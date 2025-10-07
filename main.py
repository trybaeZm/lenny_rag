
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4
import os
import json
import hashlib
import time
from typing import List, Dict, Optional
from collections import defaultdict
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
import httpx
import system

# Advanced RAG imports
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️ Cohere not installed. Reranking disabled.")

try:
    from rank_bm25 import BM25Okapi
    import nltk
    BM25_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ BM25 not installed. Hybrid search disabled.")

# === Load .env ===
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = "uploaded-files"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RAG_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "300"))

# Caching
ENABLE_PROMPT_CACHE = os.getenv("ENABLE_PROMPT_CACHE", "true").lower() == "true"
ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"

# Advanced RAG Features
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_EVALUATION = os.getenv("ENABLE_EVALUATION", "true").lower() == "true"

# Quality Thresholds
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.7"))
RAG_RERANK_THRESHOLD = float(os.getenv("RAG_RERANK_THRESHOLD", "0.3"))
RAG_TOP_K_CANDIDATES = int(os.getenv("RAG_TOP_K_CANDIDATES", "20"))
RAG_FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "5"))

# Hybrid Search Weights
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_AVAILABLE and COHERE_API_KEY else None

# === Embedding Cache ===
# In-memory cache for embeddings to reduce API calls
embedding_cache = {}

def get_embedding_cache_key(text: str, model: str) -> str:
    """Generate a unique cache key for embedding requests"""
    return hashlib.md5(f"{model}:{text}".encode()).hexdigest()

def get_cached_embedding(text: str, model: str):
    """Retrieve embedding from cache if available"""
    if not ENABLE_EMBEDDING_CACHE:
        return None
    cache_key = get_embedding_cache_key(text, model)
    return embedding_cache.get(cache_key)

def cache_embedding(text: str, model: str, embedding: list):
    """Store embedding in cache"""
    if not ENABLE_EMBEDDING_CACHE:
        return
    cache_key = get_embedding_cache_key(text, model)
    # Limit cache size to prevent memory issues
    if len(embedding_cache) > 1000:
        # Remove oldest entries (simple FIFO)
        oldest_key = next(iter(embedding_cache))
        del embedding_cache[oldest_key]
    embedding_cache[cache_key] = embedding

# === BM25 Document Store (for Hybrid Search) ===
bm25_index = None
bm25_documents = []
bm25_metadata = []

def build_bm25_index(documents: List[Dict]):
    """Build BM25 index from documents"""
    global bm25_index, bm25_documents, bm25_metadata
    
    if not BM25_AVAILABLE or not documents:
        return
    
    try:
        # Extract text and tokenize
        texts = [doc.get('chunk_text', '') for doc in documents]
        tokenized_docs = [nltk.word_tokenize(text.lower()) for text in texts]
        
        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_docs)
        bm25_documents = texts
        bm25_metadata = documents
        
        print(f"✅ Built BM25 index with {len(documents)} documents")
    except Exception as e:
        print(f"❌ BM25 index build error: {e}")

def search_bm25(query: str, top_k: int = 10) -> List[Dict]:
    """Search using BM25 keyword matching"""
    if not BM25_AVAILABLE or bm25_index is None:
        return []
    
    try:
        tokenized_query = nltk.word_tokenize(query.lower())
        scores = bm25_index.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return non-zero scores
                result = bm25_metadata[idx].copy()
                result['bm25_score'] = float(scores[idx])
                results.append(result)
        
        return results
    except Exception as e:
        print(f"❌ BM25 search error: {e}")
        return []

# === Evaluation & Monitoring ===
rag_metrics = {
    "queries_total": 0,
    "queries_with_results": 0,
    "avg_retrieval_time": 0.0,
    "avg_chunks_retrieved": 0.0,
    "avg_rerank_score": 0.0,
    "cache_hits": 0,
    "cache_misses": 0,
    "feature_usage": {
        "reranking": 0,
        "query_expansion": 0,
        "hybrid_search": 0,
        "relevance_filtering": 0,
    },
    "quality_scores": [],
}

def log_retrieval_metrics(metrics_data: Dict):
    """Log retrieval metrics for monitoring"""
    if not ENABLE_EVALUATION:
        return
    
    rag_metrics["queries_total"] += 1
    
    if metrics_data.get("chunks_retrieved", 0) > 0:
        rag_metrics["queries_with_results"] += 1
    
    # Update running averages
    n = rag_metrics["queries_total"]
    rag_metrics["avg_retrieval_time"] = (
        (rag_metrics["avg_retrieval_time"] * (n - 1) + metrics_data.get("retrieval_time", 0)) / n
    )
    rag_metrics["avg_chunks_retrieved"] = (
        (rag_metrics["avg_chunks_retrieved"] * (n - 1) + metrics_data.get("chunks_retrieved", 0)) / n
    )
    
    if metrics_data.get("rerank_score"):
        rag_metrics["avg_rerank_score"] = (
            (rag_metrics["avg_rerank_score"] * (rag_metrics["queries_with_results"] - 1) + 
             metrics_data["rerank_score"]) / rag_metrics["queries_with_results"]
        )
    
    # Track feature usage
    for feature in ["reranking", "query_expansion", "hybrid_search", "relevance_filtering"]:
        if metrics_data.get(feature):
            rag_metrics["feature_usage"][feature] += 1
    
    # Track cache performance
    if metrics_data.get("cache_hit"):
        rag_metrics["cache_hits"] += 1
    else:
        rag_metrics["cache_misses"] += 1

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Models ===
class QueryRequest(BaseModel):
    user_id: str
    user_name: str
    question: str
    session_id: str

# Accepts the simpler payload some clients send
class SimpleQuery(BaseModel):
    question: str
    session_id: str

# === File Upload ===
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...), user_name: str = Form(...)):
    try:
        file_id = str(uuid4())
        ext = file.filename.split(".")[-1]
        file_path = f"{user_id}/{file_id}.{ext}"
        content = await file.read()

        supabase.storage.from_(SUPABASE_BUCKET).upload(file_path, content, file_options={"content-type": file.content_type})
        file_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(file_path).get("publicUrl")

        supabase.table("uploaded_files").insert({
            "id": file_id,
            "user_id": user_id,
            "user_name": user_name,
            "file_name": file.filename,
            "file_type": file.content_type,
            "file_url": file_url,
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()

        return {"status": "success", "url": file_url, "file_name": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# === Ensure User + Business Data ===
def ensure_user_exists(user_id: str, user_name: str):
    try:
        exists = supabase.table("users").select("*").eq("id", user_id).execute()
        if not exists.data:
            supabase.table("users").insert({
                "id": user_id,
                "name": user_name,
                "email": f"{user_name.lower()}@autogen.com",
                "password_hash": "Temp123",
                "role": "user"
            }).execute()
            create_initial_business_data(user_id, user_name)
    except Exception as e:
        print("🚨 User creation error:", e)

def create_initial_business_data(user_id: str, user_name: str):
    try:
        supabase.table("ai_aggregated_data").insert({
            "user_id": user_id,
            "business_id": user_id,
            "user_name": user_name,
            "customer_count": 5,
            "total_sales": 12000,
            "total_orders": 10,
            "most_purchased_product": "iPhone 14 Pro",
            "customer_behavior": json.dumps({
                "peak_hours": "2 PM - 5 PM",
                "popular_locations": ["Karachi", "Lahore"]
            }),
            "revenue_forecast": json.dumps({
                "next_month": "$2000",
                "next_year": "$25000"
            }),
            "last_updated": datetime.utcnow().isoformat()
        }).execute()
    except Exception as e:
        print("🚨 Business data error:", e)

# === Chat History Store ===
def store_chat_history(user_id, business_id, question, answer, session_id, user_name=None):
    try:
        now = datetime.utcnow().isoformat()
        supabase.table("conversations").insert({
            "user_id": user_id,
            "customer_id": business_id,
            "user_name": user_name,
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "started_at": now,
            "last_updated": now
        }).execute()
    except Exception as e:
        print("❌ Chat store error:", e)

def fetch_sales_data(user_id):
    try:
        res = supabase.table("ai_aggregated_data").select("*").eq("user_id", user_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print("❌ Supabase fetch error:", e)
        return None

# === Advanced RAG Functions ===

def expand_query(query: str) -> List[str]:
    """
    Generate query variations for multi-query retrieval.
    Returns [original_query, variant1, variant2]
    """
    if not ENABLE_QUERY_EXPANSION or not client:
        return [query]
    
    try:
        prompt = f"""Generate 2 alternative phrasings of this business question to improve information retrieval.

Original: {query}

Return ONLY the 2 alternatives, one per line, no numbering or extra text."""

        resp = client.chat.completions.create(
            model=RAG_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        
        variants = resp.choices[0].message.content.strip().split('\n')
        variants = [v.strip() for v in variants if v.strip()]
        
        all_queries = [query] + variants[:2]
        print(f"🔍 Query expansion: {len(all_queries)} variants")
        return all_queries
        
    except Exception as e:
        print(f"❌ Query expansion failed: {e}")
        return [query]


def rerank_chunks(query: str, chunks: List[Dict], top_n: int = None) -> List[Dict]:
    """
    Rerank chunks using Cohere Rerank API for better relevance.
    Falls back to original chunks if reranking fails.
    """
    if top_n is None:
        top_n = RAG_FINAL_TOP_K
        
    if not ENABLE_RERANKING or not cohere_client or not chunks:
        return chunks[:top_n]
    
    try:
        # Extract text from chunks
        documents = [c.get("chunk_text", "") for c in chunks]
        
        # Call Cohere Rerank API
        results = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
            return_documents=False  # We already have the documents
        )
        
        # Reorder chunks based on rerank scores
        reranked_chunks = []
        scores = []
        for result in results.results:
            if result.relevance_score >= RAG_RERANK_THRESHOLD:
                chunk = chunks[result.index].copy()
                chunk['rerank_score'] = result.relevance_score
                reranked_chunks.append(chunk)
                scores.append(result.relevance_score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"🎯 Reranked {len(chunks)} → {len(reranked_chunks)} chunks (avg score: {avg_score:.3f})")
        
        return reranked_chunks
        
    except Exception as e:
        print(f"❌ Reranking failed: {e}, using original order")
        return chunks[:top_n]


def merge_hybrid_results(vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
    """
    Merge vector and BM25 results using weighted scoring.
    Uses Reciprocal Rank Fusion (RRF) for combining rankings.
    """
    if not bm25_results:
        return vector_results
    
    if not vector_results:
        return bm25_results
    
    # Normalize scores and combine
    merged = {}
    k = 60  # RRF constant
    
    # Add vector results
    for rank, chunk in enumerate(vector_results):
        chunk_id = chunk.get('source_id', '') + chunk.get('chunk_text', '')[:50]
        vector_score = 1 / (k + rank + 1)  # RRF score
        merged[chunk_id] = {
            'chunk': chunk,
            'score': VECTOR_WEIGHT * vector_score,
            'vector_rank': rank + 1,
        }
    
    # Add BM25 results
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk.get('source_id', '') + chunk.get('chunk_text', '')[:50]
        bm25_score = 1 / (k + rank + 1)  # RRF score
        
        if chunk_id in merged:
            merged[chunk_id]['score'] += BM25_WEIGHT * bm25_score
            merged[chunk_id]['bm25_rank'] = rank + 1
        else:
            merged[chunk_id] = {
                'chunk': chunk,
                'score': BM25_WEIGHT * bm25_score,
                'bm25_rank': rank + 1,
            }
    
    # Sort by combined score
    sorted_results = sorted(merged.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Extract chunks and add hybrid scores
    final_results = []
    for chunk_id, data in sorted_results:
        chunk = data['chunk'].copy()
        chunk['hybrid_score'] = data['score']
        chunk['vector_rank'] = data.get('vector_rank')
        chunk['bm25_rank'] = data.get('bm25_rank')
        final_results.append(chunk)
    
    print(f"🔀 Hybrid merge: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(final_results)} unique")
    
    return final_results

# === Vector Retrieval (pgvector via Supabase RPC) ===
def search_chunks(query_text: str, user_id: str, business_id: str | None = None, top_k: int = None):
    """
    Perform vector similarity search with relevance filtering.
    Returns chunks with distance scores.
    """
    if top_k is None:
        top_k = RAG_TOP_K_CANDIDATES
        
    try:
        if not client:
            return [], False
        
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Try to get embedding from cache first
        embedding = get_cached_embedding(query_text, emb_model)
        cache_hit = embedding is not None
        
        if embedding is None:
            # Cache miss - generate embedding via API
            emb_resp = client.embeddings.create(model=emb_model, input=query_text)
            embedding = emb_resp.data[0].embedding if emb_resp and emb_resp.data else []
            
            if embedding:
                # Cache the embedding for future use
                cache_embedding(query_text, emb_model, embedding)
                print(f"✅ Embedding cached for query: {query_text[:50]}...")
            else:
                return [], False
        else:
            print(f"🎯 Using cached embedding for query: {query_text[:50]}...")

        url = f"{SUPABASE_URL}/rest/v1/rpc/match_rag_chunks"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "query_embedding": embedding,
            "p_user_id": user_id,
            "p_business_id": business_id,
            "p_limit": top_k * 2,  # Get more candidates for filtering
        }
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as hx:
            resp = hx.post(url, headers=headers, json=payload)
            if resp.status_code in (200, 201):
                data = resp.json()
                chunks = data if isinstance(data, list) else []
                
                # Apply relevance filtering
                filtered_chunks = [
                    c for c in chunks
                    if c.get('distance', 999) < RAG_RELEVANCE_THRESHOLD
                ]
                
                if len(filtered_chunks) < len(chunks):
                    print(f"📊 Relevance filter: {len(chunks)} → {len(filtered_chunks)} chunks (threshold: {RAG_RELEVANCE_THRESHOLD})")
                
                return filtered_chunks[:top_k], cache_hit
            return [], cache_hit
    except Exception as e:
        print("❌ search_chunks error:", e)
        return [], False


def search_chunks_advanced(query_text: str, user_id: str, business_id: str | None = None) -> tuple[List[Dict], Dict]:
    """
    Advanced search combining all RAG improvements:
    1. Query Expansion
    2. Hybrid Search (Vector + BM25)
    3. Relevance Filtering
    4. Reranking
    5. Metrics Tracking
    """
    start_time = time.time()
    metrics = {
        "query": query_text,
        "retrieval_time": 0.0,
        "chunks_retrieved": 0,
        "rerank_score": None,
        "cache_hit": False,
        "reranking": False,
        "query_expansion": False,
        "hybrid_search": False,
        "relevance_filtering": False,
    }
    
    try:
        # Step 1: Query Expansion
        queries = expand_query(query_text)
        if len(queries) > 1:
            metrics["query_expansion"] = True
        
        # Step 2: Vector Search for all query variants
        all_vector_results = []
        seen_chunks = set()
        cache_hit = False
        
        for query_variant in queries:
            chunks, hit = search_chunks(query_variant, user_id, business_id, top_k=RAG_TOP_K_CANDIDATES)
            cache_hit = cache_hit or hit
            
            # Deduplicate
            for chunk in chunks:
                chunk_id = chunk.get('source_id', '') + chunk.get('chunk_text', '')[:50]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_vector_results.append(chunk)
        
        metrics["cache_hit"] = cache_hit
        metrics["relevance_filtering"] = True
        
        print(f"📚 Multi-query retrieval: {len(all_vector_results)} unique chunks from {len(queries)} queries")
        
        # Step 3: Hybrid Search (add BM25 if enabled)
        if ENABLE_HYBRID_SEARCH and BM25_AVAILABLE and bm25_index:
            bm25_results = search_bm25(query_text, top_k=10)
            if bm25_results:
                all_vector_results = merge_hybrid_results(all_vector_results, bm25_results)
                metrics["hybrid_search"] = True
        
        # Step 4: Rerank top candidates
        final_chunks = rerank_chunks(query_text, all_vector_results, top_n=RAG_FINAL_TOP_K)
        
        if final_chunks and 'rerank_score' in final_chunks[0]:
            metrics["reranking"] = True
            avg_rerank_score = sum(c.get('rerank_score', 0) for c in final_chunks) / len(final_chunks)
            metrics["rerank_score"] = avg_rerank_score
        
        # Track metrics
        metrics["chunks_retrieved"] = len(final_chunks)
        metrics["retrieval_time"] = time.time() - start_time
        
        log_retrieval_metrics(metrics)
        
        print(f"⚡ Advanced search completed in {metrics['retrieval_time']:.2f}s: {len(final_chunks)} chunks")
        
        return final_chunks, metrics
        
    except Exception as e:
        print(f"❌ Advanced search error: {e}")
        metrics["retrieval_time"] = time.time() - start_time
        log_retrieval_metrics(metrics)
        return [], metrics

# === Chat History Retrieval ===
def fetch_chat_history_messages(user_id: str, session_id: str, limit: int = 8):
    try:
        res = supabase.table("conversations") \
            .select("question, answer") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("last_updated", desc=False) \
            .limit(limit) \
            .execute()
        messages = []
        for row in res.data or []:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q:
                messages.append({"role": "user", "content": q})
            if a:
                messages.append({"role": "assistant", "content": a})
        return messages[-limit:]
    except Exception as e:
        print("❌ Chat history fetch error:", e)
        return []

# === Gemini Logic ===
def query_rag_ai(user_id, user_query, user_name=None, file_context=None, session_id=None):
    print(f'endpoint hit : {OPENAI_API_KEY}')
    user_data = fetch_sales_data(user_id)
    if not user_data:
        return "❌ No business data found."

    try:
        customer_behavior = json.loads(user_data.get("customer_behavior", "{}"))
        revenue_forecast = json.loads(user_data.get("revenue_forecast", "{}"))
    except:
        customer_behavior, revenue_forecast = {}, {}

    # 🎯 Load Lenny system prompt
    SYSTEM_PROMPT = system.get_system_prompt()

    # 🎯 Business Context and User Query
    business_context = f"""
📁 File:
{file_context or "No uploaded file."}

📊 Business Data:
- Name: {user_name}
- Total Sales: ${user_data.get("total_sales")}
- Total Orders: {user_data.get("total_orders")}
- Customers: {user_data.get("customer_count")}
- Popular Hours: {customer_behavior.get("peak_hours")}
- Popular Locations: {customer_behavior.get("popular_locations")}
- Revenue Forecast Next Month: {revenue_forecast.get("next_month")}
    """

    if not client:
        answer = "🤖 AI is unavailable: OPENAI_API_KEY not configured."
    else:
        try:
            # Build multi-turn chat context
            history_msgs = fetch_chat_history_messages(user_id=user_id, session_id=session_id or "", limit=8)

            # USE ADVANCED SEARCH with all improvements
            chunks, search_metrics = search_chunks_advanced(
                query_text=user_query, 
                user_id=user_id, 
                business_id=user_data.get("business_id")
            )
            
            # Build context with enhanced metadata
            context_blocks = []
            if chunks:
                for i, c in enumerate(chunks):
                    title = c.get("title") or c.get("source_table", "Unknown Source")
                    snippet = c.get("chunk_text", "")
                    
                    # Include relevance scores for transparency
                    score_info = []
                    if c.get("rerank_score"):
                        score_info.append(f"Relevance: {c['rerank_score']:.2f}")
                    elif c.get("distance"):
                        similarity = max(0, 1 - c['distance'])
                        score_info.append(f"Similarity: {similarity:.2f}")
                    
                    if c.get("hybrid_score"):
                        score_info.append(f"Hybrid: {c['hybrid_score']:.2f}")
                    
                    score_str = " | ".join(score_info) if score_info else ""
                    
                    context_blocks.append(
                        f"[Source {i+1}: {title}" + 
                        (f" ({score_str})" if score_str else "") + 
                        f"]\n{snippet}"
                    )

            augmented_question = user_query
            if context_blocks:
                augmented_question = (
                    user_query + 
                    "\n\n=== Retrieved Business Intelligence ===\n" + 
                    "\n\n".join(context_blocks) +
                    "\n=== End of Retrieved Context ==="
                )
                print(f"✅ Built context with {len(chunks)} sources, {len(augmented_question)} chars")
            else:
                augmented_question = (
                    user_query + 
                    "\n\n[Note: No specific context found in knowledge base - please provide general guidance based on business data above]"
                )
                print("⚠️ No relevant context found, proceeding without RAG")

            # Build messages with prompt caching support
            # The system prompt is marked for caching to reduce costs
            messages = []
            
            if ENABLE_PROMPT_CACHE:
                # Use prompt caching for static system prompt
                # OpenAI caches content when it exceeds ~1024 tokens
                messages.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
            else:
                # Standard message without caching
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            
            # Add conversation history (not cached as it changes)
            messages.extend(history_msgs)
            
            # Add current user query with business context
            messages.append({
                "role": "user",
                "content": f"{business_context}\n\nUser's Question: {augmented_question}"
            })

            # Create completion with appropriate parameters
            completion_params = {
                "model": RAG_OPENAI_MODEL,
                "messages": messages,
                "temperature": RAG_TEMPERATURE,
                "max_tokens": RAG_MAX_TOKENS,
            }
            
            resp = client.chat.completions.create(**completion_params)
            answer = (resp.choices[0].message.content or "").strip() if resp and resp.choices else "🤖 No response."
            
            # Log cache usage if available
            if ENABLE_PROMPT_CACHE and hasattr(resp, 'usage'):
                usage = resp.usage
                if hasattr(usage, 'prompt_tokens_details'):
                    print(f"💰 Prompt tokens: {usage.prompt_tokens}, Cached: {getattr(usage.prompt_tokens_details, 'cached_tokens', 0)}")
                    
        except Exception as e:
            answer = f"🤖 AI service temporarily unavailable. Error: {str(e)}"

    store_chat_history(user_id, user_data["business_id"], user_query, answer, session_id, user_name)
    return answer
# === Query Endpoint ===


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        ensure_user_exists(request.user_id, request.user_name)

        file_context = ""  # not needed anymore

        result = query_rag_ai(
            request.user_id,
            request.question,
            request.user_name,
            file_context,
            request.session_id
        )

        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Compatibility route for frontend expecting /api/v1/rag/query
@app.post("/api/v1/rag/query")
async def rag_query_simple(request: SimpleQuery):
    try:
        # Fallback demo user if client doesn't send auth/user context
        user_id = "00000000-0000-0000-0000-000000000001"
        user_name = "User"
        ensure_user_exists(user_id, user_name)

        file_context = ""
        result = query_rag_ai(
            user_id,
            request.question,
            user_name,
            file_context,
            request.session_id,
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Basic health/version endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "model": RAG_OPENAI_MODEL,
        "temperature": RAG_TEMPERATURE,
        "max_tokens": RAG_MAX_TOKENS,
        "prompt_cache_enabled": ENABLE_PROMPT_CACHE,
        "embedding_cache_enabled": ENABLE_EMBEDDING_CACHE,
        "advanced_features": {
            "reranking": ENABLE_RERANKING and cohere_client is not None,
            "query_expansion": ENABLE_QUERY_EXPANSION,
            "hybrid_search": ENABLE_HYBRID_SEARCH and BM25_AVAILABLE,
            "evaluation": ENABLE_EVALUATION,
        },
        "thresholds": {
            "relevance_threshold": RAG_RELEVANCE_THRESHOLD,
            "rerank_threshold": RAG_RERANK_THRESHOLD,
            "top_k_candidates": RAG_TOP_K_CANDIDATES,
            "final_top_k": RAG_FINAL_TOP_K,
        }
    }

@app.get("/cache-stats")
def cache_stats():
    """Get cache statistics"""
    cache_hit_rate = 0
    if rag_metrics["cache_hits"] + rag_metrics["cache_misses"] > 0:
        cache_hit_rate = rag_metrics["cache_hits"] / (rag_metrics["cache_hits"] + rag_metrics["cache_misses"])
    
    return {
        "embedding_cache_size": len(embedding_cache),
        "embedding_cache_enabled": ENABLE_EMBEDDING_CACHE,
        "prompt_cache_enabled": ENABLE_PROMPT_CACHE,
        "max_cache_size": 1000,
        "cache_hit_rate": f"{cache_hit_rate:.1%}",
        "cache_hits": rag_metrics["cache_hits"],
        "cache_misses": rag_metrics["cache_misses"],
    }

@app.get("/rag-metrics")
def get_rag_metrics():
    """Get comprehensive RAG performance metrics"""
    if not ENABLE_EVALUATION:
        return {"error": "Evaluation is disabled. Set ENABLE_EVALUATION=true in .env"}
    
    success_rate = 0
    if rag_metrics["queries_total"] > 0:
        success_rate = rag_metrics["queries_with_results"] / rag_metrics["queries_total"]
    
    feature_usage_pct = {}
    for feature, count in rag_metrics["feature_usage"].items():
        if rag_metrics["queries_total"] > 0:
            feature_usage_pct[feature] = f"{(count / rag_metrics['queries_total']) * 100:.1f}%"
        else:
            feature_usage_pct[feature] = "0%"
    
    return {
        "overview": {
            "total_queries": rag_metrics["queries_total"],
            "queries_with_results": rag_metrics["queries_with_results"],
            "success_rate": f"{success_rate:.1%}",
            "avg_retrieval_time": f"{rag_metrics['avg_retrieval_time']:.3f}s",
            "avg_chunks_retrieved": f"{rag_metrics['avg_chunks_retrieved']:.1f}",
            "avg_rerank_score": f"{rag_metrics['avg_rerank_score']:.3f}" if rag_metrics['avg_rerank_score'] > 0 else "N/A",
        },
        "caching": {
            "cache_hits": rag_metrics["cache_hits"],
            "cache_misses": rag_metrics["cache_misses"],
            "hit_rate": f"{(rag_metrics['cache_hits'] / (rag_metrics['cache_hits'] + rag_metrics['cache_misses']) * 100):.1f}%" if (rag_metrics['cache_hits'] + rag_metrics['cache_misses']) > 0 else "0%",
        },
        "feature_usage": {
            "reranking": {
                "used": rag_metrics["feature_usage"]["reranking"],
                "percentage": feature_usage_pct["reranking"],
            },
            "query_expansion": {
                "used": rag_metrics["feature_usage"]["query_expansion"],
                "percentage": feature_usage_pct["query_expansion"],
            },
            "hybrid_search": {
                "used": rag_metrics["feature_usage"]["hybrid_search"],
                "percentage": feature_usage_pct["hybrid_search"],
            },
            "relevance_filtering": {
                "used": rag_metrics["feature_usage"]["relevance_filtering"],
                "percentage": feature_usage_pct["relevance_filtering"],
            },
        },
        "quality_scores": rag_metrics["quality_scores"][-10:] if rag_metrics["quality_scores"] else [],
    }

@app.post("/reset-metrics")
def reset_metrics():
    """Reset all RAG metrics (useful for testing)"""
    global rag_metrics
    rag_metrics = {
        "queries_total": 0,
        "queries_with_results": 0,
        "avg_retrieval_time": 0.0,
        "avg_chunks_retrieved": 0.0,
        "avg_rerank_score": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "feature_usage": {
            "reranking": 0,
            "query_expansion": 0,
            "hybrid_search": 0,
            "relevance_filtering": 0,
        },
        "quality_scores": [],
    }
    return {"status": "success", "message": "Metrics reset successfully"}

@app.post("/build-bm25-index")
async def build_bm25_index_endpoint(user_id: str):
    """Build BM25 index for hybrid search from user's RAG chunks"""
    if not BM25_AVAILABLE:
        raise HTTPException(status_code=400, detail="BM25 not available. Install rank-bm25 and nltk.")
    
    try:
        # Fetch all chunks for the user
        res = supabase.table("rag_chunks").select("*").eq("user_id", user_id).execute()
        documents = res.data or []
        
        if not documents:
            return {"status": "warning", "message": "No documents found for this user"}
        
        build_bm25_index(documents)
        
        return {
            "status": "success",
            "message": f"BM25 index built with {len(documents)} documents",
            "document_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build BM25 index: {str(e)}")
# === Chat Session Endpoint ===
@app.get("/chat-session")
async def get_chat_session(user_id: str, session_id: str):
    try:
        res = supabase.table("conversations") \
            .select("question, answer") \
            .eq("user_id", user_id) \
            .eq("session_id", session_id) \
            .order("last_updated", desc=False) \
            .execute()
        return {"session": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# === Recent Sessions ===
@app.get("/recent-questions")
async def get_recent_questions(user_id: str):
    try:
        res = supabase.table("conversations") \
            .select("session_id, question, last_updated") \
            .eq("user_id", user_id) \
            .order("last_updated", desc=True) \
            .execute()

        seen_sessions = set()
        unique_sessions = []

        for row in res.data:
            if row["session_id"] not in seen_sessions:
                seen_sessions.add(row["session_id"])
                unique_sessions.append(row)
            if len(unique_sessions) == 5:
                break

        return {"recent": unique_sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === All History (Optional) ===
@app.get("/chat-history")
async def get_chat_history(user_id: str):
    try:
        res = supabase.table("conversations") \
            .select("question, answer") \
            .eq("user_id", user_id) \
            .order("last_updated", desc=False) \
            .execute()
        return {"history": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


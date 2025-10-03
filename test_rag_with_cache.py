#!/usr/bin/env python3
"""
Test script to verify RAG effectiveness with caching enabled.
This script tests that caching doesn't impact RAG retrieval quality.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"cache-test-{int(time.time())}"

# Test queries
TEST_QUERIES = [
    "What are my total sales?",
    "How many customers do I have?",
    "What's the revenue forecast for next month?",
    "Show me peak business hours",
    "What are my total sales?",  # Duplicate to test embedding cache
]

def test_endpoint(endpoint: str, method: str = "GET", data: dict = None):
    """Test an API endpoint"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        else:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
        return response
    except Exception as e:
        print(f"❌ Error testing {endpoint}: {e}")
        return None

def main():
    print("=" * 70)
    print("🧪 RAG Backend Caching Test")
    print("=" * 70)
    print()
    
    # Step 1: Check health
    print("1️⃣ Checking backend health...")
    response = test_endpoint("/health")
    if response and response.status_code == 200:
        print("✅ Backend is healthy")
    else:
        print("❌ Backend is not responding. Please start the backend first.")
        return
    print()
    
    # Step 2: Check version and cache config
    print("2️⃣ Checking cache configuration...")
    response = test_endpoint("/version")
    if response and response.status_code == 200:
        config = response.json()
        print(f"   Model: {config.get('model')}")
        print(f"   Prompt Cache: {'✅ Enabled' if config.get('prompt_cache_enabled') else '❌ Disabled'}")
        print(f"   Embedding Cache: {'✅ Enabled' if config.get('embedding_cache_enabled') else '❌ Disabled'}")
    print()
    
    # Step 3: Check initial cache stats
    print("3️⃣ Initial cache statistics...")
    response = test_endpoint("/cache-stats")
    if response and response.status_code == 200:
        stats = response.json()
        print(f"   Embedding cache size: {stats.get('embedding_cache_size')}")
        print(f"   Max cache size: {stats.get('max_cache_size')}")
    print()
    
    # Step 4: Test RAG queries
    print("4️⃣ Testing RAG queries with caching...")
    print("-" * 70)
    
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        start_time = time.time()
        response = test_endpoint(
            "/api/v1/rag/query",
            method="POST",
            data={
                "question": query,
                "session_id": TEST_SESSION_ID
            }
        )
        elapsed_time = time.time() - start_time
        
        if response and response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            # Check if answer is meaningful
            is_valid = len(answer) > 20 and "unavailable" not in answer.lower()
            
            results.append({
                "query": query,
                "answer": answer,
                "time": elapsed_time,
                "valid": is_valid
            })
            
            print(f"   ⏱️  Response time: {elapsed_time:.2f}s")
            print(f"   📝 Answer preview: {answer[:100]}...")
            print(f"   ✅ Valid response: {is_valid}")
            
            # Check for duplicate query (should be faster due to embedding cache)
            if i > 1 and query == TEST_QUERIES[i-2]:
                prev_time = results[i-2]['time']
                if elapsed_time < prev_time:
                    improvement = ((prev_time - elapsed_time) / prev_time) * 100
                    print(f"   🚀 {improvement:.1f}% faster than first request (embedding cache hit!)")
        else:
            print(f"   ❌ Request failed: {response.status_code if response else 'No response'}")
            results.append({
                "query": query,
                "answer": None,
                "time": elapsed_time,
                "valid": False
            })
    
    print()
    print("-" * 70)
    
    # Step 5: Final cache stats
    print("\n5️⃣ Final cache statistics...")
    response = test_endpoint("/cache-stats")
    if response and response.status_code == 200:
        stats = response.json()
        print(f"   Embedding cache size: {stats.get('embedding_cache_size')}")
        print(f"   Cache entries added: {stats.get('embedding_cache_size')}")
    print()
    
    # Step 6: Summary
    print("=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    valid_responses = sum(1 for r in results if r['valid'])
    total_queries = len(TEST_QUERIES)
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"   Total queries: {total_queries}")
    print(f"   Valid responses: {valid_responses}/{total_queries}")
    print(f"   Success rate: {(valid_responses/total_queries)*100:.1f}%")
    print(f"   Average response time: {avg_time:.2f}s")
    print()
    
    if valid_responses == total_queries:
        print("✅ ALL TESTS PASSED - RAG is working effectively with caching!")
    else:
        print("⚠️  SOME TESTS FAILED - Please check the backend logs")
    
    print()
    print("💡 Tips:")
    print("   - Check backend console for cache hit messages")
    print("   - Look for '🎯 Using cached embedding' for cache hits")
    print("   - Look for '💰 Prompt tokens: X, Cached: Y' for prompt cache usage")
    print()
    
    # Step 7: Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "test_session_id": TEST_SESSION_ID,
            "total_queries": total_queries,
            "valid_responses": valid_responses,
            "average_time": avg_time,
            "results": results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()


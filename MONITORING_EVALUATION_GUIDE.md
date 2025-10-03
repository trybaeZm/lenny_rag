# RAG Monitoring & Evaluation Guide

## 📊 Complete Monitoring Strategy

This guide covers everything you need to monitor, evaluate, and continuously improve your RAG system.

---

## 🎯 Key Metrics to Track

### 1. Retrieval Metrics

#### Success Rate
```
queries_with_results / total_queries
```
- **Target:** > 85%
- **Warning:** < 70%
- **Critical:** < 50%

**What it means:**
- % of queries that retrieved at least one chunk
- Low rate = poor index coverage or too strict filtering

#### Avg Chunks Retrieved
```
total_chunks / total_queries
```
- **Target:** 3-5 chunks
- **Warning:** < 2 chunks
- **Too many:** > 8 chunks

**What it means:**
- Average context size per query
- Too few = missing information
- Too many = noisy context

#### Avg Retrieval Time
```
total_time / total_queries
```
- **Target:** < 0.6s
- **Warning:** > 1.0s
- **Critical:** > 2.0s

**What it means:**
- Speed of entire retrieval pipeline
- Impacts user experience

### 2. Quality Metrics

#### Avg Rerank Score
```
sum(rerank_scores) / queries_with_reranking
```
- **Excellent:** > 0.8
- **Good:** 0.6-0.8
- **Warning:** 0.4-0.6
- **Poor:** < 0.4

**What it means:**
- Average relevance of retrieved chunks
- Higher = better quality matches

#### Cache Hit Rate
```
cache_hits / (cache_hits + cache_misses)
```
- **Target:** > 50%
- **Good:** > 60%
- **Excellent:** > 75%

**What it means:**
- % of queries using cached embeddings
- Higher = faster + cheaper

### 3. Feature Usage Metrics

Track which features are actually being used:
- **Reranking:** Should be ~100% if enabled
- **Query Expansion:** Should be ~100% if enabled
- **Hybrid Search:** Varies based on BM25 index
- **Relevance Filtering:** Should be ~100%

---

## 📈 Real-Time Monitoring

### Dashboard Setup

Check these endpoints regularly:

#### 1. Health Check (Every 5 mins)
```bash
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}`

#### 2. Configuration Check (Daily)
```bash
curl http://localhost:8000/version | jq
```

Verify all features are enabled as expected.

#### 3. Cache Performance (Hourly)
```bash
curl http://localhost:8000/cache-stats | jq
```

Watch for:
- Cache hit rate declining
- Cache size growing too large

#### 4. RAG Metrics (After each session)
```bash
curl http://localhost:8000/rag-metrics | jq
```

Full performance overview.

---

## 🔍 Detailed Monitoring Commands

### Check Overall Health
```bash
#!/bin/bash
# save as check_rag_health.sh

echo "=== RAG System Health Check ==="
echo ""

# Health
echo "1. System Health:"
curl -s http://localhost:8000/health | jq .
echo ""

# Metrics
echo "2. Performance Metrics:"
curl -s http://localhost:8000/rag-metrics | jq '.overview'
echo ""

# Cache
echo "3. Cache Performance:"
curl -s http://localhost:8000/cache-stats | jq '{
  hit_rate: .cache_hit_rate,
  size: .embedding_cache_size
}'
echo ""

# Features
echo "4. Feature Status:"
curl -s http://localhost:8000/version | jq '.advanced_features'
```

Run daily:
```bash
chmod +x check_rag_health.sh
./check_rag_health.sh
```

### Monitor Specific Metrics

#### Track Success Rate Over Time
```bash
# Run every hour, log to file
while true; do
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    success_rate=$(curl -s http://localhost:8000/rag-metrics | jq -r '.overview.success_rate')
    echo "$timestamp,$success_rate" >> rag_success_rate.csv
    sleep 3600
done
```

#### Alert on Low Performance
```bash
#!/bin/bash
# alert_on_low_performance.sh

success_rate=$(curl -s http://localhost:8000/rag-metrics | jq -r '.overview.success_rate' | tr -d '%')
retrieval_time=$(curl -s http://localhost:8000/rag-metrics | jq -r '.overview.avg_retrieval_time' | tr -d 's')

if (( $(echo "$success_rate < 70" | bc -l) )); then
    echo "⚠️ ALERT: Success rate below 70%: ${success_rate}%"
fi

if (( $(echo "$retrieval_time > 1.0" | bc -l) )); then
    echo "⚠️ ALERT: Retrieval time above 1.0s: ${retrieval_time}s"
fi
```

---

## 🧪 Evaluation Framework

### Method 1: Manual Testing (Quick)

Create a test set of 10-20 queries:

```python
# test_queries.py
TEST_QUERIES = [
    {
        "query": "What are my total sales?",
        "expected_keywords": ["sales", "revenue", "$12000"],
        "should_retrieve": True,
    },
    {
        "query": "How many customers do I have?",
        "expected_keywords": ["customers", "5"],
        "should_retrieve": True,
    },
    {
        "query": "Tell me about quantum physics",
        "expected_keywords": [],
        "should_retrieve": False,  # Out of scope
    },
]

def test_rag():
    for test in TEST_QUERIES:
        response = requests.post(
            "http://localhost:8000/api/v1/rag/query",
            json={"question": test["query"], "session_id": "test"}
        )
        
        answer = response.json()["response"]
        
        # Check if keywords present
        found_keywords = [
            kw for kw in test["expected_keywords"]
            if kw.lower() in answer.lower()
        ]
        
        print(f"Query: {test['query']}")
        print(f"Found: {len(found_keywords)}/{len(test['expected_keywords'])} keywords")
        print(f"Answer: {answer[:100]}...")
        print()
```

### Method 2: Automated Evaluation (Advanced)

Using RAGAs framework for automated metrics:

```python
# evaluation.py
from ragas import evaluate
from ragas.metrics import (
    faithfulness,  # Is answer based on context?
    answer_relevancy,  # Does answer address question?
    context_precision,  # Are relevant chunks ranked high?
    context_recall,  # Did we retrieve all relevant chunks?
)
from datasets import Dataset

# Your test data
data = {
    "question": ["What are my total sales?", ...],
    "answer": ["Your total sales are $12,000", ...],  # From RAG
    "contexts": [[chunk1, chunk2], ...],  # Retrieved chunks
    "ground_truth": ["Total sales: $12,000", ...],  # Reference answer
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
)

print(results)
```

**Target Scores:**
- Faithfulness: > 0.9 (answer based on retrieved context)
- Answer Relevancy: > 0.8 (answer addresses question)
- Context Precision: > 0.7 (relevant chunks ranked high)
- Context Recall: > 0.7 (retrieved all relevant info)

### Method 3: A/B Testing

Compare configurations:

```python
# ab_test.py
import requests
import time

CONFIGS = {
    "baseline": {
        "ENABLE_RERANKING": "false",
        "ENABLE_QUERY_EXPANSION": "false",
    },
    "advanced": {
        "ENABLE_RERANKING": "true",
        "ENABLE_QUERY_EXPANSION": "true",
    },
}

def run_test(config_name, queries):
    # Update .env with config
    # Restart backend
    # Run queries
    results = []
    for query in queries:
        start = time.time()
        response = requests.post(
            "http://localhost:8000/api/v1/rag/query",
            json={"question": query, "session_id": f"test-{config_name}"}
        )
        elapsed = time.time() - start
        
        results.append({
            "query": query,
            "answer": response.json()["response"],
            "time": elapsed,
        })
    
    return results

# Compare
baseline_results = run_test("baseline", TEST_QUERIES)
advanced_results = run_test("advanced", TEST_QUERIES)

# Manually compare answer quality
```

---

## 📉 Interpreting Metrics

### Scenario Analysis

#### Scenario 1: Low Success Rate (< 70%)

**Possible Causes:**
1. Relevance threshold too strict
2. Poor index coverage
3. Query-document mismatch

**Investigation Steps:**
```bash
# Check threshold
curl http://localhost:8000/version | jq '.thresholds.relevance_threshold'

# Check if documents exist
curl "http://localhost:8000/build-bm25-index?user_id=USER_ID"

# Look at actual queries
tail -f backend.log | grep "query:"
```

**Fixes:**
```bash
# More lenient threshold
RAG_RELEVANCE_THRESHOLD=0.8  # from 0.7

# OR ensure data is indexed properly
```

#### Scenario 2: Low Rerank Scores (< 0.6)

**Possible Causes:**
1. Vector search returning poor candidates
2. Queries too vague
3. Domain mismatch

**Investigation Steps:**
```bash
# Check feature usage
curl http://localhost:8000/rag-metrics | jq '.feature_usage.reranking'

# Check if reranking is actually running
tail -f backend.log | grep "Reranked"
```

**Fixes:**
```bash
# Get more candidates for reranking
RAG_TOP_K_CANDIDATES=30  # from 20

# Enable query expansion
ENABLE_QUERY_EXPANSION=true

# Lower rerank threshold
RAG_RERANK_THRESHOLD=0.2  # from 0.3
```

#### Scenario 3: Slow Retrieval (> 1.0s)

**Possible Causes:**
1. Too many features enabled
2. Query expansion overhead
3. Network latency to Supabase/Cohere

**Investigation Steps:**
```bash
# Check average time
curl http://localhost:8000/rag-metrics | jq '.overview.avg_retrieval_time'

# Check backend logs for timing
tail -f backend.log | grep "Advanced search completed"
```

**Fixes:**
```bash
# Disable expensive features
ENABLE_QUERY_EXPANSION=false
ENABLE_HYBRID_SEARCH=false

# Reduce candidates
RAG_TOP_K_CANDIDATES=10  # from 20

# Ensure caching is enabled
ENABLE_EMBEDDING_CACHE=true
```

#### Scenario 4: Low Cache Hit Rate (< 40%)

**Possible Causes:**
1. Users asking unique questions
2. Cache too small
3. Cache being cleared

**Investigation Steps:**
```bash
# Check cache stats
curl http://localhost:8000/cache-stats | jq

# Check cache size
curl http://localhost:8000/cache-stats | jq '.embedding_cache_size'
```

**Fixes:**
```python
# In main.py, increase cache size
if len(embedding_cache) > 5000:  # from 1000
    ...
```

---

## 🎯 Quality Assurance Checklist

### Daily Checks
- [ ] System health: `/health` returns OK
- [ ] Success rate > 80%
- [ ] Avg retrieval time < 0.8s
- [ ] Cache hit rate > 50%

### Weekly Checks
- [ ] Review feature usage percentages
- [ ] Check for error patterns in logs
- [ ] Run manual test queries
- [ ] Update BM25 index if data changed

### Monthly Checks
- [ ] Full evaluation with RAGAs
- [ ] A/B test configuration changes
- [ ] Review and update test queries
- [ ] Optimize thresholds based on usage

---

## 📊 Reporting Template

### Weekly RAG Report

```markdown
# RAG Performance Report - Week of [DATE]

## Overview
- Total Queries: XXX
- Success Rate: XX.X%
- Avg Retrieval Time: X.XXs
- Cache Hit Rate: XX.X%

## Quality Metrics
- Avg Rerank Score: X.XX
- Avg Chunks Retrieved: X.X

## Feature Usage
- Reranking: XX.X%
- Query Expansion: XX.X%
- Hybrid Search: XX.X%

## Issues Detected
- [List any performance issues]

## Actions Taken
- [List configuration changes]

## Next Steps
- [List planned improvements]
```

### Generate Automatically

```bash
# generate_report.sh
#!/bin/bash

date=$(date +"%Y-%m-%d")
metrics=$(curl -s http://localhost:8000/rag-metrics)

cat > "rag_report_$date.md" << EOF
# RAG Performance Report - $date

## Overview
$(echo $metrics | jq '.overview')

## Caching
$(echo $metrics | jq '.caching')

## Feature Usage
$(echo $metrics | jq '.feature_usage')
EOF

echo "Report saved to rag_report_$date.md"
```

---

## 🔧 Fine-Tuning Workflow

### 1. Baseline Measurement
```bash
# Reset metrics
curl -X POST http://localhost:8000/reset-metrics

# Run test queries
python test_advanced_rag.py

# Record baseline
curl http://localhost:8000/rag-metrics > baseline.json
```

### 2. Make ONE Change
```bash
# Example: Adjust relevance threshold
RAG_RELEVANCE_THRESHOLD=0.6  # from 0.7

# Restart backend
```

### 3. Test Impact
```bash
# Reset metrics
curl -X POST http://localhost:8000/reset-metrics

# Run same test queries
python test_advanced_rag.py

# Compare results
curl http://localhost:8000/rag-metrics > experiment.json

# Diff
diff baseline.json experiment.json
```

### 4. Decision
- **Improvement?** → Keep change, update baseline
- **No change?** → Revert, try different approach
- **Worse?** → Revert immediately

### 5. Document
```bash
echo "$(date): Changed RAG_RELEVANCE_THRESHOLD from 0.7 to 0.6" >> tuning_log.txt
echo "Result: Success rate improved from 82% to 87%" >> tuning_log.txt
```

---

## 🚨 Alerting Rules

### Critical Alerts (Immediate Action)
```bash
# Success rate < 50%
if success_rate < 50:
    alert("CRITICAL: RAG success rate below 50%")

# Retrieval time > 2.0s
if avg_retrieval_time > 2.0:
    alert("CRITICAL: RAG too slow (>2s)")

# System down
if health != "ok":
    alert("CRITICAL: RAG system down")
```

### Warning Alerts (Review Soon)
```bash
# Success rate < 70%
if success_rate < 70:
    alert("WARNING: RAG success rate below 70%")

# Cache hit rate < 30%
if cache_hit_rate < 0.3:
    alert("WARNING: Poor cache performance")

# Rerank score < 0.5
if avg_rerank_score < 0.5:
    alert("WARNING: Low relevance scores")
```

---

## ✅ Success Criteria

Your RAG system is performing well if:

- ✅ Success rate > 85%
- ✅ Avg rerank score > 0.75
- ✅ Avg retrieval time < 0.6s
- ✅ Cache hit rate > 50%
- ✅ Feature usage as expected (100% for enabled features)
- ✅ No critical errors in logs
- ✅ User satisfaction with answers

---

## 📚 Additional Resources

### Tools
- **RAGAs:** https://github.com/explodinggradients/ragas
- **LangSmith:** https://smith.langchain.com/
- **TruLens:** https://www.trulens.org/

### Dashboards
- **Grafana:** Visualize metrics over time
- **Datadog:** Monitor production RAG systems
- **Custom:** Build with Streamlit/Plotly

### Reading
- "Evaluating RAG Systems" by LangChain
- "Advanced RAG Techniques" by LlamaIndex
- Cohere Rerank documentation

---

## 🎯 Summary

**Monitor these 3 key metrics:**
1. Success Rate (> 85%)
2. Avg Rerank Score (> 0.75)
3. Retrieval Time (< 0.6s)

**Check daily:**
- `/health` - System up?
- `/rag-metrics` - Performance OK?
- Backend logs - Any errors?

**Tune weekly:**
- Adjust one threshold
- Test impact
- Keep or revert

**Evaluate monthly:**
- Run full RAGAs evaluation
- A/B test major changes
- Update test queries

With proper monitoring and evaluation, your RAG system will continuously improve! 🚀


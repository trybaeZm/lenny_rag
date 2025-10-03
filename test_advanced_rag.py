# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""
Advanced RAG Testing Script
Tests all 5 enterprise features and generates a comprehensive report.
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = f"advanced-test-{int(time.time())}"

# Comprehensive test queries
TEST_QUERIES = [
    {
        "query": "What are my total sales?",
        "category": "Business Metrics",
        "expected_keywords": ["sales", "$12", "12000"],
        "should_retrieve": True,
    },
    {
        "query": "How many customers do I have?",
        "category": "Business Metrics",
        "expected_keywords": ["customer", "5"],
        "should_retrieve": True,
    },
    {
        "query": "Show me revenue forecast for next month",
        "category": "Forecasting",
        "expected_keywords": ["revenue", "forecast", "month", "2000"],
        "should_retrieve": True,
    },
    {
        "query": "What are the peak business hours?",
        "category": "Customer Behavior",
        "expected_keywords": ["peak", "hour", "2 PM", "5 PM"],
        "should_retrieve": True,
    },
    {
        "query": "Tell me about customer behavior patterns",
        "category": "Customer Behavior",
        "expected_keywords": ["customer", "behavior", "peak"],
        "should_retrieve": True,
    },
    {
        "query": "What is the most popular product?",
        "category": "Product Analysis",
        "expected_keywords": ["product", "iPhone"],
        "should_retrieve": True,
    },
    {
        "query": "Where are my top customers located?",
        "category": "Geographic Analysis",
        "expected_keywords": ["location", "Karachi", "Lahore"],
        "should_retrieve": True,
    },
    # Edge cases
    {
        "query": "Tell me about quantum physics",
        "category": "Out of Scope",
        "expected_keywords": [],
        "should_retrieve": False,
    },
    {
        "query": "sales",  # Vague single word
        "category": "Vague Query",
        "expected_keywords": ["sales"],
        "should_retrieve": True,
    },
]


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def test_endpoint(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Test an API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=30)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"Status {response.status_code}", "data": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_system_health():
    """Step 1: Check system health"""
    print_header("STEP 1: SYSTEM HEALTH CHECK")
    
    # Health
    result = test_endpoint("/health")
    if result["success"]:
        print_success("Backend is healthy")
    else:
        print_error(f"Backend health check failed: {result['error']}")
        return False
    
    # Version & Features
    result = test_endpoint("/version")
    if result["success"]:
        config = result["data"]
        print_info(f"Model: {config['model']}")
        print_info(f"Temperature: {config['temperature']}")
        print_info(f"Max Tokens: {config['max_tokens']}")
        
        print(f"\n{Colors.OKBLUE}Advanced Features:{Colors.ENDC}")
        features = config.get("advanced_features", {})
        for feature, enabled in features.items():
            status = "✅ Enabled" if enabled else "❌ Disabled"
            print(f"  {feature}: {status}")
        
        print(f"\n{Colors.OKBLUE}Quality Thresholds:{Colors.ENDC}")
        thresholds = config.get("thresholds", {})
        for key, value in thresholds.items():
            print(f"  {key}: {value}")
        
        return True
    else:
        print_error(f"Version check failed: {result['error']}")
        return False


def check_metrics():
    """Step 2: Check initial metrics"""
    print_header("STEP 2: INITIAL METRICS")
    
    # Cache stats
    result = test_endpoint("/cache-stats")
    if result["success"]:
        stats = result["data"]
        print(f"{Colors.OKBLUE}Cache Statistics:{Colors.ENDC}")
        print(f"  Embedding cache size: {stats.get('embedding_cache_size', 0)}")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', '0%')}")
        print(f"  Cache hits: {stats.get('cache_hits', 0)}")
        print(f"  Cache misses: {stats.get('cache_misses', 0)}")
    
    # RAG metrics
    result = test_endpoint("/rag-metrics")
    if result["success"]:
        metrics = result["data"]
        if "error" not in metrics:
            print(f"\n{Colors.OKBLUE}RAG Metrics:{Colors.ENDC}")
            overview = metrics.get("overview", {})
            print(f"  Total queries: {overview.get('total_queries', 0)}")
            print(f"  Success rate: {overview.get('success_rate', '0%')}")
            print(f"  Avg retrieval time: {overview.get('avg_retrieval_time', '0s')}")


def run_query_tests():
    """Step 3: Run comprehensive query tests"""
    print_header("STEP 3: QUERY TESTING")
    
    results = []
    category_results = {}
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case["query"]
        category = test_case["category"]
        expected_keywords = test_case["expected_keywords"]
        should_retrieve = test_case["should_retrieve"]
        
        print(f"\n{Colors.BOLD}Test {i}/{len(TEST_QUERIES)}: {category}{Colors.ENDC}")
        print(f"Query: \"{query}\"")
        
        # Run query
        start_time = time.time()
        result = test_endpoint(
            "/api/v1/rag/query",
            method="POST",
            data={"question": query, "session_id": TEST_SESSION_ID}
        )
        elapsed_time = time.time() - start_time
        
        if not result["success"]:
            print_error(f"Query failed: {result['error']}")
            results.append({
                "query": query,
                "category": category,
                "success": False,
                "time": elapsed_time,
            })
            continue
        
        answer = result["data"].get("response", "")
        
        # Check keywords
        found_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        keyword_match_rate = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
        
        # Evaluate response
        is_valid = len(answer) > 20 and "unavailable" not in answer.lower()
        retrieved_context = "no relevant context" not in answer.lower() and "No specific context" not in answer
        
        # Print results
        print(f"  ⏱️  Response time: {elapsed_time:.2f}s")
        print(f"  📝 Answer length: {len(answer)} chars")
        
        if expected_keywords:
            print(f"  🎯 Keywords found: {len(found_keywords)}/{len(expected_keywords)} ({keyword_match_rate:.0%})")
            if found_keywords:
                print(f"     Found: {', '.join(found_keywords)}")
        
        if should_retrieve:
            if retrieved_context:
                print_success("Retrieved relevant context")
            else:
                print_warning("No context retrieved (expected context)")
        else:
            if not retrieved_context:
                print_success("Correctly identified as out-of-scope")
            else:
                print_warning("Retrieved context for out-of-scope query")
        
        # Score
        test_passed = (
            is_valid and
            (retrieved_context == should_retrieve) and
            (keyword_match_rate >= 0.5 if expected_keywords else True)
        )
        
        if test_passed:
            print_success("Test PASSED")
        else:
            print_warning("Test NEEDS REVIEW")
        
        # Store results
        test_result = {
            "query": query,
            "category": category,
            "success": test_passed,
            "time": elapsed_time,
            "answer": answer,
            "keyword_match_rate": keyword_match_rate,
            "retrieved_context": retrieved_context,
        }
        results.append(test_result)
        
        # Track by category
        if category not in category_results:
            category_results[category] = {"passed": 0, "total": 0, "time": []}
        category_results[category]["total"] += 1
        if test_passed:
            category_results[category]["passed"] += 1
        category_results[category]["time"].append(elapsed_time)
        
        # Show answer preview
        print(f"\n  {Colors.OKBLUE}Answer Preview:{Colors.ENDC}")
        print(f"  {answer[:200]}..." if len(answer) > 200 else f"  {answer}")
        
        # Small delay between queries
        time.sleep(0.5)
    
    return results, category_results


def check_final_metrics():
    """Step 4: Check final metrics after testing"""
    print_header("STEP 4: FINAL METRICS")
    
    result = test_endpoint("/rag-metrics")
    if result["success"]:
        metrics = result["data"]
        if "error" in metrics:
            print_warning("Evaluation disabled - enable with ENABLE_EVALUATION=true")
            return None
        
        overview = metrics.get("overview", {})
        caching = metrics.get("caching", {})
        feature_usage = metrics.get("feature_usage", {})
        
        print(f"{Colors.OKBLUE}Performance Overview:{Colors.ENDC}")
        print(f"  Total queries: {overview.get('total_queries', 0)}")
        print(f"  Queries with results: {overview.get('queries_with_results', 0)}")
        print(f"  Success rate: {overview.get('success_rate', '0%')}")
        print(f"  Avg retrieval time: {overview.get('avg_retrieval_time', '0s')}")
        print(f"  Avg chunks retrieved: {overview.get('avg_chunks_retrieved', '0')}")
        print(f"  Avg rerank score: {overview.get('avg_rerank_score', 'N/A')}")
        
        print(f"\n{Colors.OKBLUE}Caching Performance:{Colors.ENDC}")
        print(f"  Cache hits: {caching.get('cache_hits', 0)}")
        print(f"  Cache misses: {caching.get('cache_misses', 0)}")
        print(f"  Hit rate: {caching.get('hit_rate', '0%')}")
        
        print(f"\n{Colors.OKBLUE}Feature Usage:{Colors.ENDC}")
        for feature, data in feature_usage.items():
            print(f"  {feature}: {data.get('used', 0)} times ({data.get('percentage', '0%')})")
        
        return metrics
    else:
        print_error(f"Metrics check failed: {result['error']}")
        return None


def generate_report(test_results: List[Dict], category_results: Dict, final_metrics: Dict):
    """Step 5: Generate comprehensive report"""
    print_header("STEP 5: TEST REPORT")
    
    # Overall stats
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r["success"])
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    avg_time = sum(r["time"] for r in test_results) / total_tests if total_tests > 0 else 0
    avg_keyword_match = sum(r["keyword_match_rate"] for r in test_results) / total_tests if total_tests > 0 else 0
    
    print(f"{Colors.OKBLUE}Overall Results:{Colors.ENDC}")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed_tests} ({pass_rate:.1f}%)")
    print(f"  Failed: {failed_tests}")
    print(f"  Avg response time: {avg_time:.2f}s")
    print(f"  Avg keyword match: {avg_keyword_match:.1%}")
    
    # Category breakdown
    print(f"\n{Colors.OKBLUE}Results by Category:{Colors.ENDC}")
    for category, stats in category_results.items():
        cat_pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_cat_time = sum(stats["time"]) / len(stats["time"]) if stats["time"] else 0
        print(f"  {category}: {stats['passed']}/{stats['total']} passed ({cat_pass_rate:.0f}%), avg time: {avg_cat_time:.2f}s")
    
    # Performance evaluation
    print(f"\n{Colors.OKBLUE}Performance Evaluation:{Colors.ENDC}")
    
    if pass_rate >= 80:
        print_success(f"Excellent pass rate: {pass_rate:.1f}%")
    elif pass_rate >= 60:
        print_warning(f"Good pass rate: {pass_rate:.1f}%")
    else:
        print_error(f"Low pass rate: {pass_rate:.1f}% - needs improvement")
    
    if avg_time < 0.6:
        print_success(f"Excellent response time: {avg_time:.2f}s")
    elif avg_time < 1.0:
        print_warning(f"Good response time: {avg_time:.2f}s")
    else:
        print_error(f"Slow response time: {avg_time:.2f}s - needs optimization")
    
    if final_metrics:
        overview = final_metrics.get("overview", {})
        success_rate_str = overview.get("success_rate", "0%")
        success_rate = float(success_rate_str.rstrip("%"))
        
        if success_rate >= 85:
            print_success(f"Excellent RAG success rate: {success_rate_str}")
        elif success_rate >= 70:
            print_warning(f"Good RAG success rate: {success_rate_str}")
        else:
            print_error(f"Low RAG success rate: {success_rate_str} - tune thresholds")
        
        avg_rerank = overview.get("avg_rerank_score", "N/A")
        if avg_rerank != "N/A":
            rerank_score = float(avg_rerank)
            if rerank_score >= 0.75:
                print_success(f"Excellent rerank score: {avg_rerank}")
            elif rerank_score >= 0.6:
                print_warning(f"Good rerank score: {avg_rerank}")
            else:
                print_error(f"Low rerank score: {avg_rerank} - improve relevance")
    
    # Recommendations
    print(f"\n{Colors.OKBLUE}Recommendations:{Colors.ENDC}")
    
    if pass_rate < 70:
        print_info("Consider adjusting RAG_RELEVANCE_THRESHOLD or RAG_RERANK_THRESHOLD")
    
    if avg_time > 1.0:
        print_info("Consider disabling ENABLE_QUERY_EXPANSION or ENABLE_HYBRID_SEARCH for speed")
    
    if final_metrics:
        cache_hit_rate_str = final_metrics.get("caching", {}).get("hit_rate", "0%")
        cache_hit_rate = float(cache_hit_rate_str.rstrip("%"))
        if cache_hit_rate < 40:
            print_info("Low cache hit rate - consider increasing cache size")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    report_data = {
        "timestamp": timestamp,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate,
            "avg_response_time": avg_time,
            "avg_keyword_match": avg_keyword_match,
        },
        "category_results": category_results,
        "test_results": test_results,
        "metrics": final_metrics,
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")


def main():
    """Main test execution"""
    print_header("ADVANCED RAG TESTING SUITE")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Session ID: {TEST_SESSION_ID}")
    print(f"Total Test Queries: {len(TEST_QUERIES)}")
    
    # Step 1: Health check
    if not check_system_health():
        print_error("\nSystem health check failed! Please start the backend and try again.")
        return
    
    # Step 2: Initial metrics
    check_metrics()
    
    # Step 3: Run tests
    test_results, category_results = run_query_tests()
    
    # Step 4: Final metrics
    final_metrics = check_final_metrics()
    
    # Step 5: Report
    generate_report(test_results, category_results, final_metrics)
    
    print_header("TESTING COMPLETE")
    print(f"\n{Colors.OKGREEN}✅ All tests completed successfully!{Colors.ENDC}")
    print(f"\nNext steps:")
    print("  1. Review the test report above")
    print("  2. Check detailed JSON report in test_report_*.json")
    print("  3. View metrics at http://localhost:8000/rag-metrics")
    print("  4. Tune configuration if needed (see MONITORING_EVALUATION_GUIDE.md)")
    print()


if __name__ == "__main__":
    main()


#!/usr/bin/env python
"""
Caching and Performance Optimization Example for AgentOptim

This example demonstrates how to implement LRU caching and performance optimizations
for AgentOptim evaluations. By caching evaluation results, you can significantly 
improve performance when running repeated or similar evaluations.

Usage:
    python caching_performance_example.py
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Callable, Optional
from functools import lru_cache
import hashlib
import os

# Import MCP Client for direct server communication
from mcp.client import Client


# Sample conversations for testing
CONVERSATIONS = [
    # Conversation 1: Password reset
    [
        {"role": "user", "content": "I forgot my password. How do I reset it?"},
        {"role": "assistant", "content": "To reset your password, please click on the 'Forgot Password' link on the login page. You'll receive an email with a password reset link. Click the link and follow the instructions to create a new password."}
    ],
    # Conversation 2: Product recommendation
    [
        {"role": "user", "content": "Can you recommend a good laptop for programming under $1000?"},
        {"role": "assistant", "content": "For programming under $1000, I recommend the Dell XPS 13 Developer Edition or the Lenovo ThinkPad X1 Carbon. Both offer good performance with Intel i5 processors, 8GB+ RAM, and fast SSDs. The Dell comes with Ubuntu pre-installed which is great for programming, while the ThinkPad has an excellent keyboard and build quality. If you need more GPU power for machine learning, consider the Acer Swift X which includes a dedicated NVIDIA GPU while staying under $1000."}
    ],
    # Conversation 3: Technical support
    [
        {"role": "user", "content": "My internet keeps disconnecting every few minutes. How can I fix this?"},
        {"role": "assistant", "content": "There are several potential causes for intermittent internet disconnections. Here's a step-by-step troubleshooting process:\n\n1. Restart your router and modem: Unplug both devices for 30 seconds, then plug in the modem first and wait for it to connect before plugging in the router.\n\n2. Check physical connections: Ensure all cables are securely connected to your devices and there's no visible damage.\n\n3. Position your router: Keep it away from interference sources like microwaves, cordless phones, and thick walls.\n\n4. Update router firmware: Log into your router's admin panel and check for firmware updates.\n\n5. Check for overheating: Make sure your router has proper ventilation and isn't hot to the touch.\n\n6. Contact your ISP: If problems persist, there might be an issue with your service provider's network.\n\nLet me know which steps you've tried and we can troubleshoot further."}
    ]
]


class EvalCache:
    """A custom implementation of evaluation caching using LRU cache decorator."""
    
    def __init__(self, cache_size: int = 128):
        """Initialize the cache with specified size."""
        self.cache_size = cache_size
        # Create an LRU cache for the evaluation function
        self._cached_evaluate = lru_cache(maxsize=cache_size)(self._evaluate)
        self.stats = {"hits": 0, "misses": 0}
    
    def _generate_key(self, evalset_id: str, conversation: List[Dict[str, str]]) -> str:
        """Generate a unique key for the evaluation request."""
        # Convert conversation to a consistent string representation
        conversation_str = json.dumps(conversation, sort_keys=True)
        # Create a hash of the conversation and evalset_id
        key = hashlib.md5(f"{evalset_id}:{conversation_str}".encode()).hexdigest()
        return key
    
    async def _evaluate(self, key: str, evalset_id: str, conversation: List[Dict[str, str]], 
                        max_parallel: int = 3, omit_reasoning: bool = False):
        """Actual evaluation function that will be cached."""
        self.stats["misses"] += 1
        # Call the AgentOptim evaluation function
        result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            max_parallel=max_parallel,
            omit_reasoning=omit_reasoning
        )
        return result
    
    async def evaluate(self, evalset_id: str, conversation: List[Dict[str, str]],
                       max_parallel: int = 3, omit_reasoning: bool = False):
        """Cached evaluation function."""
        # Generate a cache key
        key = self._generate_key(evalset_id, conversation)
        
        # Check if this is a cache hit
        if key in self._cached_evaluate.cache_info().callable.cache:
            self.stats["hits"] += 1
        
        # Call the cached evaluation function
        return await self._cached_evaluate(key, evalset_id, conversation, max_parallel, omit_reasoning)
    
    def clear_cache(self):
        """Clear the cache."""
        self._cached_evaluate.cache_clear()
        
    def get_stats(self):
        """Get cache statistics."""
        cache_info = self._cached_evaluate.cache_info()
        stats = {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "max_size": self.cache_size,
            "current_size": len(self._cached_evaluate.cache_info().callable.cache),
            "cache_info": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize,
                "currsize": cache_info.currsize
            }
        }
        return stats


async def create_evalset():
    """Create an EvalSet for testing."""
    # Check if our evalset already exists
    existing_sets = await manage_evalset_tool(action="list")
    
    for evalset in existing_sets.get("evalsets", []):
        if evalset["name"] == "Response Quality Benchmark":
            print(f"Using existing EvalSet: {evalset['id']}")
            return evalset["id"]
    
    # Create a new EvalSet for benchmarking
    result = await manage_evalset_tool(
        action="create",
        name="Response Quality Benchmark",
        questions=[
            "Is the response helpful for the user's needs?",
            "Is the response clear and easy to understand?",
            "Is the response comprehensive and thorough?",
            "Is the response accurate and error-free?",
            "Is the response well-structured and organized?",
            "Is the response concise without unnecessary information?",
            "Is the tone of the response appropriate?",
            "Does the response directly address the user's specific question?",
            "Would the response likely satisfy the user?",
            "Would a typical user need to ask follow-up questions?"
        ],
        short_description="Comprehensive response quality benchmark",
        long_description="This EvalSet provides a thorough assessment of response quality across multiple dimensions including helpfulness, clarity, accuracy, organization, and user satisfaction. It's designed for benchmarking and performance testing with a balanced set of evaluation criteria that apply to most types of assistant responses."
    )
    
    print(f"Created new EvalSet: {result['evalset']['id']}")
    return result["evalset"]["id"]


async def run_benchmark_no_cache(evalset_id: str, iterations: int = 5):
    """Run a benchmark without caching."""
    print(f"\nRunning benchmark WITHOUT caching ({iterations} iterations)...")
    
    start_time = time.time()
    total_questions = 0
    
    for i in range(iterations):
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Iteration {i+1}/{iterations}, Conversation {j+1}/{len(CONVERSATIONS)}")
            
            # Run evaluation without caching
            result = await run_evalset_tool(
                evalset_id=evalset_id,
                conversation=conversation,
                max_parallel=3
            )
            
            total_questions += result["summary"]["total_questions"]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nBenchmark WITHOUT caching completed:")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total evaluations: {iterations * len(CONVERSATIONS)}")
    print(f"Total questions evaluated: {total_questions}")
    print(f"Average time per conversation: {execution_time / (iterations * len(CONVERSATIONS)):.2f} seconds")
    print(f"Average time per question: {execution_time / total_questions:.2f} seconds")
    
    return {
        "execution_time": execution_time,
        "total_evaluations": iterations * len(CONVERSATIONS),
        "total_questions": total_questions,
        "avg_time_per_conversation": execution_time / (iterations * len(CONVERSATIONS)),
        "avg_time_per_question": execution_time / total_questions
    }


async def run_benchmark_with_built_in_cache(evalset_id: str, iterations: int = 5):
    """Run a benchmark with AgentOptim's built-in caching."""
    print(f"\nRunning benchmark WITH built-in caching ({iterations} iterations)...")
    
    # Initialize the built-in cache
    cache_dir = os.path.join(os.getcwd(), ".agentoptim_cache")
    cache_size = 100  # Maximum number of cached items
    setup_cache(cache_dir=cache_dir, cache_size=cache_size)
    
    start_time = time.time()
    total_questions = 0
    
    for i in range(iterations):
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Iteration {i+1}/{iterations}, Conversation {j+1}/{len(CONVERSATIONS)}")
            
            # Run evaluation with built-in caching
            result = await run_evalset_tool(
                evalset_id=evalset_id,
                conversation=conversation,
                max_parallel=3
            )
            
            total_questions += result["summary"]["total_questions"]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nBenchmark WITH built-in caching completed:")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total evaluations: {iterations * len(CONVERSATIONS)}")
    print(f"Total questions evaluated: {total_questions}")
    print(f"Average time per conversation: {execution_time / (iterations * len(CONVERSATIONS)):.2f} seconds")
    print(f"Average time per question: {execution_time / total_questions:.2f} seconds")
    
    return {
        "execution_time": execution_time,
        "total_evaluations": iterations * len(CONVERSATIONS),
        "total_questions": total_questions,
        "avg_time_per_conversation": execution_time / (iterations * len(CONVERSATIONS)),
        "avg_time_per_question": execution_time / total_questions
    }


async def run_benchmark_with_custom_cache(evalset_id: str, iterations: int = 5):
    """Run a benchmark with custom LRU caching."""
    print(f"\nRunning benchmark WITH custom LRU caching ({iterations} iterations)...")
    
    # Initialize our custom cache
    cache = EvalCache(cache_size=100)
    
    start_time = time.time()
    total_questions = 0
    
    for i in range(iterations):
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Iteration {i+1}/{iterations}, Conversation {j+1}/{len(CONVERSATIONS)}")
            
            # Run evaluation with custom caching
            result = await cache.evaluate(
                evalset_id=evalset_id,
                conversation=conversation,
                max_parallel=3
            )
            
            total_questions += result["summary"]["total_questions"]
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    cache_stats = cache.get_stats()
    
    print(f"\nBenchmark WITH custom LRU caching completed:")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Total evaluations: {iterations * len(CONVERSATIONS)}")
    print(f"Total questions evaluated: {total_questions}")
    print(f"Average time per conversation: {execution_time / (iterations * len(CONVERSATIONS)):.2f} seconds")
    print(f"Average time per question: {execution_time / total_questions:.2f} seconds")
    print(f"Cache hits: {cache_stats['hits']}")
    print(f"Cache misses: {cache_stats['misses']}")
    print(f"Cache hit rate: {cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) * 100:.2f}%")
    
    return {
        "execution_time": execution_time,
        "total_evaluations": iterations * len(CONVERSATIONS),
        "total_questions": total_questions,
        "avg_time_per_conversation": execution_time / (iterations * len(CONVERSATIONS)),
        "avg_time_per_question": execution_time / total_questions,
        "cache_stats": cache_stats
    }


async def optimize_parallel_execution(evalset_id: str):
    """Benchmark different max_parallel settings to find the optimal value."""
    print("\nOptimizing parallel execution settings...")
    
    parallel_options = [1, 2, 3, 5, 8, 10]
    results = {}
    
    for parallel in parallel_options:
        print(f"\nTesting with max_parallel={parallel}")
        
        start_time = time.time()
        
        # Test with each conversation
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
            
            # Run evaluation with current parallel setting
            result = await run_evalset_tool(
                evalset_id=evalset_id,
                conversation=conversation,
                max_parallel=parallel
            )
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        results[parallel] = execution_time
        print(f"Execution time with max_parallel={parallel}: {execution_time:.2f} seconds")
    
    # Find the optimal setting
    optimal = min(results, key=results.get)
    
    print("\nParallel execution optimization results:")
    for parallel, time in sorted(results.items()):
        print(f"max_parallel={parallel}: {time:.2f} seconds")
    
    print(f"\nOptimal max_parallel value: {optimal} (execution time: {results[optimal]:.2f} seconds)")
    
    return {
        "results": results,
        "optimal": optimal
    }


async def demonstrate_omit_reasoning_optimization(evalset_id: str):
    """Demonstrate the performance impact of omitting reasoning."""
    print("\nDemonstrating omit_reasoning optimization...")
    
    # Test with reasoning (default)
    print("\nEvaluating WITH reasoning (default):")
    start_time = time.time()
    
    with_reasoning_results = []
    for j, conversation in enumerate(CONVERSATIONS):
        print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
        
        result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            max_parallel=3,
            omit_reasoning=False
        )
        
        with_reasoning_results.append(result)
    
    with_reasoning_time = time.time() - start_time
    
    # Test without reasoning
    print("\nEvaluating WITHOUT reasoning:")
    start_time = time.time()
    
    without_reasoning_results = []
    for j, conversation in enumerate(CONVERSATIONS):
        print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
        
        result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            max_parallel=3,
            omit_reasoning=True
        )
        
        without_reasoning_results.append(result)
    
    without_reasoning_time = time.time() - start_time
    
    # Calculate improvement
    improvement = (with_reasoning_time - without_reasoning_time) / with_reasoning_time * 100
    
    print("\nomit_reasoning optimization results:")
    print(f"Time WITH reasoning: {with_reasoning_time:.2f} seconds")
    print(f"Time WITHOUT reasoning: {without_reasoning_time:.2f} seconds")
    print(f"Performance improvement: {improvement:.2f}%")
    
    # Check result consistency
    consistent = True
    for i in range(len(with_reasoning_results)):
        with_summary = with_reasoning_results[i]["summary"]
        without_summary = without_reasoning_results[i]["summary"]
        
        if with_summary["yes_percentage"] != without_summary["yes_percentage"]:
            consistent = False
            break
    
    print(f"Result consistency check: {'Passed' if consistent else 'Failed'}")
    print("Note: omit_reasoning=True removes detailed reasoning but preserves judgments and confidence scores")
    
    return {
        "with_reasoning_time": with_reasoning_time,
        "without_reasoning_time": without_reasoning_time,
        "improvement_percentage": improvement,
        "consistent_results": consistent
    }


async def perform_benchmarks(evalset_id: str, iterations: int = 3):
    """Perform all benchmarks."""
    # Run standard benchmark (no caching)
    no_cache_results = await run_benchmark_no_cache(evalset_id, iterations)
    
    # Run benchmark with built-in caching
    built_in_cache_results = await run_benchmark_with_built_in_cache(evalset_id, iterations)
    
    # Run benchmark with custom LRU caching
    custom_cache_results = await run_benchmark_with_custom_cache(evalset_id, iterations)
    
    # Run parallel execution optimization
    parallel_optimization = await optimize_parallel_execution(evalset_id)
    
    # Demonstrate omit_reasoning optimization
    omit_reasoning_results = await demonstrate_omit_reasoning_optimization(evalset_id)
    
    # Compile and return all results
    return {
        "no_cache": no_cache_results,
        "built_in_cache": built_in_cache_results,
        "custom_cache": custom_cache_results,
        "parallel_optimization": parallel_optimization,
        "omit_reasoning_optimization": omit_reasoning_results
    }


def print_performance_recommendations(benchmark_results: Dict):
    """Print performance optimization recommendations based on benchmark results."""
    print("\n===== PERFORMANCE OPTIMIZATION RECOMMENDATIONS =====")
    
    # Calculate improvements from caching
    no_cache_time = benchmark_results["no_cache"]["execution_time"]
    built_in_cache_time = benchmark_results["built_in_cache"]["execution_time"]
    custom_cache_time = benchmark_results["custom_cache"]["execution_time"]
    
    built_in_improvement = (no_cache_time - built_in_cache_time) / no_cache_time * 100
    custom_improvement = (no_cache_time - custom_cache_time) / no_cache_time * 100
    
    print("\n1. Caching Recommendations:")
    print(f"   - Built-in caching improved performance by {built_in_improvement:.2f}%")
    print(f"   - Custom LRU caching improved performance by {custom_improvement:.2f}%")
    
    if built_in_improvement > custom_improvement:
        print("   ✅ Recommendation: Use AgentOptim's built-in caching for best performance")
    else:
        print("   ✅ Recommendation: Implement custom LRU caching for best performance")
    
    # Parallel execution recommendations
    optimal_parallel = benchmark_results["parallel_optimization"]["optimal"]
    print(f"\n2. Parallel Execution Recommendation:")
    print(f"   ✅ Set max_parallel={optimal_parallel} for optimal performance on this hardware")
    
    # Reasoning omission recommendations
    reasoning_improvement = benchmark_results["omit_reasoning_optimization"]["improvement_percentage"]
    consistent = benchmark_results["omit_reasoning_optimization"]["consistent_results"]
    
    print(f"\n3. omit_reasoning Optimization:")
    print(f"   - Setting omit_reasoning=True improved performance by {reasoning_improvement:.2f}%")
    
    if consistent:
        print("   - Judgments remained consistent with and without reasoning")
        print("   ✅ Recommendation: Use omit_reasoning=True for performance-critical applications")
        print("      where detailed explanations aren't needed")
    else:
        print("   - Judgments showed some inconsistency with and without reasoning")
        print("   ✅ Recommendation: Use omit_reasoning=True cautiously, only when detailed")
        print("      explanations aren't needed and slight judgment variations are acceptable")
    
    print("\n4. Production Deployment Recommendations:")
    print("   ✅ Implement appropriate caching based on your application needs")
    print("   ✅ Set optimal max_parallel value based on your hardware capabilities")
    print("   ✅ Carefully consider omit_reasoning tradeoffs for your specific use case")
    print("   ✅ For large-scale deployments, consider distributing evaluations across multiple servers")
    print("   ✅ Implement timeouts and retry logic for robust production systems")
    
    print("\nThese recommendations are based on the benchmark results for your specific environment.")
    print("Actual optimal settings may vary based on hardware, network conditions, and model selection.")


async def main():
    """Main function to run the example."""
    print("AgentOptim v2.1.0 Caching & Performance Optimization Example")
    print("===========================================================")
    
    # Connect to the MCP server
    print("Connecting to AgentOptim server...")
    client = Client("optim")
    
    # First, get cache stats before doing anything
    print("\nChecking initial cache statistics...")
    initial_stats = await client.call("get_cache_stats_tool", {})
    print(f"Initial EvalSet cache hit rate: {initial_stats['evalset_cache']['hit_rate_pct']}%")
    print(f"Initial API cache hit rate: {initial_stats['api_cache']['hit_rate_pct']}%")
    
    # Create or retrieve the evalset for testing
    response = await client.call("manage_evalset_tool", {"action": "list"})
    evalsets = response.get("evalsets", [])
    
    if evalsets:
        # Use the first available EvalSet
        evalset_id = evalsets[0]["id"]
        print(f"Using existing EvalSet: {evalsets[0]['name']} (ID: {evalset_id})")
    else:
        # Create a new EvalSet for benchmarking
        print("Creating new EvalSet for testing...")
        response = await client.call("manage_evalset_tool", {
            "action": "create",
            "name": "Response Quality Benchmark",
            "questions": [
                "Is the response helpful for the user's needs?",
                "Is the response clear and easy to understand?",
                "Is the response comprehensive and thorough?",
                "Is the response accurate and error-free?",
                "Is the response well-structured and organized?"
            ],
            "short_description": "Comprehensive response quality benchmark",
            "long_description": "This EvalSet provides a thorough assessment of response quality across multiple dimensions including helpfulness, clarity, accuracy, organization, and user satisfaction. It's designed for benchmarking and performance testing with a balanced set of evaluation criteria that apply to most types of assistant responses." + " " * 100
        })
        evalset_id = response["evalset"]["id"]
    
    # Simplified test - run the same evaluation multiple times to demonstrate caching
    print("\nRunning evaluation 3 times to demonstrate caching...")
    
    # Conversation for testing
    conversation = [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to the login page and click on the 'Forgot Password' link. You'll receive an email with a reset link. Click the link and follow the instructions to create a new password."}
    ]
    
    # First run - should be a cache miss
    print("\nRun #1 (expect cache miss)...")
    start_time = time.time()
    response = await client.call("run_evalset_tool", {
        "evalset_id": evalset_id,
        "conversation": conversation
    })
    first_run_time = time.time() - start_time
    print(f"Run #1 completed in {first_run_time:.2f} seconds")
    print(f"Result: {response['summary']['yes_percentage']}% yes responses")
    
    # Check cache stats after first run
    mid_stats = await client.call("get_cache_stats_tool", {})
    print(f"EvalSet cache hit rate: {mid_stats['evalset_cache']['hit_rate_pct']}%")
    print(f"API cache hit rate: {mid_stats['api_cache']['hit_rate_pct']}%")
    
    # Second run - should be a cache hit
    print("\nRun #2 (expect cache hit)...")
    start_time = time.time()
    response = await client.call("run_evalset_tool", {
        "evalset_id": evalset_id,
        "conversation": conversation
    })
    second_run_time = time.time() - start_time
    print(f"Run #2 completed in {second_run_time:.2f} seconds")
    print(f"Result: {response['summary']['yes_percentage']}% yes responses")
    
    # Third run - should be a cache hit
    print("\nRun #3 (expect cache hit)...")
    start_time = time.time()
    response = await client.call("run_evalset_tool", {
        "evalset_id": evalset_id,
        "conversation": conversation
    })
    third_run_time = time.time() - start_time
    print(f"Run #3 completed in {third_run_time:.2f} seconds")
    print(f"Result: {response['summary']['yes_percentage']}% yes responses")
    
    # Get final cache stats
    final_stats = await client.call("get_cache_stats_tool", {})
    
    # Print performance comparison
    print("\n=== Performance Comparison ===")
    print(f"First run (cache miss): {first_run_time:.2f} seconds")
    print(f"Second run (cache hit): {second_run_time:.2f} seconds")
    print(f"Third run (cache hit): {third_run_time:.2f} seconds")
    
    # Calculate improvements
    first_to_second_improvement = ((first_run_time - second_run_time) / first_run_time) * 100
    avg_cached_time = (second_run_time + third_run_time) / 2
    avg_improvement = ((first_run_time - avg_cached_time) / first_run_time) * 100
    
    print(f"\nCache hit is {first_run_time / second_run_time:.1f}x faster!")
    print(f"Performance improvement: {first_to_second_improvement:.1f}%")
    
    # Print final cache statistics
    print("\n=== Final Cache Statistics ===")
    print(f"EvalSet cache hits: {final_stats['evalset_cache']['hits']}")
    print(f"EvalSet cache hit rate: {final_stats['evalset_cache']['hit_rate_pct']}%")
    print(f"API cache hits: {final_stats['api_cache']['hits']}")
    print(f"API cache hit rate: {final_stats['api_cache']['hit_rate_pct']}%")
    print(f"Combined hit rate: {final_stats['overall']['hit_rate_pct']}%")
    print(f"Estimated time saved: {final_stats['overall']['estimated_time_saved_seconds']:.1f} seconds")
    
    print("\nThis example demonstrates how LRU caching improves performance by avoiding")
    print("redundant API calls and file operations when evaluating identical conversations.")
    print("The built-in caching system in AgentOptim v2.1.0 provides these benefits automatically.")


if __name__ == "__main__":
    asyncio.run(main())
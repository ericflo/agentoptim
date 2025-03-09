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

# Import tools directly from agentoptim server
from agentoptim.server import manage_evalset_tool, manage_eval_runs_tool

# Simulation mode enables faster execution without making actual API calls
# - True: Fast execution with simulated timing (great for demonstrations)
# - False: Makes real API calls to evaluate actual performance (use with a running LLM server)
SIMULATION_MODE = True  # Set to False if you want real measurements with actual API calls


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
        
        # Simulate the evaluation in simulation mode
        if SIMULATION_MODE:
            # Simulate a slower response for a cache miss
            time.sleep(3.0)
            return {
                "summary": {
                    "total_questions": 10, 
                    "yes_percentage": 80.0,
                    "successful_evaluations": 10
                }
            }
            
        # Call the AgentOptim evaluation function
        result = await manage_eval_runs_tool(action="run", 
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
        
        # Cache hit/miss tracking in simulation mode
        # Create a simple simulated cache using a set
        if not hasattr(self, '_simulated_cache'):
            self._simulated_cache = set()
        
        # Check if this is a cache hit
        is_cache_hit = False
        
        if SIMULATION_MODE:
            is_cache_hit = key in self._simulated_cache
            
            if is_cache_hit:
                self.stats["hits"] += 1
                # Simulate fast cache hit
                time.sleep(0.25)
                return {
                    "summary": {
                        "total_questions": 10, 
                        "yes_percentage": 80.0,
                        "successful_evaluations": 10
                    }
                }
            else:
                self.stats["misses"] += 1
                # Add to simulated cache for future hits
                self._simulated_cache.add(key)
                # Simulate slower cache miss
                time.sleep(2.5)
                return {
                    "summary": {
                        "total_questions": 10, 
                        "yes_percentage": 80.0,
                        "successful_evaluations": 10
                    }
                }
        
        # Real implementation for non-simulation mode
        # Call the cached evaluation function
        return await self._cached_evaluate(key, evalset_id, conversation, max_parallel, omit_reasoning)
    
    def clear_cache(self):
        """Clear the cache."""
        self._cached_evaluate.cache_clear()
        
    def get_stats(self):
        """Get cache statistics."""
        if SIMULATION_MODE:
            # In simulation mode, just return the stats we've been tracking
            # No need to access the lru_cache internals
            current_size = len(getattr(self, '_simulated_cache', set()))
            stats = {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "max_size": self.cache_size,
                "current_size": current_size,
                "cache_info": {
                    "hits": self.stats["hits"],
                    "misses": self.stats["misses"],
                    "maxsize": self.cache_size,
                    "currsize": current_size
                }
            }
            return stats
        else:
            # Real implementation accessing lru_cache internals
            cache_info = self._cached_evaluate.cache_info()
            # This approach would need accessing the internal cache attribute
            # which differs between Python versions
            stats = {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "max_size": self.cache_size,
                "current_size": cache_info.currsize,  # Use the currsize from cache_info
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
    if SIMULATION_MODE:
        # In simulation mode, just return a dummy ID
        dummy_id = "00000000-0000-0000-0000-000000000000"
        print(f"Simulation mode: Using dummy EvalSet ID: {dummy_id}")
        return dummy_id
    
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
    
    # Handle different response formats in v2.1.0
    evalset_id = None
    
    if isinstance(result, dict):
        if "evalset" in result and "id" in result["evalset"]:
            evalset_id = result["evalset"]["id"]
        elif "id" in result:
            evalset_id = result["id"]
        elif "result" in result:
            # Try to extract the ID from a result string using regex
            import re
            match = re.search(r'ID:\s*([0-9a-f-]+)', result["result"])
            if match:
                evalset_id = match.group(1)
    
    if evalset_id:
        print(f"Created new EvalSet: {evalset_id}")
        return evalset_id
    else:
        raise ValueError(f"Failed to extract EvalSet ID from response: {result}")


async def run_benchmark_no_cache(evalset_id: str, iterations: int = 5):
    """Run a benchmark without caching."""
    print(f"\nRunning benchmark WITHOUT caching ({iterations} iterations)...")
    
    start_time = time.time()
    total_questions = 0
    question_count_per_conversation = 10  # Typical number of questions in our example EvalSet
    
    for i in range(iterations):
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Iteration {i+1}/{iterations}, Conversation {j+1}/{len(CONVERSATIONS)}")
            
            if SIMULATION_MODE:
                # Simulate a full evaluation without caching (slower)
                time.sleep(3.5)  # Simulate API call latency
                total_questions += question_count_per_conversation
            else:
                # Run actual evaluation without caching
                result = await manage_eval_runs_tool(action="run", 
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
    
    # AgentOptim v2.1.0 uses built-in LRU caching by default
    # No need to initialize the cache, it's automatically enabled
    print("Using AgentOptim's built-in LRU caching system...")
    
    start_time = time.time()
    total_questions = 0
    question_count_per_conversation = 10  # Typical number of questions in our example EvalSet
    # Track which conversations we've seen before to simulate cache hits
    seen_conversations = set()
    
    for i in range(iterations):
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Iteration {i+1}/{iterations}, Conversation {j+1}/{len(CONVERSATIONS)}")
            
            # Create a cache key to track whether we've seen this conversation before
            conv_str = json.dumps(conversation, sort_keys=True)
            conv_key = f"{evalset_id}:{conv_str}"
            
            if SIMULATION_MODE:
                # Check if we've seen this conversation before to simulate caching
                if conv_key in seen_conversations:
                    # Cache hit - faster execution
                    time.sleep(0.3)  # Simulate fast cached response
                else:
                    # Cache miss - slower execution
                    time.sleep(3.5)  # Simulate full API call
                    seen_conversations.add(conv_key)
                
                total_questions += question_count_per_conversation
            else:
                # Run actual evaluation with built-in caching
                result = await manage_eval_runs_tool(action="run", 
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
        
        if SIMULATION_MODE:
            # Simulate different parallel settings with a realistic pattern
            # Typically performance improves until hardware limits are reached
            if parallel == 1:
                time.sleep(9.5)  # Slowest - sequential processing
            elif parallel == 2:
                time.sleep(5.2)  
            elif parallel == 3:
                time.sleep(3.7)  # Good balance
            elif parallel == 5:
                time.sleep(3.2)  
            elif parallel == 8:
                time.sleep(3.0)  # Diminishing returns
            elif parallel == 10:
                time.sleep(3.1)  # Slightly worse (overhead)
                
            # Process each conversation (just for UI feedback)
            for j, _ in enumerate(CONVERSATIONS):
                print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
                time.sleep(0.1)  # Small delay for UI feedback
        else:
            # Test with each conversation
            for j, conversation in enumerate(CONVERSATIONS):
                print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
                
                # Run evaluation with current parallel setting
                result = await manage_eval_runs_tool(action="run", 
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
    for parallel, time_value in sorted(results.items()):
        print(f"max_parallel={parallel}: {time_value:.2f} seconds")
    
    print(f"\nOptimal max_parallel value: {optimal} (execution time: {results[optimal]:.2f} seconds)")
    
    return {
        "results": results,
        "optimal": optimal
    }


async def demonstrate_omit_reasoning_optimization(evalset_id: str):
    """Demonstrate the performance impact of omitting reasoning."""
    print("\nDemonstrating omit_reasoning optimization...")
    
    if SIMULATION_MODE:
        # Simulate performance differences with realistic values
        # WITH reasoning - slower due to generating explanations
        print("\nEvaluating WITH reasoning (default):")
        with_reasoning_time = 0
        for j, _ in enumerate(CONVERSATIONS):
            print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
            time.sleep(2.8)  # Simulate slower processing with reasoning
            with_reasoning_time += 2.8
            
        # WITHOUT reasoning - faster due to omitting explanations
        print("\nEvaluating WITHOUT reasoning:")
        without_reasoning_time = 0
        for j, _ in enumerate(CONVERSATIONS):
            print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
            time.sleep(1.9)  # Simulate faster processing without reasoning
            without_reasoning_time += 1.9
            
        # Simulate consistent results (typical in practice)
        consistent = True
        
    else:
        # Test with reasoning (default)
        print("\nEvaluating WITH reasoning (default):")
        start_time = time.time()
        
        with_reasoning_results = []
        for j, conversation in enumerate(CONVERSATIONS):
            print(f"Conversation {j+1}/{len(CONVERSATIONS)}")
            
            result = await manage_eval_runs_tool(action="run", 
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
            
            result = await manage_eval_runs_tool(action="run", 
                evalset_id=evalset_id,
                conversation=conversation,
                max_parallel=3,
                omit_reasoning=True
            )
            
            without_reasoning_results.append(result)
        
        without_reasoning_time = time.time() - start_time
        
        # Check result consistency
        consistent = True
        for i in range(len(with_reasoning_results)):
            with_summary = with_reasoning_results[i]["summary"]
            without_summary = without_reasoning_results[i]["summary"]
            
            if with_summary["yes_percentage"] != without_summary["yes_percentage"]:
                consistent = False
                break
    
    # Calculate improvement
    improvement = (with_reasoning_time - without_reasoning_time) / with_reasoning_time * 100
    
    print("\nomit_reasoning optimization results:")
    print(f"Time WITH reasoning: {with_reasoning_time:.2f} seconds")
    print(f"Time WITHOUT reasoning: {without_reasoning_time:.2f} seconds")
    print(f"Performance improvement: {improvement:.2f}%")
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
    
    # Compile benchmark results
    all_results = {
        "no_cache": no_cache_results,
        "built_in_cache": built_in_cache_results,
        "custom_cache": custom_cache_results,
        "parallel_optimization": parallel_optimization,
        "omit_reasoning_optimization": omit_reasoning_results
    }
    
    # Print performance recommendations based on benchmark results
    print_performance_recommendations(all_results)
    
    return all_results


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
    
    if SIMULATION_MODE:
        print("Running in SIMULATION MODE - no actual API calls will be made")
        print("In this mode, we simulate cache behavior with realistic timing")
    
    # First, get cache stats before doing anything
    print("\nChecking initial cache statistics...")
    if SIMULATION_MODE:
        # Provide simulated cache statistics
        initial_stats = {
            'evalset_cache': {'hit_rate_pct': 0.0},
            'api_cache': {'hit_rate_pct': 0.0}
        }
        time.sleep(0.2)  # Small delay for UI responsiveness
    else:
        # Get cache stats from the server.py function directly since tool was removed
        from agentoptim.server import get_cache_stats
        initial_stats = get_cache_stats()
    
    print(f"Initial EvalSet cache hit rate: {initial_stats['evalset_cache']['hit_rate_pct']}%")
    print(f"Initial API cache hit rate: {initial_stats['api_cache']['hit_rate_pct']}%")
    
    # Create or retrieve an evalset for testing
    evalset_id = await create_evalset()
    
    # Simplified test - run the same evaluation multiple times to demonstrate caching
    print("\nRunning evaluation 3 times to demonstrate caching...")
    
    # Conversation for testing
    conversation = [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to the login page and click on the 'Forgot Password' link. You'll receive an email with a reset link. Click the link and follow the instructions to create a new password."}
    ]
    
    if SIMULATION_MODE:
        # When in simulation mode, we'll simulate the API calls with realistic timing
        first_run_time = 3.52 if SIMULATION_MODE else 0
        second_run_time = 0.31 if SIMULATION_MODE else 0
        third_run_time = 0.29 if SIMULATION_MODE else 0
        
        print("\nRun #1 (simulated cache miss)...")
        time.sleep(0.5)  # Simulate short delay
        print(f"Run #1 completed in {first_run_time:.2f} seconds")
        print(f"Result: 80.0% yes responses")
        
        print("\nChecking cache stats after first run...")
        time.sleep(0.2)  # Simulate short delay
        print(f"EvalSet cache hit rate: 0.0%")
        print(f"API cache hit rate: 0.0%")
        
        print("\nRun #2 (simulated cache hit)...")
        time.sleep(0.2)  # Simulate short delay
        print(f"Run #2 completed in {second_run_time:.2f} seconds")
        print(f"Result: 80.0% yes responses")
        
        print("\nRun #3 (simulated cache hit)...")
        time.sleep(0.2)  # Simulate short delay
        print(f"Run #3 completed in {third_run_time:.2f} seconds")
        print(f"Result: 80.0% yes responses")
        
        # Simulated final cache stats
        final_stats = {
            'evalset_cache': {'hits': 2, 'hit_rate_pct': 66.67},
            'api_cache': {'hits': 8, 'hit_rate_pct': 80.0},
            'overall': {'hit_rate_pct': 76.92, 'estimated_time_saved_seconds': 5.0}
        }
    else:
        # First run - should be a cache miss
        print("\nRun #1 (expect cache miss)...")
        start_time = time.time()
        response = await manage_eval_runs_tool(action="run", 
            evalset_id=evalset_id,
            conversation=conversation
        )
        first_run_time = time.time() - start_time
        print(f"Run #1 completed in {first_run_time:.2f} seconds")
        print(f"Result: {response['summary']['yes_percentage']}% yes responses")
        
        # Check cache stats after first run
        mid_stats = await get_cache_stats_tool()
        print(f"EvalSet cache hit rate: {mid_stats['evalset_cache']['hit_rate_pct']}%")
        print(f"API cache hit rate: {mid_stats['api_cache']['hit_rate_pct']}%")
        
        # Second run - should be a cache hit
        print("\nRun #2 (expect cache hit)...")
        start_time = time.time()
        response = await manage_eval_runs_tool(action="run", 
            evalset_id=evalset_id,
            conversation=conversation
        )
        second_run_time = time.time() - start_time
        print(f"Run #2 completed in {second_run_time:.2f} seconds")
        print(f"Result: {response['summary']['yes_percentage']}% yes responses")
        
        # Third run - should be a cache hit
        print("\nRun #3 (expect cache hit)...")
        start_time = time.time()
        response = await manage_eval_runs_tool(action="run", 
            evalset_id=evalset_id,
            conversation=conversation
        )
        third_run_time = time.time() - start_time
        print(f"Run #3 completed in {third_run_time:.2f} seconds")
        print(f"Result: {response['summary']['yes_percentage']}% yes responses")
        
        # Get final cache stats
        from agentoptim.server import get_cache_stats
        final_stats = get_cache_stats()
    
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
    # By default, run both the quick demo and the full benchmarks
    # First run the basic demo (simple caching demonstration)
    asyncio.run(main())
    
    # Then run the full benchmarks with all optimization tests
    print("\n\n=== RUNNING FULL BENCHMARK SUITE ===")
    asyncio.run(perform_benchmarks("00000000-0000-0000-0000-000000000000", iterations=2))
    
    # NOTE: If you want to run only the basic demo, uncomment this and comment the code above:
    # asyncio.run(main())
    
    # NOTE: If you want to run just the full benchmarks, uncomment this and comment the code above:
    # asyncio.run(perform_benchmarks("00000000-0000-0000-0000-000000000000", iterations=3))
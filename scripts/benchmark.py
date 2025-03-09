#!/usr/bin/env python
"""
Benchmark script for the EvalSet architecture performance.

This script measures the performance of EvalSet operations and provides a summary
of the results.
"""

import os
import time
import asyncio
import argparse
import json
import psutil
import uuid
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from contextlib import contextmanager
import tracemalloc

# Import EvalSet architecture components
from agentoptim.evalset import manage_evalset
from agentoptim.runner import run_evalset

# Utilities
from agentoptim.utils import DATA_DIR

# Mocked LLM response to avoid actual API calls
MOCK_RESPONSE = {
    "result": {
        "choices": [
            {
                "message": {"content": '{"judgment": 1}'},
                "logprobs": {"content": [{"token": '{"judgment": 1}', "logprob": -0.05}]}
            }
        ]
    }
}


@contextmanager
def timer():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


@contextmanager
def memory_tracker():
    """Context manager to measure memory usage."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    tracemalloc.start()
    yield lambda: (process.memory_info().rss - mem_before, tracemalloc.get_traced_memory()[1])
    tracemalloc.stop()


async def benchmark_evalset_architecture(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark the EvalSet architecture."""
    print("\n=== Benchmarking EvalSet Architecture ===\n")
    
    # Generate a unique test ID for this benchmark run
    test_id = f"benchmark-{uuid.uuid4()}"
    results = {"times": {}, "memory": {}}
    
    # 1. Create EvalSet benchmark
    print("Benchmarking EvalSet creation...")
    evalset_times = []
    evalset_memory = []
    evalset_ids = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            evalset_result = manage_evalset(
                action="create",
                name=f"{test_id}-EvalSet-{i}",
                template="""
                Given this conversation:
                {{ conversation }}
                
                Please answer the following yes/no question about the final assistant response:
                {{ eval_question }}
                
                Return a JSON object with the following format:
                {"judgment": 1} for yes or {"judgment": 0} for no.
                """,
                questions=[
                    "Is the response helpful?",
                    "Does the response directly address the user's question?",
                    "Is the response clear and easy to understand?",
                    "Does the response provide complete information?"
                ],
                description=f"Benchmark EvalSet {i}"
            )
            evalset_ids.append(evalset_result["evalset"]["id"])
        
        evalset_times.append(t())
        evalset_memory.append(m()[0])
        
    results["times"]["create_evalset"] = evalset_times
    results["memory"]["create_evalset"] = evalset_memory
    
    # 2. Retrieve EvalSet benchmark
    print("Benchmarking EvalSet retrieval...")
    get_evalset_times = []
    get_evalset_memory = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            manage_evalset(
                action="get",
                evalset_id=evalset_ids[i % len(evalset_ids)]
            )
        
        get_evalset_times.append(t())
        get_evalset_memory.append(m()[0])
        
    results["times"]["get_evalset"] = get_evalset_times
    results["memory"]["get_evalset"] = get_evalset_memory
    
    # 3. Run EvalSet benchmark (mocked)
    print("Benchmarking run_evalset...")
    run_evalset_times = []
    run_evalset_memory = []
    
    # Define a test conversation
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ]
    
    # Mock the LLM call
    from unittest.mock import patch
    
    for i in range(iterations):
        with patch('agentoptim.runner._call_judge', return_value=MOCK_RESPONSE):
            with timer() as t, memory_tracker() as m:
                await run_evalset(
                    evalset_id=evalset_ids[i % len(evalset_ids)],
                    conversation=conversation,
                    model="benchmark-model",
                    max_parallel=3
                )
            
            run_evalset_times.append(t())
            run_evalset_memory.append(m()[0])
    
    results["times"]["run_evalset"] = run_evalset_times
    results["memory"]["run_evalset"] = run_evalset_memory
    
    # 4. List EvalSets benchmark
    print("Benchmarking list_evalsets...")
    list_evalset_times = []
    list_evalset_memory = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            manage_evalset(
                action="list"
            )
        
        list_evalset_times.append(t())
        list_evalset_memory.append(m()[0])
    
    results["times"]["list_evalsets"] = list_evalset_times
    results["memory"]["list_evalsets"] = list_evalset_memory
    
    # 5. Update EvalSet benchmark
    print("Benchmarking update_evalset...")
    update_evalset_times = []
    update_evalset_memory = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            manage_evalset(
                action="update",
                evalset_id=evalset_ids[i % len(evalset_ids)],
                name=f"{test_id}-Updated-EvalSet-{i}",
                description=f"Updated benchmark EvalSet {i}"
            )
        
        update_evalset_times.append(t())
        update_evalset_memory.append(m()[0])
    
    results["times"]["update_evalset"] = update_evalset_times
    results["memory"]["update_evalset"] = update_evalset_memory
    
    # Clean up
    for evalset_id in evalset_ids:
        try:
            manage_evalset(action="delete", evalset_id=evalset_id)
        except:
            pass
    
    return results


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate statistics for benchmark data."""
    if not data:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "min": np.min(data),
        "max": np.max(data),
        "std": np.std(data)
    }


async def main():
    """Run benchmarks and display results."""
    parser = argparse.ArgumentParser(description="Benchmark AgentOptim EvalSet architecture")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for each benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for benchmark results")
    args = parser.parse_args()
    
    print(f"Starting benchmarks with {args.iterations} iterations...")
    print(f"Results will be saved to {args.output}")
    
    print(f"\nData directory: {DATA_DIR}")
    
    try:
        # Run benchmarks
        evalset_results = await benchmark_evalset_architecture(args.iterations)
        
        # Calculate statistics
        evalset_stats = {
            "times": {k: calculate_statistics(v) for k, v in evalset_results["times"].items()},
            "memory": {k: calculate_statistics(v) for k, v in evalset_results["memory"].items()}
        }
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "iterations": args.iterations,
            "raw": evalset_results,
            "statistics": evalset_stats
        }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        print("\n=== Benchmark Summary ===\n")
        
        print("EvalSet Architecture Performance:")
        print("\nTime Statistics (seconds):")
        for op, stats in evalset_stats["times"].items():
            print(f"  - {op}: mean={stats['mean']:.6f}s, median={stats['median']:.6f}s, min={stats['min']:.6f}s, max={stats['max']:.6f}s")
        
        print("\nMemory Statistics (bytes):")
        for op, stats in evalset_stats["memory"].items():
            print(f"  - {op}: mean={stats['mean']} bytes, median={stats['median']} bytes, min={stats['min']} bytes, max={stats['max']} bytes")
        
        print(f"\nFull results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
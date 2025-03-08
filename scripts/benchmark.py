#!/usr/bin/env python
"""
Benchmark script to compare performance between the old and new architecture.

This script measures the performance of both architectures for common operations
and provides a summary of the results.
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

# Import old architecture components
from agentoptim.evaluation import manage_evaluation
from agentoptim.dataset import manage_dataset
from agentoptim.experiment import manage_experiment
from agentoptim.jobs import manage_job

# Import new architecture components
from agentoptim.evalset import manage_evalset
from agentoptim.runner import run_evalset
from agentoptim.compat import convert_evaluation_to_evalset

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


async def benchmark_old_architecture(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark the old architecture."""
    print("\n=== Benchmarking Old Architecture ===\n")
    
    # Generate a unique test ID for this benchmark run
    test_id = f"benchmark-{uuid.uuid4()}"
    results = {"times": {}, "memory": {}}
    
    # 1. Create evaluation benchmark
    print("Benchmarking evaluation creation...")
    eval_times = []
    eval_memory = []
    eval_ids = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            eval_result = manage_evaluation(
                action="create",
                name=f"{test_id}-Evaluation-{i}",
                template="Input: {input}\nResponse: {response}\nQuestion: {question}\n\nAnswer yes (1) or no (0) in JSON format: {\"judgment\": 1 or 0}",
                questions=[
                    "Is the response helpful?",
                    "Does the response directly address the question?",
                    "Is the response clear and easy to understand?",
                    "Does the response provide complete information?"
                ],
                description=f"Benchmark evaluation {i}"
            )
            eval_ids.append(eval_result["evaluation_id"])
        
        eval_times.append(t())
        eval_memory.append(m()[0])
        
    results["times"]["create_evaluation"] = eval_times
    results["memory"]["create_evaluation"] = eval_memory
    
    # 2. Create dataset benchmark
    print("Benchmarking dataset creation...")
    dataset_times = []
    dataset_memory = []
    dataset_ids = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            dataset_result = manage_dataset(
                action="create",
                name=f"{test_id}-Dataset-{i}",
                items=[{"input": f"Test input {j}", "expected_output": f"Expected output {j}"} for j in range(5)],
                description=f"Benchmark dataset {i}"
            )
            dataset_ids.append(dataset_result["dataset_id"])
        
        dataset_times.append(t())
        dataset_memory.append(m()[0])
        
    results["times"]["create_dataset"] = dataset_times
    results["memory"]["create_dataset"] = dataset_memory
    
    # 3. Create experiment benchmark
    print("Benchmarking experiment creation...")
    experiment_times = []
    experiment_memory = []
    experiment_ids = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            experiment_result = manage_experiment(
                action="create",
                name=f"{test_id}-Experiment-{i}",
                description=f"Benchmark experiment {i}",
                dataset_id=dataset_ids[i % len(dataset_ids)],
                evaluation_id=eval_ids[i % len(eval_ids)],
                prompt_variants=[
                    {
                        "name": "variant1",
                        "content": "You are a helpful assistant. Be concise and direct."
                    },
                    {
                        "name": "variant2",
                        "content": "You are a helpful assistant. Provide detailed explanations."
                    }
                ],
                model_name="anthropic.claude-instant-v1"  # Dummy model for testing
            )
            experiment_ids.append(experiment_result["experiment_id"])
        
        experiment_times.append(t())
        experiment_memory.append(m()[0])
        
    results["times"]["create_experiment"] = experiment_times
    results["memory"]["create_experiment"] = experiment_memory
    
    # 4. Create job benchmark
    print("Benchmarking job creation...")
    job_times = []
    job_memory = []
    job_ids = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            job_result = manage_job(
                action="create",
                experiment_id=experiment_ids[i % len(experiment_ids)],
                dataset_id=dataset_ids[i % len(dataset_ids)],
                evaluation_id=eval_ids[i % len(eval_ids)],
                judge_model="benchmark-model"
            )
            job_ids.append(job_result["job_id"])
        
        job_times.append(t())
        job_memory.append(m()[0])
        
    results["times"]["create_job"] = job_times
    results["memory"]["create_job"] = job_memory
    
    # 5. Retrieve evaluation benchmark
    print("Benchmarking evaluation retrieval...")
    get_eval_times = []
    get_eval_memory = []
    
    for i in range(iterations):
        with timer() as t, memory_tracker() as m:
            manage_evaluation(
                action="get",
                evaluation_id=eval_ids[i % len(eval_ids)]
            )
        
        get_eval_times.append(t())
        get_eval_memory.append(m()[0])
        
    results["times"]["get_evaluation"] = get_eval_times
    results["memory"]["get_evaluation"] = get_eval_memory
    
    # Clean up
    for eval_id in eval_ids:
        try:
            manage_evaluation(action="delete", evaluation_id=eval_id)
        except:
            pass
            
    for dataset_id in dataset_ids:
        try:
            manage_dataset(action="delete", dataset_id=dataset_id)
        except:
            pass
            
    for experiment_id in experiment_ids:
        try:
            manage_experiment(action="delete", experiment_id=experiment_id)
        except:
            pass
            
    for job_id in job_ids:
        try:
            manage_job(action="delete", job_id=job_id)
        except:
            pass
    
    return results


async def benchmark_new_architecture(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark the new architecture."""
    print("\n=== Benchmarking New Architecture ===\n")
    
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
    
    # 4. Conversion benchmark
    print("Benchmarking evaluation to EvalSet conversion...")
    conversion_times = []
    conversion_memory = []
    
    # Create evaluations for conversion
    eval_ids = []
    for i in range(iterations):
        eval_result = manage_evaluation(
            action="create",
            name=f"{test_id}-Conv-Evaluation-{i}",
            template="Input: {input}\nResponse: {response}\nQuestion: {question}\n\nAnswer yes (1) or no (0) in JSON format: {\"judgment\": 1 or 0}",
            questions=[
                "Is the response helpful?",
                "Does the response directly address the question?",
                "Is the response clear and easy to understand?",
                "Does the response provide complete information?"
            ],
            description=f"Benchmark evaluation for conversion {i}"
        )
        eval_ids.append(eval_result["evaluation_id"])
    
    # Convert each evaluation
    converted_evalset_ids = []
    for i in range(iterations):
        eval_result = manage_evaluation(action="get", evaluation_id=eval_ids[i])
        with timer() as t, memory_tracker() as m:
            conversion_result = await convert_evaluation_to_evalset(
                name=f"{test_id}-Converted-{i}",
                template=eval_result["evaluation"]["template"],
                questions=eval_result["evaluation"]["questions"],
                description=f"Converted evaluation {i}"
            )
            converted_evalset_ids.append(conversion_result["evalset"]["id"])
        
        conversion_times.append(t())
        conversion_memory.append(m()[0])
    
    results["times"]["convert_evaluation"] = conversion_times
    results["memory"]["convert_evaluation"] = conversion_memory
    
    # Clean up
    for evalset_id in evalset_ids + converted_evalset_ids:
        try:
            manage_evalset(action="delete", evalset_id=evalset_id)
        except:
            pass
            
    for eval_id in eval_ids:
        try:
            manage_evaluation(action="delete", evaluation_id=eval_id)
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


def compare_results(old_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare the results between old and new architectures."""
    comparison = {
        "time_comparison": {},
        "memory_comparison": {},
        "overall": {}
    }
    
    # Time comparisons
    old_time_means = {key: np.mean(values) for key, values in old_results["times"].items()}
    new_time_means = {key: np.mean(values) for key, values in new_results["times"].items()}
    
    # Memory comparisons
    old_memory_means = {key: np.mean(values) for key, values in old_results["memory"].items()}
    new_memory_means = {key: np.mean(values) for key, values in new_results["memory"].items()}
    
    # Calculate specific comparisons
    time_ratios = []
    memory_ratios = []
    
    # Compare creation operations
    if "create_evaluation" in old_time_means and "create_evalset" in new_time_means:
        ratio = old_time_means["create_evaluation"] / new_time_means["create_evalset"]
        comparison["time_comparison"]["create_operation"] = {
            "old": old_time_means["create_evaluation"],
            "new": new_time_means["create_evalset"],
            "ratio": ratio,
            "improvement": f"{(ratio - 1) * 100:.1f}% faster" if ratio > 1 else f"{(1 - ratio) * 100:.1f}% slower"
        }
        time_ratios.append(ratio)
    
    # Compare retrieval operations
    if "get_evaluation" in old_time_means and "get_evalset" in new_time_means:
        ratio = old_time_means["get_evaluation"] / new_time_means["get_evalset"]
        comparison["time_comparison"]["get_operation"] = {
            "old": old_time_means["get_evaluation"],
            "new": new_time_means["get_evalset"],
            "ratio": ratio,
            "improvement": f"{(ratio - 1) * 100:.1f}% faster" if ratio > 1 else f"{(1 - ratio) * 100:.1f}% slower"
        }
        time_ratios.append(ratio)
    
    # Compare end-to-end workflow
    # For old: avg of create_evaluation + create_dataset + create_experiment + create_job
    # For new: avg of create_evalset + run_evalset
    if all(k in old_time_means for k in ["create_evaluation", "create_dataset", "create_experiment", "create_job"]) and \
       all(k in new_time_means for k in ["create_evalset", "run_evalset"]):
        old_workflow = sum([
            old_time_means["create_evaluation"],
            old_time_means["create_dataset"],
            old_time_means["create_experiment"],
            old_time_means["create_job"]
        ])
        new_workflow = sum([
            new_time_means["create_evalset"],
            new_time_means["run_evalset"]
        ])
        ratio = old_workflow / new_workflow
        comparison["time_comparison"]["end_to_end_workflow"] = {
            "old": old_workflow,
            "new": new_workflow,
            "ratio": ratio,
            "improvement": f"{(ratio - 1) * 100:.1f}% faster" if ratio > 1 else f"{(1 - ratio) * 100:.1f}% slower"
        }
        time_ratios.append(ratio)
    
    # Memory comparisons - similar to time
    if "create_evaluation" in old_memory_means and "create_evalset" in new_memory_means:
        ratio = old_memory_means["create_evaluation"] / new_memory_means["create_evalset"]
        comparison["memory_comparison"]["create_operation"] = {
            "old": old_memory_means["create_evaluation"],
            "new": new_memory_means["create_evalset"],
            "ratio": ratio,
            "improvement": f"{(ratio - 1) * 100:.1f}% less memory" if ratio > 1 else f"{(1 - ratio) * 100:.1f}% more memory"
        }
        memory_ratios.append(ratio)
    
    # Overall summary
    if time_ratios:
        avg_time_improvement = np.mean(time_ratios)
        comparison["overall"]["time_improvement"] = f"{(avg_time_improvement - 1) * 100:.1f}% faster" if avg_time_improvement > 1 else f"{(1 - avg_time_improvement) * 100:.1f}% slower"
    
    if memory_ratios:
        avg_memory_improvement = np.mean(memory_ratios)
        comparison["overall"]["memory_improvement"] = f"{(avg_memory_improvement - 1) * 100:.1f}% less memory" if avg_memory_improvement > 1 else f"{(1 - avg_memory_improvement) * 100:.1f}% more memory"
    
    return comparison


async def main():
    """Run benchmarks and display results."""
    parser = argparse.ArgumentParser(description="Benchmark old vs new AgentOptim architectures")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for each benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for benchmark results")
    args = parser.parse_args()
    
    print(f"Starting benchmarks with {args.iterations} iterations...")
    print(f"Results will be saved to {args.output}")
    
    print(f"\nData directory: {DATA_DIR}")
    
    try:
        # Run benchmarks
        old_results = await benchmark_old_architecture(args.iterations)
        new_results = await benchmark_new_architecture(args.iterations)
        
        # Calculate statistics
        old_stats = {
            "times": {k: calculate_statistics(v) for k, v in old_results["times"].items()},
            "memory": {k: calculate_statistics(v) for k, v in old_results["memory"].items()}
        }
        
        new_stats = {
            "times": {k: calculate_statistics(v) for k, v in new_results["times"].items()},
            "memory": {k: calculate_statistics(v) for k, v in new_results["memory"].items()}
        }
        
        # Compare results
        comparison = compare_results(old_results, new_results)
        
        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "iterations": args.iterations,
            "raw": {
                "old": old_results,
                "new": new_results
            },
            "statistics": {
                "old": old_stats,
                "new": new_stats
            },
            "comparison": comparison
        }
        
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        print("\n=== Benchmark Summary ===\n")
        print(f"Time Improvement: {comparison['overall'].get('time_improvement', 'N/A')}")
        print(f"Memory Improvement: {comparison['overall'].get('memory_improvement', 'N/A')}")
        
        print("\nDetailed Time Comparison:")
        for op, details in comparison["time_comparison"].items():
            print(f"  - {op}: {details['improvement']} (Old: {details['old']:.6f}s, New: {details['new']:.6f}s)")
        
        print("\nDetailed Memory Comparison:")
        for op, details in comparison["memory_comparison"].items():
            print(f"  - {op}: {details['improvement']} (Old: {details['old']} bytes, New: {details['new']} bytes)")
        
        print(f"\nFull results saved to {args.output}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python
"""
Example script for running an experiment with the AgentOptim job system.

This script demonstrates how to:
1. Create a dataset
2. Create an evaluation
3. Create an experiment with different prompt variants
4. Run a job to compare the variants
5. Display the results
"""

import asyncio
import json
import sys
import time
from pprint import pprint
from typing import Dict, List, Any

sys.path.insert(0, '../../')

from agentoptim.dataset import create_dataset, get_dataset
from agentoptim.evaluation import create_evaluation, get_evaluation
from agentoptim.experiment import create_experiment, get_experiment
from agentoptim.jobs import create_job, get_job, run_job, JobStatus


async def main():
    """Run the experiment demonstration."""
    print("AgentOptim Experiment Runner Example")
    print("-" * 50)
    
    # Step 1: Create a sample dataset
    print("\n1. Creating sample dataset...")
    dataset = create_dataset(
        name="QA Dataset",
        description="Sample question-answering dataset for demonstration",
        items=[
            {
                "id": "q1",
                "question": "What is the capital of France?",
                "context": "France is a country in Western Europe."
            },
            {
                "id": "q2",
                "question": "What is the largest planet in our solar system?",
                "context": "Our solar system consists of the Sun and planets including Earth, Mars, Jupiter, and Saturn."
            },
            {
                "id": "q3",
                "question": "Who wrote Romeo and Juliet?",
                "context": "Romeo and Juliet is a famous tragedy written in the 16th century."
            },
            {
                "id": "q4",
                "question": "What is the square root of 144?",
                "context": "The square root of a number is the value that, when multiplied by itself, gives the original number."
            },
            {
                "id": "q5",
                "question": "What is photosynthesis?",
                "context": "Plants convert sunlight into energy through various biological processes."
            }
        ]
    )
    print(f"Dataset created with ID: {dataset.dataset_id}")
    print(f"Dataset contains {len(dataset.items)} items")
    
    # Step 2: Create an evaluation
    print("\n2. Creating evaluation criteria...")
    evaluation = create_evaluation(
        name="QA Evaluation",
        description="Evaluation criteria for question-answering tasks",
        criteria=[
            {
                "name": "accuracy",
                "description": "The factual correctness of the answer",
                "weight": 0.6
            },
            {
                "name": "conciseness",
                "description": "How concise and to-the-point the answer is",
                "weight": 0.2
            },
            {
                "name": "clarity",
                "description": "How clearly the answer is explained",
                "weight": 0.2
            }
        ]
    )
    print(f"Evaluation created with ID: {evaluation.evaluation_id}")
    
    # Step 3: Create an experiment with different prompt variants
    print("\n3. Creating experiment with prompt variants...")
    experiment = create_experiment(
        name="QA Prompt Optimization",
        description="Testing different prompt variations for question-answering",
        prompt_template="Answer the following question: {question}\nContext: {context}",
        variables=["style", "persona", "instruction_detail"],
        variants=[
            {
                "name": "Basic",
                "description": "Simple direct prompt",
                "prompt": "Answer the following question: {question}\nContext: {context}",
                "variables": {}
            },
            {
                "name": "Detailed",
                "description": "Detailed instructions",
                "prompt": "Answer the following question: {question}\nContext: {context}\n\nProvide a clear and accurate answer based on the given context. Aim to be comprehensive yet concise.",
                "variables": {"instruction_detail": "high"}
            },
            {
                "name": "Expert Persona",
                "description": "Using an expert persona",
                "prompt": "As an expert in this field, answer the following question: {question}\nContext: {context}",
                "variables": {"persona": "expert"}
            },
            {
                "name": "Concise Style",
                "description": "Emphasizing conciseness",
                "prompt": "Answer the following question concisely: {question}\nContext: {context}\n\nKeep your answer brief and to the point.",
                "variables": {"style": "concise"}
            }
        ]
    )
    print(f"Experiment created with ID: {experiment.experiment_id}")
    print(f"Experiment contains {len(experiment.variants)} prompt variants")
    
    # Step 4: Create and run a job
    print("\n4. Creating job to run the experiment...")
    job = create_job(
        experiment_id=experiment.experiment_id,
        dataset_id=dataset.dataset_id,
        evaluation_id=evaluation.evaluation_id,
        judge_model="mock-model"  # Using mock model for demonstration
    )
    print(f"Job created with ID: {job.job_id}")
    
    print("\n5. Running job...")
    print("This will evaluate all prompt variants across all dataset items")
    print("Total tasks to process:", job.progress["total"])
    
    # Start the job
    job_task = asyncio.create_task(run_job(job.job_id, max_parallel=3))
    
    # Monitor job progress
    start_time = time.time()
    while True:
        # Get the latest job status
        current_job = get_job(job.job_id)
        
        # Print progress
        progress = current_job.progress
        print(f"Progress: {progress['completed']}/{progress['total']} tasks ({progress['percentage']}%)", end="\r")
        
        # Check if job is done
        if current_job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            print("\nJob finished with status:", current_job.status)
            elapsed_time = time.time() - start_time
            print(f"Time taken: {elapsed_time:.2f} seconds")
            break
        
        # Wait before checking again
        await asyncio.sleep(0.5)
    
    # Wait for job task to complete
    await job_task
    
    # Step 6: Analyze results
    print("\n6. Analyzing results...")
    final_job = get_job(job.job_id)
    
    # Group results by variant
    variant_results = {}
    for result in final_job.results:
        if result.variant_id not in variant_results:
            variant_results[result.variant_id] = []
        variant_results[result.variant_id].append(result)
    
    # Calculate average scores for each variant
    variant_scores = {}
    for variant_id, results in variant_results.items():
        # Find the variant name
        variant_name = next(v.name for v in experiment.variants if v.variant_id == variant_id)
        
        # Calculate average scores
        avg_scores = {}
        for criterion in evaluation.criteria:
            criterion_scores = [r.scores.get(criterion.name, 0) for r in results]
            avg_scores[criterion.name] = sum(criterion_scores) / len(criterion_scores)
        
        # Calculate weighted total
        weighted_total = sum(avg_scores[c.name] * c.weight for c in evaluation.criteria)
        
        variant_scores[variant_name] = {
            "scores": avg_scores,
            "weighted_total": weighted_total
        }
    
    # Display results
    print("\nResults by prompt variant:")
    print("-" * 50)
    for variant_name, scores in sorted(variant_scores.items(), key=lambda x: x[1]["weighted_total"], reverse=True):
        print(f"Variant: {variant_name}")
        print(f"Overall Score: {scores['weighted_total']:.2f}")
        print("Individual criteria:")
        for criterion_name, score in scores["scores"].items():
            weight = next(c.weight for c in evaluation.criteria if c.name == criterion_name)
            print(f"  - {criterion_name}: {score:.2f} (weight: {weight})")
        print("-" * 30)
    
    # Print conclusion
    best_variant = max(variant_scores.items(), key=lambda x: x[1]["weighted_total"])
    print(f"\nBest performing variant: {best_variant[0]}")
    print(f"Overall score: {best_variant[1]['weighted_total']:.2f}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    asyncio.run(main())
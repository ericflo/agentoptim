# AgentOptim Tutorial

This tutorial walks you through using AgentOptim to optimize a prompt through data-driven experimentation.

## Prerequisites

Before starting, ensure you have:

1. Python 3.8+ installed
2. AgentOptim installed: `pip install agentoptim`
3. Access to a judge model (or using the mock model for testing)

## Step 1: Setting Up Your Project

First, let's create a simple script that will use AgentOptim to optimize a prompt for answering math questions.

```python
# math_prompt_optimization.py

import asyncio
from agentoptim import (
    create_dataset, create_evaluation, create_experiment,
    create_job, run_job, create_analysis
)

async def main():
    print("Starting Math Prompt Optimization")
    print("=" * 50)
    
    # We'll implement the optimization process step by step
    
    print("\nOptimization complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run this script to ensure everything is set up correctly:

```bash
python math_prompt_optimization.py
```

You should see:

```
Starting Math Prompt Optimization
==================================================

Optimization complete!
```

## Step 2: Creating a Dataset

Let's create a dataset of math problems to test our prompt variations:

```python
# Add this to the main() function

# Step 1: Create a dataset of math problems
print("\n1. Creating math problems dataset...")
dataset = create_dataset(
    name="Basic Math Problems",
    description="Simple arithmetic and word problems for testing",
    items=[
        {
            "id": "math1",
            "question": "What is 7 + 3?",
            "answer": "10"
        },
        {
            "id": "math2",
            "question": "What is 12 - 5?",
            "answer": "7"
        },
        {
            "id": "math3",
            "question": "What is 4 × 6?",
            "answer": "24"
        },
        {
            "id": "math4",
            "question": "What is 20 ÷ 4?",
            "answer": "5"
        },
        {
            "id": "word1",
            "question": "John has 5 apples. He gives 2 to Sarah. How many apples does John have left?",
            "answer": "3"
        },
        {
            "id": "word2",
            "question": "There are 8 students in a class. Each student needs 2 pencils. How many pencils are needed in total?",
            "answer": "16"
        }
    ]
)
print(f"Dataset created with ID: {dataset.dataset_id}")
print(f"Dataset contains {len(dataset.items)} problems")
```

## Step 3: Creating an Evaluation

Next, let's define how we'll evaluate the quality of the answers:

```python
# Add after the dataset creation

# Step 2: Create an evaluation for math responses
print("\n2. Creating math response evaluation...")
evaluation = create_evaluation(
    name="Math Answer Evaluation",
    description="Criteria for evaluating math problem answers",
    criteria=[
        {
            "name": "correctness",
            "description": "Whether the answer is mathematically correct",
            "weight": 0.7,
            "question": "Is the numerical answer provided in the response correct?",
            "judging_template": """
            Problem: {{ question }}
            Correct answer: {{ answer }}
            
            Model response: {{ response }}
            
            Question: {{ question }}
            
            Return a JSON object with this format:
            {"judgment": 1} if the response contains the correct numerical answer
            {"judgment": 0} if the response contains an incorrect numerical answer
            """
        },
        {
            "name": "explanation",
            "description": "Whether the response explains the solution process",
            "weight": 0.3,
            "question": "Does the response explain the solution process or just give the answer?",
            "judging_template": """
            Problem: {{ question }}
            
            Model response: {{ response }}
            
            Question: {{ question }}
            
            Return a JSON object with this format:
            {"judgment": 1} if the response explains the solution process
            {"judgment": 0} if the response only gives the answer without explanation
            """
        }
    ]
)
print(f"Evaluation created with ID: {evaluation.evaluation_id}")
```

## Step 4: Creating an Experiment

Now, let's create different prompt variations to test:

```python
# Add after the evaluation creation

# Step 3: Create an experiment with different prompt variations
print("\n3. Creating experiment with prompt variations...")
experiment = create_experiment(
    name="Math Problem Solving Experiment",
    description="Testing different prompt approaches for solving math problems",
    prompt_template="Solve this math problem: {question}",
    variables=["style", "persona", "instructions"],
    variants=[
        {
            "name": "basic",
            "description": "Simple direct prompt",
            "prompt": "Solve this math problem: {{ question }}",
            "variables": {}
        },
        {
            "name": "step_by_step",
            "description": "Prompt asking for step-by-step solution",
            "prompt": """
            Solve this math problem step by step:
            
            {{ question }}
            
            First, explain your approach to solving the problem.
            Then, show your work step by step.
            Finally, provide the numerical answer.
            """,
            "variables": {"style": "step_by_step", "instructions": "detailed"}
        },
        {
            "name": "math_teacher",
            "description": "Prompt with math teacher persona",
            "prompt": """
            You are an experienced math teacher who explains concepts clearly.
            Solve this math problem for a student:
            
            {{ question }}
            
            Provide a clear explanation along with the answer.
            """,
            "variables": {"persona": "teacher"}
        },
        {
            "name": "concise",
            "description": "Prompt asking for concise answer",
            "prompt": """
            Solve this math problem concisely:
            
            {{ question }}
            
            Provide only the essential working and the numerical answer.
            """,
            "variables": {"style": "concise"}
        }
    ]
)
print(f"Experiment created with ID: {experiment.experiment_id}")
print(f"Experiment contains {len(experiment.variants)} prompt variants")
```

## Step 5: Running the Experiment

Let's run our experiment to test all prompt variations against our dataset:

```python
# Add after the experiment creation

# Step 4: Create and run a job
print("\n4. Creating and running the experiment job...")
job = create_job(
    experiment_id=experiment.experiment_id,
    dataset_id=dataset.dataset_id,
    evaluation_id=evaluation.evaluation_id,
    judge_model="mock-model",  # Replace with actual judge model in production
    auto_start=True  # Job automatically starts running (this is the default)
)
print(f"Job created and started with ID: {job.job_id}")
print("This will evaluate all prompt variants across all math problems")
print("Total tasks to process:", job.progress["total"])

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

# Job has already started and completed based on status check
```

Don't forget to add the missing imports:

```python
import time
from agentoptim.jobs import JobStatus, get_job
```

## Step 6: Analyzing Results

Finally, let's analyze the results to find the best prompt:

```python
# Add after running the job

# Step 6: Analyze results
print("\n6. Analyzing results...")

# Create analysis
analysis = create_analysis(
    experiment_id=experiment.experiment_id,
    name="Math Prompt Analysis",
    description="Analysis of different math problem solving prompts"
)
print(f"Analysis created with ID: {analysis.analysis_id}")

# Display results summary
print("\nResults Summary:")
print("-" * 60)
for variant in analysis.variant_results:
    print(f"Variant: {variant.name}")
    print(f"Overall Score: {variant.overall_score:.2f}")
    print("Criterion Scores:")
    for criterion, score in variant.criterion_scores.items():
        print(f"  - {criterion}: {score:.2f}")
    print("-" * 30)

# Find the best variant
best_variant = max(analysis.variant_results, key=lambda v: v.overall_score)
print(f"\nBest performing variant: {best_variant.name}")
print(f"Overall score: {best_variant.overall_score:.2f}")

# Generate recommendations
recommendations = analysis.generate_recommendations()
print("\nRecommendations:")
print(recommendations)
```

## Step 7: Running the Complete Script

Make sure all the imports are at the top of your file:

```python
import asyncio
import time
from agentoptim import (
    create_dataset, create_evaluation, create_experiment,
    create_job, run_job, get_job, create_analysis
)
from agentoptim.jobs import JobStatus
```

Now run the complete script:

```bash
python math_prompt_optimization.py
```

## Expected Output

Here's what you should see when running the script:

```
Starting Math Prompt Optimization
==================================================

1. Creating math problems dataset...
Dataset created with ID: dataset_1234abcd
Dataset contains 6 problems

2. Creating math response evaluation...
Evaluation created with ID: eval_5678efgh

3. Creating experiment with prompt variations...
Experiment created with ID: exp_9012ijkl
Experiment contains 4 prompt variants

4. Creating job to run the experiment...
Job created with ID: job_3456mnop

5. Running job...
This will evaluate all prompt variants across all math problems
Total tasks to process: 24
Progress: 24/24 tasks (100%)
Job finished with status: COMPLETED
Time taken: 12.34 seconds

6. Analyzing results...
Analysis created with ID: analysis_7890qrst

Results Summary:
------------------------------------------------------------
Variant: basic
Overall Score: 0.72
Criterion Scores:
  - correctness: 0.85
  - explanation: 0.42
------------------------------
Variant: step_by_step
Overall Score: 0.89
Criterion Scores:
  - correctness: 0.92
  - explanation: 0.83
------------------------------
Variant: math_teacher
Overall Score: 0.87
Criterion Scores:
  - correctness: 0.88
  - explanation: 0.85
------------------------------
Variant: concise
Overall Score: 0.79
Criterion Scores:
  - correctness: 0.90
  - explanation: 0.52
------------------------------

Best performing variant: step_by_step
Overall score: 0.89

Recommendations:
Based on the experiment results, the "step_by_step" prompt variant performs best overall. This variant asks for a structured solution with clear steps. For best results when solving math problems, instruct the model to:

1. Explain the approach to solving the problem first
2. Show the solution work step by step
3. Provide the final numerical answer

This approach leads to both high correctness (92%) and good explanations (83%).

Optimization complete!
```

## What's Next?

With your optimized prompt, you can now:

1. Further refine the winning prompt with more specific instructions
2. Test on a larger or more diverse dataset
3. Run comparative analysis between different sets of prompt variants
4. Use the findings in your production applications

AgentOptim makes it easy to continuously improve your prompts through data-driven experimentation, helping you get the most out of language models for your specific use cases.

Happy optimizing!
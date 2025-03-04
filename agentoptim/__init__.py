"""
# AgentOptim

A powerful toolset for optimizing prompts using data-driven experiments, focused on MCP agents.

## Overview

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent 
to optimize prompts in a data-driven way. Think of it as DSPy, but for agents - a toolkit 
that enables autonomous experimentation, evaluation, and optimization of prompts and interactions.

## Installation

```bash
pip install agentoptim
```

## Key Components

The package is organized into the following modules:

- `evaluation`: Create and manage evaluation criteria for assessing response quality
- `dataset`: Create and manage datasets for testing prompt variations
- `experiment`: Configure and run experiments with different prompt variations
- `jobs`: Run experiments and evaluations with judge models
- `analysis`: Analyze experiment results and optimize prompts
- `server`: MCP server for exposing AgentOptim tools to agents

## MCP Tools

AgentOptim provides 5 powerful MCP tools:

1. `manage_evaluation_tool`: Create, list, get, update, and delete evaluations
2. `manage_dataset_tool`: Create, list, get, update, delete, and split datasets
3. `manage_experiment_tool`: Create, list, get, update, and delete experiments
4. `run_job_tool`: Run evaluations or experiments with specified inputs and models
5. `analyze_results_tool`: Analyze experiment results to find optimal prompts

## Getting Started

Here's a simple example workflow:

```python
# Create a dataset
dataset = create_dataset(
    name="Sample QA Dataset",
    items=[{"question": "What is 2+2?", "context": "Basic arithmetic."}]
)

# Create an evaluation
evaluation = create_evaluation(
    name="Math Accuracy",
    criteria=[{"name": "correctness", "weight": 1.0}]
)

# Create an experiment with different prompt variants
experiment = create_experiment(
    name="Math QA Experiment",
    prompt_template="Answer the following question: {question}",
    variants=[
        {"name": "basic", "prompt": "Answer the following question: {question}"},
        {"name": "detailed", "prompt": "Please solve this math problem step by step: {question}"}
    ]
)

# Run the experiment
job = create_job(
    experiment_id=experiment.experiment_id,
    dataset_id=dataset.dataset_id,
    evaluation_id=evaluation.evaluation_id
)
run_job(job.job_id)

# Analyze results
analysis = create_analysis(experiment_id=experiment.experiment_id)
print(f"Best variant: {analysis.best_variant.name}")
```

See the documentation for each module for more details and examples.
"""

__version__ = "0.1.0"

# Make core components available at package level
from .evaluation import create_evaluation, get_evaluation, list_evaluations, update_evaluation, delete_evaluation
from .dataset import create_dataset, get_dataset, list_datasets, update_dataset, delete_dataset, split_dataset
from .experiment import create_experiment, get_experiment, list_experiments, update_experiment, delete_experiment
from .jobs import create_job, get_job, list_jobs, run_job
from .analysis import create_analysis, get_analysis, list_analyses, delete_analysis, compare_analyses
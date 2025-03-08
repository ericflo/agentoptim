"""
# AgentOptim

A powerful toolset for optimizing and evaluating agent responses using data-driven experiments and evaluations.

## Overview

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent 
to optimize prompts and evaluate conversations in a data-driven way. Think of it as DSPy, 
but for agents - a toolkit that enables autonomous experimentation, evaluation, and 
optimization of prompts and interactions.

## Installation

```bash
pip install agentoptim
```

## Key Components

The package is organized into the following modules:

- `evalset`: Create and manage EvalSets for evaluating conversations (new in v2.0)
- `runner`: Run evaluations against conversations using EvalSets (new in v2.0)
- `compat`: Compatibility layer for transitioning from v1.x to v2.0
- `server`: MCP server for exposing AgentOptim tools to agents
- Legacy modules (deprecated but maintained for compatibility):
  - `evaluation`: Create and manage evaluation criteria
  - `dataset`: Create and manage datasets
  - `experiment`: Configure and run experiments
  - `jobs`: Run experiments and evaluations
  - `analysis`: Analyze experiment results

## MCP Tools

AgentOptim v2.0 provides 2 powerful MCP tools:

1. `manage_evalset_tool`: Create, list, get, update, and delete EvalSets
2. `run_evalset_tool`: Evaluate conversations against an EvalSet

The following legacy tools are maintained for backward compatibility:

1. `manage_evaluation_tool`: Create, list, get, update, and delete evaluations (deprecated)
2. `manage_dataset_tool`: Create, list, get, update, delete, and split datasets (deprecated)
3. `manage_experiment_tool`: Create, list, get, update, and delete experiments (deprecated)
4. `run_job_tool`: Run evaluations or experiments with specified inputs and models (deprecated)
5. `analyze_results_tool`: Analyze experiment results to find optimal prompts (deprecated)

## Getting Started (v2.0 API)

Here's a simple example workflow using the new v2.0 API:

```python
# Create an EvalSet with evaluation criteria
evalset_result = await manage_evalset_tool(
    action="create",
    name="Response Quality Evaluation",
    template="""
    Given this conversation:
    {{ conversation }}
    
    Please answer the following yes/no question about the final assistant response:
    {{ eval_question }}
    
    Return a JSON object with the following format:
    {"judgment": 1} for yes or {"judgment": 0} for no.
    """,
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Is the response accurate?",
        "Does the response provide complete information?"
    ],
    description="Evaluation criteria for response quality"
)

# Extract the EvalSet ID
evalset_id = evalset_result.get("evalset", {}).get("id")

# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
]

# Run the evaluation on the conversation
eval_results = await run_evalset_tool(
    evalset_id=evalset_id,
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct"
)

# The results include both judgments and a summary
print(f"Yes percentage: {eval_results.get('summary', {}).get('yes_percentage')}%")
```

See the documentation and the Migration Guide for more details and examples.
"""

__version__ = "2.0.0"

# Make core components available at package level

# New v2.0 API
from .evalset import manage_evalset
from .runner import run_evalset

# Legacy API (deprecated but maintained for compatibility)
from .evaluation import create_evaluation, get_evaluation, list_evaluations, update_evaluation, delete_evaluation
from .dataset import create_dataset, get_dataset, list_datasets, update_dataset, delete_dataset, split_dataset
from .experiment import create_experiment, get_experiment, list_experiments, update_experiment, delete_experiment
from .jobs import create_job, get_job, list_jobs, run_job
from .analysis import create_analysis, get_analysis, list_analyses, delete_analysis, compare_analyses
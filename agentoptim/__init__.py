"""
# AgentOptim v2.1.0

A streamlined toolset for evaluating conversations with a simplified 2-tool architecture.

## Overview

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent 
to evaluate conversations in a data-driven way. Think of it as DSPy, 
but for agents - a toolkit that enables autonomous evaluation and optimization
of prompts and interactions.

## Installation

```bash
pip install agentoptim
```

## Key Components

The package is organized into the following modules:

- `evalset`: Create and manage EvalSets for evaluating conversations
- `evalrun`: Store and retrieve evaluation results
- `runner`: Run evaluations against conversations using EvalSets
- `utils`: Utility functions for file operations and data handling
- `server`: MCP server for exposing AgentOptim tools to agents
- `cache`: Performance optimization through caching
- `validation`: Input validation functionality
- `errors`: Error handling and logging

## MCP Tools

AgentOptim v2.1.0 provides 2 powerful MCP tools:

1. `manage_evalset_tool`: Create, list, get, update, and delete EvalSets
2. `manage_eval_runs_tool`: Run evaluations, retrieve past results, and list evaluation runs

## Getting Started

Here's a simple example workflow:

```python
# Create an EvalSet with evaluation criteria
evalset_result = await manage_evalset_tool(
    action="create",
    name="Response Quality Evaluation",
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Is the response accurate?",
        "Does the response provide complete information?"
    ],
    short_description="Basic quality evaluation",
    long_description="This EvalSet provides comprehensive evaluation criteria for AI assistant responses. It measures helpfulness, relevance, clarity, accuracy, and completeness. Use it to evaluate assistant responses to general user queries. High scores indicate responses that are helpful, relevant, clear, accurate, and complete." + " " * 50
)

# Extract the EvalSet ID
evalset_id = evalset_result["evalset"]["id"]

# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
]

# Run the evaluation on the conversation and store the results
eval_results = await manage_eval_runs_tool(
    action="run",
    evalset_id=evalset_id,
    conversation=conversation
    # Model will be auto-detected
)

# The results include both judgments and a summary
print(f"Yes percentage: {eval_results['summary']['yes_percentage']}%")
print(f"Evaluation stored with ID: {eval_results['id']}")

# Later, retrieve the evaluation results by ID
retrieved_eval = await manage_eval_runs_tool(
    action="get",
    eval_run_id=eval_results["id"]
)

# Or list all past evaluations (with pagination)
all_evals = await manage_eval_runs_tool(
    action="list",
    page=1,
    page_size=10
)
```
"""

__version__ = "2.1.0"

# Make MCP tools available at package level
from .server import manage_evalset_tool, manage_eval_runs_tool

# Also expose core implementation functions for advanced usage
from .evalset import manage_evalset
from .runner import run_evalset
from .evalrun import manage_eval_runs
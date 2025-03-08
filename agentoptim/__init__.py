"""
# AgentOptim v2.0

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
- `runner`: Run evaluations against conversations using EvalSets
- `utils`: Utility functions for file operations and data handling
- `server`: MCP server for exposing AgentOptim tools to agents
- `cache`: Performance optimization through caching
- `validation`: Input validation functionality
- `errors`: Error handling and logging

## MCP Tools

AgentOptim v2.0 provides 2 powerful MCP tools:

1. `manage_evalset_tool`: Create, list, get, update, and delete EvalSets
2. `run_evalset_tool`: Evaluate conversations against an EvalSet

## Getting Started

Here's a simple example workflow:

```python
# Create an EvalSet with evaluation criteria
evalset_result = await manage_evalset_tool(
    action="create",
    name="Response Quality Evaluation",
    template="Given this conversation:\\n{{ conversation }}\\n\\nPlease answer the following yes/no question about the final assistant response:\\n{{ eval_question }}\\n\\nReturn a JSON object with the following format:\\n{\\"judgment\\": 1} for yes or {\\"judgment\\": 0} for no.",
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
"""

__version__ = "2.0.0"

# Make core components available at package level
from .evalset import manage_evalset
from .runner import run_evalset
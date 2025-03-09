# AgentOptim Quickstart Guide

This guide will help you start using AgentOptim in under 5 minutes.

## Installation

```bash
pip install agentoptim
```

## Starting the MCP Server

Start the AgentOptim server with:

```bash
agentoptim server
```

That's it! Your server is now running and ready to use.

## Using the CLI

AgentOptim includes a powerful command-line interface:

```bash
# List all available evaluation sets
agentoptim evalset list

# Create a new evaluation set
agentoptim evalset create --name "Response Quality" \
  --questions "Is the response helpful?,Is the response clear?,Is the response accurate?" \
  --short-desc "Basic quality evaluation" \
  --long-desc "Simple evaluation set for measuring basic response quality criteria"

# Create a new evaluation set interactively (easier!)
agentoptim evalset create --wizard

# Evaluate a conversation file against an evaluation set
agentoptim run create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json

# View developer tools like cache statistics
agentoptim dev cache
```

See all available commands with `agentoptim --help`.

## 5-Minute Example

Here's a complete example to evaluate a conversation using the Python API:

```python
import asyncio
from agentoptim import manage_evalset_tool, manage_eval_runs_tool

async def main():
    # 1. Create an EvalSet with your quality criteria
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Quick Example",
        questions=[
            "Is the response helpful?",
            "Is the response clear?",
            "Is the response accurate?"
        ]
    )
    
    # Get the EvalSet ID
    evalset_id = evalset_result["evalset"]["id"]
    
    # 2. Define a conversation to evaluate
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]
    
    # 3. Run the evaluation
    results = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation
    )
    
    # 4. Print the results
    print(f"Overall score: {results['summary']['yes_percentage']}%")
    for item in results["results"]:
        print(f"{item['question']}: {'Yes' if item['judgment'] else 'No'}")

if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `quickstart.py` and run it:

```bash
python quickstart.py
```

## Next Steps

- [Tutorial](TUTORIAL.md) - More detailed tutorial
- [Examples](../examples/) - Browse example scripts
- [API Reference](API_REFERENCE.md) - Complete API documentation

## Common Operations

### Creating an EvalSet

```python
evalset = await manage_evalset_tool(
    action="create",
    name="My EvalSet",
    questions=[
        "Question 1?",
        "Question 2?",
        "Question 3?"
    ]
)
```

### Listing All EvalSets

```python
result = await manage_evalset_tool(action="list")
evalsets = result["evalsets"]
```

### Evaluating a Conversation

```python
results = await manage_eval_runs_tool(
    action="run",
    evalset_id="your_evalset_id",
    conversation=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "User question here"},
        {"role": "assistant", "content": "Assistant response here"}
    ]
)
```

### Using a Specific Judge Model

```python
results = await manage_eval_runs_tool(action="run", 
    evalset_id="your_evalset_id",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct"  # or "gpt-4", "claude-3-haiku-20240307", etc.
)
```

That's it! You're now ready to use AgentOptim for evaluating and optimizing your LLM conversations.
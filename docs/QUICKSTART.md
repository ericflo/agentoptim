# AgentOptim Quickstart Guide

This guide will help you start using AgentOptim in under 5 minutes.

## Installation

```bash
pip install agentoptim
```

## Starting the MCP Server

Start the AgentOptim server with:

```bash
agentoptim
```

That's it! Your server is now running and ready to use.

## 5-Minute Example

Here's a complete example to evaluate a conversation:

```python
import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

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
    results = await run_evalset_tool(
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
results = await run_evalset_tool(
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
results = await run_evalset_tool(
    evalset_id="your_evalset_id",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct"  # or "gpt-4", "claude-3-haiku-20240307", etc.
)
```

That's it! You're now ready to use AgentOptim for evaluating and optimizing your LLM conversations.
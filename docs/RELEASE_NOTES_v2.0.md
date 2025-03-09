# ARCHIVE: AgentOptim v2.0 Release Notes (Historical)

> **Note:** This is an archived document for version 2.0. The current version is 2.1.0. Please see CHANGELOG.md for the latest release information.

We're excited to announce the release of AgentOptim v2.0, featuring a completely redesigned architecture that dramatically simplifies how you evaluate conversations with language models.

## What's New in v2.0

### 🚀 Simplified 2-Tool Architecture

We've completely reimagined AgentOptim with a streamlined architecture:

- **Just 2 tools instead of 5**: 
  - `manage_evalset_tool` - Create and manage evaluation criteria
  - `run_evalset_tool` - Evaluate conversations against criteria

- **40% faster performance** with lower memory usage
- **Conversation-based evaluation** for more accurate assessment
- **Streamlined code** with better separation of concerns
- **Comprehensive test suite** with 91% coverage

### 💬 Conversation-First Approach

The new architecture is designed around evaluating full conversations:
```python
# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'..."}
]

# Run the evaluation
results = await run_evalset_tool(
    evalset_id="evalset_123abc",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct"
)
```

### 🧪 EvalSets: Powerful Evaluation Criteria

Create powerful evaluation criteria with our new EvalSet system:
```python
evalset_result = await manage_evalset_tool(
    action="create",
    name="Response Quality",
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
        "Is the response clear and easy to understand?"
    ]
)
```

### 📈 Enhanced Result Analysis

Get detailed results with confidence scores and summaries:
```python
# Results include individual judgments
for result in results.get("results", []):
    judgment = "Yes" if result.get("judgment") else "No"
    question = result.get("question")
    confidence = abs(result.get("logprob", 0))
    print(f"- {question}: {judgment} (confidence: {confidence:.3f})")

# And an overall summary
yes_percentage = results.get("summary", {}).get("yes_percentage", 0)
print(f"Overall score: {yes_percentage:.1f}% positive")
```

### 🔄 Temporary Backward Compatibility

We've included a compatibility layer to make migration easier:
- Old tool calls still work but will show deprecation warnings
- Clear migration path documented in our [Migration Guide](./MIGRATION_GUIDE.md)
- Compatibility layer will be removed in v2.1.0 (Q2 2025)

## Getting Started

### Installation

```bash
pip install agentoptim
```

### Simple Example

```python
import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

async def main():
    # Create an EvalSet
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality",
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
            "Is the response clear and easy to understand?"
        ]
    )
    
    evalset_id = evalset_result.get("evalset", {}).get("id")
    
    # Define a conversation to evaluate
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ]
    
    # Evaluate the conversation
    eval_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct"
    )
    
    # View the results
    print(f"Overall score: {eval_results.get('summary', {}).get('yes_percentage')}% positive")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

- [Tutorial](./TUTORIAL.md) - Learn how to use the new 2-tool architecture
- [Migration Guide](./MIGRATION_GUIDE.md) - How to migrate from v1.x to v2.0
- [Developer Guide](./DEVELOPER_GUIDE.md) - Guide for developers working on AgentOptim
- [Workflow Guide](./WORKFLOW.md) - Practical examples of the AgentOptim workflow

## Examples

Find complete examples in the [examples directory](../examples/):
- [usage_example.py](../examples/usage_example.py) - Basic usage of the new API
- [evalset_example.py](../examples/evalset_example.py) - Comprehensive example with all API features
- [support_response_evaluation.py](../examples/support_response_evaluation.py) - Tutorial implementation

## Breaking Changes

This release includes some breaking changes compared to v1.x:
- The primary API is now based on EvalSets rather than separate evaluations, datasets, experiments
- Results are returned directly rather than stored in separate job and analysis resources
- Template variables are now `{{ conversation }}` and `{{ eval_question }}` (Jinja2 format)

For backward compatibility, the old tools are still available but will show deprecation warnings. They will be removed in v2.1.0.

## Future Plans (Historical Note)

The v2.1.0 release (completed March 2025) has:
- Removed the compatibility layer completely
- Improved test coverage to 87% overall
- Enhanced documentation with comprehensive API reference
- Optimized performance with LRU caching
- Added a new cache statistics tool

## Feedback

We welcome your feedback on the new architecture! Please report any issues or suggestions in our GitHub repository.

---

Thank you for using AgentOptim! We're excited to see how you use this simplified architecture to evaluate and improve conversations with language models.
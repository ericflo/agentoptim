# AgentOptim v2.1.0 API Reference

This document provides comprehensive documentation for all AgentOptim functions, parameters, and return values. It serves as the definitive reference for developers integrating AgentOptim into their projects.

## Core Tools

AgentOptim's architecture centers on two powerful tools, designed to work together seamlessly:

1. `manage_evalset_tool`: Create and manage evaluation criteria sets
2. `run_evalset_tool`: Apply evaluation criteria to conversations

## `manage_evalset_tool`

This tool handles all operations for creating and managing EvalSets - collections of evaluation criteria for assessing conversation quality.

### Function Signature

```python
async def manage_evalset_tool(
    action: str,
    evalset_id: Optional[str] = None,
    name: Optional[str] = None,
    questions: Optional[List[str]] = None,
    short_description: Optional[str] = None,
    long_description: Optional[str] = None,
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create, retrieve, update, list, or delete evaluation sets (EvalSets).
    """
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `action` | `str` | Yes | Action to perform: "create", "get", "update", "list", or "delete" |
| `evalset_id` | `str` | For get/update/delete | ID of the EvalSet to operate on |
| `name` | `str` | For create | Name of the EvalSet |
| `questions` | `List[str]` | For create | List of evaluation questions |
| `short_description` | `str` | No | Brief description of the EvalSet |
| `long_description` | `str` | No | Detailed description of the EvalSet |
| `template` | `str` | No | Custom Jinja2 template for evaluations (uses system default if not provided) |

### Return Value

Returns a dictionary with action-specific data:

- For "create" and "update": `{"evalset": {...}}` - The created/updated EvalSet object
- For "get": `{"evalset": {...}}` - The requested EvalSet object
- For "list": `{"evalsets": [...]}` - List of all available EvalSets
- For "delete": `{"success": true}` - Confirmation of successful deletion

### EvalSet Object Structure

```json
{
  "id": "unique_evalset_id",
  "name": "EvalSet Name",
  "questions": ["Question 1?", "Question 2?", ...],
  "short_description": "Brief description",
  "long_description": "More detailed description",
  "template": "Custom template text or null if using default",
  "created_at": "2023-04-01T12:00:00Z",
  "updated_at": "2023-04-01T12:00:00Z"
}
```

### Example Usage

#### Creating an EvalSet

```python
result = await manage_evalset_tool(
    action="create",
    name="Customer Support Quality",
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Is the tone of the response appropriate and professional?"
    ],
    short_description="Evaluates customer support response quality",
    long_description="This EvalSet measures the helpfulness, clarity, and professionalism of customer support responses."
)

evalset_id = result["evalset"]["id"]
```

#### Retrieving an EvalSet

```python
result = await manage_evalset_tool(
    action="get",
    evalset_id="abc123"
)

evalset = result["evalset"]
```

#### Updating an EvalSet

```python
result = await manage_evalset_tool(
    action="update",
    evalset_id="abc123",
    name="Updated Support Quality Evaluation",
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Is the tone of the response appropriate and professional?",
        "Does the response provide complete information?"  # Added question
    ]
)

updated_evalset = result["evalset"]
```

#### Listing All EvalSets

```python
result = await manage_evalset_tool(
    action="list"
)

evalsets = result["evalsets"]
for evalset in evalsets:
    print(f"{evalset['id']}: {evalset['name']}")
```

#### Deleting an EvalSet

```python
result = await manage_evalset_tool(
    action="delete",
    evalset_id="abc123"
)

if result["success"]:
    print("EvalSet deleted successfully")
```

## `run_evalset_tool`

This tool applies an EvalSet to a conversation to evaluate its quality against defined criteria.

### Function Signature

```python
async def run_evalset_tool(
    evalset_id: str,
    conversation: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_parallel: Optional[int] = None,
    omit_reasoning: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Evaluate a conversation using an EvalSet.
    """
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evalset_id` | `str` | Yes | ID of the EvalSet to use for evaluation |
| `conversation` | `List[Dict[str, str]]` | Yes | List of message objects in the conversation |
| `model` | `str` | No | Judge model to use (default: meta-llama-3.1-8b-instruct) |
| `temperature` | `float` | No | Temperature for the judge model (default: 0.0) |
| `max_parallel` | `int` | No | Maximum number of parallel evaluations (default: 5) |
| `omit_reasoning` | `bool` | No | Whether to omit detailed reasoning in results (default: false) |

### Conversation Format

The `conversation` parameter must be a list of message objects, where each message is a dictionary with `role` and `content` keys:

```python
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
]
```

### Return Value

Returns a dictionary with evaluation results:

```json
{
  "results": [
    {
      "question": "Is the response helpful for the user's needs?",
      "judgment": 1,
      "logprob": -0.021,
      "reasoning": "The response provides clear instructions on how to reset a password...",
      "raw_result": {...}
    },
    // Additional results for each question...
  ],
  "summary": {
    "total_questions": 4,
    "yes_count": 3,
    "no_count": 1,
    "yes_percentage": 75.0
  }
}
```

#### Result Fields

| Field | Description |
|-------|-------------|
| `question` | The evaluation question |
| `judgment` | Binary judgment: 1 for yes/pass, 0 for no/fail |
| `logprob` | Log probability of the judgment (indicates confidence) |
| `reasoning` | Explanation of the judgment (if `omit_reasoning` is false) |
| `raw_result` | Raw response from the judge model (format depends on template) |

### Example Usage

#### Basic Evaluation

```python
# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
]

# Run the evaluation
results = await run_evalset_tool(
    evalset_id="abc123",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# Print summary
print(f"Overall score: {results['summary']['yes_percentage']}%")
print(f"Passed criteria: {results['summary']['yes_count']}/{results['summary']['total_questions']}")

# Print individual judgments
for item in results["results"]:
    judgment = "Yes" if item["judgment"] else "No"
    print(f"{item['question']}: {judgment}")
```

#### Advanced Usage with Custom Model

```python
results = await run_evalset_tool(
    evalset_id="abc123",
    conversation=conversation,
    model="gpt-4",
    temperature=0.2,
    max_parallel=2,
    omit_reasoning=True
)
```

## Templates

AgentOptim uses Jinja2 templates to format evaluations. You can customize templates when creating EvalSets to fit specific evaluation needs.

### Default Template

If no custom template is provided, AgentOptim uses this default template:

```
Given this conversation:
{{ conversation }}

Please answer the following yes/no question about the final assistant response:
{{ eval_question }}

Return a JSON object with the following format:
{"judgment": 1} for yes or {"judgment": 0} for no.
```

### Template Variables

| Variable | Description |
|----------|-------------|
| `{{ conversation }}` | The full conversation transcript |
| `{{ eval_question }}` | The current evaluation question |

### Custom Template Example

This example uses a 5-point scale instead of binary judgment:

```
Given this conversation:
{{ conversation }}

Please evaluate the following aspect of the final assistant response:
{{ eval_question }}

Rate your agreement on a 5-point scale:
1 - Strongly Disagree
2 - Disagree
3 - Neutral
4 - Agree
5 - Strongly Agree

Provide your rating and explanation in JSON format:
{
    "rating": [1-5 integer],
    "explanation": "Brief justification for your rating",
    "judgment": [0 or 1 - convert to binary where 4-5 = 1, 1-3 = 0]
}
```

## Model Selection

AgentOptim supports multiple judge models for evaluations. The model is selected according to this priority:

1. If a `model` parameter is explicitly provided in the `run_evalset_tool` call
2. If the `AGENTOPTIM_JUDGE_MODEL` environment variable is set
3. If the `judge_model` option is provided in client configuration
4. Default model: `meta-llama-3.1-8b-instruct`

### Supported Models

- **LM Studio Models**: Any model available in LM Studio
- **OpenAI Models**: With valid API key in environment
  - gpt-4
  - gpt-4o
  - gpt-4-turbo
  - gpt-4o-mini
  - gpt-3.5-turbo
- **Anthropic Models**: With valid API key in environment
  - claude-3-opus-20240229
  - claude-3-sonnet-20240229
  - claude-3-haiku-20240307

## Error Handling

AgentOptim tools throw exceptions with detailed error messages when operations fail. Common error types include:

### EvalSetNotFoundError

Thrown when an EvalSet with the specified ID doesn't exist.

```python
try:
    result = await manage_evalset_tool(
        action="get",
        evalset_id="nonexistent_id"
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle EvalSet not found
```

### ValidationError

Thrown when input parameters are invalid or missing.

```python
try:
    result = await manage_evalset_tool(
        action="create",
        # Missing required name and questions
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle validation error
```

### ModelNotAvailableError

Thrown when the specified judge model is not available.

```python
try:
    result = await run_evalset_tool(
        evalset_id="abc123",
        conversation=conversation,
        model="nonexistent_model"
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle model not available
```

## Performance Considerations

For optimal performance when using AgentOptim:

1. **Parallel Evaluations**: The `max_parallel` parameter in `run_evalset_tool` controls how many questions are evaluated simultaneously. Higher values increase speed but use more resources.

2. **Model Selection**: Smaller models like `meta-llama-3.1-8b-instruct` are faster than larger ones like `gpt-4`, but may provide less nuanced evaluations.

3. **Template Complexity**: More complex templates may take longer to process. Simple templates are more efficient for large batch evaluations.

4. **Caching**: AgentOptim automatically caches evaluation results. Identical conversations with the same EvalSet and model will use cached results, significantly improving performance for repeated evaluations.

## Advanced Usage

### Comparing Different Approaches

To compare multiple conversation approaches:

```python
# Define different conversations
approach1 = [...]  # First conversation approach
approach2 = [...]  # Second conversation approach

# Evaluate each approach
results1 = await run_evalset_tool(evalset_id="abc123", conversation=approach1)
results2 = await run_evalset_tool(evalset_id="abc123", conversation=approach2)

# Compare scores
print(f"Approach 1 score: {results1['summary']['yes_percentage']}%")
print(f"Approach 2 score: {results2['summary']['yes_percentage']}%")
```

### Batch Processing

For evaluating multiple conversations efficiently:

```python
async def evaluate_batch(evalset_id, conversations):
    results = []
    for i, conv in enumerate(conversations):
        print(f"Evaluating conversation {i+1}/{len(conversations)}")
        result = await run_evalset_tool(evalset_id=evalset_id, conversation=conv)
        results.append(result)
    return results

batch_results = await evaluate_batch("abc123", conversations_list)
```

## Best Practices

1. **Choose the Right Judge Model**: Select a judge model appropriate for your evaluation needs:
   - Small models like `meta-llama-3.1-8b-instruct` for speed and efficiency
   - Larger models like `gpt-4` or `claude-3-opus` for more nuanced evaluations

2. **Design Effective Questions**: Create clear, unambiguous evaluation questions that focus on specific aspects of quality.

3. **Use Specialized EvalSets**: Create different EvalSets for different types of conversations or quality criteria.

4. **Custom Templates**: Use custom templates for specialized evaluations requiring more detailed judgments.

5. **Templatized Evaluations**: For comparing similar conversations, use the same EvalSet to ensure consistent evaluation.

## API Changes in v2.1.0

Version 2.1.0 removes the legacy compatibility layer, making these breaking changes:

1. All deprecated functions from v1.x have been removed
2. The old `template_id` parameter is no longer supported in any function
3. The `compat.py` module has been completely removed

Refer to the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed information on migrating from older versions.

---

For more information and practical examples, see the [examples directory](../examples/) in the repository.
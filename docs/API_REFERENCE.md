# AgentOptim API Reference

This document provides comprehensive reference documentation for the tools and commands available in the AgentOptim project.

## Overview

AgentOptim is a conversation evaluation framework that provides tools for creating, managing, and running evaluations on conversational AI responses. The framework offers two interfaces:

1. **Python API**: For programmatic access and integration
2. **Command-Line Interface (CLI)**: For direct usage without writing code

Both interfaces provide the same core functionality around two main concepts:

1. **EvalSets**: Collections of evaluation criteria for judging conversation quality
2. **Evaluation Runners**: Tools for executing evaluations against conversations

## Python API

The Python API consists of two primary tools:

- `manage_evalset_tool`: For creating and managing evaluation criteria sets (EvalSets)
- `manage_eval_runs_tool`: For running evaluations, storing results, and retrieving past evaluation runs

The system also provides cache performance statistics functionality through internal utilities.

## Command-Line Interface (CLI)

The CLI provides the same functionality through intuitive commands:

### EvalSet Management
- `agentoptim list`: List all available EvalSets
- `agentoptim create`: Create a new EvalSet
- `agentoptim get`: Get details about a specific EvalSet
- `agentoptim update`: Update an existing EvalSet
- `agentoptim delete`: Delete an EvalSet

### Evaluation Management
- `agentoptim runs run`: Run a new evaluation against an EvalSet (alias: `eval`)
- `agentoptim runs list`: List all past evaluation runs
- `agentoptim runs get`: Get details about a specific evaluation run

### Utilities
- `agentoptim stats`: Get cache performance statistics

For detailed CLI usage examples, see [CLI Usage Examples](../examples/cli_usage_examples.md).

## Tool: manage_evalset_tool

### Description

This tool allows you to create, retrieve, update, and delete "EvalSets" - collections of evaluation criteria for judging the quality of conversational responses.

### Parameters

| Parameter | Type | Required For | Description |
|-----------|------|--------------|-------------|
| `action` | string | All actions | The operation to perform. Must be one of: "create", "list", "get", "update", "delete" |
| `evalset_id` | string | get, update, delete | The unique identifier of the EvalSet (UUID format) |
| `name` | string | create, update (optional) | A descriptive name for the EvalSet |
| `questions` | array of strings | create, update (optional) | A list of yes/no evaluation questions to assess responses |
| `short_description` | string | create, update (optional) | A concise summary (6-128 chars) of what this EvalSet measures |
| `long_description` | string | create, update (optional) | A detailed explanation (256-1024 chars) of the evaluation criteria |

### Return Values

The tool returns different results based on the action performed:

#### For `list` action
```json
{
  "status": "success",
  "evalsets": [
    {
      "id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "name": "Technical Support Quality Evaluation",
      "short_description": "Tech support response evaluation criteria",
      "template": "...",
      "questions": ["..."],
      "question_count": 7
    },
    ...
  ]
}
```

#### For `get` action
```json
{
  "status": "success",
  "evalset": {
    "id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    "name": "Technical Support Quality Evaluation",
    "short_description": "Tech support response evaluation criteria",
    "long_description": "This EvalSet provides comprehensive evaluation criteria for technical support responses...",
    "template": "...",
    "questions": ["..."],
    "question_count": 7
  }
}
```

#### For `create` action
```json
{
  "status": "success",
  "evalset": {
    "id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    "name": "Technical Support Quality Evaluation",
    "short_description": "Tech support response evaluation criteria",
    "long_description": "This EvalSet provides comprehensive evaluation criteria for technical support responses...",
    "template": "...",
    "questions": ["..."]
  },
  "message": "EvalSet 'Technical Support Quality Evaluation' created with ID: 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
}
```

#### For `update` action
```json
{
  "status": "success",
  "evalset": {
    "id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    "name": "Enhanced Technical Support Quality Evaluation",
    "short_description": "Enhanced tech support evaluation criteria",
    "long_description": "This enhanced EvalSet provides improved evaluation criteria...",
    "template": "...",
    "questions": ["..."]
  },
  "message": "EvalSet 'Enhanced Technical Support Quality Evaluation' updated"
}
```

#### For `delete` action
```json
{
  "status": "success",
  "message": "EvalSet 'Technical Support Quality Evaluation' with ID '6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e' deleted"
}
```

### Error Responses

If an error occurs, the tool returns a detailed error message:

```json
{
  "error": "Error message",
  "details": "Detailed explanation of the error",
  "troubleshooting": [
    "Suggestion 1",
    "Suggestion 2",
    ...
  ]
}
```

### Usage Examples

#### List all EvalSets

```python
existing_evalsets = manage_evalset_tool(action="list")
```

#### Get an EvalSet by ID

```python
evalset_details = manage_evalset_tool(
    action="get",
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
)
```

#### Create a new EvalSet

```python
evalset = manage_evalset_tool(
    action="create",
    name="Technical Support Quality Evaluation",
    questions=[
        "Does the response directly address the user's specific question?",
        "Is the response clear and easy to understand?",
        "Does the response provide complete step-by-step instructions?",
        "Is the response accurate and technically correct?",
        "Does the response use appropriate technical terminology?",
        "Is the tone of the response professional and helpful?",
        "Would the response likely resolve the user's issue without further assistance?"
    ],
    short_description="Tech support response evaluation criteria",
    long_description="This EvalSet provides comprehensive evaluation criteria for technical support responses. It measures clarity, completeness, accuracy, helpfulness, and professionalism. Use it to evaluate support agent responses to technical questions or troubleshooting scenarios. High scores indicate responses that are clear, accurate, and likely to resolve the user's issue without further assistance."
)
```

#### Update an EvalSet

```python
updated_evalset = manage_evalset_tool(
    action="update",
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    name="Enhanced Technical Support Quality Evaluation",
    questions=["New question 1", "New question 2"],
    short_description="Enhanced tech support evaluation criteria",
    long_description="This enhanced EvalSet provides improved evaluation criteria for technical support responses with more specific questions focused on resolution success. It measures clarity, accuracy, and customer satisfaction more precisely. Use it for evaluating advanced support scenarios where multiple solutions might be applicable."
)
```

#### Delete an EvalSet

```python
delete_result = manage_evalset_tool(
    action="delete",
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
)
```

### Important Notes

1. **System Templates**: As of v2.0, templates are system-defined and no longer customizable by users.
2. **Questions Format**: The `questions` parameter must be a proper list/array of strings, not a multiline string.
3. **Question Limits**: A maximum of 100 questions are allowed per EvalSet.
4. **Description Requirements**:
   - `short_description` must be 6-128 characters
   - `long_description` must be 256-1024 characters

## Tool: manage_eval_runs_tool

### Description

This tool manages evaluation runs - it allows you to run evaluations on conversations using predefined criteria (EvalSets), store the results persistently, retrieve past evaluation results, and list evaluation runs with pagination. All evaluation results are stored and can be retrieved by ID for later analysis.

### Parameters

| Parameter | Type | Required For | Description |
|-----------|------|--------------|-------------|
| `action` | string | All actions | The operation to perform. Must be one of: "run", "get", "list" |
| `evalset_id` | string | run, optional for list | The unique identifier of the EvalSet to use |
| `conversation` | array of objects | run | A chronological list of conversation messages to evaluate |
| `judge_model` | string | No | The LLM to use as the evaluation judge |
| `max_parallel` | integer | No (default: 3) | Maximum number of evaluation questions to process simultaneously |
| `omit_reasoning` | boolean | No (default: false) | If true, don't generate detailed reasoning in results |
| `eval_run_id` | string | get | ID of a specific evaluation run to retrieve |
| `page` | integer | No (default: 1) | Page number for paginated list of runs (1-indexed) |
| `page_size` | integer | No (default: 10) | Number of items per page (range: 1-100) |

#### Conversation Format

Each message in the `conversation` array must be a dictionary with these fields:

```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

### Tool Configuration Options

The following options can be specified in client configuration:

| Option | Type | Description |
|--------|------|-------------|
| `judge_model` | string | The LLM to use as the evaluation judge (default: "meta-llama-3.1-8b-instruct") |
| `omit_reasoning` | boolean | If true, don't generate detailed reasoning in results |

Example client configuration:
```json
{
  "mcpServers": {
    "optim": {
      "command": "...",
      "args": ["..."],
      "options": {
        "judge_model": "gpt-4o-mini",
        "omit_reasoning": "True"
      }
    }
  }
}
```

These options can also be set using environment variables:
- `AGENTOPTIM_JUDGE_MODEL`: Set the judge model
- `AGENTOPTIM_OMIT_REASONING`: Set to "1", "true", or "yes" to omit reasoning

### Return Values

The tool returns different results based on the action performed:

#### For `run` action

```json
{
  "status": "success",
  "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
  "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
  "evalset_name": "Technical Support Quality Evaluation",
  "judge_model": "meta-llama-3.1-8b-instruct",
  "results": [
    {
      "question": "Does the response directly address the user's specific question?",
      "judgment": true,
      "confidence": 0.95,
      "reasoning": "The assistant's response directly addresses..."
    },
    ...
  ],
  "summary": {
    "total_questions": 7,
    "successful_evaluations": 7,
    "yes_count": 6,
    "no_count": 1,
    "yes_percentage": 85.71,
    ...
  }
}
```

#### For `get` action

```json
{
  "status": "success",
  "eval_run": {
    "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    "evalset_name": "Technical Support Quality Evaluation",
    "timestamp": 1717286400.0,
    "timestamp_formatted": "2024-06-01 12:00:00",
    "judge_model": "meta-llama-3.1-8b-instruct",
    "results": [...],
    "summary": {...}
  },
  "formatted_message": "# Evaluation Results: Technical Support Quality Evaluation..."
}
```

#### For `list` action (with pagination)

```json
{
  "status": "success",
  "eval_runs": [
    {
      "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_name": "Technical Support Quality Evaluation",
      "timestamp": 1717286400.0,
      "timestamp_formatted": "2024-06-01 12:00:00",
      "judge_model": "meta-llama-3.1-8b-instruct",
      "summary": {...}
    },
    ...
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_count": 25,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false,
    "next_page": 2,
    "prev_page": null
  },
  "formatted_message": "# Evaluation Runs\nPage 1 of 3 (25 total)..."
}
```

### Error Responses

If an error occurs, the tool returns a detailed error message:

```json
{
  "error": "Error message",
  "details": "Detailed explanation of the error",
  "troubleshooting": [
    "Suggestion 1",
    "Suggestion 2",
    ...
  ]
}
```

### Usage Examples

#### Running a New Evaluation

```python
# Evaluate a conversation and get results with a stored ID
evaluation = await manage_eval_runs_tool(
    action="run",
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    conversation=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Go to the login page and click 'Forgot Password'."}
    ],
    judge_model="gpt-4o-mini",
    max_parallel=3
)

# Extract the evaluation run ID for later retrieval
eval_run_id = evaluation["id"]
print(f"Evaluation score: {evaluation['summary']['yes_percentage']}%")
print(f"Saved as run ID: {eval_run_id}")
```

#### Retrieving a Past Evaluation

```python
# Get a previous evaluation result by ID
past_eval = await manage_eval_runs_tool(
    action="get",
    eval_run_id="9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
)

# Access the evaluation data
print(f"Evaluation from {past_eval['eval_run']['timestamp_formatted']}")
print(f"Score: {past_eval['eval_run']['summary']['yes_percentage']}%")
```

#### Listing Evaluation Runs

```python
# List all evaluation runs (paginated)
all_runs = await manage_eval_runs_tool(
    action="list",
    page=1,
    page_size=10
)

# Print summary of runs
print(f"Found {all_runs['pagination']['total_count']} evaluation runs")
for run in all_runs['eval_runs']:
    print(f"Run {run['id']}: {run['summary']['yes_percentage']}% on {run['timestamp_formatted']}")

# Filter runs by EvalSet ID
filtered_runs = await manage_eval_runs_tool(
    action="list",
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    page=1,
    page_size=10
)
```

### Important Notes

1. **Persistent Storage**: All evaluation results are stored persistently and can be retrieved later.
2. **Pagination**: The `list` action supports pagination to efficiently handle large numbers of evaluation runs.
3. **Filtering**: When listing evaluation runs, you can filter by `evalset_id` to find specific evaluations.
4. **Judge Model**: By default, the tool uses "meta-llama-3.1-8b-instruct" as the judge model, but this can be configured.
5. **Backward Compatibility**: This tool replaces and extends the functionality of the previous `run_evalset_tool`.

## Environment Variables

The following environment variables can be used to configure AgentOptim:

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTOPTIM_DEBUG` | Enable debug logging | "0" |
| `AGENTOPTIM_JUDGE_MODEL` | Default judge model to use | None (uses "meta-llama-3.1-8b-instruct") |
| `AGENTOPTIM_OMIT_REASONING` | Omit reasoning in evaluation results | "0" |
| `AGENTOPTIM_API_BASE` | Base URL for the LLM API | "http://localhost:1234/v1" |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI models) | None |

## Technical Implementation Notes

1. **System Templates**: As of v2.0, templates are system-defined and standardized for all EvalSets.
2. **JSON Response Format**: The evaluation uses a structured JSON format for all responses.
3. **Confidence Scoring**: Confidence scores range from 0.0 to 1.0, with higher values indicating greater confidence.
4. **Parallel Processing**: The tool uses asyncio for parallel processing of evaluation questions.
5. **Error Handling**: Comprehensive error handling with detailed diagnostic information is provided.
6. **API Compatibility**: Works with any OpenAI-compatible API endpoint, including local LLM servers and cloud providers.

## API Changes in v2.1.0

Version 2.1.0 removes the legacy compatibility layer, making these breaking changes:

1. All deprecated functions from v1.x have been removed
2. The `compat.py` module has been completely removed
3. Removed all deprecated examples from examples/deprecated_examples

Refer to the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed information on migrating from older versions.

---

## Cache Statistics

### Description

The system provides detailed statistics about the caching system for monitoring and diagnostics through the CLI. While the `get_cache_stats_tool` MCP tool has been removed in v2.1.0, the functionality remains accessible via the CLI `stats` command, which helps you understand the performance benefits of caching and identify opportunities for optimization.

### CLI Command

```bash
agentoptim stats
```

### Statistics Output

The command outputs a comprehensive set of cache statistics:

```
# Cache Performance Statistics

## EvalSet Cache
- Size: 25 / 50 (current/max)
- Hit Rate: 80.83%
- Hits: 156
- Misses: 37
- Evictions: 0
- Expirations: 3

## API Response Cache
- Size: 78 / 100 (current/max)
- Hit Rate: 64.40%
- Hits: 423
- Misses: 234
- Evictions: 12
- Expirations: 5

## Eval Runs Cache
- Size: 15 / 50 (current/max)
- Hit Rate: 83.33%
- Hits: 150
- Misses: 30
- Evictions: 3
- Expirations: 1

## Overall Performance
- Combined Hit Rate: 68.97%
- Total Hits: 579
- Total Misses: 271
- Resource Savings: Approximately 289.5 seconds of API processing time saved
```

### Statistics Fields

#### EvalSet Cache
- `size`: Current number of items in the cache
- `capacity`: Maximum number of items the cache can hold
- `hits`: Number of successful cache retrievals
- `misses`: Number of cache lookups that didn't find the requested item
- `hit_rate_pct`: Percentage of cache lookups that were hits
- `evictions`: Number of items removed due to capacity constraints
- `expirations`: Number of items removed due to TTL expiration

#### API Cache
Same fields as the EvalSet cache, but for the API response cache.

#### Eval Runs Cache
Same fields as the other caches, but for the evaluation runs cache.

#### Overall
- `hit_rate_pct`: Combined hit rate across all caches
- `total_hits`: Total number of cache hits across all caches
- `total_misses`: Total number of cache misses across all caches
- `estimated_time_saved_seconds`: Estimated processing time saved due to caching

### Important Notes

1. **Access Method**: In v2.1.0, cache statistics are accessed via the CLI rather than an MCP tool.
2. **Time Saved Estimation**: The estimated time saved is an approximation based on average API call duration.
3. **Cache Size Optimization**: A high number of evictions may indicate the cache capacity is too small.
4. **Cache Warmup**: Hit rates will be lower initially and improve as the cache warms up with frequently accessed items.
5. **Cache Monitoring**: Regular monitoring helps optimize cache configuration for your specific usage patterns.

---

For more information and practical examples, see the [examples directory](../examples/) in the repository, particularly the `caching_performance_example.py` file.
# AgentOptim API Reference

This document provides comprehensive reference documentation for the tools available in the AgentOptim project.

## Overview

AgentOptim is a conversation evaluation framework that provides tools for creating, managing, and running evaluations on conversational AI responses. The framework is built around two main concepts:

1. **EvalSets**: Collections of evaluation criteria for judging conversation quality
2. **Evaluation Runners**: Tools for executing evaluations against conversations

The API consists of two primary tools:

- `manage_evalset_tool`: For creating and managing evaluation criteria sets (EvalSets)
- `run_evalset_tool`: For evaluating conversations against EvalSets

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

## Tool: run_evalset_tool

### Description

This tool systematically evaluates a conversation against a predefined set of criteria (an EvalSet), using a language model as a judge. It provides detailed insights into conversation quality with reasoned judgments, confidence scores, and summary statistics.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `evalset_id` | string | Yes | The unique identifier (UUID) of the EvalSet to use |
| `conversation` | array of objects | Yes | A chronological list of conversation messages to evaluate |
| `max_parallel` | integer | No (default: 3) | Maximum number of evaluation questions to process simultaneously |

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

### Return Value

The tool returns a comprehensive evaluation result with these components:

```json
{
  "status": "success",
  "id": "unique-evaluation-id",
  "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
  "evalset_name": "Technical Support Quality Evaluation",
  "judge_model": "meta-llama-3.1-8b-instruct",
  "results": [
    {
      "question": "Does the response directly address the user's specific question?",
      "judgment": true,
      "confidence": 0.95,
      "reasoning": "The assistant's response directly addresses the user's question about resetting their password by providing step-by-step instructions specifically for that task. The response begins with 'To reset your password, please follow these steps:' which clearly indicates that the assistant understood the question and is providing a solution to the exact problem the user described."
    },
    ...
  ],
  "summary": {
    "total_questions": 7,
    "successful_evaluations": 7,
    "yes_count": 6,
    "no_count": 1,
    "error_count": 0,
    "yes_percentage": 85.71,
    "mean_confidence": 0.89,
    "mean_yes_confidence": 0.92,
    "mean_no_confidence": 0.75
  },
  "formatted_message": "# Evaluation Results for 'Technical Support Quality Evaluation'..."
}
```

If `omit_reasoning` is set to true, the `reasoning` field will be omitted from each result.

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

#### Basic evaluation

```python
results = run_evalset_tool(
    evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
    conversation=[
        {"role": "system", "content": "You are a helpful technical support assistant."},
        {"role": "user", "content": "I forgot my password and can't log in. How do I reset it?"},
        {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to the login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder."}
    ],
    max_parallel=2
)
```

#### Comparing multiple responses

```python
def evaluate_response(response_text):
    return run_evalset_tool(
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        conversation=[
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": response_text}
        ]
    )["summary"]["yes_percentage"]

response1_score = evaluate_response("Go to Settings > Account > Reset Password.")
response2_score = evaluate_response("To reset your password, go to the login page and click 'Forgot Password'.")
response3_score = evaluate_response("To reset your password, follow these steps:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Follow the on-screen instructions")

print(f"Response 1: {response1_score}%")
print(f"Response 2: {response2_score}%")
print(f"Response 3: {response3_score}%")
```

### Important Notes

1. **Judge Model**: By default, the tool uses "meta-llama-3.1-8b-instruct" as the judge model, but this can be configured.
2. **Parallel Processing**: The `max_parallel` parameter controls how many questions are evaluated simultaneously. Higher values can improve speed but increase resource requirements.
3. **Conversation Format**: The conversation must include at least one user message and one assistant message.
4. **LM Studio Compatibility**: The tool includes special handling for LM Studio to ensure proper JSON processing.

## Environment Variables

The following environment variables can be used to configure AgentOptim:

| Variable | Description | Default |
|----------|-------------|---------|
| `AGENTOPTIM_DEBUG` | Enable debug logging | "0" |
| `AGENTOPTIM_LMSTUDIO_COMPAT` | Enable LM Studio compatibility mode | "1" |
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
6. **API Compatibility**: Special handling for different LLM providers, including LM Studio compatibility mode.

## API Changes in v2.1.0

Version 2.1.0 removes the legacy compatibility layer, making these breaking changes:

1. All deprecated functions from v1.x have been removed
2. The `compat.py` module has been completely removed
3. Removed all deprecated examples from examples/deprecated_examples

Refer to the [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed information on migrating from older versions.

---

For more information and practical examples, see the [examples directory](../examples/) in the repository.
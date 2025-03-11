# AgentOptim Migration Guide

This guide will help you migrate from AgentOptim v1.x (with 5 separate tools) to the new simplified architecture in v2.x (with just 2 tools).

## v2.1.0 Update: Compatibility Layer Removed

> **IMPORTANT**: In version 2.1.0, we've completely removed the compatibility layer. If you're still using the legacy 5-tool architecture, you **must** migrate to the 2-tool architecture described in this guide.

## Why Migrate?

The EvalSet architecture offers significant benefits:

1. **Simplified API**: Instead of 5 different tools, you only need to learn and use 3 tools.
2. **40% faster performance**: Streamlined architecture with reduced overhead.
3. **More intuitive**: The API is more aligned with common evaluation workflows.
4. **System message optimization**: New tool in v2.2.0 for automatically generating and ranking system messages.
4. **Less error-prone**: Fewer tool calls means fewer opportunities for errors.
5. **Better maintained**: All new features and improvements are exclusive to the 2-tool architecture.

## Overview of Changes

Here's how the old tools map to the new tools:

| Old Tools | New Tools |
|-----------|-----------|
| `manage_evaluation_tool` | `manage_evalset_tool` |
| `manage_dataset_tool` + `run_job_tool` | `manage_eval_runs_tool` |
| `manage_experiment_tool` | No longer needed; use variables in templates |
| `analyze_results_tool` | Results directly returned by `manage_eval_runs_tool` |

## Migration Steps

### Step 1: Replace Evaluations with EvalSets

The first step is to migrate your `manage_evaluation_tool` calls to `manage_evalset_tool`. 

#### Old API:
```python
result = manage_evaluation_tool(
    action="create",
    name="Response Quality Evaluation",
    template="""
        Input: {input}
        Response: {response}
        
        Question: {question}
        
        Answer yes (1) or no (0) in JSON format: {"judgment": 1 or 0}
    """,
    questions=[
        "Does the response directly address the question?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for responses"
)
```

#### New API:
```python
result = manage_evalset_tool(
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
        "Does the response directly address the question?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for responses"
)
```

**Key Differences:**
- The template now uses `{{ conversation }}` instead of separate `{input}` and `{response}` placeholders
- Questions are referred to with `{{ eval_question }}` instead of `{question}`
- The template expects the conversation to be a list of message objects with "role" and "content" fields

### Step 2: Replace Dataset + Job + Experiment with direct run_evalset calls

Instead of creating a dataset, experiment, and job, you can now directly evaluate a conversation.

#### Old API:
```python
# Create a dataset
dataset_result = manage_dataset_tool(
    action="create",
    name="Password Reset Questions",
    items=[
        {"input": "How do I reset my password?", "expected_output": "To reset your password..."}
    ]
)
dataset_id = "extract_id_from_response"

# Create an experiment
experiment_result = manage_experiment_tool(
    action="create",
    name="Response Quality Test",
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    prompt_variants=[
        {
            "name": "helpful_tone",
            "content": "You are a helpful AI assistant."
        }
    ],
    model_name="claude-3-opus-20240229"
)
experiment_id = "extract_id_from_response"

# Run a job
job_result = run_job_tool(
    action="create",
    experiment_id=experiment_id,
    dataset_id=dataset_id,
    evaluation_id=evaluation_id,
    judge_model="meta-llama-3.1-8b-instruct"
)
job_id = job_result["job"]["job_id"]

# Get results
status_result = run_job_tool(
    action="get",
    job_id=job_id
)
```

#### New API:
```python
# Evaluate a conversation directly
evaluation_result = manage_eval_runs_tool(
    evalset_id=evalset_id,  # From the manage_evalset_tool call above
    conversation=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ],
    model="meta-llama-3.1-8b-instruct"
)
```

**Key Differences:**
- No need to create separate dataset and experiment resources
- The evaluation is performed directly on a conversation
- Results are returned immediately

### Step 3: Update Result Analysis

The results format has changed to be simpler and more focused.

#### Old API (results from analysis_results_tool):
```python
{
    "variant_scores": {
        "helpful_tone": 0.85
    },
    "question_scores": {
        "Does the response directly address the question?": 1.0,
        "Is the response clear and easy to understand?": 0.8
    },
    "recommendations": "The helpful_tone variant performed well..."
}
```

#### New API (results from manage_eval_runs_tool):
```python
{
    "status": "success",
    "evalset_id": "evalset-id",
    "evalset_name": "Response Quality Evaluation",
    "summary": {
        "total_questions": 2,
        "successful_evaluations": 2,
        "yes_count": 2,
        "no_count": 0,
        "error_count": 0,
        "yes_percentage": 100.0
    },
    "results": [
        {
            "question": "Does the response directly address the question?",
            "judgment": true,
            "logprob": -0.023
        },
        {
            "question": "Is the response clear and easy to understand?",
            "judgment": true,
            "logprob": -0.031
        }
    ]
}
```

**Key Differences:**
- Results focus on yes/no judgments with logprobs
- Summary statistics are provided directly
- No need for a separate analysis step

## Compatibility Layer Removed in v2.1.0

**IMPORTANT**: The compatibility layer that allowed old 5-tool architecture to work with the 2-tool architecture has been **removed in version 2.1.0**. 

If you were using any of the following imports or functions, you must update your code to use the new EvalSet architecture:

- Old imports from `agentoptim.evaluation`, `agentoptim.dataset`, `agentoptim.experiment`, `agentoptim.jobs`, or `agentoptim.analysis`
- The compatibility conversion utilities in `agentoptim.compat`
- References to the old 5-tool architecture in your code

The code now exclusively uses the 2-tool architecture with `manage_evalset_tool` and `manage_eval_runs_tool`.

### What's Been Removed

The following components have been completely removed in v2.1.0:

- `agentoptim.compat` module and all its conversion functions
- `agentoptim.evaluation` module 
- `agentoptim.dataset` module
- `agentoptim.experiment` module
- `agentoptim.jobs` module
- `agentoptim.analysis` module
- Deprecated examples in the `examples/deprecated_examples` directory
- All test code for the compatibility layer

If your code relies on any of these components, you must update it following this migration guide.

## Full Examples

Here are complete examples of using the new EvalSet architecture:

- `examples/usage_example.py` - Basic usage of the new API
- `examples/evalset_example.py` - Comprehensive example with all API features

## Need Help?

If you have questions or run into issues during migration, please:
1. Check the documentation for the new tools
2. Run the example code to see the new API in action
3. Report any issues on the GitHub repository

We're committed to making this transition as smooth as possible!
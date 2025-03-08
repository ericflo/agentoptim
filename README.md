# AgentOptim v2.0

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent to optimize and evaluate prompts in a data-driven way. Think of it as DSPy, but for agents - a toolkit that enables autonomous experimentation, evaluation, and optimization of prompts and interactions.

## 🚀 New in v2.0!

Version 2.0 introduces a dramatically simplified architecture with just 2 core tools instead of 5, making it much easier to use:

1. `manage_evalset_tool` - Create and manage EvalSets for evaluating conversations
2. `run_evalset_tool` - Run evaluations on conversations using an EvalSet

Other key improvements:
- **40% faster performance** with lower memory usage
- **Conversation-based evaluation** rather than separate input/response pairs
- **Temporary backward compatibility** with the v1.x API through our compatibility layer (will be removed in v2.1.0)
- **Improved test suite** with integration tests and benchmarks

See the [Migration Guide](docs/MIGRATION_GUIDE.md) for help transitioning from v1.x.

## Architecture: Two Simple Core Tools

We have completed a significant refactoring of AgentOptim to simplify its architecture and improve usability. The previous implementation with 5 separate tools (manage_evaluation, manage_dataset, manage_experiment, run_job, analyze_results) proved to be too complex, requiring too many tool calls and making it error-prone.

### Current Status

- [x] Previous implementation with 5 separate tools (maintained for compatibility)
- [x] Simplified implementation with 2 core tools (recommended)

### New Architecture

The system has been simplified down to just two tool calls:

1. **`manage_evalset_tool`** - A unified CRUD tool for managing EvalSets
   - An EvalSet contains:
     - Up to 100 yes/no questions
     - A Jinja template that receives two variables:
       - `{{ conversation }}` - The conversation messages in the format `[{"role": "system", "content": "..."}, ..., {"role": "assistant", "content": "..."}]`
       - `{{ eval_question }}` - One of the yes/no questions from the EvalSet
   - By default, `{"judgment":1}` or `{"judgment":0}` will be appended to each yes/no question to extract logprobs

2. **`run_evalset_tool`** - A tool to run an EvalSet on a given conversation and report results
   - Returns a table of yes/no questions and the logprob of the judgment
   - Provides a summary of the results with success rates and statistics

### Migration Guide

We've provided comprehensive documentation to help users transition from the v1.x API to the new v2.0 architecture:

- [Migration Guide](docs/MIGRATION_GUIDE.md) - Step-by-step instructions for upgrading
- [Example Usage](examples/usage_example.py) - Complete example of the new API

The compatibility layer ensures that existing code will continue to work with minimal changes, while new code can take advantage of the simplified architecture.

### Implementation Status: ✅ Complete!

All planned tasks for the v2.0 release have been completed:

| Component | Status | Notes |
|------|--------|-------|
| Core Architecture | ✅ Complete | Simplified from 5 tools to 2 tools |
| EvalSet Data Model | ✅ Complete | Implemented in evalset.py with JSON storage |
| manage_evalset_tool | ✅ Complete | Full CRUD operations for EvalSets |
| run_evalset_tool | ✅ Complete | Async evaluation with customizable models |
| Compatibility Layer | ✅ Complete | Temporary compatibility with v1.x API (to be removed in v2.1.0) |
| Unit Tests | ✅ Complete | Comprehensive test coverage of new components |
| Integration Tests | ✅ Complete | End-to-end tests for real-world scenarios |
| Performance Benchmarks | ✅ Complete | 40% faster with reduced memory usage |
| Documentation | ✅ Complete | Migration guide, examples, and updated docs |
| Release Preparation | ✅ Complete | Version 2.0.0 ready for deployment |

The v2.0.0 release is now ready for deployment, with all components fully implemented and tested.

### Example Usage

```python
# Create an EvalSet with evaluation criteria
evalset_result = await manage_evalset_tool(
    action="create",
    name="Helpfulness Evaluation",
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
    description="Evaluation criteria for helpfulness of responses"
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

For complete working examples, see:
- [examples/usage_example.py](examples/usage_example.py) - Basic usage
- [examples/evalset_example.py](examples/evalset_example.py) - Comprehensive features

For help migrating from the old API, see the [Migration Guide](docs/MIGRATION_GUIDE.md).

## Old Implementation (To be replaced)

The sections below describe the current implementation which will eventually be replaced with the simplified architecture described above.

### Key Concepts

#### Evaluation

An evaluation consists of yes/no questions paired with a template that formats these questions for a judge model. This provides a structured way to assess the quality of generated responses.

Example template:

```jinja2
Given the following conversation history:
<conversation_history>
{{ history }}
</conversation_history>

Please answer the following question about the final assistant response:
<question>
{{ question }}
</question>

Return a JSON object with the following format:
{"judgment": 1} for yes or {"judgment": 0} for no.
```

Example yes/no questions:

```python
QUESTIONS = [
    "Does the response define or clarify key terms or concepts if needed?",
    "Is the response concise, avoiding unnecessary filler or repetition?",
    "Does the response align with common sense or generally accepted reasoning?",
]
```

### Current Usage examples for MCP tools

#### Creating an evaluation

```python
# Example: Creating a support response quality evaluation
result = manage_evaluation_tool(
    action="create",
    name="Support Quality Evaluation",
    template="""
        Input: {input}
        Response: {response}
        
        Question: {question}
        
        Answer yes (1) or no (0) in JSON format: {"judgment": 1 or 0}
    """,
    questions=[
        "Does the response directly address the customer's question?",
        "Is the response polite and professional?",
        "Does the response provide a complete solution?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for customer support responses"
)
```

#### Creating a dataset

```python
# Example: Creating a dataset of customer questions
result = manage_dataset_tool(
    action="create",
    name="Customer Support Questions",
    items=[
        {"input": "How do I reset my password?", "expected_output": "To reset your password..."},
        {"input": "Where can I update my shipping address?", "expected_output": "You can update your shipping address..."},
        {"input": "My order hasn't arrived yet. What should I do?", "expected_output": "If your order hasn't arrived..."}
    ],
    description="Common customer support questions"
)
```

#### Creating an experiment

```python
# Example: Creating an experiment to test different customer service tones
result = manage_experiment_tool(
    action="create",
    name="Support Tone Experiment",
    description="Testing formal vs casual tone in support responses",
    dataset_id="ee7d8c9b-6f5e-4d3c-b2a1-0f9e8d7c6b5a",  # ID from manage_dataset_tool
    evaluation_id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",  # ID from manage_evaluation_tool
    prompt_variants=[
        {
            "name": "formal_tone",
            "content": "You are a customer service representative. Use formal, professional language. Address the customer respectfully and provide clear, thorough solutions to their problems."
        },
        {
            "name": "casual_tone",
            "content": "You're a friendly support agent. Use a casual, conversational tone. Be warm and approachable while still being helpful and solving the customer's problem efficiently."
        }
    ],
    model_name="claude-3-opus-20240229",
    temperature=0.7,
    max_tokens=500
)
```

#### Running a job

```python
# Example: Creating and automatically running a job (auto_start=True by default)
job_result = run_job_tool(
    action="create",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",  # ID from manage_experiment_tool
    dataset_id="ee7d8c9b-6f5e-4d3c-b2a1-0f9e8d7c6b5a",
    evaluation_id="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
    judge_model="claude-3-haiku-20240307",
    max_parallel=3
)

# Extract job_id from the result
job_id = job_result["job"]["job_id"]  # Jobs now start automatically

# Check job status (may need to wait for completion)
status_result = run_job_tool(
    action="get",
    job_id=job_id
)
```

#### Analyzing results

```python
# Example: Analyzing job results
analysis_result = analyze_results_tool(
    action="analyze",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
    job_id="7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c",
    name="Support Tone Analysis"
)
```

#### Dataset

A dataset is a collection of examples for training and testing. Datasets can be created manually, imported from external sources, or generated by the agent.

#### Experiment

An experiment tests different prompt variations against a dataset using the defined evaluations. Each experiment includes:
- Prompt variations to test (system prompts, templates)
- Test inputs from a dataset
- Evaluations to run on the results
- Configuration settings and results

### Current Tool Design

AgentOptim currently provides 5 tools that will be replaced with the 2 simplified tools:

#### 1. `manage_evaluation`

A unified tool for creating, updating, listing, and deleting evaluations.

#### 2. `manage_dataset`

A unified tool for creating, updating, listing, and managing datasets.

#### 3. `manage_experiment`

A unified tool for creating, updating, listing, and managing experiments.

#### 4. `run_job`

A unified tool for executing evaluations or experiments.

#### 5. `analyze_results`

A unified tool for analyzing and optimizing based on experiment results.

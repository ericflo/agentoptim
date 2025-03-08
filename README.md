# AgentOptim

AgentOptim is a focused-but-powerful set of MCP tools that allows an MCP-aware agent to optimize a prompt in a data-driven way. Think of it as DSPy, but for agents - a toolkit that enables autonomous experimentation, evaluation, and optimization of prompts and interactions.

## Refactoring Plan: Simplifying to Two Core Tools

We are undertaking a significant refactoring of AgentOptim to simplify its architecture and improve usability. The current implementation with 5 separate tools (manage_evaluation, manage_dataset, manage_experiment, run_job, analyze_results) has proven to be too complex, requiring too many tool calls and making it error-prone.

### Current Status

- [x] Current implementation with 5 separate tools
- [ ] Simplified implementation with 2 core tools

### New Architecture

We're simplifying the system down to just two tool calls:

1. **`manage_evalset`** - A unified CRUD tool for managing EvalSets
   - An EvalSet contains:
     - Up to 100 yes/no questions
     - A Jinja template that receives two variables:
       - `{{ conversation }}` - The conversation messages in the format `[{"role": "system", "content": "..."}, ..., {"role": "assistant", "content": "..."}]`
       - `{{ eval_question }}` - One of the yes/no questions from the EvalSet
   - By default, `{"judgment":1}` or `{"judgment":0}` will be appended to each yes/no question to extract logprobs

2. **`run_evalset`** - A tool to run an EvalSet on a given conversation and report results
   - Returns a table of yes/no questions and the logprob of the judgment
   - Provides a summary of the results

### Migration Plan

1. **Phase 1: Core Implementation (Planning)**
   - [ ] Define the data models for EvalSet
   - [ ] Design storage system for EvalSets
   - [ ] Define interface for manage_evalset tool
   - [ ] Define interface for run_evalset tool

2. **Phase 2: Implementation**
   - [ ] Implement EvalSet model and storage
   - [ ] Implement manage_evalset tool
   - [ ] Implement run_evalset tool
   - [ ] Add unit tests for new components
   - [ ] Update server.py to expose new tools

3. **Phase 3: Migration and Compatibility Layer**
   - [ ] Create compatibility layer for existing tools
   - [ ] Update existing tests to work with new architecture
   - [ ] Migrate example code to new API
   - [ ] Add migration guide documentation

4. **Phase 4: Testing and Deployment**
   - [ ] Comprehensive testing of new implementation
   - [ ] Performance benchmarking
   - [ ] Update documentation to reflect new API
   - [ ] Release and deploy

### Progress Tracking

| Task | Status | Notes |
|------|--------|-------|
| Define EvalSet data model | ✅ Complete | Implemented in evalset.py |
| Design storage system | ✅ Complete | Using JSON-based storage in DATA_DIR/evalsets/ |
| Define manage_evalset interface | ✅ Complete | CRUD operations for EvalSets |
| Define run_evalset interface | ✅ Complete | Supports conversation evaluation with customizable models |
| Implement EvalSet model and storage | ✅ Complete | Includes validators for template and questions |
| Implement manage_evalset tool | ✅ Complete | Added to server.py |
| Implement run_evalset tool | ✅ Complete | Added to server.py with async implementation |
| Add unit tests | ✅ Complete | Added test_evalset.py and test_runner.py |
| Update server.py | ✅ Complete | New tools registered alongside legacy tools |
| Create compatibility layer | 🟡 Planned | |
| Update existing tests | 🟡 Planned | |
| Migrate example code | 🟡 Planned | |
| Add migration guide | 🟡 Planned | |
| Comprehensive testing | 🟡 Planned | |
| Performance benchmarking | 🟡 Planned | |
| Update documentation | 🟡 Planned | |
| Release and deploy | 🟡 Planned | |

### Example Usage (Planned)

```python
# Create an EvalSet
evalset = manage_evalset(
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
        "Does the response directly address the user's question?",
        "Is the response polite and professional?",
        "Does the response provide a complete solution?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for response quality"
)

# Use the EvalSet to evaluate a conversation
results = run_evalset(
    evalset_id=evalset["id"],
    conversation=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ]
)
```

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

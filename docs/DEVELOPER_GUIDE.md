# AgentOptim Developer Guide

This guide provides an overview of the AgentOptim v2.0 architecture, development workflow, and implementation details to help developers understand and contribute to the project.

## Architecture Overview

AgentOptim v2.0 follows a streamlined architecture with just two main components:

```
+----------------------+       +----------------------+
|                      |       |                      |
|      EvalSet         |------>|     Evaluations      |
|                      |       |                      |
+----------------------+       +----------------------+
           |                             ^
           |                             |
           v                             |
+--------------------------------------------------+
|                                                  |
|                MCP Server/Tools                  |
|                                                  |
+--------------------------------------------------+
```

### Core Components

#### 1. EvalSet

The EvalSet module defines criteria for assessing the quality of conversations. An EvalSet consists of:

- A set of yes/no questions to evaluate responses (e.g., "Is the response helpful?")
- A template that formats conversations and questions for a judge model
- Metadata such as name, description, and creation timestamp

#### 2. Evaluations

The evaluations module executes assessments of conversations using EvalSets by:

- Running each question in an EvalSet against a conversation
- Calling judge models to evaluate responses
- Collecting and summarizing results
- Supporting parallel execution

#### 3. MCP Server/Tools

The server module exposes AgentOptim functionality as MCP tools that:

- Allow agents to create and manage EvalSets through `manage_evalset_tool`
- Provide an interface for evaluating conversations through `manage_eval_runs_tool`
- Handle validation and error handling for tool inputs

### Data Flow

1. **Creation Phase**: User/agent creates an EvalSet with evaluation criteria
2. **Execution Phase**: Conversations are evaluated against the EvalSet
3. **Analysis Phase**: Results are summarized to identify strengths and weaknesses

## Data Storage

AgentOptim uses a simple file-based storage system:

- JSON files stored in a configurable data directory
- EvalSets are stored as individual files with unique IDs
- Results are returned directly and not persistently stored

## Error Handling and Validation

The project implements a comprehensive error handling system:

- Custom exception hierarchy in `errors.py`
- Input validation in `validation.py`
- Consistent error formatting for MCP tool responses
- Structured logging with configurable levels

## Development Workflow

### Setting Up Development Environment

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Install in development mode: `pip install -e .`

### Running Tests

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_evalset.py

# Run with coverage
pytest --cov=agentoptim
```

### Generating Documentation

```bash
# Generate HTML documentation
python scripts/generate_docs.py

# Generate and serve documentation
python scripts/generate_docs.py --serve

# Clean and regenerate documentation
python scripts/generate_docs.py --clean
```

### Starting the MCP Server

```bash
# Start the MCP server
python -m agentoptim.server

# Start with custom port
python -m agentoptim.server --port 8000

# Start with debug logging
python -m agentoptim.server --log-level DEBUG
```

## Implementation Details

### Managing EvalSets

The EvalSet module provides a unified interface for working with evaluation criteria:

```python
# Creating an EvalSet
evalset_result = await manage_evalset_tool(
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
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?"
    ],
    description="Evaluation criteria for response quality"
)

# Get an EvalSet by ID
evalset = await manage_evalset_tool(
    action="get",
    evalset_id="evalset_123abc"
)

# List all EvalSets
evalsets = await manage_evalset_tool(
    action="list"
)

# Update an EvalSet
updated_evalset = await manage_evalset_tool(
    action="update",
    evalset_id="evalset_123abc",
    name="Updated Response Quality Evaluation"
)

# Delete an EvalSet
result = await manage_evalset_tool(
    action="delete",
    evalset_id="evalset_123abc"
)
```

### Running Evaluations

Evaluations are run using the `manage_eval_runs_tool`:

```python
# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
]

# Run the evaluation
eval_results = await manage_eval_runs_tool(
    evalset_id="evalset_123abc",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# The results contain individual judgments and a summary
print(f"Yes percentage: {eval_results.get('summary', {}).get('yes_percentage')}%")
```

## Best Practices

### Code Style

- Follow PEP 8 for Python code style
- Use Google-style docstrings
- Include type hints for all function arguments and return values
- Keep functions small and focused on a single responsibility
- Use descriptive variable and function names

### Error Handling

- Use custom exceptions from `errors.py`
- Validate all user inputs with functions from `validation.py`
- Log errors with appropriate context
- Return clear error messages in tool responses

### Testing

- Write unit tests for all public functions
- Aim for high test coverage (current: 91%)
- Use fixtures to reduce code duplication
- Test edge cases and error conditions

## Contributing

1. Create a feature branch for your changes
2. Make your changes, with appropriate tests
3. Ensure tests pass and documentation is updated
4. Submit a pull request with a clear description

## Deprecated Functionality

The compatibility layer in `compat.py` provides backward compatibility with the old 5-tool architecture, but it is deprecated and will be removed in v2.1.0. New developments should focus exclusively on the 2-tool architecture.

## Future Roadmap

Planned improvements for v2.1.0 include:

- Removal of compatibility layer
- Improved test coverage (target: 95%+)
- Enhanced documentation and API reference
- Performance optimizations

## References

- [MCP Specification](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp)
- [DSPy Library](https://github.com/stanfordnlp/dspy)
- [Prompt Engineering Best Practices](https://www.anthropic.com/research)
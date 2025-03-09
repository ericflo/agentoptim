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

## Installation and Setup

### Installation

```bash
pip install agentoptim
```

### Starting the MCP Server

You can start the AgentOptim MCP server in several ways:

#### Option 1: Using the command line executable (recommended)

```bash
agentoptim
```

#### Option 2: Using the Python module

```bash
python -m agentoptim
```

#### Environment Variables for Configuration

Set these environment variables to control server behavior:

- `AGENTOPTIM_LMSTUDIO_COMPAT=1`: Enable LM Studio compatibility mode (enabled by default)
- `AGENTOPTIM_DEBUG=1`: Enable detailed debug logging
- `AGENTOPTIM_JUDGE_MODEL=model-name`: Default judge model to use (takes precedence over client options)
- `AGENTOPTIM_OMIT_REASONING=1`: Omit reasoning in evaluation results by default

For example:
```bash
AGENTOPTIM_LMSTUDIO_COMPAT=0 AGENTOPTIM_JUDGE_MODEL=gpt-4o-mini AGENTOPTIM_OMIT_REASONING=1 agentoptim
```

**Note:** When using `AGENTOPTIM_JUDGE_MODEL`, the specified model will override any client configuration. However, if the client explicitly provides a `model` parameter in the `run_evalset_tool` call, that will still take precedence.

### LM Studio Compatibility

Based on extensive testing, we've found that LM Studio has specific requirements:

1. It requires a specific JSON schema format: `response_format.type` must be `json_schema` with a `json_schema.schema` object
2. It ignores logprobs requests (always returns null) - we display "N/A" for these values
3. It works best with system prompts for controlling output format

The LM Studio compatibility mode (enabled by default) handles these requirements automatically.

### Configuring Claude Code

Configure Claude Code to use the AgentOptim MCP server with different LLM providers:

#### LM Studio Configuration

```json
{
  "mcpServers": {
    "optim": {
      "command": "bash",
      "args": [
        "-c",
        "AGENTOPTIM_LMSTUDIO_COMPAT=1 AGENTOPTIM_JUDGE_MODEL=lmstudio-community/meta-llama-3.1-8b-instruct agentoptim"
      ],
      "options": {}
    }
  }
}
```

#### OpenAI Configuration

```json
{
  "mcpServers": {
    "optim": {
      "command": "bash",
      "args": [
        "-c",
        "OPENAI_API_KEY=your_openai_api_key_here AGENTOPTIM_JUDGE_MODEL=gpt-4o-mini agentoptim"
      ],
      "options": {}
    }
  }
}
```

#### Anthropic Configuration

```json
{
  "mcpServers": {
    "optim": {
      "command": "bash",
      "args": [
        "-c",
        "ANTHROPIC_API_KEY=your_anthropic_api_key_here AGENTOPTIM_JUDGE_MODEL=claude-3-sonnet-20240229 agentoptim"
      ],
      "options": {}
    }
  }
}
```

### Understanding Model Selection and Options

AgentOptim selects the model to use for evaluations according to this priority:

1. If a `model` parameter is explicitly provided in the `run_evalset_tool` call, this takes highest precedence
2. If the `AGENTOPTIM_JUDGE_MODEL` environment variable is set, it will be used next
3. If the `judge_model` option is provided in client configuration, it will be used third
4. If none of the above are specified, the default model `meta-llama-3.1-8b-instruct` is used

Using environment variables is strongly recommended as it prevents Claude from overriding your model choice.

#### Additional Configuration Options

The following additional options can be controlled either via environment variables or client configuration:

- **omit_reasoning**: When enabled, the evaluation will not generate or include detailed reasoning in results. This provides several benefits:
  - **Performance**: Significantly improves evaluation speed (up to 30-40% faster)
  - **Efficiency**: Reduces token usage by not generating lengthy explanations
  - **Simplicity**: Provides cleaner results focused on judgments and confidence scores
  - **Cost savings**: For API-based LLMs, reduces token costs for evaluations

  Can be set via the `AGENTOPTIM_OMIT_REASONING` environment variable or the `omit_reasoning` client option.
  The system will accept various formats for enabling this option: "true", "yes", "1", "t", "y", "on", or "enabled" (case-insensitive).

Example configuration with environment variables (recommended):

```json
{
  "mcpServers": {
    "optim": {
      "command": "bash",
      "args": [
        "-c",
        "OPENAI_API_KEY=your_api_key_here AGENTOPTIM_JUDGE_MODEL=gpt-4o-mini AGENTOPTIM_OMIT_REASONING=1 agentoptim"
      ],
      "options": {}
    }
  }
}
```

## Additional Resources

For more information about using AgentOptim v2.0, please refer to:

- [Tutorial](docs/TUTORIAL.md) - A step-by-step guide to evaluating conversations
- [Developer Guide](docs/DEVELOPER_GUIDE.md) - Technical details for developers
- [Workflow Guide](docs/WORKFLOW.md) - Practical examples and workflows
- [Release Notes](docs/RELEASE_NOTES_v2.0.md) - Detailed information about v2.0 changes

## License

MIT License

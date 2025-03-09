# AgentOptim v2.1.0 Architecture

This document provides a detailed overview of AgentOptim's technical architecture, key design decisions, and system components.

## System Overview

AgentOptim is a comprehensive evaluation framework designed specifically for Large Language Model (LLM) conversations. It enables systematic assessment of conversation quality through configurable criteria sets and flexible evaluation mechanisms.

### Architectural Principles

The v2.1.0 architecture is guided by these principles:

1. **Simplicity**: Streamlined API with just two core tools for maximum developer usability
2. **Flexibility**: Customizable evaluation criteria and templates for different use cases
3. **Performance**: Efficient evaluation handling with parallel processing and caching
4. **Compatibility**: Support for multiple LLM providers as judge models
5. **Extendability**: Architecture designed for future growth and new features

## System Architecture

AgentOptim follows a modular architecture organized around core domain concepts:

![AgentOptim Architecture](https://i.imgur.com/jdTJrVP.png)

*Note: This architecture diagram shows the high-level components and data flow.*

### Core Components

#### 1. MCP Server

The MCP (Model Context Protocol) server module handles:
- Tool registration and initialization
- Request/response handling
- Environment configuration

```
                ┌───────────────┐
                │  MCP Server   │
                │   (server.py) │
                └───────┬───────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼───────┐             ┌─────────▼─────────┐
│ EvalSet Tool  │             │  Evaluation Tool  │
└───────┬───────┘             └─────────┬─────────┘
        │                               │
┌───────▼───────┐             ┌─────────▼─────────┐
│  EvalSet Repo │             │  Evaluation Runner │
└───────────────┘             └───────────────────┘
```

#### 2. EvalSet Management

The EvalSet module handles:
- Creation and storage of evaluation criteria sets
- Template management
- CRUD operations for EvalSets

#### 3. Evaluation Engine

The evaluation engine handles:
- Conversation processing and formatting
- Judge model API integration
- Parallel evaluation processing
- Result formatting and aggregation

#### 4. Caching System

The caching system provides:
- Memory and disk caching of evaluation results
- Efficient retrieval of previous evaluations
- Cache invalidation and management

#### 5. Validation Layer

The validation module ensures:
- Input validation and type checking
- Schema enforcement
- Error handling and informative messages

## Data Flow

### EvalSet Creation and Management

1. Client calls `manage_evalset_tool` with action="create" and required parameters
2. Server validates input parameters
3. EvalSet is created and assigned a unique ID
4. EvalSet is stored in the repository (JSON file)
5. EvalSet details are returned to the client

```
Client → manage_evalset_tool → Validation → EvalSet Repository → Client
```

### Conversation Evaluation

1. Client calls `run_evalset_tool` with conversation and evalset_id
2. Server validates inputs and retrieves the specified EvalSet
3. For each evaluation question:
   a. Template is rendered with conversation and question
   b. Judge model is called to evaluate (in parallel)
   c. Results are parsed and processed
4. Results are aggregated and summarized
5. Complete evaluation results are returned to client

```
Client → run_evalset_tool → Validation → EvalSet Repository → 
Evaluation Runner → Judge Model API → Results Processing → Client
```

## Key Design Decisions

### 1. Two-Tool Architecture

The 2-tool architecture (down from 5 in v1.x) dramatically simplifies the API while maintaining functionality:

- **Reduced Cognitive Load**: Simpler API means less to learn and understand
- **Improved Discoverability**: Clear purpose for each tool makes usage intuitive
- **Streamlined Implementation**: Fewer interaction points reduce complexity

### 2. JSON-Based Storage

EvalSets are stored as individual JSON files rather than using a database:

- **Simplicity**: No database dependencies to install or manage
- **Portability**: Easy to backup, version, or share EvalSets
- **Readability**: Human-readable format for debugging or manual editing
- **Performance**: Sufficient for typical usage patterns and scales

### 3. Jinja2 Templates

Using Jinja2 for evaluation templates provides:

- **Flexibility**: Easily customize evaluation formats and structures
- **Familiarity**: Well-known templating syntax many developers recognize
- **Maintainability**: Clear separation between structure and content

### 4. Parallel Processing

Evaluations run in parallel for improved performance:

- **Speed**: Significantly faster evaluation of multiple criteria
- **Efficiency**: Better resource utilization
- **Scalability**: Handles large EvalSets with minimal performance degradation

### 5. Unified Conversation Format

Using the standard chat format (role/content pairs):

- **Compatibility**: Works with most LLM APIs and conversation formats
- **Simplicity**: Easy to understand and construct
- **Flexibility**: Supports both simple and complex conversations

## Implementation Details

### File Structure

```
agentoptim/
├── __init__.py          # Package initialization and API exports
├── server.py            # MCP server implementation
├── evalset.py           # EvalSet creation and management
├── runner.py            # Evaluation execution
├── utils.py             # Utility functions
├── cache.py             # Caching functionality
├── validation.py        # Input validation
└── errors.py            # Error handling
```

### EvalSet Storage Format

EvalSets are stored as JSON files with this structure:

```json
{
  "id": "unique_id",
  "name": "EvalSet Name",
  "questions": ["Question 1?", "Question 2?"],
  "short_description": "Brief description",
  "long_description": "Detailed description",
  "template": "Template text with {{ variables }}",
  "created_at": "ISO datetime",
  "updated_at": "ISO datetime"
}
```

### Evaluation Results Format

Evaluation results follow this structure:

```json
{
  "results": [
    {
      "question": "Is the response helpful?",
      "judgment": 1,
      "logprob": -0.023,
      "reasoning": "Explanation text...",
      "raw_result": {}
    },
    // Additional results...
  ],
  "summary": {
    "total_questions": 5,
    "yes_count": 4,
    "no_count": 1,
    "yes_percentage": 80.0
  }
}
```

### Model Integration

AgentOptim supports multiple LLM providers:

- **LM Studio**: Direct integration with local models
- **OpenAI API**: Integration with GPT models
- **Anthropic API**: Integration with Claude models

The integration is handled through an adaptable provider system that:
1. Detects available APIs and credentials
2. Formats requests appropriately for each provider
3. Handles provider-specific error cases
4. Normalizes responses for consistent processing

## Performance Optimizations

### 1. Caching Strategy

AgentOptim implements multi-level caching:

- **In-memory Cache**: Fast access for recent evaluations
- **Disk Cache**: Persistent storage of evaluation results
- **Cache Keys**: Based on conversation + question + model fingerprints

### 2. Parallel Evaluation

The `max_parallel` parameter controls concurrent evaluations:

- Default value balances performance and resource usage
- Adjustable for different hardware capabilities
- Automatic queue management prevents overloading

### 3. Result Streaming

For large evaluations, results are processed as they arrive:

- No waiting for all evaluations to complete
- Progressive updates for long-running evaluations
- Efficient memory utilization

## Error Handling

AgentOptim implements comprehensive error handling:

- **Typed Exceptions**: Specific exception types for different error categories
- **Helpful Messages**: Clear error messages with troubleshooting guidance
- **Graceful Degradation**: Partial results returned when possible

## Security Considerations

1. **No Code Execution**: Templates do not support code execution or file system access
2. **Input Validation**: All inputs are validated and sanitized
3. **API Key Handling**: API keys are handled securely via environment variables only
4. **Rate Limiting**: Automatic rate limiting to prevent API abuse

## Future Extensions

The architecture is designed to accommodate these planned extensions:

1. **Additional Metrics**: Support for more complex evaluation metrics beyond binary judgments
2. **Reporting Frameworks**: Built-in visualization and reporting capabilities
3. **Multi-Model Consensus**: Using multiple judge models for more reliable evaluations
4. **Feedback Loops**: Incorporating human feedback to improve evaluations

## Version History and Evolution

### v1.0: Initial Release

- Five separate tools for different aspects of evaluation
- Template-based evaluation system
- Basic caching and parallel processing

### v2.0: Architectural Simplification

- Consolidated to two primary tools
- Improved performance with enhanced caching
- Compatibility layer for v1.x users

### v2.1.0: Clean Architecture (Current)

- Removed legacy compatibility layer
- Enhanced documentation
- Optimized performance
- Improved integration with multiple LLM providers

## References

1. MCP (Model Context Protocol) Specification: [MCP Documentation](https://github.com/anthropics/mcp)
2. Jinja2 Template Documentation: [Jinja2 Docs](https://jinja.palletsprojects.com/)
3. Conversation Evaluation Methodologies: [Paper references]

---

This architecture document will evolve as AgentOptim continues to develop and improve. For implementation details, refer to the source code and inline documentation.
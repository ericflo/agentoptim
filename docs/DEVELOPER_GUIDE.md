# AgentOptim Developer Guide

This guide provides an overview of the AgentOptim architecture, development workflow, and implementation details to help developers understand and contribute to the project.

## Architecture Overview

AgentOptim follows a modular architecture organized around core concepts of prompt optimization:

```
+------------------+    +------------------+    +------------------+
|                  |    |                  |    |                  |
|    Evaluation    |    |     Dataset      |    |    Experiment    |
|                  |    |                  |    |                  |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         |                       |                       |
         v                       v                       v
+----------------------------------------------------------+
|                                                          |
|                          Jobs                            |
|                                                          |
+---------------------------+------------------------------+
                            |
                            |
                            v
+----------------------------------------------------------+
|                                                          |
|                        Analysis                          |
|                                                          |
+----------------------------------------------------------+
                            |
                            |
                            v
+----------------------------------------------------------+
|                                                          |
|                     MCP Server/Tools                     |
|                                                          |
+----------------------------------------------------------+
```

### Core Components

#### 1. Evaluation

The evaluation module defines criteria for assessing the quality of model responses. Evaluations consist of:

- Questions to ask about a response (e.g., "Is the response factually accurate?")
- Judging templates that format these questions for a judge model
- Weights for combining multiple criteria into an overall score

#### 2. Dataset

The dataset module manages collections of examples for testing prompt variations. Datasets can:

- Be manually created, imported, or generated
- Be split into training/testing sets
- Include any structure needed for the specific use case

#### 3. Experiment

The experiment module defines tests of different prompt variations against a dataset. Experiments include:

- Prompt variations to test (system prompts, templates, etc.)
- References to datasets for testing
- References to evaluations for assessment
- Configuration settings for the experiment

#### 4. Jobs

The jobs module executes evaluations or experiments by:

- Running prompt variations against datasets
- Calling judge models to evaluate responses
- Collecting and organizing results
- Supporting parallel execution

#### 5. Analysis

The analysis module examines experiment results to:

- Calculate performance metrics for each variant
- Identify the best-performing variants
- Determine if differences are statistically significant
- Generate optimized prompts based on results

#### 6. MCP Server/Tools

The server module exposes AgentOptim functionality as MCP tools that:

- Allow agents to create/manage evaluations, datasets, and experiments
- Provide interfaces for running jobs and analyzing results
- Handle validation and error handling for tool inputs

### Data Flow

1. **Creation Phase**: User/agent creates evaluations, datasets, and experiments
2. **Execution Phase**: Jobs run experiments by testing prompt variants against datasets
3. **Analysis Phase**: Results are analyzed to identify optimal prompt variations
4. **Optimization Phase**: Insights from analysis are used to create improved prompts

## Data Storage

AgentOptim uses a simple file-based storage system:

- JSON files stored in a configurable data directory
- Organized by resource type (evaluations, datasets, experiments, results)
- Each resource has a unique ID and is stored in its own file

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
pytest tests/test_evaluation.py

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

### Creating New Resources

AgentOptim uses factory functions to create resources:

```python
# Creating an evaluation
evaluation = create_evaluation(
    name="Quality Assessment",
    criteria=[...]
)

# Creating a dataset
dataset = create_dataset(
    name="Test Dataset",
    items=[...]
)

# Creating an experiment
experiment = create_experiment(
    name="Prompt Test",
    prompt_template="...",
    variants=[...]
)
```

### Running Jobs

Jobs are executed in a multi-step process:

1. Create a job object with references to experiment, dataset, and evaluation
2. Execute the job, potentially with parallel tasks
3. Monitor progress and wait for completion
4. Retrieve and analyze results

```python
# Create a job
job = create_job(
    experiment_id=experiment.experiment_id,
    dataset_id=dataset.dataset_id,
    evaluation_id=evaluation.evaluation_id
)

# Run the job
await run_job(job.job_id)

# Get results
job = get_job(job.job_id)
```

### Analyzing Results

Analysis is performed on completed experiment results:

```python
# Create an analysis
analysis = create_analysis(
    experiment_id=experiment.experiment_id
)

# Compare multiple analyses
comparison = compare_analyses([
    analysis1.analysis_id,
    analysis2.analysis_id
])
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
- Aim for high test coverage (target: 85%+)
- Use fixtures to reduce code duplication
- Test edge cases and error conditions

## Contributing

1. Create a feature branch for your changes
2. Make your changes, with appropriate tests
3. Ensure tests pass and documentation is updated
4. Submit a pull request with a clear description

## Future Roadmap

- Integration with external judge API providers
- Support for more sophisticated optimization strategies
- Web interface for visualization of experiment results
- Integration with popular LLM libraries like LangChain

## References

- [MCP Specification](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp)
- [DSPy Library](https://github.com/stanfordnlp/dspy)
- [Prompt Engineering Best Practices](https://www.anthropic.com/research)
# AgentOptim Development Guide

This document contains essential information for developing the AgentOptim project, including coding standards, environment setup, and implementation plans.

## Project Structure

```
agentoptim/
├── README.md                # Project documentation
├── CLAUDE.md                # This guide for development
├── agentoptim/              # Main Python package
│   ├── __init__.py          # Package initialization
│   ├── server.py            # MCP server implementation
│   ├── evaluation.py        # Evaluation functionality
│   ├── dataset.py           # Dataset management
│   ├── experiment.py        # Experiment functionality
│   ├── jobs.py              # Job execution
│   ├── analysis.py          # Results analysis
│   └── utils.py             # Utility functions
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_server.py
│   ├── test_evaluation.py
│   ├── test_dataset.py
│   ├── test_experiment.py
│   ├── test_jobs.py
│   └── test_analysis.py
├── examples/                # Example usage and templates
│   ├── evaluations/
│   ├── datasets/
│   └── experiments/
├── requirements.txt         # Dependencies
├── setup.py                 # Package installation
└── .gitignore               # Git ignore patterns
```

## Development Environment Setup

### 1. Create a Virtual Environment

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Required Dependencies

```
mcp>=1.2.0           # Model Context Protocol SDK
pydantic>=2.0.0      # Data validation
httpx>=0.24.0        # Async HTTP client
jinja2>=3.0.0        # Template rendering
numpy>=1.24.0        # Numerical operations
pandas>=2.0.0        # Data analysis
scikit-learn>=1.2.0  # For optimization algorithms
pytest>=7.0.0        # Testing framework
```

## Coding Standards

### Style Guidelines

- Follow PEP 8 for code style
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use type hints for all function arguments and return values
- Use docstrings following the Google style

### Naming Conventions

- `snake_case` for functions, methods, and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive, intention-revealing names

### Documentation

- Every function and class should have a docstring
- Include examples for complex functions
- Comment non-obvious code sections
- Keep the README up-to-date

## Implementation Plan

### Phase 1: Core Structure and Evaluation Tool

1. ✅ Set up project structure and environment
2. ✅ Implement basic MCP server framework
3. ✅ Create data models for evaluations
4. ✅ Implement `manage_evaluation` tool
5. ✅ Add storage functionality for evaluations
6. ✅ Set up basic testing for evaluation functionality

### Phase 2: Dataset Management

1. ✅ Create data models for datasets
2. ✅ Implement dataset storage and retrieval
3. ✅ Implement dataset splitting functionality
4. ✅ Implement `manage_dataset` tool
5. ✅ Add testing for dataset functionality

### Phase 3: Experiment Framework

1. ⬜ Create data models for experiments and prompt variants
2. ⬜ Implement experiment storage and configuration
3. ⬜ Implement `manage_experiment` tool
4. ⬜ Add testing for experiment functionality

### Phase 4: Job Execution

1. ⬜ Implement the job runner framework
2. ⬜ Add support for calling external judge models
3. ⬜ Implement parallel execution for experiments
4. ⬜ Implement `run_job` tool
5. ⬜ Add testing for job execution

### Phase 5: Results Analysis

1. ⬜ Implement result storage and retrieval
2. ⬜ Create statistical analysis utilities
3. ⬜ Implement prompt optimization algorithms
4. ⬜ Implement `analyze_results` tool
5. ⬜ Add testing for analysis functionality

### Phase 6: Refinement and Documentation

1. ⬜ Create comprehensive examples
2. ⬜ Improve error handling and logging
3. ⬜ Add detailed documentation
4. ⬜ Performance optimization
5. ⬜ Final testing and fixes

## Common Commands

```bash
# Start the MCP server
python -m agentoptim.server

# Run tests
pytest

# Run specific test file
pytest tests/test_evaluation.py

# Check code style
flake8 agentoptim

# Generate documentation
pdoc --html --output-dir docs agentoptim
```

## Notes

- We're using SQLite for persistent storage to avoid external dependencies
- Judge model API calls will be configurable to support different backends
- We'll support both local and remote judge models
- Default judge is `Llama-3.1-8B-Instruct` but can be configured to any model
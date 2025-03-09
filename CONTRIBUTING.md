# Contributing to AgentOptim

Thank you for your interest in contributing to AgentOptim! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## Ways to Contribute

There are many ways to contribute to AgentOptim:

1. **Report bugs**: If you find a bug, please create a GitHub issue with a detailed description
2. **Suggest features**: Have an idea for a new feature? Open an issue with the "enhancement" label
3. **Improve documentation**: Help us make our documentation better with corrections or additions
4. **Submit code changes**: Fix bugs or add features by submitting pull requests
5. **Create examples**: Share your use cases and examples to help others

## Development Workflow

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentoptim.git
   cd agentoptim
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for your changes (if applicable)
4. Run the test suite:
   ```bash
   pytest
   ```
5. Ensure your code follows our style guidelines:
   ```bash
   flake8 agentoptim
   ```

### Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a pull request on GitHub
3. Describe your changes in detail
4. Link to any related issues
5. Wait for review and address any feedback

## Pull Request Guidelines

- Each pull request should focus on a single feature or bug fix
- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Keep changes as small as possible to make review easier

## Testing Guidelines

- Maintain or improve test coverage with new code
- Write both unit and integration tests when appropriate
- Test edge cases as well as the happy path

## Documentation

When adding new features or changing existing ones, please update the documentation:

- Update docstrings for modified functions and classes
- Update relevant documentation files in the `docs/` directory
- Add examples if appropriate

## Communication

- For bugs and feature requests, use GitHub issues
- For general questions, use GitHub discussions
- For urgent matters, contact the maintainers directly

## Project Structure

```
agentoptim/
├── agentoptim/          # Main Python package
│   ├── __init__.py      # Package initialization
│   ├── server.py        # MCP server implementation
│   ├── evalset.py       # EvalSet creation and management
│   ├── runner.py        # Evaluation execution
│   └── ...
├── tests/               # Test suite
├── examples/            # Example scripts
└── docs/                # Documentation
```

## License

By contributing to AgentOptim, you agree that your contributions will be licensed under the project's MIT License.

Thank you for contributing to AgentOptim!
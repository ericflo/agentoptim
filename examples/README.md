# AgentOptim Examples

This directory contains example code demonstrating how to use AgentOptim v2.0's 2-tool architecture.

## Current Architecture Examples

The following examples demonstrate the recommended way to use AgentOptim:

- [usage_example.py](./usage_example.py) - Basic usage of the new API:
  - Creating an EvalSet
  - Running an evaluation on a conversation
  - Comparing two different responses

- [evalset_example.py](./evalset_example.py) - Comprehensive example with all API features:
  - Creating, getting, updating, and listing EvalSets
  - Running evaluations with different models
  - Comparing multiple responses
  - Advanced configuration options

- [support_response_evaluation.py](./support_response_evaluation.py) - Tutorial example:
  - Follows the tutorial in docs/TUTORIAL.md
  - Evaluates customer support response quality
  - Compares different response styles
  - Provides specific recommendations

## Deprecated Examples

For examples of the old 5-tool architecture (deprecated and scheduled to be removed in v2.1.0), see the [deprecated_examples](./deprecated_examples) directory.

## Getting Started

To run these examples:

1. Make sure you have AgentOptim installed:
   ```bash
   pip install agentoptim
   ```

2. Run an example:
   ```bash
   python usage_example.py
   ```

For more information, see the [AgentOptim documentation](../docs/TUTORIAL.md).
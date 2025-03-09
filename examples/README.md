# AgentOptim Examples

This directory contains comprehensive examples demonstrating AgentOptim's capabilities for evaluating, comparing, and improving AI conversations.

## Basic Examples

- [usage_example.py](./usage_example.py) - Simple introduction to AgentOptim:
  - Creating an EvalSet with basic criteria
  - Evaluating a single conversation
  - Interpreting evaluation results

- [evalset_example.py](./evalset_example.py) - Comprehensive API walkthrough:
  - Creating, retrieving, updating, and listing EvalSets
  - Managing evaluation criteria
  - Working with different API options

- [cli_usage_examples.md](./cli_usage_examples.md) - Command-line interface examples:
  - Managing evaluation sets through the CLI
  - Running evaluations from the command line
  - Using different output formats
  - Integrating with other tools

- [support_response_evaluation.py](./support_response_evaluation.py) - Customer support quality:
  - Evaluating different support response styles
  - Comparing detailed vs. brief responses
  - Identifying best practices for support conversations

## Advanced Use Cases

- [conversation_comparison.py](./conversation_comparison.py) - Compare conversation approaches:
  - Evaluate formal vs. casual vs. technical styles
  - Generate visualizations of style effectiveness
  - Identify optimal approaches for different contexts

- [prompt_testing.py](./prompt_testing.py) - Test system prompts:
  - Compare different system prompt strategies
  - Analyze strengths of different prompt designs
  - Identify optimal prompt patterns

- [multilingual_evaluation.py](./multilingual_evaluation.py) - Evaluate in multiple languages:
  - Test responses across different languages
  - Identify inconsistencies in multilingual support
  - Ensure quality across language boundaries

## Specialized Techniques

- [custom_template_example.py](./custom_template_example.py) - Create specialized templates:
  - Design custom evaluation formats (Likert scales, multi-criteria)
  - Implement domain-specific evaluations (code review, etc.)
  - Compare template effectiveness

- [batch_evaluation.py](./batch_evaluation.py) - Process multiple conversations:
  - Efficiently evaluate large sets of conversations
  - Generate comprehensive evaluation reports
  - Identify patterns across conversation datasets

- [automated_reporting.py](./automated_reporting.py) - Generate detailed reports:
  - Create visualization-rich evaluation reports
  - Export results in multiple formats (HTML, Markdown)
  - Design custom dashboards for quality monitoring

- [caching_performance_example.py](./caching_performance_example.py) - Optimize performance:
  - Demonstrate LRU caching benefits
  - Measure API request reductions
  - Track cache hit rates and time savings

## Advanced Evaluation Methods

- [conversation_benchmark.py](./conversation_benchmark.py) - Standardized benchmarking:
  - Create conversation quality benchmarks
  - Evaluate across standardized test cases
  - Track quality improvements over time

- [model_comparison.py](./model_comparison.py) - Compare judge models:
  - Evaluate using different LLM judge models
  - Analyze model biases and patterns
  - Select optimal judge models for different tasks

- [response_improvement.py](./response_improvement.py) - Iterative improvement:
  - Identify specific ways to improve responses
  - Implement targeted enhancements
  - Measure quality improvements

## Getting Started

To run these examples:

1. Install AgentOptim:
   ```bash
   pip install agentoptim
   ```

2. Start the AgentOptim server:
   ```bash
   agentoptim
   ```

3. Run any example:
   ```bash
   python examples/usage_example.py
   ```

Each example includes detailed comments and documentation to help you understand how to apply these techniques to your own projects.

## Additional Resources

- [Quickstart Guide](../docs/QUICKSTART.md) - Get started in under 5 minutes
- [Full Tutorial](../docs/TUTORIAL.md) - Step-by-step guide to AgentOptim
- [API Reference](../docs/API_REFERENCE.md) - Complete API documentation
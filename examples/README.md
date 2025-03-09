# AgentOptim v2.1.0 Examples

This directory contains comprehensive examples demonstrating AgentOptim v2.1.0's capabilities for evaluating, comparing, and improving AI conversations.

**Note about v2.1.0**: The examples in this directory have been fully updated for AgentOptim v2.1.0, which includes several important changes from previous versions:

1. Templates are now system-defined (the `template` parameter has been removed)
2. All EvalSets require `short_description` and `long_description` parameters
3. The compatibility layer has been removed
4. Error handling has been improved with more detailed error messages
5. Caching has been enhanced for better performance

See the [Updates for v2.1.0](#updates-for-v210) section for more details.

## Basic Examples

- [usage_example.py](./usage_example.py) - Simple introduction to AgentOptim:
  - Using the `manage_evalset_tool` to list and create EvalSets
  - Running a quick evaluation on a conversation
  - Viewing and interpreting evaluation results
  - Complete end-to-end example with minimal code

- [evalset_example.py](./evalset_example.py) - Comprehensive API walkthrough:
  - Creating, retrieving, updating, and listing EvalSets
  - Managing evaluation criteria for different use cases
  - Working with the v2.0+ API structure
  - Handling different response formats

- [cli_usage_examples.md](./cli_usage_examples.md) - Command-line interface examples:
  - Managing evaluation sets through the CLI
  - Running evaluations from the command line
  - Using different output formats
  - Integrating with other tools

- [support_response_evaluation.py](./support_response_evaluation.py) - Customer support quality:
  - Creating specialized criteria for support response evaluation
  - Comparing detailed, brief, and vague response styles
  - Analyzing results to identify best practices
  - Generating recommendations for support conversations

## Advanced Use Cases

- [conversation_comparison.py](./conversation_comparison.py) - Compare conversation approaches:
  - Evaluate formal vs. casual vs. technical vs. empathetic styles
  - Compare effectiveness across multiple evaluation criteria
  - Generate detailed reports and recommendations
  - Identify optimal approaches for different user needs

- [prompt_testing.py](./prompt_testing.py) - Test system prompts:
  - Compare five different system prompt strategies:
    - Basic professional prompt
    - Detailed role instruction prompt  
    - Persona-based prompt
    - Principles-based prompt
    - Contextual constraint prompt
  - Analyze strengths of each prompt design
  - Generate detailed criterion-based comparisons
  - Produce optimized prompt recommendations

- [multilingual_evaluation.py](./multilingual_evaluation.py) - Evaluate in multiple languages:
  - Test responses in 5 languages (English, Spanish, French, German, Japanese)
  - Conduct detailed analysis across 8 language-specific criteria
  - Generate consistency comparison across all languages
  - Visualize results with language performance metrics
  - Provide specific recommendations for multilingual improvement

## Specialized Techniques

- [custom_template_example.py](./custom_template_example.py) - Create specialized evaluation criteria:
  - Design question sets for different evaluation scenarios:
    - Basic yes/no questions
    - Likert-scale statements
    - Multi-criteria technical evaluations
    - Domain-specific code review questions
  - Compare effectiveness of different question approaches
  - Guidelines for designing effective evaluation criteria in v2.1.0

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

## Updates for v2.1.0

The examples in this directory have been updated for AgentOptim v2.1.0, with the following changes:

1. Removed dependencies on the compat layer, which has been removed in v2.1.0
2. Updated API calls to use the current parameter structure:
   - Added required `short_description` and `long_description` parameters
   - Removed `template` parameter as templates are now system-defined
   - Improved error handling and response parsing
3. Simplified some examples to run more quickly
4. Added robust error handling for all API interactions
5. Updated examples to handle the EvalSet deduplication feature
6. Enhanced result parsing to support different response formats
7. Added verification steps to ensure EvalSets exist before using them

For the full evaluation experience, consider running the examples with a local LLM server that supports the OpenAI API format. All examples have been tested with both cloud-based and local LLM models.

## Additional Resources

- [Full Tutorial](../docs/TUTORIAL.md) - Step-by-step guide to AgentOptim
- [API Reference](../docs/API_REFERENCE.md) - Complete API documentation
- [Migration Guide](../docs/MIGRATION_GUIDE.md) - Guide for migrating from v1.x to v2.x
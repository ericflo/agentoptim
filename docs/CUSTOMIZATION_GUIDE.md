# AgentOptim v2.2.0 Customization Guide

This guide covers advanced customization options for AgentOptim v2.2.0, with a focus on system message optimization. Learn how to extend, customize, and tailor AgentOptim to your specific requirements.

## Table of Contents

- [System Message Optimization Customization](#system-message-optimization-customization)
- [EvalSet Customization](#evalset-customization)
- [Model and Provider Customization](#model-and-provider-customization)
- [Output Format Customization](#output-format-customization)
- [Advanced Configuration](#advanced-configuration)
- [CLI Customization](#cli-customization)

## System Message Optimization Customization

### Custom Templates

AgentOptim system message optimization uses templates to guide the generation of system messages. You can customize these templates for specific domains or use cases.

#### Creating a Domain-Specific Template

```python
import asyncio
from agentoptim import optimize_system_messages_tool

async def create_customer_support_template():
    """Create a custom template for customer support system messages."""
    
    # Define a custom template structure
    domain_template = """
    You are a customer support assistant specialized in {specific_area}.
    
    Your goal is to help users with {specific_tasks} while maintaining these priorities:
    1. {priority_1}
    2. {priority_2}
    3. {priority_3}
    
    When responding to users:
    - {response_style_1}
    - {response_style_2}
    - {response_style_3}
    
    Important constraints:
    - {constraint_1}
    - {constraint_2}
    """
    
    # Use the template in the optimization process
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id="your-evalset-id",
        user_query="How do I reset my password?",
        template=domain_template,
        template_variables={
            "specific_area": "account security and access management",
            "specific_tasks": "password resets, account recovery, and security setup",
            "priority_1": "Ensure user security and privacy at all times",
            "priority_2": "Provide clear, actionable steps",
            "priority_3": "Balance security with user convenience",
            "response_style_1": "Use concise, step-by-step instructions",
            "response_style_2": "Include security best practices when relevant",
            "response_style_3": "Confirm when users should receive system emails",
            "constraint_1": "Never ask for the user's current password",
            "constraint_2": "Remind users to update passwords on other services if they reuse passwords"
        }
    )
    
    return result

if __name__ == "__main__":
    asyncio.run(create_customer_support_template())
```

#### Using Meta-Templates

For even more customization, you can create meta-templates that guide the system message generator:

```python
meta_template = """
Generate system messages that conform to these structural requirements:
1. Start with a clear role definition
2. Include specific knowledge areas: {knowledge_areas}
3. Provide guidance on response format: {response_format}
4. Set tone guidelines: {tone}
5. Establish boundaries: {boundaries}
6. End with essential reminders: {reminders}
"""

result = await optimize_system_messages_tool(
    # ... other parameters
    meta_template=meta_template,
    meta_template_variables={
        "knowledge_areas": "APIs, database design, coding standards",
        "response_format": "code snippets with explanations",
        "tone": "professional but approachable",
        "boundaries": "avoid implementing full applications",
        "reminders": "always explain the 'why' behind coding decisions"
    }
)
```

### Custom Evaluation Criteria

You can create custom evaluation criteria tailored to specific system message needs:

```python
# First, create a specialized EvalSet for your domain
evalset_result = await manage_evalset_tool(
    action="create",
    name="Security-Focused System Message Evaluation",
    questions=[
        "Does the system message establish clear security boundaries?",
        "Does it guide the assistant to verify user identity when appropriate?",
        "Does it include guidance on handling sensitive information?",
        "Does it instruct the assistant to recommend secure practices?",
        "Does it avoid suggesting insecure workarounds?",
        "Does it provide a framework for asking security clarification questions?",
        "Does it balance security with usability?",
        "Does it avoid unnecessary technical jargon?",
        "Does it establish an appropriate tone for security-related communications?",
        "Does it include direction on when to escalate security concerns?"
    ],
    short_description="Evaluates system messages specifically for security-focused applications",
    long_description="This EvalSet is designed to assess system messages for applications where security is a primary concern. It evaluates whether the system message provides appropriate security guidance while maintaining usability."
)

# Then use this specialized EvalSet for optimization
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id=evalset_result["evalset"]["id"],
    user_query="How can I securely share my account information?",
    # ... other parameters
)
```

### Weighted Evaluation

You can customize the importance of different evaluation criteria by applying weights:

```python
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    criterion_weights={
        "Does the system message establish clear security boundaries?": 2.0,
        "Does it include guidance on handling sensitive information?": 1.5,
        "Does it avoid suggesting insecure workarounds?": 1.8,
        # Default weight for unspecified criteria is 1.0
    }
)
```

## EvalSet Customization

### Custom Evaluation Templates

You can customize the evaluation template used to assess system messages:

```python
evalset_result = await manage_evalset_tool(
    action="create",
    name="Custom Template Evaluation",
    questions=[
        "Does the response meet criterion A?",
        "Does the response meet criterion B?"
    ],
    template="""
    You are an expert evaluator assessing system messages for AI assistants.
    
    You will be given:
    1. A user query: "{{ user_query }}"
    2. A system message to evaluate: "{{ system_message }}"
    
    Your task is to assess whether the system message would be effective at guiding an AI assistant to respond appropriately to the user query.
    
    For each criterion, provide a YES or NO judgment, along with a confidence score between 0 and 1.
    
    Criteria to evaluate:
    {% for question in questions %}
    {{ loop.index }}. {{ question }}
    {% endfor %}
    
    Format your response as follows:
    
    Criterion 1: YES/NO
    Confidence: [0-1]
    Reasoning: Your detailed reasoning here.
    
    Criterion 2: YES/NO
    Confidence: [0-1]
    Reasoning: Your detailed reasoning here.
    
    ... and so on for each criterion.
    """
)
```

### Custom Scoring Functions

You can implement custom scoring functions for evaluation results:

```python
# Custom scoring function
def custom_scoring(evaluation_results):
    """Custom scoring that weights 'safety' criteria more heavily."""
    total_score = 0
    total_weight = 0
    
    # Define weights for different question patterns
    weights = {
        "safety": 2.0,
        "security": 1.8,
        "harmful": 1.5,
        "default": 1.0
    }
    
    for result in evaluation_results:
        question = result["question"].lower()
        judgment = 1 if result["judgment"] == "YES" else 0
        
        # Determine weight based on question content
        weight = weights["default"]
        for key, value in weights.items():
            if key in question:
                weight = value
                break
        
        total_score += judgment * weight
        total_weight += weight
    
    # Calculate weighted percentage
    weighted_score = (total_score / total_weight) * 100 if total_weight > 0 else 0
    return weighted_score

# Use the custom scoring with the optimization tool
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    scoring_function=custom_scoring
)
```

## Model and Provider Customization

### Using Different Judge Models

You can customize which models are used for evaluation:

```python
# Using OpenAI GPT-4 as the judge
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    judge_model="gpt-4",
    judge_provider="openai"
)

# Using Claude Opus as the judge
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    judge_model="claude-3-opus-20240229",
    judge_provider="anthropic"
)

# Using a local model via LM Studio
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    judge_model="meta-llama-3.1-70b-instruct",
    judge_provider="local",
    api_base_url="http://localhost:1234/v1"
)
```

### Custom Generation Models

You can specify different models for generating system message candidates:

```python
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    generator_model="claude-3-opus-20240229",  # More powerful model for generation
    judge_model="meta-llama-3.1-8b-instruct"    # Faster model for evaluation
)
```

### Advanced API Configuration

For fine-grained control over API calls:

```python
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id="your-evalset-id",
    user_query="How do I reset my password?",
    generator_model="gpt-4",
    generator_model_config={
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 2000
    },
    judge_model="gpt-4",
    judge_model_config={
        "temperature": 0.1,  # Low temperature for more consistent evaluation
        "top_p": 0.99,
        "max_tokens": 1000
    }
)
```

## Output Format Customization

### Custom Result Formatting

You can customize how optimization results are formatted:

```python
def custom_formatter(result):
    """Custom formatter for optimization results."""
    output = "# System Message Optimization Results\n\n"
    
    # Add a summary section
    output += "## Summary\n"
    output += f"Query: {result['user_message']}\n"
    output += f"Candidates generated: {len(result['candidates'])}\n"
    output += f"Best score: {result['best_score']:.2f}%\n\n"
    
    # Add the best system message
    output += "## Best System Message\n"
    output += f"```\n{result['best_system_message']}\n```\n\n"
    
    # Add performance analysis
    output += "## Performance Analysis\n"
    output += "| Criterion | Score |\n"
    output += "|-----------|-------|\n"
    
    for criterion, score in result['candidates'][0]['criterion_scores'].items():
        output += f"| {criterion} | {score:.2f}% |\n"
    
    return output

# Use with the CLI or API
formatted_result = custom_formatter(optimization_result)
print(formatted_result)

# Save to a file
with open("optimization_report.md", "w") as f:
    f.write(formatted_result)
```

### Integration with External Reporting Tools

```python
async def export_to_reporting_system(optimization_result, system):
    """Export optimization results to an external reporting system."""
    if system == "slack":
        # Format for Slack message blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "System Message Optimization Results"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Query:* {optimization_result['user_message']}\n*Score:* {optimization_result['best_score']:.2f}%"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Best System Message:*\n```{optimization_result['best_system_message']}```"
                }
            }
        ]
        
        # Send to Slack (implementation would depend on your Slack integration)
        # slack_client.post_message(channel="ai-optimizations", blocks=blocks)
        
    elif system == "html_dashboard":
        # Generate HTML for a dashboard
        html = f"""
        <div class="optimization-card">
            <h3>Optimization Result</h3>
            <p><strong>Query:</strong> {optimization_result['user_message']}</p>
            <p><strong>Score:</strong> {optimization_result['best_score']:.2f}%</p>
            <div class="system-message">
                <pre>{optimization_result['best_system_message']}</pre>
            </div>
        </div>
        """
        
        # Save or update dashboard
        # with open("dashboard.html", "a") as f:
        #     f.write(html)
```

## Advanced Configuration

### Environment Variables for Global Configuration

You can use environment variables to configure AgentOptim globally:

```bash
# Set judge model and other parameters
export AGENTOPTIM_JUDGE_MODEL="claude-3-sonnet-20240229"
export AGENTOPTIM_GENERATOR_MODEL="claude-3-opus-20240229"
export AGENTOPTIM_DEFAULT_DIVERSITY="high"
export AGENTOPTIM_API_KEY="your-api-key"
export AGENTOPTIM_MAX_PARALLEL=6
export AGENTOPTIM_TIMEOUT=120000
```

### Configuration Files

You can create a configuration file for persistent settings:

```python
# config.py
OPTIMIZATION_CONFIG = {
    "generator": {
        "model": "claude-3-opus-20240229",
        "temperature": 0.7,
        "default_num_candidates": 5,
        "default_diversity": "medium",
        "timeout": 60000
    },
    "judge": {
        "model": "meta-llama-3.1-8b-instruct",
        "temperature": 0.1,
        "timeout": 30000
    },
    "templates": {
        "customer_support": "path/to/customer_support_template.jinja",
        "coding_assistant": "path/to/coding_assistant_template.jinja",
        "general": "path/to/general_template.jinja"
    },
    "evalsets": {
        "general": "general-evalset-id",
        "security": "security-evalset-id",
        "technical": "technical-evalset-id"
    }
}
```

Then import and use this config:

```python
from config import OPTIMIZATION_CONFIG

# Use configuration in optimization
result = await optimize_system_messages_tool(
    action="optimize",
    evalset_id=OPTIMIZATION_CONFIG["evalsets"]["security"],
    user_query="How do I reset my password?",
    generator_model=OPTIMIZATION_CONFIG["generator"]["model"],
    judge_model=OPTIMIZATION_CONFIG["judge"]["model"],
    num_candidates=OPTIMIZATION_CONFIG["generator"]["default_num_candidates"]
)
```

## CLI Customization

### Creating Custom Commands

You can extend the AgentOptim CLI with custom commands:

```python
# custom_commands.py
import click
import asyncio
from agentoptim import optimize_system_messages_tool

@click.command()
@click.argument("evalset_id")
@click.argument("query_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file for results")
@click.option("--candidates", "-c", default=3, help="Number of candidates to generate")
def batch_optimize(evalset_id, query_file, output, candidates):
    """Run optimization for multiple queries from a file."""
    # Read queries from file
    with open(query_file, "r") as f:
        queries = [line.strip() for line in f if line.strip()]
    
    click.echo(f"Running optimization for {len(queries)} queries...")
    
    # Run optimizations
    results = []
    for query in queries:
        click.echo(f"Optimizing for: {query}")
        result = asyncio.run(optimize_system_messages_tool(
            action="optimize",
            evalset_id=evalset_id,
            user_query=query,
            num_candidates=candidates
        ))
        results.append(result)
    
    # Save results
    if output:
        with open(output, "w") as f:
            for i, result in enumerate(results):
                f.write(f"Query {i+1}: {result['user_message']}\n")
                f.write(f"Best system message (Score: {result['best_score']:.2f}%):\n")
                f.write(f"{result['best_system_message']}\n\n")
        
        click.echo(f"Results saved to {output}")
    
    return results

if __name__ == "__main__":
    batch_optimize()
```

### Integration with Existing CLI Workflows

```bash
#!/bin/bash
# optimize_and_test.sh

# Run optimization
RESULT=$(agentoptim optimize create "$EVALSET_ID" "$USER_QUERY" --format json -q)

# Extract the best system message
SYSTEM_MESSAGE=$(echo $RESULT | jq -r '.best_system_message')

# Create a test conversation with the optimized system message
cat > test_conversation.json << EOL
[
  {"role": "system", "content": "$SYSTEM_MESSAGE"},
  {"role": "user", "content": "$USER_QUERY"}
]
EOL

# Test the optimized system message
agentoptim run create "$EVALSET_ID" test_conversation.json

# Clean up
rm test_conversation.json
```

## Conclusion

By leveraging these customization options, you can tailor AgentOptim's system message optimization capabilities to your specific requirements. From custom templates and evaluation criteria to specialized models and reporting, these techniques allow you to build a fully customized system message optimization workflow.

For more details on implementation, refer to:
- [ADVANCED_TUTORIAL.md](ADVANCED_TUTORIAL.md) - Advanced examples and techniques
- [API_REFERENCE.md](API_REFERENCE.md) - Detailed API documentation
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Best practices for effective optimization
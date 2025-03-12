# AgentOptim v2.2.0 Advanced Tutorial: System Message Optimization

This advanced tutorial focuses on using AgentOptim's new system message optimization feature introduced in v2.2.0. You'll learn how to generate, evaluate, and improve system messages for optimal performance.

## Prerequisites

Before starting, ensure you have:

1. Python 3.8+ installed
2. AgentOptim v2.2.0 installed: `pip install agentoptim>=2.2.0`
3. Access to a judge model (e.g., meta-llama-3.1-8b-instruct)
4. Familiarity with basic AgentOptim concepts (see [TUTORIAL.md](TUTORIAL.md))

## Introduction to System Message Optimization

System messages play a crucial role in guiding AI behavior, but writing effective ones can be challenging. AgentOptim v2.2.0 introduces a powerful tool to help: `optimize_system_messages_tool`.

**Key capabilities include:**

- Generating diverse candidate system messages tailored to specific queries
- Evaluating candidate performance using existing EvalSets
- Scoring and ranking candidates with detailed performance breakdowns
- Self-optimization for continuous improvement
- Persistence for storing and retrieving optimization runs

## Optimization Workflow Overview

The system message optimization workflow consists of these steps:

1. **Define evaluation criteria** - Create or select an EvalSet for testing
2. **Specify a user query** - The system message will be optimized for this query
3. **Generate candidates** - The tool creates multiple system message candidates
4. **Evaluate performance** - Each candidate is tested against the query
5. **Rank candidates** - Results are sorted by performance scores
6. **Select the winner** - The best-performing system message is identified

Let's go through each step in detail.

## Step 1: Creating an EvalSet for System Message Evaluation

First, you need an EvalSet with criteria specifically designed to evaluate system message quality. While you can use any existing EvalSet, it's better to create one tailored for this purpose:

```python
import asyncio
from agentoptim import manage_evalset_tool, optimize_system_messages_tool

async def main():
    # Create an EvalSet specifically for evaluating system messages
    evalset_result = await manage_evalset_tool(
        action="create",
        name="System Message Quality Assessment",
        questions=[
            "Does the system message provide clear guidance on the assistant's role?",
            "Does the system message include relevant constraints or boundaries?",
            "Is the system message specific enough for the user's needs?",
            "Does the system message avoid unnecessary verbosity?",
            "Is the system message free from biased or prescriptive language?",
            "Would the system message help produce responses tailored to this query?",
            "Does the system message include helpful context?",
            "Does the system message specify an appropriate tone for responses?",
            "Is the system message concise while still being complete?",
            "Would the system message likely prevent harmful outputs?",
        ],
        short_description="Evaluates system message quality and effectiveness",
        long_description="This EvalSet assesses whether a system message effectively guides the model to provide helpful, safe, and appropriate responses to the given user query. It evaluates clarity, specificity, tone guidance, and other factors that contribute to high-quality system messages."
    )
    
    evalset_id = evalset_result["evalset"]["id"]
    print(f"Created EvalSet with ID: {evalset_id}")
    
    # Continue with the rest of the tutorial...

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 2: Basic System Message Optimization

Now that we have our EvalSet, let's use it to optimize a system message for a specific user query:

```python
async def optimize_basic(evalset_id, user_query="How do I reset my password?"):
    """Basic system message optimization."""
    print(f"\nOptimizing system message for query: '{user_query}'")
    
    # Run the optimization process
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query=user_query,
        num_candidates=3,  # Generate 3 candidate system messages
        diversity_level="medium"  # Control diversity of candidates
    )
    
    # Display results
    print("\nOptimization complete!")
    print(f"Optimization run ID: {result['id']}")
    
    # Get the top-ranked system message
    best_system_message = result["candidates"][0]["system_message"]
    best_score = result["candidates"][0]["score"]
    
    print(f"\nBest System Message (Score: {best_score:.1f}%):")
    print("-" * 80)
    print(best_system_message)
    print("-" * 80)
    
    return result["id"]
```

Add this function to your script and call it from your main function:

```python
async def main():
    # Create EvalSet (as shown earlier)
    # ...
    
    # Optimize a system message
    optimization_id = await optimize_basic(evalset_id)
    
    # Continue with more examples...
```

## Step 3: Customizing the Optimization Process

For greater control, you can customize various aspects of the optimization process:

```python
async def optimize_customized(evalset_id):
    """Demonstrates customization options for system message optimization."""
    print("\nRunning customized optimization...")
    
    # Define a starting point (base system message)
    base_system_message = "You are a helpful customer support assistant."
    
    # Additional instructions to guide the generation
    additional_instructions = """
    Focus on creating system messages that:
    - Emphasize security best practices
    - Maintain a friendly but professional tone
    - Include guidance on handling sensitive information
    - Keep responses concise and step-based
    """
    
    # Run the customized optimization
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query="How do I secure my account with two-factor authentication?",
        base_system_message=base_system_message,
        num_candidates=5,  # Generate more candidates
        diversity_level="high",  # Increase diversity
        additional_instructions=additional_instructions,
        max_parallel=4  # Process 4 candidates in parallel
    )
    
    # Display results summary
    print(f"\nGenerated {len(result['candidates'])} diverse candidates")
    
    # Show performance breakdown for the winner
    winner = result["candidates"][0]
    print(f"\nWinning system message performance breakdown:")
    
    for criterion, score in winner["criterion_scores"].items():
        print(f"- {criterion}: {score:.1f}%")
    
    return result["id"]
```

## Step 4: Retrieving and Comparing Results

After running optimizations, you can retrieve previous results and compare them:

```python
async def retrieve_and_compare(optimization_id):
    """Retrieve and analyze previous optimization runs."""
    print("\nRetrieving previous optimization run...")
    
    # Get a specific optimization run by ID
    result = await optimize_system_messages_tool(
        action="get",
        optimization_run_id=optimization_id
    )
    
    optimization_run = result["optimization_run"]
    
    # Analyze candidate diversity
    print("\nComparing system message candidates:")
    print("-" * 80)
    
    for i, candidate in enumerate(optimization_run["candidates"][:3]):  # Top 3
        print(f"Candidate {i+1} (Score: {candidate['score']:.1f}%):")
        print(f"Length: {len(candidate['content'])} characters")
        # Print just the first 100 characters as a preview
        print(f"Preview: {candidate['content'][:100]}...")
        print()
    
    # List all optimization runs
    print("\nListing recent optimization runs:")
    runs = await optimize_system_messages_tool(
        action="list",
        page=1,
        page_size=5
    )
    
    for run in runs["optimization_runs"]:
        print(f"- ID: {run['id']}")
        print(f"  Query: '{run['user_message']}'")
        print(f"  Best score: {run['best_score']:.1f}%")
        print()
```

## Step 5: Self-Optimization and Advanced Features

AgentOptim can also self-optimize its system message generator to improve performance over time:

```python
async def trigger_self_optimization(evalset_id):
    """Trigger self-optimization of the system message generator."""
    print("\nTriggering self-optimization of the generator...")
    
    # Run optimization with self-optimization enabled
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query="How do I install Python on Windows?",
        num_candidates=3,
        self_optimize=True  # Enable self-optimization
    )
    
    # Check self-optimization results
    if "self_optimization" in result and result["self_optimization"]:
        opt_result = result["self_optimization"]
        print(f"\nSelf-optimization completed!")
        print(f"Generator version: {opt_result['old_version']} → {opt_result['new_version']}")
        print(f"Success rate: {opt_result['success_rate']:.2f}")
    else:
        print("\nSelf-optimization was not triggered or did not complete.")
    
    return result["id"]
```

## Complete Example Script

Here's the complete script combining all the examples:

```python
# advanced_system_message_optimization.py

import asyncio
from agentoptim import manage_evalset_tool, optimize_system_messages_tool

async def main():
    # Step 1: Create an EvalSet for system message evaluation
    evalset_result = await manage_evalset_tool(
        action="create",
        name="System Message Quality Assessment",
        questions=[
            "Does the system message provide clear guidance on the assistant's role?",
            "Does the system message include relevant constraints or boundaries?",
            "Is the system message specific enough for the user's needs?",
            "Does the system message avoid unnecessary verbosity?",
            "Is the system message free from biased or prescriptive language?",
            "Would the system message help produce responses tailored to this query?",
            "Does the system message include helpful context?",
            "Does the system message specify an appropriate tone for responses?",
            "Is the system message concise while still being complete?",
            "Would the system message likely prevent harmful outputs?",
        ],
        short_description="Evaluates system message quality and effectiveness",
        long_description="This EvalSet assesses whether a system message effectively guides the model to provide helpful, safe, and appropriate responses to the given user query. It evaluates clarity, specificity, tone guidance, and other factors that contribute to high-quality system messages."
    )
    
    evalset_id = evalset_result["evalset"]["id"]
    print(f"Created EvalSet with ID: {evalset_id}")
    
    # Step 2: Basic system message optimization
    basic_id = await optimize_basic(evalset_id)
    
    # Step 3: Customized optimization
    custom_id = await optimize_customized(evalset_id)
    
    # Step 4: Retrieve and compare results
    await retrieve_and_compare(custom_id)
    
    # Step 5: Self-optimization (advanced)
    await trigger_self_optimization(evalset_id)
    
    print("\nAdvanced tutorial completed!")

async def optimize_basic(evalset_id, user_query="How do I reset my password?"):
    """Basic system message optimization."""
    print(f"\nOptimizing system message for query: '{user_query}'")
    
    # Run the optimization process
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query=user_query,
        num_candidates=3,
        diversity_level="medium"
    )
    
    # Display results
    print("\nOptimization complete!")
    print(f"Optimization run ID: {result['id']}")
    
    # Get the top-ranked system message
    best_system_message = result["candidates"][0]["system_message"]
    best_score = result["candidates"][0]["score"]
    
    print(f"\nBest System Message (Score: {best_score:.1f}%):")
    print("-" * 80)
    print(best_system_message)
    print("-" * 80)
    
    return result["id"]

async def optimize_customized(evalset_id):
    """Demonstrates customization options for system message optimization."""
    print("\nRunning customized optimization...")
    
    # Define a starting point (base system message)
    base_system_message = "You are a helpful customer support assistant."
    
    # Additional instructions to guide the generation
    additional_instructions = """
    Focus on creating system messages that:
    - Emphasize security best practices
    - Maintain a friendly but professional tone
    - Include guidance on handling sensitive information
    - Keep responses concise and step-based
    """
    
    # Run the customized optimization
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query="How do I secure my account with two-factor authentication?",
        base_system_message=base_system_message,
        num_candidates=5,
        diversity_level="high",
        additional_instructions=additional_instructions,
        max_parallel=4
    )
    
    # Display results summary
    print(f"\nGenerated {len(result['candidates'])} diverse candidates")
    
    # Show performance breakdown for the winner
    winner = result["candidates"][0]
    print(f"\nWinning system message performance breakdown:")
    
    for criterion, score in winner["criterion_scores"].items():
        print(f"- {criterion}: {score:.1f}%")
    
    return result["id"]

async def retrieve_and_compare(optimization_id):
    """Retrieve and analyze previous optimization runs."""
    print("\nRetrieving previous optimization run...")
    
    # Get a specific optimization run by ID
    result = await optimize_system_messages_tool(
        action="get",
        optimization_run_id=optimization_id
    )
    
    optimization_run = result["optimization_run"]
    
    # Analyze candidate diversity
    print("\nComparing system message candidates:")
    print("-" * 80)
    
    for i, candidate in enumerate(optimization_run["candidates"][:3]):
        print(f"Candidate {i+1} (Score: {candidate['score']:.1f}%):")
        print(f"Length: {len(candidate['content'])} characters")
        print(f"Preview: {candidate['content'][:100]}...")
        print()
    
    # List all optimization runs
    print("\nListing recent optimization runs:")
    runs = await optimize_system_messages_tool(
        action="list",
        page=1,
        page_size=5
    )
    
    for run in runs["optimization_runs"]:
        print(f"- ID: {run['id']}")
        print(f"  Query: '{run['user_message']}'")
        print(f"  Best score: {run['best_score']:.1f}%")
        print()

async def trigger_self_optimization(evalset_id):
    """Trigger self-optimization of the system message generator."""
    print("\nTriggering self-optimization of the generator...")
    
    # Run optimization with self-optimization enabled
    result = await optimize_system_messages_tool(
        action="optimize",
        evalset_id=evalset_id,
        user_query="How do I install Python on Windows?",
        num_candidates=3,
        self_optimize=True
    )
    
    # Check self-optimization results
    if "self_optimization" in result and result["self_optimization"]:
        opt_result = result["self_optimization"]
        print(f"\nSelf-optimization completed!")
        print(f"Generator version: {opt_result['old_version']} → {opt_result['new_version']}")
        print(f"Success rate: {opt_result['success_rate']:.2f}")
    else:
        print("\nSelf-optimization was not triggered or did not complete.")
    
    return result["id"]

if __name__ == "__main__":
    asyncio.run(main())
```

## CLI Usage for System Message Optimization

In addition to the Python API, you can use the command-line interface:

```bash
# Create an EvalSet for system message evaluation
agentoptim evalset create --wizard

# Basic optimization
agentoptim optimize create <evalset-id> "How do I reset my password?"

# Customized optimization
agentoptim optimize create <evalset-id> "How do I secure my account?" \
  --base "You are a helpful customer support assistant." \
  --diversity high \
  --num-candidates 5 \
  --instructions "Focus on security best practices"

# Export results to HTML
agentoptim optimize get <optimization-id> --format html --output results.html

# List optimization runs
agentoptim optimize list

# Trigger self-optimization
agentoptim optimize meta <evalset-id>
```

## Advanced Usage Tips

1. **Combine with A/B testing**: Use `run create` to test responses with different system messages
2. **Domain specialization**: Create domain-specific EvalSets for different use cases
3. **Iterative refinement**: Use the best system message as a base for further optimization
4. **Benchmark different models**: Compare system message performance across different models
5. **Export for team collaboration**: Use the HTML export to share optimization results
6. **Automation**: Create scheduled tasks to periodically optimize system messages

## Conclusion

The system message optimization tool offers a powerful way to improve the effectiveness of your AI applications. By systematically generating, evaluating, and refining system messages, you can achieve better performance and more consistent results.

For more examples and use cases, check out the [system_message_optimization.py](../examples/system_message_optimization.py) example in the examples directory.

---

Next steps:
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Learn best practices for system message optimization
- [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) - Discover advanced customization options
- [API_REFERENCE.md](API_REFERENCE.md) - Reference documentation for all APIs
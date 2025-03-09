# AgentOptim v2.1.0 Tutorial

This tutorial walks you through using AgentOptim v2.1.0 to evaluate the quality of conversations with a language model.

## Prerequisites

Before starting, ensure you have:

1. Python 3.8+ installed
2. AgentOptim installed: `pip install agentoptim`
3. Access to a judge model (e.g., meta-llama-3.1-8b-instruct)

## Introduction to AgentOptim

AgentOptim simplifies conversation evaluation and optimization with two interfaces:

1. **Python API** - For programmatic access and integration
   - `manage_evalset_tool` - Create and manage sets of evaluation criteria
   - `manage_eval_runs_tool` - Run evaluations on conversations using an EvalSet

2. **Command-Line Interface (CLI)** - For quick access and direct usage
   - `agentoptim evalset` commands - Manage evaluation sets
   - `agentoptim run` commands - Manage evaluation runs
   - Plus developer tools via `agentoptim dev`

This tutorial covers both approaches, so you can choose the one that best fits your workflow.

## Step 1: Setting Up Your Project

Let's create a simple script that uses AgentOptim to evaluate different support responses:

```python
# support_response_evaluation.py

import asyncio
from agentoptim import manage_evalset_tool, manage_eval_runs_tool

async def main():
    print("AgentOptim Support Response Evaluation")
    print("=" * 50)
    
    # We'll implement the evaluation process step by step
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run this script to ensure everything is set up correctly:

```bash
python support_response_evaluation.py
```

## Step 2: Creating an EvalSet

You can create an EvalSet using either the Python API or the CLI.

### Option A: Using the Python API

```python
# Add this to the main() function

# Step 1: Create an EvalSet with evaluation criteria
print("\n1. Creating Support Response Quality EvalSet...")

evalset_result = await manage_evalset_tool(
    action="create",
    name="Support Response Quality",
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Does the response provide accurate information?",
        "Does the response provide complete information?",
        "Is the tone of the response appropriate and professional?",
        "Does the response avoid unnecessary information?"
    ],
    short_description="Support quality evaluation",
    long_description="Evaluation criteria for measuring customer support response quality across helpfulness, clarity, accuracy, completeness, and professionalism dimensions"
)

# Extract the EvalSet ID
evalset_id = evalset_result.get("evalset", {}).get("id")
print(f"EvalSet created with ID: {evalset_id}")
print(f"EvalSet contains {len(evalset_result.get('evalset', {}).get('questions', []))} evaluation questions")
```

### Option B: Using the CLI

Alternatively, you can create the same EvalSet from the command line. First, create a file named `support_questions.txt` with each question on a separate line:

```
Is the response helpful for the user's needs?
Does the response directly address the user's question?
Is the response clear and easy to understand?
Does the response provide accurate information?
Does the response provide complete information?
Is the tone of the response appropriate and professional?
Does the response avoid unnecessary information?
```

Then create the EvalSet with the CLI:

```bash
agentoptim evalset create \
  --name "Support Response Quality" \
  --questions support_questions.txt \
  --short-desc "Support quality evaluation" \
  --long-desc "Evaluation criteria for measuring customer support response quality across helpfulness, clarity, accuracy, completeness, and professionalism dimensions"
```

The command will display the generated ID and other details about your new EvalSet.

You can verify it was created successfully by listing all EvalSets:

```bash
agentoptim evalset list
```

## Step 3: Define Conversations to Evaluate

Let's define several conversation examples with different response approaches:

```python
# Add after the EvalSet creation

# Step 2: Define conversations to evaluate
print("\n2. Defining conversations to evaluate...")

# Good response - complete, clear, and helpful
good_conversation = [
    {"role": "system", "content": "You are a helpful customer support assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to our login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder. If you still need help, let me know and I'd be happy to assist further."}
]

# Average response - addresses the question but lacks detail
average_conversation = [
    {"role": "system", "content": "You are a helpful customer support assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "To reset your password, go to the login page and click on the 'Forgot Password' link. You'll receive an email with instructions to create a new password."}
]

# Poor response - unhelpful and lacks specific information
poor_conversation = [
    {"role": "system", "content": "You are a helpful customer support assistant."},
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "You should be able to find password reset options in your account settings or on the login page."}
]

print("Defined 3 conversations with different response quality levels")
```

## Step 3: Evaluate the Conversations

You can evaluate conversations using either the Python API or the CLI.

### Option A: Using the Python API

```python
# Add this to the main() function
print("\n3. Running evaluations...")

# Evaluate the good response
print("\nEvaluating good response...")
good_results = await manage_eval_runs_tool(
    action="run",
    evalset_id=evalset_id,
    conversation=good_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# Evaluate the average response
print("\nEvaluating average response...")
average_results = await manage_eval_runs_tool(
    action="run",
    evalset_id=evalset_id,
    conversation=average_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# Evaluate the poor response
print("\nEvaluating poor response...")
poor_results = await manage_eval_runs_tool(
    action="run",
    evalset_id=evalset_id,
    conversation=poor_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

print("\nAll evaluations completed!")
```

### Option B: Using the CLI

First, save each conversation to a JSON file. For example, `good_conversation.json`:

```json
[
  {"role": "system", "content": "You are a helpful customer support assistant."},
  {"role": "user", "content": "How do I reset my password?"},
  {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to our login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder. If you still need help, let me know and I'd be happy to assist further."}
]
```

Repeat this for `average_conversation.json` and `poor_conversation.json` with their respective contents.

Then evaluate each with the CLI:

```bash
# Assuming your EvalSet ID is 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e

# Evaluate good response
agentoptim run create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e good_conversation.json \
  --model "meta-llama-3.1-8b-instruct" \
  --concurrency 3 \
  --output good_results.json \
  --format json

# Evaluate average response
agentoptim run create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e average_conversation.json \
  --model "meta-llama-3.1-8b-instruct" \
  --concurrency 3 \
  --output average_results.json \
  --format json

# Evaluate poor response
agentoptim run create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e poor_conversation.json \
  --model "meta-llama-3.1-8b-instruct" \
  --concurrency 3 \
  --output poor_results.json \
  --format json
```

The results will be saved to the respective output files for analysis.

## Step 4: Analyze and Compare Results

### Option A: Using the Python API

```python
# Add this to the main() function
print("\n4. Analyzing results...")

def print_results(name, results):
    summary = results.get("summary", {})
    yes_percentage = summary.get("yes_percentage", 0)
    yes_count = summary.get("yes_count", 0)
    total = summary.get("total_questions", 0)
    
    print(f"\n{name} Response Results:")
    print(f"Overall score: {yes_percentage:.1f}% positive ({yes_count}/{total} criteria)")
    print("Individual judgments:")
    
    for item in results.get("results", []):
        judgment = "✅ Yes" if item.get("judgment") else "❌ No"
        question = item.get("question")
        confidence = item.get("confidence", 0)
        print(f"  {judgment} | {question} (confidence: {confidence:.3f})")

# Print results for each response
print_results("Good", good_results)
print_results("Average", average_results)
print_results("Poor", poor_results)

# Compare overall scores
print("\nComparison Summary:")
print("-" * 60)
print(f"Good Response: {good_results['summary']['yes_percentage']:.1f}% positive")
print(f"Average Response: {average_results['summary']['yes_percentage']:.1f}% positive")
print(f"Poor Response: {poor_results['summary']['yes_percentage']:.1f}% positive")
```

### Option B: Using the CLI

Using the JSON output files from the previous step, you can compare the results using simple command-line tools:

```bash
# Extract and display the overall scores
echo "Comparison Summary:"
echo "---------------------------------"
echo -n "Good Response: " 
jq '.summary.yes_percentage' good_results.json
echo -n "Average Response: " 
jq '.summary.yes_percentage' average_results.json
echo -n "Poor Response: " 
jq '.summary.yes_percentage' poor_results.json

# Compare which questions failed in the poor response
echo -e "\nFailed criteria in poor response:"
jq '.results[] | select(.judgment==false) | .question' poor_results.json

# Find the best-performing response
echo -e "\nBest performing response:"
GOOD=$(jq '.summary.yes_percentage' good_results.json)
AVG=$(jq '.summary.yes_percentage' average_results.json)
POOR=$(jq '.summary.yes_percentage' poor_results.json)

if (( $(echo "$GOOD > $AVG" | bc -l) )) && (( $(echo "$GOOD > $POOR" | bc -l) )); then
    echo "Good response is best with ${GOOD}% positive"
elif (( $(echo "$AVG > $POOR" | bc -l) )); then
    echo "Average response is best with ${AVG}% positive"
else
    echo "Poor response is best with ${POOR}% positive"
fi
```

## Step 5: Conclusion

You've now learned how to use AgentOptim with both the Python API and the CLI to:

1. Create evaluation sets with quality criteria
2. Evaluate conversations against these criteria
3. Analyze and compare evaluation results

Try creating your own evaluation sets for different use cases, and experiment with different judge models to see which works best for your specific needs.

For more examples and advanced usage, check out:
- [CLI Usage Examples](../examples/cli_usage_examples.md)
- [Complete Examples Directory](../examples/)
- [API Reference](API_REFERENCE.md)
# AgentOptim v2.0 Tutorial

This tutorial walks you through using AgentOptim v2.0 to evaluate the quality of conversations with a language model.

## Prerequisites

Before starting, ensure you have:

1. Python 3.8+ installed
2. AgentOptim installed: `pip install agentoptim`
3. Access to a judge model (e.g., meta-llama-3.1-8b-instruct)

## Introduction to the 2-Tool Architecture

AgentOptim v2.0 simplifies prompt optimization with just two tools:

1. `manage_evalset_tool` - Create and manage sets of evaluation criteria
2. `run_evalset_tool` - Run evaluations on conversations using an EvalSet

This streamlined approach makes it easier to evaluate and improve conversation quality.

## Step 1: Setting Up Your Project

Let's create a simple script that uses AgentOptim to evaluate different support responses:

```python
# support_response_evaluation.py

import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

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

First, let's create an EvalSet with criteria for evaluating customer support responses:

```python
# Add this to the main() function

# Step 1: Create an EvalSet with evaluation criteria
print("\n1. Creating Support Response Quality EvalSet...")

evalset_result = await manage_evalset_tool(
    action="create",
    name="Support Response Quality",
    template="""
    Given this conversation:
    {{ conversation }}
    
    Please answer the following yes/no question about the final assistant response:
    {{ eval_question }}
    
    Return a JSON object with the following format:
    {"judgment": 1} for yes or {"judgment": 0} for no.
    """,
    questions=[
        "Is the response helpful for the user's needs?",
        "Does the response directly address the user's question?",
        "Is the response clear and easy to understand?",
        "Does the response provide accurate information?",
        "Does the response provide complete information?",
        "Is the tone of the response appropriate and professional?",
        "Does the response avoid unnecessary information?"
    ],
    description="Evaluation criteria for customer support response quality"
)

# Extract the EvalSet ID
evalset_id = evalset_result.get("evalset", {}).get("id")
print(f"EvalSet created with ID: {evalset_id}")
print(f"EvalSet contains {len(evalset_result.get('evalset', {}).get('questions', []))} evaluation questions")
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

## Step 4: Running the Evaluations

Now, let's evaluate each conversation:

```python
# Add after defining the conversations

# Step 3: Evaluate the conversations
print("\n3. Running evaluations...")

# Evaluate the good response
print("\nEvaluating good response...")
good_results = await run_evalset_tool(
    evalset_id=evalset_id,
    conversation=good_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# Evaluate the average response
print("\nEvaluating average response...")
average_results = await run_evalset_tool(
    evalset_id=evalset_id,
    conversation=average_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

# Evaluate the poor response
print("\nEvaluating poor response...")
poor_results = await run_evalset_tool(
    evalset_id=evalset_id,
    conversation=poor_conversation,
    model="meta-llama-3.1-8b-instruct",
    max_parallel=3
)

print("\nAll evaluations completed!")
```

## Step 5: Analyzing Results

Let's compare the results of our evaluations:

```python
# Add after running the evaluations

# Step 4: Analyze and compare results
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
        logprob = item.get("logprob", 0)
        confidence = abs(logprob)  # Higher absolute value = higher confidence
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

# Determine which response performed best
best_score = max(
    good_results['summary']['yes_percentage'],
    average_results['summary']['yes_percentage'],
    poor_results['summary']['yes_percentage']
)

if best_score == good_results['summary']['yes_percentage']:
    best_response = "detailed step-by-step"
elif best_score == average_results['summary']['yes_percentage']:
    best_response = "brief but direct"
else:
    best_response = "vague"

print(f"\nBest performing response style: {best_response}")

print("\nRecommendations:")
print("Based on the evaluation results, customer support responses should:")
if best_score == good_results['summary']['yes_percentage']:
    print("1. Provide step-by-step instructions when applicable")
    print("2. Anticipate follow-up questions")
    print("3. Offer additional helpful information")
    print("4. Use a friendly, professional tone")
else:
    print("1. Provide more specific information")
    print("2. Include step-by-step instructions")
    print("3. Ensure completeness of information")
```

## Step 6: Running the Complete Script

Your complete script should look like this (with imports at the top):

```python
import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

async def main():
    print("AgentOptim Support Response Evaluation")
    print("=" * 50)
    
    # Step 1: Create an EvalSet with evaluation criteria
    print("\n1. Creating Support Response Quality EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Support Response Quality",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please answer the following yes/no question about the final assistant response:
        {{ eval_question }}
        
        Return a JSON object with the following format:
        {"judgment": 1} for yes or {"judgment": 0} for no.
        """,
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Does the response provide accurate information?",
            "Does the response provide complete information?",
            "Is the tone of the response appropriate and professional?",
            "Does the response avoid unnecessary information?"
        ],
        description="Evaluation criteria for customer support response quality"
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"EvalSet created with ID: {evalset_id}")
    print(f"EvalSet contains {len(evalset_result.get('evalset', {}).get('questions', []))} evaluation questions")
    
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
    
    # Step 3: Evaluate the conversations
    print("\n3. Running evaluations...")
    
    # Evaluate the good response
    print("\nEvaluating good response...")
    good_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=good_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Evaluate the average response
    print("\nEvaluating average response...")
    average_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=average_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # Evaluate the poor response
    print("\nEvaluating poor response...")
    poor_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=poor_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nAll evaluations completed!")
    
    # Step 4: Analyze and compare results
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
            logprob = item.get("logprob", 0)
            confidence = abs(logprob)  # Higher absolute value = higher confidence
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
    
    # Determine which response performed best
    best_score = max(
        good_results['summary']['yes_percentage'],
        average_results['summary']['yes_percentage'],
        poor_results['summary']['yes_percentage']
    )
    
    if best_score == good_results['summary']['yes_percentage']:
        best_response = "detailed step-by-step"
    elif best_score == average_results['summary']['yes_percentage']:
        best_response = "brief but direct"
    else:
        best_response = "vague"
    
    print(f"\nBest performing response style: {best_response}")
    
    print("\nRecommendations:")
    print("Based on the evaluation results, customer support responses should:")
    if best_score == good_results['summary']['yes_percentage']:
        print("1. Provide step-by-step instructions when applicable")
        print("2. Anticipate follow-up questions")
        print("3. Offer additional helpful information")
        print("4. Use a friendly, professional tone")
    else:
        print("1. Provide more specific information")
        print("2. Include step-by-step instructions")
        print("3. Ensure completeness of information")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run the complete script:

```bash
python support_response_evaluation.py
```

## Expected Output

Here's what you can expect to see:

```
AgentOptim Support Response Evaluation
==================================================

1. Creating Support Response Quality EvalSet...
EvalSet created with ID: evalset_abc123
EvalSet contains 7 evaluation questions

2. Defining conversations to evaluate...
Defined 3 conversations with different response quality levels

3. Running evaluations...

Evaluating good response...

Evaluating average response...

Evaluating poor response...

All evaluations completed!

4. Analyzing results...

Good Response Results:
Overall score: 100.0% positive (7/7 criteria)
Individual judgments:
  ✅ Yes | Is the response helpful for the user's needs? (confidence: 0.023)
  ✅ Yes | Does the response directly address the user's question? (confidence: 0.019)
  ✅ Yes | Is the response clear and easy to understand? (confidence: 0.021)
  ✅ Yes | Does the response provide accurate information? (confidence: 0.025)
  ✅ Yes | Does the response provide complete information? (confidence: 0.018)
  ✅ Yes | Is the tone of the response appropriate and professional? (confidence: 0.022)
  ✅ Yes | Does the response avoid unnecessary information? (confidence: 0.029)

Average Response Results:
Overall score: 71.4% positive (5/7 criteria)
Individual judgments:
  ✅ Yes | Is the response helpful for the user's needs? (confidence: 0.026)
  ✅ Yes | Does the response directly address the user's question? (confidence: 0.021)
  ✅ Yes | Is the response clear and easy to understand? (confidence: 0.024)
  ✅ Yes | Does the response provide accurate information? (confidence: 0.028)
  ❌ No | Does the response provide complete information? (confidence: 0.032)
  ✅ Yes | Is the tone of the response appropriate and professional? (confidence: 0.025)
  ❌ No | Does the response avoid unnecessary information? (confidence: 0.031)

Poor Response Results:
Overall score: 28.6% positive (2/7 criteria)
Individual judgments:
  ❌ No | Is the response helpful for the user's needs? (confidence: 0.035)
  ❌ No | Does the response directly address the user's question? (confidence: 0.031)
  ✅ Yes | Is the response clear and easy to understand? (confidence: 0.028)
  ❌ No | Does the response provide accurate information? (confidence: 0.033)
  ❌ No | Does the response provide complete information? (confidence: 0.038)
  ✅ Yes | Is the tone of the response appropriate and professional? (confidence: 0.027)
  ❌ No | Does the response avoid unnecessary information? (confidence: 0.030)

Comparison Summary:
------------------------------------------------------------
Good Response: 100.0% positive
Average Response: 71.4% positive
Poor Response: 28.6% positive

Best performing response style: detailed step-by-step

Recommendations:
Based on the evaluation results, customer support responses should:
1. Provide step-by-step instructions when applicable
2. Anticipate follow-up questions
3. Offer additional helpful information
4. Use a friendly, professional tone

Evaluation complete!
```

## What's Next?

Now that you've evaluated different response styles, you can:

1. Create more specific EvalSets for different types of support questions
2. Test different system prompts to guide response styles
3. Evaluate responses for different customer scenarios
4. Use your findings to improve your production AI assistants

AgentOptim v2.0 makes it easy to evaluate and improve AI conversations through data-driven assessment, helping you deliver better user experiences.

For more advanced usage and features, check out the [examples directory](../examples/) in the repository.

Happy optimizing!
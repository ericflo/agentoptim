# AgentOptim v2.0 Workflow Guide

This guide provides a comprehensive overview of the AgentOptim v2.0 workflow, from EvalSet creation to conversation evaluation, with practical examples.

## Streamlined 2-Tool Workflow

AgentOptim v2.0 provides a simplified workflow with just two powerful tools:

1. **Create an EvalSet** - Define criteria for evaluating conversations
2. **Run Evaluations** - Evaluate conversations against your criteria

This streamlined approach makes it easier to assess and improve conversation quality.

## Quick Start Example

Here's a complete example of the AgentOptim workflow:

```python
import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

async def main():
    # 1. Create an EvalSet with evaluation criteria
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality",
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
            "Is the tone of the response appropriate?"
        ],
        description="Basic criteria for evaluating response quality"
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    
    # 2. Define a conversation to evaluate
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ]
    
    # 3. Run the evaluation
    eval_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # 4. View the results
    print(f"Overall score: {eval_results.get('summary', {}).get('yes_percentage')}% positive")
    print("Individual judgments:")
    for result in eval_results.get("results", []):
        judgment = "Yes" if result.get("judgment") else "No"
        question = result.get("question")
        print(f"- {question}: {judgment}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tool Usage

### Managing EvalSets with `manage_evalset_tool`

The `manage_evalset_tool` provides CRUD operations for EvalSets:

#### Creating an EvalSet

```python
evalset_result = await manage_evalset_tool(
    action="create",
    name="Code Review Quality",
    template="""
    Given this conversation:
    {{ conversation }}
    
    Please answer the following yes/no question about the final assistant response:
    {{ eval_question }}
    
    Return a JSON object with the following format:
    {"judgment": 1} for yes or {"judgment": 0} for no.
    """,
    questions=[
        "Does the code review correctly identify issues?",
        "Does the review provide helpful suggestions for improvement?",
        "Is the review clear and easy to understand?",
        "Is the tone of the review constructive and professional?"
    ],
    description="Evaluation criteria for code review quality"
)

evalset_id = evalset_result.get("evalset", {}).get("id")
print(f"Created EvalSet with ID: {evalset_id}")
```

#### Getting an EvalSet

```python
evalset = await manage_evalset_tool(
    action="get",
    evalset_id="evalset_123abc"
)

print(f"EvalSet name: {evalset.get('evalset', {}).get('name')}")
print(f"Number of questions: {len(evalset.get('evalset', {}).get('questions', []))}")
```

#### Listing EvalSets

```python
evalsets = await manage_evalset_tool(
    action="list"
)

print(f"Found {len(evalsets.get('evalsets', []))} EvalSets:")
for evalset in evalsets.get("evalsets", []):
    print(f"- {evalset.get('name')} (ID: {evalset.get('id')})")
```

#### Updating an EvalSet

```python
updated_evalset = await manage_evalset_tool(
    action="update",
    evalset_id="evalset_123abc",
    name="Updated Code Review Quality",
    description="Improved evaluation criteria for code review quality"
)

print(f"Updated EvalSet: {updated_evalset.get('evalset', {}).get('name')}")
```

#### Deleting an EvalSet

```python
result = await manage_evalset_tool(
    action="delete",
    evalset_id="evalset_123abc"
)

print(f"Deletion success: {result.get('success', False)}")
```

### Evaluating Conversations with `run_evalset_tool`

The `run_evalset_tool` evaluates conversations against an EvalSet:

```python
# Define a conversation to evaluate
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you explain how to implement a binary search algorithm?"},
    {"role": "assistant", "content": "Binary search is an efficient algorithm for finding an item in a sorted list. Here's how it works: 1) Start with the middle element. 2) If the target is equal to the middle element, return. 3) If the target is less than the middle element, repeat the search on the left half. 4) If the target is greater than the middle element, repeat the search on the right half. 5) Continue until you find the target or determine it's not in the list. The time complexity is O(log n)."}
]

# Run the evaluation
eval_results = await run_evalset_tool(
    evalset_id="evalset_123abc",
    conversation=conversation,
    model="meta-llama-3.1-8b-instruct",  # Specify the judge model
    max_parallel=3  # Run up to 3 evaluations in parallel
)

# Process the results
summary = eval_results.get("summary", {})
yes_percentage = summary.get("yes_percentage", 0)
yes_count = summary.get("yes_count", 0)
total = summary.get("total_questions", 0)

print(f"Overall score: {yes_percentage:.1f}% positive ({yes_count}/{total} criteria)")
print("Individual judgments:")
for item in eval_results.get("results", []):
    judgment = "Yes" if item.get("judgment") else "No"
    question = item.get("question")
    logprob = item.get("logprob", 0)
    confidence = abs(logprob)  # Higher absolute value = higher confidence
    print(f"- {question}: {judgment} (confidence: {confidence:.3f})")
```

## Comparing Multiple Conversations

A key use case is comparing different conversation styles:

```python
async def compare_conversations(evalset_id, conversations, model="meta-llama-3.1-8b-instruct"):
    """Compare multiple conversations using the same EvalSet."""
    results = {}
    
    for name, conversation in conversations.items():
        print(f"Evaluating {name}...")
        eval_result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            model=model,
            max_parallel=3
        )
        results[name] = eval_result
    
    # Compare results
    print("\nComparison Summary:")
    print("-" * 60)
    for name, result in results.items():
        yes_percentage = result.get("summary", {}).get("yes_percentage", 0)
        print(f"{name}: {yes_percentage:.1f}% positive")
    
    # Find the best conversation
    best_name = max(results, key=lambda x: results[x].get("summary", {}).get("yes_percentage", 0))
    best_score = results[best_name].get("summary", {}).get("yes_percentage", 0)
    
    print(f"\nBest performing conversation: {best_name} ({best_score:.1f}%)")
    return results

# Usage example
conversations = {
    "Detailed": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I make a chocolate cake?"},
        {"role": "assistant", "content": "Here's a detailed recipe for chocolate cake: 1) Preheat oven to 350°F. 2) Mix 2 cups flour, 2 cups sugar, 3/4 cup cocoa powder, 2 tsp baking soda, 1 tsp baking powder, 1 tsp salt. 3) Add 2 eggs, 1 cup milk, 1/2 cup oil, 2 tsp vanilla, and 1 cup hot coffee. 4) Pour into greased pans. 5) Bake 30-35 minutes. Let me know if you need frosting instructions too!"}
    ],
    "Concise": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I make a chocolate cake?"},
        {"role": "assistant", "content": "Mix dry ingredients (flour, sugar, cocoa, baking soda/powder, salt). Add wet ingredients (eggs, milk, oil, vanilla, hot coffee). Bake at 350°F for 30-35 minutes."}
    ],
    "Conversational": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I make a chocolate cake?"},
        {"role": "assistant", "content": "Making a chocolate cake is fun! Start by mixing your dry ingredients like flour, sugar, and cocoa. Then add eggs, milk, oil, and vanilla. A secret tip: add hot coffee for extra richness! Bake at 350°F for about 30 minutes or until a toothpick comes out clean. Enjoy your delicious cake!"}
    ]
}

comparison_results = await compare_conversations(evalset_id, conversations)
```

## Full Workflow Example: Improving Customer Support Responses

This example demonstrates evaluating and improving customer support responses:

```python
import asyncio
from agentoptim import manage_evalset_tool, run_evalset_tool

async def main():
    # 1. Create a support quality EvalSet
    print("Creating Support Quality EvalSet...")
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
    
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"Created EvalSet with ID: {evalset_id}")
    
    # 2. Define conversations with different response styles
    print("\nDefining conversations with different response styles...")
    
    password_reset = {
        "Detailed": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to our login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder. If you still need help, let me know and I'd be happy to assist further."}
        ],
        "Brief": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "To reset your password, go to the login page and click on the 'Forgot Password' link. You'll receive an email with instructions to create a new password."}
        ],
        "Vague": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "You should be able to find password reset options in your account settings or on the login page."}
        ]
    }
    
    order_status = {
        "Detailed": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How can I check my order status?"},
            {"role": "assistant", "content": "You can check your order status in several ways:\n\n1. Visit our website and log in to your account\n2. Navigate to 'My Orders' in your account dashboard\n3. Find your order in the list and click on it to view detailed status\n\nAlternatively, you can:\n- Use our mobile app and check the Orders section\n- Click the tracking link in your order confirmation email\n- Reply to this message with your order number, and I can check it for you\n\nIs there a specific order you're concerned about?"}
        ],
        "Brief": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How can I check my order status?"},
            {"role": "assistant", "content": "You can check your order status by logging into your account on our website and going to 'My Orders'. You can also click the tracking link in your order confirmation email."}
        ],
        "Vague": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": "How can I check my order status?"},
            {"role": "assistant", "content": "Order status information is available in your account."}
        ]
    }
    
    # 3. Evaluate all conversations
    print("\nEvaluating password reset conversations...")
    password_results = {}
    for style, conversation in password_reset.items():
        print(f"  Evaluating {style} style...")
        result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            model="meta-llama-3.1-8b-instruct",
            max_parallel=3
        )
        password_results[style] = result
    
    print("\nEvaluating order status conversations...")
    order_results = {}
    for style, conversation in order_status.items():
        print(f"  Evaluating {style} style...")
        result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            model="meta-llama-3.1-8b-instruct",
            max_parallel=3
        )
        order_results[style] = result
    
    # 4. Analyze and compare results
    print("\n=============================================")
    print("RESULTS SUMMARY")
    print("=============================================")
    
    print("\nPassword Reset Scenario:")
    print("-" * 40)
    for style, result in password_results.items():
        yes_percentage = result.get("summary", {}).get("yes_percentage", 0)
        print(f"{style}: {yes_percentage:.1f}% positive")
    
    print("\nOrder Status Scenario:")
    print("-" * 40)
    for style, result in order_results.items():
        yes_percentage = result.get("summary", {}).get("yes_percentage", 0)
        print(f"{style}: {yes_percentage:.1f}% positive")
    
    # 5. Generate recommendations
    print("\n=============================================")
    print("RECOMMENDATIONS")
    print("=============================================")
    
    # Find best style for each scenario
    best_password_style = max(password_results, key=lambda x: password_results[x].get("summary", {}).get("yes_percentage", 0))
    best_order_style = max(order_results, key=lambda x: order_results[x].get("summary", {}).get("yes_percentage", 0))
    
    print(f"\nBest style for password reset: {best_password_style}")
    print(f"Best style for order status: {best_order_style}")
    
    # Overall recommendation
    if best_password_style == best_order_style:
        print(f"\nOverall recommendation: The {best_password_style} style performs best across different scenarios.")
    else:
        print(f"\nDifferent styles work best for different scenarios:")
        print(f"- Password reset: {best_password_style}")
        print(f"- Order status: {best_order_style}")
    
    # Detailed analysis of the best style
    best_style = best_password_style if best_password_style == best_order_style else "Detailed"
    print(f"\nKey characteristics of the {best_style} style:")
    if best_style == "Detailed":
        print("1. Provide step-by-step instructions")
        print("2. Include alternative methods when applicable")
        print("3. Anticipate follow-up questions")
        print("4. Offer additional help for edge cases")
    elif best_style == "Brief":
        print("1. Get straight to the point")
        print("2. Include all necessary information without elaboration")
        print("3. Use concise language with clear instructions")
    else:
        print("1. Provide more specific information")
        print("2. Include step-by-step instructions")
        print("3. Ensure completeness of information")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tips for Effective Evaluation

1. **Create focused EvalSets** - Define specific criteria for different aspects of conversation quality
2. **Use realistic conversations** - Test with examples that reflect actual user interactions
3. **Compare different approaches** - Evaluate multiple conversation styles to find the best approach
4. **Use appropriate models** - Choose judge models with appropriate capabilities
5. **Balance detail and brevity** - Evaluate both detailed and concise responses
6. **Consider audience needs** - Create criteria that reflect your specific users' needs
7. **Iterate on your EvalSets** - Refine criteria based on evaluation results
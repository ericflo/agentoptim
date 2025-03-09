#!/usr/bin/env python
"""
Evaluating customer support responses using AgentOptim v2.0.

This script demonstrates how to use AgentOptim's EvalSet architecture to:
1. Create an EvalSet with quality criteria for support responses
2. Evaluate multiple conversation examples with different response styles
3. Analyze and compare results to identify the best approach

See the full tutorial in docs/TUTORIAL.md
"""

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
        {"reasoning": "Your reasoning here", "judgment": true, "confidence": 0.9} for yes
        or {"reasoning": "Your reasoning here", "judgment": false, "confidence": 0.7} for no.
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
    
    if not evalset_id:
        # If EvalSet creation failed, print error and exit
        print("Error: Failed to create EvalSet. Please check your configuration.")
        print(f"Error details: {evalset_result.get('error', 'Unknown error')}")
        return
        
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
        max_parallel=3
    )
    
    # Evaluate the average response
    print("\nEvaluating average response...")
    average_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=average_conversation,
        max_parallel=3
    )
    
    # Evaluate the poor response
    print("\nEvaluating poor response...")
    poor_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=poor_conversation,
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
            confidence = item.get("confidence", 0)
            reasoning = item.get("reasoning", "")
            print(f"  {judgment} | {question} (confidence: {confidence:.3f})")
            if reasoning:
                print(f"    Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
    
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
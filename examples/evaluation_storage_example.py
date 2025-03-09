"""
Example showing the evaluation storage and retrieval functionality in AgentOptim v2.1.1.

This example demonstrates:
1. Creating an EvalSet with evaluation criteria
2. Running an evaluation and storing the results
3. Retrieving a stored evaluation by ID
4. Listing all evaluations with filtering and pagination
5. Comparing results from multiple evaluations

To run this example:
1. Make sure you have AgentOptim v2.1.1+ installed
2. Start the AgentOptim server with: agentoptim server
3. Run this script with: python evaluation_storage_example.py
"""

import asyncio
import time
from pprint import pprint

from agentoptim import manage_evalset_tool, manage_eval_runs_tool


async def create_evalset():
    """Create a sample EvalSet for evaluation."""
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Password Reset Instructions",
        questions=[
            "Does the response directly address the user's question about password reset?",
            "Are the instructions clear and easy to follow?",
            "Does the response include all necessary steps?",
            "Is the tone helpful and professional?",
            "Is the response concise without unnecessary information?"
        ],
        short_description="Evaluating password reset instructions",
        long_description="This EvalSet evaluates how well an assistant provides password reset instructions. It checks for completeness, clarity, helpfulness, professionalism, and conciseness. High scores indicate a response that provides clear, complete instructions that directly address the user's need to reset their password."
    )
    
    # Handle different possible response formats
    if "evalset" in evalset_result:
        return evalset_result["evalset"]["id"]
    elif "id" in evalset_result:
        return evalset_result["id"]
    elif "result" in evalset_result and isinstance(evalset_result["result"], str):
        # Extract ID from result message using regex
        import re
        id_match = re.search(r"ID: ([a-f0-9\-]+)", evalset_result["result"])
        if id_match:
            return id_match.group(1)
    
    # Print the actual response structure for debugging
    print("Unexpected response structure:", evalset_result)
    # Try to find an ID in the response
    if isinstance(evalset_result, dict):
        for key, value in evalset_result.items():
            if isinstance(value, dict) and "id" in value:
                return value["id"]
            elif key == "id" and isinstance(value, str):
                return value
    
    raise ValueError("Could not find evalset ID in response")


async def run_evaluations(evalset_id):
    """Run multiple evaluations with different responses and store the results."""
    print("\n📊 Running and storing evaluations...")
    
    # A good response
    good_conversation = [
        {"role": "system", "content": "You are a helpful customer service assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to the login page\n2. Click on 'Forgot Password' below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder."}
    ]
    
    # A mediocre response
    mediocre_conversation = [
        {"role": "system", "content": "You are a helpful customer service assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "You can reset your password by clicking on 'Forgot Password' on the login page and then following the instructions sent to your email."}
    ]
    
    # Run and store both evaluations
    good_eval = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=good_conversation,
        max_parallel=5  # Process all 5 questions at once
    )
    
    # Wait a moment to ensure timestamps are different
    time.sleep(1)
    
    mediocre_eval = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=mediocre_conversation,
        max_parallel=5
    )
    
    return good_eval["id"], mediocre_eval["id"]


async def retrieve_evaluation(eval_run_id):
    """Retrieve a stored evaluation by ID."""
    print(f"\n🔍 Retrieving evaluation with ID: {eval_run_id}")
    
    eval_result = await manage_eval_runs_tool(
        action="get",
        eval_run_id=eval_run_id
    )
    
    # Print out the formatted results
    print("\n" + eval_result["formatted_message"])
    
    return eval_result["eval_run"]


async def list_evaluations(evalset_id=None):
    """List all evaluations with optional filtering by EvalSet ID."""
    print("\n📋 Listing evaluations" + (f" for EvalSet: {evalset_id}" if evalset_id else ""))
    
    # List all evaluations, or filter by EvalSet ID if provided
    list_params = {
        "action": "list",
        "page": 1,
        "page_size": 10
    }
    
    if evalset_id:
        list_params["evalset_id"] = evalset_id
        
    eval_list = await manage_eval_runs_tool(**list_params)
    
    # Print out the list with summary information
    print("\n" + eval_list["formatted_message"])
    
    return eval_list["eval_runs"]


async def compare_evaluations(good_id, mediocre_id):
    """Compare two evaluations to see the differences."""
    print(f"\n⚖️ Comparing evaluations: {good_id} vs {mediocre_id}")
    
    # Retrieve both evaluations
    good_eval = await manage_eval_runs_tool(
        action="get",
        eval_run_id=good_id
    )
    
    mediocre_eval = await manage_eval_runs_tool(
        action="get",
        eval_run_id=mediocre_id
    )
    
    # Extract the summary data
    good_summary = good_eval["eval_run"]["summary"]
    mediocre_summary = mediocre_eval["eval_run"]["summary"]
    
    # Compare the scores
    print("\n📊 Comparison Results:")
    print(f"Good response score: {good_summary['yes_percentage']}%")
    print(f"Mediocre response score: {mediocre_summary['yes_percentage']}%")
    print(f"Difference: {good_summary['yes_percentage'] - mediocre_summary['yes_percentage']}%")
    
    # Compare individual question results
    print("\n📝 Question-by-Question Comparison:")
    good_results = good_eval["eval_run"]["results"]
    mediocre_results = mediocre_eval["eval_run"]["results"]
    
    for i, (good_q, mediocre_q) in enumerate(zip(good_results, mediocre_results)):
        print(f"\nQ{i+1}: {good_q['question']}")
        print(f"  Good: {'✅ Yes' if good_q['judgment'] else '❌ No'}")
        print(f"  Mediocre: {'✅ Yes' if mediocre_q['judgment'] else '❌ No'}")
        
        if good_q['judgment'] != mediocre_q['judgment']:
            print(f"  Difference: {'✅ Good response better' if good_q['judgment'] else '❌ Mediocre response better'}")
    
    return {
        "good_score": good_summary['yes_percentage'],
        "mediocre_score": mediocre_summary['yes_percentage'],
        "difference": good_summary['yes_percentage'] - mediocre_summary['yes_percentage']
    }


async def main():
    """Main function to run the example."""
    print("🚀 AgentOptim Evaluation Storage Example")
    print("========================================")
    
    # 1. Create an EvalSet
    evalset_id = await create_evalset()
    print(f"\n✅ Created EvalSet with ID: {evalset_id}")
    
    # 2. Run and store evaluations
    good_id, mediocre_id = await run_evaluations(evalset_id)
    print(f"\n✅ Stored evaluations with IDs:")
    print(f"   - Good response: {good_id}")
    print(f"   - Mediocre response: {mediocre_id}")
    
    # 3. Retrieve an evaluation by ID
    good_eval = await retrieve_evaluation(good_id)
    
    # 4. List all evaluations
    all_evals = await list_evaluations()
    print(f"\n✅ Found {len(all_evals)} total evaluations")
    
    # 5. List evaluations filtered by EvalSet ID
    filtered_evals = await list_evaluations(evalset_id)
    print(f"\n✅ Found {len(filtered_evals)} evaluations for EvalSet ID: {evalset_id}")
    
    # 6. Compare evaluations
    comparison = await compare_evaluations(good_id, mediocre_id)
    
    print("\n🏁 Example Complete")
    print("=================")
    print("\nThis example demonstrated:")
    print("1. Creating an EvalSet with evaluation criteria")
    print("2. Running evaluations and storing the results")
    print("3. Retrieving stored evaluations by ID")
    print("4. Listing and filtering evaluations")
    print("5. Comparing results from multiple evaluations")
    
    print("\nKey features of AgentOptim's evaluation storage:")
    print("- Persistent storage of all evaluation results")
    print("- Unique IDs for retrieving specific evaluations")
    print("- Pagination for handling large numbers of evaluations")
    print("- Filtering capability to find evaluations by EvalSet")
    print("- Formatted output for easy reading of results")
    print("- Complete storage of conversation context with results")


if __name__ == "__main__":
    asyncio.run(main())
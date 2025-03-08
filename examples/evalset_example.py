"""
Comprehensive example of the AgentOptim v2.0 EvalSet architecture.

This example demonstrates:
1. Creating an EvalSet with evaluation criteria
2. Getting EvalSet details
3. Updating an EvalSet
4. Running evaluations with different models
5. Comparing evaluation results
6. Listing and deleting EvalSets
"""

import asyncio
import json
from pprint import pprint

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim v2.0 EvalSet Architecture Demo ===")
    
    # 1. Create an EvalSet with evaluation criteria
    print("\n1. Creating a new EvalSet for evaluating response quality...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality Evaluation",
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
            "Is the response accurate?",
            "Does the response provide complete information?"
        ],
        description="Evaluation criteria for response quality"
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"EvalSet created with ID: {evalset_id}")
    
    # 2. Get EvalSet details
    print("\n2. Getting EvalSet details...")
    evalset_details = await manage_evalset_tool(
        action="get",
        evalset_id=evalset_id
    )
    print("EvalSet details:")
    pprint(evalset_details)
    
    # 3. Update the EvalSet
    print("\n3. Updating the EvalSet...")
    updated_evalset = await manage_evalset_tool(
        action="update",
        evalset_id=evalset_id,
        name="Enhanced Response Quality Evaluation",
        description="Improved evaluation criteria for response quality"
    )
    print("Updated EvalSet:")
    pprint(updated_evalset)
    
    # 4. Define conversations to evaluate
    good_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password. If you don't receive the email within a few minutes, check your spam folder or contact support for assistance."}
    ]
    
    average_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to the login page and click 'Forgot Password'."}
    ]
    
    poor_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "You should be able to find that in the settings."}
    ]
    
    # 5. Run evaluations with different models
    print("\n4. Running evaluations with different models...")
    
    print("\na) Evaluating good response with default model...")
    good_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=good_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nb) Evaluating average response with default model...")
    average_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=average_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    print("\nc) Evaluating poor response with default model...")
    poor_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=poor_conversation,
        model="meta-llama-3.1-8b-instruct",
        max_parallel=3
    )
    
    # 6. Compare evaluation results
    print("\n5. Comparing evaluation results:")
    print(f"Good response: {good_results.get('summary', {}).get('yes_percentage')}% positive")
    print(f"Average response: {average_results.get('summary', {}).get('yes_percentage')}% positive")
    print(f"Poor response: {poor_results.get('summary', {}).get('yes_percentage')}% positive")
    
    # 7. List all EvalSets
    print("\n6. Listing all EvalSets...")
    evalsets = await manage_evalset_tool(action="list")
    print(f"Found {len(evalsets.get('evalsets', []))} EvalSets:")
    for evalset in evalsets.get("evalsets", []):
        print(f"  - {evalset.get('name')} (ID: {evalset.get('id')})")
    
    # 8. Delete the EvalSet (commented out to preserve the EvalSet)
    print("\n7. Deleting the EvalSet... (uncomment to actually delete)")
    # delete_result = await manage_evalset_tool(
    #     action="delete",
    #     evalset_id=evalset_id
    # )
    # print(f"EvalSet deleted: {delete_result.get('success', False)}")


if __name__ == "__main__":
    asyncio.run(main())
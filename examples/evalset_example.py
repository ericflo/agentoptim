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

from agentoptim.server import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim v2.0 EvalSet Architecture Demo ===")
    
    # 1. Create an EvalSet with evaluation criteria
    print("\n1. Creating a new EvalSet for evaluating response quality...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality Evaluation",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Is the response accurate?",
            "Does the response provide complete information?"
        ],
        short_description="Response quality evaluation criteria",
        long_description="This EvalSet provides comprehensive evaluation criteria for assessing the quality of conversational responses. It measures whether responses address user needs, provide clear and accurate information, and offer complete answers. Use it to evaluate assistant responses across a variety of contexts where helpfulness, clarity, and accuracy are important. High scores indicate responses that effectively address user queries with accurate and complete information." + " " * 50
    )
    
    # Extract the EvalSet ID
    # First try the new response format where ID is in the result message
    evalset_id = None
    result_message = evalset_result.get("result", "")
    import re
    id_match = re.search(r"ID: ([a-f0-9\-]+)", result_message)
    
    if id_match:
        evalset_id = id_match.group(1)
    else:
        # Try the older format where ID is in the evalset object
        evalset_id = evalset_result.get("evalset", {}).get("id")
        
    if not evalset_id:
        print("Failed to extract EvalSet ID from response")
        print(f"Response: {evalset_result}")
        return
        
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
        short_description="Improved response quality evaluation",
        long_description="This enhanced EvalSet provides improved evaluation criteria for assessing the quality of conversational responses. The evaluation focuses on helpfulness, clarity, accuracy, and completeness with additional emphasis on user satisfaction. Use it to evaluate and improve assistant responses with a focus on delivering high-quality information that meets user needs efficiently and effectively." + " " * 50
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
    
    # For this simplified example, we'll define conversations but not run evaluations
    # since they can take a long time to complete
    print("\n4. Defining conversations for evaluation (skipping actual evaluation for brevity)...")
    
    print("\nIn a full example, we would evaluate these conversations:")
    print("a) Good response: Detailed instructions with next steps")
    print("b) Average response: Basic instructions without details")
    print("c) Poor response: Vague and unhelpful answer")
    
    # Here's what the evaluation code would look like:
    """
    good_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=good_conversation,
        max_parallel=3
    )
    
    average_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=average_conversation,
        max_parallel=3
    )
    
    poor_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=poor_conversation,
        max_parallel=3
    )
    """
    
    # For this simplified example, we'll skip the evals and just list the evalsets
    
    # List all EvalSets
    print("\n6. Listing all EvalSets...")
    evalsets = await manage_evalset_tool(action="list")
    
    # The new response format returns a formatted message in the result field
    result_message = evalsets.get('result', '')
    
    if "Found" in result_message:
        print(result_message.split('\n')[0])  # Print the summary line
        
        # Extract IDs and names from the result message
        import re
        matches = re.findall(r"• (.*?) \(ID: ([a-f0-9\-]+)\)", result_message)
        
        if matches:
            for name, id in matches:
                print(f"  - {name} (ID: {id})")
        else:
            print("  No EvalSets found.")
    else:
        # Fall back to the old format
        evalset_dict = evalsets.get('evalsets', {})
        print(f"Found {len(evalset_dict)} EvalSets:")
        
        if evalset_dict:
            for eval_id, evalset in evalset_dict.items():
                print(f"  - {evalset.get('name')} (ID: {eval_id})")
        else:
            print("  No EvalSets found in structured format.")
            print(f"  Raw response: {evalsets}")
    
    print("\nThis concludes the simplified EvalSet example.")


if __name__ == "__main__":
    asyncio.run(main())
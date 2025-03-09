"""Example usage of the AgentOptim EvalSet architecture."""

import asyncio
import json

from agentoptim.server import manage_evalset_tool, run_evalset_tool


async def main():
    """Demonstrate using the new EvalSet architecture."""
    print("Creating a new EvalSet for evaluating helpfulness...")
    
    # First, create an EvalSet with evaluation criteria
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Helpfulness Evaluation",
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
        description="Evaluation criteria for helpfulness of responses"
    )
    
    print("\nEvalSet created successfully")
    # Extract the EvalSet ID from the response
    evalset_id = evalset_result.get("evalset", {}).get("id")
    if not evalset_id:
        print("Failed to extract EvalSet ID from response")
        return
    
    print(f"EvalSet ID: {evalset_id}")
    
    # Define a conversation to evaluate
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
    ]
    
    print("\nEvaluating conversation...")
    
    # Run the evaluation on the conversation
    eval_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=conversation,
        max_parallel=3
    )
    
    print("\nEvaluation results:")
    print(eval_results)
    
    # Compare with a different response
    print("\nNow comparing with a different, less helpful response...")
    
    less_helpful_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "You should be able to find that in the settings."}
    ]
    
    # Run the evaluation on the less helpful conversation
    less_helpful_results = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=less_helpful_conversation,
        max_parallel=3
    )
    
    print("\nLess helpful response evaluation results:")
    print(less_helpful_results)
    
    # List available EvalSets
    print("\nListing available EvalSets:")
    evalsets = await manage_evalset_tool(action="list")
    print(evalsets)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
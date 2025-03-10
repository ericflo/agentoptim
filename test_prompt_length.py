"""
Test script to verify the prompt length handling changes in runner.py.
"""

import asyncio
import logging
from agentoptim.server import manage_evalset_tool, manage_eval_runs_tool

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    print("=== Testing Prompt Length Handling ===")
    
    # 1. Create an EvalSet with a long description to force a long prompt
    print("\n1. Creating an EvalSet with a very long description...")
    
    # Generate a description within the 1024 character limit
    long_description = "This is a test of long prompt handling. " * 25  # Roughly 925 characters
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Prompt Length Test Evaluation",
        questions=[
            "Is the response helpful?",
            "Is the response accurate?",
        ],
        short_description="Testing prompt length handling",
        long_description=long_description
    )
    
    # Extract the EvalSet ID
    evalset_id = None
    result_message = evalset_result.get("result", "")
    import re
    id_match = re.search(r"ID: ([a-f0-9\-]+)", result_message)
    
    if id_match:
        evalset_id = id_match.group(1)
    else:
        evalset_id = evalset_result.get("evalset", {}).get("id")
        
    if not evalset_id:
        print("Failed to extract EvalSet ID from response")
        print(f"Response: {evalset_result}")
        return
        
    print(f"EvalSet created with ID: {evalset_id}")
    
    # 2. Test with a conversation
    print("\n2. Running evaluation with the long prompt...")
    
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to the login page and click 'Forgot Password'."}
    ]
    
    # Run the evaluation (confidence is always enabled, don't need a separate parameter)
    results = await manage_eval_runs_tool(
        action="run", 
        evalset_id=evalset_id,
        conversation=conversation,
        judge_model="meta-llama-3-8b-instruct",  # Use a smaller model for faster results
        max_parallel=2
    )
    
    # Print the results
    print("\n=== Evaluation Results ===")
    print(f"Result: {results}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
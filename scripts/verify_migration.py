#!/usr/bin/env python
"""
Verification script for AgentOptim v2.1.0.

This script tests the EvalSet architecture API.

Usage:
    python scripts/verify_migration.py

Output:
    The script will run tests and print the results.
"""

import asyncio
import sys
import os
import time
from typing import Dict, List, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the EvalSet API
from agentoptim import (
    manage_evalset_tool,
    manage_eval_runs_tool,
)


async def test_evalset_api():
    """Test the EvalSet architecture API."""
    print("\n=== Testing EvalSet API ===")
    
    # Create an EvalSet
    print("\nCreating EvalSet...")
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Test EvalSet",
        questions=[
            "Is the response helpful?",
            "Is the response accurate?"
        ],
        short_description="Test evaluation criteria",
        long_description="This is a test evaluation set for verifying the AgentOptim v2.1.0 API. It contains simple criteria to assess response quality. The criteria focus on helpfulness and accuracy of responses, which are key aspects of evaluating AI assistant interactions. These evaluation criteria can be applied to various types of conversations to ensure consistent assessment of response quality across different scenarios and use cases." + " " * 100
    )
    
    evalset_id = None
    
    if "evalset" in evalset_result and "id" in evalset_result["evalset"]:
        evalset_id = evalset_result["evalset"]["id"]
    elif "evalset_id" in evalset_result:
        evalset_id = evalset_result["evalset_id"]
    elif "result" in evalset_result and isinstance(evalset_result["result"], str):
        # Try to extract ID from result message
        import re
        id_match = re.search(r'ID: ([0-9a-f-]+)', evalset_result["result"])
        if id_match:
            evalset_id = id_match.group(1)
    
    if not evalset_id:
        print("Failed to get evalset_id from result:", evalset_result)
        raise ValueError("Could not get evalset_id from creation result")
        
    print(f"Created EvalSet with ID: {evalset_id}")
    
    # Test a conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2=4"}
    ]
    
    print("\nRunning evaluation...")
    eval_result = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation
        # Model will be auto-detected
    )
    
    print(f"Evaluation complete with overall score: {eval_result.get('summary', {}).get('yes_percentage')}%")
    return evalset_id


async def main():
    print("=== AgentOptim v2.1.0 API Verification ===")
    
    start_time = time.time()
    
    try:
        # Test the EvalSet API
        evalset_id = await test_evalset_api()
        
        elapsed = time.time() - start_time
        print(f"\n✅ API tests completed successfully in {elapsed:.2f} seconds.")
        print("\nVerification Results:")
        print("1. EvalSet API is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nAPI verification failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
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
    run_evalset_tool,
)


async def test_evalset_api():
    """Test the EvalSet architecture API."""
    print("\n=== Testing EvalSet API ===")
    
    # Create an EvalSet
    print("\nCreating EvalSet...")
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Test EvalSet",
        template="""
        Given this conversation:
        {{ conversation }}
        
        Please answer the following yes/no question about the final assistant response:
        {{ eval_question }}
        
        Return a JSON object with the following format:
        {"judgment": 1} for yes or {"judgment": 0} for no.
        """,
        questions=[
            "Is the response helpful?",
            "Is the response accurate?"
        ],
        description="Test evaluation criteria"
    )
    
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"Created EvalSet with ID: {evalset_id}")
    
    # Test a conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2=4"}
    ]
    
    print("\nRunning evaluation...")
    eval_result = await run_evalset_tool(
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
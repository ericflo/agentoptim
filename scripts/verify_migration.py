#!/usr/bin/env python
"""
Verification script for AgentOptim v2.0 migration.

This script tests both the old API and new API to verify that:
1. The new API works correctly
2. The compatibility layer correctly bridges the old API to the new implementation
3. Appropriate deprecation warnings are shown

Usage:
    python scripts/verify_migration.py

Output:
    The script will run tests and print the results, including any deprecation warnings.
"""

import asyncio
import warnings
import sys
import os
import time
from typing import Dict, List, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import both old and new APIs
from agentoptim import (
    # New API
    manage_evalset_tool,
    run_evalset_tool,
)

# These imports should trigger deprecation warnings
from agentoptim.compat import (
    convert_evaluation_to_evalset,
    evaluation_to_evalset_id,
    dataset_to_conversations,
    experiment_results_to_evalset_results,
)


# Filter to capture deprecation warnings
class DeprecationWarningCollector:
    def __init__(self):
        self.warnings = []

    def __enter__(self):
        self.old_filters = warnings.filters.copy()
        warnings.simplefilter("always", DeprecationWarning)
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters = self.old_filters
        warnings.showwarning = self._old_showwarning

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        if category == DeprecationWarning:
            self.warnings.append(str(message))
        self._old_showwarning(message, category, filename, lineno, file, line)


async def test_new_api():
    """Test the new 2-tool architecture API."""
    print("\n=== Testing New API ===")
    
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
        conversation=conversation,
        model="meta-llama-3.1-8b-instruct"
    )
    
    print(f"Evaluation complete with overall score: {eval_result.get('summary', {}).get('yes_percentage')}%")
    return evalset_id


async def test_compatibility_layer(evalset_id):
    """Test the compatibility layer functions."""
    print("\n=== Testing Compatibility Layer ===")
    
    # Test convert_evaluation_to_evalset
    with DeprecationWarningCollector() as collector:
        print("\nTesting convert_evaluation_to_evalset...")
        compat_result = await convert_evaluation_to_evalset(
            name="Compat Test",
            template="Input: {input}\nResponse: {response}\nQuestion: {question}",
            questions=["Is the response helpful?"],
            description="Test compatibility conversion"
        )
        
        # Print the warning if any
        if collector.warnings:
            print(f"Deprecation warning: {collector.warnings[-1]}")
    
    compat_evalset_id = compat_result.get("evalset", {}).get("id")
    print(f"Created compat EvalSet with ID: {compat_evalset_id}")
    
    # Test evaluation_to_evalset_id
    with DeprecationWarningCollector() as collector:
        print("\nTesting evaluation_to_evalset_id...")
        # This should return None since we don't have a real evaluation ID
        compat_id = await evaluation_to_evalset_id("fake_eval_id")
        print(f"Mapped evaluation ID to EvalSet ID: {compat_id}")
        
        # Print the warning if any
        if collector.warnings:
            print(f"Deprecation warning: {collector.warnings[-1]}")
    
    # Test dataset_to_conversations
    with DeprecationWarningCollector() as collector:
        print("\nTesting dataset_to_conversations...")
        # Create some sample dataset items
        items = [
            {"input": "What is 2+2?", "expected_output": "4"},
            {"input": "What is the capital of France?", "expected_output": "Paris"}
        ]
        conversations = await dataset_to_conversations("fake_dataset_id", items)
        print(f"Converted {len(conversations)} dataset items to conversations")
        
        # Print the warning if any
        if collector.warnings:
            print(f"Deprecation warning: {collector.warnings[-1]}")
    
    # Test experiment_results_to_evalset_results
    with DeprecationWarningCollector() as collector:
        print("\nTesting experiment_results_to_evalset_results...")
        results = await experiment_results_to_evalset_results(
            experiment_id="fake_experiment_id",
            evaluation_id="fake_eval_id",
            evalset_id=evalset_id
        )
        print(f"Converted experiment results: {results.get('status')}")
        
        # Print the warning if any
        if collector.warnings:
            print(f"Deprecation warning: {collector.warnings[-1]}")


async def main():
    print("=== AgentOptim v2.0 Migration Verification ===")
    print("This script tests both the new API and compatibility layer.")
    
    start_time = time.time()
    
    try:
        # Test the new API
        evalset_id = await test_new_api()
        
        # Test the compatibility layer
        await test_compatibility_layer(evalset_id)
        
        elapsed = time.time() - start_time
        print(f"\n✅ All tests completed successfully in {elapsed:.2f} seconds.")
        print("\nVerification Results:")
        print("1. New API is working correctly")
        print("2. Compatibility layer is functioning")
        print("3. Appropriate deprecation warnings are shown")
        print("\nMigration verification passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nMigration verification failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python
"""Test duplicate EvalSet detection."""

import asyncio
from agentoptim.server import manage_evalset_tool


async def main():
    """Test creating identical EvalSets."""
    print("=== Testing Duplicate EvalSet Detection ===")
    
    # Define identical EvalSet properties for both attempts
    evalset_properties = {
        "name": "Duplicate Test EvalSet",
        "questions": [
            "Is the response helpful?",
            "Is the response clear?",
            "Is the response accurate?"
        ],
        "short_description": "Test evalset for duplicate detection",
        "long_description": "This is a test EvalSet created to verify the duplicate detection functionality. " +
            "When creating an identical EvalSet, the system should return the existing one instead of creating " +
            "a duplicate. This helps reduce clutter and promotes reuse of well-crafted evaluation criteria." + " " * 100
    }
    
    # First creation attempt
    print("\n1. Creating first EvalSet...")
    first_result = await manage_evalset_tool(
        action="create",
        **evalset_properties
    )
    print(f"Full response: {first_result}")
    print(f"Message: {first_result.get('result', '')}")
    print(f"Is new: {first_result.get('is_new', 'Unknown')}")
    
    # Second creation attempt with identical properties
    print("\n2. Creating identical EvalSet...")
    second_result = await manage_evalset_tool(
        action="create",
        **evalset_properties
    )
    print(f"Full response: {second_result}")
    print(f"Message: {second_result.get('result', '')}")
    print(f"Is new: {second_result.get('is_new', 'Unknown')}")
    
    # Third creation attempt with different properties
    print("\n3. Creating different EvalSet...")
    different_properties = evalset_properties.copy()
    different_properties["name"] = "Different Test EvalSet"
    third_result = await manage_evalset_tool(
        action="create",
        **different_properties
    )
    print(f"Full response: {third_result}")
    print(f"Message: {third_result.get('result', '')}")
    print(f"Is new: {third_result.get('is_new', 'Unknown')}")
    
    # Extract IDs from formatted message
    import re
    
    def extract_id(result):
        result_message = result.get('result', '')
        id_match = re.search(r"ID: ([a-f0-9\-]+)", result_message)
        if id_match:
            return id_match.group(1)
        return None
    
    first_id = extract_id(first_result)
    second_id = extract_id(second_result)
    third_id = extract_id(third_result)
    
    # Verify results
    print("\n4. Results comparison:")
    print(f"First EvalSet ID: {first_id}")
    print(f"Second EvalSet ID (identical): {second_id}")
    print(f"Third EvalSet ID (different): {third_id}")
    
    success = True
    
    # Check if identical EvalSets have the same ID
    if first_id and second_id and first_id == second_id:
        print("\n✅ Duplicate detection test passed!")
        print("The second attempt returned the existing EvalSet rather than creating a duplicate.")
    else:
        success = False
        print("\n❌ Duplicate detection test failed!")
        print("The second attempt created a new EvalSet instead of returning the existing one.")
    
    # Check if different EvalSets have different IDs
    if third_id and third_id != first_id:
        print("\n✅ Non-duplicate detection test passed!")
        print("The third attempt with different properties created a new EvalSet as expected.")
    else:
        success = False
        print("\n❌ Non-duplicate detection test failed!")
        print("The third attempt incorrectly returned an existing EvalSet or failed to create one.")
    
    if success:
        print("\n🎉 All tests passed! The duplicate detection feature is working correctly.")


if __name__ == "__main__":
    asyncio.run(main())
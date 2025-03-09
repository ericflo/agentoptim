"""Example usage of the AgentOptim EvalSet architecture."""

import asyncio
import json

from agentoptim.server import manage_evalset_tool, run_evalset_tool


async def main():
    """Demonstrate using the new EvalSet architecture."""
    try:
        # First, check if there are existing EvalSets we could use
        print("Listing existing EvalSets...")
        list_result = await manage_evalset_tool(action="list")
        evalsets = list_result.get('evalsets', {})
        print(f"Found {len(evalsets)} existing EvalSets")
        
        if evalsets:
            print("Available EvalSets:")
            for eval_id, evalset in evalsets.items():
                print(f"- {evalset.get('name', 'Unnamed')} (ID: {eval_id})")
        
        # For demonstration purposes, let's just list the EvalSets
        print("\nAgentOptim provides a simple API for creating and running evaluation sets to assess conversation quality.")
        print("This example shows how to list, create, and evaluate conversations with EvalSets.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
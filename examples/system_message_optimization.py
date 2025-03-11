"""
Example demonstrating the system message optimization functionality in AgentOptim v2.2.0.

This example shows:
1. Creating an EvalSet for judging system message performance
2. Using the system message optimization tool to generate candidates
3. Examining the optimization results and performance breakdown
4. Using the optimized system message in a conversation

To run this example:
1. Make sure you have AgentOptim v2.2.0+ installed
2. Start the AgentOptim server with: agentoptim server
3. Run this script with: python system_message_optimization.py
"""

import asyncio
import json
from pprint import pprint

from agentoptim.server import manage_evalset_tool, optimize_system_messages_tool


async def main():
    """Demonstrate using the system message optimization functionality."""
    try:
        print("\n=== 1. Creating an EvalSet for System Message Evaluation ===\n")
        
        # Create an evaluation set specifically for judging system messages
        evalset_result = await manage_evalset_tool(
            action="create",
            name="System Message Quality Evaluation",
            questions=[
                "Does the system message result in a response that directly addresses the user's query?",
                "Does the system message help produce a concise response without unnecessary information?",
                "Does the system message maintain an appropriate tone for this type of query?",
                "Does the system message guide the model to provide accurate information?",
                "Does the system message avoid introducing bias or unnecessary limitations?"
            ],
            short_description="System message quality evaluation",
            long_description="This EvalSet measures how well a system message guides an AI to generate helpful, accurate, concise, and appropriate responses to user queries. It evaluates directness, conciseness, tone, accuracy, and lack of bias."
        )
        
        # Extract the EvalSet ID
        evalset_id = evalset_result["evalset"]["id"]
        print(f"Created EvalSet with ID: {evalset_id}")
        
        print("\n=== 2. Optimizing System Messages for a User Query ===\n")
        
        # Define the user query we want to optimize for
        user_query = "I'm having trouble setting up two-factor authentication on my account. Can you help me?"
        
        print(f"Optimizing system messages for query: '{user_query}'")
        
        # Run the optimization process
        optimization_result = await optimize_system_messages_tool(
            action="optimize",
            user_query=user_query,
            num_candidates=3,  # Generate 3 candidate system messages
            evalset_id=evalset_id,
            domain="customer_support",  # Specialize for customer support
            base_system_message="You are a helpful assistant."  # Optional starting point
        )
        
        # Extract the run ID for later reference
        optimization_run_id = optimization_result["id"]
        print(f"Optimization completed with run ID: {optimization_run_id}")
        
        print("\n=== 3. Examining Optimization Results ===\n")
        
        # Get the top candidate
        top_candidate = optimization_result["candidates"][0]
        
        print("Best System Message:")
        print("-" * 80)
        print(top_candidate["system_message"])
        print("-" * 80)
        
        print("\nPerformance Breakdown:")
        for criterion, score in top_candidate["performance"]["criterion_scores"].items():
            print(f"- {criterion}: {score:.2f}")
        
        print(f"\nOverall Score: {top_candidate['performance']['overall_score']:.2f}")
        print(f"Confidence: {top_candidate['performance']['confidence']:.2f}")
        
        print("\n=== 4. Retrieving Past Optimization Results ===\n")
        
        # Retrieve the optimization run we just created
        retrieved_result = await optimize_system_messages_tool(
            action="get",
            optimization_run_id=optimization_run_id
        )
        
        print(f"Retrieved optimization run with ID: {retrieved_result['id']}")
        print(f"Created at: {retrieved_result['created_at']}")
        print(f"Number of candidates: {len(retrieved_result['candidates'])}")
        
        print("\n=== 5. Listing All Optimization Runs ===\n")
        
        # List all optimization runs
        list_result = await optimize_system_messages_tool(
            action="list",
            page=1,
            page_size=10
        )
        
        print(f"Found {list_result['total']} optimization runs")
        for run in list_result["optimization_runs"]:
            print(f"- {run['id']}: '{run['user_query']}' ({run['created_at']})")
        
        print("\nSystem Message Optimization Example Complete!")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
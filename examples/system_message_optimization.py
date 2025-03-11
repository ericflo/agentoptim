"""
Example of using the system message optimization tool.

This example demonstrates how to:
1. Create an EvalSet for response quality
2. Optimize system messages for a user query 
3. Retrieve and review the optimization results
4. Compare different optimized system messages
"""

import asyncio
import json
from agentoptim import manage_evalset_tool, optimize_system_messages_tool

async def main():
    print("🚀 System Message Optimization Example")
    print("=" * 80)
    print()
    
    # Create an EvalSet for general response quality
    print("Creating EvalSet for response quality evaluation...")
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Response Quality Evaluation",
        questions=[
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Is the response complete, providing all necessary information?",
            "Is the response accurate and factually correct?",
            "Is the tone of the response appropriate for the query?",
            "Would the response likely satisfy the user's needs?",
            "Does the response avoid unnecessary information?",
            "Does the response have a logical structure and flow?",
            "Is the response free of problematic biases?",
            "Does the response strike a good balance between brevity and completeness?"
        ],
        short_description="Comprehensive response quality evaluation criteria",
        long_description="This EvalSet provides comprehensive evaluation criteria for assessing the quality of AI responses. It measures clarity, completeness, accuracy, helpfulness, and appropriateness of tone. Use it to evaluate and optimize system messages for improved user experience and satisfaction." + " " * 50
    )
    
    # Get the EvalSet ID
    evalset_id = evalset_result["evalset"]["id"]
    print(f"Created EvalSet with ID: {evalset_id}")
    print()
    
    # Define a user question to optimize for
    user_message = "What are the key components of a balanced diet, and how should I adjust my eating habits to improve my overall health?"
    print(f"User message: {user_message}")
    print()
    
    # Run system message optimization
    print("Optimizing system messages...")
    optimization_result = await optimize_system_messages_tool(
        action="optimize",
        user_message=user_message,
        evalset_id=evalset_id,
        num_candidates=3,  # Generate 3 candidates for this example
        diversity_level="high",
        additional_instructions="Focus on creating system messages that help the assistant provide actionable, science-based nutrition advice without overwhelming the user with too much information."
    )
    
    # Get the optimization run ID
    optimization_run_id = optimization_result["id"]
    print(f"Optimization completed with ID: {optimization_run_id}")
    print()
    
    # Print the best system message
    print("🏆 Best System Message:")
    print("-" * 80)
    print(optimization_result["best_system_message"])
    print("-" * 80)
    print(f"Score: {optimization_result['best_score']}%")
    print()
    
    # Retrieve the optimization run
    print("Retrieving optimization details...")
    optimization_details = await optimize_system_messages_tool(
        action="get",
        optimization_run_id=optimization_run_id
    )
    
    # Print all candidate scores
    print("\nAll Candidate Scores:")
    candidates = optimization_details["optimization_run"]["candidates"]
    for i, candidate in enumerate(candidates, 1):
        print(f"Candidate {i}: {candidate['score']}%")
    
    # Now run another optimization with a different user message
    print("\n" + "=" * 80)
    print("Running another optimization for a different query...")
    
    user_message_2 = "How do I troubleshoot a slow internet connection?"
    optimization_result_2 = await optimize_system_messages_tool(
        action="optimize",
        user_message=user_message_2,
        evalset_id=evalset_id,
        num_candidates=3
    )
    
    print(f"Second optimization completed with ID: {optimization_result_2['id']}")
    
    # List all optimization runs
    print("\nListing all optimization runs:")
    all_runs = await optimize_system_messages_tool(
        action="list",
        page=1,
        page_size=10
    )
    
    print(f"Found {all_runs['pagination']['total_count']} optimization runs:")
    for run in all_runs["optimization_runs"]:
        print(f"- {run['id']}: {run['user_message'][:40]}...")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    asyncio.run(main())
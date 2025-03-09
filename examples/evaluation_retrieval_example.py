#!/usr/bin/env python3
"""
AgentOptim: Evaluation Storage and Retrieval Example

This example demonstrates how to:
1. Create an EvalSet for conversation evaluation
2. Run an evaluation and store the results
3. Retrieve past evaluation results by ID
4. List all evaluation runs with pagination
"""

import asyncio
import json
import sys
from datetime import datetime
from pprint import pprint

# Import AgentOptim tools
from agentoptim import manage_evalset_tool, manage_eval_runs_tool


async def main():
    """Run the evaluation storage and retrieval example."""
    print("\n✨ AgentOptim Evaluation Storage and Retrieval Example ✨\n")
    
    # 1️⃣ First, check if we have an appropriate EvalSet or create a new one
    existing_sets = await manage_evalset_tool(action="list")
    
    # Find an EvalSet for response quality evaluation or create a new one
    evalset_id = None
    for evalset_data in existing_sets.get("evalsets", {}).values():
        if "response quality" in evalset_data.get("name", "").lower():
            evalset_id = evalset_data["id"]
            print(f"Found existing EvalSet: {evalset_data['name']} (ID: {evalset_id})")
            break
    
    if not evalset_id:
        # Create a new EvalSet for response quality evaluation
        print("Creating new EvalSet for response quality evaluation...")
        result = await manage_evalset_tool(
            action="create",
            name="Response Quality Evaluation",
            questions=[
                "Does the response directly address the user's question?",
                "Is the response clear and concise?",
                "Is the response accurate and factually correct?",
                "Is the response well-structured and organized?",
                "Does the response provide sufficient detail?",
                "Is the tone appropriate and helpful?"
            ],
            short_description="General response quality evaluation criteria",
            long_description="This EvalSet provides a comprehensive framework for evaluating the quality of AI assistant responses. It measures clarity, accuracy, organization, completeness, and tone. Use it to evaluate assistant responses to general knowledge questions and requests. Higher scores indicate responses that are clear, accurate, well-organized, complete, and helpful in tone." + " " * 50
        )
        
        evalset_id = result["evalset"]["id"]
        print(f"Created new EvalSet with ID: {evalset_id}")
    
    # 2️⃣ Define a conversation to evaluate
    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the benefits of regular exercise?"},
        {"role": "assistant", "content": """Regular exercise offers numerous benefits:

1. Physical health: Improves cardiovascular health, builds strength, enhances flexibility, and helps maintain healthy weight.

2. Mental health: Reduces stress, anxiety, and depression while improving mood through endorphin release.

3. Cognitive function: Enhances memory, focus, and overall brain function.

4. Sleep quality: Promotes better, deeper sleep patterns.

5. Energy levels: Increases stamina and reduces fatigue.

6. Longevity: Associated with longer lifespan and healthier aging.

7. Disease prevention: Reduces risk of many chronic conditions including heart disease, diabetes, and some cancers.

For optimal benefits, aim for at least 150 minutes of moderate activity weekly, combining cardiovascular exercise with strength training and flexibility work."""}
    ]
    
    # 3️⃣ Run the evaluation and save the results
    print("\nRunning evaluation and saving results...")
    eval_result = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation
    )
    
    # Save the evaluation run ID
    eval_run_id = eval_result["id"]
    print(f"Evaluation completed with ID: {eval_run_id}")
    print(f"Overall score: {eval_result['summary']['yes_percentage']}%")
    
    # 4️⃣ Retrieve the evaluation results by ID
    print("\nRetrieving evaluation by ID...")
    retrieved_eval = await manage_eval_runs_tool(
        action="get",
        eval_run_id=eval_run_id
    )
    
    # Display basic information about the retrieved evaluation
    eval_data = retrieved_eval["eval_run"]
    timestamp = datetime.fromtimestamp(eval_data["timestamp"])
    print(f"Retrieved evaluation from: {timestamp}")
    print(f"EvalSet: {eval_data['evalset_name']}")
    print(f"Score: {eval_data['summary']['yes_percentage']}%")
    print(f"Judge model: {eval_data['judge_model'] or 'auto-detected'}")
    
    # 5️⃣ List all evaluation runs (paginated)
    print("\nListing all evaluation runs (page 1)...")
    all_runs = await manage_eval_runs_tool(
        action="list",
        page=1,
        page_size=5  # Show 5 runs per page
    )
    
    # Display pagination information
    pagination = all_runs["pagination"]
    print(f"Found {pagination['total_count']} evaluation runs in {pagination['total_pages']} pages")
    
    # Display a simple table of runs
    print("\nRecent evaluations:")
    print("------------------------------------")
    print("ID (truncated) | EvalSet | Date | Score")
    print("------------------------------------")
    for run in all_runs["eval_runs"]:
        run_id = run["id"][:8] + "..."  # Truncate ID for display
        evalset_name = run["evalset_name"][:15] + "..." if len(run["evalset_name"]) > 15 else run["evalset_name"]
        date = run["timestamp_formatted"]
        score = f"{run['summary'].get('yes_percentage', 0)}%" if "summary" in run else "N/A"
        print(f"{run_id} | {evalset_name} | {date} | {score}")
    
    # 6️⃣ List evaluations filtered by EvalSet ID
    print("\nListing evaluations for specific EvalSet...")
    filtered_runs = await manage_eval_runs_tool(
        action="list",
        evalset_id=evalset_id,
        page=1,
        page_size=5
    )
    
    # Display filtered results
    filtered_count = filtered_runs["pagination"]["total_count"]
    print(f"Found {filtered_count} evaluations for EvalSet ID: {evalset_id}")
    if filtered_count > 0:
        print("Most recent evaluation:")
        most_recent = filtered_runs["eval_runs"][0]
        timestamp = most_recent["timestamp_formatted"]
        score = most_recent["summary"].get("yes_percentage", 0) if "summary" in most_recent else "N/A"
        print(f"- ID: {most_recent['id']}")
        print(f"- Date: {timestamp}")
        print(f"- Score: {score}%")
    
    print("\n✅ Example completed successfully!")
    

if __name__ == "__main__":
    asyncio.run(main())
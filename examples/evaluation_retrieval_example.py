#!/usr/bin/env python
"""
Example showing how to use the evaluation storage functionality.

This example demonstrates:
1. Creating an EvalSet
2. Running an evaluation and storing the result
3. Retrieving a past evaluation by ID
4. Listing all evaluation runs with pagination
5. Filtering evaluation runs by EvalSet ID
"""

import asyncio
import sys
from typing import Dict, List, Any
from agentoptim import manage_evalset_tool, manage_eval_runs_tool


async def create_evalset() -> Dict[str, Any]:
    """Create an example EvalSet for evaluating response quality."""
    result = await manage_evalset_tool(
        action="create",
        name="Response Quality Evaluation",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Is the response accurate?",
            "Does the response provide complete information?"
        ],
        short_description="Basic quality evaluation criteria",
        long_description="This EvalSet provides comprehensive evaluation criteria for assistant responses. "
                         "It measures helpfulness, directness, clarity, accuracy, and completeness. "
                         "Use it to evaluate the quality of responses to general questions and requests. "
                         "High scores indicate responses that effectively address user needs with accurate "
                         "and complete information presented in a clear, understandable way." + " " * 50
    )
    
    print(f"✅ Created EvalSet: {result['evalset']['name']}")
    print(f"   ID: {result['evalset']['id']}")
    return result['evalset']


async def run_evaluation(evalset_id: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Run an evaluation and store the result."""
    print(f"\n📊 Running evaluation with EvalSet: {evalset_id}...")
    
    result = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation,
        max_parallel=2  # Run up to 2 evaluations in parallel
    )
    
    print(f"✅ Evaluation complete!")
    print(f"   Run ID: {result['id']}")
    print(f"   Score: {result['summary']['yes_percentage']}%")
    print(f"   Questions: {result['summary']['total_questions']}")
    print(f"   Yes answers: {result['summary']['yes_count']}")
    print(f"   No answers: {result['summary']['no_count']}")
    
    return result


async def get_evaluation(eval_run_id: str) -> Dict[str, Any]:
    """Retrieve a past evaluation by ID."""
    print(f"\n🔍 Retrieving evaluation with ID: {eval_run_id}...")
    
    result = await manage_eval_runs_tool(
        action="get",
        eval_run_id=eval_run_id
    )
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    
    print(f"✅ Retrieved evaluation!")
    print(f"   EvalSet: {result['eval_run']['evalset_name']}")
    print(f"   Time: {result['eval_run']['timestamp_formatted']}")
    print(f"   Score: {result['eval_run']['summary']['yes_percentage']}%")
    
    return result['eval_run']


async def list_evaluations(page: int = 1, page_size: int = 5, evalset_id: str = None) -> Dict[str, Any]:
    """List evaluation runs with pagination and optional filtering."""
    filter_text = f" for EvalSet {evalset_id}" if evalset_id else ""
    print(f"\n📋 Listing evaluation runs{filter_text} (page {page}, {page_size} per page)...")
    
    params = {
        "action": "list",
        "page": page,
        "page_size": page_size
    }
    
    if evalset_id:
        params["evalset_id"] = evalset_id
    
    result = await manage_eval_runs_tool(**params)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    
    runs = result['eval_runs']
    pagination = result['pagination']
    
    print(f"✅ Found {pagination['total_count']} evaluation runs")
    print(f"   Page {pagination['page']} of {pagination['total_pages']}")
    
    # Print a table of results
    print("\n┌───────────────────────────────────────────────────────────────┐")
    print("│ ID                                     │ Score │ Date          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    for run in runs:
        run_id = run['id'][:8] + "..." + run['id'][-8:]  # Truncate ID for display
        score = f"{run['summary']['yes_percentage']:.1f}%".ljust(5)
        date = run['timestamp_formatted'][:10]  # Just the date part
        print(f"│ {run_id} │ {score} │ {date} │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Show navigation options
    if pagination['has_prev']:
        print(f"◀ Previous page: page={pagination['prev_page']}")
    if pagination['has_next']:
        print(f"▶ Next page: page={pagination['next_page']}")
    
    return result


async def run_demo():
    """Run the full demo."""
    print("🚀 AgentOptim Evaluation Storage Demo")
    print("======================================")
    
    # Step 1: Create an EvalSet
    evalset = await create_evalset()
    evalset_id = evalset['id']
    
    # Step 2: Define some conversations to evaluate
    good_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password. If you don't receive the email within a few minutes, please check your spam folder. Let me know if you need any further assistance!"}
    ]
    
    mediocre_conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "Click on Forgot Password."}
    ]
    
    # Step 3: Run evaluations
    good_result = await run_evaluation(evalset_id, good_conversation)
    mediocre_result = await run_evaluation(evalset_id, mediocre_conversation)
    
    # Step 4: Retrieve a past evaluation
    retrieved_eval = await get_evaluation(good_result['id'])
    
    # Step 5: List all evaluations
    await list_evaluations()
    
    # Step 6: List evaluations filtered by EvalSet
    await list_evaluations(evalset_id=evalset_id)
    
    print("\n🎉 Demo completed!")
    print("This example showed how to:")
    print("1. Create an EvalSet")
    print("2. Run evaluations and store the results")
    print("3. Retrieve a past evaluation by ID")
    print("4. List all evaluation runs with pagination")
    print("5. Filter evaluation runs by EvalSet ID")


if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted.")
        sys.exit(1)
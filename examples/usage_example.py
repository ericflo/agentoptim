"""Example usage of the AgentOptim EvalSet architecture."""

import asyncio
import json
import uuid

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
        
        # Create a simple EvalSet for conversation quality assessment
        print("\nCreating a new EvalSet...")
        evalset_id = str(uuid.uuid4())
        create_result = await manage_evalset_tool(
            action="create",
            evalset_id=evalset_id,
            name="Basic Helpfulness Evaluation",
            short_description="Evaluates the helpfulness of assistant responses",
            long_description="This EvalSet measures how well an assistant responds to user queries by assessing helpfulness, accuracy, and clarity. It is designed to evaluate the quality of responses in a conversational context, focusing on whether the assistant provides information that is not only accurate but also presented in a way that directly addresses the user's needs. The evaluation considers both the factual correctness of the information provided and the manner in which it is communicated. This comprehensive assessment helps identify strengths and weaknesses in conversational AI systems and provides insights for improving response quality." + " " * 50,
            questions=[
                "How helpful was the assistant's response to the user query?",
                "Was the assistant's response accurate and factually correct?",
                "Did the assistant provide a clear and concise answer?"
            ]
        )
        
        # Check if there was an error in creation
        if 'error' in create_result and create_result['error']:
            print(f"Error creating EvalSet: {create_result.get('message', 'Unknown error')}")
            return
        
        # Extract the actual evalset_id from the result
        if 'evalset_id' in create_result:
            evalset_id = create_result['evalset_id']
        elif 'evalset' in create_result and 'id' in create_result['evalset']:
            evalset_id = create_result['evalset']['id']
        elif 'result' in create_result and isinstance(create_result['result'], str):
            # Try to extract the ID from the result message
            import re
            id_match = re.search(r'ID: ([0-9a-f-]+)', create_result['result'])
            if id_match:
                evalset_id = id_match.group(1)
                print(f"Extracted ID from result message: {evalset_id}")
        
        # Let's verify the EvalSet exists by trying to get it
        get_result = await manage_evalset_tool(
            action="get",
            evalset_id=evalset_id
        )
        
        # Check if the get operation returned an error
        if 'error' in get_result and get_result['error']:
            print(f"Warning: EvalSet verification failed: {get_result.get('message', 'No error message')}")
        elif 'result' in get_result and isinstance(get_result['result'], str) and 'EvalSet' in get_result['result']:
            print(f"Verified EvalSet exists: {evalset_id}")
        else:
            print(f"Verification result: {json.dumps(get_result, indent=2)}")
            
        print(f"Created EvalSet with ID: {evalset_id}")
        
        # Run an evaluation using the created EvalSet
        print("\nRunning an evaluation...")
        
        # Sample conversation to evaluate
        conversation = [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."}
        ]
        
        # Run the evaluation
        eval_result = await run_evalset_tool(
            evalset_id=evalset_id,
            conversation=conversation,
            max_parallel=3  # Optionally control parallelism for speed
        )
        
        # Display the evaluation results
        print("\nEvaluation Results:")
        
        # Check for error first
        if eval_result.get('error'):
            print(f"Error: {eval_result.get('message', 'Unknown error')}")
        else:
            # Format 1: v2.1.0 structure with results list
            results = eval_result.get('results', [])
            if results and isinstance(results, list):
                print("\nResults for each question:")
                for i, result in enumerate(results):
                    print(f"\nQuestion {i+1}: {result.get('question', 'Unknown')}")
                    print(f"Judgment: {'Yes' if result.get('judgment', False) else 'No'}")
                    print(f"Confidence: {result.get('confidence', 0)}")
                    if 'reasoning' in result:
                        print(f"Reasoning: {result.get('reasoning', '')}")
                        
            # Format 2: Summary statistics
            summary = eval_result.get('summary', {})
            if summary:
                print("\nSummary:")
                print(f"Total questions: {summary.get('total_questions', 0)}")
                print(f"Yes percentage: {summary.get('yes_percentage', 0)}%")
                print(f"Mean confidence: {summary.get('mean_confidence', 0)}")
                    
            # Format 3: Check if there's a formatted result field
            formatted_result = eval_result.get('result', '')
            if formatted_result:
                print("\nFormatted Result:")
                print(formatted_result)
        
        print("\nAgentOptim provides a simple API for creating and running evaluation sets to assess conversation quality.")
        print("This example showed how to list, create, and evaluate conversations with EvalSets.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
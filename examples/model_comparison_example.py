#!/usr/bin/env python
"""
Model Comparison Example for AgentOptim

This example demonstrates how to use AgentOptim to compare responses from different
large language models on the same input queries. This is useful for benchmarking
model performance across various tasks and determining which model provides the
best responses for specific use cases.

Usage:
    python model_comparison_example.py
"""

import asyncio
import os
import json
import pandas as pd
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from agentoptim.server import manage_evalset_tool, run_evalset_tool


# Configure these with your actual API credentials if needed
# os.environ["OPENAI_API_KEY"] = "your_key_here"  # For OpenAI models

# Sample questions to ask each model
QUESTIONS = [
    "Explain the concept of quantum entanglement in simple terms.",
    "What steps should I take to improve my public speaking skills?",
    "How do I propagate a monstera plant from cuttings?",
    "Explain the differences between classical and operant conditioning.",
    "What are three strategies for managing a remote software development team?"
]

# Models to compare
MODELS = [
    "meta-llama-3.1-8b-instruct",  # Default model
    "meta-llama-3.1-70b-instruct",  # More capable model
    # Uncomment and add your own models if you have API access
    # "gpt-4o-mini",
    # "claude-3-haiku-20240307",
]

# The temperature setting for each model
TEMPERATURE = 0.7


async def create_evaluation_set():
    """Create an EvalSet for evaluating model responses."""
    
    # First check if our evalset already exists
    existing_sets = await manage_evalset_tool(action="list")
    
    for evalset in existing_sets.get("evalsets", []):
        if evalset["name"] == "Response Quality Evaluation":
            print(f"Using existing EvalSet: {evalset['id']}")
            return evalset["id"]
    
    # Create a new EvalSet for evaluating model quality
    result = await manage_evalset_tool(
        action="create",
        name="Response Quality Evaluation",
        questions=[
            "Is the response accurate and factually correct?",
            "Is the response clear and easy to understand?",
            "Is the response well-structured and organized?",
            "Is the response comprehensive and complete?",
            "Is the response helpful for the user's needs?",
            "Is the response concise without unnecessary information?",
            "Does the response avoid any misleading statements?"
        ],
        short_description="Evaluation of model response quality",
        long_description="This EvalSet measures the quality of model responses across multiple dimensions including accuracy, clarity, organization, completeness, helpfulness, conciseness, and truthfulness. It's designed to compare responses from different LLMs on the same input."
    )
    
    print(f"Created new EvalSet: {result['evalset']['id']}")
    return result["evalset"]["id"]


async def get_model_response(model: str, question: str, temperature: float = 0.7) -> str:
    """
    Get a response from a specified model.
    
    This function should be replaced with actual calls to your LLM APIs.
    It's currently implemented to work with one API endpoint using the model name parameter.
    """
    # Create a conversation with just the user's question
    conversation = [
        {"role": "user", "content": question}
    ]
    
    # For simplicity, we're using the AgentOptim API endpoint, but you could replace this
    # with a direct call to various LLM APIs (OpenAI, Anthropic, etc.)
    try:
        # Here we're using a simple approach - just querying the model through
        # the LM Studio API that AgentOptim uses
        from agentoptim.runner import call_llm_api
        
        response = await call_llm_api(
            messages=conversation,
            model=model,
            temperature=temperature
        )
        
        if "error" in response:
            print(f"Error getting response from {model}: {response['error']}")
            return f"Error: {response['error']}"
        
        # Extract the content from the response
        if "choices" in response and len(response["choices"]) > 0:
            if "message" in response["choices"][0]:
                return response["choices"][0]["message"]["content"]
            elif "text" in response["choices"][0]:
                return response["choices"][0]["text"]
        
        return "No valid response received"
    
    except Exception as e:
        print(f"Exception when calling {model}: {str(e)}")
        return f"Error: {str(e)}"


async def evaluate_model_response(evalset_id: str, question: str, response: str) -> Dict[str, Any]:
    """Evaluate a model's response using the evaluation set."""
    
    # Create a conversation with the user's question and the model's response
    conversation = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response}
    ]
    
    # Run the evaluation
    result = await run_evalset_tool(
        evalset_id=evalset_id,
        conversation=conversation,
        max_parallel=3
    )
    
    return result


async def run_model_comparison():
    """Run the comparison between different models."""
    # Create or get evaluation set
    evalset_id = await create_evaluation_set()
    
    # Store all results
    all_results = []
    
    # Process each question
    for question_idx, question in enumerate(QUESTIONS):
        print(f"\nProcessing question {question_idx + 1}/{len(QUESTIONS)}:")
        print(f"Q: {question}")
        
        # Get and evaluate responses for each model
        for model in MODELS:
            print(f"  Getting response from {model}...")
            response = await get_model_response(model, question, TEMPERATURE)
            
            print(f"  Evaluating response from {model}...")
            eval_result = await evaluate_model_response(evalset_id, question, response)
            
            # Store the results
            all_results.append({
                "question": question,
                "model": model,
                "response": response,
                "evaluation": eval_result
            })
            
            # Print a brief summary
            summary = eval_result["summary"]
            print(f"  {model} score: {summary['yes_percentage']}% "
                  f"({summary['yes_count']}/{summary['total_questions']} criteria)")
    
    return all_results


def generate_reports(results: List[Dict[str, Any]]):
    """Generate comparative reports from the results."""
    # Convert results to a structured DataFrame
    data = []
    
    for result in results:
        model = result["model"]
        question = result["question"]
        response = result["response"]
        evaluation = result["evaluation"]
        
        summary = evaluation["summary"]
        yes_percentage = summary["yes_percentage"]
        yes_count = summary["yes_count"]
        total_questions = summary["total_questions"]
        
        # Extract confidence if available
        mean_confidence = summary.get("mean_confidence", None)
        
        # Add to our data collection
        data.append({
            "model": model,
            "question": question,
            "response": response,
            "yes_percentage": yes_percentage,
            "yes_count": yes_count,
            "total_questions": total_questions,
            "mean_confidence": mean_confidence
        })
    
    # Create a DataFrame for analysis
    df = pd.DataFrame(data)
    
    # Calculate average scores per model
    model_scores = df.groupby("model")[["yes_percentage", "mean_confidence"]].mean().reset_index()
    
    print("\n===== MODEL COMPARISON RESULTS =====")
    print("\nAverage scores by model:")
    for _, row in model_scores.iterrows():
        confidence_str = f", Confidence: {row['mean_confidence']:.2f}" if pd.notna(row["mean_confidence"]) else ""
        print(f"{row['model']}: {row['yes_percentage']:.2f}%{confidence_str}")
    
    # Calculate scores by question type
    question_scores = df.groupby(["question", "model"])["yes_percentage"].mean().unstack().reset_index()
    
    print("\nScores by question:")
    for _, row in question_scores.iterrows():
        print(f"\nQ: {row['question']}")
        for model in MODELS:
            if model in row:
                print(f"  {model}: {row[model]:.2f}%")
    
    # Create a visualization
    try:
        plt.figure(figsize=(10, 6))
        
        # Bar chart of average model scores
        plt.bar(model_scores["model"], model_scores["yes_percentage"])
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
        plt.title("Model Comparison: Average Quality Scores")
        plt.xlabel("Model")
        plt.ylabel("Quality Score (%)")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig("model_comparison_results.png")
        print("\nVisualization saved as 'model_comparison_results.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Save detailed results to a JSON file
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Detailed results saved to 'model_comparison_results.json'")


async def main():
    """Main function to run the example."""
    print("Starting Model Comparison Example")
    print("=================================")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"Models: {', '.join(MODELS)}")
    print("=================================\n")
    
    results = await run_model_comparison()
    generate_reports(results)


if __name__ == "__main__":
    asyncio.run(main())
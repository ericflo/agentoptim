#!/usr/bin/env python
"""
Example script demonstrating how to analyze experiment results.

This script shows how to:
1. Create a simple experiment with multiple prompt variants
2. Run a job to gather results 
3. Analyze the results to identify the best performing variant
4. Compare multiple analyses
"""

import os
import sys
import asyncio
import random
from pprint import pprint

# Add the parent directory to the path so we can import agentoptim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agentoptim.dataset import manage_dataset
from agentoptim.evaluation import manage_evaluation
from agentoptim.experiment import manage_experiment
from agentoptim.jobs import manage_job
from agentoptim.analysis import analyze_results


async def main():
    """Run the example."""
    print("AgentOptim Experiment Analysis Example")
    print("-------------------------------------")
    
    # 1. Create a sample dataset
    print("\n1. Creating a sample dataset...")
    
    dataset_items = []
    for i in range(10):
        dataset_items.append({
            "id": f"item{i}",
            "question": f"Sample question {i}?",
            "context": f"This is sample context for question {i}."
        })
    
    dataset_response = manage_dataset(
        action="create",
        name="Sample Dataset",
        description="A sample dataset for testing",
        items=dataset_items
    )
    print(dataset_response)
    dataset_id = dataset_response.split("ID: ")[1].split()[0]
    
    # 2. Create an evaluation
    print("\n2. Creating an evaluation...")
    
    evaluation_response = manage_evaluation(
        action="create",
        name="Clarity and Accuracy",
        description="Evaluates responses for clarity and accuracy",
        template="""
            Evaluate the following response to the given question and context.
            
            Question: {question}
            Context: {context}
            Response: {response}
            
            For the criterion: {criterion}
            
            Rate on a scale of 1-5 where 1 is very poor and 5 is excellent.
            Provide only a single number as your rating.
        """,
        questions=[
            "How clear and understandable is the response?",
            "How accurate is the response given the context?"
        ]
    )
    print(evaluation_response)
    evaluation_id = evaluation_response.split("ID: ")[1].split()[0]
    
    # 3. Create an experiment with multiple prompt variants
    print("\n3. Creating an experiment with multiple prompt variants...")
    
    prompt_variants = [
        {
            "name": "Simple Instructions",
            "type": "system",
            "template": """
                Answer the following question based on the provided context.
                
                Context: {context}
                Question: {question}
            """
        },
        {
            "name": "Detailed Instructions",
            "type": "system",
            "template": """
                You are an expert assistant. Answer the following question based 
                only on the information in the provided context. Be clear, concise,
                and accurate. If the answer cannot be determined from the context,
                say so.
                
                Context: {context}
                Question: {question}
            """
        },
        {
            "name": "Step-by-Step Instructions",
            "type": "system",
            "template": """
                Follow these steps to answer the question:
                1. Carefully read the context provided
                2. Identify key information related to the question
                3. Formulate a clear and concise answer
                4. Double-check your answer against the context
                
                Context: {context}
                Question: {question}
            """
        }
    ]
    
    experiment_response = manage_experiment(
        action="create",
        name="Instruction Styles Experiment",
        description="Testing different instruction styles",
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        prompt_variants=prompt_variants,
        model_name="test-model",
        temperature=0.0
    )
    print(experiment_response)
    experiment_id = experiment_response.split("ID: ")[1].split(")")[0].split()[0]
    
    # 4. Create and run a job
    print("\n4. Creating and running a job...")
    
    job_command = manage_job(
        action="create",
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        judge_model="mock-model"
    )
    job_id = job_command["job"]["job_id"]
    print(f"Job created with ID: {job_id}")
    
    # For this example, we'll simulate job results since we're using a mock model
    # In a real application, you would run the job with:
    # manage_job(action="run", job_id=job_id)
    
    # Simulate running the job
    print("Simulating job execution...")
    
    # Instead of actually running the job, we'll manually update the experiment
    # with simulated results
    
    # Create simulated results
    simulated_results = {}
    
    # Simple instructions variant - decent clarity, lower accuracy
    simulated_results["var1"] = {
        "scores": {
            "clarity": [3.8, 4.0, 3.9, 4.1, 3.7, 3.6, 3.9, 4.0, 3.8, 3.7],
            "accuracy": [3.2, 3.5, 3.3, 3.4, 3.6, 3.1, 3.3, 3.4, 3.2, 3.5]
        }
    }
    
    # Detailed instructions variant - good clarity, good accuracy
    simulated_results["var2"] = {
        "scores": {
            "clarity": [4.1, 4.3, 4.2, 4.0, 3.9, 4.2, 4.1, 4.3, 4.0, 4.2],
            "accuracy": [4.3, 4.5, 4.4, 4.2, 4.3, 4.1, 4.4, 4.3, 4.2, 4.5]
        }
    }
    
    # Step-by-step instructions variant - excellent clarity, very good accuracy
    simulated_results["var3"] = {
        "scores": {
            "clarity": [4.5, 4.7, 4.6, 4.8, 4.7, 4.6, 4.5, 4.7, 4.6, 4.5],
            "accuracy": [4.0, 4.2, 4.1, 4.3, 4.0, 4.2, 4.1, 4.0, 4.3, 4.2]
        }
    }
    
    # Update the experiment with the results
    manage_experiment(
        action="update",
        experiment_id=experiment_id,
        results=simulated_results,
        status="completed"
    )
    
    # 5. Analyze the results
    print("\n5. Analyzing the experiment results...")
    
    analysis_response = analyze_results(
        action="analyze",
        experiment_id=experiment_id,
        name="Instruction Styles Analysis",
        description="Analysis of different instruction styles"
    )
    print(analysis_response)
    analysis_id = analysis_response.split("ID: ")[1].split("\n")[0]
    
    # 6. Get the analysis details
    print("\n6. Getting the analysis details...")
    
    analysis_details = analyze_results(
        action="get",
        analysis_id=analysis_id
    )
    print(analysis_details)
    
    # 7. Create a second experiment with different variants for comparison
    print("\n7. Creating a second experiment for comparison...")
    
    prompt_variants_2 = [
        {
            "name": "Formal Style",
            "type": "system",
            "template": """
                Please provide a comprehensive answer to the following inquiry,
                utilizing the contextual information provided below.
                
                Context: {context}
                Inquiry: {question}
            """
        },
        {
            "name": "Conversational Style",
            "type": "system",
            "template": """
                Hey there! I'd like you to help answer this question based on 
                the information I'm giving you. Keep it friendly and easy to understand.
                
                Here's the background info: {context}
                And here's what I'm wondering: {question}
            """
        }
    ]
    
    experiment_response_2 = manage_experiment(
        action="create",
        name="Communication Styles Experiment",
        description="Testing different communication styles",
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        prompt_variants=prompt_variants_2,
        model_name="test-model",
        temperature=0.0
    )
    print(experiment_response_2)
    experiment_id_2 = experiment_response_2.split("ID: ")[1].split(")")[0].split()[0]
    
    # Simulate results for second experiment
    simulated_results_2 = {}
    
    # Formal style - good clarity, very good accuracy
    simulated_results_2["var1"] = {
        "scores": {
            "clarity": [4.0, 4.2, 3.9, 4.1, 4.0, 3.9, 4.1, 4.0, 4.2, 4.1],
            "accuracy": [4.4, 4.5, 4.3, 4.6, 4.4, 4.5, 4.3, 4.4, 4.5, 4.3]
        }
    }
    
    # Conversational style - very good clarity, good accuracy
    simulated_results_2["var2"] = {
        "scores": {
            "clarity": [4.3, 4.5, 4.4, 4.6, 4.3, 4.5, 4.2, 4.4, 4.3, 4.5],
            "accuracy": [3.8, 4.0, 3.9, 4.1, 3.8, 3.9, 4.0, 3.9, 4.0, 3.8]
        }
    }
    
    # Update the second experiment
    manage_experiment(
        action="update",
        experiment_id=experiment_id_2,
        results=simulated_results_2,
        status="completed"
    )
    
    # Analyze the second experiment
    analysis_response_2 = analyze_results(
        action="analyze",
        experiment_id=experiment_id_2,
        name="Communication Styles Analysis",
        description="Analysis of different communication styles"
    )
    print(analysis_response_2)
    analysis_id_2 = analysis_response_2.split("ID: ")[1].split("\n")[0]
    
    # 8. Compare the analyses
    print("\n8. Comparing the analyses...")
    
    comparison = analyze_results(
        action="compare",
        analysis_ids=[analysis_id, analysis_id_2]
    )
    print(comparison)
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
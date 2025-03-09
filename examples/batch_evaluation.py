"""
Example of using AgentOptim to efficiently evaluate multiple conversations in batch.

This example demonstrates how to:
1. Create an EvalSet with quality criteria
2. Load and process multiple conversations from a dataset
3. Efficiently evaluate a large batch of conversations
4. Generate summary statistics and identify patterns

Use case: Evaluating a dataset of support conversations for quality monitoring
"""

import asyncio
import json
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pprint import pprint

from agentoptim.server import manage_evalset_tool, run_evalset_tool


# Helper function to load sample conversations
def load_sample_conversations(num_samples=10):
    """
    Generate sample conversations for demonstration purposes.
    In a real application, you would load these from a database or file.
    """
    # Sample user queries
    queries = [
        "How do I reset my password?",
        "Can I get a refund for my purchase?",
        "The app keeps crashing when I try to upload photos",
        "I need to change my shipping address",
        "Is there a way to download my account data?",
        "How do I connect my account to social media?",
        "The payment is showing as pending for 3 days",
        "Can I transfer my subscription to another account?",
        "I'm getting an error code XJ-42 when logging in",
        "How do I enable dark mode in the app?"
    ]
    
    # Sample system prompts
    system_prompts = [
        "You are a helpful customer support assistant.",
        "You are a technical support specialist who helps solve customer issues.",
        "You are a customer service agent for an e-commerce platform.",
    ]
    
    # Generate conversations with varying quality
    conversations = []
    for i in range(min(num_samples, len(queries))):
        # Select a system prompt
        system_prompt = system_prompts[i % len(system_prompts)]
        
        # Create a conversation ID
        conv_id = f"conv_{i+1:03d}"
        
        # Generate timestamps
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a conversation with metadata
        conversation = {
            "id": conv_id,
            "timestamp": timestamp,
            "channel": "chat" if i % 3 != 0 else "email",
            "category": ["account", "technical", "billing"][i % 3],
            "conversation": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": queries[i]},
                # Quality varies by index (good, average, poor in rotation)
                {"role": "assistant", "content": generate_response(queries[i], quality=i % 3)}
            ]
        }
        
        conversations.append(conversation)
    
    return conversations


def generate_response(query, quality=0):
    """Generate responses of different quality levels for demonstration"""
    # Quality levels: 0=good, 1=average, 2=poor
    
    responses = {
        "How do I reset my password?": [
            # Good response
            "To reset your password, please follow these steps:\n\n1. Go to the login page\n2. Click on 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder. Let me know if you need any further assistance!",
            # Average response
            "You can reset your password by clicking the 'Forgot Password' link on the login page and following the instructions sent to your email.",
            # Poor response
            "Password resets can be done through the account section. Check your email after requesting it."
        ],
        "Can I get a refund for my purchase?": [
            # Good response
            "Yes, you can request a refund for your purchase if it's within 30 days of the order date. Here's how to do it:\n\n1. Go to your Orders page in your account\n2. Find the specific order and click 'Request Refund'\n3. Select the reason for your refund\n4. Submit the request\n\nRefunds typically process within 5-7 business days, depending on your payment method. If your purchase was more than 30 days ago, please let me know and we can discuss other options for your situation.",
            # Average response
            "We offer refunds within 30 days of purchase. You can request one through your Orders page by clicking the Request Refund button.",
            # Poor response
            "Check the refund policy on our website for information about getting your money back."
        ]
    }
    
    # For queries not specifically defined, use generic responses
    if query not in responses:
        return [
            f"I'd be happy to help you with your question about {query.lower()}. Here's a detailed explanation with step-by-step instructions...[detailed multi-paragraph response with clear steps]",
            f"I can help with your question about {query.lower()}. The solution is to navigate to the relevant section and follow the basic instructions provided.",
            f"You should be able to find information about {query.lower()} in our help documentation."
        ][quality]
    
    return responses[query][quality]


async def main():
    print("=== AgentOptim Batch Evaluation ===")
    print("This example demonstrates evaluating multiple conversations efficiently")
    
    # Step 1: Create an EvalSet with quality criteria
    print("\n1. Creating support quality EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Support Quality Evaluation",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Does the response provide a complete solution?",
            "Is the tone professional and appropriate?",
            "Does the response provide all necessary details?",
            "Would this response likely resolve the customer's issue?"
        ],
        short_description="Evaluates customer support response quality",
        long_description="This EvalSet measures the quality of customer support responses based on helpfulness, clarity, completeness, and tone. It helps identify high and low-quality responses to improve overall support quality and consistency across agents. Use this for regular quality monitoring and agent training." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = None
    
    # First check for evalset_id directly in the response
    if "evalset_id" in evalset_result:
        evalset_id = evalset_result["evalset_id"]
    # Next check if it's in an evalset object
    elif "evalset" in evalset_result and "id" in evalset_result["evalset"]:
        evalset_id = evalset_result["evalset"]["id"]
    # Finally try to extract from result message
    elif "result" in evalset_result and isinstance(evalset_result["result"], str):
        import re
        id_match = re.search(r"ID: ([a-f0-9\-]+)", evalset_result["result"])
        if id_match:
            evalset_id = id_match.group(1)
    
    if not evalset_id:
        print("Error: Could not extract EvalSet ID from response")
        print(f"Response: {evalset_result}")
        return
        
    print(f"EvalSet created with ID: {evalset_id}")
    
    # Step 2: Load sample conversations
    print("\n2. Loading conversation dataset...")
    
    # In a real application, you would load conversations from a database or file
    # For this example, we'll generate sample conversations
    sample_size = 10
    conversations = load_sample_conversations(sample_size)
    print(f"Loaded {len(conversations)} conversations for evaluation")
    
    # Step 3: For demonstration purposes, we'll simulate batch evaluation
    print("\n3. In a real scenario, we would run batch evaluation on all conversations")
    print("   For demonstration purposes, we'll use simulated results to show the analysis")
    print("   In a real application, you would run:")
    print(f"   - run_evalset_tool(evalset_id={evalset_id}, conversation=conversation)")
    print("   for each conversation in the dataset")
    
    # Create a dictionary to store results
    all_results = {}
    
    # Generate simulated results for each conversation
    for i, conv_data in enumerate(conversations):
        conv_id = conv_data["id"]
        print(f"\nSimulating evaluation for conversation {i+1}/{len(conversations)}: {conv_id}")
        
        # Extract the conversation messages - we'll use this to determine quality level
        conversation = conv_data["conversation"]
        # Determine quality level based on the pattern in generate_response function
        # where quality is determined by i % 3 (good, average, poor)
        quality_level = i % 3
        
        # Simulate evaluation results based on quality level
        if quality_level == 0:  # Good response
            yes_count = 7  # All criteria pass
            confidence = 0.93
        elif quality_level == 1:  # Average response
            yes_count = 5  # Most criteria pass
            confidence = 0.85
        else:  # Poor response
            yes_count = 2  # Few criteria pass
            confidence = 0.78
            
        no_count = 7 - yes_count
        yes_percentage = (yes_count / 7) * 100
        
        # Create simulated evaluation results
        question_results = []
        questions = [
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Does the response provide a complete solution?",
            "Is the tone professional and appropriate?",
            "Does the response provide all necessary details?",
            "Would this response likely resolve the customer's issue?"
        ]
        
        # Generate judgments based on quality level
        if quality_level == 0:  # Good - all pass
            judgments = [True, True, True, True, True, True, True]
        elif quality_level == 1:  # Average - some fail
            judgments = [True, True, True, False, True, False, True]
        else:  # Poor - most fail
            judgments = [False, True, True, False, False, False, False]
            
        # Create result items
        for q_idx, question in enumerate(questions):
            result_item = {
                "question": question,
                "judgment": judgments[q_idx],
                "confidence": confidence - (0.05 * (q_idx % 3))  # Slight variation in confidence
            }
            question_results.append(result_item)
            
        # Create simulated eval_results
        eval_results = {
            "summary": {
                "yes_percentage": yes_percentage,
                "yes_count": yes_count,
                "no_count": no_count,
                "total_questions": 7,
                "mean_confidence": confidence
            },
            "results": question_results
        }
        
        # Store results with metadata
        all_results[conv_id] = {
            "metadata": {
                "timestamp": conv_data["timestamp"],
                "channel": conv_data["channel"],
                "category": conv_data["category"]
            },
            "evaluation": eval_results
        }
    
    print("\nBatch evaluation completed!")
    
    # Step 4: Generate summary statistics
    print("\n4. Analyzing batch evaluation results:")
    
    # Overall statistics
    overall_scores = [r["evaluation"]["summary"]["yes_percentage"] for r in all_results.values()]
    avg_score = sum(overall_scores) / len(overall_scores)
    min_score = min(overall_scores)
    max_score = max(overall_scores)
    
    print(f"\nOverall Quality Metrics:")
    print(f"Average score: {avg_score:.1f}%")
    print(f"Minimum score: {min_score:.1f}%")
    print(f"Maximum score: {max_score:.1f}%")
    print(f"Score range: {max_score - min_score:.1f}%")
    
    # Identify highest and lowest scoring conversations
    best_conv_id = max(all_results.keys(), key=lambda k: all_results[k]["evaluation"]["summary"]["yes_percentage"])
    worst_conv_id = min(all_results.keys(), key=lambda k: all_results[k]["evaluation"]["summary"]["yes_percentage"])
    
    print(f"\nHighest scoring conversation: {best_conv_id} ({all_results[best_conv_id]['evaluation']['summary']['yes_percentage']:.1f}%)")
    print(f"Lowest scoring conversation: {worst_conv_id} ({all_results[worst_conv_id]['evaluation']['summary']['yes_percentage']:.1f}%)")
    
    # Calculate scores by category
    categories = set(r["metadata"]["category"] for r in all_results.values())
    category_scores = {cat: [] for cat in categories}
    
    for conv_id, result in all_results.items():
        category = result["metadata"]["category"]
        score = result["evaluation"]["summary"]["yes_percentage"]
        category_scores[category].append(score)
    
    print("\nScores by Category:")
    for category, scores in category_scores.items():
        avg = sum(scores) / len(scores)
        print(f"{category.capitalize()}: {avg:.1f}% ({len(scores)} conversations)")
    
    # Calculate scores by channel
    channels = set(r["metadata"]["channel"] for r in all_results.values())
    channel_scores = {ch: [] for ch in channels}
    
    for conv_id, result in all_results.items():
        channel = result["metadata"]["channel"]
        score = result["evaluation"]["summary"]["yes_percentage"]
        channel_scores[channel].append(score)
    
    print("\nScores by Channel:")
    for channel, scores in channel_scores.items():
        avg = sum(scores) / len(scores)
        print(f"{channel.capitalize()}: {avg:.1f}% ({len(scores)} conversations)")
    
    # Analyze scores by criteria
    criteria = [q["question"] for q in all_results[list(all_results.keys())[0]]["evaluation"]["results"]]
    criteria_scores = {criterion: [] for criterion in criteria}
    
    for conv_id, result in all_results.items():
        for item in result["evaluation"]["results"]:
            question = item["question"]
            judgment = 1 if item["judgment"] else 0
            criteria_scores[question].append(judgment)
    
    print("\nPerformance by Criterion:")
    for criterion, judgments in criteria_scores.items():
        pass_rate = (sum(judgments) / len(judgments)) * 100
        print(f"- {criterion}: {pass_rate:.1f}% pass rate")
    
    # Identify strongest and weakest criteria
    best_criterion = max(criteria_scores.keys(), key=lambda k: sum(criteria_scores[k]) / len(criteria_scores[k]))
    worst_criterion = min(criteria_scores.keys(), key=lambda k: sum(criteria_scores[k]) / len(criteria_scores[k]))
    
    best_rate = (sum(criteria_scores[best_criterion]) / len(criteria_scores[best_criterion])) * 100
    worst_rate = (sum(criteria_scores[worst_criterion]) / len(criteria_scores[worst_criterion])) * 100
    
    print(f"\nStrongest criterion: {best_criterion} ({best_rate:.1f}%)")
    print(f"Weakest criterion: {worst_criterion} ({worst_rate:.1f}%)")
    
    # Step 5: Export results to CSV
    print("\n5. Exporting results...")
    
    # Prepare data for export
    export_data = []
    for conv_id, result in all_results.items():
        export_row = {
            "conversation_id": conv_id,
            "timestamp": result["metadata"]["timestamp"],
            "channel": result["metadata"]["channel"],
            "category": result["metadata"]["category"],
            "overall_score": result["evaluation"]["summary"]["yes_percentage"]
        }
        
        # Add individual criteria results
        for item in result["evaluation"]["results"]:
            question = item["question"].replace("?", "").replace(" ", "_").lower()
            export_row[question] = "Yes" if item["judgment"] else "No"
        
        export_data.append(export_row)
    
    # Export to CSV
    csv_filename = "batch_evaluation_results.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            if export_data:
                fieldnames = export_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in export_data:
                    writer.writerow(row)
            print(f"Results exported to {csv_filename}")
    except:
        print(f"Warning: Could not export results to CSV")
    
    # Step 6: Visualization (if matplotlib is available)
    try:
        # Create bar chart of category scores
        plt.figure(figsize=(12, 6))
        categories = list(category_scores.keys())
        avg_category_scores = [sum(scores) / len(scores) for scores in category_scores.values()]
        
        plt.bar(categories, avg_category_scores, color='skyblue')
        plt.axhline(y=avg_score, color='r', linestyle='-', label=f'Overall Average ({avg_score:.1f}%)')
        plt.xlabel('Category')
        plt.ylabel('Average Score (%)')
        plt.title('Support Quality by Category')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('batch_results_by_category.png')
        print("\nCreated visualization 'batch_results_by_category.png'")
    except:
        print("\nSkipping visualization (matplotlib may not be available)")
    
    # Step 7: Generate recommendations
    print("\n6. Recommendations based on analysis:")
    
    # Generate recommendations based on findings
    print("\nQuality Improvement Recommendations:")
    
    if worst_rate < 70:
        print(f"1. Focus training on improving: {worst_criterion}")
    
    if max_score - min_score > 30:
        print("2. Address inconsistency in response quality across conversations")
    
    # Category-specific recommendations
    lowest_category = min(category_scores.keys(), key=lambda k: sum(category_scores[k]) / len(category_scores[k]))
    lowest_cat_score = sum(category_scores[lowest_category]) / len(category_scores[lowest_category])
    
    if lowest_cat_score < avg_score - 10:
        print(f"3. Provide additional training for {lowest_category} support")
    
    # Channel-specific recommendations
    if len(channels) > 1:
        lowest_channel = min(channel_scores.keys(), key=lambda k: sum(channel_scores[k]) / len(channel_scores[k]))
        lowest_ch_score = sum(channel_scores[lowest_channel]) / len(channel_scores[lowest_channel])
        
        if lowest_ch_score < avg_score - 10:
            print(f"4. Review response templates and guidelines for {lowest_channel} channel")
    
    # General recommendations
    print("\nGeneral Quality Enhancement Steps:")
    print("1. Create response templates for frequently asked questions")
    print("2. Implement regular batch evaluations to track improvement")
    print("3. Set quality score targets by category and channel")
    print("4. Review lowest-scoring conversations for specific improvement opportunities")
    print(f"5. Share examples of high-quality responses (like conversation {best_conv_id})")


if __name__ == "__main__":
    asyncio.run(main())
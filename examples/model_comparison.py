"""
Example of using AgentOptim to compare different judge models for evaluations.

This example demonstrates how to:
1. Create an evaluation set for consistent comparison
2. Evaluate the same conversations with different judge models
3. Compare judgment patterns across models
4. Analyze confidence levels and agreement rates

Use case: Selecting the most appropriate judge model for specific evaluation needs
"""

import asyncio
import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from agentoptim import manage_evalset_tool, run_evalset_tool


async def main():
    print("=== AgentOptim Judge Model Comparison ===")
    print("This example compares different models for evaluation judgments")
    
    # Step 1: Create an EvalSet for comparison
    print("\n1. Creating test EvalSet...")
    
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Judge Model Comparison Test",
        questions=[
            "Is the response helpful for the user's needs?",
            "Does the response directly address the user's question?",
            "Is the response clear and easy to understand?",
            "Is the response factually accurate (where applicable)?",
            "Does the response provide complete information?",
            "Is the tone of the response appropriate and professional?",
            "Does the response avoid unnecessary information?"
        ],
        short_description="Test set for comparing judge models",
        long_description="This EvalSet contains a diverse set of evaluation criteria to test how different judge models perform. It includes both objective and subjective criteria to compare judgment patterns across models." + " " * 100
    )
    
    # Extract the EvalSet ID
    evalset_id = evalset_result.get("evalset", {}).get("id")
    print(f"Comparison test EvalSet created with ID: {evalset_id}")
    
    # Step 2: Define test conversations of varying quality
    print("\n2. Defining test conversations...")
    
    # Good response - comprehensive, clear, and factual
    good_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain how photosynthesis works?"},
        {"role": "assistant", "content": "Photosynthesis is the process by which plants, algae, and some bacteria convert light energy into chemical energy. Here's how it works:\n\n1. Plants absorb sunlight using chlorophyll, a green pigment in their leaves\n2. They take in carbon dioxide (CO₂) from the air through small pores called stomata\n3. They absorb water (H₂O) through their roots\n4. Using the energy from sunlight, plants convert CO₂ and H₂O into glucose (C₆H₁₂O₆) and oxygen (O₂)\n5. Glucose serves as food for the plant, providing energy and materials for growth\n6. Oxygen is released into the atmosphere as a byproduct\n\nThe overall chemical equation is:\n6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂\n\nThis process is crucial for life on Earth as it produces oxygen for animals to breathe and forms the base of most food chains. Photosynthesis also helps remove carbon dioxide from the atmosphere, playing a role in regulating Earth's climate."}
    ]
    
    # Average response - correct but lacks detail
    average_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain how photosynthesis works?"},
        {"role": "assistant", "content": "Photosynthesis is how plants make their own food using sunlight. The plants take in carbon dioxide from the air and water from the soil. Using the energy from sunlight, they convert these materials into glucose (sugar) and oxygen. The glucose provides energy for the plant, and the oxygen is released into the air. This process happens in the chloroplasts of plant cells, which contain chlorophyll, the green pigment that captures light energy."}
    ]
    
    # Poor response - incomplete and contains inaccuracies
    poor_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain how photosynthesis works?"},
        {"role": "assistant", "content": "Photosynthesis is how plants make food. They use sunlight and water to create energy. The plants use their leaves to catch the sunlight and then turn it into food. This is why plants need to be in the sun to grow properly."}
    ]
    
    # Ambiguous response - technically correct but potentially misleading
    ambiguous_conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain how photosynthesis works?"},
        {"role": "assistant", "content": "Photosynthesis works through a series of complex biochemical reactions that convert light energy into chemical energy. The process involves electron transport chains, photosystems I and II, and the Calvin cycle, which ultimately produces glucose. The light-dependent reactions occur in the thylakoid membrane, while the light-independent reactions take place in the stroma. The electron transport chain generates ATP and NADPH, which are then used in the Calvin cycle to reduce carbon dioxide into carbohydrates."}
    ]
    
    print("Defined 4 test conversations with varying quality:")
    print("1. Good: Comprehensive, clear, and factual")
    print("2. Average: Correct but lacks detail")
    print("3. Poor: Incomplete with inaccuracies")
    print("4. Ambiguous: Technically correct but potentially misleading")
    
    # Step 3: Define judge models to compare
    print("\n3. Setting up judge models to compare...")
    
    # List of judge models to compare
    # Note: This example assumes you have API keys set for these services
    # You may need to modify this list based on what's available in your environment
    judge_models = [
        "meta-llama-3.1-8b-instruct",      # Default, smaller model
        "gpt-3.5-turbo",                    # OpenAI's GPT-3.5
        "gpt-4o-mini",                      # OpenAI's smaller GPT-4 variant
        "claude-3-haiku-20240307"           # Anthropic's smaller Claude model
    ]
    
    # Check which models are available by environment variables
    available_models = []
    
    # Always include the default model
    available_models.append("meta-llama-3.1-8b-instruct")
    
    # Check for OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        if "gpt-3.5-turbo" in judge_models:
            available_models.append("gpt-3.5-turbo")
        if "gpt-4o-mini" in judge_models:
            available_models.append("gpt-4o-mini")
    
    # Check for Anthropic API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        if "claude-3-haiku-20240307" in judge_models:
            available_models.append("claude-3-haiku-20240307")
    
    print(f"Will compare {len(available_models)} judge models:")
    for model in available_models:
        print(f"- {model}")
    
    # Step 4: Evaluate conversations with each judge model
    print("\n4. Running evaluations with different judge models...")
    
    # Store all results by model and conversation
    all_results = {}
    
    # In AgentOptim v2.1.0, the model selection is controlled via environment variables
    # rather than the run_evalset_tool parameter. We'll set the environment variable
    # for each model we want to test.
    for model in available_models:
        print(f"\nEvaluating with {model}:")
        model_results = {}
        
        # Set the model via environment variable for this iteration
        os.environ["AGENTOPTIM_JUDGE_MODEL"] = model
        
        for conv_type, conversation in [
            ("good", good_conversation),
            ("average", average_conversation),
            ("poor", poor_conversation),
            ("ambiguous", ambiguous_conversation)
        ]:
            print(f"  Evaluating {conv_type} conversation...")
            try:
                eval_result = await run_evalset_tool(
                    evalset_id=evalset_id,
                    conversation=conversation,
                    max_parallel=3
                )
                model_results[conv_type] = eval_result
                print(f"    Score: {eval_result['summary']['yes_percentage']:.1f}%")
            except Exception as e:
                print(f"    Error with {model} on {conv_type} conversation: {e}")
                model_results[conv_type] = None
        
        all_results[model] = model_results
    
    print("\nAll evaluations completed!")
    
    # Step 5: Compare evaluation results across models
    print("\n5. Comparing evaluation results across models:")
    
    # Overall scores comparison
    print("\nOverall Scores Comparison:")
    print("-" * 60)
    print(f"{'Model':<30} | {'Good':<10} | {'Average':<10} | {'Poor':<10} | {'Ambiguous':<10}")
    print("-" * 60)
    
    for model in available_models:
        good_score = all_results[model].get("good", {}).get("summary", {}).get("yes_percentage", 0)
        avg_score = all_results[model].get("average", {}).get("summary", {}).get("yes_percentage", 0)
        poor_score = all_results[model].get("poor", {}).get("summary", {}).get("yes_percentage", 0)
        ambig_score = all_results[model].get("ambiguous", {}).get("summary", {}).get("yes_percentage", 0)
        
        print(f"{model:<30} | {good_score:>8.1f}% | {avg_score:>8.1f}% | {poor_score:>8.1f}% | {ambig_score:>8.1f}%")
    
    print("-" * 60)
    
    # Calculate score range (max - min) for each conversation type
    score_ranges = {}
    for conv_type in ["good", "average", "poor", "ambiguous"]:
        scores = [
            all_results[model].get(conv_type, {}).get("summary", {}).get("yes_percentage", 0)
            for model in available_models
            if all_results[model].get(conv_type) is not None
        ]
        if scores:
            score_ranges[conv_type] = max(scores) - min(scores)
    
    print("\nScore range (max - min) across models:")
    for conv_type, score_range in score_ranges.items():
        print(f"{conv_type.capitalize()}: {score_range:.1f}%")
    
    # Step 6: Analyze judgment patterns across models
    print("\n6. Analyzing judgment patterns:")
    
    # Create dictionary to store judgments by question and model
    question_judgments = defaultdict(lambda: defaultdict(dict))
    
    # Extract questions from first available result
    first_model = available_models[0]
    first_conv = "good"
    questions = [
        item["question"]
        for item in all_results[first_model][first_conv]["results"]
    ]
    
    # Collect judgments for each question across models and conversations
    for model in available_models:
        for conv_type in ["good", "average", "poor", "ambiguous"]:
            if all_results[model].get(conv_type) is not None:
                for item in all_results[model][conv_type]["results"]:
                    question = item["question"]
                    judgment = item["judgment"]
                    question_judgments[question][model][conv_type] = judgment
    
    # Calculate agreement rates for each question
    agreement_rates = {}
    for question in questions:
        agreement_count = 0
        comparison_count = 0
        
        # Compare each model pair for each conversation type
        for i, model1 in enumerate(available_models):
            for j, model2 in enumerate(available_models):
                if i < j:  # Only compare each pair once
                    for conv_type in ["good", "average", "poor", "ambiguous"]:
                        # Check if both models have judgments for this conversation type
                        if (conv_type in question_judgments[question][model1] and
                            conv_type in question_judgments[question][model2]):
                            comparison_count += 1
                            if question_judgments[question][model1][conv_type] == question_judgments[question][model2][conv_type]:
                                agreement_count += 1
        
        if comparison_count > 0:
            agreement_rates[question] = (agreement_count / comparison_count) * 100
    
    # Print agreement rates by question
    print("\nModel agreement rates by question:")
    print("-" * 80)
    for question, rate in sorted(agreement_rates.items(), key=lambda x: x[1]):
        print(f"{question:<70} | {rate:>6.1f}%")
    print("-" * 80)
    
    # Calculate overall agreement rate
    overall_agreement = sum(agreement_rates.values()) / len(agreement_rates) if agreement_rates else 0
    print(f"Overall agreement rate: {overall_agreement:.1f}%")
    
    # Step 7: Analyze confidence levels (using logprobs)
    print("\n7. Analyzing confidence levels:")
    
    # Create dictionary to store confidence levels by model and conversation type
    confidence_levels = defaultdict(lambda: defaultdict(list))
    
    # Collect confidence levels (absolute logprob values) for each model and conversation
    for model in available_models:
        for conv_type in ["good", "average", "poor", "ambiguous"]:
            if all_results[model].get(conv_type) is not None:
                for item in all_results[model][conv_type]["results"]:
                    if "logprob" in item:
                        # Higher absolute value means higher confidence
                        confidence = abs(item.get("logprob", 0))
                        confidence_levels[model][conv_type].append(confidence)
    
    # Calculate average confidence by model and conversation type
    print("\nAverage confidence levels (higher is more confident):")
    print("-" * 60)
    print(f"{'Model':<30} | {'Good':<10} | {'Average':<10} | {'Poor':<10} | {'Ambiguous':<10}")
    print("-" * 60)
    
    for model in available_models:
        good_conf = sum(confidence_levels[model]["good"]) / len(confidence_levels[model]["good"]) if confidence_levels[model]["good"] else 0
        avg_conf = sum(confidence_levels[model]["average"]) / len(confidence_levels[model]["average"]) if confidence_levels[model]["average"] else 0
        poor_conf = sum(confidence_levels[model]["poor"]) / len(confidence_levels[model]["poor"]) if confidence_levels[model]["poor"] else 0
        ambig_conf = sum(confidence_levels[model]["ambiguous"]) / len(confidence_levels[model]["ambiguous"]) if confidence_levels[model]["ambiguous"] else 0
        
        print(f"{model:<30} | {good_conf:>8.3f} | {avg_conf:>8.3f} | {poor_conf:>8.3f} | {ambig_conf:>8.3f}")
    
    print("-" * 60)
    
    # Step 8: Generate visualizations
    print("\n8. Generating visualizations...")
    
    try:
        # Create bar chart comparing model scores for each conversation type
        plt.figure(figsize=(15, 8))
        
        # Set up bar positions
        bar_width = 0.2
        index = np.arange(4)  # 4 conversation types
        
        # Plot bars for each model
        for i, model in enumerate(available_models):
            scores = [
                all_results[model].get("good", {}).get("summary", {}).get("yes_percentage", 0),
                all_results[model].get("average", {}).get("summary", {}).get("yes_percentage", 0),
                all_results[model].get("poor", {}).get("summary", {}).get("yes_percentage", 0),
                all_results[model].get("ambiguous", {}).get("summary", {}).get("yes_percentage", 0)
            ]
            plt.bar(index + i * bar_width, scores, bar_width, label=model)
        
        plt.xlabel('Conversation Type')
        plt.ylabel('Score (%)')
        plt.title('Judge Model Comparison by Conversation Type')
        plt.xticks(index + bar_width * (len(available_models) - 1) / 2, ['Good', 'Average', 'Poor', 'Ambiguous'])
        plt.ylim(0, 110)  # To make room for annotations
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.grid(axis='y', alpha=0.3)
        
        # Save chart
        plt.tight_layout()
        plt.savefig('model_comparison_scores.png')
        print("Created 'model_comparison_scores.png'")
        
        # Create a heatmap of agreement rates by question
        plt.figure(figsize=(10, 12))
        
        # Prepare data
        questions_short = [q.split('?')[0] + '?' for q in questions]  # Shorten question text
        agreement_data = [agreement_rates[q] for q in questions]
        
        # Create horizontal bar chart
        bars = plt.barh(questions_short, agreement_data, color='skyblue')
        
        # Add value labels
        for bar, value in zip(bars, agreement_data):
            plt.text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.1f}%', 
                    va='center', fontweight='bold')
        
        plt.xlabel('Agreement Rate (%)')
        plt.title('Model Agreement Rate by Question')
        plt.xlim(0, 110)  # To make room for annotations
        plt.axvline(x=overall_agreement, color='red', linestyle='--', 
                   label=f'Overall Agreement ({overall_agreement:.1f}%)')
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        
        # Save chart
        plt.tight_layout()
        plt.savefig('model_agreement_rates.png')
        print("Created 'model_agreement_rates.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Step 9: Generate recommendations
    print("\n9. Model selection recommendations:")
    
    # Calculate overall score by model
    model_overall_scores = {}
    for model in available_models:
        scores = []
        for conv_type in ["good", "average", "poor", "ambiguous"]:
            score = all_results[model].get(conv_type, {}).get("summary", {}).get("yes_percentage", 0)
            if score > 0:  # Only count valid scores
                scores.append(score)
        
        if scores:
            model_overall_scores[model] = sum(scores) / len(scores)
    
    # Calculate score spread (max - min score) for each model
    model_score_spreads = {}
    for model in available_models:
        scores = []
        for conv_type in ["good", "average", "poor", "ambiguous"]:
            score = all_results[model].get(conv_type, {}).get("summary", {}).get("yes_percentage", 0)
            if score > 0:  # Only count valid scores
                scores.append(score)
        
        if scores:
            model_score_spreads[model] = max(scores) - min(scores)
    
    # Calculate average confidence by model
    model_avg_confidence = {}
    for model in available_models:
        all_conf = []
        for conv_type in ["good", "average", "poor", "ambiguous"]:
            all_conf.extend(confidence_levels[model][conv_type])
        
        if all_conf:
            model_avg_confidence[model] = sum(all_conf) / len(all_conf)
    
    # Print model characteristics
    print("\nModel characteristics:")
    print("-" * 75)
    print(f"{'Model':<30} | {'Avg Score':<10} | {'Score Spread':<12} | {'Avg Confidence':<15}")
    print("-" * 75)
    
    for model in available_models:
        avg_score = model_overall_scores.get(model, 0)
        score_spread = model_score_spreads.get(model, 0)
        avg_conf = model_avg_confidence.get(model, 0)
        
        print(f"{model:<30} | {avg_score:>8.1f}% | {score_spread:>10.1f}% | {avg_conf:>13.3f}")
    
    print("-" * 75)
    
    # Generate specific recommendations
    print("\nRecommendations for model selection:")
    
    # Find model with highest score discrimination (highest score spread)
    best_discriminator = max(model_score_spreads.items(), key=lambda x: x[1])[0] if model_score_spreads else None
    
    # Find model with highest agreement with other models
    model_agreement_rates = {}
    for model in available_models:
        agreement_count = 0
        comparison_count = 0
        
        for question in questions:
            for other_model in available_models:
                if model != other_model:
                    for conv_type in ["good", "average", "poor", "ambiguous"]:
                        if (conv_type in question_judgments[question][model] and
                            conv_type in question_judgments[question][other_model]):
                            comparison_count += 1
                            if question_judgments[question][model][conv_type] == question_judgments[question][other_model][conv_type]:
                                agreement_count += 1
        
        if comparison_count > 0:
            model_agreement_rates[model] = (agreement_count / comparison_count) * 100
    
    most_agreeable = max(model_agreement_rates.items(), key=lambda x: x[1])[0] if model_agreement_rates else None
    
    # Find model with highest confidence
    most_confident = max(model_avg_confidence.items(), key=lambda x: x[1])[0] if model_avg_confidence else None
    
    # Print recommendations
    print(f"\n1. For best discrimination between good and poor responses: {best_discriminator}")
    print(f"   (Shows the largest score difference between high and low quality responses)")
    
    print(f"\n2. For most reliable consensus judgments: {most_agreeable}")
    print(f"   (Highest agreement rate with other judge models)")
    
    print(f"\n3. For highest confidence judgments: {most_confident}")
    print(f"   (Makes the most decisive judgments with highest certainty)")
    
    print("\n4. General recommendations:")
    print("   - Use smaller models (8B-10B parameters) for routine evaluations")
    print("   - Use larger models for complex or nuanced judgments")
    print("   - Consider using multiple judge models for critical evaluations")
    print("   - Look for score patterns rather than absolute values")
    
    print("\nThis comparison helps identify the most appropriate judge model for your specific evaluation needs.")


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Confidence Visualization Example

This example demonstrates how to use AgentOptim's visualization tools
to analyze and plot confidence scores from evaluation runs.
"""

import os
import sys
import json
import asyncio
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentoptim import manage_evalset_tool, manage_eval_runs_tool
from agentoptim.visualization import (
    generate_confidence_histogram,
    plot_calibration_curve,
    plot_multi_model_calibration,
    export_confidence_data,
    format_confidence_cli
)
from agentoptim.confidence import calculate_ece, calculate_calibration_curve

# Create example data directory
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(EXAMPLES_DIR, exist_ok=True)

COLORS = {
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
    "bold": "\033[1m"
}

async def create_sample_evalset():
    """Create a sample evaluation set for testing."""
    print(f"{COLORS['cyan']}Creating sample evaluation set...{COLORS['reset']}")
    
    # Create a simple evaluation set
    evalset_result = await manage_evalset_tool(
        action="create",
        name="Factual Knowledge",
        questions=[
            "Does the response correctly state that Paris is the capital of France?",
            "Does the response correctly state that water boils at 100°C at sea level?",
            "Does the response correctly identify the first president of the United States?",
            "Does the response correctly state that the Earth revolves around the Sun?",
            "Does the response correctly identify the chemical symbol for gold?",
            "Does the response correctly state that a triangle has three sides?",
            "Does the response correctly identify the year World War II ended?",
            "Does the response correctly state that DNA stands for deoxyribonucleic acid?",
            "Does the response correctly identify Shakespeare as the author of Romeo and Juliet?",
            "Does the response correctly state that the human body has 206 bones?"
        ],
        short_description="Basic factual knowledge evaluation",
        confidence_config={
            "method": "combo_exemplars",
            "instructions": "Also provide your confidence in your judgment on a scale from 0 to 1."
        }
    )
    
    evalset_id = evalset_result["evalset"]["id"]
    print(f"{COLORS['green']}Created evaluation set with ID: {evalset_id}{COLORS['reset']}")
    return evalset_id

async def run_sample_evaluation(evalset_id, model_name=None):
    """Run a sample evaluation with the given evalset ID."""
    print(f"{COLORS['cyan']}Running evaluation...{COLORS['reset']}")
    
    # Create a sample conversation with factual responses
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that provides factual information."},
        {"role": "user", "content": "Please tell me about Paris and France."},
        {"role": "assistant", "content": "Paris is the capital city of France. It's known for landmarks like the Eiffel Tower and the Louvre Museum."},
        {"role": "user", "content": "What temperature does water boil at?"},
        {"role": "assistant", "content": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure at sea level."},
        {"role": "user", "content": "Who was the first U.S. president?"},
        {"role": "assistant", "content": "George Washington was the first President of the United States, serving from 1789 to 1797."},
        {"role": "user", "content": "What revolves around what in our solar system?"},
        {"role": "assistant", "content": "In our solar system, the Earth and other planets revolve around the Sun. The Moon revolves around the Earth."},
        {"role": "user", "content": "What's the chemical symbol for gold?"},
        {"role": "assistant", "content": "The chemical symbol for gold is Au, which comes from the Latin word 'aurum'."},
        {"role": "user", "content": "How many sides does a triangle have?"},
        {"role": "assistant", "content": "A triangle has three sides."},
        {"role": "user", "content": "When did World War II end?"},
        {"role": "assistant", "content": "World War II ended in 1945. In Europe, it ended on May 8, 1945 (V-E Day), and in the Pacific on September 2, 1945 (V-J Day) following Japan's surrender."},
        {"role": "user", "content": "What does DNA stand for?"},
        {"role": "assistant", "content": "DNA stands for deoxyribonucleic acid. It's a molecule that carries genetic information in all living organisms."},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "assistant", "content": "Romeo and Juliet was written by William Shakespeare. It's one of his most famous tragedies."},
        {"role": "user", "content": "How many bones are in the human body?"},
        {"role": "assistant", "content": "The adult human body typically has 206 bones."}
    ]
    
    # Run the evaluation
    eval_result = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation,
        judge_model=model_name
    )
    
    eval_run_id = eval_result["id"]
    print(f"{COLORS['green']}Completed evaluation with ID: {eval_run_id}{COLORS['reset']}")
    return eval_run_id

async def run_sample_evaluation_less_confident(evalset_id, model_name=None):
    """Run another sample evaluation with less certain responses."""
    print(f"{COLORS['cyan']}Running second evaluation with less certain responses...{COLORS['reset']}")
    
    # Create a sample conversation with some less certain or incorrect responses
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that provides information."},
        {"role": "user", "content": "Please tell me about Paris and France."},
        {"role": "assistant", "content": "I believe Paris is the capital city of France, though there are many other important cities like Lyon and Marseille."},
        {"role": "user", "content": "What temperature does water boil at?"},
        {"role": "assistant", "content": "Water typically boils at around 100 degrees Celsius, though this can vary slightly depending on altitude and atmospheric pressure."},
        {"role": "user", "content": "Who was the first U.S. president?"},
        {"role": "assistant", "content": "George Washington is generally considered the first President of the United States, though some scholars debate the significance of earlier leaders."},
        {"role": "user", "content": "What revolves around what in our solar system?"},
        {"role": "assistant", "content": "In our solar system, I believe the planets, including Earth, revolve around the Sun, while moons revolve around their respective planets."},
        {"role": "user", "content": "What's the chemical symbol for gold?"},
        {"role": "assistant", "content": "If I recall correctly, the chemical symbol for gold is Au, derived from the Latin term."},
        {"role": "user", "content": "How many sides does a triangle have?"},
        {"role": "assistant", "content": "A standard triangle has three sides, by definition."},
        {"role": "user", "content": "When did World War II end?"},
        {"role": "assistant", "content": "World War II ended sometime in the mid-1940s, with different dates for different theaters of war."},
        {"role": "user", "content": "What does DNA stand for?"},
        {"role": "assistant", "content": "DNA stands for deoxyribonucleic acid, if my memory serves me correctly."},
        {"role": "user", "content": "Who wrote Romeo and Juliet?"},
        {"role": "assistant", "content": "Romeo and Juliet is typically attributed to William Shakespeare, though some scholars have questioned the authorship of various works in the Shakespeare canon."},
        {"role": "user", "content": "How many bones are in the human body?"},
        {"role": "assistant", "content": "I believe the average adult human body has approximately 206 bones, though this number can vary slightly between individuals."}
    ]
    
    # Run the evaluation
    eval_result = await manage_eval_runs_tool(
        action="run",
        evalset_id=evalset_id,
        conversation=conversation,
        judge_model=model_name
    )
    
    eval_run_id = eval_result["id"]
    print(f"{COLORS['green']}Completed second evaluation with ID: {eval_run_id}{COLORS['reset']}")
    return eval_run_id

async def fetch_evaluation_results(eval_run_id):
    """Fetch the evaluation results for the given run ID."""
    result = await manage_eval_runs_tool(
        action="get",
        eval_run_id=eval_run_id
    )
    return result

def create_histogram_visualization(eval_results, output_dir):
    """Create a histogram of confidence scores."""
    print(f"{COLORS['cyan']}Creating confidence histogram...{COLORS['reset']}")
    
    # Extract confidence scores
    confidence_scores = []
    for result in eval_results["eval_run"]["results"]:
        if "confidence" in result and result["confidence"] is not None:
            confidence_scores.append(result["confidence"])
    
    # Generate and save histogram
    output_path = os.path.join(output_dir, "confidence_histogram.png")
    title = f"Confidence Distribution: {eval_results['eval_run']['evalset_name']}"
    
    fig = generate_confidence_histogram(
        confidence_scores,
        output_path=output_path,
        title=title,
        bins=10,
        show_stats=True
    )
    
    # Print simple ASCII histogram to console
    print("\nConfidence Score Distribution:")
    hist, bin_edges = np.histogram(confidence_scores, bins=10, range=(0, 1))
    max_count = max(hist)
    for i in range(len(hist)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bar_width = int((hist[i] / max_count) * 40)  # Scale to 40 chars width
        print(f"{bin_start:.1f}-{bin_end:.1f}: {'#' * bar_width} {hist[i]}")
    
    print(f"{COLORS['green']}Saved histogram to {output_path}{COLORS['reset']}")

def create_calibration_curve(eval_results, output_dir):
    """Create a calibration curve for the confidence scores."""
    print(f"{COLORS['cyan']}Creating calibration curve...{COLORS['reset']}")
    
    # Extract confidence scores and judgments
    confidence_scores = []
    judgments = []
    for result in eval_results["eval_run"]["results"]:
        if "confidence" in result and result["confidence"] is not None:
            confidence_scores.append(result["confidence"])
            judgments.append(result["judgment"])
    
    # Generate and save calibration curve
    output_path = os.path.join(output_dir, "calibration_curve.png")
    title = f"Calibration Curve: {eval_results['eval_run']['evalset_name']}"
    
    fig = plot_calibration_curve(
        confidence_scores,
        judgments,
        output_path=output_path,
        title=title,
        num_bins=5,  # Fewer bins for small sample size
        show_ece=True
    )
    
    # Calculate and display ECE
    ece = calculate_ece(confidence_scores, judgments)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    
    # Display text-based calibration data
    bin_edges, bin_accs, bin_confs, bin_counts = calculate_calibration_curve(
        confidence_scores, judgments, 5
    )
    
    print("\nCalibration Data:")
    print(f"{'Bin Range':<15} {'Confidence':<12} {'Accuracy':<12} {'Samples':<10} {'Gap':<10}")
    print(f"{'-' * 59}")
    for i in range(len(bin_confs)):
        if np.isnan(bin_confs[i]) or np.isnan(bin_accs[i]):
            continue
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_range = f"{bin_start:.2f}-{bin_end:.2f}"
        gap = bin_accs[i] - bin_confs[i]
        print(f"{bin_range:<15} {bin_confs[i]:.4f}{'  ':5} {bin_accs[i]:.4f}{'  ':5} {bin_counts[i]:<10} {gap:.4f}")
    
    print(f"{COLORS['green']}Saved calibration curve to {output_path}{COLORS['reset']}")

def compare_calibration_curves(eval_results1, eval_results2, output_dir):
    """Compare calibration curves between two evaluation runs."""
    print(f"{COLORS['cyan']}Comparing calibration curves...{COLORS['reset']}")
    
    # Extract confidence scores and judgments from first evaluation
    confidence_scores1 = []
    judgments1 = []
    for result in eval_results1["eval_run"]["results"]:
        if "confidence" in result and result["confidence"] is not None:
            confidence_scores1.append(result["confidence"])
            judgments1.append(result["judgment"])
    
    # Extract confidence scores and judgments from second evaluation
    confidence_scores2 = []
    judgments2 = []
    for result in eval_results2["eval_run"]["results"]:
        if "confidence" in result and result["confidence"] is not None:
            confidence_scores2.append(result["confidence"])
            judgments2.append(result["judgment"])
    
    # Prepare model data dictionary
    model_data = {
        "Run A (Confident)": (confidence_scores1, judgments1),
        "Run B (Less Confident)": (confidence_scores2, judgments2)
    }
    
    # Generate and save multi-model calibration plot
    output_path = os.path.join(output_dir, "calibration_comparison.png")
    title = "Calibration Comparison"
    
    fig = plot_multi_model_calibration(
        model_data,
        output_path=output_path,
        title=title,
        num_bins=5  # Fewer bins for small sample size
    )
    
    # Calculate and display ECE for both models
    ece1 = calculate_ece(confidence_scores1, judgments1)
    ece2 = calculate_ece(confidence_scores2, judgments2)
    
    print("\nCalibration Comparison:")
    print(f"{'Model':<25} {'ECE':<10} {'Mean Confidence':<15} {'Samples':<10}")
    print(f"{'-' * 60}")
    print(f"{'Run A (Confident)':<25} {ece1:.4f}{'  ':5} {np.mean(confidence_scores1):.4f}{'  ':5} {len(confidence_scores1):<10}")
    print(f"{'Run B (Less Confident)':<25} {ece2:.4f}{'  ':5} {np.mean(confidence_scores2):.4f}{'  ':5} {len(confidence_scores2):<10}")
    
    print(f"{COLORS['green']}Saved calibration comparison to {output_path}{COLORS['reset']}")

def export_confidence_report(eval_results, output_dir):
    """Export confidence data in various formats."""
    print(f"{COLORS['cyan']}Exporting confidence reports...{COLORS['reset']}")
    
    # Export as JSON
    json_path = os.path.join(output_dir, "confidence_data.json")
    json_data = export_confidence_data(eval_results, "json", json_path)
    
    # Export as CSV
    csv_path = os.path.join(output_dir, "confidence_data.csv")
    csv_data = export_confidence_data(eval_results, "csv", csv_path)
    
    # Export as HTML
    html_path = os.path.join(output_dir, "confidence_data.html")
    html_data = export_confidence_data(eval_results, "html", html_path)
    
    # Export as Markdown
    md_path = os.path.join(output_dir, "confidence_data.md")
    md_data = export_confidence_data(eval_results, "markdown", md_path)
    
    print(f"{COLORS['green']}Exported confidence data to multiple formats in {output_dir}{COLORS['reset']}")

def create_confidence_summary(eval_results):
    """Create a text-based summary of confidence metrics."""
    print(f"{COLORS['cyan']}Generating confidence summary...{COLORS['reset']}")
    
    # Extract confidence scores and judgments
    confidence_scores = []
    judgments = []
    for result in eval_results["eval_run"]["results"]:
        if "confidence" in result and result["confidence"] is not None:
            confidence_scores.append(result["confidence"])
            judgments.append(result["judgment"])
    
    # Calculate basic metrics
    mean_confidence = np.mean(confidence_scores)
    median_confidence = np.median(confidence_scores)
    min_confidence = min(confidence_scores)
    max_confidence = max(confidence_scores)
    variance = np.var(confidence_scores)
    
    # Calculate calibration metrics
    ece = calculate_ece(confidence_scores, judgments)
    accuracy = np.mean(judgments)
    gap = accuracy - mean_confidence
    
    # Print summary report
    print(f"\n{COLORS['bold']}Confidence Summary: {eval_results['eval_run']['evalset_name']}{COLORS['reset']}")
    print(f"Judge Model: {eval_results['eval_run']['judge_model']}")
    print(f"Date: {eval_results['eval_run']['timestamp']}")
    print("")
    
    # Confidence statistics
    print(f"{COLORS['yellow']}Confidence Statistics{COLORS['reset']}")
    print(f"Mean confidence:     {mean_confidence:.4f}")
    print(f"Median confidence:   {median_confidence:.4f}")
    print(f"Confidence variance: {variance:.4f}")
    print(f"Confidence range:    {min_confidence:.2f} - {max_confidence:.2f}")
    print("")
    
    # Calibration metrics
    print(f"{COLORS['yellow']}Calibration Metrics{COLORS['reset']}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confidence-accuracy gap: {gap:.4f}")
    print("")
    
    # Distribution of confidence scores by range
    print(f"{COLORS['yellow']}Confidence Distribution{COLORS['reset']}")
    ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    for start, end in ranges:
        count = sum(1 for c in confidence_scores if start <= c < end)
        percentage = (count / len(confidence_scores)) * 100
        print(f"{start:.1f}-{end:.1f}: {count} scores ({percentage:.1f}%)")

async def run_visualizations():
    """Run all visualization examples."""
    # Set up output directory
    output_dir = os.path.join(EXAMPLES_DIR, f"confidence_viz_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{COLORS['bold']}AgentOptim Confidence Visualization Example{COLORS['reset']}")
    print(f"Output directory: {output_dir}")
    print("----------------------------------------")
    
    # Create evalset and run evaluations
    evalset_id = await create_sample_evalset()
    
    # Run two evaluations with different confidence patterns
    eval_run_id1 = await run_sample_evaluation(evalset_id)
    eval_run_id2 = await run_sample_evaluation_less_confident(evalset_id)
    
    # Fetch results
    eval_results1 = await fetch_evaluation_results(eval_run_id1)
    eval_results2 = await fetch_evaluation_results(eval_run_id2)
    
    # Check if confidence scores are present
    has_confidence1 = any("confidence" in result for result in eval_results1["eval_run"]["results"])
    has_confidence2 = any("confidence" in result for result in eval_results2["eval_run"]["results"])
    
    if not has_confidence1 or not has_confidence2:
        print(f"{COLORS['red']}Error: Confidence scores not found in evaluation results.{COLORS['reset']}")
        print("Make sure your evaluation was run with confidence elicitation enabled.")
        return
    
    # Create various visualizations
    create_histogram_visualization(eval_results1, output_dir)
    create_calibration_curve(eval_results1, output_dir)
    compare_calibration_curves(eval_results1, eval_results2, output_dir)
    export_confidence_report(eval_results1, output_dir)
    create_confidence_summary(eval_results1)
    
    print("----------------------------------------")
    print(f"{COLORS['green']}All visualizations complete! Files saved to {output_dir}{COLORS['reset']}")
    print(f"\nTo view these visualizations with the CLI, try:")
    print(f"{COLORS['cyan']}agentoptim run visualize {eval_run_id1} --type histogram --color{COLORS['reset']}")
    print(f"{COLORS['cyan']}agentoptim run visualize {eval_run_id1} --type calibration --save-image calibration.png{COLORS['reset']}")
    print(f"{COLORS['cyan']}agentoptim run visualize {eval_run_id1} --type distribution --color{COLORS['reset']}")
    print(f"{COLORS['cyan']}agentoptim run visualize {eval_run_id1} --type summary{COLORS['reset']}")
    print(f"{COLORS['cyan']}agentoptim run visualize {eval_run_id1} --type calibration --compare {eval_run_id2}{COLORS['reset']}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AgentOptim Confidence Visualization Example")
    args = parser.parse_args()
    
    # Run all visualizations
    asyncio.run(run_visualizations())

if __name__ == "__main__":
    main()
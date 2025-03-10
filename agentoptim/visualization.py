"""
Visualization module for confidence scores and evaluation results.

This module provides functions for visualizing confidence scores, 
including calibration curves, distribution histograms, and CLI output formatting.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
from .confidence import calculate_ece, calculate_calibration_curve

# ANSI color codes for CLI output
COLORS = {
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "reset": "\033[0m",
    "bold": "\033[1m"
}

def format_confidence_cli(confidence: float, judgment: bool = None) -> str:
    """
    Format a confidence score for CLI output with color coding.
    
    Args:
        confidence: A confidence score between 0 and 1
        judgment: Optional boolean judgment (True/False)
        
    Returns:
        A formatted string with color coding
    """
    if confidence is None:
        return f"{COLORS['yellow']}No confidence{COLORS['reset']}"
    
    # Determine color based on confidence value
    if confidence >= 0.9:
        color = COLORS['green']
    elif confidence >= 0.7:
        color = COLORS['blue']
    elif confidence >= 0.5:
        color = COLORS['yellow']
    else:
        color = COLORS['red']
    
    # Format with percentage and color
    confidence_str = f"{confidence:.1%}"
    
    # Add judgment indicator if provided
    if judgment is not None:
        if judgment:
            judgment_str = f"{COLORS['green']}✓{COLORS['reset']}"
        else:
            judgment_str = f"{COLORS['red']}✗{COLORS['reset']}"
        return f"{color}{confidence_str}{COLORS['reset']} {judgment_str}"
    
    return f"{color}{confidence_str}{COLORS['reset']}"

def generate_confidence_histogram(
    confidence_scores: List[float], 
    output_path: str = None,
    title: str = "Confidence Score Distribution",
    bins: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True
) -> plt.Figure:
    """
    Generate a histogram showing the distribution of confidence scores.
    
    Args:
        confidence_scores: List of confidence scores (values between 0 and 1)
        output_path: Optional file path to save the plot
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size as (width, height) tuple
        show_stats: Whether to show statistics on the plot
        
    Returns:
        Matplotlib figure object
    """
    if not confidence_scores:
        raise ValueError("No confidence scores provided")
    
    # Remove any None values
    scores = [score for score in confidence_scores if score is not None]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    counts, bins, patches = ax.hist(
        scores, 
        bins=bins, 
        alpha=0.7, 
        color='skyblue',
        edgecolor='black'
    )
    
    # Add mean line
    mean_score = np.mean(scores)
    ax.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, 
               label=f'Mean: {mean_score:.2f}')
    
    # Add statistics if requested
    if show_stats:
        stats_text = (
            f"Mean: {mean_score:.2f}\n"
            f"Median: {np.median(scores):.2f}\n"
            f"Std Dev: {np.std(scores):.2f}\n"
            f"Min: {min(scores):.2f}\n"
            f"Max: {max(scores):.2f}\n"
            f"Count: {len(scores)}"
        )
        
        # Add text box with statistics
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
    
    # Set labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_calibration_curve(
    confidence_scores: List[float],
    judgments: List[bool],
    output_path: str = None,
    title: str = "Calibration Curve",
    figsize: Tuple[int, int] = (10, 6),
    num_bins: int = 10,
    show_ece: bool = True
) -> plt.Figure:
    """
    Plot a calibration curve showing how well confidence scores align with outcomes.
    
    Args:
        confidence_scores: List of confidence scores (values between 0 and 1)
        judgments: List of boolean judgments (True/False)
        output_path: Optional file path to save the plot
        title: Plot title
        figsize: Figure size as (width, height) tuple
        num_bins: Number of bins for calibration
        show_ece: Whether to show the Expected Calibration Error
        
    Returns:
        Matplotlib figure object
    """
    if not confidence_scores or not judgments:
        raise ValueError("Empty confidence scores or judgments")
    
    if len(confidence_scores) != len(judgments):
        raise ValueError("Length of confidence scores and judgments must match")
    
    # Filter out None values
    valid_data = [(conf, jud) for conf, jud in zip(confidence_scores, judgments) 
                  if conf is not None]
    
    if not valid_data:
        raise ValueError("No valid data points after filtering None values")
    
    filtered_confidence, filtered_judgments = zip(*valid_data)
    
    # Calculate calibration curve
    bin_edges, bin_accs, bin_confs, bin_counts = calculate_calibration_curve(
        filtered_confidence, filtered_judgments, num_bins
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot calibration curve
    ax.plot(bin_confs, bin_accs, 'o-', color='#3498db', linewidth=2,
            label='Model Calibration')
    
    # Add bin sample counts
    for x, y, count in zip(bin_confs, bin_accs, bin_counts):
        if np.isnan(x) or np.isnan(y):
            continue
        ax.text(x, y+0.02, f'n={count}', ha='center', va='bottom', 
                fontsize=8, alpha=0.7)
    
    # Calculate ECE if requested
    if show_ece:
        ece = calculate_ece(filtered_confidence, filtered_judgments, num_bins)
        ax.text(0.05, 0.05, f'ECE: {ece:.4f}', transform=ax.transAxes,
                bbox={'boxstyle': 'round', 'alpha': 0.5})
    
    # Set labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid and legend
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multi_model_calibration(
    model_data: Dict[str, Tuple[List[float], List[bool]]],
    output_path: str = None,
    title: str = "Multi-Model Calibration Comparison",
    figsize: Tuple[int, int] = (12, 8),
    num_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curves for multiple models for comparison.
    
    Args:
        model_data: Dictionary mapping model names to tuples of 
                   (confidence_scores, judgments)
        output_path: Optional file path to save the plot
        title: Plot title
        figsize: Figure size as (width, height) tuple
        num_bins: Number of bins for calibration
        
    Returns:
        Matplotlib figure object
    """
    if not model_data:
        raise ValueError("No model data provided")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Colors for different models
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Table data for ECE values
    ece_data = []
    
    # Plot each model's calibration curve
    for i, (model_name, (confidence_scores, judgments)) in enumerate(model_data.items()):
        # Filter out None values
        valid_data = [(conf, jud) for conf, jud in zip(confidence_scores, judgments) 
                    if conf is not None]
        
        if not valid_data:
            continue
            
        filtered_confidence, filtered_judgments = zip(*valid_data)
        
        # Calculate calibration curve
        bin_edges, bin_accs, bin_confs, bin_counts = calculate_calibration_curve(
            filtered_confidence, filtered_judgments, num_bins
        )
        
        # Calculate ECE
        ece = calculate_ece(filtered_confidence, filtered_judgments, num_bins)
        ece_data.append((model_name, ece))
        
        # Get color (cycle through colors if more models than colors)
        color = colors[i % len(colors)]
        
        # Plot calibration curve
        ax.plot(bin_confs, bin_accs, 'o-', color=color, linewidth=2,
                label=f'{model_name} (ECE: {ece:.4f})')
    
    # Set labels and title
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add grid and legend
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def export_confidence_data(
    eval_results: Dict[str, Any],
    output_format: str = "json",
    output_path: str = None
) -> Union[str, Dict, pd.DataFrame]:
    """
    Export confidence data in various formats.
    
    Args:
        eval_results: Evaluation results containing confidence data
        output_format: Format to export ("json", "csv", "html", "markdown")
        output_path: Optional file path to save the exported data
        
    Returns:
        Exported data in the requested format
    """
    # Extract confidence data
    results = eval_results.get("results", [])
    
    # Prepare data for export
    export_data = []
    for result in results:
        judgment = result.get("judgment", None)
        question = result.get("question", "")
        confidence = result.get("confidence", None)
        confidence_method = result.get("confidence_method", "")
        
        export_data.append({
            "question": question,
            "judgment": judgment,
            "confidence": confidence,
            "confidence_method": confidence_method
        })
    
    # Handle different output formats
    if output_format == "json":
        json_data = {
            "evalset_id": eval_results.get("eval_run", {}).get("evalset_id", ""),
            "eval_run_id": eval_results.get("eval_run", {}).get("id", ""),
            "timestamp": eval_results.get("eval_run", {}).get("timestamp", ""),
            "confidence_data": export_data,
            "summary": {
                "mean_confidence": np.mean([r.get("confidence", 0) for r in results if r.get("confidence") is not None]),
                "ece": calculate_ece(
                    [r.get("confidence") for r in results if r.get("confidence") is not None],
                    [r.get("judgment") for r in results if r.get("confidence") is not None]
                )
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        return json_data
    
    elif output_format in ["csv", "html", "markdown"]:
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        # Add formatted confidence column
        if "confidence" in df.columns:
            df["confidence_pct"] = df["confidence"].apply(
                lambda x: f"{x:.1%}" if x is not None else "N/A"
            )
        
        if output_format == "csv" and output_path:
            df.to_csv(output_path, index=False)
            return df
        elif output_format == "html" and output_path:
            html_output = df.to_html(index=False)
            with open(output_path, 'w') as f:
                f.write(html_output)
            return html_output
        elif output_format == "markdown" and output_path:
            markdown_output = df.to_markdown(index=False)
            with open(output_path, 'w') as f:
                f.write(markdown_output)
            return markdown_output
        
        return df
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
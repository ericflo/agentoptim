"""
Result analysis module for AgentOptim.

This module provides functionality for analyzing experiment results, computing
statistics, and suggesting optimizations for prompt variants.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from pydantic import BaseModel, Field

from agentoptim.utils import (
    RESULTS_DIR,
    generate_id,
    save_json,
    load_json,
    list_json_files,
    validate_action,
    validate_required_params,
    format_error,
    format_success,
    format_list,
    ValidationError,
)
from agentoptim.experiment import get_experiment, Experiment
from agentoptim.jobs import get_job, Job, JobResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VariantAnalysis(BaseModel):
    """Analysis results for a single prompt variant."""
    
    variant_id: str
    variant_name: str
    sample_size: int
    average_scores: Dict[str, float] = Field(default_factory=dict)
    median_scores: Dict[str, float] = Field(default_factory=dict)
    min_scores: Dict[str, float] = Field(default_factory=dict)
    max_scores: Dict[str, float] = Field(default_factory=dict)
    std_deviation: Dict[str, float] = Field(default_factory=dict)
    confidence_interval: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    percentiles: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    raw_scores: Dict[str, List[float]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResultsAnalysis(BaseModel):
    """Analysis results for an experiment."""
    
    id: str = Field(default_factory=generate_id)
    experiment_id: str
    job_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    variant_results: Dict[str, VariantAnalysis] = Field(default_factory=dict)
    best_variants: Dict[str, str] = Field(default_factory=dict)  # criterion -> variant_id
    overall_best_variant: Optional[str] = None  # variant_id
    comparison_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultsAnalysis":
        """Create from dictionary."""
        return cls(**data)


def get_analysis_path(analysis_id: str) -> str:
    """Get the file path for an analysis."""
    return os.path.join(RESULTS_DIR, f"{analysis_id}.json")


def list_analyses() -> List[Dict[str, Any]]:
    """List all available analyses."""
    analyses = []
    for analysis_id in list_json_files(RESULTS_DIR):
        analysis_data = load_json(get_analysis_path(analysis_id))
        if analysis_data:
            analyses.append(analysis_data)
    return analyses


def get_analysis(analysis_id: str) -> Optional[ResultsAnalysis]:
    """Get a specific analysis by ID."""
    analysis_data = load_json(get_analysis_path(analysis_id))
    if analysis_data:
        return ResultsAnalysis.from_dict(analysis_data)
    return None


def delete_analysis(analysis_id: str) -> bool:
    """Delete an analysis."""
    path = get_analysis_path(analysis_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def compute_variant_statistics(
    variant_id: str,
    variant_name: str,
    scores: Dict[str, List[float]]
) -> VariantAnalysis:
    """
    Compute statistical metrics for a prompt variant's scores.
    
    Args:
        variant_id: ID of the prompt variant
        variant_name: Name of the prompt variant
        scores: Dictionary of criterion names to lists of scores
        
    Returns:
        VariantAnalysis object with computed statistics
    """
    results = VariantAnalysis(
        variant_id=variant_id,
        variant_name=variant_name,
        sample_size=len(next(iter(scores.values()))) if scores else 0,
        raw_scores=scores
    )
    
    # No scores to analyze
    if not scores or not results.sample_size:
        return results
    
    for criterion, score_list in scores.items():
        if not score_list:
            continue
            
        # Convert to numpy array for calculations
        score_array = np.array(score_list)
        
        # Basic statistics
        results.average_scores[criterion] = float(np.mean(score_array))
        results.median_scores[criterion] = float(np.median(score_array))
        results.min_scores[criterion] = float(np.min(score_array))
        results.max_scores[criterion] = float(np.max(score_array))
        results.std_deviation[criterion] = float(np.std(score_array, ddof=1))
        
        # Confidence interval (95%)
        if len(score_array) > 1:
            sem = stats.sem(score_array)
            ci = stats.t.interval(
                0.95, 
                len(score_array) - 1, 
                loc=results.average_scores[criterion], 
                scale=sem
            )
            results.confidence_interval[criterion] = (float(ci[0]), float(ci[1]))
        
        # Percentiles
        percentiles = {
            "25th": float(np.percentile(score_array, 25)),
            "50th": float(np.percentile(score_array, 50)),
            "75th": float(np.percentile(score_array, 75)),
            "90th": float(np.percentile(score_array, 90))
        }
        results.percentiles[criterion] = percentiles
    
    return results


def analyze_experiment_results(
    experiment_id: str,
    job_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> ResultsAnalysis:
    """
    Analyze results from an experiment.
    
    Args:
        experiment_id: ID of the experiment to analyze
        job_id: Optional job ID if analyzing a specific job's results
        name: Optional name for the analysis
        description: Optional description
        
    Returns:
        ResultsAnalysis object with computed metrics
    """
    # Get the experiment data
    experiment = get_experiment(experiment_id)
    if not experiment:
        raise ValueError(f"Experiment with ID '{experiment_id}' not found")
    
    # Check if experiment has results
    if not experiment.results and not job_id:
        raise ValueError(f"Experiment '{experiment.name}' has no results to analyze")
    
    # If job_id is provided, get that specific job
    job = None
    if job_id:
        job = get_job(job_id)
        if not job:
            raise ValueError(f"Job with ID '{job_id}' not found")
        if job.experiment_id != experiment_id:
            raise ValueError(f"Job '{job_id}' is not associated with experiment '{experiment_id}'")
        if not job.results:
            raise ValueError(f"Job '{job_id}' has no results to analyze")
    
    # Create the analysis object
    analysis = ResultsAnalysis(
        experiment_id=experiment_id,
        job_id=job_id,
        name=name or f"Analysis of {experiment.name}",
        description=description,
    )
    
    # Group results by variant and criterion
    variant_to_scores = {}
    variant_name_map = {}
    
    # Get variant names for lookup
    for variant in experiment.prompt_variants:
        variant_name_map[variant.id] = variant.name
    
    # Process job results if available
    if job:
        for result in job.results:
            variant_id = result.variant_id
            
            if variant_id not in variant_to_scores:
                variant_to_scores[variant_id] = {}
            
            for criterion, score in result.scores.items():
                if criterion not in variant_to_scores[variant_id]:
                    variant_to_scores[variant_id][criterion] = []
                
                variant_to_scores[variant_id][criterion].append(score)
    
    # Process experiment results if available and no job_id provided
    elif experiment.results:
        for variant_id, scores in experiment.results.items():
            if isinstance(scores, dict) and "scores" in scores:
                variant_to_scores[variant_id] = scores["scores"]
    
    # Compute statistics for each variant
    for variant_id, scores in variant_to_scores.items():
        variant_name = variant_name_map.get(variant_id, f"Unknown variant ({variant_id})")
        variant_analysis = compute_variant_statistics(variant_id, variant_name, scores)
        analysis.variant_results[variant_id] = variant_analysis
    
    # Find the best variant for each criterion
    criteria = set()
    for variant_analysis in analysis.variant_results.values():
        criteria.update(variant_analysis.average_scores.keys())
    
    for criterion in criteria:
        best_score = -float('inf')
        best_variant_id = None
        
        for variant_id, variant_analysis in analysis.variant_results.items():
            if criterion in variant_analysis.average_scores:
                score = variant_analysis.average_scores[criterion]
                if score > best_score:
                    best_score = score
                    best_variant_id = variant_id
        
        if best_variant_id:
            analysis.best_variants[criterion] = best_variant_id
    
    # Determine overall best variant (highest average across all criteria)
    if analysis.variant_results:
        variant_overall_scores = {}
        
        for variant_id, variant_analysis in analysis.variant_results.items():
            if variant_analysis.average_scores:
                variant_overall_scores[variant_id] = sum(variant_analysis.average_scores.values()) / len(variant_analysis.average_scores)
        
        if variant_overall_scores:
            analysis.overall_best_variant = max(variant_overall_scores.items(), key=lambda x: x[1])[0]
    
    # Perform statistical significance tests between variants
    if len(analysis.variant_results) > 1 and criteria:
        comparison_results = {}
        
        # List of variant IDs for comparison
        variant_ids = list(analysis.variant_results.keys())
        
        for i in range(len(variant_ids)):
            for j in range(i + 1, len(variant_ids)):
                variant1_id = variant_ids[i]
                variant2_id = variant_ids[j]
                variant1_name = analysis.variant_results[variant1_id].variant_name
                variant2_name = analysis.variant_results[variant2_id].variant_name
                
                comparison_key = f"{variant1_name} vs {variant2_name}"
                comparison_results[comparison_key] = {}
                
                for criterion in criteria:
                    if (criterion in analysis.variant_results[variant1_id].raw_scores and
                        criterion in analysis.variant_results[variant2_id].raw_scores):
                        
                        scores1 = analysis.variant_results[variant1_id].raw_scores[criterion]
                        scores2 = analysis.variant_results[variant2_id].raw_scores[criterion]
                        
                        # Perform t-test if we have enough samples
                        if len(scores1) > 1 and len(scores2) > 1:
                            t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
                            
                            comparison_results[comparison_key][criterion] = {
                                "p_value": float(p_value),
                                "significant": p_value < 0.05,
                                "better_variant": variant1_id if np.mean(scores1) > np.mean(scores2) else variant2_id,
                                "mean_difference": float(np.mean(scores1) - np.mean(scores2)),
                            }
        
        analysis.comparison_metrics["pairwise_tests"] = comparison_results
    
    # Generate recommendations based on the analysis
    if analysis.overall_best_variant:
        best_variant_name = analysis.variant_results[analysis.overall_best_variant].variant_name
        analysis.recommendations.append(f"The best performing variant overall is '{best_variant_name}'")
    
    for criterion, variant_id in analysis.best_variants.items():
        variant_name = analysis.variant_results[variant_id].variant_name
        analysis.recommendations.append(f"For '{criterion}', the best variant is '{variant_name}'")
    
    # Add statistical insights
    if "pairwise_tests" in analysis.comparison_metrics:
        for comparison, results in analysis.comparison_metrics["pairwise_tests"].items():
            for criterion, stats_data in results.items():
                if stats_data["significant"]:
                    better_id = stats_data["better_variant"]
                    better_name = analysis.variant_results[better_id].variant_name
                    analysis.recommendations.append(
                        f"For '{criterion}', '{better_name}' is significantly better than the alternative (p={stats_data['p_value']:.3f})"
                    )
    
    # Save the analysis
    save_json(analysis.to_dict(), get_analysis_path(analysis.id))
    
    return analysis


def compare_analyses(analysis_ids: List[str]) -> Dict[str, Any]:
    """
    Compare multiple analyses.
    
    Args:
        analysis_ids: List of analysis IDs to compare
        
    Returns:
        Dictionary with comparison results
    """
    if len(analysis_ids) < 2:
        raise ValueError("Need at least two analyses to compare")
    
    analyses = []
    for analysis_id in analysis_ids:
        analysis = get_analysis(analysis_id)
        if not analysis:
            raise ValueError(f"Analysis with ID '{analysis_id}' not found")
        analyses.append(analysis)
    
    # Prepare comparison data
    comparison = {
        "analyses": [],
        "overall_best_variants": [],
        "performance_by_criterion": {},
    }
    
    # Collect basic info about each analysis
    for analysis in analyses:
        exp = get_experiment(analysis.experiment_id)
        exp_name = exp.name if exp else "Unknown experiment"
        
        comparison["analyses"].append({
            "id": analysis.id,
            "name": analysis.name,
            "experiment_id": analysis.experiment_id,
            "experiment_name": exp_name,
            "variant_count": len(analysis.variant_results),
        })
        
        if analysis.overall_best_variant:
            best_variant = analysis.variant_results[analysis.overall_best_variant]
            comparison["overall_best_variants"].append({
                "analysis_id": analysis.id,
                "variant_id": analysis.overall_best_variant,
                "variant_name": best_variant.variant_name,
                "average_scores": best_variant.average_scores,
            })
    
    # Compare performance by criterion across analyses
    all_criteria = set()
    for analysis in analyses:
        for variant in analysis.variant_results.values():
            all_criteria.update(variant.average_scores.keys())
    
    for criterion in all_criteria:
        comparison["performance_by_criterion"][criterion] = []
        
        for analysis in analyses:
            best_id = analysis.best_variants.get(criterion)
            if best_id and best_id in analysis.variant_results:
                best_variant = analysis.variant_results[best_id]
                score = best_variant.average_scores.get(criterion)
                
                if score is not None:
                    comparison["performance_by_criterion"][criterion].append({
                        "analysis_id": analysis.id,
                        "variant_id": best_id,
                        "variant_name": best_variant.variant_name,
                        "score": score,
                    })
    
    return comparison


def analyze_results(
    action: str,
    experiment_id: Optional[str] = None,
    job_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    analysis_ids: Optional[List[str]] = None,
) -> str:
    """
    Analyze experiment results to find the best performing prompts.
    
    Args:
        action: One of "analyze", "list", "get", "delete", "compare"
        experiment_id: Required for analyze
        job_id: Optional job ID for analyze
        analysis_id: Required for get, delete
        name: Optional name for the analysis
        description: Optional description
        analysis_ids: Required for compare
        
    Returns:
        A string response describing the result
    """
    valid_actions = ["analyze", "list", "get", "delete", "compare"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "list":
            analyses = list_analyses()
            return format_list(analyses)
        
        elif action == "get":
            validate_required_params({"analysis_id": analysis_id}, ["analysis_id"])
            analysis = get_analysis(analysis_id)
            if not analysis:
                return format_error(f"Analysis with ID '{analysis_id}' not found")
            
            # Format the analysis output
            analysis_dict = analysis.to_dict()
            result = [
                f"Analysis: {analysis_dict['name']} (ID: {analysis_dict['id']})",
                f"Experiment: {analysis_dict['experiment_id']}",
            ]
            
            if analysis_dict.get('description'):
                result.append(f"Description: {analysis_dict['description']}")
            
            if analysis_dict.get('job_id'):
                result.append(f"Job: {analysis_dict['job_id']}")
            
            # Add variant results
            result.append(f"\nVariant Results ({len(analysis_dict['variant_results'])}):")
            for variant_id, variant_data in analysis_dict["variant_results"].items():
                result.append(f"\n- {variant_data['variant_name']} (ID: {variant_id})")
                result.append(f"  Sample size: {variant_data['sample_size']}")
                
                # Show average scores
                if variant_data.get('average_scores'):
                    result.append("  Average scores:")
                    for criterion, score in variant_data['average_scores'].items():
                        result.append(f"    {criterion}: {score:.2f}")
            
            # Add best variants
            if analysis_dict.get('best_variants'):
                result.append("\nBest Variants by Criterion:")
                for criterion, variant_id in analysis_dict['best_variants'].items():
                    variant_name = next(
                        (v['variant_name'] for v_id, v in analysis_dict['variant_results'].items() 
                         if v_id == variant_id),
                        f"Unknown ({variant_id})"
                    )
                    result.append(f"- {criterion}: {variant_name}")
            
            # Add overall best variant
            if analysis_dict.get('overall_best_variant'):
                overall_id = analysis_dict['overall_best_variant']
                overall_name = next(
                    (v['variant_name'] for v_id, v in analysis_dict['variant_results'].items() 
                     if v_id == overall_id),
                    f"Unknown ({overall_id})"
                )
                result.append(f"\nOverall Best Variant: {overall_name}")
            
            # Add recommendations
            if analysis_dict.get('recommendations'):
                result.append("\nRecommendations:")
                for rec in analysis_dict['recommendations']:
                    result.append(f"- {rec}")
            
            return "\n".join(result)
        
        elif action == "analyze":
            validate_required_params({"experiment_id": experiment_id}, ["experiment_id"])
            
            analysis = analyze_experiment_results(
                experiment_id=experiment_id,
                job_id=job_id,
                name=name,
                description=description,
            )
            
            return format_success(
                f"Analysis '{analysis.name}' created with ID: {analysis.id}\n"
                f"Overall best variant: {analysis.variant_results[analysis.overall_best_variant].variant_name if analysis.overall_best_variant else 'None'}"
            )
        
        elif action == "delete":
            validate_required_params({"analysis_id": analysis_id}, ["analysis_id"])
            
            if delete_analysis(analysis_id):
                return format_success(f"Analysis with ID '{analysis_id}' deleted")
            else:
                return format_error(f"Analysis with ID '{analysis_id}' not found")
        
        elif action == "compare":
            validate_required_params({"analysis_ids": analysis_ids}, ["analysis_ids"])
            
            if not isinstance(analysis_ids, list) or len(analysis_ids) < 2:
                return format_error("analysis_ids must be a list with at least two IDs")
            
            comparison = compare_analyses(analysis_ids)
            
            # Format the comparison output
            result = ["Analysis Comparison:"]
            
            # Add analyses info
            result.append("\nAnalyses:")
            for analysis_info in comparison["analyses"]:
                result.append(f"- {analysis_info['name']} (ID: {analysis_info['id']})")
                result.append(f"  Experiment: {analysis_info['experiment_name']}")
                result.append(f"  Variants: {analysis_info['variant_count']}")
            
            # Add best variant comparison
            if comparison.get("overall_best_variants"):
                result.append("\nBest Overall Variants:")
                for best in comparison["overall_best_variants"]:
                    analysis_name = next(
                        (a['name'] for a in comparison['analyses'] if a['id'] == best['analysis_id']),
                        f"Analysis {best['analysis_id']}"
                    )
                    result.append(f"- {analysis_name}: {best['variant_name']}")
            
            # Add criterion comparison
            if comparison.get("performance_by_criterion"):
                result.append("\nPerformance by Criterion:")
                for criterion, performances in comparison["performance_by_criterion"].items():
                    result.append(f"\n{criterion}:")
                    
                    # Sort by score (descending)
                    sorted_performances = sorted(performances, key=lambda x: x["score"], reverse=True)
                    
                    for perf in sorted_performances:
                        analysis_name = next(
                            (a['name'] for a in comparison['analyses'] if a['id'] == perf['analysis_id']),
                            f"Analysis {perf['analysis_id']}"
                        )
                        result.append(f"- {analysis_name}: {perf['variant_name']} ({perf['score']:.2f})")
            
            return "\n".join(result)
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")
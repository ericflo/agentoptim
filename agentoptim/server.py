"""MCP server implementation for AgentOptim."""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Optional, Any

from mcp.server.fastmcp import FastMCP

from agentoptim.evaluation import manage_evaluation
from agentoptim.dataset import manage_dataset
from agentoptim.experiment import manage_experiment
from agentoptim.jobs import manage_job
from agentoptim.analysis import analyze_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), "agentoptim.log")),
    ],
)

logger = logging.getLogger("agentoptim")

# Initialize FastMCP server
mcp = FastMCP("agentoptim")


@mcp.tool()
async def manage_evaluation_tool(
    action: str,
    evaluation_id: Optional[str] = None,
    name: Optional[str] = None,
    template: Optional[str] = None,
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> str:
    """
    Manage evaluation definitions for assessing response quality.
    
    Args:
        action: One of "create", "list", "get", "update", "delete"
        evaluation_id: Required for get, update, delete
        name: Required for create
        template: Required for create, optional for update
        questions: Required for create, optional for update
        description: Optional description
    """
    logger.info(f"manage_evaluation_tool called with action={action}")
    try:
        result = manage_evaluation(
            action=action,
            evaluation_id=evaluation_id,
            name=name,
            template=template,
            questions=questions,
            description=description,
        )
        return result
    except Exception as e:
        logger.error(f"Error in manage_evaluation_tool: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


@mcp.tool()
async def manage_dataset_tool(
    action: str,
    dataset_id: Optional[str] = None,
    name: Optional[str] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    description: Optional[str] = None,
    source: Optional[str] = None,
    tags: Optional[List[str]] = None,
    filepath: Optional[str] = None,
    input_field: Optional[str] = None,
    output_field: Optional[str] = None,
    test_ratio: Optional[float] = None,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Manage datasets for experiments and evaluations.
    
    Args:
        action: One of "create", "list", "get", "update", "delete", "split", "sample", "import"
        dataset_id: Required for get, update, delete, split, sample
        name: Required for create and import
        items: Required for create, optional for update
        description: Optional description
        source: Optional source information
        tags: Optional list of tags
        filepath: Required for import
        input_field: Field name for input when importing
        output_field: Field name for expected output when importing
        test_ratio: Ratio for splitting dataset (default: 0.2)
        sample_size: Number of items to sample
        seed: Random seed for reproducibility
    """
    logger.info(f"manage_dataset_tool called with action={action}")
    try:
        result = manage_dataset(
            action=action,
            dataset_id=dataset_id,
            name=name,
            items=items,
            description=description,
            source=source,
            tags=tags,
            filepath=filepath,
            input_field=input_field,
            output_field=output_field,
            test_ratio=test_ratio,
            sample_size=sample_size,
            seed=seed,
        )
        return result
    except Exception as e:
        logger.error(f"Error in manage_dataset_tool: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


@mcp.tool()
async def manage_experiment_tool(
    action: str,
    experiment_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    dataset_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    prompt_variants: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    new_name: Optional[str] = None,
) -> str:
    """
    Manage experiments for prompt optimization.
    
    Args:
        action: One of "create", "list", "get", "update", "delete", "duplicate"
        experiment_id: Required for get, update, delete, duplicate
        name: Required for create
        description: Optional description
        dataset_id: ID of dataset to use (required for create)
        evaluation_id: ID of evaluation to use (required for create)
        prompt_variants: List of prompt variants to test (required for create)
        model_name: Name of model to use (required for create)
        temperature: Model temperature setting
        max_tokens: Maximum tokens to generate
        status: Experiment status
        results: Experiment results
        metadata: Additional metadata
        new_name: New name for duplicated experiment
    """
    logger.info(f"manage_experiment_tool called with action={action}")
    try:
        result = manage_experiment(
            action=action,
            experiment_id=experiment_id,
            name=name,
            description=description,
            dataset_id=dataset_id,
            evaluation_id=evaluation_id,
            prompt_variants=prompt_variants,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            status=status,
            results=results,
            metadata=metadata,
            new_name=new_name,
        )
        return result
    except Exception as e:
        logger.error(f"Error in manage_experiment_tool: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


@mcp.tool()
async def run_job_tool(
    action: str,
    job_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge_parameters: Optional[Dict[str, Any]] = None,
    max_parallel: Optional[int] = None,
) -> str:
    """
    Execute and manage jobs for prompt optimization experiments.
    
    Args:
        action: One of "create", "list", "get", "delete", "run", "cancel"
        job_id: Required for get, delete, run, cancel
        experiment_id: Required for create, optional for list
        dataset_id: Required for create
        evaluation_id: Required for create
        judge_model: Optional model to use for evaluation
        judge_parameters: Optional parameters for the judge model
        max_parallel: Maximum number of parallel tasks
    """
    logger.info(f"run_job_tool called with action={action}")
    try:
        result = manage_job(
            action=action,
            job_id=job_id,
            experiment_id=experiment_id,
            dataset_id=dataset_id,
            evaluation_id=evaluation_id,
            judge_model=judge_model,
            judge_parameters=judge_parameters,
            max_parallel=max_parallel,
        )
        
        # Special handling for the "run" action to handle async execution
        if action == "run" and job_id:
            # Start a background task to run the job
            # The manage_job function already handles this properly
            return f"Job {job_id} started. Use 'get' action to check progress."
        
        return result
    except Exception as e:
        logger.error(f"Error in run_job_tool: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


@mcp.tool()
async def analyze_results_tool(
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
    """
    logger.info(f"analyze_results_tool called with action={action}")
    try:
        result = analyze_results(
            action=action,
            experiment_id=experiment_id,
            job_id=job_id,
            analysis_id=analysis_id,
            name=name,
            description=description,
            analysis_ids=analysis_ids,
        )
        return result
    except Exception as e:
        logger.error(f"Error in analyze_results_tool: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
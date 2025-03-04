"""MCP server implementation for AgentOptim."""

import os
import sys
import logging
from typing import Dict, List, Optional, Any

from mcp.server.fastmcp import FastMCP

from agentoptim.evaluation import manage_evaluation
from agentoptim.dataset import manage_dataset
# Import other module functions as they're implemented
# from agentoptim.experiment import manage_experiment
# from agentoptim.jobs import run_job
# from agentoptim.analysis import analyze_results

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


# Register other tools as they're implemented
# @mcp.tool()
# async def manage_experiment_tool():
#     pass

# @mcp.tool()
# async def run_job_tool():
#     pass

# @mcp.tool()
# async def analyze_results_tool():
#     pass


def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
"""MCP server implementation for AgentOptim."""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any

from mcp.server.fastmcp import FastMCP

from agentoptim.evaluation import manage_evaluation
from agentoptim.dataset import manage_dataset
from agentoptim.experiment import manage_experiment
from agentoptim.jobs import manage_job, get_job
from agentoptim.analysis import analyze_results

# Import necessary utilities
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Ensure data directories exist
ensure_data_directories()

# Configure logging
log_file_path = os.path.join(DATA_DIR, "agentoptim.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path),
    ],
)

logger = logging.getLogger("agentoptim")
logger.info(f"Logging to {log_file_path}")

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
    
    This tool helps manage evaluations, which define a set of criteria for judging response quality.
    An evaluation consists of a template and a list of questions that will be used to evaluate
    responses. The template formats the input, response, and evaluation question for the judge model.
    
    Args:
        action: The operation to perform. Must be one of:
               - "create" - Create a new evaluation
               - "list" - List all available evaluations
               - "get" - Get details of a specific evaluation
               - "update" - Update an existing evaluation
               - "delete" - Delete an evaluation
               
        evaluation_id: The unique identifier of the evaluation.
                      Required for "get", "update", and "delete" actions.
                      Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                      
        name: The name of the evaluation.
              Required for "create" action.
              Example: "Response Clarity Evaluation"
              
        template: The template string that formats the input for the judge model.
                 Required for "create" action, optional for "update".
                 Must include placeholders for input, question, and optional response.
                 Example: "Input: {input}\nResponse: {response}\nQuestion: {question}"
                 
        questions: A list of questions to evaluate responses against.
                  Required for "create" action, optional for "update".
                  Each question should be answerable with yes/no.
                  Example: ["Is the response clear and concise?", "Does the response answer the question?"]
                  
        description: Optional description of the evaluation purpose.
                    Example: "Evaluates responses for clarity, conciseness, and helpfulness"
    
    Returns:
        For "list" action: A formatted list of all evaluations with their IDs and names
        For "get" action: Detailed information about the specified evaluation
        For "create" action: Confirmation message with the new evaluation ID
        For "update" action: Confirmation message that the evaluation was updated
        For "delete" action: Confirmation message that the evaluation was deleted
        
    Error messages will clearly indicate what went wrong, such as:
    - "Invalid action: 'foo'. Valid actions are: create, list, get, update, delete"
    - "Missing required parameters: evaluation_id"
    - "Evaluation with ID 'xyz' not found"
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
        
        # If result is a dictionary with a formatted_message, return that for MCP
        if isinstance(result, dict) and "formatted_message" in result:
            return result["formatted_message"]
            
        return result
    except Exception as e:
        logger.error(f"Error in manage_evaluation_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"Error: Invalid action '{action}'. Valid actions are: create, list, get, update, delete\n\nExamples:\n- manage_evaluation_tool(action=\"create\", name=\"Response Quality\", template=\"...\", questions=[...])\n- manage_evaluation_tool(action=\"list\")\n- manage_evaluation_tool(action=\"get\", evaluation_id=\"...\")"
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "create":
                example = {
                    "action": "create",
                    "name": "Response Quality Evaluation",
                    "template": "Input: {input}\nResponse: {response}\nQuestion: {question}",
                    "questions": [
                        "Is the response clear and concise?",
                        "Does the response fully answer the question?",
                        "Is the response accurate?"
                    ],
                    "description": "Evaluates responses for clarity, completeness, and accuracy"
                }
                return f"Error: {error_msg}.\n\nThe 'create' action requires name, template, and questions parameters.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action in ["get", "update", "delete"]:
                example = {
                    "action": action,
                    "evaluation_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                }
                # Add fields for update
                if action == "update":
                    example["template"] = "New template text with {input} and {question}"
                    example["questions"] = ["Updated question 1", "Updated question 2"]
                
                return f"Error: {error_msg}.\n\nThe '{action}' action requires the evaluation_id parameter.\n\nExample:\n{json.dumps(example, indent=2)}"
            else:
                return f"Error: {error_msg}. Please check the tool documentation for required parameters."
        elif "not found" in error_msg:
            return f"Error: {error_msg}.\n\nThe evaluation_id you provided doesn't exist. Use manage_evaluation_tool(action=\"list\") to see available evaluations."
        elif "validate_string" in error_msg and "name" in error_msg:
            return f"Error: Invalid name parameter. The name must be a non-empty string.\n\nExample: name=\"Response Quality Evaluation\""
        elif "template" in error_msg:
            return f"Error: {error_msg}.\n\nThe template should include placeholders like {{input}}, {{response}}, and {{question}}.\n\nExample template: \"Input: {{input}}\\nResponse: {{response}}\\nQuestion: {{question}}\""
        elif "questions" in error_msg:
            if "must be a list" in error_msg:
                return f"Error: {error_msg}.\n\nQuestions must be provided as a list of strings.\n\nExample: questions=[\"Is the response clear?\", \"Does the response answer the question?\"]"
            elif "maximum" in error_msg:
                return f"Error: {error_msg}.\n\nYou have provided too many questions. Please limit your questions to 100 or fewer."
            else:
                return f"Error: {error_msg}.\n\nEach question should be a string that can be answered with yes/no by the judge model."
        else:
            return f"Error: {error_msg}"


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
    Manage datasets for prompt optimization experiments and evaluations.
    
    This tool helps you create, manage, and manipulate datasets that can be used for testing
    different prompt variations. Datasets contain examples that include inputs (like user questions)
    and optionally expected outputs (for comparison purposes).
    
    Args:
        action: The operation to perform. Must be one of:
               - "create" - Create a new dataset with provided items
               - "list" - List all available datasets
               - "get" - Get details of a specific dataset
               - "update" - Update an existing dataset
               - "delete" - Delete a dataset
               - "split" - Split a dataset into training and testing subsets
               - "sample" - Create a smaller sample from a larger dataset
               - "import" - Import a dataset from a file
               
        dataset_id: The unique identifier of the dataset.
                   Required for "get", "update", "delete", "split", and "sample" actions.
                   Example: "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p"
                   
        name: The name of the dataset.
              Required for "create" and "import" actions.
              Example: "Customer Support Questions"
              
        items: A list of dictionary items containing the dataset examples.
              Required for "create" action, optional for "update".
              Each item should be a dictionary with consistent keys.
              Example: [{"input": "How do I reset my password?", "output": "To reset your password..."}]
              
        description: Optional description of the dataset.
                    Example: "Contains 500 customer support questions with expert answers"
                    
        source: Optional information about the dataset source.
               Example: "Exported from our support ticketing system"
               
        tags: Optional list of tags for categorizing the dataset.
             Example: ["customer-support", "account-issues", "passwords"]
             
        filepath: Path to a file to import as a dataset.
                 Required for "import" action.
                 Supports JSON and CSV files.
                 Example: "/path/to/dataset.json"
                 
        input_field: Field name to use as input when importing from file.
                    Example: "question" or "user_message"
                    
        output_field: Field name to use as expected output when importing.
                     Example: "answer" or "response"
                     
        test_ratio: Fraction of data to use for testing when splitting.
                   Must be between 0.0 and 1.0. Default is 0.2 (20%).
                   Example: 0.3 (for 30% test, 70% train)
                   
        sample_size: Number of items to include when sampling.
                    Example: 50 (to create a dataset with 50 randomly selected items)
                    
        seed: Random seed for reproducible sampling or splitting.
             Example: 42
    
    Returns:
        For "list" action: A formatted list of all datasets with IDs and names
        For "get" action: Detailed information about the specified dataset
        For "create"/"import" action: Confirmation message with the new dataset ID
        For "update" action: Confirmation message that the dataset was updated
        For "delete" action: Confirmation message that the dataset was deleted
        For "split" action: Information about the resulting train/test datasets
        For "sample" action: Information about the sampled dataset
        
    Error messages will clearly indicate what went wrong, such as:
    - "Invalid action: 'foo'. Valid actions are: create, list, get, update, delete, split, sample, import"
    - "Missing required parameters: dataset_id"
    - "Dataset with ID 'xyz' not found"
    - "Invalid test_ratio: 1.5. Must be between 0.0 and 1.0"
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
        
        # If result is a dictionary with a formatted_message, return that for MCP
        if isinstance(result, dict) and "formatted_message" in result:
            return result["formatted_message"]
            
        return result
    except Exception as e:
        logger.error(f"Error in manage_dataset_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"Error: Invalid action '{action}'. Valid actions are: create, list, get, update, delete, split, sample, import\n\nExamples:\n- manage_dataset_tool(action=\"create\", name=\"Customer Questions\", items=[...])\n- manage_dataset_tool(action=\"list\")\n- manage_dataset_tool(action=\"get\", dataset_id=\"...\")"
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "create":
                example = {
                    "action": "create",
                    "name": "Customer Support Questions",
                    "items": [
                        {"input": "How do I reset my password?", "expected_output": "To reset your password..."},
                        {"input": "Where can I update my shipping address?", "expected_output": "You can update your shipping address..."}
                    ],
                    "description": "Dataset of common customer support inquiries"
                }
                return f"Error: {error_msg}.\n\nThe 'create' action requires name and items parameters.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action == "import":
                example = {
                    "action": "import",
                    "name": "Survey Responses",
                    "filepath": "/path/to/data.json",
                    "input_field": "question",
                    "output_field": "answer"
                }
                return f"Error: {error_msg}.\n\nThe 'import' action requires name, filepath, and typically input_field parameters.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action in ["get", "update", "delete"]:
                example = {
                    "action": action,
                    "dataset_id": "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p"
                }
                # Add fields for update
                if action == "update":
                    example["name"] = "Updated Dataset Name"
                    example["description"] = "Updated dataset description"
                
                return f"Error: {error_msg}.\n\nThe '{action}' action requires the dataset_id parameter.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action == "split":
                example = {
                    "action": "split",
                    "dataset_id": "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p",
                    "test_ratio": 0.2,
                    "seed": 42
                }
                return f"Error: {error_msg}.\n\nThe 'split' action requires the dataset_id parameter and optionally test_ratio.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action == "sample":
                example = {
                    "action": "sample",
                    "dataset_id": "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p",
                    "sample_size": 50,
                    "seed": 42
                }
                return f"Error: {error_msg}.\n\nThe 'sample' action requires dataset_id and sample_size parameters.\n\nExample:\n{json.dumps(example, indent=2)}"
            else:
                return f"Error: {error_msg}. Please check the tool documentation for required parameters."
        elif "not found" in error_msg:
            return f"Error: {error_msg}.\n\nThe dataset_id you provided doesn't exist. Use manage_dataset_tool(action=\"list\") to see available datasets."
        elif "test_ratio" in error_msg:
            return f"Error: Invalid test_ratio value: {test_ratio}.\n\nThe test_ratio must be a number between 0.0 and 1.0, representing the proportion of data to use for testing.\n\nExample: test_ratio=0.2 (for 20% test, 80% train)"
        elif "sample_size" in error_msg and "exceeds" in error_msg:
            return f"Error: {error_msg}.\n\nPlease specify a smaller sample_size or use a larger dataset. The sample size cannot be larger than the dataset size."
        elif "filepath" in error_msg and "not found" in error_msg:
            return f"Error: File not found: {filepath}.\n\nPlease check that:\n1. The file path is correct and absolute\n2. The file exists and is readable\n3. The file is in a supported format (JSON or CSV)"
        elif "items" in error_msg:
            if "must be a list" in error_msg:
                return f"Error: {error_msg}.\n\nItems must be provided as a list of dictionaries, where each dictionary represents a dataset example.\n\nExample: items=[{{\"input\": \"Question 1\", \"output\": \"Answer 1\"}}, {{\"input\": \"Question 2\", \"output\": \"Answer 2\"}}]"
            else:
                return f"Error: {error_msg}.\n\nEach item should be a dictionary with consistent keys for input and expected output."
        elif "input_field" in error_msg or "output_field" in error_msg:
            return f"Error: {error_msg}.\n\nWhen importing data, you need to specify which fields in your data file contain the inputs and expected outputs.\n\nExample: input_field=\"question\", output_field=\"answer\""
        else:
            return f"Error: {error_msg}"


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
    Manage experiments for testing and optimizing different prompt variations.
    
    This tool helps you create and manage experiments that test different prompt variants
    against a dataset using specified evaluation criteria. An experiment connects a dataset, 
    an evaluation, and multiple prompt variations to be tested.
    
    Args:
        action: The operation to perform. Must be one of:
               - "create" - Create a new experiment
               - "list" - List all available experiments
               - "get" - Get details of a specific experiment
               - "update" - Update an existing experiment
               - "delete" - Delete an experiment
               - "duplicate" - Create a copy of an existing experiment
               
        experiment_id: The unique identifier of the experiment.
                      Required for "get", "update", "delete", and "duplicate" actions.
                      Example: "9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r"
                      
        name: The name of the experiment.
              Required for "create" action.
              Example: "Customer Service Tone Optimization"
              
        description: Optional description of the experiment purpose.
                    Example: "Testing different tones for customer service responses"
                    
        dataset_id: ID of the dataset to use for the experiment.
                   Required for "create" action.
                   Must reference an existing dataset created with manage_dataset_tool.
                   Example: "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p"
                   
        evaluation_id: ID of the evaluation to use for judging responses.
                      Required for "create" action.
                      Must reference an existing evaluation created with manage_evaluation_tool.
                      Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                      
        prompt_variants: List of prompt variations to test in the experiment.
                        Required for "create" action, optional for "update".
                        Each variant should be a dictionary with at least "name" and "content" keys.
                        Example: [
                            {
                                "name": "formal_tone",
                                "content": "You are a customer service representative. Use formal language..."
                            },
                            {
                                "name": "casual_tone",
                                "content": "You're a friendly support agent. Keep things casual..."
                            }
                        ]
                        
        model_name: Name of the model to use for generating responses.
                   Required for "create" action.
                   Example: "claude-3-opus-20240229"
                   
        temperature: Temperature setting for the model (controls randomness).
                    Optional, defaults to 0.7.
                    Must be between 0.0 and 1.0.
                    Example: 0.5
                    
        max_tokens: Maximum number of tokens to generate in responses.
                   Optional, defaults to model-specific value.
                   Example: 1000
                   
        status: Current status of the experiment.
               For "update" action only.
               Valid values: "draft", "running", "completed", "failed", "cancelled"
               Example: "running"
               
        results: Results data for the experiment.
                For internal use or "update" action.
                Example: {"variant_scores": {"formal_tone": 0.85, "casual_tone": 0.92}}
                
        metadata: Additional metadata for the experiment.
                 Optional for "create" and "update" actions.
                 Example: {"priority": "high", "requested_by": "marketing_team"}
                 
        new_name: New name for the duplicated experiment.
                 Required for "duplicate" action.
                 Example: "Customer Service Tone Optimization V2"
    
    Returns:
        For "list" action: A formatted list of all experiments with IDs and names
        For "get" action: Detailed information about the specified experiment
        For "create" action: Confirmation message with the new experiment ID
        For "update" action: Confirmation message that the experiment was updated
        For "delete" action: Confirmation message that the experiment was deleted
        For "duplicate" action: Confirmation message with the new experiment ID
        
    Error messages will clearly indicate what went wrong, such as:
    - "Invalid action: 'foo'. Valid actions are: create, list, get, update, delete, duplicate"
    - "Missing required parameters: experiment_id"
    - "Experiment with ID 'xyz' not found"
    - "Dataset with ID 'abc' not found"
    - "Evaluation with ID 'def' not found"
    - "Each prompt variant must have 'name' and 'content' fields"
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
        
        # If result is a dictionary with a formatted_message, return that for MCP
        if isinstance(result, dict) and "formatted_message" in result:
            return result["formatted_message"]
            
        return result
    except Exception as e:
        logger.error(f"Error in manage_experiment_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"Error: Invalid action '{action}'. Valid actions are: create, list, get, update, delete, duplicate"
        elif "required parameters" in error_msg:
            return f"Error: {error_msg}. Please check the tool documentation for required parameters."
        elif "not found" in error_msg and "experiment" in error_msg:
            return f"Error: {error_msg}. Please check that the experiment_id exists or use the 'list' action to see available experiments."
        elif "not found" in error_msg and "dataset" in error_msg:
            return f"Error: {error_msg}. Please use manage_dataset_tool to list or create datasets first."
        elif "not found" in error_msg and "evaluation" in error_msg:
            return f"Error: {error_msg}. Please use manage_evaluation_tool to list or create evaluations first."
        elif "prompt_variants" in error_msg or "prompt variant" in error_msg:
            basic_example = {
                "name": "formal_tone", 
                "type": "system", 
                "content": "You are a formal customer service representative..."
            }
            
            # Create a more detailed error message based on the specific error
            if "missing required fields" in error_msg:
                # The error message already contains the specific missing fields
                return f"Error: {error_msg}\n\nPlease provide the missing fields. Example prompt variant:\n{json.dumps(basic_example, indent=2)}"
            elif "must be a non-empty list" in error_msg:
                # The prompt_variants parameter must be a list
                return f"Error: {error_msg}\n\nThe prompt_variants parameter must be a list of variant objects. Example:\n{json.dumps([basic_example], indent=2)}"
            else:
                # Generic prompt variant error
                return f"Error: {error_msg}\n\nEach prompt variant must include at least 'name' and 'content' (or 'template') fields. Example:\n{json.dumps(basic_example, indent=2)}"
        elif "temperature" in error_msg:
            return f"Error: Invalid temperature: {temperature}. Must be between 0.0 and 1.0."
        else:
            return f"Error: {error_msg}"


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
    auto_start: Optional[bool] = True,
) -> str:
    """
    Execute and manage jobs that run prompt optimization experiments.
    
    This tool allows you to create, run, and manage jobs that execute experiments
    with different prompt variants against datasets. Each job runs an experiment
    by generating responses from each prompt variant for the dataset items, then
    evaluating those responses using the specified evaluation criteria.
    
    Args:
        action: The operation to perform. Must be one of:
               - "create" - Create a new job and automatically start it (unless auto_start=False)
               - "list" - List all available jobs
               - "get" - Get details and status of a specific job
               - "delete" - Delete a job
               - "run" - Start execution of a job
               - "cancel" - Cancel a running job
               
        job_id: The unique identifier of the job.
               Required for "get", "delete", "run", and "cancel" actions.
               Example: "7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c"
               
        experiment_id: ID of the experiment to run.
                      Required for "create" action.
                      Must reference an existing experiment created with manage_experiment_tool.
                      Example: "9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r"
                      
        dataset_id: ID of the dataset to use for the job.
                   Required for "create" action.
                   Must reference an existing dataset created with manage_dataset_tool.
                   Example: "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p"
                   
        evaluation_id: ID of the evaluation to use for the job.
                      Required for "create" action.
                      Must reference an existing evaluation created with manage_evaluation_tool.
                      Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                      
        judge_model: Name of the model to use for evaluation (acts as judge).
                    Optional for "create" action.
                    If not provided, the default judge model will be used.
                    Example: "claude-3-haiku-20240307"
                    
        judge_parameters: Parameters for the judge model.
                         Optional for "create" action.
                         Example: {"temperature": 0.1, "max_tokens": 500}
                         
        max_parallel: Maximum number of tasks to run in parallel.
                     Optional for "create" and "run" actions.
                     Higher values may speed up execution but increase resource usage.
                     Example: 5
                     
        auto_start: Whether to automatically start the job after creation.
                   Only applies to "create" action.
                   Defaults to True.
                   Example: True
    
    Returns:
        For "list" action: A formatted list of all jobs with their IDs and statuses
        For "get" action: Detailed information about the job including current status
        For "create" action: Confirmation message with the new job ID and that it has started
        For "run" action: Message confirming the job has started
        For "cancel" action: Message confirming the job has been cancelled
        For "delete" action: Confirmation message that the job was deleted
        
    Error messages will clearly indicate what went wrong, such as:
    - "Invalid action: 'foo'. Valid actions are: create, list, get, delete, run, cancel"
    - "Missing required parameters: job_id"
    - "Job with ID 'xyz' not found"
    - "Experiment with ID 'abc' not found"
    - "Dataset with ID 'def' not found"
    - "Evaluation with ID 'ghi' not found"
    - "Job is already running"
    - "Job is not in a runnable state"
    
    Important Note:
    Jobs are executed asynchronously. When a job is created, it automatically starts running
    (unless auto_start=False). Use the "get" action to check its progress. Jobs may take several 
    minutes to complete depending on the size of the dataset and the number of prompt variants.
    """
    logger.info(f"run_job_tool called with action={action}")
    try:
        # If creating a job, handle the auto-start behavior
        if action == "create":
            # Create the job
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
            
            # If job creation was successful and auto_start is True, start it automatically
            if isinstance(result, dict) and result.get("status") == "success" and auto_start:
                new_job_id = result.get("job", {}).get("job_id")
                if new_job_id:
                    # Start the job
                    run_result = manage_job(
                        action="run",
                        job_id=new_job_id,
                        max_parallel=max_parallel
                    )
                    
                    # Return a combined result message
                    api_base = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234")
                    is_local_env = ("localhost" in api_base or 
                                   "127.0.0.1" in api_base or
                                   ".local" in api_base or 
                                   api_base.startswith("http://0.0.0.0"))
                    
                    model_info = f" using judge model '{result.get('job', {}).get('judge_model', 'unknown')}'"
                    
                    if is_local_env:
                        success_message = (
                            f"Job created with ID: {new_job_id} and started{model_info}.\n"
                            f"Make sure your local model server is running at {api_base}.\n"
                            f"Use run_job_tool(action=\"get\", job_id=\"{new_job_id}\") to check progress."
                        )
                    else:
                        success_message = (
                            f"Job created with ID: {new_job_id} and started{model_info}.\n"
                            f"Use run_job_tool(action=\"get\", job_id=\"{new_job_id}\") to check progress."
                        )
                    
                    # Include the job in the response
                    return {
                        "status": "success",
                        "message": success_message,
                        "job": result.get("job", {})
                    }
                
            # Return the original result if auto_start is False or if something went wrong
            return result
        elif action == "run" and job_id:
            # Run the job
            result = manage_job(
                action=action,
                job_id=job_id,
                max_parallel=max_parallel
            )
            
            # Get the job to include model information in the message
            try:
                job = get_job(job_id)
                model_info = f" using judge model '{job.judge_model}'"
                
                # Check if this is a local/development environment
                api_base = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234")
                is_local_env = ("localhost" in api_base or 
                               "127.0.0.1" in api_base or
                               ".local" in api_base or 
                               api_base.startswith("http://0.0.0.0"))
                
                if is_local_env:
                    # Add a warning about local development usage
                    return (
                        f"Job {job_id} started{model_info}. "
                        f"Make sure your local model server is running at {api_base}. "
                        f"Use 'get' action to check progress and get results when complete."
                    )
                else:
                    # Standard response
                    return f"Job {job_id} started{model_info}. Use 'get' action to check progress and get results when complete."
            except:
                # Fallback if we can't get job details
                return f"Job {job_id} started. Use 'get' action to check progress and get results when complete."
        else:
            # Handle all other actions normally
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
            
            # If result is a dictionary with a formatted_message, return that for MCP
            if isinstance(result, dict) and "formatted_message" in result:
                return result["formatted_message"]
                
            return result
    except Exception as e:
        logger.error(f"Error in run_job_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"Error: Invalid action '{action}'. Valid actions are: create, list, get, delete, run, cancel\n\nExample: run_job_tool(action=\"create\", experiment_id=\"...\", dataset_id=\"...\", evaluation_id=\"...\")"
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "create":
                example = {
                    "action": "create",
                    "experiment_id": "9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
                    "dataset_id": "8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p",
                    "evaluation_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                }
                return f"Error: {error_msg}.\n\nThe 'create' action requires experiment_id, dataset_id, and evaluation_id parameters.\n\nExample: {json.dumps(example, indent=2)}"
            elif action in ["get", "delete", "run", "cancel"]:
                example = {"action": action, "job_id": "7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c"}
                return f"Error: {error_msg}.\n\nThe '{action}' action requires the job_id parameter.\n\nExample: {json.dumps(example, indent=2)}"
            else:
                return f"Error: {error_msg}. Please check the tool documentation for required parameters."
        elif "not found" in error_msg and "job" in error_msg:
            return f"Error: {error_msg}.\n\nThe job_id you provided doesn't exist. Use run_job_tool(action=\"list\") to see available jobs."
        elif "not found" in error_msg and "experiment" in error_msg:
            return f"Error: {error_msg}.\n\nPlease use manage_experiment_tool(action=\"list\") to see available experiments or create a new experiment first."
        elif "not found" in error_msg and "dataset" in error_msg:
            return f"Error: {error_msg}.\n\nPlease use manage_dataset_tool(action=\"list\") to see available datasets or create a new dataset first."
        elif "not found" in error_msg and "evaluation" in error_msg:
            return f"Error: {error_msg}.\n\nPlease use manage_evaluation_tool(action=\"list\") to see available evaluations or create a new evaluation first."
        elif "already running" in error_msg:
            return f"Error: Job {job_id} is already running.\n\nUse run_job_tool(action=\"get\", job_id=\"{job_id}\") to check its progress."
        elif "status is" in error_msg and "because its" in error_msg:
            if "PENDING" in error_msg:
                return f"Error: {error_msg}.\n\nUse run_job_tool(action=\"run\", job_id=\"{job_id}\") to start the job."
            elif "COMPLETED" in error_msg or "FAILED" in error_msg:
                return f"Error: {error_msg}.\n\nThis job has already finished. If you want to run it again, create a new job."
            elif "CANCELLED" in error_msg:
                return f"Error: {error_msg}.\n\nThis job was cancelled. If you want to run it again, create a new job."
            else:
                return f"Error: {error_msg}"
        elif "max_parallel" in error_msg:
            return f"Error: {error_msg}.\n\nThe max_parallel parameter must be a positive integer. Example: max_parallel=5"
        else:
            return f"Error: {error_msg}"



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
    
    This tool helps you analyze the results of prompt optimization experiments
    to determine which prompt variants performed best, understand performance patterns,
    and generate optimized prompt versions based on what was learned.
    
    Args:
        action: The operation to perform. Must be one of:
               - "analyze" - Analyze an experiment's results and create a new analysis
               - "list" - List all available analyses
               - "get" - Get details of a specific analysis
               - "delete" - Delete an analysis
               - "compare" - Compare multiple analyses
               
        experiment_id: The ID of the experiment to analyze.
                      Required for "analyze" action.
                      Must reference an experiment that has been run to completion.
                      Example: "9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r"
               
        job_id: Optional ID of a specific job to analyze.
               Only needed for "analyze" action if the experiment has multiple jobs.
               Example: "7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c"
               
        analysis_id: The unique identifier of the analysis.
                    Required for "get" and "delete" actions.
                    Example: "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o"
                    
        name: Optional name for the analysis.
              If not provided for "analyze" action, a name will be generated.
              Example: "Formal vs. Casual Tone Analysis"
              
        description: Optional description for the analysis.
                    Example: "Comparing performance of formal and casual tones across demographics"
                    
        analysis_ids: List of analysis IDs to compare.
                     Required for "compare" action.
                     Example: ["5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o", "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"]
    
    Returns:
        For "list" action: A formatted list of all analyses with their IDs and names
        For "get" action: Detailed information about the specified analysis
        For "analyze" action: A summary of the experiment results, including:
                            - Overall performance of each prompt variant
                            - Detailed metrics and scores
                            - Recommendations for improvements
                            - Patterns observed across different inputs
        For "delete" action: Confirmation message that the analysis was deleted
        For "compare" action: Comparative analysis of multiple experiments
        
    Error messages will clearly indicate what went wrong, such as:
    - "Invalid action: 'foo'. Valid actions are: analyze, list, get, delete, compare"
    - "Missing required parameters: analysis_id"
    - "Analysis with ID 'xyz' not found"
    - "Experiment with ID 'abc' not found"
    - "Experiment 'abc' has not been run yet"
    - "Job with ID 'def' not found"
    
    Important Note:
    The "analyze" action performs a detailed statistical analysis of experiment results,
    which can take a few seconds to complete for large experiments. The result includes
    both quantitative metrics and qualitative insights about each prompt variant's performance.
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
        
        # If result is a dictionary with a formatted_message, return that for MCP
        if isinstance(result, dict) and "formatted_message" in result:
            return result["formatted_message"]
            
        return result
    except Exception as e:
        logger.error(f"Error in analyze_results_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"Error: Invalid action '{action}'. Valid actions are: analyze, list, get, delete, compare\n\nExamples:\n- analyze_results_tool(action=\"analyze\", experiment_id=\"9c8d...\", name=\"My Analysis\")\n- analyze_results_tool(action=\"list\")\n- analyze_results_tool(action=\"get\", analysis_id=\"5d4c...\")"
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "analyze":
                example = {
                    "action": "analyze",
                    "experiment_id": "9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
                    "name": "Tone Analysis Results",
                    "description": "Analysis of different customer service tones"
                }
                return f"Error: {error_msg}.\n\nThe 'analyze' action requires the experiment_id parameter.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action == "get" or action == "delete":
                example = {"action": action, "analysis_id": "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o"}
                return f"Error: {error_msg}.\n\nThe '{action}' action requires the analysis_id parameter.\n\nExample:\n{json.dumps(example, indent=2)}"
            elif action == "compare":
                example = {
                    "action": "compare",
                    "analysis_ids": [
                        "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o",
                        "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
                    ]
                }
                return f"Error: {error_msg}.\n\nThe 'compare' action requires the analysis_ids parameter (a list of analysis IDs).\n\nExample:\n{json.dumps(example, indent=2)}"
            else:
                return f"Error: {error_msg}. Please check the tool documentation for required parameters."
        elif "not found" in error_msg and "analysis" in error_msg:
            return f"Error: {error_msg}.\n\nThe analysis_id you provided doesn't exist. Use analyze_results_tool(action=\"list\") to see available analyses."
        elif "not found" in error_msg and "experiment" in error_msg:
            return f"Error: {error_msg}.\n\nPlease use manage_experiment_tool(action=\"list\") to see available experiments or create a new experiment first."
        elif "not found" in error_msg and "job" in error_msg:
            return f"Error: {error_msg}.\n\nPlease use run_job_tool(action=\"list\") to see available jobs or create a new job first."
        elif "not been run" in error_msg or "results" in error_msg and "no" in error_msg:
            return f"Error: The experiment has not been run yet or has no results.\n\nPlease use run_job_tool to run the experiment first with:\n1. run_job_tool(action=\"create\", experiment_id=\"{experiment_id}\", ...)\n2. run_job_tool(action=\"run\", job_id=\"...\")"
        elif "analysis_ids" in error_msg:
            example = {
                "action": "compare",
                "analysis_ids": [
                    "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o",
                    "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
                ]
            }
            return f"Error: {error_msg}.\n\nThe 'compare' action requires a list of at least two valid analysis IDs.\n\nExample:\n{json.dumps(example, indent=2)}"
        else:
            return f"Error: {error_msg}"




def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
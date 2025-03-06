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
    wait: Optional[bool] = True,
    poll_interval: Optional[int] = 5,
    timeout_minutes: Optional[int] = 10,
    stdio_friendly: Optional[bool] = True,
) -> str:
    """
    Execute and manage jobs that run prompt optimization experiments.
    
    This tool allows you to create, run, and manage jobs that execute experiments
    with different prompt variants against datasets. Each job runs an experiment
    by generating responses from each prompt variant for the dataset items, then
    evaluating those responses using the specified evaluation criteria.
    
    QUICKSTART EXAMPLES:
    
    1. Create a job, run it, and wait for completion (default behavior):
       ```
       job_result = run_job_tool(
           action="create", 
           experiment_id="9c8d...", 
           dataset_id="8a7b...", 
           evaluation_id="6f8d..."
       )
       # Job completes and returns results automatically!
       status = job_result["job"]["status"]  # Will be "COMPLETED"
       results = job_result["job"]["results"]  # Contains all job results
       ```
       
    1b. Create a job without waiting (asynchronous mode):
       ```
       job_result = run_job_tool(
           action="create", 
           experiment_id="9c8d...", 
           dataset_id="8a7b...", 
           evaluation_id="6f8d...",
           wait=False  # Return immediately without waiting
       )
       # Job started but not yet complete
       job_id = job_result["job"]["job_id"]
       
       # Check status later:
       status = run_job_tool(action="get", job_id=job_id)
       ```
       
    2. Check a job's status:
       ```
       run_job_tool(action="get", job_id="7r6q...")
       ```
       
    3. List all jobs:
       ```
       run_job_tool(action="list")
       ```
    
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
                    Example: "claude-3-haiku-20240307" or "lmstudio-community/meta-llama-3.1-8b-instruct"
                    
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
                   
        wait: Whether to block and wait for the job to complete before returning.
              Only applies to "create" and "run" actions when auto_start is True.
              Defaults to True.
              If False, the function will return immediately without waiting.
              Example: False
              
        poll_interval: Seconds to wait between status checks when wait=True.
                      Only applies when wait=True.
                      Defaults to 5 seconds.
                      Example: 10
                      
        timeout_minutes: Maximum time in minutes to wait before declaring a job as failed.
                        Optional, defaults to 10 minutes.
                        Jobs that show no progress may be stuck due to LLM connectivity issues.
                        Example: 30
                        
        stdio_friendly: Whether to optimize job execution for stdio-based MCP transport.
                      Only applies when wait=True.
                      Defaults to True (recommended for Claude Desktop).
                      Set to False for HTTP-based MCP transport for slightly better performance.
                      Example: True
    
    Returns:
        For "list" action: A formatted list of all jobs with their IDs and statuses
        For "get" action: Detailed information about the job including current status
        For "create" action with wait=False (default): Confirmation message with the new job ID and that it has started
        For "create" action with wait=True: Complete job results after waiting for job completion
        For "run" action with wait=False (default): Message confirming the job has started
        For "run" action with wait=True: Complete job results after waiting for job completion
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
    Jobs are executed synchronously by default. When a job is created, it automatically starts running
    and waits for completion (unless configured otherwise). You can choose between two execution modes:
    
    1. Synchronous mode (default, wait=True):
       - Job starts and the tool blocks until job completes
       - Returns complete results when finished
       - Provides the simplest workflow - create job and get results in one step
       - Specify poll_interval to control how often status is checked
    
    2. Asynchronous mode (wait=False): 
       - Job starts and the tool returns immediately
       - Use the "get" action later to check progress
       - Better for larger jobs that may take a long time to complete
    
    Jobs may take several minutes to complete depending on the size of the dataset and 
    the number of prompt variants.
    
    WORKFLOW EXAMPLES:
    
    1. Complete workflow with synchronous execution (default):
       ```
       # Create job and wait for completion in one step
       job_result = run_job_tool(
           action="create",
           experiment_id="exp_id_here",
           dataset_id="dataset_id_here",
           evaluation_id="eval_id_here",
           judge_model="lmstudio-community/meta-llama-3.1-8b-instruct",
           poll_interval=10  # Check status every 10 seconds
       )
       
       # Job is already complete! Analyze results immediately
       analysis = analyze_results_tool(
           action="analyze",
           experiment_id="exp_id_here"
       )
       ```
       
    2. Asynchronous workflow option:
       ```
       # Create job without waiting
       job_result = run_job_tool(
           action="create",
           experiment_id="exp_id_here",
           dataset_id="dataset_id_here",
           evaluation_id="eval_id_here",
           judge_model="lmstudio-community/meta-llama-3.1-8b-instruct",
           wait=False  # Don't wait for completion
       )
       
       # Extract job_id from the result
       job_id = job_result["job"]["job_id"]
       
       # Check job progress
       job_status = run_job_tool(action="get", job_id=job_id)
       
       # When job completes, analyze the results
       analysis = analyze_results_tool(
           action="analyze",
           experiment_id="exp_id_here"
       )
       ```
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
                    
                    # If wait=True, wait for the job to complete
                    if wait:
                        # Get a reference to the job
                        job_data = result.get("job", {})
                        job_id = new_job_id
                        
                        # Poll for job completion
                        logger.info(f"Waiting for job {job_id} to complete (poll_interval={poll_interval}s)")
                        import time
                        
                        start_time = time.time()
                        while True:
                            # Get current job status
                            try:
                                job = get_job(job_id)
                                status = job.status
                                progress = job.progress
                                
                                # Check if job is done
                                if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                                    elapsed_time = time.time() - start_time
                                    logger.info(f"Job {job_id} completed with status {status} in {elapsed_time:.2f}s")
                                    
                                    # Return completed job data
                                    if status == "COMPLETED":
                                        completion_message = (
                                            f"Job completed successfully in {elapsed_time:.2f} seconds.\n"
                                            f"Processed {progress.get('completed', 0)} tasks with "
                                            f"{job.results.get('succeeded', 0)} successes and "
                                            f"{job.results.get('failed', 0)} failures."
                                        )
                                    elif status == "FAILED":
                                        # Provide helpful error message if the job failed
                                        error_detail = job.error or "Unknown error"
                                        api_base = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234/v1")
                                        
                                        completion_message = (
                                            f"Job FAILED after {elapsed_time:.2f} seconds.\n"
                                            f"Error: {error_detail}\n\n"
                                            f"TROUBLESHOOTING:\n"
                                            f"1. Check if your LLM server is running at {api_base}\n"
                                            f"2. Try testing with: curl {api_base}/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"meta-llama-3.1-8b-instruct\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'\n"
                                            f"3. Set environment variables if needed:\n"
                                            f"   export AGENTOPTIM_API_BASE=http://localhost:YOUR_PORT/v1"
                                        )
                                    else:
                                        completion_message = f"Job ended with status: {status} after {elapsed_time:.2f} seconds."
                                        
                                    return {
                                        "status": "success",
                                        "message": completion_message,
                                        "job": {
                                            "job_id": job_id,
                                            "status": status,
                                            "progress": progress,
                                            "results": job.results,
                                            "experiment_id": job.experiment_id,
                                            "dataset_id": job.dataset_id,
                                            "evaluation_id": job.evaluation_id,
                                            "judge_model": job.judge_model,
                                            "elapsed_time": elapsed_time
                                        }
                                    }
                                
                                # Sleep before next check
                                time.sleep(poll_interval)
                            except Exception as e:
                                logger.error(f"Error polling job status: {str(e)}")
                                # Return what we know so far
                                return {
                                    "status": "error",
                                    "message": f"Error while waiting for job to complete: {str(e)}",
                                    "job": job_data
                                }
                    
                    # Include the job in the response (if not waiting)
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
                
                # If wait is True, poll until completion
                if wait:
                    # Poll for job completion
                    logger.info(f"Waiting for job {job_id} to complete (poll_interval={poll_interval}s)")
                    import time
                    
                    start_time = time.time()
                    while True:
                        # Get current job status
                        try:
                            current_job = get_job(job_id)
                            status = current_job.status
                            progress = current_job.progress
                            
                            # Check if job is done
                            if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                                elapsed_time = time.time() - start_time
                                logger.info(f"Job {job_id} completed with status {status} in {elapsed_time:.2f}s")
                                
                                # Return completed job data
                                if status == "COMPLETED":
                                    completion_message = (
                                        f"Job completed successfully in {elapsed_time:.2f} seconds.\n"
                                        f"Processed {progress.get('completed', 0)} tasks with "
                                        f"{current_job.results.get('succeeded', 0)} successes and "
                                        f"{current_job.results.get('failed', 0)} failures."
                                    )
                                elif status == "FAILED":
                                    # Provide helpful error message if the job failed
                                    error_detail = current_job.error or "Unknown error"
                                    api_base = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234/v1")
                                    
                                    completion_message = (
                                        f"Job FAILED after {elapsed_time:.2f} seconds.\n"
                                        f"Error: {error_detail}\n\n"
                                        f"TROUBLESHOOTING:\n"
                                        f"1. Check if your LLM server is running at {api_base}\n"
                                        f"2. Try testing with: curl {api_base}/chat/completions -H 'Content-Type: application/json' -d '{{\"model\":\"meta-llama-3.1-8b-instruct\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'\n"
                                        f"3. Set environment variables if needed:\n"
                                        f"   export AGENTOPTIM_API_BASE=http://localhost:YOUR_PORT/v1"
                                    )
                                else:
                                    completion_message = f"Job ended with status: {status} after {elapsed_time:.2f} seconds."
                                    
                                return {
                                    "status": "success",
                                    "message": completion_message,
                                    "job": {
                                        "job_id": job_id,
                                        "status": status,
                                        "progress": progress,
                                        "results": current_job.results,
                                        "experiment_id": current_job.experiment_id,
                                        "dataset_id": current_job.dataset_id,
                                        "evaluation_id": current_job.evaluation_id,
                                        "judge_model": current_job.judge_model,
                                        "elapsed_time": elapsed_time
                                    }
                                }
                            
                            # Sleep before next check
                            time.sleep(poll_interval)
                        except Exception as e:
                            logger.error(f"Error polling job status: {str(e)}")
                            # Return error
                            return {
                                "status": "error",
                                "message": f"Error while waiting for job to complete: {str(e)}",
                                "job": {"job_id": job_id}
                            }
                
                # If not waiting, return standard response
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
            except Exception as e:
                # Fallback if we can't get job details
                logger.error(f"Error getting job details for {job_id}: {str(e)}")
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
            # Show different examples based on action with improved formatting
            if action == "create":
                example_code = """run_job_tool(
    action="create",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
    dataset_id="8a7b6c5d-4e3f-2g1h-0i9j-8k7l6m5n4o3p",
    evaluation_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
)"""
                return f"""Error: {error_msg}.

The 'create' action requires experiment_id, dataset_id, and evaluation_id parameters.

Example:
```python
{example_code}
```

First, make sure you've created:
1. A dataset with manage_dataset_tool()
2. An evaluation with manage_evaluation_tool()
3. An experiment with manage_experiment_tool()

Then use the IDs from those resources in the create action."""
            elif action in ["get", "delete", "run", "cancel"]:
                example_code = f"""run_job_tool(
    action="{action}",
    job_id="7r6q5p4o-3n2m-1l0k-9j8i-7h6g5f4e3d2c"
)"""
                return f"""Error: {error_msg}.

The '{action}' action requires the job_id parameter.

Example:
```python
{example_code}
```

To get a list of available job IDs, first run:
```python
run_job_tool(action="list")
```"""
            else:
                return f"""Error: {error_msg}.

Please check the tool documentation for required parameters.

Most common actions:
- "create" - Creates and runs a new job
- "get" - Gets details about an existing job
- "list" - Lists all jobs"""
        elif "not found" in error_msg and "job" in error_msg:
            return f"""Error: {error_msg}

The job_id you provided doesn't exist. To see all available jobs:

```python
run_job_tool(action="list")
```"""
        elif "not found" in error_msg and "experiment" in error_msg:
            return f"""Error: {error_msg}

You need to create an experiment first or use an existing one. To list available experiments:

```python
manage_experiment_tool(action="list")
```

To create a new experiment:

```python
manage_experiment_tool(
    action="create",
    name="My Experiment",
    dataset_id="your_dataset_id",
    evaluation_id="your_evaluation_id",
    model_name="claude-3-opus-20240229",
    prompt_variants=[...]
)
```"""
        elif "not found" in error_msg and "dataset" in error_msg:
            return f"""Error: {error_msg}

You need to create a dataset first or use an existing one. To list available datasets:

```python
manage_dataset_tool(action="list")
```

To create a new dataset:

```python
manage_dataset_tool(
    action="create",
    name="My Dataset",
    items=[
        {"input": "Example input 1"},
        {"input": "Example input 2"}
    ]
)
```"""
        elif "not found" in error_msg and "evaluation" in error_msg:
            return f"""Error: {error_msg}

You need to create an evaluation first or use an existing one. To list available evaluations:

```python
manage_evaluation_tool(action="list")
```

To create a new evaluation:

```python
manage_evaluation_tool(
    action="create",
    name="My Evaluation",
    template="Input: {input}\nResponse: {response}\nQuestion: {question}",
    questions=["Is the response helpful?", "Is the response accurate?"]
)
```"""
        elif "already running" in error_msg:
            return f"""Error: Job {job_id} is already running.

To check the job's progress:

```python
run_job_tool(action="get", job_id="{job_id}")
```"""
        elif "status is" in error_msg and "because its" in error_msg:
            if "PENDING" in error_msg:
                return f"""Error: {error_msg}

This job is pending and needs to be started. To start the job:

```python
run_job_tool(action="run", job_id="{job_id}")
```"""
            elif "COMPLETED" in error_msg or "FAILED" in error_msg:
                return f"""Error: {error_msg}

This job has already finished. To analyze its results:

```python
# Get job details
job = run_job_tool(action="get", job_id="{job_id}")

# Analyze results for the experiment
experiment_id = job["job"]["experiment_id"]
analyze_results_tool(action="analyze", experiment_id=experiment_id)
```

If you want to run the experiment again, create a new job."""
            elif "CANCELLED" in error_msg:
                return f"""Error: {error_msg}

This job was cancelled. If you want to run it again, create a new job:

```python
# Get the original job to see its configuration
job = run_job_tool(action="get", job_id="{job_id}")

# Create a new job with the same parameters
run_job_tool(
    action="create",
    experiment_id=job["job"]["experiment_id"],
    dataset_id=job["job"]["dataset_id"],
    evaluation_id=job["job"]["evaluation_id"]
)
```"""
            else:
                return f"Error: {error_msg}"
        elif "max_parallel" in error_msg:
            return f"""Error: {error_msg}

The max_parallel parameter must be a positive integer. This controls how many tasks run in parallel.

Example:
```python
run_job_tool(
    action="create",
    experiment_id="...",
    dataset_id="...",
    evaluation_id="...",
    max_parallel=5  # Process 5 tasks at a time
)
```"""
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
    
    QUICKSTART EXAMPLES:
    
    1. Analyze experiment results:
       ```
       analyze_results_tool(
           action="analyze", 
           experiment_id="9c8d..."
       )
       ```
       
    2. Get details of a specific analysis:
       ```
       analyze_results_tool(action="get", analysis_id="5d4c...")
       ```
       
    3. List all analyses:
       ```
       analyze_results_tool(action="list")
       ```
    
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
    
    COMPLETE WORKFLOW EXAMPLE:
    
    ```python
    # 1. Create a dataset
    dataset = manage_dataset_tool(
        action="create",
        name="Customer Service Messages",
        items=[
            {"input": "How do I reset my password?"},
            {"input": "When will my order arrive?"}
        ]
    )
    dataset_id = "extract_id_from_response"
    
    # 2. Create an evaluation
    evaluation = manage_evaluation_tool(
        action="create",
        name="Response Quality Evaluation",
        template="Input: {input}\nResponse: {response}\nQuestion: {question}",
        questions=["Is the response helpful?", "Is the response clear?"]
    )
    evaluation_id = "extract_id_from_response"
    
    # 3. Create an experiment with different prompt variants
    experiment = manage_experiment_tool(
        action="create",
        name="Tone Optimization",
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        model_name="claude-3-opus-20240229",
        prompt_variants=[
            {
                "name": "formal_tone",
                "type": "system",
                "content": "Use a formal, professional tone."
            },
            {
                "name": "casual_tone",
                "type": "system",
                "content": "Use a casual, friendly tone."
            }
        ]
    )
    experiment_id = "extract_id_from_response"
    
    # 4. Create and run a job (automatically starts)
    job = run_job_tool(
        action="create",
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        evaluation_id=evaluation_id
    )
    
    # The job is already running! Check status if needed:
    # job_status = run_job_tool(action="get", job_id=job["job"]["job_id"])
    
    # 5. Once the job completes, analyze the results
    analysis = analyze_results_tool(
        action="analyze",
        experiment_id=experiment_id,
        name="Tone Optimization Analysis"
    )
    ```
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
        
        # Enhance error messages for common problems with improved formatting
        if "action" in error_msg and "valid actions" not in error_msg:
            return f"""Error: Invalid action '{action}'.

Valid actions are: analyze, list, get, delete, compare

Examples:
```python
# Analyze experiment results
analyze_results_tool(action="analyze", experiment_id="9c8d...")

# List all analyses
analyze_results_tool(action="list")

# Get a specific analysis
analyze_results_tool(action="get", analysis_id="5d4c...")
```"""
        elif "required parameters" in error_msg:
            # Show different examples based on action with improved formatting
            if action == "analyze":
                example_code = """analyze_results_tool(
    action="analyze",
    experiment_id="9c8d7e6f-5g4h-3i2j-1k0l-9m8n7o6p5q4r",
    name="Tone Analysis Results",  # Optional
    description="Analysis of different customer service tones"  # Optional
)"""
                return f"""Error: {error_msg}.

The 'analyze' action requires the experiment_id parameter.

Example:
```python
{example_code}
```

NOTE: Make sure the experiment has finished running and has results before analysis."""
            elif action == "get" or action == "delete":
                example_code = f"""analyze_results_tool(
    action="{action}",
    analysis_id="5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o"
)"""
                return f"""Error: {error_msg}.

The '{action}' action requires the analysis_id parameter.

Example:
```python
{example_code}
```

To get a list of available analysis IDs, first run:
```python
analyze_results_tool(action="list")
```"""
            elif action == "compare":
                example_code = """analyze_results_tool(
    action="compare",
    analysis_ids=[
        "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o",
        "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
    ]
)"""
                return f"""Error: {error_msg}.

The 'compare' action requires the analysis_ids parameter (a list of at least two analysis IDs).

Example:
```python
{example_code}
```"""
            else:
                return f"""Error: {error_msg}.

Please check the tool documentation for required parameters.

Most common actions:
- "analyze" - Analyzes an experiment's results
- "get" - Gets details of a specific analysis
- "list" - Lists all analyses"""
        elif "not found" in error_msg and "analysis" in error_msg:
            return f"""Error: {error_msg}

The analysis_id you provided doesn't exist. To see all available analyses:

```python
analyze_results_tool(action="list")
```"""
        elif "not found" in error_msg and "experiment" in error_msg:
            return f"""Error: {error_msg}

The experiment_id you provided doesn't exist or can't be found. To list available experiments:

```python
manage_experiment_tool(action="list")
```"""
        elif "not found" in error_msg and "job" in error_msg:
            return f"""Error: {error_msg}

The job_id you provided doesn't exist or can't be found. To list all available jobs:

```python
run_job_tool(action="list")
```"""
        elif "not been run" in error_msg or ("results" in error_msg and "no" in error_msg):
            return f"""Error: The experiment has not been run yet or has no results.

You need to create and run a job for this experiment first:

```python
# Create a job for the experiment
job_result = run_job_tool(
    action="create",
    experiment_id="{experiment_id}",
    dataset_id="dataset_id_here",
    evaluation_id="evaluation_id_here"
)

# Job will automatically start running
# Check the job's status to see when it completes
job_status = run_job_tool(
    action="get", 
    job_id=job_result["job"]["job_id"]
)
```"""
        elif "analysis_ids" in error_msg:
            example_code = """analyze_results_tool(
    action="compare",
    analysis_ids=[
        "5d4c3b2a-1z0y-9x8w-7v6u-5t4s3r2q1p0o",
        "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
    ]
)"""
            return f"""Error: {error_msg}

The 'compare' action requires a list of at least two valid analysis IDs.

Example:
```python
{example_code}
```

To get a list of available analysis IDs, run:
```python
analyze_results_tool(action="list")
```"""
        else:
            return f"Error: {error_msg}"




def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
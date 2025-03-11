"""MCP server implementation for AgentOptim v2.2.0."""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

from mcp.server.fastmcp import FastMCP

# Import EvalSet tools
from agentoptim.evalset import manage_evalset, get_cache_statistics
from agentoptim.runner import run_evalset, get_api_cache_stats
from agentoptim.evalrun import (
    EvalRun, save_eval_run, get_eval_run, list_eval_runs, 
    manage_eval_runs, get_formatted_eval_run
)

# Import System Message Optimization tools
from agentoptim.sysopt import (
    manage_optimization_runs, get_sysopt_stats
)

# Import necessary utilities
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Ensure data directories exist
ensure_data_directories()

# Check for debug mode
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"
# Environment variable overrides for judge model and omit reasoning
DEFAULT_JUDGE_MODEL = os.environ.get("AGENTOPTIM_JUDGE_MODEL", None)
DEFAULT_OMIT_REASONING = os.environ.get("AGENTOPTIM_OMIT_REASONING", "0").lower() in ('true', 't', '1', 'yes', 'y', 'on', 'enabled')

# Configure logging - only log to file and stderr, not stdout (to avoid breaking MCP's stdio transport)
log_file_path = os.path.join(DATA_DIR, "agentoptim.log")
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # Changed from stdout to stderr
        logging.FileHandler(log_file_path),
    ],
)

# Add environment info to log
logger = logging.getLogger("agentoptim")
logger.debug(f"Logging to {log_file_path}")
logger.debug(f"Debug mode: {DEBUG_MODE}")

# Initialize FastMCP server
mcp = FastMCP("agentoptim")


@mcp.tool()
async def manage_evalset_tool(
    action: str,
    evalset_id: Optional[str] = None,
    name: Optional[str] = None,
    questions: Optional[List[str]] = None,
    short_description: Optional[str] = None,
    long_description: Optional[str] = None
) -> dict:
    """
    Manage evaluation criteria sets (EvalSets) for systematically assessing conversation quality.
    
    ## Overview
    This tool allows you to create, retrieve, update, and delete "EvalSets" - collections of 
    evaluation criteria for judging the quality of conversational responses. Each EvalSet contains:
    
    1. A set of yes/no evaluation questions (e.g., "Is the response helpful?")
    2. A system-defined template with formatting instructions for the evaluation model
    3. Metadata like name, short description, and long description
    
    ## IMPORTANT: List Before Creating
    
    ALWAYS use `action="list"` to check for existing EvalSets before creating a new one:
    ```python
    # First, list all existing EvalSets
    existing_evalsets = manage_evalset_tool(action="list")
    
    # Then examine the list to avoid creating duplicates
    # Only create a new EvalSet if nothing suitable exists
    ```
    
    This prevents duplicate EvalSets and promotes reuse of well-crafted evaluation criteria.
    
    ## Creating Effective EvalSets
    
    To create a high-quality EvalSet:
    
    - Use specific, measurable questions (e.g., "Does the response provide a step-by-step solution?")
    - Include diverse evaluation criteria (helpfulness, accuracy, clarity, etc.)
    - Maintain consistent evaluation standards across questions
    - Provide clear descriptions of what the EvalSet measures and when to use it
    
    ## Arguments
    
    action: The operation to perform. Must be one of:
           - "create" - Create a new EvalSet with evaluation criteria
           - "list" - List all available EvalSets in the system
           - "get" - Get complete details of a specific EvalSet by ID
           - "update" - Modify an existing EvalSet's properties
           - "delete" - Permanently remove an EvalSet
           
    evalset_id: The unique identifier of the EvalSet (UUID format).
              Required for "get", "update", and "delete" actions.
              Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
              
    name: A descriptive name for the EvalSet.
          Required for "create" action, optional for "update".
          Should be specific to the evaluation purpose.
          Example: "Technical Support Response Quality Evaluation"
          
    # template parameter has been removed - system now uses a standard template
             
    questions: A list of yes/no evaluation questions to assess responses.
              Required for "create" action, optional for "update".
              CRITICAL: Must be a proper JSON array/list of strings, NOT a single multiline string.
              Each question must be clear, specific, and answerable with yes/no.
              CORRECT EXAMPLE: [
                "Does the response directly address the user's question?",
                "Is the response clear and easy to understand?",
                "Does the response provide complete information?"
              ]
              INCORRECT EXAMPLE: "Does the response directly address the user's question?
              Is the response clear and easy to understand?
              Does the response provide complete information?"
              
    short_description: A concise summary of what this EvalSet measures (6-128 characters).
                    REQUIRED for "create" action, optional for "update".
                    Should provide a quick understanding of the EvalSet's purpose.
                    Example: "Technical support response quality evaluation criteria"
    
    long_description: A detailed explanation of the evaluation criteria (256-1024 characters).
                    REQUIRED for "create" action, optional for "update".
                    Should provide comprehensive information about:
                    - What aspects of conversation are being evaluated
                    - How to interpret the results
                    - When this EvalSet is appropriate to use
                    Example: "This EvalSet provides comprehensive evaluation criteria for technical support responses. It measures clarity, completeness, accuracy, helpfulness, and professionalism. Use it to evaluate support agent responses to technical questions or troubleshooting scenarios. High scores indicate responses that are clear, accurate, and likely to resolve the user's issue without further assistance." + " " * 50
    
    # description parameter has been removed - use short_description and long_description instead
    
    ## System Template
    
    The system template automatically includes:
    
    1. Instructions for the judge model to evaluate the conversation 
    2. Formatting for the {{ conversation }} and {{ eval_question }} variables
    3. Clear directions for JSON output format with these fields:
       - "reasoning": Detailed explanation for the judgment (can be omitted with omit_reasoning option)
       - "judgment": Boolean true/false for yes/no answers
       - "confidence": Number between 0 and 1 indicating confidence level
    4. Format examples showing proper JSON syntax
    
    ## Return Values
    
    For "list" action: Dictionary with "evalsets" key containing all available EvalSets
    For "get" action: Dictionary with "evalset" key containing the complete EvalSet data
    For "create" action: Dictionary with "evalset" key containing the new EvalSet with generated ID
    For "update" action: Dictionary with "evalset" key containing the updated EvalSet
    For "delete" action: Dictionary with "status" and "message" confirming deletion
    
    ## Usage Examples
    
    ### Proper Workflow: List First, Then Create Only If Needed
    
    ```python
    # STEP 1: ALWAYS list existing EvalSets first to avoid duplicates
    existing_evalsets = manage_evalset_tool(action="list")
    
    # STEP 2: Examine the list to see if a suitable EvalSet already exists
    # Only proceed with creation if nothing suitable exists
    
    # STEP 3: If needed, create a new EvalSet with required fields
    # IMPORTANT: Note that 'questions' is an array/list of strings, NOT a multiline string
    evalset = manage_evalset_tool(
        action="create",
        name="Technical Support Quality Evaluation",
        # The template is now system-defined, no need to provide it
        # CRITICAL: questions must be a list/array of strings, not a multiline string
        questions=[
            "Does the response directly address the user's specific question?",
            "Is the response clear and easy to understand?",
            "Does the response provide complete step-by-step instructions?",
            "Is the response accurate and technically correct?",
            "Does the response use appropriate technical terminology?",
            "Is the tone of the response professional and helpful?",
            "Would the response likely resolve the user's issue without further assistance?"
        ],
        short_description="Tech support response evaluation criteria",
        long_description="This EvalSet provides comprehensive evaluation criteria for technical support responses. It measures clarity, completeness, accuracy, helpfulness, and professionalism. Use it to evaluate support agent responses to technical questions or troubleshooting scenarios. High scores indicate responses that are clear, accurate, and likely to resolve the user's issue without further assistance." + " " * 50
    )
    
    # Get a specific EvalSet by ID
    evalset_details = manage_evalset_tool(
        action="get",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
    )
    
    # Update an existing EvalSet
    updated_evalset = manage_evalset_tool(
        action="update",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        name="Enhanced Technical Support Quality Evaluation",
        questions=["New question 1", "New question 2"],
        short_description="Enhanced tech support evaluation criteria",
        long_description="This enhanced EvalSet provides improved evaluation criteria for technical support responses with more specific questions focused on resolution success. It measures clarity, accuracy, and customer satisfaction more precisely. Use it for evaluating advanced support scenarios where multiple solutions might be applicable." + " " * 50
    )
    
    # Delete an EvalSet
    delete_result = manage_evalset_tool(
        action="delete",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
    )
    ```
    """
    logger.info(f"manage_evalset_tool called with action={action}")
    try:
        result = manage_evalset(
            action=action,
            evalset_id=evalset_id,
            name=name,
            questions=questions,
            short_description=short_description,
            long_description=long_description,
        )
        
        # If result is a dictionary with a formatted_message, use that message
        if isinstance(result, dict) and "formatted_message" in result:
            return {"result": result["formatted_message"]}
            
        # If result is a string, wrap it in a dict
        if isinstance(result, str):
            return {"result": result}
            
        # If result is already a dict but doesn't have formatted_message
        return result
    except Exception as e:
        logger.error(f"Error in manage_evalset_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        error_response = {"error": error_msg}
        
        # Enhance error messages for common problems
        if "action" in error_msg and "valid actions" not in error_msg:
            error_response["details"] = f"Invalid action '{action}'. Valid actions are: create, list, get, update, delete"
            error_response["examples"] = [
                'manage_evalset_tool(action="create", name="Response Quality", template="...", questions=[...])',
                'manage_evalset_tool(action="list")',
                'manage_evalset_tool(action="get", evalset_id="...")'
            ]
        elif "questions" in error_msg and ("list_type" in error_msg or "str" in error_msg):
            error_response["details"] = "The 'questions' parameter must be a proper list of strings, not a multiline string."
            error_response["troubleshooting"] = [
                'Make sure questions is formatted as a JSON array/list of strings',
                'Each question should be a separate string in the list',
                'Example: questions=["Question 1?", "Question 2?", "Question 3?"]',
                'Do NOT use a single multiline string for multiple questions',
                'Incorrect: questions="Question 1?\nQuestion 2?\nQuestion 3?"'
            ]
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "create":
                example = {
                    "action": "create",
                    "name": "Response Quality Evaluation",
                    # Template is now system-defined and not required
                    "questions": [
                        "Is the response clear and concise?",
                        "Does the response fully answer the question?",
                        "Is the response accurate?"
                    ],
                    "short_description": "General response quality evaluation criteria",
                    "long_description": "This EvalSet evaluates conversation responses for clarity, completeness, and accuracy. It can be used to assess the quality of AI assistant responses across a wide range of general knowledge topics. High scores indicate responses that are clear, comprehensive, and factually correct." + " " * 50
                }
                error_response["details"] = "The 'create' action requires name, questions, short_description, and long_description parameters."
                error_response["example"] = example
            elif action in ["get", "update", "delete"]:
                example = {
                    "action": action,
                    "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                }
                # Add fields for update
                if action == "update":
                    # Template is now system-defined and not required
                    example["questions"] = ["Updated question 1", "Updated question 2"]
                    example["short_description"] = "Updated quality evaluation criteria"
                    example["long_description"] = "This updated EvalSet provides improved evaluation criteria with a stronger focus on response relevance and accuracy. Use it to evaluate assistant responses to complex queries where precision is critical." + " " * 50
                
                error_response["details"] = f"The '{action}' action requires the evalset_id parameter."
                error_response["example"] = example
            else:
                error_response["details"] = "Please check the tool documentation for required parameters."
        elif "not found" in error_msg:
            error_response["details"] = "The evalset_id you provided doesn't exist. Use manage_evalset_tool(action=\"list\") to see available EvalSets."
        elif "template" in error_msg:
            error_response["details"] = "The template should include placeholders like {{ conversation }} and {{ eval_question }}."
            error_response["example"] = "Given this conversation: {{ conversation }}\n\nPlease answer this question: {{ eval_question }}"
        elif "questions" in error_msg:
            if "must be a list" in error_msg:
                error_response["details"] = "Questions must be provided as a list of strings."
                error_response["example"] = ["Is the response clear?", "Does the response answer the question?"]
            elif "maximum" in error_msg:
                error_response["details"] = "You have provided too many questions. Please limit your questions to 100 or fewer."
            else:
                error_response["details"] = "Each question should be a string that can be answered with yes/no by the judge model."
        
        return error_response


@mcp.tool()
async def manage_eval_runs_tool(
    action: str,
    evalset_id: Optional[str] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
    judge_model: Optional[str] = None,
    max_parallel: int = 3,
    omit_reasoning: bool = False,
    eval_run_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
) -> dict:
    """
    Manage evaluation runs: create, retrieve, and list conversation evaluations.
    
    ## Overview
    This tool allows you to run evaluations on conversations using predefined criteria (EvalSets),
    retrieve past evaluation results, and list previously run evaluations with pagination.
    All evaluation results are stored persistently and can be retrieved by ID for later analysis.
    
    ## Actions
    
    The tool supports these core actions:
    
    1. `run` - Evaluate a conversation against an EvalSet's criteria
    2. `get` - Retrieve a specific evaluation run by ID
    3. `list` - List all evaluation runs with pagination (optionally filtered by EvalSet)
    
    ## Arguments
    
    action: The operation to perform. Must be one of: "run", "get", "list"
           
    evalset_id: The unique identifier of the EvalSet to use.
              REQUIRED for "run" action, optional filter for "list" action.
              Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
              
    conversation: Chronological list of conversation messages to evaluate.
                REQUIRED for "run" action.
                Each message must be a dictionary with "role" and "content" fields.
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "How do I reset my password?"},
                    {"role": "assistant", "content": "Go to the login page and click 'Forgot Password'."}
                ]
                
    judge_model: The LLM model to use for evaluation. OPTIONAL for "run" action.
                If not specified, a model will be auto-detected or the default used.
                
    max_parallel: Maximum number of evaluation questions to process simultaneously.
                OPTIONAL. Defaults to 3. Higher values improve speed but use more resources.
                
    omit_reasoning: If true, don't generate detailed reasoning in evaluation results.
                   OPTIONAL. Defaults to false.
                   
    eval_run_id: ID of a specific evaluation run to retrieve.
                REQUIRED for "get" action.
                Example: "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                
    page: Page number for paginated list of runs (1-indexed).
         OPTIONAL for "list" action. Defaults to 1.
         
    page_size: Number of items per page for paginated lists.
              OPTIONAL for "list" action. Default: 10, range: 1-100.
    
    ## Return Value
    
    The tool returns a response based on the action performed:
    
    ### For "run" action
    
    ```json
    {
      "status": "success",
      "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_name": "Technical Support Quality Evaluation",
      "judge_model": "meta-llama-3.1-8b-instruct",
      "results": [
        {
          "question": "Does the response directly address the user's specific question?",
          "judgment": true,
          "confidence": 0.95,
          "reasoning": "The assistant's response directly addresses..."
        },
        ...
      ],
      "summary": {
        "total_questions": 7,
        "successful_evaluations": 7,
        "yes_count": 6,
        "no_count": 1,
        "yes_percentage": 85.71,
        ...
      }
    }
    ```
    
    ### For "get" action
    
    ```json
    {
      "status": "success",
      "eval_run": {
        "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        "evalset_name": "Technical Support Quality Evaluation",
        "timestamp": 1717286400.0,
        "judge_model": "meta-llama-3.1-8b-instruct",
        "results": [...],
        "summary": {...}
      }
    }
    ```
    
    ### For "list" action (with pagination)
    
    ```json
    {
      "status": "success",
      "eval_runs": [
        {
          "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
          "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
          "evalset_name": "Technical Support Quality Evaluation",
          "timestamp": 1717286400.0,
          "timestamp_formatted": "2024-06-01 12:00:00",
          "judge_model": "meta-llama-3.1-8b-instruct",
          "summary": {...}
        },
        ...
      ],
      "pagination": {
        "page": 1,
        "page_size": 10,
        "total_count": 25,
        "total_pages": 3,
        "has_next": true,
        "has_prev": false,
        "next_page": 2,
        "prev_page": null
      }
    }
    ```
    
    ## Usage Examples
    
    ### Running a New Evaluation
    
    ```python
    # Evaluate a conversation and get results with a stored ID
    evaluation = await manage_eval_runs_tool(
        action="run",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        conversation=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to the login page and click 'Forgot Password'."}
        ],
        judge_model="gpt-4o-mini",
        max_parallel=3
    )
    
    # Extract the evaluation run ID for later retrieval
    eval_run_id = evaluation["id"]
    print(f"Evaluation score: {evaluation['summary']['yes_percentage']}%")
    print(f"Saved as run ID: {eval_run_id}")
    ```
    
    ### Retrieving a Past Evaluation
    
    ```python
    # Get a previous evaluation result by ID
    past_eval = await manage_eval_runs_tool(
        action="get",
        eval_run_id="9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
    )
    
    # Access the evaluation data
    print(f"Evaluation from {past_eval['eval_run']['timestamp_formatted']}")
    print(f"Score: {past_eval['eval_run']['summary']['yes_percentage']}%")
    ```
    
    ### Listing Evaluation Runs
    
    ```python
    # List all evaluation runs (paginated)
    all_runs = await manage_eval_runs_tool(
        action="list",
        page=1,
        page_size=10
    )
    
    # Print summary of runs
    print(f"Found {all_runs['pagination']['total_count']} evaluation runs")
    for run in all_runs['eval_runs']:
        print(f"Run {run['id']}: {run['summary']['yes_percentage']}% on {run['timestamp_formatted']}")
    
    # Filter runs by EvalSet ID
    filtered_runs = await manage_eval_runs_tool(
        action="list",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        page=1,
        page_size=10
    )
    ```
    """
    logger.info(f"manage_eval_runs_tool called with action={action}")
    
    try:
        # Check for judge_model and omit_reasoning in options (passed from client config)
        tool_judge_model = DEFAULT_JUDGE_MODEL  # Use env var as first priority
        tool_omit_reasoning = DEFAULT_OMIT_REASONING  # Use env var as first priority
        
        # Log environment variable settings if present
        if DEFAULT_JUDGE_MODEL:
            logger.info(f"Using judge_model from environment: {DEFAULT_JUDGE_MODEL}")
        if DEFAULT_OMIT_REASONING:
            logger.info(f"Using omit_reasoning from environment: {DEFAULT_OMIT_REASONING}")
        
        try:
            if hasattr(mcp, '_mcp_server'):
                request_context = mcp._mcp_server.request_context
                
                if request_context and hasattr(request_context, 'init_options'):
                    options = request_context.init_options or {}
                    logger.info(f"DEBUG: Received init_options: {options}")
                    # Only use client options if env vars are not set
                    if not DEFAULT_JUDGE_MODEL and 'judge_model' in options:
                        tool_judge_model = options['judge_model']
                        logger.info(f"Using judge_model from client options: {tool_judge_model}")
                    
                    # Check for omit_reasoning flag (only if env var not set)
                    if not DEFAULT_OMIT_REASONING and 'omit_reasoning' in options:
                        omit_reasoning_str = str(options['omit_reasoning']).lower()
                        # Accept various forms of "true" for better user experience
                        tool_omit_reasoning = omit_reasoning_str in ('true', 't', '1', 'yes', 'y', 'on', 'enabled')
                        logger.info(f"omit_reasoning option set to: {tool_omit_reasoning}")
                        
                if tool_omit_reasoning:
                    logger.info("Reasoning will be omitted from evaluation results")
        except (AttributeError, LookupError) as e:
            # This can happen during tests or when no request context is available
            logger.debug(f"Could not access request context: {e}")
        
        # Override with explicitly passed parameters
        if judge_model is not None:
            tool_judge_model = judge_model
        if omit_reasoning is not None:
            tool_omit_reasoning = omit_reasoning
            
        # For "run" action, we need to call run_evalset directly and then save the results
        if action == "run":
            # Validate required parameters
            from agentoptim.utils import validate_required_params
            validate_required_params({
                "evalset_id": evalset_id,
                "conversation": conversation
            }, ["evalset_id", "conversation"])
            
            # Call the runner function
            eval_model = tool_judge_model  # Use the configured model, which might be None for auto-detection
            logger.info(f"Running evaluation with model: {eval_model or 'auto-detect'}")
            
            result = await run_evalset(
                evalset_id=evalset_id,
                conversation=conversation,
                judge_model=eval_model,
                max_parallel=max_parallel,
                omit_reasoning=tool_omit_reasoning
            )
            
            # Check for errors
            if "error" in result:
                return result
            
            # Create EvalRun object from the result
            eval_run = EvalRun(
                evalset_id=result.get("evalset_id"),
                evalset_name=result.get("evalset_name"),
                judge_model=result.get("judge_model"),
                results=result.get("results", []),
                conversation=conversation,
                summary=result.get("summary", {})
            )
            
            # Save to disk
            save_success = save_eval_run(eval_run)
            if not save_success:
                logger.warning(f"Failed to save evaluation run with ID: {eval_run.id}")
            
            # Format the response
            if "formatted_message" in result:
                result["result"] = result.pop("formatted_message")
            
            # Add the run ID to the result for future reference
            result["id"] = eval_run.id
            
            return result
        else:
            # For "get" and "list" actions, use the manage_eval_runs function
            return manage_eval_runs(
                action=action,
                evalset_id=evalset_id,
                conversation=conversation,
                judge_model=tool_judge_model,
                max_parallel=max_parallel,
                omit_reasoning=tool_omit_reasoning,
                eval_run_id=eval_run_id,
                page=page,
                page_size=page_size
            )
            
    except Exception as e:
        logger.error(f"Error in manage_eval_runs_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        error_response = {"error": error_msg}
        
        # Enhanced error messages with detailed troubleshooting information
        if action == "run":
            if "evalset_id" in error_msg and "not found" in error_msg:
                error_response["details"] = "The evalset_id you provided doesn't exist in the system."
                error_response["troubleshooting"] = [
                    "First, use manage_evalset_tool(action=\"list\") to see all available EvalSets",
                    "Check that you've copied the EvalSet ID correctly, including all hyphens",
                    "If you need to create a new EvalSet, use manage_evalset_tool(action=\"create\", ...)",
                    "EvalSet IDs are case-sensitive UUIDs in the format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                ]
            elif "conversation" in error_msg:
                example_conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well! How can I help you today?"}
                ]
                error_response["details"] = "The conversation parameter has an invalid format."
                error_response["troubleshooting"] = [
                    "Ensure conversation is a list of dictionaries (not a string or other type)",
                    "Each message must have both 'role' and 'content' fields",
                    "Valid roles are: 'system', 'user', 'assistant'",
                    "The conversation should include at least one 'user' and one 'assistant' message",
                    "Check for typos in field names (e.g., 'role' not 'roles')",
                    "Make sure content fields are strings"
                ]
                error_response["example"] = example_conversation
            elif "max_parallel" in error_msg:
                error_response["details"] = "The max_parallel parameter has an invalid value."
                error_response["troubleshooting"] = [
                    "max_parallel must be a positive integer",
                    "Recommended values are between 1 and 5",
                    "Too high values may cause memory or rate-limit issues",
                    "Try max_parallel=1 if you're experiencing stability problems"
                ]
            elif "model" in error_msg:
                error_response["details"] = "The specified model is invalid or unavailable."
                error_response["troubleshooting"] = [
                    "Check that the model name is spelled correctly",
                    "Ensure your LLM server or provider supports this model",
                    "Verify that the model can understand and follow instructions",
                    "Models will be auto-detected from your LLM API",
                    "For cloud APIs, make sure your API key has access to the requested model"
                ]
            else:
                # Generic troubleshooting for other errors
                error_response["troubleshooting"] = [
                    "Check that your LLM server is running and accessible",
                    "Verify that the EvalSet exists and has valid questions",
                    "Try with a simpler conversation to diagnose the issue",
                    "Check for proper JSON formatting in your template",
                    "Try setting AGENTOPTIM_DEBUG=1 for more verbose logs"
                ]
        elif action == "get":
            if "eval_run_id" in error_msg or "not found" in error_msg:
                error_response["details"] = "The eval_run_id you provided doesn't exist in the system."
                error_response["troubleshooting"] = [
                    "First, use manage_eval_runs_tool(action=\"list\") to see all available evaluation runs",
                    "Check that you've copied the run ID correctly, including all hyphens",
                    "Run IDs are case-sensitive UUIDs in the format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                ]
        elif action == "list":
            if "page" in error_msg or "page_size" in error_msg:
                error_response["details"] = "Invalid pagination parameters."
                error_response["troubleshooting"] = [
                    "page must be a positive integer (1 or greater)",
                    "page_size must be between 1 and 100",
                    "Try the default parameters: page=1, page_size=10"
                ]
        else:
            # Generic troubleshooting for other errors
            error_response["troubleshooting"] = [
                "Check the action parameter: must be one of 'run', 'get', or 'list'",
                "For 'run', make sure evalset_id and conversation are provided",
                "For 'get', make sure eval_run_id is provided",
                "For 'list', both page and page_size are optional (default: page=1, page_size=10)"
            ]
        
        return error_response


@mcp.tool()
async def optimize_system_messages_tool(
    action: str,
    user_message: Optional[str] = None,
    evalset_id: Optional[str] = None,
    base_system_message: Optional[str] = None,
    num_candidates: int = 5,
    generator_id: str = "default",
    generator_model: Optional[str] = None,
    diversity_level: str = "medium",
    max_parallel: int = 3,
    additional_instructions: str = "",
    optimization_run_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    self_optimize: bool = False
) -> dict:
    """
    Optimize system messages for a given user message through automated generation, evaluation, and ranking.
    
    ## Overview
    This tool helps find the most effective system message for any given user query by generating
    multiple candidate system messages, evaluating each one against your evaluation criteria (EvalSet),
    and ranking them based on performance. It can also self-optimize to continually improve its
    candidate generation capabilities.
    
    ## Actions
    
    The tool supports these actions:
    
    1. `optimize` - Generate, evaluate, and rank system messages for a user query
    2. `get` - Retrieve a specific optimization run by ID
    3. `list` - List all optimization runs with pagination (optionally filtered by EvalSet)
    
    ## Arguments
    
    action: The operation to perform. Must be one of: "optimize", "get", "list"
           
    user_message: The user message/query to optimize system messages for.
                REQUIRED for "optimize" action.
                Example: "How do I reset my password?"
                
    evalset_id: ID of the EvalSet to use for evaluating candidate system messages.
              REQUIRED for "optimize" action, optional filter for "list" action.
              Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
              
    base_system_message: Optional starting system message to use as a foundation.
                      OPTIONAL for "optimize" action.
                      Example: "You are a helpful assistant that provides clear and accurate information."
    
    num_candidates: Number of system message candidates to generate.
                   OPTIONAL for "optimize" action. Default: 5. Range: 1-20.
                   
    generator_id: ID of the system message generator to use.
                 OPTIONAL for "optimize" action. Default: "default".
                 
    generator_model: Model to use for generating candidate system messages.
                    OPTIONAL for "optimize" action. If not specified, a model will be auto-detected.
                    
    diversity_level: Controls how diverse the generated candidates should be.
                    OPTIONAL for "optimize" action. Values: "low", "medium", "high". Default: "medium".
                    
    max_parallel: Maximum number of evaluation questions to process simultaneously.
                 OPTIONAL for "optimize" action. Default: 3.
                 
    additional_instructions: Additional instructions to guide system message generation.
                           OPTIONAL for "optimize" action.
                           
    optimization_run_id: ID of a specific optimization run to retrieve.
                        REQUIRED for "get" action.
                        Example: "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                        
    page: Page number for paginated list of runs (1-indexed).
         OPTIONAL for "list" action. Default: 1.
         
    page_size: Number of items per page for paginated lists.
              OPTIONAL for "list" action. Default: 10, range: 1-100.
              
    self_optimize: Whether to trigger a meta-prompt self-optimization after the run.
                  OPTIONAL for "optimize" action. Default: false.
                  When true, the tool will analyze its own performance and improve its generator.
    
    ## Return Value
    
    The tool returns a response based on the action performed:
    
    ### For "optimize" action
    
    ```json
    {
      "status": "success",
      "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
      "evalset_name": "Response Quality Evaluation",
      "best_system_message": "You are a technical support assistant specializing in...",
      "best_score": 92.5,
      "candidates": [
        {
          "content": "You are a technical support assistant specializing in...",
          "score": 92.5,
          "criterion_scores": {...},
          "rank": 1
        },
        ...
      ]
    }
    ```
    
    ### For "get" action
    
    ```json
    {
      "status": "success",
      "optimization_run": {
        "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        "user_message": "How do I reset my password?",
        "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        "timestamp": 1717286400.0,
        "candidates": [...],
        ...
      }
    }
    ```
    
    ### For "list" action (with pagination)
    
    ```json
    {
      "status": "success",
      "optimization_runs": [
        {
          "id": "9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
          "user_message": "How do I reset my password?",
          "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
          "timestamp": 1717286400.0,
          ...
        },
        ...
      ],
      "pagination": {
        "page": 1,
        "page_size": 10,
        "total_count": 25,
        "total_pages": 3,
        "has_next": true,
        ...
      }
    }
    ```
    
    ## Usage Examples
    
    ### Running System Message Optimization
    
    ```python
    # Basic optimization of system messages
    result = await optimize_system_messages_tool(
        action="optimize",
        user_message="How do I reset my password?",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        num_candidates=5,
        diversity_level="high"
    )
    
    # Extract the optimization run ID and best system message
    optimization_run_id = result["id"]
    best_system_message = result["best_system_message"]
    print(f"Best system message (score: {result['best_score']}%):")
    print(best_system_message)
    
    # Optimization with self-improvement of the generator
    result_with_improvement = await optimize_system_messages_tool(
        action="optimize",
        user_message="What's the best way to learn programming?",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        num_candidates=5,
        diversity_level="high",
        self_optimize=True  # This will trigger self-optimization
    )
    
    # Check if self-optimization was successful
    if "self_optimization" in result_with_improvement and "error" not in result_with_improvement["self_optimization"]:
        print(f"Generator self-optimized from v{result_with_improvement['self_optimization']['old_version']} to v{result_with_improvement['self_optimization']['new_version']}")
    ```
    
    ### Retrieving a Past Optimization Run
    
    ```python
    # Get a previous optimization run by ID
    past_optimization = await optimize_system_messages_tool(
        action="get",
        optimization_run_id="9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
    )
    
    # Access the optimization data
    run = past_optimization["optimization_run"]
    print(f"Optimization from {run['timestamp']}")
    print(f"Best system message: {run['candidates'][0]['content']}")
    ```
    
    ### Listing Optimization Runs
    
    ```python
    # List all optimization runs (paginated)
    all_runs = await optimize_system_messages_tool(
        action="list",
        page=1,
        page_size=10
    )
    
    # Print summary of runs
    print(f"Found {all_runs['pagination']['total_count']} optimization runs")
    for run in all_runs['optimization_runs']:
        print(f"Run {run['id']}: {run['user_message'][:30]}...")
    
    # Filter runs by EvalSet ID
    filtered_runs = await optimize_system_messages_tool(
        action="list",
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        page=1,
        page_size=10
    )
    ```
    """
    logger.info(f"optimize_system_messages_tool called with action={action}")
    
    # Create progress tracking callback for real-time updates
    progress_updates = []
    
    def track_progress(current: int, total: int, message: str):
        """Track progress for the optimization process."""
        progress = {
            "current": current,
            "total": total,
            "percent": round((current / total) * 100, 1),
            "message": message
        }
        progress_updates.append(progress)
        logger.debug(f"Progress: {progress['percent']}% - {message}")
    
    try:
        # Call the manage_optimization_runs function with progress tracking
        result = await manage_optimization_runs(
            action=action,
            user_message=user_message,
            evalset_id=evalset_id,
            base_system_message=base_system_message,
            num_candidates=num_candidates,
            generator_id=generator_id,
            generator_model=generator_model,
            diversity_level=diversity_level,
            max_parallel=max_parallel,
            additional_instructions=additional_instructions,
            optimization_run_id=optimization_run_id,
            page=page,
            page_size=page_size,
            self_optimize=self_optimize,
            progress_callback=track_progress
        )
        
        # Add progress updates to the result if available
        if progress_updates and action == "optimize":
            result["progress"] = progress_updates
            
        return result
            
    except Exception as e:
        logger.error(f"Error in optimize_system_messages_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        error_response = {"error": error_msg}
        
        # Enhanced error messages with detailed troubleshooting information
        if action == "optimize":
            if "evalset_id" in error_msg and "not found" in error_msg:
                error_response["details"] = "The evalset_id you provided doesn't exist in the system."
                error_response["troubleshooting"] = [
                    "First, use manage_evalset_tool(action=\"list\") to see all available EvalSets",
                    "Check that you've copied the EvalSet ID correctly, including all hyphens",
                    "If you need to create a new EvalSet, use manage_evalset_tool(action=\"create\", ...)",
                    "EvalSet IDs are case-sensitive UUIDs in the format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                ]
            elif "user_message" in error_msg:
                error_response["details"] = "The user_message parameter is required for optimization."
                error_response["troubleshooting"] = [
                    "Provide a user_message parameter with the query you want to optimize for",
                    "Make sure the user_message is not empty",
                    "Try with a specific, clear user query for best results"
                ]
            elif "num_candidates" in error_msg:
                error_response["details"] = "The num_candidates parameter has an invalid value."
                error_response["troubleshooting"] = [
                    "num_candidates must be between 1 and 20",
                    "The recommended range is 3-10 for most use cases",
                    "Higher values give more options but take longer to evaluate"
                ]
            else:
                # Generic troubleshooting for other errors
                error_response["troubleshooting"] = [
                    "Check that your EvalSet exists and has valid questions",
                    "Verify that the user message is clear and specific",
                    "Try with default parameters first before customizing",
                    "Check for proper formatting of all parameters",
                    "Try setting AGENTOPTIM_DEBUG=1 for more verbose logs"
                ]
        elif action == "get":
            if "optimization_run_id" in error_msg or "not found" in error_msg:
                error_response["details"] = "The optimization_run_id you provided doesn't exist in the system."
                error_response["troubleshooting"] = [
                    "First, use optimize_system_messages_tool(action=\"list\") to see all available optimization runs",
                    "Check that you've copied the run ID correctly, including all hyphens",
                    "Run IDs are case-sensitive UUIDs in the format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                ]
        elif action == "list":
            if "page" in error_msg or "page_size" in error_msg:
                error_response["details"] = "Invalid pagination parameters."
                error_response["troubleshooting"] = [
                    "page must be a positive integer (1 or greater)",
                    "page_size must be between 1 and 100",
                    "Try the default parameters: page=1, page_size=10"
                ]
        else:
            # Generic troubleshooting for other errors
            error_response["troubleshooting"] = [
                "Check the action parameter: must be one of 'optimize', 'get', or 'list'",
                "For 'optimize', make sure user_message and evalset_id are provided",
                "For 'get', make sure optimization_run_id is provided",
                "For 'list', both page and page_size are optional (default: page=1, page_size=10)"
            ]
        
        return error_response

def get_cache_stats() -> dict:
    """
    Get statistics about the caching system for monitoring and diagnostics.
    
    This function provides detailed information about the cache performance for both
    EvalSets, API responses, and evaluation runs to assist with monitoring and optimization.
    
    Returns:
        A dictionary containing cache statistics
    """
    from agentoptim.evalrun import get_eval_runs_cache_stats
    
    evalset_stats = get_cache_statistics()
    api_stats = get_api_cache_stats()
    evalruns_stats = get_eval_runs_cache_stats()
    sysopt_stats = get_sysopt_stats()
    
    # Calculate overall stats
    total_hits = evalset_stats["hits"] + api_stats["hits"] + evalruns_stats["hits"]
    total_misses = evalset_stats["misses"] + api_stats["misses"] + evalruns_stats["misses"]
    total_requests = total_hits + total_misses
    overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
    
    # Format a message with the statistics
    formatted_message = "\n".join([
        "# Cache Performance Statistics",
        "",
        "## EvalSet Cache",
        f"- Size: {evalset_stats['size']} / {evalset_stats['capacity']} (current/max)",
        f"- Hit Rate: {evalset_stats['hit_rate_pct']}%",
        f"- Hits: {evalset_stats['hits']}",
        f"- Misses: {evalset_stats['misses']}",
        f"- Evictions: {evalset_stats['evictions']}",
        f"- Expirations: {evalset_stats['expirations']}",
        "",
        "## API Response Cache",
        f"- Size: {api_stats['size']} / {api_stats['capacity']} (current/max)",
        f"- Hit Rate: {api_stats['hit_rate_pct']}%",
        f"- Hits: {api_stats['hits']}",
        f"- Misses: {api_stats['misses']}",
        f"- Evictions: {api_stats['evictions']}",
        f"- Expirations: {api_stats['expirations']}",
        "",
        "## Eval Runs Cache",
        f"- Size: {evalruns_stats['size']} / {evalruns_stats['capacity']} (current/max)",
        f"- Hit Rate: {evalruns_stats['hit_rate_pct']}%",
        f"- Hits: {evalruns_stats['hits']}",
        f"- Misses: {evalruns_stats['misses']}",
        f"- Evictions: {evalruns_stats['evictions']}",
        f"- Expirations: {evalruns_stats['expirations']}",
        "",
        "## System Optimization Stats",
        f"- Total Optimization Runs: {sysopt_stats.get('total_optimization_runs', 0)}",
        f"- Total Generators: {sysopt_stats.get('total_generators', 0)}",
        "",
        "## Overall Performance",
        f"- Combined Hit Rate: {round(overall_hit_rate, 2)}%",
        f"- Total Hits: {total_hits}",
        f"- Total Misses: {total_misses}",
        f"- Resource Savings: Approximately {round(total_hits * 0.5, 1)} seconds of API processing time saved"
    ])
    
    return {
        "status": "success",
        "evalset_cache": evalset_stats,
        "api_cache": api_stats,
        "evalrun_cache": evalruns_stats,
        "sysopt_stats": sysopt_stats,
        "overall": {
            "hit_rate_pct": round(overall_hit_rate, 2),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "estimated_time_saved_seconds": round(total_hits * 0.5, 1)  # Assuming 0.5s per cached response
        },
        "formatted_message": formatted_message
    }


def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
"""MCP server implementation for AgentOptim v2.0."""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

from mcp.server.fastmcp import FastMCP

# Import EvalSet tools
from agentoptim.evalset import manage_evalset
from agentoptim.runner import run_evalset

# Import necessary utilities
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Ensure data directories exist
ensure_data_directories()

# Check for debug and LM Studio compatibility modes
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"
# Our testing showed LM Studio requires special handling:
# 1. No response_format parameter (causes 400 error)
# 2. No logprobs support (always returns null)
# 3. System prompts work well and help control the output format
LMSTUDIO_COMPAT = os.environ.get("AGENTOPTIM_LMSTUDIO_COMPAT", "1") == "1"  # Enable by default
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
logger.info(f"Logging to {log_file_path}")
logger.info(f"Debug mode: {DEBUG_MODE}")
logger.info(f"LM Studio compatibility mode: {LMSTUDIO_COMPAT}")

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
async def run_evalset_tool(
    evalset_id: str,
    conversation: List[Dict[str, str]],
    model: Optional[str] = None,  # Will use judge_model from options if not provided
    max_parallel: int = 3
) -> dict:
    """
    Execute a comprehensive evaluation of conversation quality using an EvalSet's criteria.
    
    ## Overview
    This tool systematically evaluates a conversation against a predefined set of criteria (an EvalSet),
    using an LLM as a judge. It provides detailed insights into conversation quality with reasoned 
    judgments, confidence scores, and summary statistics.
    
    ## How It Works
    1. The tool loads the specified EvalSet and its evaluation questions
    2. For each question, it prompts an LLM judge model to evaluate the conversation
    3. The judge model provides a reasoned judgment, confidence score, and yes/no determination
    4. Results are aggregated into detailed per-question results and summary statistics
    
    ## Arguments
    
    evalset_id: The unique identifier (UUID) of the EvalSet to use.
              REQUIRED. Must reference an existing EvalSet created with manage_evalset_tool.
              Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
             
    conversation: A chronological list of conversation messages to evaluate.
                REQUIRED. Each message must be a dictionary with "role" and "content" fields.
                Valid roles include: "system", "user", "assistant"
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "How do I reset my password?"},
                    {"role": "assistant", "content": "Go to the login page and click 'Forgot Password'."}
                ]
                
    model: The LLM to use as the evaluation judge.
          OPTIONAL. If not provided, will use:
          1. First, the "judge_model" option from client configuration if available
          2. Otherwise, falls back to "meta-llama-3.1-8b-instruct"
          Should be a model capable of reasoning about conversations and following instructions.
          Example: "meta-llama-3.1-8b-instruct", "gpt-4o-mini", or "anthropic/claude-3-sonnet"
          
    max_parallel: Maximum number of evaluation questions to process simultaneously.
                OPTIONAL. Defaults to 3.
                Higher values can improve speed but increase resource requirements.
                Recommended range: 1-5 (depending on your hardware)
                Example: 5 for faster evaluation on powerful systems
                
    The following tool options are available in client configuration:
      - judge_model: The LLM to use as the evaluation judge
      - omit_reasoning: If set to "True", don't generate detailed reasoning in results
        Example client configuration with omit_reasoning:
        ```json
        {
          "mcpServers": {
            "optim": {
              "command": "...",
              "args": ["..."],
              "options": {
                "judge_model": "gpt-4o-mini",
                "omit_reasoning": "True"
              }
            }
          }
        }
        ```
    
    ## Return Value
    
    The tool returns a comprehensive evaluation result with these components:
    
    - status: Success/error status of the evaluation
    - id: Unique ID for this evaluation run
    - evalset_id: ID of the EvalSet used
    - evalset_name: Name of the EvalSet used
    - model: Name of the judge model used
    - results: List of detailed results for each evaluation question, including:
      - question: The evaluation question
      - judgment: Boolean true/false indicating yes/no answer
      - confidence: Number between 0-1 indicating confidence level
      - reasoning: The judge's explanation for their judgment
    - summary: Aggregated statistics including:
      - total_questions: Number of questions evaluated
      - successful_evaluations: Number of questions successfully evaluated
      - yes_count & no_count: Counts of yes/no judgments
      - yes_percentage: Percentage of yes responses
      - mean_confidence: Average confidence across all judgments
      - mean_yes_confidence & mean_no_confidence: Average confidence for yes/no judgments
    - result: A formatted markdown report of the evaluation results
    
    ## Usage Examples
    
    ### Basic Usage
    
    ```python
    # Evaluate a simple password reset conversation
    results = run_evalset_tool(
        evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
        conversation=[
            {"role": "system", "content": "You are a helpful technical support assistant."},
            {"role": "user", "content": "I forgot my password and can't log in. How do I reset it?"},
            {"role": "assistant", "content": "To reset your password, please follow these steps:\n\n1. Go to the login page\n2. Click on the 'Forgot Password' link below the login form\n3. Enter the email address associated with your account\n4. Check your email for a password reset link\n5. Click the link and follow the instructions to create a new password\n\nIf you don't receive the email within a few minutes, please check your spam folder."}
        ],
        model="meta-llama-3.1-8b-instruct",
        max_parallel=2
    )
    
    # Get the overall success rate
    success_rate = results["summary"]["yes_percentage"]
    print(f"Response quality score: {success_rate}%")
    
    # Get mean confidence score
    mean_confidence = results["summary"]["mean_confidence"]
    print(f"Mean confidence: {mean_confidence:.2f}")
    
    # Access individual evaluation results
    for result in results["results"]:
        print(f"Q: {result['question']}")
        print(f"A: {'Yes' if result['judgment'] else 'No'} (confidence: {result['confidence']:.2f})")
        print(f"Reasoning: {result['reasoning']}\n")
    
    # Compare multiple conversation strategies
    def evaluate_response(response_text):
        return run_evalset_tool(
            evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
            conversation=[
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": response_text}
            ]
        )["summary"]["yes_percentage"]
    
    response1_score = evaluate_response("Go to Settings > Account > Reset Password.")
    response2_score = evaluate_response("To reset your password, go to the login page and click 'Forgot Password'.")
    response3_score = evaluate_response("To reset your password, follow these steps:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Follow the on-screen instructions")
    
    print(f"Response 1: {response1_score}%")
    print(f"Response 2: {response2_score}%")
    print(f"Response 3: {response3_score}%")
    ```
    """
    logger.info(f"run_evalset_tool called with evalset_id={evalset_id}")
    try:
        # Check for judge_model and omit_reasoning in options (passed from client config)
        judge_model = DEFAULT_JUDGE_MODEL  # Use env var as first priority
        omit_reasoning = DEFAULT_OMIT_REASONING  # Use env var as first priority
        
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
                        judge_model = options['judge_model']
                        logger.info(f"Using judge_model from client options: {judge_model}")
                    
                    # Check for omit_reasoning flag (only if env var not set)
                    if not DEFAULT_OMIT_REASONING and 'omit_reasoning' in options:
                        omit_reasoning_str = str(options['omit_reasoning']).lower()
                        # Accept various forms of "true" for better user experience
                        omit_reasoning = omit_reasoning_str in ('true', 't', '1', 'yes', 'y', 'on', 'enabled')
                        logger.info(f"omit_reasoning option set to: {omit_reasoning}")
                        
                if omit_reasoning:
                    logger.info("Reasoning will be omitted from evaluation results")
        except (AttributeError, LookupError) as e:
            # This can happen during tests or when no request context is available
            logger.debug(f"Could not access request context: {e}")
        
        # Use model parameter if provided, otherwise use judge_model from options/env,
        # or fall back to default model
        eval_model = model or judge_model or "meta-llama-3.1-8b-instruct"
        logger.info(f"Evaluating with model: {eval_model}")
                
        # Call the async function and await its result
        result = await run_evalset(
            evalset_id=evalset_id,
            conversation=conversation,
            model=eval_model,
            max_parallel=max_parallel,
            omit_reasoning=omit_reasoning
        )
        
        # If result is a dictionary with a formatted_message, use that
        if isinstance(result, dict) and "formatted_message" in result:
            # Keep the formatted message but also include the structured data
            result_copy = result.copy()
            # Move formatted_message to result field to maintain compatibility
            result_copy["result"] = result_copy.pop("formatted_message")
            return result_copy
            
        # If result is a string, wrap it in a dict
        if isinstance(result, str):
            return {"result": result}
            
        # Otherwise return as is (should be a dict)
        return result
    except Exception as e:
        logger.error(f"Error in run_evalset_tool: {str(e)}", exc_info=True)
        error_msg = str(e)
        error_response = {"error": error_msg}
        
        # Enhanced error messages with detailed troubleshooting information
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
                "The default model 'meta-llama-3.1-8b-instruct' should work with LM Studio",
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
        
        return error_response


def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
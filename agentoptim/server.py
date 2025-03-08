"""MCP server implementation for AgentOptim v2.0."""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any

from mcp.server.fastmcp import FastMCP

# Import EvalSet tools
from agentoptim.evalset import manage_evalset
from agentoptim.runner import run_evalset

# Import necessary utilities
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Ensure data directories exist
ensure_data_directories()

# Check for debug mode from environment variable
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"
LMSTUDIO_COMPAT = os.environ.get("AGENTOPTIM_LMSTUDIO_COMPAT", "0") == "1"

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
    template: Optional[str] = None,
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> dict:
    """
    Manage EvalSet definitions for assessing conversation quality.
    
    This tool helps manage EvalSets, which define a set of yes/no questions and a template
    for evaluating conversations. The template formats the conversation and evaluation question
    for the judge model.
    
    Args:
        action: The operation to perform. Must be one of:
               - "create" - Create a new EvalSet
               - "list" - List all available EvalSets
               - "get" - Get details of a specific EvalSet
               - "update" - Update an existing EvalSet
               - "delete" - Delete an EvalSet
               
        evalset_id: The unique identifier of the EvalSet.
                  Required for "get", "update", and "delete" actions.
                  Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                  
        name: The name of the EvalSet.
              Required for "create" action.
              Example: "Response Quality Evaluation"
              
        template: The template string that formats the input for the judge model.
                 Required for "create" action, optional for "update".
                 Must include placeholders for {{ conversation }} and {{ eval_question }}.
                 Example: "Given this conversation: {{ conversation }}\n\nPlease answer this question about the final response: {{ eval_question }}"
                 
        questions: A list of yes/no questions to evaluate responses against.
                  Required for "create" action, optional for "update".
                  Each question should be answerable with yes/no.
                  Example: ["Is the response clear and concise?", "Does the response answer the question?"]
                  
        description: Optional description of the EvalSet purpose.
                    Example: "Evaluates responses for clarity, conciseness, and helpfulness"
    
    Returns:
        For "list" action: A formatted list of all EvalSets with their IDs and names
        For "get" action: Detailed information about the specified EvalSet
        For "create" action: Confirmation message with the new EvalSet ID
        For "update" action: Confirmation message that the EvalSet was updated
        For "delete" action: Confirmation message that the EvalSet was deleted
    
    Example:
        ```python
        # Create an EvalSet
        evalset = manage_evalset_tool(
            action="create",
            name="Response Quality Evaluation",
            template='''
            Given this conversation:
            {{ conversation }}
            
            Please answer the following yes/no question about the final assistant response:
            {{ eval_question }}
            
            Return a JSON object with the following format:
            {"judgment": 1} for yes or {"judgment": 0} for no.
            ''',
            questions=[
                "Does the response directly address the user's question?",
                "Is the response polite and professional?",
                "Does the response provide a complete solution?",
                "Is the response clear and easy to understand?"
            ],
            description="Evaluation criteria for response quality"
        )
        ```
    """
    logger.info(f"manage_evalset_tool called with action={action}")
    try:
        result = manage_evalset(
            action=action,
            evalset_id=evalset_id,
            name=name,
            template=template,
            questions=questions,
            description=description,
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
        elif "required parameters" in error_msg:
            # Show different examples based on action
            if action == "create":
                example = {
                    "action": "create",
                    "name": "Response Quality Evaluation",
                    "template": "Given this conversation: {{ conversation }}\n\nPlease answer this question about the final response: {{ eval_question }}\n\nReturn a JSON object with the following format:\n{\"judgment\": 1} for yes or {\"judgment\": 0} for no.",
                    "questions": [
                        "Is the response clear and concise?",
                        "Does the response fully answer the question?",
                        "Is the response accurate?"
                    ],
                    "description": "Evaluates responses for clarity, completeness, and accuracy"
                }
                error_response["details"] = "The 'create' action requires name, template, and questions parameters."
                error_response["example"] = example
            elif action in ["get", "update", "delete"]:
                example = {
                    "action": action,
                    "evalset_id": "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                }
                # Add fields for update
                if action == "update":
                    example["template"] = "New template text with {{ conversation }} and {{ eval_question }}"
                    example["questions"] = ["Updated question 1", "Updated question 2"]
                
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
    model: str = "meta-llama-3.1-8b-instruct",
    max_parallel: int = 3
) -> dict:
    """
    Run an EvalSet evaluation on a conversation.
    
    This tool evaluates a conversation against an EvalSet's yes/no questions,
    returning detailed results and a summary of the evaluation.
    
    Args:
        evalset_id: The unique identifier of the EvalSet to use.
                   Must reference an existing EvalSet created with manage_evalset_tool.
                   Example: "6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e"
                  
        conversation: List of conversation messages in the format:
                     [{"role": "system", "content": "..."}, 
                      {"role": "user", "content": "..."},
                      {"role": "assistant", "content": "..."}]
                     Each message must have at least "role" and "content" fields.
                    
        model: Name of the model to use for evaluation (acts as judge).
               Optional, defaults to "meta-llama-3.1-8b-instruct".
               Example: "claude-3-haiku-20240307" or "lmstudio-community/meta-llama-3.1-8b-instruct"
               
        max_parallel: Maximum number of questions to evaluate in parallel.
                     Optional, defaults to 3.
                     Higher values may speed up execution but increase resource usage.
                     Example: 5
    
    Returns:
        Detailed results of the evaluation, including:
        - Summary statistics (success rate, yes/no percentages)
        - Detailed results for each question
        - Judgment (yes/no) and logprob for each question
    
    Example:
        ```python
        # Run an evaluation on a conversation
        results = run_evalset_tool(
            evalset_id="6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e",
            conversation=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": "To reset your password, please go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password."}
            ],
            model="meta-llama-3.1-8b-instruct"
        )
        ```
    """
    logger.info(f"run_evalset_tool called with evalset_id={evalset_id}")
    try:
        # Call the async function and await its result
        result = await run_evalset(
            evalset_id=evalset_id,
            conversation=conversation,
            model=model,
            max_parallel=max_parallel
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
        
        # Enhance error messages for common problems
        if "evalset_id" in error_msg and "not found" in error_msg:
            error_response["details"] = "The evalset_id you provided doesn't exist. Use manage_evalset_tool(action=\"list\") to see available EvalSets."
        elif "conversation" in error_msg:
            example_conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well! How can I help you today?"}
            ]
            error_response["details"] = "The conversation parameter must be a list of message objects, each with 'role' and 'content' fields."
            error_response["example"] = example_conversation
        elif "max_parallel" in error_msg:
            error_response["details"] = "The max_parallel parameter must be a positive integer that controls how many evaluations to run in parallel."
        elif "model" in error_msg:
            error_response["details"] = "The model parameter must be a valid model identifier. Make sure your LLM server supports this model."
        
        return error_response


def main():
    """Run the MCP server."""
    logger.info("Starting AgentOptim MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
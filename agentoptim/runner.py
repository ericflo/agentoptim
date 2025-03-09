"""Runner functionality for evaluating conversations with EvalSets."""

import os
import json
import httpx
import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import jinja2
import uuid

from agentoptim.evalset import get_evalset
from agentoptim.cache import cached, LRUCache
from agentoptim.utils import (
    format_error,
    format_success,
    ValidationError,
    validate_required_params,
)

# Configure logging
logger = logging.getLogger(__name__)

# Ensure we have at least one file handler for comprehensive logging
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug_agentoptim.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Added file handler for debugging to {log_path}")

# Check for LM Studio compatibility mode
# Our testing showed LM Studio requires special handling
# 1. Special response_format schema structure
# 2. System prompts work well and help control the output format
# 3. Be explicit about expected JSON formatting
LMSTUDIO_COMPAT = os.environ.get("AGENTOPTIM_LMSTUDIO_COMPAT", "1") == "1"  # Enable by default
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"

# Log compatibility mode
if LMSTUDIO_COMPAT:
    logger.info("LM Studio compatibility mode is ENABLED")
    logger.info("* Using special response_format structure for json_schema")
    logger.info("* Using system prompts for better control")
    logger.info("* Using verbalized confidence scores")
else:
    logger.info("LM Studio compatibility mode is DISABLED")

# Default timeout for API calls
DEFAULT_TIMEOUT = 60  # seconds

# Get API base URL from environment or use default
API_BASE = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234/v1")

# Create LRU cache for API responses to avoid re-evaluating identical questions/conversations
# Default: 100 items capacity, 1 hour TTL
API_RESPONSE_CACHE = LRUCache(capacity=100, ttl=3600)


class EvalResult(BaseModel):
    """Model for a single evaluation result.
    
    Attributes:
        question: The evaluation question being answered
        judgment: Boolean value (true/false) indicating the yes/no judgment
        confidence: Number between 0.0 and 1.0 indicating confidence level
        reasoning: Detailed explanation for the judgment (None when omit_reasoning=True)
        error: Error message if evaluation failed, None otherwise
    """
    
    question: str
    judgment: Optional[bool] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None


class EvalResults(BaseModel):
    """Model for evaluation results.
    
    Attributes:
        evalset_id: ID of the EvalSet used for evaluation
        evalset_name: Name of the EvalSet used
        results: List of EvalResult objects with evaluation results
                (reasoning field will be None when omit_reasoning=True)
        conversation: List of conversation messages that were evaluated
        summary: Dictionary with summary statistics (yes/no counts, percentages, etc.)
    """
    
    evalset_id: str
    evalset_name: str
    results: List[EvalResult]
    conversation: List[Dict[str, str]]
    summary: Dict[str, Any] = Field(default_factory=dict)


async def call_llm_api(
    prompt: Optional[str] = None,
    model: str = "meta-llama-3.1-8b-instruct",
    temperature: float = 0.0,
    max_tokens: int = 1024,  # Increased to 1024 to ensure complete responses
    logit_bias: Optional[Dict[int, float]] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    omit_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Call the LLM API to get structured model response with verbalized confidence.
    
    Args:
        prompt: The prompt to send to the model (legacy parameter)
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate (default: 1024 to ensure complete responses)
        logit_bias: Optional logit bias to apply
        messages: List of message objects with role and content (preferred over prompt)
    
    Returns:
        Response from the LLM API
    """
    # Initialize response_json so it's accessible in except block
    response_json = {}
    response = None
    
    try:
        # Allow either messages or prompt parameter
        if messages is None:
            if prompt is None:
                raise ValueError("Either messages or prompt must be provided")
            messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # No longer using logprobs - we use verbalized confidence scores instead
        # This provides better compatibility across different providers
        
        # LM Studio requires a specific response_format with schema field
        if LMSTUDIO_COMPAT:
            logger.info("Using LM Studio-compatible JSON schema format")
            # Use the CORRECT JSON schema format for LM Studio
            # Following their API requirements exactly
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judgment_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "judgment": {
                                "type": "boolean", 
                                "description": "Your yes/no judgment as a boolean value: true if the answer to the evaluation question is 'yes', false if the answer is 'no'. Must be a proper JSON boolean literal (true or false, not True/False)."
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "A number between 0.0 and 1.0 indicating how confident you are in your judgment. Use 0.9-1.0 for very high confidence, 0.7-0.8 for moderate confidence, 0.5-0.6 for medium confidence, and below 0.5 for low confidence. Be well-calibrated - don't simply use 1.0 for everything."
                            },
                            **({"reasoning": {
                                "type": "string",
                                "description": "Provide a clear, detailed explanation justifying your judgment. Include relevant evidence from the conversation and specific reasoning that led to your conclusion. This field should help others understand exactly why you determined the response was helpful or not helpful."
                            }} if not omit_reasoning else {})
                        },
                        "required": ["judgment", "confidence"] + (["reasoning"] if not omit_reasoning else []),
                        "additionalProperties": False
                    }
                }
            }
        else:
            # For other providers, use standard OpenAI format
            logger.info("Using standard OpenAI JSON response format")
            payload["response_format"] = {
                "type": "json_object"
            }
        
        if logit_bias:
            payload["logit_bias"] = logit_bias
        
        headers = {"Content-Type": "application/json"}
        
        # Add OpenAI API key if LMSTUDIO_COMPAT is disabled and OPENAI_API_KEY is set
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not LMSTUDIO_COMPAT and openai_api_key and "openai.com" in API_BASE:
            logger.info("Using OpenAI API with authentication")
            headers["Authorization"] = f"Bearer {openai_api_key}"
        
        logger.info(f"Calling LLM API with model: {model}")
        if DEBUG_MODE:
            logger.debug(f"API request payload: {json.dumps(payload, indent=2)}")
            # Log the headers (except Authorization to protect the API key)
            safe_headers = {k: v if k.lower() != "authorization" else "Bearer sk-***" for k, v in headers.items()}
            logger.debug(f"API headers: {json.dumps(safe_headers, indent=2)}")
        
        # Try up to 3 times with different payloads for LM Studio compatibility
        max_retries = 3
        retry_count = 0
        last_error = None
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            while retry_count < max_retries:
                try:
                    logger.info(f"API call attempt {retry_count+1}/{max_retries}")
                    if retry_count > 0:
                        # On retry, simplify the payload
                        # Increase max tokens on retries in case we're hitting length limits
                        payload["max_tokens"] = max_tokens * (retry_count + 1)
                        logger.info(f"Retry: Increased max_tokens to {payload['max_tokens']}")
                        
                        if "response_format" in payload:
                            del payload["response_format"]
                            logger.info("Retry: Removed response_format parameter")
                    
                    if DEBUG_MODE:
                        logger.debug(f"Retry {retry_count} payload: {json.dumps(payload, indent=2)}")
                    
                    response = await client.post(
                        f"{API_BASE}/chat/completions", 
                        json=payload,
                        headers=headers
                    )
                    
                    if response.status_code == 200:
                        break  # Success, exit the retry loop
                    
                    # If we get here, we had an error
                    error_msg = f"LLM API error: {response.status_code}"
                    try:
                        error_details = response.json()
                        error_msg += f", {json.dumps(error_details)}"
                        
                        # Special handling for auth errors
                        if response.status_code == 401:
                            if "openai.com" in API_BASE:
                                if not openai_api_key:
                                    logger.error("Authentication failed: OPENAI_API_KEY environment variable is not set")
                                else:
                                    logger.error("Authentication failed: The provided OpenAI API key was rejected")
                                    # Check if the key looks properly formatted
                                    if not openai_api_key.startswith("sk-"):
                                        logger.error("The API key doesn't start with 'sk-', which is unusual for OpenAI keys")
                    except:
                        error_msg += f", {response.text}"
                    
                    logger.error(f"Attempt {retry_count+1} failed: {error_msg}")
                    last_error = error_msg
                    retry_count += 1
                    
                except Exception as e:
                    error_msg = f"Exception during API call: {str(e)}"
                    logger.error(error_msg)
                    last_error = error_msg
                    retry_count += 1
            
            # Check if we succeeded or exhausted retries
            if response.status_code != 200:
                error_msg = last_error or f"LLM API error: {response.status_code}"
                logger.error(f"All retries failed. Last error: {error_msg}")
                return {
                    "error": error_msg,
                    "details": response.text if hasattr(response, 'text') else "No response text"
                }
            
            response_json = response.json()
            if DEBUG_MODE:
                logger.debug(f"API response: {json.dumps(response_json, indent=2)}")
            
            # Handle LM Studio compatibility mode
            if "choices" in response_json and response_json["choices"]:
                choice = response_json["choices"][0]
                
                # No longer using or expecting logprobs
                # We use verbalized confidence scores for all models including LM Studio
                # For debugging only - analyze content to see what the model likely determined
                content = ""
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].lower()
                elif "text" in choice:
                    content = choice["text"].lower()
                    
                # If LM Studio is using text field instead of message, convert it
                if "text" in choice and "message" not in choice:
                    logger.info("Converting LM Studio 'text' field to 'message' format")
                    choice["message"] = {
                        "role": "assistant", 
                        "content": choice["text"]
                    }
            
            return response_json
    
    except Exception as e:
        # Special handling for common errors
        error_str = str(e)
        # Using 'in' search instead of any() to be more robust
        if isinstance(e, NameError) and ('true' in error_str.lower() or 'false' in error_str.lower()):
            # This is common when the LLM returns literals like 'true' or 'false' in the response
            error_msg = f"JSON parsing error: {error_str}. The model is using unquoted JSON literals."
            logger.error(error_msg)
            
            # Extract the actual content that's causing the issue
            model_content = "unknown"
            if "choices" in response_json and response_json["choices"]:
                choice = response_json["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    model_content = choice["message"]["content"]
                elif "text" in choice:
                    model_content = choice["text"]
            
            # Try to fix the response directly by replacing unquoted literals
            try:
                if "false" in error_str:
                    fixed_content = model_content.replace("false", "false")  # This is a no-op but marks it as handled
                    logger.info("Trying to fix unquoted 'false' in model response")
                    # Parse the model's output as Python code and convert to JSON
                    import ast
                    # Replace all true/false literals with True/False
                    python_fixed = model_content.replace("true", "True").replace("false", "False")
                    try:
                        # Parse as Python dict literal
                        parsed = ast.literal_eval(python_fixed)
                        # Process the parsed result
                        if isinstance(parsed, dict):
                            if "judgment" in parsed:
                                if parsed["judgment"] in [True, False, 0, 1]:
                                    return {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": json.dumps({
                                                        "reasoning": parsed.get("reasoning", "No reasoning provided"),
                                                        "judgment": bool(parsed["judgment"]),
                                                        "confidence": float(parsed.get("confidence", 0.5))
                                                    })
                                                }
                                            }
                                        ]
                                    }
                    except (SyntaxError, ValueError) as syn_err:
                        logger.warning(f"Failed to fix JSON with Python literal_eval: {syn_err}")
            except Exception as fix_err:
                logger.warning(f"Failed to fix JSON response: {fix_err}")
            
            # Comprehensive, helpful error message with detailed instructions
            return {
                "error": "JSON parsing error: The model response contains unquoted JSON boolean literals or invalid syntax.",
                "details": "This error typically occurs when the model outputs unquoted 'true' or 'false' values or uses Python-style 'True/False' instead of JSON literals. The response must be valid JSON with proper boolean literals.",
                "troubleshooting_steps": [
                    "Check that your template emphasizes using proper JSON format",
                    "Add a system message that explicitly requires JSON boolean literals",
                    "Ensure the model is properly handling the json_schema response format",
                    "Try a different model that has better JSON formatting capabilities",
                    "Try using a more explicit example in your prompt"
                ],
                "model_content": model_content,  # Include for debugging
                "raw_error": str(e)  # Include the actual error for debugging
            }
        else:
            # Handle other exceptions with detailed error information
            error_message = str(e)
            error_type = type(e).__name__
            logger.error(f"Error calling LLM API: {error_type}: {error_message}")
            
            # Get traceback for more debugging information
            import traceback
            tb_str = traceback.format_exc()
            logger.error(f"Traceback: {tb_str}")
            
            # Create a detailed error response with troubleshooting suggestions
            return {
                "error": f"LLM API Error: {error_type}",
                "details": error_message,
                "traceback": tb_str[:500] + "..." if len(tb_str) > 500 else tb_str,  # Include truncated traceback
                "troubleshooting_steps": [
                    "Check that the LLM server is running and accessible",
                    "Verify the model name is correct and available in your LLM server or cloud provider",
                    "Ensure your API_BASE environment variable points to the correct endpoint",
                    "Verify your API key is correct and has access to the selected model",
                    "For OpenAI, set OPENAI_API_KEY environment variable with a valid API key",
                    "For OpenAI, set AGENTOPTIM_LMSTUDIO_COMPAT=0 to disable LM Studio mode",
                    "Check for API rate limits or quota issues",
                    "Try with a shorter prompt if hitting context length limits",
                    "Inspect the server logs for more information",
                    "If using LM Studio, make sure it's properly configured and responding",
                    "Try with a different model that has better JSON compatibility"
                ],
                "request_info": {
                    "model": model,
                    "api_base": API_BASE,
                    "timeout": DEFAULT_TIMEOUT
                }
            }


async def evaluate_question(
    conversation: List[Dict[str, str]],
    question: str,
    template: str,
    judge_model: str = "meta-llama-3.1-8b-instruct",
    omit_reasoning: bool = False
) -> EvalResult:
    """
    Evaluate a single question against a conversation.
    
    Utilizes a cache to avoid redundant evaluations of the same conversation/question pairs.
    
    Args:
        conversation: List of conversation messages
        question: The evaluation question
        template: The evaluation template
        judge_model: LLM model to use for evaluation
        omit_reasoning: If True, don't generate or include reasoning in results
    
    Returns:
        Evaluation result for the question
    """
    # Generate a cache key for this specific evaluation
    # We need to include all parameters that affect the result
    # Convert conversation to a tuple of tuples for hashability
    conv_tuple = tuple(tuple((k, v) for k, v in msg.items()) for msg in conversation)
    cache_key = (conv_tuple, question, judge_model, omit_reasoning)
    
    # Check if we have a cached result for this evaluation
    cached_result = API_RESPONSE_CACHE.get(cache_key)
    if cached_result is not None:
        logger.info(f"Cache hit for question: {question}")
        return cached_result
    try:
        # Format the conversation for better readability
        formatted_conversation = "### BEGIN CONVERSATION ###\n\n"
        for message in conversation:
            role = message.get("role", "").capitalize()
            content = message.get("content", "")
            formatted_conversation += f"{role}: {content}\n\n"
        formatted_conversation += "### END CONVERSATION ###\n\n"
            
        # Create a fresh Jinja environment for each evaluation to prevent state leakage
        jinja_env = jinja2.Environment(autoescape=True)
        
        # Add a unique evaluation ID to help track this specific evaluation
        eval_id = str(uuid.uuid4())[:8]
        logger.debug(f"Evaluation ID {eval_id} for question: {question}")
        
        # Render the template with Jinja2
        jinja_template = jinja_env.from_string(template)
        rendered_prompt = jinja_template.render(
            conversation=formatted_conversation,
            # Also provide raw conversation in case templates need JSON format
            conversation_json=json.dumps(conversation, ensure_ascii=False),
            eval_question=question,
            eval_id=eval_id  # Add the evaluation ID to the context
        )
        
        # Add specific formatting guidance for LM Studio
        if LMSTUDIO_COMPAT:
            # Now that we've confirmed LM Studio supports JSON schema format
            # We'll use it to strictly enforce the response format
            
            # We can keep the JSON instructions in the prompt
            # Add a clear note about the expected format with confidence estimation
            # Based on the verbalized-uq research, this format is shown to be more effective
            if omit_reasoning:
                rendered_prompt += """

IMPORTANT INSTRUCTIONS FOR EVALUATION:

CAREFULLY ANALYZE THE EXACT CONVERSATION PROVIDED ABOVE between ### BEGIN CONVERSATION ### and ### END CONVERSATION ### markers. Respond ONLY in the valid JSON format specified below.

Your evaluation must be based ONLY on this specific conversation. Do not reference any other conversations, scenarios, or prior knowledge. Only evaluate what is explicitly present in this conversation.

Your JSON response MUST include these TWO fields:

1. "judgment": (REQUIRED)
   • Must be a boolean value: true for Yes, false for No
   • Must use JSON boolean literals (true/false)
   • Do NOT use Python booleans (True/False)
   • Do NOT use strings (like "true"/"false")
   • This field must be JSON parseable

2. "confidence": (REQUIRED)
   • A number between 0.0 and 1.0
   • Indicates how certain you are in your judgment
   • Be well-calibrated - don't just use 1.0 for everything
   • Use this scale:
     - 0.95-1.0: Virtually certain (overwhelming evidence)
     - 0.85-0.94: Very confident (strong evidence)
     - 0.70-0.84: Moderately confident (good evidence)
     - 0.50-0.69: Somewhat confident (mixed evidence)
     - Below 0.5: Limited confidence (weak evidence)

EXAMPLE OF CORRECT FORMAT:
```json
{
  "judgment": true,
  "confidence": 0.92
}
```

CRITICAL: Your response MUST be valid JSON that can be parsed. Incorrect formats will cause errors. DO NOT include a "reasoning" field in your response."""
            else:
                rendered_prompt += """

IMPORTANT INSTRUCTIONS FOR EVALUATION:

CAREFULLY ANALYZE THE EXACT CONVERSATION PROVIDED ABOVE between ### BEGIN CONVERSATION ### and ### END CONVERSATION ### markers. Respond ONLY in the valid JSON format specified below.

Your evaluation must be based ONLY on this specific conversation. Do not reference any other conversations, scenarios, or prior knowledge. Only evaluate what is explicitly present in this conversation.

Your JSON response MUST include these THREE fields:

1. "reasoning": (REQUIRED)
   • Provide a thorough explanation (3-5 sentences)
   • Include specific evidence from the EXACT conversation provided
   • Quote relevant parts of the conversation to support your judgment
   • Be objective and focus on observable elements in the response

2. "judgment": (REQUIRED)
   • Must be a boolean value: true for Yes, false for No
   • Must use JSON boolean literals (true/false)
   • Do NOT use Python booleans (True/False)
   • Do NOT use strings (like "true"/"false")
   • This field must be JSON parseable

3. "confidence": (REQUIRED)
   • A number between 0.0 and 1.0
   • Indicates how certain you are in your judgment
   • Be well-calibrated - don't just use 1.0 for everything
   • Use this scale:
     - 0.95-1.0: Virtually certain (overwhelming evidence)
     - 0.85-0.94: Very confident (strong evidence)
     - 0.70-0.84: Moderately confident (good evidence)
     - 0.50-0.69: Somewhat confident (mixed evidence)
     - Below 0.5: Limited confidence (weak evidence)

EXAMPLE OF CORRECT FORMAT:
```json
{
  "reasoning": "The assistant's response directly addresses the user's question by providing specific instructions on how to reset their password. The response is clear, concise, and gives the exact location of the 'Forgot Password' link, which is precisely what the user needs to solve their problem. The information provided would allow the user to accomplish their task without further assistance.",
  "judgment": true,
  "confidence": 0.92
}
```

CRITICAL: Your response MUST be valid JSON that can be parsed. Incorrect formats will cause errors."""
            
            # LM Studio needs simpler prompts sometimes
            if len(rendered_prompt) > 2000:
                # Shorten if necessary
                logger.info("Prompt is very long, shortening for LM Studio compatibility")
                
                # Base prompt structure that's common for both with/without reasoning
                shortened_base = f"""Please evaluate the following question about the SPECIFIC conversation provided below:

### BEGIN CONVERSATION ###
{formatted_conversation}
### END CONVERSATION ###

QUESTION: {question}

IMPORTANT: Analyze ONLY the conversation provided between the ### BEGIN CONVERSATION ### and ### END CONVERSATION ### markers. Do not reference any other conversations or prior knowledge.

You MUST provide your evaluation in JSON format with """
                
                if omit_reasoning:
                    rendered_prompt = shortened_base + """these TWO REQUIRED fields:

1. "judgment": true for Yes, false for No (must be JSON boolean: true/false)
2. "confidence": Number from 0.0 to 1.0 showing your confidence level

Example of CORRECT format:
```json
{{
  "judgment": true,
  "confidence": 0.88
}}
```

CRITICAL: Your response MUST be valid JSON. Use true/false (not True/False or strings) for the judgment field to avoid errors. DO NOT include a reasoning field in your response."""
                else:
                    rendered_prompt = shortened_base + """these THREE REQUIRED fields:

1. "reasoning": A detailed explanation of your reasoning (3-5 sentences), quoting specific parts of the conversation
2. "judgment": true for Yes, false for No (must be JSON boolean: true/false)
3. "confidence": Number from 0.0 to 1.0 showing your confidence level

Example of CORRECT format:
```json
{{
  "reasoning": "The assistant's response directly addresses the user's question by providing specific instructions. The assistant says 'To reset your password, go to settings and click on the Reset button.' This information is clear, accurate, and would enable the user to successfully complete their task without further assistance.",
  "judgment": true,
  "confidence": 0.88
}}
```

CRITICAL: Your response MUST be valid JSON. Use true/false (not True/False or strings) for the judgment field to avoid errors. Include detailed reasoning with specific evidence from THIS conversation."""
                
            # Create system prompt without f-string issues
            system_content = "You are an expert evaluation assistant that provides detailed, thoughtful judgments in valid JSON format. "
            system_content += f"Your evaluation ID is {eval_id}. "
            system_content += "You analyze ONLY the specific conversation provided to you in each request, not referring to any other conversations or knowledge.\n\n"
            system_content += "You MUST ONLY evaluate the exact conversation provided between the ### BEGIN CONVERSATION ### and ### END CONVERSATION ### markers. "
            system_content += "Do not reference details from outside this specific conversation or make assumptions based on other knowledge.\n\n"
            
            # Adjust the required fields based on omit_reasoning
            if omit_reasoning:
                system_content += "Your responses MUST include both of these required fields:\n\n"
                system_content += "1) 'judgment': A boolean true/false value (using proper JSON literals: true or false, NOT True/False or strings). "
                system_content += "Use true for 'yes' answers and false for 'no' answers to the evaluation question.\n\n"
                system_content += "2) 'confidence': A well-calibrated number between 0.0 and 1.0 indicating your confidence level. "
                system_content += "Use 0.9+ for near certainty, 0.7-0.8 for strong confidence, 0.5-0.6 for moderate confidence, and below 0.5 for low confidence. Avoid overconfidence.\n\n"
            else:
                system_content += "Your responses MUST include all THREE of these required fields:\n\n"
                system_content += "1) 'reasoning': A thorough explanation (3-5 sentences) that clearly justifies your judgment with specific evidence from the provided conversation. "
                system_content += "Quote relevant parts of the conversation to support your judgment. Explain your thought process and the criteria you used.\n\n"
                system_content += "2) 'judgment': A boolean true/false value (using proper JSON literals: true or false, NOT True/False or strings). "
                system_content += "Use true for 'yes' answers and false for 'no' answers to the evaluation question.\n\n"
                system_content += "3) 'confidence': A well-calibrated number between 0.0 and 1.0 indicating your confidence level. "
                system_content += "Use 0.9+ for near certainty, 0.7-0.8 for strong confidence, 0.5-0.6 for moderate confidence, and below 0.5 for low confidence. Avoid overconfidence.\n\n"
            
            system_content += "CRITICAL: Always use valid JSON syntax with proper JSON boolean literals (true/false). "
            system_content += "Do not use Python style True/False or quoted strings like 'true'/'false'. The response MUST be parseable as JSON. "
            
            if omit_reasoning:
                system_content += "NEVER return a response that is missing any of the two required fields.\n\n"
                system_content += "FORMAT EXAMPLE:\n```json\n"
                system_content += '{\n'
                system_content += '  "judgment": true,\n'
                system_content += '  "confidence": 0.85\n}\n```'
            else:
                system_content += "NEVER return a response that is missing any of the three required fields.\n\n"
                system_content += "FORMAT EXAMPLE:\n```json\n"
                system_content += '{\n  "reasoning": "The response clearly addresses the user\'s concern by providing step-by-step instructions to solve their problem. The assistant says to resolve the issue by taking specific steps. This demonstrates clear guidance. The tone is professional yet friendly, and the information is accurate and concise.",\n'
                system_content += '  "judgment": true,\n'
                system_content += '  "confidence": 0.85\n}\n```'
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": rendered_prompt}
            ]
        else:
            # Normal mode - just use the rendered prompt as user message
            messages = [{"role": "user", "content": rendered_prompt}]
        
        logger.info(f"Evaluating question: {question}")
        
        # Call the LLM API with proper message format
        response = await call_llm_api(
            messages=messages,  # Now passing prepared messages with system prompt for LM Studio
            model=judge_model,
            omit_reasoning=omit_reasoning
        )
        
        # Check for errors
        if "error" in response:
            return EvalResult(
                question=question,
                error=response["error"]
            )
        
        # Extract the response and parse the results
        try:
            # Check if the response has the expected structure
            if "choices" not in response or not response["choices"]:
                logger.error(f"Unexpected API response format: missing 'choices' field: {json.dumps(response)}")
                return EvalResult(
                    question=question,
                    error="Invalid API response format: missing 'choices'"
                )
            
            choice = response["choices"][0]
            
            # Check if message field exists
            if "message" not in choice:
                # Handle LM Studio specific format which might be different
                if "text" in choice:
                    content = choice["text"].strip()
                else:
                    logger.error(f"Unexpected response format - no message or text field in choice: {json.dumps(choice)}")
                    return EvalResult(
                        question=question,
                        error="Invalid response format: missing both 'message' and 'text' fields"
                    )
            else:
                message = choice["message"]
                if "content" not in message:
                    logger.error(f"Unexpected message format - no content field: {json.dumps(message)}")
                    return EvalResult(
                        question=question,
                        error="Invalid message format: missing 'content' field"
                    )
                content = message["content"].strip()
            
            # Default judgment to None (will be populated by parsing)
            judgment = None
            confidence = None
            reasoning = None
            
            # Define regex patterns for extracting values from non-JSON responses (fallback)
            judgment_patterns = [
                re.compile(r"judgment.*?[=:]\s*(?P<judgment>true|false)", re.IGNORECASE),
                re.compile(r"answer.*?[=:]\s*(?P<judgment>yes|no)", re.IGNORECASE)
            ]
            confidence_patterns = [
                re.compile(r"confidence.*?[=:]\s*(?P<confidence>\d*\.?\d+)", re.IGNORECASE),
                re.compile(r"probability.*?[=:]\s*(?P<confidence>\d*\.?\d+)", re.IGNORECASE)
            ]
            
            # Try to parse JSON response - with improved parsing for LM Studio
            try:
                # First try to parse as JSON
                try:
                    # Log the original content for debugging
                    logger.debug(f"Attempting to parse JSON content: {content}")
                    
                    # Pre-sanitize common issues before even attempting to parse
                    # This helps avoid NameError exceptions by fixing Python-style literals
                    sanitized_content = (content.replace("True", "true")
                                          .replace("False", "false")
                                          .replace("None", "null")
                                          .replace("'", '"'))  # Replace single quotes with double quotes
                    
                    # Add more logging for debugging
                    logger.debug(f"Sanitized content: {sanitized_content}")
                    
                    # First try with the sanitized content since it's more likely to work
                    judgment_obj = None
                    parse_error = None
                    
                    try:
                        judgment_obj = json.loads(sanitized_content)
                        logger.debug("Successfully parsed sanitized content as JSON")
                    except Exception as sanitized_err:
                        parse_error = str(sanitized_err)
                        logger.warning(f"Failed to parse sanitized content: {parse_error}")
                        
                        # If sanitized content failed, fall back to trying original
                        try:
                            judgment_obj = json.loads(content)
                            logger.debug("Successfully parsed original content as JSON")
                        except Exception as original_err:
                            logger.warning(f"Failed to parse original content: {str(original_err)}")
                            
                            # Last resort: try with explicit safe eval for Python dict literals
                            try:
                                import ast
                                logger.debug("Attempting to parse as Python literal with ast.literal_eval")
                                # Try to eval it as a Python literal, then convert to proper JSON
                                python_dict = ast.literal_eval(content)
                                
                                # Convert Python dict to judgment_obj
                                judgment_obj = {}
                                if isinstance(python_dict, dict):
                                    # Convert all keys and values to proper JSON types
                                    for k, v in python_dict.items():
                                        if isinstance(v, bool):
                                            judgment_obj[k] = bool(v)  # Ensure it's JSON boolean
                                        elif v is None:
                                            judgment_obj[k] = None
                                        else:
                                            judgment_obj[k] = v
                                    logger.debug("Successfully parsed with ast.literal_eval")
                            except Exception as ast_err:
                                logger.warning(f"ast.literal_eval parsing failed: {str(ast_err)}")
                                logger.warning(f"All JSON parsing methods failed. Using regex fallback.")
                                # Re-raise to trigger the regex fallback
                                raise json.JSONDecodeError("Cannot parse content: " + parse_error, 
                                                         content, 0)
                    
                    # Check for judgment field with proper boolean handling
                    if isinstance(judgment_obj, dict) and "judgment" in judgment_obj:
                        # Extract confidence if provided
                        confidence = None
                        reasoning = None
                        
                        # Get confidence score from model if available
                        if "confidence" in judgment_obj:
                            try:
                                conf_val = judgment_obj["confidence"]
                                if isinstance(conf_val, (int, float)):
                                    confidence = min(max(float(conf_val), 0.0), 1.0)  # Clamp to 0-1
                                elif isinstance(conf_val, str):
                                    # Try to parse string as number
                                    confidence = min(max(float(conf_val), 0.0), 1.0)  # Clamp to 0-1
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid confidence value: {judgment_obj.get('confidence')}")
                        else:
                            # If confidence is missing, provide a default of 0.7
                            logger.warning("Confidence value missing from model response. Using default value of 0.7")
                            confidence = 0.7
                        
                        # Get reasoning if available and not omitted
                        if omit_reasoning:
                            # If reasoning is intentionally omitted, set to None
                            reasoning = None
                        elif "reasoning" in judgment_obj and isinstance(judgment_obj["reasoning"], str):
                            reasoning = judgment_obj["reasoning"]
                        else:
                            # If reasoning is missing but not intentionally omitted, provide a default explanation
                            logger.warning("Reasoning missing from model response. Using default reasoning")
                            reasoning = "No reasoning provided by evaluation model"
                        
                        try:
                            # Handle boolean judgment value directly
                            if isinstance(judgment_obj["judgment"], bool):
                                judgment = judgment_obj["judgment"]
                            # Handle string representations of boolean
                            elif isinstance(judgment_obj["judgment"], str):
                                judgment_str = judgment_obj["judgment"].lower()
                                if judgment_str in ["true", "yes", "1"]:
                                    judgment = True
                                elif judgment_str in ["false", "no", "0"]:
                                    judgment = False
                                else:
                                    judgment = bool(judgment_obj["judgment"])
                            # Handle numeric values (0 = False, anything else = True)
                            elif isinstance(judgment_obj["judgment"], (int, float)):
                                judgment = bool(judgment_obj["judgment"])
                            else:
                                judgment = False
                                
                            logger.info(f"Parsed judgment value: {judgment} from {judgment_obj['judgment']}")
                        except Exception as e:
                            logger.warning(f"Error parsing judgment value: {e}, using default True")
                            judgment = True
                    # If judgment field is missing but it's a simple boolean
                    elif isinstance(judgment_obj, bool):
                        judgment = judgment_obj
                    # If it's a simple number
                    elif isinstance(judgment_obj, (int, float)):
                        judgment = bool(judgment_obj)
                    else:
                        # Default to text analysis if JSON doesn't have expected structure
                        lower_content = content.lower()
                        judgment = ("yes" in lower_content or 
                                   "true" in lower_content or 
                                   "1" in lower_content or 
                                   "correct" in lower_content)
                except json.JSONDecodeError:
                    # Try to parse using regex patterns if JSON parsing fails
                    # First try to extract judgment
                    for pattern in judgment_patterns:
                        match = pattern.search(content)
                        if match:
                            judgment_str = match.group("judgment").lower()
                            if judgment_str in ["true", "yes", "1"]:
                                judgment = True
                            elif judgment_str in ["false", "no", "0"]:
                                judgment = False
                            break
                    
                    # Then try to extract confidence
                    for pattern in confidence_patterns:
                        match = pattern.search(content)
                        if match:
                            try:
                                conf_str = match.group("confidence")
                                confidence = min(max(float(conf_str), 0.0), 1.0)  # Clamp to 0-1
                                break
                            except ValueError:
                                pass
                    
                    # If patterns didn't match, fall back to simple content analysis
                    if judgment is None:
                        # Handle case when it's a direct response instead of JSON
                        if content.lower() in ["true", "yes", "1"]:
                            judgment = True
                        elif content.lower() in ["false", "no", "0"]:
                            judgment = False
                        else:
                            # If it's not parseable JSON or a direct boolean, analyze text
                            lower_content = content.lower()
                            judgment = ("yes" in lower_content or 
                                      "true" in lower_content or 
                                      "1" in lower_content or 
                                      "correct" in lower_content)
            except Exception as e:
                # Fallback to basic text analysis
                logger.warning(f"Error parsing judgment: {str(e)}")
                lower_content = content.lower()
                judgment = ("yes" in lower_content or 
                           "true" in lower_content or 
                           "1" in lower_content or 
                           "correct" in lower_content)
            
            # Use verbalized confidence directly from the model
            # This is more consistent across different providers and models
            
            # Log what we found 
            logger.info(f"Evaluation result: {judgment} (confidence: {confidence})")
            
            # Use the verbalized confidence directly
            final_confidence = confidence
            
            if final_confidence is not None:
                logger.info(f"Using verbalized confidence: {final_confidence:.4f}")
            else:
                logger.info("No confidence value provided by model")
            
            # Create the result object
            result = EvalResult(
                question=question,
                judgment=judgment,
                confidence=final_confidence,
                reasoning=reasoning
            )
            
            # Cache the result for future use
            API_RESPONSE_CACHE.put(cache_key, result)
            
            return result
        
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response for question '{question}': {str(e)}")
            logger.error(f"Response: {json.dumps(response)}")
            return EvalResult(
                question=question,
                error=f"Error parsing response: {str(e)}",
                confidence=None,
                reasoning=None
            )
    
    except Exception as e:
        logger.error(f"Error evaluating question '{question}': {str(e)}")
        return EvalResult(
            question=question,
            error=f"Error evaluating question: {str(e)}"
        )


def get_api_cache_stats() -> Dict[str, Any]:
    """Get statistics about the API response cache.
    
    Returns:
        Dictionary with statistics about the LRU cache for API responses
    """
    return API_RESPONSE_CACHE.get_stats()


async def run_evalset(
    evalset_id: str,
    conversation: List[Dict[str, str]],
    judge_model: str = "meta-llama-3.1-8b-instruct",
    max_parallel: int = 3,
    omit_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Run an EvalSet evaluation on a conversation.
    
    Args:
        evalset_id: ID of the EvalSet to use
        conversation: List of conversation messages
        judge_model: LLM model to use for evaluations (from server config)
        max_parallel: Maximum number of parallel evaluations
        omit_reasoning: If True, don't generate or include reasoning in results
    
    Returns:
        Dictionary with evaluation results
    """
    # Log the model being used for evaluation
    logger.info(f"Using judge model: {judge_model}")
        
    try:
        # Validate parameters
        validate_required_params({
            "evalset_id": evalset_id,
            "conversation": conversation
        }, ["evalset_id", "conversation"])
        
        # Validate max_parallel
        if not isinstance(max_parallel, int) or max_parallel <= 0:
            return format_error(f"max_parallel must be a positive integer, got {max_parallel}")
        
        # Get the EvalSet
        evalset = get_evalset(evalset_id)
        if not evalset:
            return format_error(f"EvalSet with ID '{evalset_id}' not found")
        
        # Validate conversation
        if not isinstance(conversation, list) or len(conversation) == 0:
            return format_error("Conversation must be a non-empty list of message objects")
        
        for msg in conversation:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return format_error("Each conversation message must be a dict with 'role' and 'content' fields")
        
        # Process all questions with rate limiting
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_question(question):
            async with semaphore:
                return await evaluate_question(
                    conversation=conversation,
                    question=question,
                    template=evalset.template,
                    judge_model=judge_model,
                    omit_reasoning=omit_reasoning
                )
        
        # Run all questions in parallel with rate limiting
        eval_results = await asyncio.gather(
            *[process_question(question) for question in evalset.questions]
        )
        
        # Create summary
        total_questions = len(eval_results)
        successful_evals = [r for r in eval_results if r.error is None]
        yes_count = sum(1 for r in successful_evals if r.judgment is True)
        no_count = sum(1 for r in successful_evals if r.judgment is False)
        error_count = sum(1 for r in eval_results if r.error is not None)
        
        if total_questions - error_count > 0:
            yes_percentage = (yes_count / (total_questions - error_count)) * 100
        else:
            yes_percentage = 0
        
        # Calculate mean confidence for valid results
        valid_confidences = [r.confidence for r in successful_evals if r.confidence is not None]
        mean_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else None
        
        # Calculate confidence statistics for Yes and No answers separately
        yes_confidences = [r.confidence for r in successful_evals if r.judgment is True and r.confidence is not None]
        no_confidences = [r.confidence for r in successful_evals if r.judgment is False and r.confidence is not None]
        
        mean_yes_confidence = sum(yes_confidences) / len(yes_confidences) if yes_confidences else None
        mean_no_confidence = sum(no_confidences) / len(no_confidences) if no_confidences else None
        
        summary = {
            "total_questions": total_questions,
            "successful_evaluations": len(successful_evals),
            "yes_count": yes_count,
            "no_count": no_count,
            "error_count": error_count,
            "yes_percentage": round(yes_percentage, 2),
            "mean_confidence": round(mean_confidence, 2) if mean_confidence is not None else None,
            "mean_yes_confidence": round(mean_yes_confidence, 2) if mean_yes_confidence is not None else None,
            "mean_no_confidence": round(mean_no_confidence, 2) if mean_no_confidence is not None else None
        }
        
        # Create results object
        results = EvalResults(
            evalset_id=evalset_id,
            evalset_name=evalset.name,
            results=eval_results,
            conversation=conversation,
            summary=summary
        )
        
        # Format response
        formatted_results = []
        formatted_results.append(f"# Evaluation Results for '{evalset.name}'")
        formatted_results.append("")
        formatted_results.append(f"## Summary")
        formatted_results.append(f"- Total questions: {total_questions}")
        formatted_results.append(f"- Success rate: {len(successful_evals)}/{total_questions} questions evaluated successfully")
        
        if error_count > 0:
            formatted_results.append(f"- Errors: {error_count} questions failed to evaluate")
            # Add more detailed help message if boolean errors
            boolean_errors = any("JSON boolean" in (r.error or "") for r in eval_results)
            if boolean_errors:
                formatted_results.append("  ⚠️ There were issues with JSON format. Try running again with different prompt.")
        
        formatted_results.append(f"- Yes responses: {yes_count} ({round(yes_percentage, 2)}%)")
        formatted_results.append(f"- No responses: {no_count} ({round(100 - yes_percentage, 2)}%)")
        
        # Add confidence information to summary if available
        if summary["mean_confidence"] is not None:
            formatted_results.append(f"- Mean confidence: {summary['mean_confidence']:.2f}")
            
            # Add yes/no confidence breakdowns if available
            if summary["mean_yes_confidence"] is not None and yes_count > 0:
                formatted_results.append(f"- Mean confidence in Yes responses: {summary['mean_yes_confidence']:.2f}")
            if summary["mean_no_confidence"] is not None and no_count > 0:
                formatted_results.append(f"- Mean confidence in No responses: {summary['mean_no_confidence']:.2f}")
        formatted_results.append("")
        
        formatted_results.append(f"## Detailed Results")
        for i, result in enumerate(eval_results, 1):
            if result.error:
                formatted_results.append(f"{i}. **Q**: {result.question}")
                formatted_results.append(f"   **Error**: {result.error}")
            else:
                judgment_text = "Yes" if result.judgment else "No"
                formatted_results.append(f"{i}. **Q**: {result.question}")
                # Format the display based on available confidence information
                # Show confidence if available
                confidence_display = ""
                
                # If we have a confidence value, display it
                if result.confidence is not None:
                    confidence_display = f"(confidence: {result.confidence:.2f})"
                else:
                    confidence_display = "(confidence: N/A)"
                
                # Format the result with confidence and optional reasoning
                formatted_results.append(f"   **A**: {judgment_text} {confidence_display}")
                
                # Include reasoning if available and not omitted
                if not omit_reasoning and result.reasoning is not None and result.reasoning:
                    # Limit reasoning to 200 chars and add ellipsis if needed
                    reasoning_text = result.reasoning[:200] + ("..." if len(result.reasoning) > 200 else "")
                    formatted_results.append(f"   **Reasoning**: {reasoning_text}")
        
        # Prepare results for response, removing reasoning field if omitted
        results_for_response = []
        for result in eval_results:
            result_dict = result.model_dump()
            # Remove reasoning field if omit_reasoning is true
            if omit_reasoning and 'reasoning' in result_dict:
                result_dict.pop('reasoning')
            results_for_response.append(result_dict)
            
        return {
            "status": "success",
            "id": str(uuid.uuid4()),
            "evalset_id": evalset_id,
            "evalset_name": evalset.name,
            "judge_model": judge_model,
            "results": results_for_response,
            "summary": summary,
            "formatted_message": "\n".join(formatted_results)
        }
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        logger.error(f"Error running EvalSet: {str(e)}")
        return format_error(f"Error running EvalSet: {str(e)}")
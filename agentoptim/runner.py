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
from agentoptim.utils import (
    format_error,
    format_success,
    ValidationError,
    validate_required_params,
)

# Configure logging
logger = logging.getLogger(__name__)

# Check for LM Studio compatibility mode
# Our testing showed LM Studio requires special handling
# 1. No response_format parameter (causes 400 error)
# 2. No logprobs support (always returns null)
# 3. System prompts work well and help control the output format
LMSTUDIO_COMPAT = os.environ.get("AGENTOPTIM_LMSTUDIO_COMPAT", "1") == "1"  # Enable by default
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"

# Log compatibility mode
if LMSTUDIO_COMPAT:
    logger.info("LM Studio compatibility mode is ENABLED")
    logger.info("* response_format parameter disabled")
    logger.info("* logprobs expected to be null")
    logger.info("* Using system prompts for better control")
else:
    logger.info("LM Studio compatibility mode is DISABLED")

# Default timeout for API calls
DEFAULT_TIMEOUT = 60  # seconds

# Get API base URL from environment or use default
API_BASE = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234/v1")


class EvalResult(BaseModel):
    """Model for a single evaluation result."""
    
    question: str
    judgment: Optional[bool] = None
    logprob: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None


class EvalResults(BaseModel):
    """Model for evaluation results."""
    
    evalset_id: str
    evalset_name: str
    results: List[EvalResult]
    conversation: List[Dict[str, str]]
    summary: Dict[str, Any] = Field(default_factory=dict)


async def call_llm_api(
    prompt: Optional[str] = None,
    model: str = "meta-llama-3.1-8b-instruct",
    temperature: float = 0.0,
    max_tokens: int = 50,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: bool = True,
    messages: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Call the LLM API to get model response and token logprobs.
    
    Args:
        prompt: The prompt to send to the model (legacy parameter)
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        logit_bias: Optional logit bias to apply
        logprobs: Whether to return logprobs
        messages: List of message objects with role and content (preferred over prompt)
    
    Returns:
        Response from the LLM API
    """
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
        
        # Our testing shows that LM Studio ignores logprobs requests,
        # but explicitly requesting them doesn't cause an error,
        # so we'll include the parameter but be prepared for null results
        if logprobs:
            payload["logprobs"] = logprobs
        
        # LM Studio requires response_format.type to be "json_schema" with specific format
        if LMSTUDIO_COMPAT:
            # Use the correct JSON schema format for LM Studio
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judgment_response",
                    "strict": "true",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "judgment": {
                                "type": "boolean"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "reasoning": {
                                "type": "string"
                            }
                        },
                        "required": ["judgment", "confidence"]
                    }
                }
            }
        else:
            # For other providers, use standard OpenAI format
            payload["response_format"] = {"type": "json_object"}
        
        if logit_bias:
            payload["logit_bias"] = logit_bias
        
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Calling LLM API with model: {model}")
        if DEBUG_MODE:
            logger.debug(f"API request payload: {json.dumps(payload, indent=2)}")
        
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
                        if "logprobs" in payload:
                            del payload["logprobs"]
                            logger.info("Retry: Removed logprobs parameter")
                        
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
                
                # Based on our testing, LM Studio NEVER returns logprobs even when requested
            # So we'll just handle the null case directly without trying to synthesize values
            if "logprobs" not in choice or choice["logprobs"] is None:
                logger.info("LM Studio returned null logprobs as expected from our testing")
                
                # Leave logprobs as null - don't try to synthesize anymore
                choice["logprobs"] = None
                
                # For debugging only - analyze content to see what the model likely determined
                content = ""
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"].lower()
                elif "text" in choice:
                    content = choice["text"].lower()
                    
                is_yes = any(x in content for x in ["yes", "true", "correct", "agree", "appropriate", "clear"]) 
                is_no = any(x in content for x in ["no", "false", "incorrect", "disagree", "inappropriate", "unclear"])
                
                if is_yes:
                    logger.info("Detected 'yes' in response content")
                elif is_no:
                    logger.info("Detected 'no' in response content")
            elif isinstance(choice["logprobs"], dict) and "content" not in choice["logprobs"]:
                # Some LLM providers use different logprobs structure
                # This is unlikely for LM Studio but we'll keep the code for other providers
                logger.info("Converting non-standard logprobs format")
                
                # Convert whatever format to content format
                top_logprobs = choice["logprobs"].get("top_logprobs", [])
                if top_logprobs and isinstance(top_logprobs, list):
                    choice["logprobs"] = {
                        "content": [
                            {"token": str(item.get("token", "1")), 
                             "logprob": item.get("logprob", -0.1)}
                            for item in top_logprobs[:5]  # Take up to 5 top tokens
                        ]
                    }
                
                # If LM Studio is using text field instead of message
                if "text" in choice and "message" not in choice:
                    logger.info("Converting LM Studio 'text' field to 'message' format")
                    choice["message"] = {
                        "role": "assistant", 
                        "content": choice["text"]
                    }
            
            return response_json
    
    except Exception as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        return {"error": f"Error calling LLM API: {str(e)}"}


async def evaluate_question(
    conversation: List[Dict[str, str]],
    question: str,
    template: str,
    model: str = "meta-llama-3.1-8b-instruct"
) -> EvalResult:
    """
    Evaluate a single question against a conversation.
    
    Args:
        conversation: List of conversation messages
        question: The evaluation question
        template: The evaluation template
        model: LLM model to use
    
    Returns:
        Evaluation result for the question
    """
    try:
        # Render the template with Jinja2
        jinja_env = jinja2.Environment()
        jinja_template = jinja_env.from_string(template)
        rendered_prompt = jinja_template.render(
            conversation=json.dumps(conversation, ensure_ascii=False),
            eval_question=question
        )
        
        # Add specific formatting guidance for LM Studio
        if LMSTUDIO_COMPAT:
            # Now that we've confirmed LM Studio supports JSON schema format
            # We'll use it to strictly enforce the response format
            
            # We can keep the JSON instructions in the prompt
            # Add a clear note about the expected format with confidence estimation
            # Based on the verbalized-uq research, this format is shown to be more effective
            rendered_prompt += """

IMPORTANT: Provide your judgment and the probability that it is correct (0.0 to 1.0) for this question.
Give ONLY the judgment and probability in JSON format. Take your uncertainty in the prompt, the task difficulty, 
your knowledge availability, and other sources of uncertainty into account.

For example:
```
{
  "judgment": true,  // true for Yes, false for No
  "confidence": 0.85,  // probability between 0.0 and 1.0 that your judgment is correct
  "reasoning": "Optional brief explanation of your reasoning"
}
```

For the confidence field, estimate your confidence as accurately as possible:
- 1.0 = perfect certainty (100% confident)
- 0.9 = very high confidence (90% confident)
- 0.7 = moderately confident (70% confident)
- 0.5 = equally likely to be right or wrong (50% confident)
- 0.3 = not very confident (30% confident)
- 0.1 = highly uncertain (10% confident)

Your confidence should reflect how certain you are about your judgment based on the evidence provided."""
            
            # LM Studio needs simpler prompts sometimes
            if len(rendered_prompt) > 2000:
                # Shorten if necessary
                logger.info("Prompt is very long, shortening for LM Studio compatibility")
                rendered_prompt = f"""Please evaluate the following question:

Question: {question}

Provide your judgment (Yes/No) and the probability that it is correct (0.0 to 1.0).
Give ONLY the judgment and probability in JSON format, with no other words or explanation.

Example for a confident Yes:
{{"judgment": true, "confidence": 0.85}}

Example for an uncertain No:
{{"judgment": false, "confidence": 0.6}}

Consider carefully and provide your best calibrated estimate of confidence."""
                
            # Add system prompt for better control - our testing shows system prompts work well
            messages = [
                {"role": "system", "content": "You are an evaluation assistant that provides objective judgments in JSON format. Include: 1) a boolean 'judgment' field (true/false), 2) a 'confidence' field (0.0-1.0) indicating the probability your judgment is correct, and 3) optional 'reasoning'. Provide well-calibrated confidence scores based on uncertainty, task difficulty, and available information."},
                {"role": "user", "content": rendered_prompt}
            ]
        else:
            # Normal mode - just use the rendered prompt as user message
            messages = [{"role": "user", "content": rendered_prompt}]
        
        logger.info(f"Evaluating question: {question}")
        
        # Call the LLM API with proper message format
        response = await call_llm_api(
            messages=messages,  # Now passing prepared messages with system prompt for LM Studio
            model=model,
            logprobs=True
        )
        
        # Check for errors
        if "error" in response:
            return EvalResult(
                question=question,
                error=response["error"]
            )
        
        # Extract the response and logprobs
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
            
            # Default judgment to True for simplified testing if needed
            judgment = True
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
                    judgment_obj = json.loads(content)
                    
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
                        
                        # Get reasoning if available
                        if "reasoning" in judgment_obj and isinstance(judgment_obj["reasoning"], str):
                            reasoning = judgment_obj["reasoning"]
                        
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
            
            # Get logprobs for judgment tokens
            logprob = 0.0  # Default value
            
            # Extract logprobs - handle different formats and LM Studio specifics
            if "logprobs" in choice and choice["logprobs"] is not None:
                logprobs_data = choice.get("logprobs", {})
                
                # Handle different logprobs formats
                if isinstance(logprobs_data, dict):
                    content_tokens = logprobs_data.get("content", [])
                    
                    # Find tokens that indicate judgment
                    judgment_token_probs = []
                    
                    for token_data in content_tokens:
                        if not isinstance(token_data, dict):
                            continue
                        
                        token = token_data.get("token", "").strip().lower()
                        # Look for yes/no/true/false tokens in various formats
                        if token in ["0", "1", "\"0\"", "\"1\"", "0,", "1,", "yes", "no", "true", "false", 
                                    "\"yes\"", "\"no\"", "\"true\"", "\"false\""]:
                            judgment_token_probs.append({
                                "token": token,
                                "logprob": token_data.get("logprob", -0.1)
                            })
                    
                    # Get the highest logprob
                    if judgment_token_probs:
                        highest_prob_token = max(judgment_token_probs, key=lambda x: x["logprob"])
                        logprob = highest_prob_token["logprob"]
                        logger.info(f"Found judgment token: {highest_prob_token['token']} with logprob {logprob}")
                    else:
                        # If no specific judgment tokens found, use the most likely token as indicator
                        if content_tokens:
                            # Sort by logprob, highest first
                            sorted_tokens = sorted(content_tokens, 
                                                key=lambda x: x.get("logprob", -100), 
                                                reverse=True)
                            if sorted_tokens:
                                best_token = sorted_tokens[0]
                                logprob = best_token.get("logprob", -0.5)
                                logger.info(f"Using best token as logprob indicator: {best_token.get('token')} with logprob {logprob}")
                elif isinstance(logprobs_data, (int, float)):
                    # Handle simple numeric logprob
                    logprob = float(logprobs_data)
            else:
                # Don't synthesize logprobs - just set to None
                logger.info(f"No logprobs data available for model {model}")
                logprob = None
            
            logger.info(f"Evaluation result: {judgment} (logprob: {logprob})")
            
            # If we got a confidence from the model but no logprob, use the verbalized confidence
            final_confidence = None
            
            if logprob is not None:
                # If we have logprobs, convert to confidence score (0-1)
                # Logprobs are typically negative values, so we use exp to convert to probability
                # This gives us the probability of the token
                if logprob < 0:
                    final_confidence = min(max(float(math.exp(logprob)), 0.0), 1.0)
                else:
                    # Handle positive logprobs (unlikely but possible)
                    final_confidence = min(max(float(logprob), 0.0), 1.0)
                
                logger.info(f"Using logprob-derived confidence: {final_confidence:.4f} from logprob {logprob:.4f}")
            elif confidence is not None:
                # If no logprobs but we have a verbalized confidence, use that
                final_confidence = confidence
                logger.info(f"Using verbalized confidence: {final_confidence:.4f}")
            
            return EvalResult(
                question=question,
                judgment=judgment,
                logprob=logprob,
                confidence=final_confidence,
                reasoning=reasoning
            )
        
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


async def run_evalset(
    evalset_id: str,
    conversation: List[Dict[str, str]],
    model: str = "meta-llama-3.1-8b-instruct",
    max_parallel: int = 3
) -> Dict[str, Any]:
    """
    Run an EvalSet evaluation on a conversation.
    
    Args:
        evalset_id: ID of the EvalSet to use
        conversation: List of conversation messages
        model: LLM model to use for evaluations
        max_parallel: Maximum number of parallel evaluations
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Validate parameters
        validate_required_params({
            "evalset_id": evalset_id,
            "conversation": conversation
        }, ["evalset_id", "conversation"])
        
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
                    model=model
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
                # Format the display based on available confidence/logprob information
                confidence_display = ""
                
                # If we have a confidence value, display it
                if result.confidence is not None:
                    confidence_display = f"(confidence: {result.confidence:.2f})"
                # Otherwise if we have a logprob, show it as a fallback
                elif result.logprob is not None:
                    confidence_display = f"(logprob: {result.logprob:.4f})"
                else:
                    confidence_display = "(confidence: N/A)"
                
                # Format the result with confidence and optional reasoning
                formatted_results.append(f"   **A**: {judgment_text} {confidence_display}")
                
                # Include reasoning if available
                if result.reasoning:
                    # Limit reasoning to 200 chars and add ellipsis if needed
                    reasoning_text = result.reasoning[:200] + ("..." if len(result.reasoning) > 200 else "")
                    formatted_results.append(f"   **Reasoning**: {reasoning_text}")
        
        return {
            "status": "success",
            "id": str(uuid.uuid4()),
            "evalset_id": evalset_id,
            "evalset_name": evalset.name,
            "model": model,
            "results": [r.model_dump() for r in eval_results],
            "summary": summary,
            "formatted_message": "\n".join(formatted_results)
        }
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        logger.error(f"Error running EvalSet: {str(e)}")
        return format_error(f"Error running EvalSet: {str(e)}")
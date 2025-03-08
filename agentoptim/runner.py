"""Runner functionality for evaluating conversations with EvalSets."""

import os
import json
import httpx
import asyncio
import logging
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

# Default timeout for API calls
DEFAULT_TIMEOUT = 60  # seconds

# Get API base URL from environment or use default
API_BASE = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234/v1")


class EvalResult(BaseModel):
    """Model for a single evaluation result."""
    
    question: str
    judgment: Optional[bool] = None
    logprob: Optional[float] = None
    error: Optional[str] = None


class EvalResults(BaseModel):
    """Model for evaluation results."""
    
    evalset_id: str
    evalset_name: str
    results: List[EvalResult]
    conversation: List[Dict[str, str]]
    summary: Dict[str, Any] = Field(default_factory=dict)


async def call_llm_api(
    prompt: str,
    model: str = "meta-llama-3.1-8b-instruct",
    temperature: float = 0.0,
    max_tokens: int = 50,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: bool = True
) -> Dict[str, Any]:
    """
    Call the LLM API to get model response and token logprobs.
    
    Args:
        prompt: The prompt to send to the model
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        logit_bias: Optional logit bias to apply
        logprobs: Whether to return logprobs
    
    Returns:
        Response from the LLM API
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        }
        
        if logit_bias:
            payload["logit_bias"] = logit_bias
        
        headers = {"Content-Type": "application/json"}
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE}/chat/completions", 
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code}, {response.text}")
                return {
                    "error": f"LLM API error: {response.status_code}",
                    "details": response.text
                }
            
            return response.json()
    
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
        
        # Call the LLM API
        response = await call_llm_api(
            prompt=rendered_prompt,
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
            content = response["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON response
            try:
                judgment_obj = json.loads(content)
                judgment = bool(judgment_obj.get("judgment", 0))
            except json.JSONDecodeError:
                # If not JSON, look for judgment values in text
                judgment = "1" in content or "yes" in content.lower()
            
            # Get logprobs for judgment tokens (0 and 1)
            # Since different models' tokenizers will differ, we look for probable tokens
            logprobs_data = response["choices"][0].get("logprobs", {})
            content_tokens = logprobs_data.get("content", [])
            
            # Find the token with highest logprob for judgment values
            judgment_token_probs = []
            
            for token_data in content_tokens:
                token = token_data.get("token", "").strip().lower()
                if token in ["0", "1", "\"0\"", "\"1\"", "0,", "1,", "yes", "no", "true", "false"]:
                    judgment_token_probs.append({
                        "token": token,
                        "logprob": token_data.get("logprob", -100)
                    })
            
            # Get the highest logprob
            if judgment_token_probs:
                highest_prob_token = max(judgment_token_probs, key=lambda x: x["logprob"])
                logprob = highest_prob_token["logprob"]
            else:
                logprob = None
            
            return EvalResult(
                question=question,
                judgment=judgment,
                logprob=logprob
            )
        
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return EvalResult(
                question=question,
                error=f"Error parsing response: {str(e)}"
            )
    
    except Exception as e:
        logger.error(f"Error evaluating question: {str(e)}")
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
        
        summary = {
            "total_questions": total_questions,
            "successful_evaluations": len(successful_evals),
            "yes_count": yes_count,
            "no_count": no_count,
            "error_count": error_count,
            "yes_percentage": round(yes_percentage, 2)
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
        formatted_results.append("")
        
        formatted_results.append(f"## Detailed Results")
        for i, result in enumerate(eval_results, 1):
            if result.error:
                formatted_results.append(f"{i}. **Q**: {result.question}")
                formatted_results.append(f"   **Error**: {result.error}")
            else:
                judgment_text = "Yes" if result.judgment else "No"
                formatted_results.append(f"{i}. **Q**: {result.question}")
                formatted_results.append(f"   **A**: {judgment_text} (logprob: {result.logprob:.4f})")
        
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
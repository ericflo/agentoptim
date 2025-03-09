"""Evaluation run functionality for storing and retrieving evaluation results.

This module provides functions for managing evaluation run results, including:
- Running new evaluations
- Storing evaluation results
- Retrieving past evaluation results
- Listing evaluation runs with pagination
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from agentoptim.constants import MAX_EVAL_RUNS
from agentoptim.utils import (
    DATA_DIR,
    generate_id,
    save_json,
    load_json,
    list_json_files,
    validate_action,
    validate_required_params,
    format_error,
    format_success,
    format_list,
    ValidationError,
)
from agentoptim.cache import LRUCache

# Directory for storing EvalRuns
EVAL_RUNS_DIR = os.path.join(DATA_DIR, "eval_runs")
os.makedirs(EVAL_RUNS_DIR, exist_ok=True)

# Create LRU cache for eval runs
eval_run_cache = LRUCache(capacity=50, ttl=1800)  # 30 minute TTL for EvalRuns


class EvalRun(BaseModel):
    """Model for an evaluation run result."""
    
    id: str = Field(default_factory=generate_id)
    evalset_id: str
    evalset_name: str
    timestamp: float = Field(default_factory=time.time)
    judge_model: Optional[str] = None
    results: List[Dict[str, Any]]
    conversation: List[Dict[str, str]]
    summary: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalRun":
        """Create from dictionary."""
        return cls(**data)
    
    def format_timestamp(self) -> str:
        """Format the timestamp as a human-readable string."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_eval_run_path(eval_run_id: str) -> str:
    """Get the file path for an EvalRun."""
    return os.path.join(EVAL_RUNS_DIR, f"{eval_run_id}.json")


def list_eval_runs(
    page: int = 1, 
    page_size: int = 10,
    evalset_id: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    List available evaluation runs with pagination support.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        evalset_id: Optional filter by EvalSet ID
        
    Returns:
        Tuple containing (list of eval runs for the page, total count of eval runs)
    """
    # Ensure valid pagination parameters
    page = max(1, page)
    page_size = max(1, min(100, page_size))  # Cap page size between 1-100
    
    # Get all evaluation run IDs
    all_run_ids = list_json_files(EVAL_RUNS_DIR)
    total_runs = len(all_run_ids)
    
    # Sort run IDs by creation time (most recent first)
    # This uses the file modification time as a proxy for creation time
    run_ids_with_mtime = []
    for run_id in all_run_ids:
        file_path = get_eval_run_path(run_id)
        try:
            mtime = os.path.getmtime(file_path)
            run_ids_with_mtime.append((run_id, mtime))
        except OSError:
            # Skip files that can't be accessed
            continue
    
    # Sort by modification time, newest first
    sorted_run_ids = [run_id for run_id, _ in sorted(run_ids_with_mtime, key=lambda x: x[1], reverse=True)]
    
    # Calculate pagination indices
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_run_ids = sorted_run_ids[start_idx:end_idx]
    
    # Load the eval runs for this page
    eval_runs = []
    for run_id in page_run_ids:
        eval_run_dict = get_eval_run_summary(run_id)
        if eval_run_dict:
            # Apply evalset_id filter if provided
            if evalset_id is None or eval_run_dict.get("evalset_id") == evalset_id:
                eval_runs.append(eval_run_dict)
    
    # For filtered results, recalculate total count
    if evalset_id:
        filtered_count = 0
        for run_id in sorted_run_ids:
            eval_run_dict = get_eval_run_summary(run_id)
            if eval_run_dict and eval_run_dict.get("evalset_id") == evalset_id:
                filtered_count += 1
        total_runs = filtered_count
        
    return eval_runs, total_runs


def get_eval_run_summary(eval_run_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a summary of an evaluation run without loading all the results.
    
    Args:
        eval_run_id: ID of the evaluation run
        
    Returns:
        Dictionary with evaluation run summary info, or None if not found
    """
    # Check cache first
    cached_run = eval_run_cache.get(eval_run_id)
    if cached_run is not None:
        # Return summary information only
        return {
            "id": cached_run.id,
            "evalset_id": cached_run.evalset_id,
            "evalset_name": cached_run.evalset_name,
            "timestamp": cached_run.timestamp,
            "timestamp_formatted": cached_run.format_timestamp(),
            "judge_model": cached_run.judge_model,
            "summary": cached_run.summary,
            "conversation_length": len(cached_run.conversation),
            "result_count": len(cached_run.results)
        }
    
    # Not in cache, try to load from disk
    file_path = get_eval_run_path(eval_run_id)
    data = load_json(file_path)
    if not data:
        return None
    
    # Return summary information
    return {
        "id": data.get("id"),
        "evalset_id": data.get("evalset_id"),
        "evalset_name": data.get("evalset_name"),
        "timestamp": data.get("timestamp"),
        "timestamp_formatted": datetime.fromtimestamp(data.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": data.get("judge_model"),
        "summary": data.get("summary", {}),
        "conversation_length": len(data.get("conversation", [])),
        "result_count": len(data.get("results", []))
    }


def get_eval_run(eval_run_id: str) -> Optional[EvalRun]:
    """
    Get a specific evaluation run by ID.
    
    First checks the LRU cache for the EvalRun, and if not found,
    loads it from disk and caches it for future use.
    
    Args:
        eval_run_id: ID of the evaluation run
        
    Returns:
        EvalRun object if found, None otherwise
    """
    # Try to get from cache first
    cached_eval_run = eval_run_cache.get(eval_run_id)
    if cached_eval_run is not None:
        return cached_eval_run
    
    # Not in cache, load from disk
    file_path = get_eval_run_path(eval_run_id)
    data = load_json(file_path)
    if not data:
        return None
    
    try:
        eval_run = EvalRun.from_dict(data)
        # Cache for future use
        eval_run_cache.put(eval_run_id, eval_run)
        return eval_run
    except Exception as e:
        # Log the error but don't crash
        print(f"Error loading EvalRun {eval_run_id}: {str(e)}")
        return None


def save_eval_run(eval_run: EvalRun) -> bool:
    """
    Save an evaluation run to disk and cache.
    
    Args:
        eval_run: The EvalRun object to save
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Save to disk
        file_path = get_eval_run_path(eval_run.id)
        save_json(eval_run.to_dict(), file_path)
        
        # Cache for future use
        eval_run_cache.put(eval_run.id, eval_run)
        
        # Check if we're exceeding the maximum number of eval runs
        cleanup_old_eval_runs()
        
        return True
    except Exception as e:
        # Log the error but don't crash
        print(f"Error saving EvalRun {eval_run.id}: {str(e)}")
        return False


def delete_eval_run(eval_run_id: str) -> bool:
    """
    Delete an evaluation run from disk and cache.
    
    Args:
        eval_run_id: ID of the evaluation run to delete
        
    Returns:
        True if deleted successfully, False if not found
    """
    file_path = get_eval_run_path(eval_run_id)
    if os.path.exists(file_path):
        try:
            # Remove from disk
            os.remove(file_path)
            
            # Remove from cache
            eval_run_cache.remove(eval_run_id)
            
            return True
        except Exception as e:
            # Log the error but don't crash
            print(f"Error deleting EvalRun {eval_run_id}: {str(e)}")
            return False
    return False


def cleanup_old_eval_runs() -> int:
    """
    Remove oldest eval runs if the total count exceeds MAX_EVAL_RUNS.
    
    Returns:
        Number of eval runs removed
    """
    run_ids = list_json_files(EVAL_RUNS_DIR)
    if len(run_ids) <= MAX_EVAL_RUNS:
        return 0  # Nothing to do
    
    # Get file modification times
    run_ids_with_mtime = []
    for run_id in run_ids:
        file_path = get_eval_run_path(run_id)
        try:
            mtime = os.path.getmtime(file_path)
            run_ids_with_mtime.append((run_id, mtime))
        except OSError:
            # Skip files that can't be accessed
            continue
    
    # Sort by modification time, oldest first
    sorted_run_ids = [run_id for run_id, _ in sorted(run_ids_with_mtime, key=lambda x: x[1])]
    
    # Calculate how many to remove
    to_remove = len(sorted_run_ids) - MAX_EVAL_RUNS
    if to_remove <= 0:
        return 0
    
    # Remove oldest runs
    removed_count = 0
    for i in range(to_remove):
        if i < len(sorted_run_ids):
            run_id = sorted_run_ids[i]
            if delete_eval_run(run_id):
                removed_count += 1
    
    return removed_count


def get_eval_runs_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the EvalRun cache performance.
    
    Returns:
        A dictionary with cache statistics including hit rate
    """
    return eval_run_cache.get_stats()


def get_formatted_eval_run(eval_run: EvalRun) -> Dict[str, Any]:
    """
    Create a nicely formatted representation of an evaluation run.
    
    Args:
        eval_run: The EvalRun object to format
        
    Returns:
        Dictionary with formatted evaluation results
    """
    # Format the timestamp
    timestamp_str = datetime.fromtimestamp(eval_run.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # Create formatted message
    formatted_results = []
    formatted_results.append(f"# Evaluation Results: {eval_run.evalset_name}")
    formatted_results.append(f"Run ID: {eval_run.id}")
    formatted_results.append(f"Time: {timestamp_str}")
    formatted_results.append(f"Judge Model: {eval_run.judge_model or 'auto-detected'}")
    formatted_results.append("")
    
    # Format summary section
    formatted_results.append(f"## Summary")
    summary = eval_run.summary
    
    if summary:
        total_questions = summary.get("total_questions", 0)
        successful_evals = summary.get("successful_evaluations", 0)
        yes_count = summary.get("yes_count", 0)
        no_count = summary.get("no_count", 0)
        error_count = summary.get("error_count", 0)
        yes_percentage = summary.get("yes_percentage", 0)
        
        formatted_results.append(f"- Total questions: {total_questions}")
        formatted_results.append(f"- Success rate: {successful_evals}/{total_questions} questions evaluated successfully")
        
        if error_count > 0:
            formatted_results.append(f"- Errors: {error_count} questions failed to evaluate")
        
        formatted_results.append(f"- Yes responses: {yes_count} ({round(yes_percentage, 2)}%)")
        formatted_results.append(f"- No responses: {no_count} ({round(100 - yes_percentage, 2)}%)")
        
        # Add confidence information to summary if available
        if "mean_confidence" in summary and summary["mean_confidence"] is not None:
            formatted_results.append(f"- Mean confidence: {summary['mean_confidence']:.2f}")
            
            # Add yes/no confidence breakdowns if available
            if "mean_yes_confidence" in summary and summary["mean_yes_confidence"] is not None and yes_count > 0:
                formatted_results.append(f"- Mean confidence in Yes responses: {summary['mean_yes_confidence']:.2f}")
            if "mean_no_confidence" in summary and summary["mean_no_confidence"] is not None and no_count > 0:
                formatted_results.append(f"- Mean confidence in No responses: {summary['mean_no_confidence']:.2f}")
    
    formatted_results.append("")
    
    # Format detailed results section
    formatted_results.append(f"## Detailed Results")
    for i, result in enumerate(eval_run.results, 1):
        if "error" in result and result["error"]:
            formatted_results.append(f"{i}. **Q**: {result.get('question', 'Unknown question')}")
            formatted_results.append(f"   **Error**: {result['error']}")
        else:
            judgment = result.get("judgment")
            judgment_text = "Yes" if judgment else "No"
            formatted_results.append(f"{i}. **Q**: {result.get('question', 'Unknown question')}")
            
            # Format the display based on available confidence information
            confidence_display = ""
            confidence = result.get("confidence")
            
            # If we have a confidence value, display it
            if confidence is not None:
                confidence_display = f"(confidence: {confidence:.2f})"
            else:
                confidence_display = "(confidence: N/A)"
            
            # Format the result with confidence and optional reasoning
            formatted_results.append(f"   **A**: {judgment_text} {confidence_display}")
            
            # Include reasoning if available
            reasoning = result.get("reasoning")
            if reasoning:
                # Limit reasoning to 200 chars and add ellipsis if needed
                reasoning_text = reasoning[:200] + ("..." if len(reasoning) > 200 else "")
                formatted_results.append(f"   **Reasoning**: {reasoning_text}")
    
    # Add conversation section
    formatted_results.append("")
    formatted_results.append(f"## Conversation")
    formatted_results.append("```")
    for message in eval_run.conversation:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "")
        
        # For very long messages, truncate with ellipsis
        if len(content) > 300:
            content = content[:300] + "..."
            
        formatted_results.append(f"{role}: {content}")
        formatted_results.append("")  # Empty line between messages
    formatted_results.append("```")
    
    # Return both the raw EvalRun data and the formatted message
    return {
        "id": eval_run.id,
        "evalset_id": eval_run.evalset_id,
        "evalset_name": eval_run.evalset_name,
        "timestamp": eval_run.timestamp,
        "timestamp_formatted": timestamp_str,
        "judge_model": eval_run.judge_model,
        "results": eval_run.results,
        "summary": eval_run.summary,
        "formatted_message": "\n".join(formatted_results)
    }


def manage_eval_runs(
    action: str,
    evalset_id: Optional[str] = None,
    conversation: Optional[List[Dict[str, str]]] = None,
    judge_model: Optional[str] = None,
    max_parallel: int = 3,
    omit_reasoning: bool = False,
    eval_run_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 10
) -> Dict[str, Any]:
    """
    Manage evaluation runs: run evaluations, retrieve past results, and list runs.
    
    Args:
        action: One of "run", "get", "list"
        evalset_id: ID of the EvalSet to use (required for "run", optional filter for "list")
        conversation: Conversation to evaluate (required for "run")
        judge_model: Optional LLM model to use for evaluations
        max_parallel: Maximum number of parallel evaluations (for "run")
        omit_reasoning: If True, don't generate or include reasoning in results
        eval_run_id: ID of a specific evaluation run (required for "get")
        page: Page number for list pagination (1-indexed)
        page_size: Number of items per page (10-100)
    
    Returns:
        Dictionary containing the result of the operation
    """
    from agentoptim.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
    
    valid_actions = ["run", "get", "list"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle pagination parameters
        page = max(1, page)  # Minimum page is 1
        page_size = max(1, min(MAX_PAGE_SIZE, page_size))  # Ensure page_size is within limits
        
        if action == "list":
            # List evaluation runs with pagination
            eval_runs, total_count = list_eval_runs(
                page=page, 
                page_size=page_size,
                evalset_id=evalset_id
            )
            
            # Calculate pagination metadata
            total_pages = (total_count + page_size - 1) // page_size
            has_next = page < total_pages
            has_prev = page > 1
            
            # Create a formatted message for human-readable output
            formatted_message = []
            filtered_text = f" for EvalSet '{evalset_id}'" if evalset_id else ""
            formatted_message.append(f"# Evaluation Runs{filtered_text}")
            formatted_message.append(f"Page {page} of {total_pages} ({total_count} total)")
            formatted_message.append("")
            
            if eval_runs:
                for i, run in enumerate(eval_runs):
                    formatted_message.append(f"## {i+1}. Run: {run['id']}")
                    formatted_message.append(f"- EvalSet: {run['evalset_name']} ({run['evalset_id']})")
                    formatted_message.append(f"- Time: {run['timestamp_formatted']}")
                    formatted_message.append(f"- Judge Model: {run['judge_model'] or 'auto-detected'}")
                    
                    # Add summary if available
                    if "summary" in run and run["summary"]:
                        summary = run["summary"]
                        yes_percentage = summary.get("yes_percentage", 0)
                        total_questions = summary.get("total_questions", 0)
                        formatted_message.append(f"- Score: {yes_percentage}% ({summary.get('yes_count', 0)}/{total_questions})")
                    
                    # Add separator for readability
                    formatted_message.append("")
            else:
                formatted_message.append("No evaluation runs found.")
            
            # Return the runs list with pagination metadata
            return {
                "status": "success",
                "eval_runs": eval_runs,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_next": has_next,
                    "has_prev": has_prev,
                    "next_page": page + 1 if has_next else None,
                    "prev_page": page - 1 if has_prev else None
                },
                "formatted_message": "\n".join(formatted_message)
            }
            
        elif action == "get":
            # Validate required parameters
            validate_required_params({"eval_run_id": eval_run_id}, ["eval_run_id"])
            
            # Get the evaluation run
            eval_run = get_eval_run(eval_run_id)
            if not eval_run:
                return format_error(f"Evaluation run with ID '{eval_run_id}' not found")
            
            # Format the evaluation run for output
            formatted_run = get_formatted_eval_run(eval_run)
            
            return {
                "status": "success",
                "eval_run": formatted_run,
                "formatted_message": formatted_run["formatted_message"]
            }
            
        elif action == "run":
            # This will be implemented by importing run_evalset from runner
            # and saving the results. We'll implement this in the server.py module
            # to avoid circular imports between runner.py and evalrun.py
            
            # In server.py, we'll:
            # 1. Call run_evalset from runner
            # 2. Create an EvalRun object from the results
            # 3. Save it using save_eval_run
            # 4. Return the results with both id and formatted message
            
            # For now, return a not implemented error since this will be handled in server.py
            return format_error("The 'run' action is implemented in server.py to avoid circular imports")
            
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")
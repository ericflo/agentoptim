"""EvalSet functionality for AgentOptim."""

import os
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from agentoptim.constants import MAX_QUESTIONS_PER_EVALSET, MAX_EVALSETS
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
from agentoptim.cache import (
    cache_evalset,
    get_cached_evalset,
    invalidate_evalset,
    get_evalset_cache_stats,
)

# Directory for storing EvalSets
EVALSETS_DIR = os.path.join(DATA_DIR, "evalsets")
os.makedirs(EVALSETS_DIR, exist_ok=True)

# Default template used for all evaluations
DEFAULT_TEMPLATE = """
Given this conversation:
{{ conversation }}

Please answer the following yes/no question about the final assistant response:
{{ eval_question }}

IMPORTANT INSTRUCTIONS FOR EVALUATION:

Analyze the conversation thoroughly before answering. Respond ONLY in valid JSON format with these THREE required fields:

1. "reasoning": Provide a detailed explanation (3-5 sentences) with specific evidence from the conversation
2. "judgment": Boolean true for Yes, false for No (must use JSON boolean literals: true/false)
3. "confidence": Number between 0.0 and 1.0 indicating your certainty

Example format:
```json
{
  "reasoning": "The assistant's response directly addresses the user's question by providing specific instructions. The information is clear, accurate, and would enable the user to accomplish their task without further assistance.",
  "judgment": true,
  "confidence": 0.92
}
```
"""


class EvalSet(BaseModel):
    """Model for an EvalSet."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    template: str = Field(default=DEFAULT_TEMPLATE)  # Use the system default template
    questions: List[str] = Field(default_factory=list)
    short_description: Optional[str] = None  # Concise summary (6-128 chars)
    long_description: Optional[str] = None  # Detailed explanation (256-1024 chars)
    
    @field_validator('questions')
    def validate_questions(cls, questions):
        """Validate that the number of questions is within limits."""
        if len(questions) > 100:
            raise ValueError("Maximum of 100 questions allowed per EvalSet")
        return questions
    
    @field_validator('short_description')
    def validate_short_description(cls, short_description):
        """Validate short_description length (6-128 characters)."""
        if short_description and len(short_description) < 6:
            raise ValueError(f"short_description must be at least 6 characters long (currently {len(short_description)} characters)")
        if short_description and len(short_description) > 128:
            raise ValueError(f"short_description must not exceed 128 characters (currently {len(short_description)} characters)")
        return short_description
    
    @field_validator('long_description')
    def validate_long_description(cls, long_description):
        """Validate long_description length (256-1024 characters)."""
        if long_description and len(long_description) < 256:
            raise ValueError(f"long_description must be at least 256 characters long (currently {len(long_description)} characters)")
        if long_description and len(long_description) > 1024:
            raise ValueError(f"long_description must not exceed 1024 characters (currently {len(long_description)} characters)")
        return long_description
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalSet":
        """Create from dictionary."""
        return cls(**data)


def get_evalset_path(evalset_id: str) -> str:
    """Get the file path for an EvalSet."""
    return os.path.join(EVALSETS_DIR, f"{evalset_id}.json")


def list_evalsets() -> List[EvalSet]:
    """List all available EvalSets."""
    evalsets = []
    for evalset_id in list_json_files(EVALSETS_DIR):
        evalset_data = load_json(get_evalset_path(evalset_id))
        if evalset_data:
            evalsets.append(EvalSet.from_dict(evalset_data))
    return evalsets


def get_evalset(evalset_id: str) -> Optional[EvalSet]:
    """Get a specific EvalSet by ID.
    
    First checks the LRU cache for the EvalSet, and if not found,
    loads it from disk and caches it for future use.
    """
    # Try to get from cache first
    cached_evalset = get_cached_evalset(evalset_id)
    if cached_evalset is not None:
        return cached_evalset
        
    # Not in cache, load from disk
    evalset_data = load_json(get_evalset_path(evalset_id))
    if evalset_data:
        evalset = EvalSet.from_dict(evalset_data)
        # Cache for future use
        cache_evalset(evalset_id, evalset)
        return evalset
    return None


def create_evalset(
    name: str, 
    questions: List[str],
    short_description: str,
    long_description: str
) -> EvalSet:
    """
    Create a new EvalSet.
    
    Before creating a new EvalSet, consider using list_evalsets() to see if a similar one
    already exists. This helps prevent duplication and promotes reuse of well-crafted EvalSets.
    
    Args:
        name: The name of the EvalSet
        questions: List of yes/no evaluation questions
        short_description: A concise summary (6-128 chars) of what this EvalSet measures
        long_description: A detailed explanation (256-1024 chars) of the evaluation criteria, 
                         how to interpret results, and when to use this EvalSet
        
    Returns:
        The created EvalSet object
    """
    evalset = EvalSet(
        name=name, 
        questions=questions, 
        short_description=short_description,
        long_description=long_description
        # template will use the DEFAULT_TEMPLATE
    )
    
    # Save to disk
    save_json(evalset.to_dict(), get_evalset_path(evalset.id))
    
    # Cache the new EvalSet
    cache_evalset(evalset.id, evalset)
    
    return evalset


def update_evalset(
    evalset_id: str,
    name: Optional[str] = None,
    questions: Optional[List[str]] = None,
    short_description: Optional[str] = None,
    long_description: Optional[str] = None
) -> Optional[EvalSet]:
    """
    Update an existing EvalSet.
    
    Args:
        evalset_id: ID of the EvalSet to update
        name: Optional new name
        questions: Optional new list of questions
        short_description: Optional new concise summary (6-128 chars)
        long_description: Optional new detailed explanation (256-1024 chars)
        
    Returns:
        The updated EvalSet object, or None if not found
    """
    evalset = get_evalset(evalset_id)
    if not evalset:
        return None
    
    if name is not None:
        evalset.name = name
    if questions is not None:
        evalset.questions = questions
    if short_description is not None:
        evalset.short_description = short_description
    if long_description is not None:
        evalset.long_description = long_description
    
    # Save to disk
    save_json(evalset.to_dict(), get_evalset_path(evalset.id))
    
    # Update the cache
    cache_evalset(evalset.id, evalset)
    
    return evalset


def delete_evalset(evalset_id: str) -> bool:
    """Delete an EvalSet from both disk and cache."""
    path = get_evalset_path(evalset_id)
    if os.path.exists(path):
        # Remove from disk
        os.remove(path)
        
        # Invalidate cache entry
        invalidate_evalset(evalset_id)
        
        return True
    return False


def get_cache_statistics() -> Dict[str, Any]:
    """Get statistics about the EvalSet cache performance.
    
    Returns:
        A dictionary with cache statistics including hit rate
    """
    return get_evalset_cache_stats()


def manage_evalset(
    action: str,
    evalset_id: Optional[str] = None,
    name: Optional[str] = None,
    questions: Optional[List[str]] = None,
    short_description: Optional[str] = None,
    long_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Manage EvalSet definitions for assessing conversation quality.
    
    Before creating a new EvalSet, consider first listing existing EvalSets with action="list" 
    to see if a similar one already exists. This promotes reuse and prevents duplication.
    
    Args:
        action: One of "create", "list", "get", "update", "delete", "cache_stats"
        evalset_id: Required for get, update, delete
        name: Required for create
        questions: Required for create, a list of yes/no questions (max 100)
        short_description: A concise summary (6-128 chars) of what this EvalSet measures (required for create)
        long_description: A detailed explanation (256-1024 chars) of the evaluation criteria (required for create)
    
    Returns:
        A dictionary containing the result of the operation
    """
    valid_actions = ["create", "list", "get", "update", "delete", "cache_stats"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "cache_stats":
            stats = get_cache_statistics()
            formatted_message = "\n".join([
                "# EvalSet Cache Statistics",
                f"- Cache Size: {stats['size']} / {stats['capacity']} (current/max)",
                f"- Hit Rate: {stats['hit_rate_pct']}%",
                f"- Hits: {stats['hits']}",
                f"- Misses: {stats['misses']}",
                f"- Evictions: {stats['evictions']}",
                f"- Expirations: {stats['expirations']}"
            ])
            return {
                "status": "success",
                "stats": stats,
                "formatted_message": formatted_message
            }
            
        elif action == "list":
            evalsets = list_evalsets()
            evalset_dicts = [e.to_dict() for e in evalsets]
            
            # Add evalsets directly to make it easier to access in the CLI
            result = format_list(evalset_dicts)
            result["evalsets"] = {e["id"]: e for e in evalset_dicts}
            return result
        
        elif action == "get":
            validate_required_params({"evalset_id": evalset_id}, ["evalset_id"])
            evalset = get_evalset(evalset_id)
            if not evalset:
                return format_error(f"EvalSet with ID '{evalset_id}' not found")
            
            eval_dict = evalset.to_dict()
            result = {
                "id": eval_dict["id"],
                "name": eval_dict["name"],
                "short_description": eval_dict.get("short_description", ""),
                "long_description": eval_dict.get("long_description", ""),
                "description": eval_dict.get("description"),
                "template": eval_dict["template"],
                "questions": eval_dict["questions"],
                "question_count": len(eval_dict["questions"])
            }
            
            return {
                "status": "success",
                "evalset": result,
                "formatted_message": "\n".join([
                    f"EvalSet: {eval_dict['name']} (ID: {eval_dict['id']})",
                    f"Short Description: {eval_dict.get('short_description', 'None')}",
                    f"Long Description: {eval_dict.get('long_description', 'None')}",
                    f"Template:\n{eval_dict['template']}",
                    f"\nQuestions ({len(eval_dict['questions'])}):",
                    *[f"{i}. {q}" for i, q in enumerate(eval_dict['questions'], 1)]
                ])
            }
        
        elif action == "create":
            validate_required_params({
                "name": name, 
                "questions": questions,
                "short_description": short_description,
                "long_description": long_description
            }, ["name", "questions", "short_description", "long_description"])
            
            # Validate questions
            if not isinstance(questions, list):
                return format_error("Questions must be a list of strings")
            
            if len(questions) > MAX_QUESTIONS_PER_EVALSET:
                return format_error(f"Maximum of {MAX_QUESTIONS_PER_EVALSET} questions allowed per EvalSet")
                
            # Check if we've reached the maximum number of EvalSets
            existing_evalsets = list_json_files(EVALSETS_DIR)
            if len(existing_evalsets) >= MAX_EVALSETS:
                return format_error(
                    f"Maximum of {MAX_EVALSETS} EvalSets reached. "
                    "Please delete an existing EvalSet before creating a new one."
                )
            
            # Validate short_description
            if short_description and len(short_description) < 6:
                return format_error(f"short_description must be at least 6 characters long (currently {len(short_description)} characters)")
            if short_description and len(short_description) > 128:
                return format_error(f"short_description must not exceed 128 characters (currently {len(short_description)} characters)")
                
            # Validate long_description
            if long_description and len(long_description) < 256:
                return format_error(f"long_description must be at least 256 characters long (currently {len(long_description)} characters)")
            if long_description and len(long_description) > 1024:
                return format_error(f"long_description must not exceed 1024 characters (currently {len(long_description)} characters)")
            
            # Template is now system-defined, no need to validate
            
            evalset = create_evalset(
                name=name,
                questions=questions,
                short_description=short_description,
                long_description=long_description
            )
            
            return {
                "status": "success",
                "evalset": evalset.to_dict(),
                "message": f"EvalSet '{name}' created with ID: {evalset.id}",
                "formatted_message": f"EvalSet '{name}' created with ID: {evalset.id}"
            }
        
        elif action == "update":
            validate_required_params({"evalset_id": evalset_id}, ["evalset_id"])
            
            # Validate questions if provided
            if questions is not None:
                if not isinstance(questions, list):
                    return format_error("Questions must be a list of strings")
                
                if len(questions) > 100:
                    return format_error("Maximum of 100 questions allowed per EvalSet")
            
            # Template is now system-defined, no need to validate
            
            # Validate short_description if provided
            if short_description is not None:
                if len(short_description) < 6:
                    return format_error(f"short_description must be at least 6 characters long (currently {len(short_description)} characters)")
                if len(short_description) > 128:
                    return format_error(f"short_description must not exceed 128 characters (currently {len(short_description)} characters)")
                
            # Validate long_description if provided
            if long_description is not None:
                if len(long_description) < 256:
                    return format_error(f"long_description must be at least 256 characters long (currently {len(long_description)} characters)")
                if len(long_description) > 1024:
                    return format_error(f"long_description must not exceed 1024 characters (currently {len(long_description)} characters)")
            
            evalset = update_evalset(
                evalset_id=evalset_id,
                name=name,
                questions=questions,
                short_description=short_description,
                long_description=long_description
            )
            
            if not evalset:
                return format_error(f"EvalSet with ID '{evalset_id}' not found")
            
            return {
                "status": "success",
                "evalset": evalset.to_dict(),
                "message": f"EvalSet '{evalset.name}' updated",
                "formatted_message": f"EvalSet '{evalset.name}' updated"
            }
        
        elif action == "delete":
            validate_required_params({"evalset_id": evalset_id}, ["evalset_id"])
            
            # Get the name before deleting
            evalset = get_evalset(evalset_id)
            if not evalset:
                return format_error(f"EvalSet with ID '{evalset_id}' not found")
            
            name = evalset.name
            
            if delete_evalset(evalset_id):
                return {
                    "status": "success",
                    "message": f"EvalSet '{name}' with ID '{evalset_id}' deleted",
                    "formatted_message": f"EvalSet '{name}' with ID '{evalset_id}' deleted"
                }
            else:
                return format_error(f"EvalSet with ID '{evalset_id}' not found")
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")
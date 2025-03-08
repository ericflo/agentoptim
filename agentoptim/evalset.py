"""EvalSet functionality for AgentOptim."""

import os
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

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
    description: Optional[str] = None
    
    @field_validator('questions')
    def validate_questions(cls, questions):
        """Validate that the number of questions is within limits."""
        if len(questions) > 100:
            raise ValueError("Maximum of 100 questions allowed per EvalSet")
        return questions
    
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
    """Get a specific EvalSet by ID."""
    evalset_data = load_json(get_evalset_path(evalset_id))
    if evalset_data:
        return EvalSet.from_dict(evalset_data)
    return None


def create_evalset(
    name: str, 
    questions: List[str],
    description: Optional[str] = None
) -> EvalSet:
    """
    Create a new EvalSet.
    
    Args:
        name: The name of the EvalSet
        questions: List of yes/no evaluation questions
        description: Optional description
        
    Returns:
        The created EvalSet object
    """
    evalset = EvalSet(
        name=name, 
        questions=questions, 
        description=description
        # template will use the DEFAULT_TEMPLATE
    )
    
    save_json(evalset.to_dict(), get_evalset_path(evalset.id))
    return evalset


def update_evalset(
    evalset_id: str,
    name: Optional[str] = None,
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Optional[EvalSet]:
    """
    Update an existing EvalSet.
    
    Args:
        evalset_id: ID of the EvalSet to update
        name: Optional new name
        questions: Optional new list of questions
        description: Optional new description
        
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
    if description is not None:
        evalset.description = description
    
    save_json(evalset.to_dict(), get_evalset_path(evalset.id))
    return evalset


def delete_evalset(evalset_id: str) -> bool:
    """Delete an EvalSet."""
    path = get_evalset_path(evalset_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def manage_evalset(
    action: str,
    evalset_id: Optional[str] = None,
    name: Optional[str] = None,
    template: Optional[str] = None,  # Kept for backward compatibility but ignored
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Manage EvalSet definitions for assessing conversation quality.
    
    Args:
        action: One of "create", "list", "get", "update", "delete"
        evalset_id: Required for get, update, delete
        name: Required for create
        template: Deprecated - templates are now system-defined (ignored)
        questions: Required for create, a list of yes/no questions (max 100)
        description: Optional description
    
    Returns:
        A dictionary containing the result of the operation
    """
    valid_actions = ["create", "list", "get", "update", "delete"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "list":
            evalsets = list_evalsets()
            return format_list([e.to_dict() for e in evalsets])
        
        elif action == "get":
            validate_required_params({"evalset_id": evalset_id}, ["evalset_id"])
            evalset = get_evalset(evalset_id)
            if not evalset:
                return format_error(f"EvalSet with ID '{evalset_id}' not found")
            
            eval_dict = evalset.to_dict()
            result = {
                "id": eval_dict["id"],
                "name": eval_dict["name"],
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
                    f"Description: {eval_dict.get('description', 'None')}",
                    f"Template:\n{eval_dict['template']}",
                    f"\nQuestions ({len(eval_dict['questions'])}):",
                    *[f"{i}. {q}" for i, q in enumerate(eval_dict['questions'], 1)]
                ])
            }
        
        elif action == "create":
            validate_required_params({
                "name": name, 
                "questions": questions
            }, ["name", "questions"])
            
            # Validate questions
            if not isinstance(questions, list):
                return format_error("Questions must be a list of strings")
            
            if len(questions) > 100:
                return format_error("Maximum of 100 questions allowed per EvalSet")
            
            # Template is now system-defined, no need to validate
            
            evalset = create_evalset(
                name=name,
                questions=questions,
                description=description
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
            # Inform the user if they try to update the template
            if template is not None:
                # Log a warning but don't fail the request
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Attempt to update template for EvalSet {evalset_id} ignored - templates are now system-defined")
            
            evalset = update_evalset(
                evalset_id=evalset_id,
                name=name,
                questions=questions,
                description=description
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
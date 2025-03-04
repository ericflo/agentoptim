"""Evaluation functionality for AgentOptim."""

import os
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentoptim.utils import (
    EVALUATIONS_DIR,
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


class Evaluation(BaseModel):
    """Model for an evaluation."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    template: str
    questions: List[str]
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evaluation":
        """Create from dictionary."""
        return cls(**data)


def get_evaluation_path(evaluation_id: str) -> str:
    """Get the file path for an evaluation."""
    return os.path.join(EVALUATIONS_DIR, f"{evaluation_id}.json")


def list_evaluations() -> List[Evaluation]:
    """List all available evaluations."""
    evaluations = []
    for evaluation_id in list_json_files(EVALUATIONS_DIR):
        evaluation_data = load_json(get_evaluation_path(evaluation_id))
        if evaluation_data:
            evaluations.append(Evaluation.from_dict(evaluation_data))
    return evaluations


def get_evaluation(evaluation_id: str) -> Optional[Evaluation]:
    """Get a specific evaluation by ID."""
    evaluation_data = load_json(get_evaluation_path(evaluation_id))
    if evaluation_data:
        return Evaluation.from_dict(evaluation_data)
    return None


def create_evaluation(
    name: str, template: str, questions: List[str], description: Optional[str] = None
) -> Evaluation:
    """Create a new evaluation."""
    evaluation = Evaluation(
        name=name, template=template, questions=questions, description=description
    )
    save_json(evaluation.to_dict(), get_evaluation_path(evaluation.id))
    return evaluation


def update_evaluation(
    evaluation_id: str,
    name: Optional[str] = None,
    template: Optional[str] = None,
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> Optional[Evaluation]:
    """Update an existing evaluation."""
    evaluation = get_evaluation(evaluation_id)
    if not evaluation:
        return None
    
    if name is not None:
        evaluation.name = name
    if template is not None:
        evaluation.template = template
    if questions is not None:
        evaluation.questions = questions
    if description is not None:
        evaluation.description = description
    
    save_json(evaluation.to_dict(), get_evaluation_path(evaluation.id))
    return evaluation


def delete_evaluation(evaluation_id: str) -> bool:
    """Delete an evaluation."""
    path = get_evaluation_path(evaluation_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def manage_evaluation(
    action: str,
    evaluation_id: Optional[str] = None,
    name: Optional[str] = None,
    template: Optional[str] = None,
    questions: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> str:
    """
    Manage evaluation definitions for assessing response quality.
    
    Args:
        action: One of "create", "list", "get", "update", "delete"
        evaluation_id: Required for get, update, delete
        name: Required for create
        template: Required for create, optional for update
        questions: Required for create, optional for update
        description: Optional description
    
    Returns:
        A string response describing the result
    """
    valid_actions = ["create", "list", "get", "update", "delete"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "list":
            evaluations = list_evaluations()
            return format_list([e.to_dict() for e in evaluations])
        
        elif action == "get":
            validate_required_params({"evaluation_id": evaluation_id}, ["evaluation_id"])
            evaluation = get_evaluation(evaluation_id)
            if not evaluation:
                return format_error(f"Evaluation with ID '{evaluation_id}' not found")
            
            eval_dict = evaluation.to_dict()
            result = [
                f"Evaluation: {eval_dict['name']} (ID: {eval_dict['id']})",
                f"Description: {eval_dict.get('description', 'None')}",
                f"Template:\n{eval_dict['template']}",
                "Questions:",
            ]
            
            for i, question in enumerate(eval_dict["questions"], 1):
                result.append(f"{i}. {question}")
            
            return "\n".join(result)
        
        elif action == "create":
            validate_required_params(
                {"name": name, "template": template, "questions": questions},
                ["name", "template", "questions"],
            )
            
            if not isinstance(questions, list) or not questions:
                return format_error("Questions must be a non-empty list")
            
            if len(questions) > 100:
                return format_error("Maximum of 100 questions allowed per evaluation")
            
            evaluation = create_evaluation(name, template, questions, description)
            return format_success(
                f"Evaluation '{name}' created with ID: {evaluation.id}"
            )
        
        elif action == "update":
            validate_required_params({"evaluation_id": evaluation_id}, ["evaluation_id"])
            
            if questions is not None and len(questions) > 100:
                return format_error("Maximum of 100 questions allowed per evaluation")
            
            evaluation = update_evaluation(
                evaluation_id, name, template, questions, description
            )
            if not evaluation:
                return format_error(f"Evaluation with ID '{evaluation_id}' not found")
            
            return format_success(f"Evaluation '{evaluation.name}' updated")
        
        elif action == "delete":
            validate_required_params({"evaluation_id": evaluation_id}, ["evaluation_id"])
            
            if delete_evaluation(evaluation_id):
                return format_success(f"Evaluation with ID '{evaluation_id}' deleted")
            else:
                return format_error(f"Evaluation with ID '{evaluation_id}' not found")
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")
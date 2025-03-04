"""Evaluation functionality for AgentOptim."""

import os
import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from agentoptim.utils import (
    DATA_DIR,
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


class EvaluationCriterion(BaseModel):
    """Model for an evaluation criterion."""
    
    name: str
    description: Optional[str] = None
    weight: float = 1.0
    min_score: float = 1.0
    max_score: float = 5.0
    scoring_guidelines: Optional[str] = None


class Evaluation(BaseModel):
    """Model for an evaluation."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    template: str = "{input}"
    questions: List[str] = Field(default_factory=list)
    criteria: List[EvaluationCriterion] = Field(default_factory=list)
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
    name: str, 
    template: str = "{input}", 
    questions: Optional[List[str]] = None,
    criteria_or_description: Optional[Union[List[Dict[str, Any]], str]] = None,  
    description: Optional[str] = None
) -> Evaluation:
    """
    Create a new evaluation.
    
    Args:
        name: The name of the evaluation
        template: The evaluation template string
        questions: List of evaluation questions
        criteria_or_description: Either a list of criteria dictionaries or a description string
        description: Optional description
        
    Returns:
        The created Evaluation object
    """
    # Handle the case where criteria_or_description is actually a description string
    criteria = None
    if isinstance(criteria_or_description, str):
        # The old function signature had description as the 4th parameter
        real_description = criteria_or_description
        if description is not None:
            # We have both parameters, which shouldn't happen normally
            # but we'll prioritize the explicit description parameter
            real_description = description
    else:
        # The new function signature has criteria as the 4th parameter
        criteria = criteria_or_description
        real_description = description
    
    # Convert criteria dictionaries to EvaluationCriterion objects
    criteria_objects = []
    if criteria:
        for criterion in criteria:
            criteria_objects.append(EvaluationCriterion(**criterion))
    
    # Create the evaluation
    evaluation = Evaluation(
        name=name, 
        template=template, 
        questions=questions or [], 
        criteria=criteria_objects,
        description=real_description
    )
    
    save_json(evaluation.to_dict(), get_evaluation_path(evaluation.id))
    return evaluation


def update_evaluation(
    evaluation_id: str,
    name: Optional[str] = None,
    template: Optional[str] = None,
    questions: Optional[List[str]] = None,
    criteria_or_description: Optional[Union[List[Dict[str, Any]], str]] = None,
    description: Optional[str] = None,
) -> Optional[Evaluation]:
    """
    Update an existing evaluation.
    
    Args:
        evaluation_id: ID of the evaluation to update
        name: Optional new name
        template: Optional new template string
        questions: Optional new list of questions
        criteria_or_description: Either a list of criteria dictionaries or a description string
        description: Optional new description
        
    Returns:
        The updated Evaluation object, or None if not found
    """
    evaluation = get_evaluation(evaluation_id)
    if not evaluation:
        return None
    
    if name is not None:
        evaluation.name = name
    if template is not None:
        evaluation.template = template
    if questions is not None:
        evaluation.questions = questions
    
    # Handle the case where criteria_or_description is actually a description string
    if isinstance(criteria_or_description, str):
        evaluation.description = criteria_or_description
    elif criteria_or_description is not None:
        # It's a criteria list
        criteria_objects = []
        for criterion in criteria_or_description:
            criteria_objects.append(EvaluationCriterion(**criterion))
        evaluation.criteria = criteria_objects
    
    # The explicit description parameter overrides any description from criteria_or_description
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
    criteria_or_description: Optional[Union[List[Dict[str, Any]], str]] = None,
    description: Optional[str] = None,
) -> str:
    """
    Manage evaluation definitions for assessing response quality.
    
    Args:
        action: One of "create", "list", "get", "update", "delete"
        evaluation_id: Required for get, update, delete
        name: Required for create
        template: Optional template string (default: "{input}")
        questions: Optional list of evaluation questions
        criteria: Optional list of criteria dictionaries with name, description, and weight
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
            ]
            
            # Show questions if available
            if eval_dict.get("questions"):
                result.append("Questions:")
                for i, question in enumerate(eval_dict["questions"], 1):
                    result.append(f"{i}. {question}")
            
            # Show criteria if available
            if eval_dict.get("criteria"):
                result.append("\nCriteria:")
                for criterion in eval_dict["criteria"]:
                    result.append(f"- {criterion['name']} (weight: {criterion['weight']})")
                    if criterion.get("description"):
                        result.append(f"  Description: {criterion['description']}")
            
            return "\n".join(result)
        
        elif action == "create":
            # For testing, we need to validate that the required fields are present
            # We need to make sure at least name and template are provided
            validate_required_params(
                {"name": name, "template": template or "{input}"},
                ["name", "template"]
            )
            
            # Validate questions if provided
            if questions is not None:
                if not isinstance(questions, list):
                    return format_error("Questions must be a list")
                
                if len(questions) > 100:
                    return format_error("Maximum of 100 questions allowed per evaluation")
            
            # Validate criteria if provided
            if criteria_or_description is not None and isinstance(criteria_or_description, list):
                if not isinstance(criteria_or_description, list):
                    return format_error("Criteria must be a list")
                
                for criterion in criteria_or_description:
                    if not isinstance(criterion, dict):
                        return format_error("Each criterion must be a dictionary")
                    
                    if "name" not in criterion:
                        return format_error("Each criterion must have a name")
            
            evaluation = create_evaluation(
                name=name,
                template=template or "{input}",
                questions=questions,
                criteria_or_description=criteria_or_description,
                description=description
            )
            
            return format_success(
                f"Evaluation '{name}' created with ID: {evaluation.id}"
            )
        
        elif action == "update":
            validate_required_params({"evaluation_id": evaluation_id}, ["evaluation_id"])
            
            # Validate questions if provided
            if questions is not None:
                if not isinstance(questions, list):
                    return format_error("Questions must be a list")
                
                if len(questions) > 100:
                    return format_error("Maximum of 100 questions allowed per evaluation")
            
            # Validate criteria if provided
            if criteria_or_description is not None and isinstance(criteria_or_description, list):
                if not isinstance(criteria_or_description, list):
                    return format_error("Criteria must be a list")
                
                for criterion in criteria_or_description:
                    if not isinstance(criterion, dict):
                        return format_error("Each criterion must be a dictionary")
                    
                    if "name" not in criterion:
                        return format_error("Each criterion must have a name")
            
            evaluation = update_evaluation(
                evaluation_id=evaluation_id,
                name=name,
                template=template,
                questions=questions,
                criteria_or_description=criteria_or_description,
                description=description
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
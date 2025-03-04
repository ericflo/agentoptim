"""Experiment framework for AgentOptim."""

import os
import json
import uuid
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field

from agentoptim.utils import (
    EXPERIMENTS_DIR,
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


class PromptVariantType(str, Enum):
    """Types of prompt variants that can be tested."""
    
    SYSTEM = "system"
    USER = "user"
    STANDALONE = "standalone"
    COMBINED = "combined"


class PromptVariable(BaseModel):
    """A variable that can be used in a prompt template."""
    
    name: str
    description: Optional[str] = None
    options: List[str]
    

class PromptVariant(BaseModel):
    """A variation of a prompt to be tested."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    type: PromptVariantType
    template: str
    description: Optional[str] = None
    variables: List[PromptVariable] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVariant":
        """Create from dictionary."""
        return cls(**data)


class Experiment(BaseModel):
    """Model for an experiment."""
    
    id: str = Field(default_factory=generate_id)
    name: str
    description: Optional[str] = None
    dataset_id: str
    evaluation_id: str
    prompt_variants: List[PromptVariant]
    model_name: str
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    status: str = "created"  # created, running, completed, failed
    results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(**data)


def get_experiment_path(experiment_id: str) -> str:
    """Get the file path for an experiment."""
    return os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")


def list_experiments() -> List[Dict[str, Any]]:
    """List all available experiments."""
    experiments = []
    for experiment_id in list_json_files(EXPERIMENTS_DIR):
        experiment_data = load_json(get_experiment_path(experiment_id))
        if experiment_data:
            experiments.append(experiment_data)
    return experiments


def get_experiment(experiment_id: str) -> Optional[Experiment]:
    """Get a specific experiment by ID."""
    experiment_data = load_json(get_experiment_path(experiment_id))
    if experiment_data:
        return Experiment.from_dict(experiment_data)
    return None


def create_experiment(
    name: str,
    dataset_id: str,
    evaluation_id: str,
    prompt_variants: List[Dict[str, Any]],
    model_name: str,
    description: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Experiment:
    """
    Create a new experiment.
    
    Args:
        name: Name of the experiment
        dataset_id: ID of the dataset to use
        evaluation_id: ID of the evaluation to use
        prompt_variants: List of prompt variants to test
        model_name: Name of the model to use
        description: Optional description
        temperature: Model temperature setting
        max_tokens: Maximum tokens to generate
        metadata: Additional metadata
        
    Returns:
        The created experiment
    """
    # Convert dict variants to PromptVariant objects
    variants = []
    for variant_data in prompt_variants:
        # Process variables if they exist
        if "variables" in variant_data:
            variable_list = []
            for var_data in variant_data["variables"]:
                variable_list.append(PromptVariable(**var_data))
            variant_data["variables"] = variable_list
        
        variants.append(PromptVariant(**variant_data))
    
    experiment = Experiment(
        name=name,
        description=description,
        dataset_id=dataset_id,
        evaluation_id=evaluation_id,
        prompt_variants=variants,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=metadata or {},
    )
    
    save_json(experiment.to_dict(), get_experiment_path(experiment.id))
    return experiment


def update_experiment(
    experiment_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    dataset_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    prompt_variants: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Experiment]:
    """
    Update an existing experiment.
    
    Args:
        experiment_id: ID of the experiment to update
        name: New name
        description: New description
        dataset_id: New dataset ID
        evaluation_id: New evaluation ID
        prompt_variants: New prompt variants
        model_name: New model name
        temperature: New temperature
        max_tokens: New max tokens
        status: New status
        results: New results
        metadata: New metadata
        
    Returns:
        The updated experiment or None if not found
    """
    experiment = get_experiment(experiment_id)
    if not experiment:
        return None
    
    if name is not None:
        experiment.name = name
    if description is not None:
        experiment.description = description
    if dataset_id is not None:
        experiment.dataset_id = dataset_id
    if evaluation_id is not None:
        experiment.evaluation_id = evaluation_id
    if prompt_variants is not None:
        # Convert dict variants to PromptVariant objects
        variants = []
        for variant_data in prompt_variants:
            # Process variables if they exist
            if "variables" in variant_data:
                variable_list = []
                for var_data in variant_data["variables"]:
                    variable_list.append(PromptVariable(**var_data))
                variant_data["variables"] = variable_list
            
            variants.append(PromptVariant(**variant_data))
        experiment.prompt_variants = variants
    if model_name is not None:
        experiment.model_name = model_name
    if temperature is not None:
        experiment.temperature = temperature
    if max_tokens is not None:
        experiment.max_tokens = max_tokens
    if status is not None:
        experiment.status = status
    if results is not None:
        experiment.results = results
    if metadata is not None:
        experiment.metadata = metadata
    
    save_json(experiment.to_dict(), get_experiment_path(experiment.id))
    return experiment


def delete_experiment(experiment_id: str) -> bool:
    """Delete an experiment."""
    path = get_experiment_path(experiment_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def duplicate_experiment(
    experiment_id: str, 
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
) -> Optional[Experiment]:
    """
    Create a duplicate of an existing experiment.
    
    Args:
        experiment_id: ID of the experiment to duplicate
        new_name: Name for the new experiment (default: "Copy of [original name]")
        new_description: Description for the new experiment
        
    Returns:
        The duplicated experiment or None if original not found
    """
    experiment = get_experiment(experiment_id)
    if not experiment:
        return None
    
    # Create a copy of the experiment data
    experiment_data = experiment.to_dict()
    
    # Remove ID and results, reset status
    experiment_data.pop("id")
    experiment_data.pop("results", None)
    experiment_data["status"] = "created"
    
    # Update name and description
    if new_name:
        experiment_data["name"] = new_name
    else:
        experiment_data["name"] = f"Copy of {experiment_data['name']}"
    
    if new_description:
        experiment_data["description"] = new_description
    
    # Create new experiment
    return create_experiment(**experiment_data)


def manage_experiment(
    action: str,
    experiment_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    dataset_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
    prompt_variants: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    status: Optional[str] = None,
    results: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    new_name: Optional[str] = None,
) -> str:
    """
    Manage experiments for prompt optimization.
    
    Args:
        action: One of "create", "list", "get", "update", "delete", "duplicate"
        experiment_id: Required for get, update, delete, duplicate
        name: Required for create
        description: Optional description
        dataset_id: ID of dataset to use (required for create)
        evaluation_id: ID of evaluation to use (required for create)
        prompt_variants: List of prompt variants to test (required for create)
        model_name: Name of model to use (required for create)
        temperature: Model temperature setting
        max_tokens: Maximum tokens to generate
        status: Experiment status
        results: Experiment results
        metadata: Additional metadata
        new_name: New name for duplicated experiment
    
    Returns:
        A string response describing the result
    """
    valid_actions = ["create", "list", "get", "update", "delete", "duplicate"]
    
    try:
        validate_action(action, valid_actions)
        
        # Handle each action
        if action == "list":
            experiments = list_experiments()
            return format_list(experiments)
        
        elif action == "get":
            validate_required_params({"experiment_id": experiment_id}, ["experiment_id"])
            experiment = get_experiment(experiment_id)
            if not experiment:
                return format_error(f"Experiment with ID '{experiment_id}' not found")
            
            exp_dict = experiment.to_dict()
            result = [
                f"Experiment: {exp_dict['name']} (ID: {exp_dict['id']})",
                f"Description: {exp_dict.get('description', 'None')}",
                f"Status: {exp_dict['status']}",
                f"Dataset ID: {exp_dict['dataset_id']}",
                f"Evaluation ID: {exp_dict['evaluation_id']}",
                f"Model: {exp_dict['model_name']} (temp: {exp_dict['temperature']})",
            ]
            
            if exp_dict.get('max_tokens'):
                result.append(f"Max Tokens: {exp_dict['max_tokens']}")
            
            # Add prompt variants
            result.append(f"\nPrompt Variants ({len(exp_dict['prompt_variants'])}):")
            for i, variant in enumerate(exp_dict["prompt_variants"], 1):
                result.append(f"\n{i}. {variant['name']} ({variant['type']})")
                if variant.get('description'):
                    result.append(f"   Description: {variant['description']}")
                result.append(f"   Template: {variant['template'][:100]}...")
                
                if variant.get('variables'):
                    result.append(f"\n   Variables:")
                    for var in variant['variables']:
                        options_str = ", ".join(var['options'][:3])
                        if len(var['options']) > 3:
                            options_str += f", ... ({len(var['options'])} total)"
                        result.append(f"   - {var['name']}: {options_str}")
            
            # Add results if they exist
            if exp_dict.get('results'):
                result.append("\nResults Summary:")
                try:
                    for variant_id, scores in exp_dict['results'].items():
                        # Find the variant name
                        variant_name = next((v['name'] for v in exp_dict['prompt_variants'] 
                                            if v['id'] == variant_id), variant_id)
                        
                        if isinstance(scores, dict) and 'average_score' in scores:
                            result.append(f"- {variant_name}: {scores['average_score']:.2f}")
                        else:
                            result.append(f"- {variant_name}: {scores}")
                except:
                    result.append(str(exp_dict['results']))
            
            return "\n".join(result)
        
        elif action == "create":
            validate_required_params(
                {
                    "name": name,
                    "dataset_id": dataset_id,
                    "evaluation_id": evaluation_id,
                    "prompt_variants": prompt_variants,
                    "model_name": model_name,
                },
                ["name", "dataset_id", "evaluation_id", "prompt_variants", "model_name"],
            )
            
            if not isinstance(prompt_variants, list) or not prompt_variants:
                return format_error("Prompt variants must be a non-empty list")
            
            for i, variant in enumerate(prompt_variants):
                if "name" not in variant or "type" not in variant or "template" not in variant:
                    return format_error(f"Prompt variant at index {i} is missing required fields")
            
            experiment = create_experiment(
                name=name,
                description=description,
                dataset_id=dataset_id,
                evaluation_id=evaluation_id,
                prompt_variants=prompt_variants,
                model_name=model_name,
                temperature=temperature if temperature is not None else 0.0,
                max_tokens=max_tokens,
                metadata=metadata,
            )
            
            return format_success(
                f"Experiment '{name}' created with ID: {experiment.id} "
                f"({len(experiment.prompt_variants)} prompt variants)"
            )
        
        elif action == "update":
            validate_required_params({"experiment_id": experiment_id}, ["experiment_id"])
            
            if prompt_variants is not None:
                if not isinstance(prompt_variants, list) or not prompt_variants:
                    return format_error("Prompt variants must be a non-empty list")
                
                for i, variant in enumerate(prompt_variants):
                    if "name" not in variant or "type" not in variant or "template" not in variant:
                        return format_error(f"Prompt variant at index {i} is missing required fields")
            
            experiment = update_experiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                dataset_id=dataset_id,
                evaluation_id=evaluation_id,
                prompt_variants=prompt_variants,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                status=status,
                results=results,
                metadata=metadata,
            )
            
            if not experiment:
                return format_error(f"Experiment with ID '{experiment_id}' not found")
            
            return format_success(f"Experiment '{experiment.name}' updated")
        
        elif action == "delete":
            validate_required_params({"experiment_id": experiment_id}, ["experiment_id"])
            
            if delete_experiment(experiment_id):
                return format_success(f"Experiment with ID '{experiment_id}' deleted")
            else:
                return format_error(f"Experiment with ID '{experiment_id}' not found")
        
        elif action == "duplicate":
            validate_required_params({"experiment_id": experiment_id}, ["experiment_id"])
            
            experiment = duplicate_experiment(
                experiment_id=experiment_id,
                new_name=new_name,
            )
            
            if not experiment:
                return format_error(f"Experiment with ID '{experiment_id}' not found")
            
            return format_success(
                f"Experiment duplicated successfully with ID: {experiment.id} and name: '{experiment.name}'"
            )
    
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        return format_error(f"Unexpected error: {str(e)}")
"""Utility functions for AgentOptim."""

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Define base directories for storage
DATA_DIR = os.environ.get("AGENTOPTIM_DATA_DIR", os.path.expanduser("~/.agentoptim"))
EVALUATIONS_DIR = os.path.join(DATA_DIR, "evaluations")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Ensure directories exist
for directory in [DATA_DIR, EVALUATIONS_DIR, DATASETS_DIR, EXPERIMENTS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


def generate_id() -> str:
    """Generate a unique ID for objects."""
    return str(uuid.uuid4())


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save
        filepath: Path to the JSON file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        The loaded data
    """
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, "r") as f:
        return json.load(f)


def list_json_files(directory: str) -> List[str]:
    """
    List all JSON files in a directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of filenames without the .json extension
    """
    if not os.path.exists(directory):
        return []
    
    return [
        os.path.splitext(filename)[0]
        for filename in os.listdir(directory)
        if filename.endswith(".json")
    ]


class ValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


def validate_action(action: str, valid_actions: List[str]) -> None:
    """
    Validate that an action is valid.
    
    Args:
        action: The action to validate
        valid_actions: List of valid actions
        
    Raises:
        ValidationError: If the action is not valid
    """
    if action not in valid_actions:
        valid_actions_str = ", ".join(valid_actions)
        raise ValidationError(
            f"Invalid action: '{action}'. Valid actions are: {valid_actions_str}"
        )


def validate_required_params(params: Dict[str, Any], required: List[str]) -> None:
    """
    Validate that required parameters are present.
    
    Args:
        params: Dictionary of parameters
        required: List of required parameter names
        
    Raises:
        ValidationError: If any required parameter is missing
    """
    missing = [param for param in required if param not in params or params[param] is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValidationError(f"Missing required parameters: {missing_str}")


def format_error(message: str) -> str:
    """Format an error message for consistent output."""
    return f"Error: {message}"


def format_success(message: str) -> str:
    """Format a success message for consistent output."""
    return f"Success: {message}"


def format_list(items: List[Dict[str, Any]], name_field: str = "name") -> str:
    """
    Format a list of items for display.
    
    Args:
        items: List of item dictionaries
        name_field: The field to use as the item name
        
    Returns:
        Formatted string listing the items
    """
    if not items:
        return "No items found."
    
    result = []
    for item in items:
        name = item.get(name_field, "Unnamed")
        id_value = item.get("id", "No ID")
        description = item.get("description", "")
        
        if description:
            result.append(f"- {name} (ID: {id_value}): {description}")
        else:
            result.append(f"- {name} (ID: {id_value})")
    
    return "\n".join(result)
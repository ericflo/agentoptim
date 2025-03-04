"""Utility functions for AgentOptim."""

import os
import json
import uuid
import logging
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable

from .errors import ValidationError, ResourceNotFoundError, OperationError, setup_logging

# Set up module logger
logger = logging.getLogger(__name__)

# Define base directories for storage
DATA_DIR = os.environ.get("AGENTOPTIM_DATA_DIR", os.path.expanduser("~/.agentoptim"))
EVALUATIONS_DIR = os.path.join(DATA_DIR, "evaluations")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

def get_data_path(subfolder: Optional[str] = None) -> str:
    """
    Get the path to the data directory or a subdirectory.
    
    Args:
        subfolder: Optional subfolder within the data directory
        
    Returns:
        Full path to the data directory or specified subfolder
    """
    if subfolder:
        path = os.path.join(DATA_DIR, subfolder)
        os.makedirs(path, exist_ok=True)
        return path
    return DATA_DIR

def get_data_dir() -> str:
    """
    Get the path to the main data directory.
    
    Returns:
        Full path to the data directory
    """
    return DATA_DIR

# Ensure directories exist
def ensure_data_directories():
    """Create all necessary data directories if they don't exist."""
    for directory in [DATA_DIR, EVALUATIONS_DIR, DATASETS_DIR, EXPERIMENTS_DIR, RESULTS_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise OperationError(
                operation="directory_creation",
                message=f"Failed to create data directory: {directory}",
                resource_type="storage"
            )

# Initialize directories
ensure_data_directories()


def generate_id() -> str:
    """Generate a unique ID for objects."""
    return str(uuid.uuid4())


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save
        filepath: Path to the JSON file
        
    Raises:
        OperationError: If the save operation fails
    """
    try:
        # Ensure parent directory exists
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.debug(f"Successfully saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save data to {filepath}: {str(e)}")
        raise OperationError(
            operation="save_json",
            message=f"Failed to save data to {filepath}: {str(e)}",
            resource_type="file",
            resource_id=filepath
        )


def load_json(filepath: str, required: bool = False) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        required: Whether the file must exist
        
    Returns:
        The loaded data, or None if the file doesn't exist and required=False
        
    Raises:
        ResourceNotFoundError: If the file doesn't exist and required=True
        OperationError: If the load operation fails
    """
    if not os.path.exists(filepath):
        if required:
            logger.error(f"Required file not found: {filepath}")
            raise ResourceNotFoundError(
                resource_type="file",
                resource_id=filepath,
                message=f"Required file not found: {filepath}"
            )
        return None
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded data from {filepath}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {filepath}: {str(e)}")
        raise OperationError(
            operation="load_json", 
            message=f"Failed to parse JSON from {filepath}: {str(e)}",
            resource_type="file",
            resource_id=filepath
        )
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {str(e)}")
        raise OperationError(
            operation="load_json",
            message=f"Failed to load data from {filepath}: {str(e)}",
            resource_type="file",
            resource_id=filepath
        )


def list_json_files(directory: str) -> List[str]:
    """
    List all JSON files in a directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of filenames without the .json extension
        
    Raises:
        OperationError: If the directory exists but cannot be read
    """
    if not os.path.exists(directory):
        logger.debug(f"Directory does not exist: {directory}")
        return []
    
    try:
        files = [
            os.path.splitext(filename)[0]
            for filename in os.listdir(directory)
            if filename.endswith(".json")
        ]
        logger.debug(f"Found {len(files)} JSON files in {directory}")
        return files
    except PermissionError as e:
        logger.error(f"Permission denied when reading directory {directory}: {str(e)}")
        raise OperationError(
            operation="list_files",
            message=f"Permission denied when reading directory: {directory}",
            resource_type="directory",
            resource_id=directory
        )
    except Exception as e:
        logger.error(f"Failed to list files in directory {directory}: {str(e)}")
        raise OperationError(
            operation="list_files",
            message=f"Failed to list files in directory: {directory}",
            resource_type="directory",
            resource_id=directory
        )


def validate_action(action: str, valid_actions: List[str]) -> None:
    """
    Validate that an action is valid.
    
    Args:
        action: The action to validate
        valid_actions: List of valid actions
        
    Raises:
        ValidationError: If the action is not valid
    """
    from .validation import validate_one_of
    
    try:
        validate_one_of(action, "action", valid_actions)
    except ValidationError as e:
        logger.error(f"Invalid action validation: {e.message}")
        # Re-raise with appropriate error message for backward compatibility
        valid_actions_str = ", ".join(valid_actions)
        raise ValidationError(
            f"Invalid action: '{action}'. Valid actions are: {valid_actions_str}",
            field="action",
            value=action
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
        logger.error(f"Missing required parameters: {missing_str}")
        raise ValidationError(
            f"Missing required parameters: {missing_str}",
            field="parameters"
        )


def format_error(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format an error message for consistent API output.
    
    Args:
        message: The error message
        details: Optional dictionary of additional error details
        
    Returns:
        A dictionary with error details formatted for API response
    """
    logger.debug(f"Formatting error message: {message}")
    
    # Create the standard error dictionary for API responses and tests
    response = {
        "error": True,
        "message": message
    }
    
    # Add any details if provided
    if details:
        response["details"] = details
    
    # For MCP presentation, also create a formatted string version
    # Basic formatting to make errors stand out and be clear
    error_msg = f"Error: {message}"
    
    # Add helpful details if provided
    if details:
        details_str = "\n".join([f"- {k}: {v}" for k, v in details.items()])
        error_msg += f"\n\nDetails:\n{details_str}"
    
    # Add a helpful suggestion if possible based on message content
    if "required parameters" in message:
        error_msg += "\n\nPlease check that you've provided all required parameters."
    elif "not found" in message:
        error_msg += "\n\nUse the 'list' action to see available resources."
    elif "invalid action" in message.lower():
        error_msg += "\n\nCheck the documentation for valid actions and their parameters."
    elif "permission" in message.lower():
        error_msg += "\n\nCheck file/directory permissions and try again."
    
    # Add the formatted message for MCP display
    response["formatted_message"] = error_msg
    
    return response


def format_success(message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format a success message for consistent API output.
    
    Args:
        message: The success message
        data: Optional data to include in the response
        
    Returns:
        A dictionary with success details formatted for API response
    """
    logger.debug(f"Formatting success message: {message}")
    
    # Create the standard success dictionary for API responses and tests
    response = {
        "error": False,
        "message": message
    }
    
    # Include data if provided
    if data:
        response["data"] = data
    
    # For MCP presentation, also create a formatted string version
    # Format the base success message
    success_msg = f"Success: {message}"
    
    # Add formatted data if provided
    if data:
        if len(data) == 1 and "id" in data:
            # Simple ID response
            success_msg += f"\nID: {data['id']}"
        else:
            # More complex data to present
            data_str = "\n".join([f"- {k}: {v}" for k, v in data.items()])
            success_msg += f"\n\nDetails:\n{data_str}"
    
    # Add next step suggestions based on the message content
    if "created" in message.lower():
        resource_type = message.split("'")[1].split("'")[0] if "'" in message else "resource"
        success_msg += f"\n\nYou can now use this {resource_type} in other operations."
    elif "updated" in message.lower():
        success_msg += "\n\nThe changes have been saved successfully."
    elif "deleted" in message.lower():
        success_msg += "\n\nThe resource has been permanently removed."
    
    # Add the formatted message for MCP display
    response["formatted_message"] = success_msg
    
    return response


def format_list(
    items: List[Dict[str, Any]], 
    name_field: str = "name",
    include_fields: Optional[List[str]] = None,
    resource_type: str = "items"
) -> Dict[str, Any]:
    """
    Format a list of items for API response.
    
    Args:
        items: List of item dictionaries
        name_field: The field to use as the item name
        include_fields: Optional list of fields to include in the response
        resource_type: The type of resources being listed (e.g., "evaluations", "datasets")
        
    Returns:
        A dictionary with formatted list data for API response
    """
    logger.debug(f"Formatting list of {len(items)} {resource_type}")
    
    # Create the standard dictionary response for API and tests
    formatted_items = []
    for item in items:
        name = item.get(name_field, "Unnamed")
        id_value = item.get("id", "No ID")
        
        formatted_item = {
            "id": id_value,
            "name": name
        }
        
        # Include description if present
        if "description" in item and item["description"]:
            formatted_item["description"] = item["description"]
        
        # Include additional requested fields
        if include_fields:
            for field in include_fields:
                if field in item and field not in formatted_item:
                    formatted_item[field] = item[field]
        
        formatted_items.append(formatted_item)
    
    if not items:
        message = f"No {resource_type} found."
    else:
        message = f"Found {len(items)} {resource_type}."
        
    response = {
        "error": False,
        "message": message,
        "items": formatted_items,
        "count": len(items)
    }
    
    # For MCP presentation, also create a formatted string version
    if not items:
        formatted_message = f"No {resource_type} found."
    else:
        # Start with a header line
        result = [f"Found {len(items)} {resource_type}:"]
        result.append("")  # Empty line for readability
        
        # Format each item
        for item in items:
            name = item.get(name_field, "Unnamed")
            id_value = item.get("id", "No ID")
            
            # Start with ID and name
            item_line = f"• {name} (ID: {id_value})"
            result.append(item_line)
            
            # Add indented description if present
            if "description" in item and item["description"]:
                result.append(f"  Description: {item['description']}")
            
            # Add additional fields with indentation
            if include_fields:
                for field in include_fields:
                    if field in item and field not in [name_field, "id", "description"]:
                        # Format the value based on type
                        if isinstance(item[field], dict):
                            value = f"<{len(item[field])} key-value pairs>"
                        elif isinstance(item[field], list):
                            value = f"<{len(item[field])} items>"
                        else:
                            value = str(item[field])
                        
                        # Add formatted field
                        field_name = field.replace("_", " ").title()
                        result.append(f"  {field_name}: {value}")
            
            # Add empty line between items for readability
            result.append("")
        
        # Add helpful usage hint at the end
        singular_type = resource_type[:-1] if resource_type.endswith("s") else resource_type
        result.append(f"Use 'get' action with the {singular_type}_id to see more details about a specific {singular_type}.")
        
        formatted_message = "\n".join(result)
    
    # Add the formatted message for MCP display
    response["formatted_message"] = formatted_message
    
    return response


def format_list_text(items: List[Dict[str, Any]], name_field: str = "name") -> str:
    """
    Format a list of items for text display.
    
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


def get_resource_path(resource_type: str, resource_id: str, extension: str = "json") -> str:
    """
    Get the file path for a resource.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        resource_id: ID of the resource
        extension: File extension (default: "json")
        
    Returns:
        Full file path to the resource
        
    Raises:
        ValueError: If the resource type is unknown
    """
    directory_map = {
        "evaluation": EVALUATIONS_DIR,
        "dataset": DATASETS_DIR,
        "experiment": EXPERIMENTS_DIR,
        "result": RESULTS_DIR,
        "analysis": os.path.join(RESULTS_DIR, "analyses"),
    }
    
    if resource_type not in directory_map:
        error_msg = f"Unknown resource type: {resource_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    directory = directory_map[resource_type]
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    return os.path.join(directory, f"{resource_id}.{extension}")


def resource_exists(resource_type: str, resource_id: str) -> bool:
    """
    Check if a resource exists.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        resource_id: ID of the resource
        
    Returns:
        True if the resource exists, False otherwise
    """
    try:
        path = get_resource_path(resource_type, resource_id)
        return os.path.exists(path)
    except ValueError:
        return False


def save_resource(
    resource_type: str, 
    resource_id: str, 
    data: Dict[str, Any],
    update_cache: bool = True
) -> None:
    """
    Save a resource to storage.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        resource_id: ID of the resource
        data: The resource data to save
        update_cache: Whether to update the cache with this resource
        
    Raises:
        OperationError: If the save operation fails
    """
    path = get_resource_path(resource_type, resource_id)
    save_json(data, path)
    logger.info(f"Saved {resource_type} with ID {resource_id}")
    
    # Update cache if enabled
    if update_cache:
        from .cache import cache_resource
        cache_resource(resource_type, resource_id, data)


def load_resource(
    resource_type: str, 
    resource_id: str, 
    required: bool = True,
    use_cache: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Load a resource from storage.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        resource_id: ID of the resource
        required: Whether the resource must exist
        use_cache: Whether to use the cache for this lookup
        
    Returns:
        The resource data, or None if it doesn't exist and required=False
        
    Raises:
        ResourceNotFoundError: If the resource doesn't exist and required=True
        OperationError: If the load operation fails
    """
    # Check cache first if enabled
    if use_cache:
        from .cache import get_cached_resource
        cached_data = get_cached_resource(resource_type, resource_id)
        if cached_data is not None:
            return cached_data
    
    # Cache miss or caching disabled, load from disk
    path = get_resource_path(resource_type, resource_id)
    data = load_json(path, required=False)
    
    if data is None and required:
        logger.error(f"{resource_type.capitalize()} with ID {resource_id} not found")
        raise ResourceNotFoundError(
            resource_type=resource_type,
            resource_id=resource_id
        )
    
    # Cache the result if it exists and caching is enabled
    if data is not None and use_cache:
        from .cache import cache_resource
        cache_resource(resource_type, resource_id, data)
        
    return data


@functools.lru_cache(maxsize=32)
def list_resources(resource_type: str, use_cache: bool = True) -> List[str]:
    """
    List all resources of a given type.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        use_cache: Whether to use function caching for this list operation
        
    Returns:
        List of resource IDs
        
    Raises:
        OperationError: If the directory exists but cannot be read
    """
    # Implementation note: we're using Python's built-in LRU cache for the
    # entire function rather than our custom cache, since this is more
    # appropriate for the list operation which returns multiple IDs
    
    # Disable caching if requested (bypass the lru_cache decorator)
    if not use_cache:
        # Clear the cache for this function to force a refresh
        list_resources.cache_clear()
    
    try:
        directory_map = {
            "evaluation": EVALUATIONS_DIR,
            "dataset": DATASETS_DIR,
            "experiment": EXPERIMENTS_DIR,
            "result": RESULTS_DIR,
            "analysis": os.path.join(RESULTS_DIR, "analyses"),
        }
        
        if resource_type not in directory_map:
            logger.error(f"Unknown resource type: {resource_type}")
            return []
            
        directory = directory_map[resource_type]
        resource_ids = list_json_files(directory)
        
        logger.debug(f"Listed {len(resource_ids)} resources of type {resource_type}")
        return resource_ids
    except Exception as e:
        logger.error(f"Failed to list {resource_type} resources: {str(e)}")
        return []


def delete_resource(
    resource_type: str, 
    resource_id: str, 
    required: bool = True,
    clear_cache: bool = True
) -> bool:
    """
    Delete a resource from storage.
    
    Args:
        resource_type: Type of resource (e.g., "evaluation", "dataset")
        resource_id: ID of the resource
        required: Whether the resource must exist
        clear_cache: Whether to remove the resource from cache
        
    Returns:
        True if the resource was deleted, False if it didn't exist
        
    Raises:
        ResourceNotFoundError: If the resource doesn't exist and required=True
        OperationError: If the delete operation fails
    """
    path = get_resource_path(resource_type, resource_id)
    
    if not os.path.exists(path):
        if required:
            logger.error(f"{resource_type.capitalize()} with ID {resource_id} not found")
            raise ResourceNotFoundError(
                resource_type=resource_type,
                resource_id=resource_id
            )
        return False
    
    try:
        # Remove the resource from disk
        os.remove(path)
        logger.info(f"Deleted {resource_type} with ID {resource_id}")
        
        # Invalidate cache if enabled
        if clear_cache:
            from .cache import invalidate_resource
            invalidate_resource(resource_type, resource_id)
            
        return True
    except Exception as e:
        logger.error(f"Failed to delete {resource_type} with ID {resource_id}: {str(e)}")
        raise OperationError(
            operation="delete_resource",
            message=f"Failed to delete {resource_type} with ID {resource_id}: {str(e)}",
            resource_type=resource_type,
            resource_id=resource_id
        )
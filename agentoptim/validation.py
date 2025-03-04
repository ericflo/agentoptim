"""
Validation module for AgentOptim.

This module provides validation utilities to ensure consistency
in input validation across the project.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Union, TypeVar, Callable, Set

from .errors import ValidationError


T = TypeVar('T')


def validate_required(value: Optional[Any], field_name: str) -> None:
    """Validate that a required field is present and not None.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        
    Raises:
        ValidationError: If the value is None
    """
    if value is None:
        raise ValidationError(f"'{field_name}' is required", field=field_name, value=value)


def validate_string(
    value: Any, 
    field_name: str, 
    min_length: int = 1, 
    max_length: Optional[int] = None,
    required: bool = True
) -> Optional[str]:
    """Validate a string value.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        min_length: The minimum allowed length
        max_length: The maximum allowed length, if any
        required: Whether the field is required
        
    Returns:
        The validated string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    if len(value) < min_length:
        raise ValidationError(
            f"'{field_name}' must be at least {min_length} characters",
            field=field_name,
            value=value
        )
    
    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"'{field_name}' must be at most {max_length} characters",
            field=field_name,
            value=value
        )
    
    return value


def validate_number(
    value: Any, 
    field_name: str, 
    min_value: Optional[float] = None, 
    max_value: Optional[float] = None,
    required: bool = True
) -> Optional[float]:
    """Validate a numeric value.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        min_value: The minimum allowed value, if any
        max_value: The maximum allowed value, if any
        required: Whether the field is required
        
    Returns:
        The validated number, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        raise ValidationError(
            f"'{field_name}' must be a number, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    if min_value is not None and num_value < min_value:
        raise ValidationError(
            f"'{field_name}' must be at least {min_value}",
            field=field_name,
            value=value
        )
    
    if max_value is not None and num_value > max_value:
        raise ValidationError(
            f"'{field_name}' must be at most {max_value}",
            field=field_name,
            value=value
        )
    
    return num_value


def validate_bool(value: Any, field_name: str, required: bool = True) -> Optional[bool]:
    """Validate a boolean value.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required: Whether the field is required
        
    Returns:
        The validated boolean, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, bool):
        # Handle string representations
        if isinstance(value, str):
            if value.lower() in ('true', 't', 'yes', 'y', '1'):
                return True
            elif value.lower() in ('false', 'f', 'no', 'n', '0'):
                return False
        
        # Handle numeric representations
        if value == 1:
            return True
        elif value == 0:
            return False
            
        raise ValidationError(
            f"'{field_name}' must be a boolean, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    return value


def validate_list(
    value: Any, 
    field_name: str, 
    min_items: int = 0, 
    max_items: Optional[int] = None,
    item_validator: Optional[Callable[[Any, str], Any]] = None,
    required: bool = True
) -> Optional[List[Any]]:
    """Validate a list value and optionally its items.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        min_items: The minimum allowed length
        max_items: The maximum allowed length, if any
        item_validator: A function to validate each item in the list
        required: Whether the field is required
        
    Returns:
        The validated list, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, list):
        raise ValidationError(
            f"'{field_name}' must be a list, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    if len(value) < min_items:
        raise ValidationError(
            f"'{field_name}' must contain at least {min_items} items",
            field=field_name,
            value=value
        )
    
    if max_items is not None and len(value) > max_items:
        raise ValidationError(
            f"'{field_name}' must contain at most {max_items} items",
            field=field_name,
            value=value
        )
    
    # Validate individual items if validator provided
    if item_validator:
        validated_items = []
        for i, item in enumerate(value):
            try:
                validated_item = item_validator(item, f"{field_name}[{i}]")
                validated_items.append(validated_item)
            except ValidationError as e:
                # Re-raise with updated field name
                raise ValidationError(
                    f"Invalid item in '{field_name}' at index {i}: {e.message}",
                    field=f"{field_name}[{i}]",
                    value=item
                )
        
        return validated_items
    
    return value


def validate_dict(
    value: Any, 
    field_name: str, 
    required_keys: Optional[Set[str]] = None,
    optional_keys: Optional[Set[str]] = None,
    allow_extra_keys: bool = False,
    key_validators: Optional[Dict[str, Callable[[Any, str], Any]]] = None,
    required: bool = True
) -> Optional[Dict[str, Any]]:
    """Validate a dictionary value and optionally its keys and values.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required_keys: Set of keys that must be present
        optional_keys: Set of keys that may be present
        allow_extra_keys: Whether to allow keys not in required_keys or optional_keys
        key_validators: Dict mapping keys to validation functions for their values
        required: Whether the field is required
        
    Returns:
        The validated dictionary, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, dict):
        raise ValidationError(
            f"'{field_name}' must be a dictionary, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    # Check for required keys
    if required_keys:
        missing_keys = required_keys - value.keys()
        if missing_keys:
            missing_str = ", ".join(f"'{k}'" for k in missing_keys)
            raise ValidationError(
                f"'{field_name}' is missing required keys: {missing_str}",
                field=field_name,
                value=value
            )
    
    # Check for disallowed keys
    if not allow_extra_keys and (required_keys or optional_keys):
        allowed_keys = set()
        if required_keys:
            allowed_keys.update(required_keys)
        if optional_keys:
            allowed_keys.update(optional_keys)
        
        extra_keys = value.keys() - allowed_keys
        if extra_keys:
            extra_str = ", ".join(f"'{k}'" for k in extra_keys)
            raise ValidationError(
                f"'{field_name}' contains disallowed keys: {extra_str}",
                field=field_name,
                value=value
            )
    
    # Validate values with key-specific validators
    if key_validators:
        validated_dict = {}
        for key, val in value.items():
            if key in key_validators:
                try:
                    validated_val = key_validators[key](val, f"{field_name}.{key}")
                    validated_dict[key] = validated_val
                except ValidationError as e:
                    # Re-raise with updated field name
                    raise ValidationError(
                        f"Invalid value for '{field_name}.{key}': {e.message}",
                        field=f"{field_name}.{key}",
                        value=val
                    )
            else:
                validated_dict[key] = val
        
        return validated_dict
    
    return value


def validate_one_of(
    value: Any, 
    field_name: str, 
    allowed_values: List[Any],
    required: bool = True
) -> Optional[Any]:
    """Validate that a value is one of a set of allowed values.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        allowed_values: List of allowed values
        required: Whether the field is required
        
    Returns:
        The validated value, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if value not in allowed_values:
        allowed_str = ", ".join(f"'{v}'" for v in allowed_values)
        raise ValidationError(
            f"'{field_name}' must be one of: {allowed_str}",
            field=field_name,
            value=value
        )
    
    return value


def validate_uuid(value: Any, field_name: str, required: bool = True) -> Optional[str]:
    """Validate that a value is a valid UUID string.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required: Whether the field is required
        
    Returns:
        The validated UUID string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    try:
        uuid_obj = uuid.UUID(value)
        return str(uuid_obj)
    except ValueError:
        raise ValidationError(
            f"'{field_name}' must be a valid UUID",
            field=field_name,
            value=value
        )


def validate_email(value: Any, field_name: str, required: bool = True) -> Optional[str]:
    """Validate an email address.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required: Whether the field is required
        
    Returns:
        The validated email string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    # Simple email validation regex
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, value):
        raise ValidationError(
            f"'{field_name}' must be a valid email address",
            field=field_name,
            value=value
        )
    
    return value


def validate_url(value: Any, field_name: str, required: bool = True) -> Optional[str]:
    """Validate a URL.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required: Whether the field is required
        
    Returns:
        The validated URL string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    # Simple URL validation regex
    url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, value):
        raise ValidationError(
            f"'{field_name}' must be a valid URL",
            field=field_name,
            value=value
        )
    
    return value


def validate_id_format(value: Any, field_name: str, prefix: str, required: bool = True) -> Optional[str]:
    """Validate an ID with a specific format (e.g., 'eval_1234').
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        prefix: The expected prefix for the ID
        required: Whether the field is required
        
    Returns:
        The validated ID string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    # ID format validation
    id_pattern = f'^{prefix}_[a-zA-Z0-9]+$'
    if not re.match(id_pattern, value):
        raise ValidationError(
            f"'{field_name}' must start with '{prefix}_' followed by alphanumeric characters",
            field=field_name,
            value=value
        )
    
    return value


def validate_template(
    value: Any, 
    field_name: str, 
    required_variables: Optional[Set[str]] = None,
    required: bool = True
) -> Optional[str]:
    """Validate a template string, optionally checking for required variables.
    
    Args:
        value: The value to validate
        field_name: The name of the field being validated
        required_variables: Set of variable names that must be present in the template
        required: Whether the field is required
        
    Returns:
        The validated template string, or None if not required and value is None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        if required:
            raise ValidationError(f"'{field_name}' is required", field=field_name, value=None)
        return None
    
    if not isinstance(value, str):
        raise ValidationError(
            f"'{field_name}' must be a string, got {type(value).__name__}",
            field=field_name,
            value=value
        )
    
    # Check for required variables in the template
    if required_variables:
        # This is a very simple check that might not catch all template patterns
        # A more sophisticated parser could be used for proper template validation
        found_vars = set()
        for var in required_variables:
            # Check for both {{ var }} and {var} formats
            if f"{{{{ {var} }}}}" in value or f"{{{var}}}" in value:
                found_vars.add(var)
        
        missing_vars = required_variables - found_vars
        if missing_vars:
            missing_str = ", ".join(f"'{v}'" for v in missing_vars)
            raise ValidationError(
                f"'{field_name}' is missing required variables: {missing_str}",
                field=field_name,
                value=value
            )
    
    return value

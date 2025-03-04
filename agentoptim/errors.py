"""
Error handling module for AgentOptim.

This module defines custom exceptions and error handling utilities
to ensure consistent error handling and reporting across the project.
"""

import logging
import sys
from typing import Optional, Dict, Any, List, Union, Tuple

# Configure logging
logger = logging.getLogger("agentoptim")


class AgentOptimError(Exception):
    """Base exception class for all AgentOptim errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception with a message and optional details.
        
        Args:
            message: The error message
            details: Additional error context details
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
        
        # Log the error
        logger.error(f"{self.__class__.__name__}: {message}", exc_info=True, extra=self.details)


class ValidationError(AgentOptimError):
    """Exception raised for input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """Initialize validation error with field information.
        
        Args:
            message: The validation error message
            field: The name of the field that failed validation
            value: The invalid value
        """
        details = {"field": field, "value": value}
        super().__init__(message, details)


class ResourceNotFoundError(AgentOptimError):
    """Exception raised when a requested resource doesn't exist."""
    
    def __init__(self, 
                 resource_type: str, 
                 resource_id: str,
                 message: Optional[str] = None):
        """Initialize resource not found error.
        
        Args:
            resource_type: Type of resource (e.g., "dataset", "evaluation")
            resource_id: ID of the resource that wasn't found
            message: Optional custom message
        """
        self.resource_type = resource_type
        self.resource_id = resource_id
        msg = message or f"{resource_type.capitalize()} with ID '{resource_id}' not found"
        details = {"resource_type": resource_type, "resource_id": resource_id}
        super().__init__(msg, details)


class DuplicateResourceError(AgentOptimError):
    """Exception raised when attempting to create a duplicate resource."""
    
    def __init__(self, 
                 resource_type: str, 
                 identifier: Union[str, Dict[str, Any]],
                 message: Optional[str] = None):
        """Initialize duplicate resource error.
        
        Args:
            resource_type: Type of resource (e.g., "dataset", "evaluation")
            identifier: ID or other unique identifier for the duplicate resource
            message: Optional custom message
        """
        self.resource_type = resource_type
        self.identifier = identifier
        
        if isinstance(identifier, dict):
            id_str = ", ".join(f"{k}='{v}'" for k, v in identifier.items())
        else:
            id_str = f"'{identifier}'"
            
        msg = message or f"A {resource_type} with the identifier {id_str} already exists"
        details = {"resource_type": resource_type, "identifier": identifier}
        super().__init__(msg, details)


class OperationError(AgentOptimError):
    """Exception raised when an operation fails."""
    
    def __init__(self, 
                 operation: str, 
                 message: str,
                 resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None):
        """Initialize operation error.
        
        Args:
            operation: The name of the operation that failed
            message: Error message explaining the failure
            resource_type: Optional type of resource involved
            resource_id: Optional ID of resource involved
        """
        details = {
            "operation": operation,
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        super().__init__(message, details)


class ConfigurationError(AgentOptimError):
    """Exception raised for configuration errors."""
    
    def __init__(self, 
                 message: str, 
                 config_key: Optional[str] = None,
                 config_value: Optional[Any] = None):
        """Initialize configuration error.
        
        Args:
            message: Error message explaining the configuration issue
            config_key: The configuration key with the issue
            config_value: The problematic configuration value
        """
        details = {"config_key": config_key, "config_value": config_value}
        super().__init__(message, details)


class ExternalServiceError(AgentOptimError):
    """Exception raised when an external service call fails."""
    
    def __init__(self, 
                 service_name: str, 
                 message: str,
                 status_code: Optional[int] = None,
                 response: Optional[Any] = None):
        """Initialize external service error.
        
        Args:
            service_name: Name of the external service
            message: Error message explaining the failure
            status_code: HTTP status code or error code from the service
            response: Raw response from the service
        """
        details = {
            "service_name": service_name,
            "status_code": status_code,
            "response": response
        }
        super().__init__(message, details)


class JobError(AgentOptimError):
    """Exception raised for job execution errors."""
    
    def __init__(self, 
                 job_id: str, 
                 message: str,
                 task_id: Optional[str] = None):
        """Initialize job error.
        
        Args:
            job_id: ID of the job that experienced the error
            message: Error message explaining the failure
            task_id: Optional ID of the specific task that failed
        """
        details = {"job_id": job_id, "task_id": task_id}
        super().__init__(message, details)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """Configure logging for the entire application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to stderr only)
        log_format: Custom log format string
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Default format includes timestamp, level, and message
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Always add a console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # Create logger for agentoptim
    app_logger = logging.getLogger("agentoptim")
    app_logger.setLevel(numeric_level)
    
    # Log configuration info
    app_logger.info(f"Logging initialized at level {level}")
    if log_file:
        app_logger.info(f"Log file: {log_file}")


def format_error_for_response(error: Exception) -> Dict[str, Any]:
    """Format an exception for consistent API responses.
    
    Args:
        error: The exception to format
    
    Returns:
        A dictionary with error details for API response
    """
    if isinstance(error, AgentOptimError):
        response = {
            "error": True,
            "error_type": error.__class__.__name__,
            "message": error.message,
        }
        
        if hasattr(error, 'details') and error.details:
            # Clean None values and convert non-serializable objects to strings
            cleaned_details = {}
            for k, v in error.details.items():
                if v is not None:
                    try:
                        # Test JSON serialization
                        import json
                        json.dumps(v)
                        cleaned_details[k] = v
                    except (TypeError, OverflowError):
                        # Convert to string if not serializable
                        cleaned_details[k] = str(v)
            
            response["details"] = cleaned_details
    else:
        # Generic error handling for unexpected exceptions
        response = {
            "error": True,
            "error_type": error.__class__.__name__,
            "message": str(error),
        }
    
    return response
"""
Tests for the errors module.
"""

import unittest
import logging
import json
import io
import sys
from unittest.mock import patch, MagicMock
import pytest

from agentoptim.errors import (
    AgentOptimError,
    ValidationError,
    ResourceNotFoundError,
    DuplicateResourceError,
    OperationError,
    ConfigurationError,
    ExternalServiceError,
    JobError,
    setup_logging,
    format_error_for_response
)


class TestErrors(unittest.TestCase):
    """Test the errors module."""
    
    def test_agent_optim_error(self):
        """Test the base AgentOptimError class."""
        # Basic error with just a message
        error = AgentOptimError("Test error message")
        self.assertEqual("Test error message", error.message)
        self.assertEqual({}, error.details)
        self.assertEqual("Test error message", str(error))
        
        # Error with details
        details = {"param": "value", "code": 123}
        error = AgentOptimError("Test with details", details)
        self.assertEqual("Test with details", error.message)
        self.assertEqual(details, error.details)
    
    def test_validation_error(self):
        """Test the ValidationError class."""
        # Basic validation error
        error = ValidationError("Invalid value")
        self.assertEqual("Invalid value", error.message)
        self.assertEqual({"field": None, "value": None}, error.details)
        
        # Validation error with field and value
        error = ValidationError("Name too short", "name", "a")
        self.assertEqual("Name too short", error.message)
        self.assertEqual({"field": "name", "value": "a"}, error.details)
    
    def test_resource_not_found_error(self):
        """Test the ResourceNotFoundError class."""
        # Basic resource not found error
        error = ResourceNotFoundError("dataset", "data_123")
        self.assertEqual("Dataset with ID 'data_123' not found", error.message)
        self.assertEqual("dataset", error.resource_type)
        self.assertEqual("data_123", error.resource_id)
        
        # With custom message
        error = ResourceNotFoundError("experiment", "exp_456", "Could not locate experiment")
        self.assertEqual("Could not locate experiment", error.message)
        self.assertEqual("experiment", error.resource_type)
        self.assertEqual("exp_456", error.resource_id)
    
    def test_duplicate_resource_error(self):
        """Test the DuplicateResourceError class."""
        # With string identifier
        error = DuplicateResourceError("dataset", "data_123")
        self.assertEqual("A dataset with the identifier 'data_123' already exists", error.message)
        self.assertEqual("dataset", error.resource_type)
        self.assertEqual("data_123", error.identifier)
        
        # With dict identifier
        identifier = {"name": "test", "owner": "user1"}
        error = DuplicateResourceError("dataset", identifier)
        self.assertIn("A dataset with the identifier", error.message)
        self.assertIn("name='test'", error.message)
        self.assertIn("owner='user1'", error.message)
        self.assertEqual(identifier, error.identifier)
        
        # With custom message
        error = DuplicateResourceError("experiment", "exp_456", "Experiment already exists")
        self.assertEqual("Experiment already exists", error.message)
    
    def test_operation_error(self):
        """Test the OperationError class."""
        # Basic operation error
        error = OperationError("create_dataset", "Failed to create dataset")
        self.assertEqual("Failed to create dataset", error.message)
        self.assertEqual({
            "operation": "create_dataset",
            "resource_type": None,
            "resource_id": None
        }, error.details)
        
        # With resource information
        error = OperationError(
            "delete_dataset", 
            "Failed to delete dataset",
            "dataset",
            "data_123"
        )
        self.assertEqual("Failed to delete dataset", error.message)
        self.assertEqual({
            "operation": "delete_dataset",
            "resource_type": "dataset",
            "resource_id": "data_123"
        }, error.details)
    
    def test_configuration_error(self):
        """Test the ConfigurationError class."""
        # Basic configuration error
        error = ConfigurationError("Invalid configuration")
        self.assertEqual("Invalid configuration", error.message)
        self.assertEqual({
            "config_key": None,
            "config_value": None
        }, error.details)
        
        # With config details
        error = ConfigurationError(
            "Invalid log level", 
            "log_level",
            "INVALID"
        )
        self.assertEqual("Invalid log level", error.message)
        self.assertEqual({
            "config_key": "log_level",
            "config_value": "INVALID"
        }, error.details)
    
    def test_external_service_error(self):
        """Test the ExternalServiceError class."""
        # Basic external service error
        error = ExternalServiceError("API", "Request failed")
        self.assertEqual("Request failed", error.message)
        self.assertEqual({
            "service_name": "API",
            "status_code": None,
            "response": None
        }, error.details)
        
        # With status code and response
        response = {"error": "Not found"}
        error = ExternalServiceError(
            "OpenAI", 
            "API request failed",
            404,
            response
        )
        self.assertEqual("API request failed", error.message)
        self.assertEqual({
            "service_name": "OpenAI",
            "status_code": 404,
            "response": response
        }, error.details)
    
    def test_job_error(self):
        """Test the JobError class."""
        # Basic job error
        error = JobError("job_123", "Job failed")
        self.assertEqual("Job failed", error.message)
        self.assertEqual({
            "job_id": "job_123",
            "task_id": None
        }, error.details)
        
        # With task ID
        error = JobError(
            "job_123", 
            "Task failed",
            "task_456"
        )
        self.assertEqual("Task failed", error.message)
        self.assertEqual({
            "job_id": "job_123",
            "task_id": "task_456"
        }, error.details)
    
    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        with patch('logging.getLogger') as mock_get_logger, \
             patch('logging.StreamHandler') as mock_stream_handler, \
             patch('logging.Formatter') as mock_formatter, \
             patch('logging.FileHandler') as mock_file_handler:
            
            # Create mock objects for the test
            mock_root_logger = MagicMock()
            mock_app_logger = MagicMock()
            mock_get_logger.side_effect = [mock_root_logger, mock_app_logger]
            
            mock_console = MagicMock()
            mock_stream_handler.return_value = mock_console
            
            # Call the function to test
            setup_logging()
            
            # Verify logging was set up correctly
            mock_get_logger.assert_any_call()  # root logger
            mock_get_logger.assert_any_call("agentoptim")  # app logger
            
            mock_root_logger.setLevel.assert_called_once()
            mock_app_logger.setLevel.assert_called_once()
            
            mock_formatter.assert_called_once()
            mock_console.setLevel.assert_called_once()
            mock_console.setFormatter.assert_called_once()
            mock_root_logger.addHandler.assert_called_once_with(mock_console)
            
            # No file handler should be created with default settings
            mock_file_handler.assert_not_called()
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file output."""
        with patch('logging.getLogger') as mock_get_logger, \
             patch('logging.StreamHandler') as mock_stream_handler, \
             patch('logging.Formatter') as mock_formatter, \
             patch('logging.FileHandler') as mock_file_handler:
            
            # Create mock objects for the test
            mock_root_logger = MagicMock()
            mock_app_logger = MagicMock()
            mock_get_logger.side_effect = [mock_root_logger, mock_app_logger]
            
            mock_console = MagicMock()
            mock_file = MagicMock()
            mock_stream_handler.return_value = mock_console
            mock_file_handler.return_value = mock_file
            
            # Call the function to test
            setup_logging(level="DEBUG", log_file="/path/to/log.txt")
            
            # Verify file handler was created
            mock_file_handler.assert_called_once_with("/path/to/log.txt")
            mock_file.setLevel.assert_called_once()
            mock_file.setFormatter.assert_called_once()
            
            # Both handlers should be added
            self.assertEqual(2, mock_root_logger.addHandler.call_count)
    
    def test_setup_logging_invalid_level(self):
        """Test setup_logging with an invalid log level."""
        with pytest.raises(ValueError) as excinfo:
            setup_logging(level="INVALID_LEVEL")
        assert "Invalid log level" in str(excinfo.value)
    
    def test_setup_logging_custom_format(self):
        """Test setup_logging with a custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        
        with patch('logging.getLogger') as mock_get_logger, \
             patch('logging.StreamHandler') as mock_stream_handler, \
             patch('logging.Formatter') as mock_formatter, \
             patch('logging.FileHandler') as mock_file_handler:
            
            # Create mock objects for the test
            mock_root_logger = MagicMock()
            mock_app_logger = MagicMock()
            mock_get_logger.side_effect = [mock_root_logger, mock_app_logger]
            
            # Call the function to test
            setup_logging(log_format=custom_format)
            
            # Verify custom format was used
            mock_formatter.assert_called_once_with(custom_format)
    
    def test_format_error_for_response_agent_optim_error(self):
        """Test format_error_for_response with AgentOptimError."""
        # Create a test error
        details = {"resource_id": "123", "extra": "value"}
        error = AgentOptimError("Test error", details)
        
        # Format the error
        result = format_error_for_response(error)
        
        # Verify the result
        self.assertTrue(result["error"])
        self.assertEqual("AgentOptimError", result["error_type"])
        self.assertEqual("Test error", result["message"])
        self.assertEqual(details, result["details"])
    
    def test_format_error_for_response_standard_exception(self):
        """Test format_error_for_response with standard Python exception."""
        # Create a standard exception
        error = ValueError("Invalid value")
        
        # Format the error
        result = format_error_for_response(error)
        
        # Verify the result
        self.assertTrue(result["error"])
        self.assertEqual("ValueError", result["error_type"])
        self.assertEqual("Invalid value", result["message"])
        self.assertNotIn("details", result)
    
    def test_format_error_for_response_non_serializable(self):
        """Test format_error_for_response with non-JSON-serializable values."""
        # Create a custom object that's not JSON serializable
        class CustomObj:
            def __str__(self):
                return "CustomObj instance"
        
        # Create an error with the custom object in details
        details = {"obj": CustomObj(), "normal": "value"}
        error = AgentOptimError("Test error", details)
        
        # Format the error
        result = format_error_for_response(error)
        
        # Verify the result
        self.assertTrue(result["error"])
        self.assertEqual("Test error", result["message"])
        self.assertEqual("CustomObj instance", result["details"]["obj"])
        self.assertEqual("value", result["details"]["normal"])
        
        # Make sure the resulting object is JSON serializable
        try:
            json.dumps(result)
        except (TypeError, OverflowError):
            self.fail("Result should be JSON serializable")


if __name__ == "__main__":
    unittest.main()
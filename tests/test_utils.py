"""
Tests for the utils module.
"""

import os
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

from agentoptim.utils import (
    DATA_DIR,
    get_data_path,
    get_data_dir,
    ensure_data_directories,
    generate_id,
    save_json,
    load_json,
    list_json_files,
    validate_action,
    validate_required_params,
    format_error,
    format_success,
    format_list,
    format_list_text,
    get_resource_path,
    resource_exists,
    save_resource,
    load_resource,
    list_resources,
    delete_resource
)
from agentoptim.errors import ValidationError, ResourceNotFoundError, OperationError


class TestUtils:
    """Test the utils module functions."""
    
    def setup_method(self):
        """Set up before each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {"name": "Test", "id": "test_123"}
    
    def teardown_method(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_data_path(self):
        """Test the get_data_path function."""
        # Test without subfolder
        path = get_data_path()
        assert path == DATA_DIR
        
        # Test with subfolder
        with patch('os.makedirs') as mock_makedirs:
            path = get_data_path("test_folder")
            assert path == os.path.join(DATA_DIR, "test_folder")
            mock_makedirs.assert_called_once()
    
    def test_get_data_dir(self):
        """Test the get_data_dir function."""
        assert get_data_dir() == DATA_DIR
    
    def test_ensure_data_directories(self):
        """Test the ensure_data_directories function."""
        with patch('os.makedirs') as mock_makedirs:
            ensure_data_directories()
            assert mock_makedirs.call_count >= 5  # DATA_DIR and at least 4 subdirectories
    
    def test_ensure_data_directories_error(self):
        """Test error handling in ensure_data_directories."""
        with patch('os.makedirs') as mock_makedirs:
            mock_makedirs.side_effect = PermissionError("Permission denied")
            with pytest.raises(OperationError) as excinfo:
                ensure_data_directories()
            assert "Failed to create data directory" in str(excinfo.value)
    
    def test_generate_id(self):
        """Test the generate_id function."""
        id1 = generate_id()
        id2 = generate_id()
        
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2  # IDs should be unique
    
    def test_save_json(self):
        """Test the save_json function."""
        filepath = os.path.join(self.temp_dir, "test.json")
        data = {"name": "Test", "value": 123}
        
        save_json(data, filepath)
        
        # Verify file was created and contains the correct data
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data == data
    
    def test_save_json_directory_creation(self):
        """Test that save_json creates parent directories if needed."""
        filepath = os.path.join(self.temp_dir, "nested", "dir", "test.json")
        data = {"name": "Test", "value": 123}
        
        save_json(data, filepath)
        
        # Verify file was created and contains the correct data
        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data == data
    
    def test_save_json_error(self):
        """Test error handling in save_json."""
        filepath = os.path.join(self.temp_dir, "test.json")
        data = {"name": "Test", "value": 123}
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            with pytest.raises(OperationError) as excinfo:
                save_json(data, filepath)
            assert "Failed to save data" in str(excinfo.value)
    
    def test_load_json(self):
        """Test the load_json function."""
        filepath = os.path.join(self.temp_dir, "test.json")
        data = {"name": "Test", "value": 123}
        
        # Save the file first
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        # Test loading
        loaded_data = load_json(filepath)
        assert loaded_data == data
    
    def test_load_json_not_required(self):
        """Test load_json with non-existent file and required=False."""
        filepath = os.path.join(self.temp_dir, "nonexistent.json")
        result = load_json(filepath, required=False)
        assert result is None
    
    def test_load_json_required(self):
        """Test load_json with non-existent file and required=True."""
        filepath = os.path.join(self.temp_dir, "nonexistent.json")
        with pytest.raises(ResourceNotFoundError):
            load_json(filepath, required=True)
    
    def test_load_json_invalid_json(self):
        """Test load_json with invalid JSON content."""
        filepath = os.path.join(self.temp_dir, "invalid.json")
        
        # Create a file with invalid JSON
        with open(filepath, 'w') as f:
            f.write("{not valid json")
        
        with pytest.raises(OperationError) as excinfo:
            load_json(filepath)
        assert "parse JSON" in str(excinfo.value)
    
    def test_load_json_error(self):
        """Test general error handling in load_json."""
        filepath = os.path.join(self.temp_dir, "test.json")
        data = {"name": "Test", "value": 123}
        
        # Save the file first
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            with pytest.raises(OperationError) as excinfo:
                load_json(filepath)
            assert "Failed to load data" in str(excinfo.value)
    
    def test_list_json_files(self):
        """Test the list_json_files function."""
        # Create test files
        os.makedirs(os.path.join(self.temp_dir, "json_test"), exist_ok=True)
        test_dir = os.path.join(self.temp_dir, "json_test")
        
        with open(os.path.join(test_dir, "file1.json"), 'w') as f:
            f.write("{}")
        with open(os.path.join(test_dir, "file2.json"), 'w') as f:
            f.write("{}")
        with open(os.path.join(test_dir, "not_json.txt"), 'w') as f:
            f.write("text")
        
        # Test listing
        files = list_json_files(test_dir)
        assert len(files) == 2
        assert "file1" in files
        assert "file2" in files
        assert "not_json" not in files
    
    def test_list_json_files_nonexistent_dir(self):
        """Test list_json_files with a non-existent directory."""
        directory = os.path.join(self.temp_dir, "nonexistent")
        files = list_json_files(directory)
        assert files == []
    
    def test_list_json_files_permission_error(self):
        """Test list_json_files with permission error."""
        with patch('os.listdir') as mock_listdir:
            mock_listdir.side_effect = PermissionError("Permission denied")
            with pytest.raises(OperationError) as excinfo:
                list_json_files(self.temp_dir)
            assert "Permission denied" in str(excinfo.value)
    
    def test_list_json_files_other_error(self):
        """Test list_json_files with other error."""
        with patch('os.listdir') as mock_listdir:
            mock_listdir.side_effect = Exception("Unexpected error")
            with pytest.raises(OperationError) as excinfo:
                list_json_files(self.temp_dir)
            assert "Failed to list files" in str(excinfo.value)
    
    def test_validate_action(self):
        """Test the validate_action function."""
        valid_actions = ["create", "list", "get", "update", "delete"]
        
        # Valid action should not raise
        validate_action("create", valid_actions)
        
        # Invalid action should raise ValidationError
        with pytest.raises(ValidationError) as excinfo:
            validate_action("invalid", valid_actions)
        assert "Invalid action" in str(excinfo.value)
    
    def test_validate_required_params(self):
        """Test the validate_required_params function."""
        params = {"name": "Test", "id": "test_123", "optional": None}
        
        # All required params present
        validate_required_params(params, ["name", "id"])
        
        # Missing required param
        with pytest.raises(ValidationError) as excinfo:
            validate_required_params(params, ["name", "id", "missing"])
        assert "Missing required parameters" in str(excinfo.value)
        assert "missing" in str(excinfo.value)
        
        # None value for required param
        with pytest.raises(ValidationError) as excinfo:
            validate_required_params(params, ["name", "optional"])
        assert "Missing required parameters" in str(excinfo.value)
        assert "optional" in str(excinfo.value)
    
    def test_format_error(self):
        """Test the format_error function."""
        message = "Something went wrong"
        result = format_error(message)
        
        assert result["error"] is True
        assert result["message"] == message
    
    def test_format_success(self):
        """Test the format_success function."""
        message = "Operation completed successfully"
        
        # Without data
        result = format_success(message)
        assert result["error"] is False
        assert result["message"] == message
        assert "data" not in result
        
        # With data
        data = {"id": "123", "name": "Test"}
        result = format_success(message, data)
        assert result["error"] is False
        assert result["message"] == message
        assert result["data"] == data
    
    def test_format_list(self):
        """Test the format_list function."""
        items = [
            {"id": "1", "name": "Item 1", "description": "First item", "extra": "value"},
            {"id": "2", "name": "Item 2", "extra": "another value"},
        ]
        
        # Basic test
        result = format_list(items)
        assert result["error"] is False
        assert "Found 2 items" in result["message"]
        assert len(result["items"]) == 2
        assert result["count"] == 2
        assert "extra" not in result["items"][0]
        
        # With include_fields
        result = format_list(items, include_fields=["extra"])
        assert "extra" in result["items"][0]
        
        # Empty list
        result = format_list([])
        assert result["error"] is False
        assert "No items found" in result["message"]
        assert result["items"] == []
        assert result["count"] == 0
    
    def test_format_list_text(self):
        """Test the format_list_text function."""
        items = [
            {"id": "1", "name": "Item 1", "description": "First item"},
            {"id": "2", "name": "Item 2"},
        ]
        
        result = format_list_text(items)
        assert "Item 1 (ID: 1): First item" in result
        assert "Item 2 (ID: 2)" in result
        
        # Empty list
        result = format_list_text([])
        assert result == "No items found."
    
    def test_get_resource_path(self):
        """Test the get_resource_path function."""
        # Test known resource types
        for resource_type in ["evaluation", "dataset", "experiment", "result"]:
            path = get_resource_path(resource_type, "test_123")
            assert f"{resource_type}s" in path  # Directories include an 's' at the end
            assert "test_123.json" in path
        
        # Special case for analysis which is in the "analyses" directory under results
        analysis_path = get_resource_path("analysis", "test_123")
        assert "analyses" in analysis_path
        assert "results" in analysis_path
        assert "test_123.json" in analysis_path
        
        # Test custom extension
        path = get_resource_path("evaluation", "test_123", extension="xml")
        assert "test_123.xml" in path
        
        # Test unknown resource type
        with pytest.raises(ValueError):
            get_resource_path("unknown", "test_123")
    
    def test_resource_exists(self):
        """Test the resource_exists function."""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            assert resource_exists("evaluation", "test_123") is True
            
            mock_exists.return_value = False
            assert resource_exists("evaluation", "test_123") is False
            
            # Test with unknown resource type
            assert resource_exists("unknown", "test_123") is False
    
    def test_save_resource(self):
        """Test the save_resource function."""
        data = {"id": "test_123", "name": "Test Resource"}
        
        with patch('agentoptim.utils.save_json') as mock_save_json, \
             patch('agentoptim.cache.cache_resource') as mock_cache_resource:
                
            # Test with cache update
            save_resource("evaluation", "test_123", data)
            mock_save_json.assert_called_once()
            mock_cache_resource.assert_called_once_with("evaluation", "test_123", data)
            
            # Reset mocks
            mock_save_json.reset_mock()
            mock_cache_resource.reset_mock()
            
            # Test without cache update
            save_resource("evaluation", "test_123", data, update_cache=False)
            mock_save_json.assert_called_once()
            mock_cache_resource.assert_not_called()
    
    def test_load_resource(self):
        """Test the load_resource function."""
        data = {"id": "test_123", "name": "Test Resource"}
        
        with patch('agentoptim.cache.get_cached_resource') as mock_get_cached, \
             patch('agentoptim.utils.load_json') as mock_load_json, \
             patch('agentoptim.cache.cache_resource') as mock_cache_resource:
            
            # Test cache hit
            mock_get_cached.return_value = data
            result = load_resource("evaluation", "test_123")
            assert result == data
            mock_get_cached.assert_called_once()
            mock_load_json.assert_not_called()
            
            # Reset mocks
            mock_get_cached.reset_mock()
            mock_load_json.reset_mock()
            
            # Test cache miss, file found
            mock_get_cached.return_value = None
            mock_load_json.return_value = data
            result = load_resource("evaluation", "test_123")
            assert result == data
            mock_get_cached.assert_called_once()
            mock_load_json.assert_called_once()
            mock_cache_resource.assert_called_once()
            
            # Reset mocks
            mock_get_cached.reset_mock()
            mock_load_json.reset_mock()
            mock_cache_resource.reset_mock()
            
            # Test cache miss, file not found, not required
            mock_get_cached.return_value = None
            mock_load_json.return_value = None
            result = load_resource("evaluation", "test_123", required=False)
            assert result is None
            mock_get_cached.assert_called_once()
            mock_load_json.assert_called_once()
            mock_cache_resource.assert_not_called()
            
            # Reset mocks
            mock_get_cached.reset_mock()
            mock_load_json.reset_mock()
            
            # Test cache miss, file not found, required
            mock_get_cached.return_value = None
            mock_load_json.return_value = None
            with pytest.raises(ResourceNotFoundError):
                load_resource("evaluation", "test_123", required=True)
            
            # Test no cache usage
            mock_get_cached.reset_mock()
            mock_load_json.reset_mock()
            mock_load_json.return_value = data
            result = load_resource("evaluation", "test_123", use_cache=False)
            assert result == data
            mock_get_cached.assert_not_called()
            mock_load_json.assert_called_once()
    
    def test_list_resources(self):
        """Test the list_resources function."""
        # Clear the LRU cache to ensure clean state
        list_resources.cache_clear()
        
        with patch('agentoptim.utils.list_json_files') as mock_list_json:
            mock_list_json.return_value = ["resource1", "resource2"]
            
            # Test with valid resource type
            resources = list_resources("evaluation")
            assert resources == ["resource1", "resource2"]
            
            # Test with invalid resource type
            resources = list_resources("unknown")
            assert resources == []
            
            # Reset mock for further testing
            mock_list_json.reset_mock()
            mock_list_json.side_effect = None
            
            # Test caching (should use cached results without calling list_json_files again)
            resources = list_resources("evaluation")
            assert resources == ["resource1", "resource2"]
            mock_list_json.assert_not_called()
            
            # Reset for exception test
            list_resources.cache_clear()
            mock_list_json.side_effect = Exception("Error")
            resources = list_resources("evaluation")
            assert resources == []
    
    def test_delete_resource(self):
        """Test the delete_resource function."""
        with patch('os.path.exists') as mock_exists, \
             patch('os.remove') as mock_remove, \
             patch('agentoptim.cache.invalidate_resource') as mock_invalidate:
            
            # Test resource exists
            mock_exists.return_value = True
            result = delete_resource("evaluation", "test_123")
            assert result is True
            mock_remove.assert_called_once()
            mock_invalidate.assert_called_once()
            
            # Reset mocks
            mock_exists.reset_mock()
            mock_remove.reset_mock()
            mock_invalidate.reset_mock()
            
            # Test resource doesn't exist, not required
            mock_exists.return_value = False
            result = delete_resource("evaluation", "test_123", required=False)
            assert result is False
            mock_remove.assert_not_called()
            mock_invalidate.assert_not_called()
            
            # Test resource doesn't exist, required
            mock_exists.return_value = False
            with pytest.raises(ResourceNotFoundError):
                delete_resource("evaluation", "test_123", required=True)
            
            # Test deletion error
            mock_exists.return_value = True
            mock_remove.side_effect = PermissionError("Permission denied")
            with pytest.raises(OperationError) as excinfo:
                delete_resource("evaluation", "test_123")
            assert "Failed to delete" in str(excinfo.value)
            
            # Test without cache clearing
            mock_exists.return_value = True
            mock_remove.reset_mock()  # Reset the mock to clear its history
            mock_remove.side_effect = None
            mock_invalidate.reset_mock()
            result = delete_resource("evaluation", "test_123", clear_cache=False)
            assert result is True
            assert mock_remove.call_count == 1  # Use call_count instead of assert_called_once
            mock_invalidate.assert_not_called()
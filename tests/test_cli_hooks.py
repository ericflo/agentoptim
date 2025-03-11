"""Tests for the CLI hooks module."""

import pytest
from unittest.mock import MagicMock, patch

from agentoptim.cli_hooks import (
    register_cli_extension,
    load_builtin_extensions,
    apply_extensions
)

def test_register_cli_extension():
    """Test registering a CLI extension."""
    # Clear the registry
    from agentoptim.cli_hooks import _CLI_EXTENSION_HOOKS
    _CLI_EXTENSION_HOOKS.clear()
    
    # Create a mock hook
    mock_hook = MagicMock(return_value=True)
    
    # Register the hook
    register_cli_extension("test", mock_hook)
    
    # Verify the hook was registered
    assert "test" in _CLI_EXTENSION_HOOKS
    assert _CLI_EXTENSION_HOOKS["test"] == mock_hook

def test_apply_extensions():
    """Test applying CLI extensions."""
    # Clear the registry
    from agentoptim.cli_hooks import _CLI_EXTENSION_HOOKS
    _CLI_EXTENSION_HOOKS.clear()
    
    # Create mock hooks
    mock_hook1 = MagicMock(return_value=True)
    mock_hook2 = MagicMock(return_value=False)
    mock_hook3 = MagicMock(side_effect=Exception("Test error"))
    
    # Register the hooks
    register_cli_extension("test1", mock_hook1)
    register_cli_extension("test2", mock_hook2)
    register_cli_extension("test3", mock_hook3)
    
    # Create a mock subparsers object
    mock_subparsers = MagicMock()
    
    # Apply the extensions
    applied = apply_extensions(mock_subparsers)
    
    # Verify the hooks were called
    mock_hook1.assert_called_once_with(mock_subparsers)
    mock_hook2.assert_called_once_with(mock_subparsers)
    mock_hook3.assert_called_once_with(mock_subparsers)
    
    # Verify only the successful hook was returned
    assert "test1" in applied
    assert "test2" not in applied
    assert "test3" not in applied

def test_load_builtin_extensions():
    """Test loading builtin extensions."""
    # Clear the registry
    from agentoptim.cli_hooks import _CLI_EXTENSION_HOOKS
    _CLI_EXTENSION_HOOKS.clear()
    
    # Mock the importlib.import_module function
    with patch("agentoptim.cli_hooks.importlib.import_module") as mock_import:
        # Create a mock sysopt module
        mock_sysopt = MagicMock()
        mock_sysopt.register_cli_commands = MagicMock()
        
        # Configure the mock to return our mock module
        mock_import.return_value = mock_sysopt
        
        # Call load_builtin_extensions
        load_builtin_extensions()
        
        # Verify the module was imported
        mock_import.assert_called_with("agentoptim.sysopt")
        
        # Verify the hook was registered
        assert "sysopt" in _CLI_EXTENSION_HOOKS
        assert _CLI_EXTENSION_HOOKS["sysopt"] == mock_sysopt.register_cli_commands
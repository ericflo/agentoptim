"""Tests for the system message optimization CLI hooks."""

import pytest
from unittest.mock import MagicMock, patch

from agentoptim.sysopt.hooks import (
    register_cli_commands,
    handle_optimize_command
)

def test_register_cli_commands():
    """Test registering CLI commands."""
    # Create a mock subparsers object
    mock_subparsers = MagicMock()
    
    # Create a mock optimize parser
    mock_optimize_parser = MagicMock()
    
    # Mock the sysopt_cli module
    with patch("importlib.import_module") as mock_import:
        # Create a mock sysopt_cli module
        mock_sysopt_cli = MagicMock()
        mock_sysopt_cli.optimize_setup_parser = MagicMock(return_value=mock_optimize_parser)
        
        # Configure the mock to return our mock module
        mock_import.return_value = mock_sysopt_cli
        
        # Call register_cli_commands
        result = register_cli_commands(mock_subparsers)
        
        # Verify the result
        assert result is True
        
        # Verify the module was imported
        mock_import.assert_called_with("agentoptim.sysopt_cli")
        
        # Verify the optimize_setup_parser was called
        mock_sysopt_cli.optimize_setup_parser.assert_called_once_with(mock_subparsers)
        
        # Verify set_defaults was called
        mock_optimize_parser.set_defaults.assert_called_once()

@pytest.mark.asyncio
async def test_handle_optimize_command():
    """Test handling optimize command."""
    # Create mock args
    mock_args = MagicMock()
    
    # Mock the asyncio module
    with patch("asyncio.get_event_loop") as mock_get_loop, \
         patch("asyncio._get_running_loop") as mock_get_running_loop, \
         patch("asyncio.run") as mock_run, \
         patch("importlib.import_module") as mock_import:
        
        # Create a mock loop
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop
        
        # Create a mock sysopt_cli module
        mock_sysopt_cli = MagicMock()
        mock_sysopt_cli.handle_optimize = MagicMock(return_value=0)
        
        # Configure the mocks
        mock_import.return_value = mock_sysopt_cli
        
        # Test when we're already in an async context
        mock_get_running_loop.return_value = MagicMock()
        
        # Call handle_optimize_command
        result = handle_optimize_command(mock_args)
        
        # Verify the result
        assert result == 0
        
        # Verify the module was imported
        mock_import.assert_called_with("agentoptim.sysopt_cli")
        
        # Verify handle_optimize was called
        mock_sysopt_cli.handle_optimize.assert_called_once_with(mock_args)
        
        # Verify run_until_complete was called
        mock_loop.run_until_complete.assert_called_once()
        
        # Reset mocks
        mock_import.reset_mock()
        mock_sysopt_cli.handle_optimize.reset_mock()
        mock_loop.run_until_complete.reset_mock()
        
        # Test when we're not in an async context
        mock_get_running_loop.return_value = None
        
        # Call handle_optimize_command
        result = handle_optimize_command(mock_args)
        
        # Verify the result
        assert result == 0
        
        # Verify the module was imported
        mock_import.assert_called_with("agentoptim.sysopt_cli")
        
        # Verify handle_optimize was called indirectly via asyncio.run
        assert mock_run.called
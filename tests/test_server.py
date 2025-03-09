"""
Tests for the MCP server implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, Mock

from agentoptim.server import (
    manage_evalset_tool,
    manage_eval_runs_tool,
    main
)


@pytest.mark.asyncio
async def test_manage_evalset_tool():
    """Test the manage_evalset_tool endpoint."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {"status": "success", "message": "Success"}
        
        result = await manage_evalset_tool(
            action="list"
        )
        
        mock_manage.assert_called_once_with(
            action="list",
            evalset_id=None,
            name=None,
            questions=None,
            short_description=None,
            long_description=None
        )
        
        assert result == {"status": "success", "message": "Success"}


@pytest.mark.asyncio
async def test_manage_evalset_tool_create():
    """Test the manage_evalset_tool with create action."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {
            "status": "success", 
            "evalset": {
                "id": "test-id",
                "name": "Test EvalSet",
                "questions": ["Q1", "Q2"]
            }
        }
        
        result = await manage_evalset_tool(
            action="create",
            name="Test EvalSet",
            questions=["Q1", "Q2"],
            short_description="Short description",
            long_description="Long description"
        )
        
        mock_manage.assert_called_once_with(
            action="create",
            evalset_id=None,
            name="Test EvalSet",
            questions=["Q1", "Q2"],
            short_description="Short description",
            long_description="Long description"
        )
        
        assert result["status"] == "success"
        assert "evalset" in result
        assert result["evalset"]["id"] == "test-id"


@pytest.mark.asyncio
async def test_manage_evalset_tool_get():
    """Test the manage_evalset_tool with get action."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {
            "status": "success", 
            "evalset": {
                "id": "test-id",
                "name": "Test EvalSet",
                "questions": ["Q1", "Q2"]
            }
        }
        
        result = await manage_evalset_tool(
            action="get",
            evalset_id="test-id"
        )
        
        mock_manage.assert_called_once_with(
            action="get",
            evalset_id="test-id",
            name=None,
            questions=None,
            short_description=None,
            long_description=None
        )
        
        assert result["status"] == "success"
        assert "evalset" in result
        assert result["evalset"]["id"] == "test-id"


@pytest.mark.asyncio
async def test_manage_evalset_tool_update():
    """Test the manage_evalset_tool with update action."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {
            "status": "success", 
            "evalset": {
                "id": "test-id",
                "name": "Updated EvalSet",
                "questions": ["Q1", "Q2", "Q3"]
            }
        }
        
        result = await manage_evalset_tool(
            action="update",
            evalset_id="test-id",
            name="Updated EvalSet",
            questions=["Q1", "Q2", "Q3"]
        )
        
        mock_manage.assert_called_once_with(
            action="update",
            evalset_id="test-id",
            name="Updated EvalSet",
            questions=["Q1", "Q2", "Q3"],
            short_description=None,
            long_description=None
        )
        
        assert result["status"] == "success"
        assert "evalset" in result
        assert result["evalset"]["name"] == "Updated EvalSet"
        assert len(result["evalset"]["questions"]) == 3


@pytest.mark.asyncio
async def test_manage_evalset_tool_delete():
    """Test the manage_evalset_tool with delete action."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {
            "status": "success", 
            "message": "EvalSet deleted successfully"
        }
        
        result = await manage_evalset_tool(
            action="delete",
            evalset_id="test-id"
        )
        
        mock_manage.assert_called_once_with(
            action="delete",
            evalset_id="test-id",
            name=None,
            questions=None,
            short_description=None,
            long_description=None
        )
        
        assert result["status"] == "success"
        assert "message" in result
        assert "deleted" in result["message"]


@pytest.mark.asyncio
async def test_manage_evalset_tool_formatted_message():
    """Test the manage_evalset_tool handling of formatted_message."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = {
            "status": "success", 
            "formatted_message": "# Formatted Message\n\nThis is a formatted message."
        }
        
        result = await manage_evalset_tool(
            action="list"
        )
        
        assert "result" in result
        assert result["result"] == "# Formatted Message\n\nThis is a formatted message."


@pytest.mark.asyncio
async def test_manage_evalset_tool_string_result():
    """Test the manage_evalset_tool handling of string result."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.return_value = "Plain text result"
        
        result = await manage_evalset_tool(
            action="list"
        )
        
        assert "result" in result
        assert result["result"] == "Plain text result"


@pytest.mark.asyncio
async def test_manage_evalset_tool_error():
    """Test the manage_evalset_tool endpoint error handling."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid input")
        
        result = await manage_evalset_tool(
            action="invalid"
        )
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "Invalid input" in result["error"]


@pytest.mark.asyncio
async def test_manage_evalset_tool_error_action():
    """Test the manage_evalset_tool error handling for invalid action."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid action")
        
        result = await manage_evalset_tool(
            action="invalid"
        )
        
        assert "error" in result
        assert "details" in result
        assert "Valid actions" in result["details"]
        assert "examples" in result


@pytest.mark.asyncio
async def test_manage_evalset_tool_error_questions():
    """Test the manage_evalset_tool error handling for invalid questions."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.side_effect = ValueError("questions must be list_type")
        
        result = await manage_evalset_tool(
            action="create",
            name="Test",
            questions="This is not a list"
        )
        
        assert "error" in result
        assert "details" in result
        assert "questions" in result["details"]
        assert "troubleshooting" in result


@pytest.mark.asyncio
async def test_manage_evalset_tool_error_required_params():
    """Test the manage_evalset_tool error handling for missing required parameters."""
    with patch('agentoptim.server.manage_evalset') as mock_manage:
        mock_manage.side_effect = ValueError("Missing required parameters")
        
        # Test for create action
        result = await manage_evalset_tool(
            action="create"
        )
        
        assert "error" in result
        assert "details" in result
        assert "example" in result
        
        # Test for get/update/delete actions
        for action in ["get", "update", "delete"]:
            result = await manage_evalset_tool(
                action=action
            )
            
            assert "error" in result
            assert "details" in result
            assert "example" in result


























def test_main():
    """Test the main function."""
    with patch('agentoptim.server.mcp.run') as mock_run:
        main()
        mock_run.assert_called_once_with(transport="stdio")
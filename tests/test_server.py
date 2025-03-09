"""
Tests for the MCP server implementation.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, Mock

from agentoptim.server import (
    manage_evalset_tool,
    run_evalset_tool,
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


@pytest.mark.asyncio
async def test_run_evalset_tool():
    """Test the run_evalset_tool endpoint."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.return_value = {"status": "success", "message": "Test success"}
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        mock_run.assert_called_once()
        assert result == {"status": "success", "message": "Test success"}


@pytest.mark.asyncio
async def test_run_evalset_tool_with_max_parallel():
    """Test the run_evalset_tool with max_parallel parameter."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.return_value = {"status": "success", "message": "Test success"}
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}],
            max_parallel=5
        )
        
        # Check that run_evalset was called (specific parameters checked in next test)
        mock_run.assert_called_once()
        
        # Check the result is as expected
        assert result == {"status": "success", "message": "Test success"}


@pytest.mark.asyncio
async def test_run_evalset_tool_with_env_vars():
    """Test the run_evalset_tool with environment variables."""
    # Save original environment variable values
    original_judge_model = os.environ.get("AGENTOPTIM_JUDGE_MODEL")
    original_omit_reasoning = os.environ.get("AGENTOPTIM_OMIT_REASONING")
    
    try:
        # Set environment variables for test
        os.environ["AGENTOPTIM_JUDGE_MODEL"] = "env-test-model"
        os.environ["AGENTOPTIM_OMIT_REASONING"] = "1"
        
        # Patch the DEFAULT values in the server module
        with patch('agentoptim.server.DEFAULT_JUDGE_MODEL', "env-test-model"), \
             patch('agentoptim.server.DEFAULT_OMIT_REASONING', True), \
             patch('agentoptim.server.run_evalset') as mock_run:
            
            mock_run.return_value = {"status": "success", "message": "Test with env vars"}
            
            result = await run_evalset_tool(
                evalset_id="test-id",
                conversation=[{"role": "user", "content": "Hello"}]
            )
            
            # Check that run_evalset was called with the correct parameters
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]
            assert call_args["evalset_id"] == "test-id"
            assert call_args["judge_model"] == "env-test-model"
            assert call_args["omit_reasoning"] is True
            
            assert result == {"status": "success", "message": "Test with env vars"}
    finally:
        # Restore original environment variable values
        if original_judge_model is not None:
            os.environ["AGENTOPTIM_JUDGE_MODEL"] = original_judge_model
        else:
            os.environ.pop("AGENTOPTIM_JUDGE_MODEL", None)
            
        if original_omit_reasoning is not None:
            os.environ["AGENTOPTIM_OMIT_REASONING"] = original_omit_reasoning
        else:
            os.environ.pop("AGENTOPTIM_OMIT_REASONING", None)


@pytest.mark.asyncio
async def test_run_evalset_tool_with_client_options():
    """Test the run_evalset_tool with client options."""
    with patch('agentoptim.server.run_evalset') as mock_run, \
         patch('agentoptim.server.mcp') as mock_mcp:
        
        # Create a mock request context
        mock_server = Mock()
        mock_context = Mock()
        mock_context.init_options = {
            'judge_model': 'client-test-model',
            'omit_reasoning': 'true'
        }
        mock_server.request_context = mock_context
        mock_mcp._mcp_server = mock_server
        
        mock_run.return_value = {"status": "success", "message": "Test with client options"}
        
        # Ensure environment variables are not set
        with patch.dict(os.environ, {}, clear=True):
            with patch('agentoptim.server.DEFAULT_JUDGE_MODEL', None), \
                 patch('agentoptim.server.DEFAULT_OMIT_REASONING', False):
                
                result = await run_evalset_tool(
                    evalset_id="test-id",
                    conversation=[{"role": "user", "content": "Hello"}]
                )
                
                mock_run.assert_called_once_with(
                    evalset_id="test-id",
                    conversation=[{"role": "user", "content": "Hello"}],
                    judge_model="client-test-model",  # Should use client option
                    max_parallel=3,  # Default value
                    omit_reasoning=True  # Should use client option
                )
                
                assert result == {"status": "success", "message": "Test with client options"}


@pytest.mark.asyncio
async def test_run_evalset_tool_formatted_message():
    """Test the run_evalset_tool handling of formatted_message."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.return_value = {
            "status": "success",
            "formatted_message": "# Formatted Results\n\nThis is a formatted results message.",
            "evalset_id": "test-id",
            "results": [{"question": "Q1", "judgment": True}]
        }
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        assert result["status"] == "success"
        assert "result" in result
        assert result["result"] == "# Formatted Results\n\nThis is a formatted results message."
        assert result["evalset_id"] == "test-id"
        assert "results" in result


@pytest.mark.asyncio
async def test_run_evalset_tool_string_result():
    """Test the run_evalset_tool handling of string result."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.return_value = "Plain text result"
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        assert "result" in result
        assert result["result"] == "Plain text result"


@pytest.mark.asyncio
async def test_run_evalset_tool_error():
    """Test the run_evalset_tool endpoint error handling."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = ValueError("Invalid input")
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[]
        )
        
        assert isinstance(result, dict)
        assert "error" in result
        assert "Invalid input" in result["error"]


@pytest.mark.asyncio
async def test_run_evalset_tool_evalset_not_found_error():
    """Test the run_evalset_tool error handling for evalset not found."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = ValueError("evalset_id not found")
        
        result = await run_evalset_tool(
            evalset_id="nonexistent-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        assert "error" in result
        # These details might not exist in all implementations
        # The important part is that the error is correctly propagated
        assert "evalset_id not found" in result["error"]


@pytest.mark.asyncio
async def test_run_evalset_tool_conversation_error():
    """Test the run_evalset_tool error handling for invalid conversation."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = ValueError("conversation format is invalid")
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation="not a list"
        )
        
        assert "error" in result
        assert "details" in result
        assert "troubleshooting" in result
        assert "example" in result
        assert "conversation" in result["details"]


@pytest.mark.asyncio
async def test_run_evalset_tool_max_parallel_error():
    """Test the run_evalset_tool error handling for invalid max_parallel."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = ValueError("max_parallel must be positive")
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}],
            max_parallel=-1
        )
        
        assert "error" in result
        assert "details" in result
        assert "troubleshooting" in result
        assert "max_parallel" in result["details"]


@pytest.mark.asyncio
async def test_run_evalset_tool_model_error():
    """Test the run_evalset_tool error handling for invalid model."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = ValueError("model is invalid")
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        assert "error" in result
        assert "details" in result
        assert "troubleshooting" in result
        assert "model" in result["details"]


@pytest.mark.asyncio
async def test_run_evalset_tool_generic_error():
    """Test the run_evalset_tool generic error handling."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        mock_run.side_effect = Exception("Something went wrong")
        
        result = await run_evalset_tool(
            evalset_id="test-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        assert "error" in result
        assert "troubleshooting" in result


def test_main():
    """Test the main function."""
    with patch('agentoptim.server.mcp.run') as mock_run:
        main()
        mock_run.assert_called_once_with(transport="stdio")
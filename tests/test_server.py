"""
Tests for the MCP server implementation.
"""

import pytest
from unittest.mock import patch, MagicMock

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
            template=None,
            questions=None,
            description=None
        )
        
        assert result == {"status": "success", "message": "Success"}


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


def test_main():
    """Test the main function."""
    with patch('agentoptim.server.mcp.run') as mock_run:
        main()
        mock_run.assert_called_once_with(transport="stdio")
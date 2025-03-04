"""
Tests for the MCP server implementation.
"""

import pytest
from unittest.mock import patch, MagicMock

from agentoptim.server import (
    manage_evaluation_tool,
    manage_dataset_tool,
    manage_experiment_tool,
    run_job_tool,
    analyze_results_tool,
    main
)


@pytest.mark.asyncio
async def test_manage_evaluation_tool():
    """Test the manage_evaluation_tool endpoint."""
    with patch('agentoptim.server.manage_evaluation') as mock_manage:
        mock_manage.return_value = "Success"
        
        result = await manage_evaluation_tool(
            action="list"
        )
        
        mock_manage.assert_called_once_with(
            action="list",
            evaluation_id=None,
            name=None,
            template=None,
            questions=None,
            description=None
        )
        
        assert result == "Success"


@pytest.mark.asyncio
async def test_manage_evaluation_tool_error():
    """Test the manage_evaluation_tool endpoint error handling."""
    with patch('agentoptim.server.manage_evaluation') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid input")
        
        result = await manage_evaluation_tool(
            action="invalid"
        )
        
        assert result.startswith("Error:")
        assert "Invalid input" in result


@pytest.mark.asyncio
async def test_manage_dataset_tool():
    """Test the manage_dataset_tool endpoint."""
    with patch('agentoptim.server.manage_dataset') as mock_manage:
        mock_manage.return_value = "Success"
        
        result = await manage_dataset_tool(
            action="list"
        )
        
        mock_manage.assert_called_once()
        assert result == "Success"


@pytest.mark.asyncio
async def test_manage_dataset_tool_error():
    """Test the manage_dataset_tool endpoint error handling."""
    with patch('agentoptim.server.manage_dataset') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid input")
        
        result = await manage_dataset_tool(
            action="invalid"
        )
        
        assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_manage_experiment_tool():
    """Test the manage_experiment_tool endpoint."""
    with patch('agentoptim.server.manage_experiment') as mock_manage:
        mock_manage.return_value = "Success"
        
        result = await manage_experiment_tool(
            action="list"
        )
        
        mock_manage.assert_called_once()
        assert result == "Success"


@pytest.mark.asyncio
async def test_manage_experiment_tool_error():
    """Test the manage_experiment_tool endpoint error handling."""
    with patch('agentoptim.server.manage_experiment') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid input")
        
        result = await manage_experiment_tool(
            action="invalid"
        )
        
        assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_run_job_tool():
    """Test the run_job_tool endpoint."""
    with patch('agentoptim.server.manage_job') as mock_manage:
        mock_manage.return_value = "Success"
        
        # Test normal action
        result = await run_job_tool(
            action="list"
        )
        
        mock_manage.assert_called_once()
        assert result == "Success"
        
        # Reset mock
        mock_manage.reset_mock()
        
        # Test run action
        result = await run_job_tool(
            action="run",
            job_id="job_123"
        )
        
        mock_manage.assert_called_once()
        assert "started" in result


@pytest.mark.asyncio
async def test_run_job_tool_error():
    """Test the run_job_tool endpoint error handling."""
    with patch('agentoptim.server.manage_job') as mock_manage:
        mock_manage.side_effect = ValueError("Invalid input")
        
        result = await run_job_tool(
            action="invalid"
        )
        
        assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_analyze_results_tool():
    """Test the analyze_results_tool endpoint."""
    with patch('agentoptim.server.analyze_results') as mock_analyze:
        mock_analyze.return_value = "Success"
        
        result = await analyze_results_tool(
            action="list"
        )
        
        mock_analyze.assert_called_once()
        assert result == "Success"


@pytest.mark.asyncio
async def test_analyze_results_tool_error():
    """Test the analyze_results_tool endpoint error handling."""
    with patch('agentoptim.server.analyze_results') as mock_analyze:
        mock_analyze.side_effect = ValueError("Invalid input")
        
        result = await analyze_results_tool(
            action="invalid"
        )
        
        assert result.startswith("Error:")


def test_main():
    """Test the main function."""
    with patch('agentoptim.server.mcp.run') as mock_run:
        main()
        mock_run.assert_called_once_with(transport="stdio")
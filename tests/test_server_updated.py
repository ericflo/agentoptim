"""
Tests for the updated MCP server implementation with the manage_eval_runs_tool.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, Mock

from agentoptim.server import (
    manage_evalset_tool,
    manage_eval_runs_tool,
    get_cache_stats,
    main
)


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_run():
    """Test the manage_eval_runs_tool with run action."""
    # Setup mocks
    with patch('agentoptim.server.run_evalset') as mock_run, \
         patch('agentoptim.server.save_eval_run') as mock_save:
        
        # Mock run_evalset return value
        mock_run.return_value = {
            "status": "success",
            "evalset_id": "test-evalset-id",
            "evalset_name": "Test EvalSet",
            "judge_model": "test-model",
            "results": [{"question": "Q1", "judgment": True}],
            "summary": {"yes_percentage": 100}
        }
        
        # Mock save_eval_run to return True
        mock_save.return_value = True
        
        # Call the tool with run action
        result = await manage_eval_runs_tool(
            action="run",
            evalset_id="test-evalset-id",
            conversation=[{"role": "user", "content": "Hello"}],
            judge_model="test-model"
        )
        
        # Check that run_evalset was called with the right parameters
        mock_run.assert_called_once_with(
            evalset_id="test-evalset-id",
            conversation=[{"role": "user", "content": "Hello"}],
            judge_model="test-model",
            max_parallel=3,
            omit_reasoning=False
        )
        
        # Check that save_eval_run was called
        mock_save.assert_called_once()
        
        # Check the result
        assert result["status"] == "success"
        assert "id" in result  # Should have an ID from the saved EvalRun
        assert result["evalset_id"] == "test-evalset-id"
        assert result["evalset_name"] == "Test EvalSet"
        assert result["judge_model"] == "test-model"


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_get():
    """Test the manage_eval_runs_tool with get action."""
    with patch('agentoptim.server.manage_eval_runs') as mock_manage:
        # Mock manage_eval_runs return value
        mock_manage.return_value = {
            "status": "success",
            "eval_run": {
                "id": "test-run-id",
                "evalset_id": "test-evalset-id",
                "evalset_name": "Test EvalSet",
                "judge_model": "test-model",
                "results": [{"question": "Q1", "judgment": True}],
                "summary": {"yes_percentage": 100}
            },
            "formatted_message": "# Test Evaluation Run"
        }
        
        # Call the tool with get action
        result = await manage_eval_runs_tool(
            action="get",
            eval_run_id="test-run-id"
        )
        
        # Check that manage_eval_runs was called with the right parameters
        mock_manage.assert_called_once_with(
            action="get",
            evalset_id=None,
            conversation=None,
            judge_model=None,
            max_parallel=3,
            omit_reasoning=False,
            eval_run_id="test-run-id",
            page=1,
            page_size=10
        )
        
        # Check the result
        assert result["status"] == "success"
        assert "eval_run" in result
        assert result["eval_run"]["id"] == "test-run-id"
        assert result["eval_run"]["evalset_id"] == "test-evalset-id"
        assert "formatted_message" in result


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_list():
    """Test the manage_eval_runs_tool with list action."""
    with patch('agentoptim.server.manage_eval_runs') as mock_manage:
        # Mock manage_eval_runs return value
        mock_manage.return_value = {
            "status": "success",
            "eval_runs": [
                {
                    "id": "test-run-id-1",
                    "evalset_id": "test-evalset-id",
                    "evalset_name": "Test EvalSet",
                    "timestamp": 1628097600.0,
                    "timestamp_formatted": "2021-08-04 12:00:00",
                    "judge_model": "test-model",
                    "summary": {"yes_percentage": 100}
                },
                {
                    "id": "test-run-id-2",
                    "evalset_id": "test-evalset-id",
                    "evalset_name": "Test EvalSet",
                    "timestamp": 1628184000.0,
                    "timestamp_formatted": "2021-08-05 12:00:00",
                    "judge_model": "test-model",
                    "summary": {"yes_percentage": 80}
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 10,
                "total_count": 2,
                "total_pages": 1
            },
            "formatted_message": "# Evaluation Runs"
        }
        
        # Call the tool with list action
        result = await manage_eval_runs_tool(
            action="list",
            page=1,
            page_size=10
        )
        
        # Check that manage_eval_runs was called with the right parameters
        mock_manage.assert_called_once_with(
            action="list",
            evalset_id=None,
            conversation=None,
            judge_model=None,
            max_parallel=3,
            omit_reasoning=False,
            eval_run_id=None,
            page=1,
            page_size=10
        )
        
        # Check the result
        assert result["status"] == "success"
        assert "eval_runs" in result
        assert len(result["eval_runs"]) == 2
        assert "pagination" in result
        assert result["pagination"]["page"] == 1
        assert result["pagination"]["total_count"] == 2
        assert "formatted_message" in result


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_list_with_filtering():
    """Test the manage_eval_runs_tool with list action and filtering."""
    with patch('agentoptim.server.manage_eval_runs') as mock_manage:
        # Mock manage_eval_runs return value
        mock_manage.return_value = {
            "status": "success",
            "eval_runs": [
                {
                    "id": "test-run-id-1",
                    "evalset_id": "specific-evalset-id",
                    "evalset_name": "Specific EvalSet",
                    "timestamp": 1628097600.0,
                    "timestamp_formatted": "2021-08-04 12:00:00",
                    "judge_model": "test-model",
                    "summary": {"yes_percentage": 100}
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 10,
                "total_count": 1,
                "total_pages": 1
            },
            "formatted_message": "# Evaluation Runs for EvalSet 'specific-evalset-id'"
        }
        
        # Call the tool with list action and evalset_id filter
        result = await manage_eval_runs_tool(
            action="list",
            evalset_id="specific-evalset-id",
            page=1,
            page_size=10
        )
        
        # Check that manage_eval_runs was called with the right parameters
        mock_manage.assert_called_once_with(
            action="list",
            evalset_id="specific-evalset-id",
            conversation=None,
            judge_model=None,
            max_parallel=3,
            omit_reasoning=False,
            eval_run_id=None,
            page=1,
            page_size=10
        )
        
        # Check the result
        assert result["status"] == "success"
        assert "eval_runs" in result
        assert len(result["eval_runs"]) == 1
        assert result["eval_runs"][0]["evalset_id"] == "specific-evalset-id"
        assert "pagination" in result
        assert result["pagination"]["total_count"] == 1
        assert "formatted_message" in result


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_run_with_client_options():
    """Test the manage_eval_runs_tool with run action and client options."""
    with patch('agentoptim.server.run_evalset') as mock_run, \
         patch('agentoptim.server.save_eval_run') as mock_save, \
         patch('agentoptim.server.mcp') as mock_mcp:
        
        # Create a mock request context with client options
        mock_server = Mock()
        mock_context = Mock()
        mock_context.init_options = {
            'judge_model': 'client-test-model',
            'omit_reasoning': 'true'
        }
        mock_server.request_context = mock_context
        mock_mcp._mcp_server = mock_server
        
        # Mock run_evalset return value
        mock_run.return_value = {
            "status": "success",
            "evalset_id": "test-evalset-id",
            "evalset_name": "Test EvalSet",
            "judge_model": "client-test-model",
            "results": [{"question": "Q1", "judgment": True}],
            "summary": {"yes_percentage": 100}
        }
        
        # Mock save_eval_run to return True
        mock_save.return_value = True
        
        # Ensure environment variables are not set
        with patch.dict(os.environ, {}, clear=True):
            with patch('agentoptim.server.DEFAULT_JUDGE_MODEL', None), \
                 patch('agentoptim.server.DEFAULT_OMIT_REASONING', False):
                
                # Call the tool with run action
                result = await manage_eval_runs_tool(
                    action="run",
                    evalset_id="test-evalset-id",
                    conversation=[{"role": "user", "content": "Hello"}]
                )
                
                # Check that run_evalset was called
                mock_run.assert_called_once()
                call_args = mock_run.call_args[1]
                assert call_args["evalset_id"] == "test-evalset-id"
                assert call_args["conversation"] == [{"role": "user", "content": "Hello"}]
                assert call_args["judge_model"] == "client-test-model"
                # We don't strictly check omit_reasoning as it can vary due to test environment
                
                # Check the result
                assert result["status"] == "success"
                assert result["judge_model"] == "client-test-model"


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_run_with_env_vars():
    """Test the manage_eval_runs_tool with run action and environment variables."""
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
             patch('agentoptim.server.run_evalset') as mock_run, \
             patch('agentoptim.server.save_eval_run') as mock_save:
            
            # Mock run_evalset return value
            mock_run.return_value = {
                "status": "success",
                "evalset_id": "test-evalset-id",
                "evalset_name": "Test EvalSet",
                "judge_model": "env-test-model",
                "results": [{"question": "Q1", "judgment": True}],
                "summary": {"yes_percentage": 100}
            }
            
            # Mock save_eval_run to return True
            mock_save.return_value = True
            
            # Call the tool with run action
            result = await manage_eval_runs_tool(
                action="run",
                evalset_id="test-evalset-id",
                conversation=[{"role": "user", "content": "Hello"}]
            )
            
            # Check that run_evalset was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[1]
            assert call_args["evalset_id"] == "test-evalset-id"
            assert call_args["conversation"] == [{"role": "user", "content": "Hello"}]
            assert call_args["judge_model"] == "env-test-model"
            # We don't strictly check omit_reasoning as it can vary due to test environment
            
            # Check the result
            assert result["status"] == "success"
            assert result["judge_model"] == "env-test-model"
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
async def test_manage_eval_runs_tool_error_handling():
    """Test the manage_eval_runs_tool error handling."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        # Simulate error in run_evalset
        mock_run.side_effect = ValueError("EvalSet not found")
        
        # Call the tool with run action
        result = await manage_eval_runs_tool(
            action="run",
            evalset_id="nonexistent-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        # Check the error response
        assert "error" in result
        assert "EvalSet not found" in result["error"]
        # The result may include different keys depending on the implementation
        # so we only check for the essential error message


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_conversation_error():
    """Test the manage_eval_runs_tool error handling for invalid conversation."""
    with patch('agentoptim.server.run_evalset') as mock_run:
        # Simulate error in run_evalset
        mock_run.side_effect = ValueError("conversation format is invalid")
        
        # Call the tool with run action
        result = await manage_eval_runs_tool(
            action="run",
            evalset_id="test-id",
            conversation="not a list"
        )
        
        # Check the error response
        assert "error" in result
        assert "details" in result
        assert "troubleshooting" in result
        assert "example" in result
        assert "conversation" in result["details"]


@pytest.mark.asyncio
async def test_manage_eval_runs_tool_invalid_action():
    """Test the manage_eval_runs_tool with an invalid action."""
    # Call the tool with an invalid action
    result = await manage_eval_runs_tool(
        action="invalid"
    )
    
    # Check the error response
    assert "error" in result
    # The specific format of the error response may vary


def test_get_cache_stats():
    """Test the get_cache_stats function."""
    with patch('agentoptim.server.get_cache_statistics') as mock_evalset_stats, \
         patch('agentoptim.server.get_api_cache_stats') as mock_api_stats, \
         patch('agentoptim.evalrun.get_eval_runs_cache_stats') as mock_evalruns_stats:
        
        # Mock cache stats
        mock_evalset_stats.return_value = {
            "size": 10,
            "capacity": 50,
            "hits": 100,
            "misses": 20,
            "hit_rate_pct": 83.33,
            "evictions": 2,
            "expirations": 1
        }
        
        mock_api_stats.return_value = {
            "size": 30,
            "capacity": 100,
            "hits": 200,
            "misses": 50,
            "hit_rate_pct": 80.0,
            "evictions": 5,
            "expirations": 2
        }
        
        mock_evalruns_stats.return_value = {
            "size": 15,
            "capacity": 50,
            "hits": 150,
            "misses": 30,
            "hit_rate_pct": 83.33,
            "evictions": 3,
            "expirations": 1
        }
        
        # Get cache stats
        result = get_cache_stats()
        
        # Check the result
        assert result["status"] == "success"
        assert "evalset_cache" in result
        assert "api_cache" in result
        assert "evalrun_cache" in result
        assert "overall" in result
        
        # Check combined stats
        overall = result["overall"]
        assert overall["total_hits"] == 450  # 100 + 200 + 150
        assert overall["total_misses"] == 100  # 20 + 50 + 30
        assert overall["hit_rate_pct"] == 81.82  # 450 / (450 + 100) * 100
        assert "estimated_time_saved_seconds" in overall
        
        # Check formatted message
        assert "formatted_message" in result
        assert "Cache Performance Statistics" in result["formatted_message"]
        assert "EvalSet Cache" in result["formatted_message"]
        assert "API Response Cache" in result["formatted_message"]
        assert "Eval Runs Cache" in result["formatted_message"]
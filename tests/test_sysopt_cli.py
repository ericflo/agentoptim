"""Tests for the system message optimization CLI."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import argparse

from agentoptim.sysopt_cli import (
    optimize_setup_parser,
    handle_optimize,
    handle_optimize_create,
    handle_optimize_get,
    handle_optimize_list,
    handle_optimize_meta,
    format_optimization_result,
    format_optimization_list
)

# Test the optimize_setup_parser function
def test_optimize_setup_parser():
    # Create a parent parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="resource")
    
    # Set up the optimize subparser
    optimize_parser = optimize_setup_parser(subparsers)
    
    # Verify the optimize parser was created
    assert optimize_parser is not None
    
    # Parse a simple command to verify the parser works
    args = parser.parse_args(["optimize", "list"])
    assert args.resource == "optimize"
    assert args.action == "list"
    
    # Test the create command
    args = parser.parse_args(["optimize", "create", "test-evalset-id", "test-user-message"])
    assert args.resource == "optimize"
    assert args.action == "create"
    assert args.evalset_id == "test-evalset-id"
    assert args.user_message == "test-user-message"
    
    # Test the get command
    args = parser.parse_args(["optimize", "get", "test-run-id"])
    assert args.resource == "optimize"
    assert args.action == "get"
    assert args.optimization_run_id == "test-run-id"
    
    # Test the meta command
    args = parser.parse_args(["optimize", "meta", "test-evalset-id"])
    assert args.resource == "optimize"
    assert args.action == "meta"
    assert args.evalset_id == "test-evalset-id"

# Test format_optimization_result function
def test_format_optimization_result():
    # Create a sample result
    result = {
        "id": "test-id",
        "evalset_id": "test-evalset-id",
        "evalset_name": "Test EvalSet",
        "user_message": "Test user message",
        "best_system_message": "You are a helpful assistant.",
        "best_score": 90.5,
        "candidates": [
            {
                "content": "You are a helpful assistant.",
                "score": 90.5,
                "criterion_scores": {"Question 1": 100, "Question 2": 80},
                "rank": 1
            },
            {
                "content": "You are a knowledgeable AI.",
                "score": 85.2,
                "criterion_scores": {"Question 1": 100, "Question 2": 70},
                "rank": 2
            }
        ]
    }
    
    # Test JSON format
    json_result = format_optimization_result(result, "json")
    assert json.loads(json_result)["id"] == "test-id"
    
    # Test markdown format
    md_result = format_optimization_result(result, "markdown")
    assert "System Message Optimization Results" in md_result
    
    # Test HTML format
    html_result = format_optimization_result(result, "html")
    assert "<html>" in html_result
    assert "You are a helpful assistant." in html_result
    
    # Test text format (quiet)
    quiet_result = format_optimization_result(result, "text", quiet=True)
    assert quiet_result == "You are a helpful assistant."

# Test format_optimization_list function
def test_format_optimization_list():
    # Create a sample result
    result = {
        "optimization_runs": [
            {
                "id": "test-id-1",
                "user_message": "Test user message 1",
                "evalset_name": "Test EvalSet",
                "best_score": 90.5,
                "timestamp_formatted": "2025-04-01 12:00"
            },
            {
                "id": "test-id-2",
                "user_message": "Test user message 2",
                "evalset_name": "Test EvalSet",
                "best_score": 85.2,
                "timestamp_formatted": "2025-04-01 13:00"
            }
        ],
        "pagination": {
            "page": 1,
            "page_size": 10,
            "total_count": 2,
            "total_pages": 1,
            "has_next": False,
            "has_prev": False
        }
    }
    
    # Test JSON format
    json_result = format_optimization_list(result, "json")
    assert len(json.loads(json_result)["optimization_runs"]) == 2
    
    # Test CSV format
    csv_result = format_optimization_list(result, "csv")
    assert "ID,User Message,EvalSet,Best Score,Candidates,Timestamp" in csv_result
    
    # Test markdown format
    md_result = format_optimization_list(result, "markdown")
    assert "| ID | User Message | EvalSet | Score | Date |" in md_result
    
    # Test text format (quiet)
    quiet_result = format_optimization_list(result, "text", quiet=True)
    assert "test-id-1\ntest-id-2" == quiet_result

# Test the handle_optimize_create function
@pytest.mark.asyncio
async def test_handle_optimize_create():
    # Mock the manage_optimization_runs function
    with patch("agentoptim.sysopt_cli.manage_optimization_runs") as mock_manage, \
         patch("agentoptim.sysopt_cli.FancySpinner") as mock_spinner:
        
        # Configure the mock
        mock_manage.return_value = {
            "id": "test-id",
            "evalset_id": "test-evalset-id",
            "evalset_name": "Test EvalSet",
            "best_system_message": "You are a helpful assistant.",
            "best_score": 90.5,
            "candidates": []
        }
        
        # Create args
        args = MagicMock()
        args.evalset_id = "test-evalset-id"
        args.user_message = "Test user message"
        args.base = None
        args.num_candidates = 5
        args.generator = "default"
        args.model = None
        args.diversity = "medium"
        args.concurrency = 3
        args.instructions = None
        args.self_optimize = False
        args.interactive = False
        args.format = "text"
        args.output = None
        
        # Call the function
        result = await handle_optimize_create(args)
        
        # Verify the result
        assert result == 0
        mock_manage.assert_called_once()
        
        # Test with error
        mock_manage.return_value = {"error": "Test error"}
        result = await handle_optimize_create(args)
        assert result == 1

# Test the handle_optimize_get function
@pytest.mark.asyncio
async def test_handle_optimize_get():
    # Mock the manage_optimization_runs function
    with patch("agentoptim.sysopt_cli.manage_optimization_runs") as mock_manage:
        
        # Configure the mock
        mock_manage.return_value = {
            "optimization_run": {
                "id": "test-id",
                "evalset_id": "test-evalset-id",
                "evalset_name": "Test EvalSet",
                "best_system_message": "You are a helpful assistant.",
                "best_score": 90.5,
                "candidates": []
            }
        }
        
        # Create args
        args = MagicMock()
        args.optimization_run_id = "test-id"
        args.format = "text"
        args.output = None
        
        # Call the function
        result = await handle_optimize_get(args)
        
        # Verify the result
        assert result == 0
        mock_manage.assert_called_once()
        
        # Test with error
        mock_manage.return_value = {"error": "Test error"}
        result = await handle_optimize_get(args)
        assert result == 1

# Test the handle_optimize_list function
@pytest.mark.asyncio
async def test_handle_optimize_list():
    # Mock the manage_optimization_runs function
    with patch("agentoptim.sysopt_cli.manage_optimization_runs") as mock_manage:
        
        # Configure the mock
        mock_manage.return_value = {
            "optimization_runs": [
                {
                    "id": "test-id-1",
                    "user_message": "Test user message 1",
                    "evalset_name": "Test EvalSet",
                    "best_score": 90.5,
                    "timestamp_formatted": "2025-04-01 12:00"
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 10,
                "total_count": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False
            }
        }
        
        # Create args
        args = MagicMock()
        args.evalset_id = None
        args.page = 1
        args.limit = 10
        args.format = "text"
        args.output = None
        args.quiet = False
        
        # Call the function
        result = await handle_optimize_list(args)
        
        # Verify the result
        assert result == 0
        mock_manage.assert_called_once()
        
        # Test with error
        mock_manage.return_value = {"error": "Test error"}
        result = await handle_optimize_list(args)
        assert result == 1

# Test the handle_optimize function
@pytest.mark.asyncio
async def test_handle_optimize():
    # Test dispatching to different handlers
    with patch("agentoptim.sysopt_cli.handle_optimize_create") as mock_create, \
         patch("agentoptim.sysopt_cli.handle_optimize_get") as mock_get, \
         patch("agentoptim.sysopt_cli.handle_optimize_list") as mock_list, \
         patch("agentoptim.sysopt_cli.handle_optimize_meta") as mock_meta:
        
        # Configure the mocks
        mock_create.return_value = 0
        mock_get.return_value = 0
        mock_list.return_value = 0
        mock_meta.return_value = 0
        
        # Create args for each action
        create_args = MagicMock()
        create_args.action = "create"
        
        get_args = MagicMock()
        get_args.action = "get"
        
        list_args = MagicMock()
        list_args.action = "list"
        
        meta_args = MagicMock()
        meta_args.action = "meta"
        
        unknown_args = MagicMock()
        unknown_args.action = "unknown"
        
        # Call the function for each action
        assert await handle_optimize(create_args) == 0
        assert await handle_optimize(get_args) == 0
        assert await handle_optimize(list_args) == 0
        assert await handle_optimize(meta_args) == 0
        assert await handle_optimize(unknown_args) == 1
        
        # Verify the mocks were called
        mock_create.assert_called_once()
        mock_get.assert_called_once()
        mock_list.assert_called_once()
        mock_meta.assert_called_once()
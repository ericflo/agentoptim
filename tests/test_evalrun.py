"""
Tests for the evalrun module.

These tests verify the functionality of the evaluation run storage
and retrieval capabilities.
"""

import os
import json
import shutil
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from agentoptim.evalrun import (
    EvalRun,
    get_eval_run,
    get_eval_run_summary,
    save_eval_run,
    delete_eval_run,
    list_eval_runs,
    cleanup_old_eval_runs,
    manage_eval_runs,
    get_formatted_eval_run,
    EVAL_RUNS_DIR
)


@pytest.fixture
def temp_eval_runs_dir():
    """Create a temporary directory for test data."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Patch the EVAL_RUNS_DIR to use our temporary directory
    with patch("agentoptim.evalrun.EVAL_RUNS_DIR", temp_dir):
        yield temp_dir
    
    # Clean up after the test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_eval_run():
    """Create a sample EvalRun for testing."""
    return EvalRun(
        evalset_id="test-evalset-id",
        evalset_name="Test EvalSet",
        judge_model="test-model",
        results=[
            {
                "question": "Is the response helpful?",
                "judgment": True,
                "confidence": 0.95,
                "reasoning": "The response directly addresses the user's question."
            },
            {
                "question": "Is the response accurate?",
                "judgment": False,
                "confidence": 0.85,
                "reasoning": "The response contains factual errors."
            }
        ],
        conversation=[
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "You can reset your password by clicking the 'Forgot Password' link."}
        ],
        summary={
            "total_questions": 2,
            "successful_evaluations": 2,
            "yes_count": 1,
            "no_count": 1,
            "yes_percentage": 50.0,
            "error_count": 0,
            "mean_confidence": 0.9,
            "mean_yes_confidence": 0.95,
            "mean_no_confidence": 0.85
        }
    )


@pytest.fixture
def sample_eval_run2():
    """Create a second sample EvalRun for testing."""
    return EvalRun(
        evalset_id="other-evalset-id",
        evalset_name="Other EvalSet",
        judge_model="test-model",
        results=[
            {
                "question": "Is the response helpful?",
                "judgment": True,
                "confidence": 0.9,
                "reasoning": "The response is helpful."
            }
        ],
        conversation=[
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "The weather is sunny."}
        ],
        summary={
            "total_questions": 1,
            "successful_evaluations": 1,
            "yes_count": 1,
            "no_count": 0,
            "yes_percentage": 100.0,
            "error_count": 0,
            "mean_confidence": 0.9
        }
    )


@pytest.fixture
def eval_run_cache_mock():
    """Create a mock for the eval_run_cache."""
    with patch("agentoptim.evalrun.eval_run_cache") as mock_cache:
        # Default to cache miss for get
        mock_cache.get.return_value = None
        yield mock_cache


class TestEvalRunModel:
    """Test the EvalRun model."""
    
    def test_evalrun_attributes(self, sample_eval_run):
        """Test that the EvalRun model sets attributes correctly."""
        # Check basic attributes
        assert sample_eval_run.evalset_id == "test-evalset-id"
        assert sample_eval_run.evalset_name == "Test EvalSet"
        assert sample_eval_run.judge_model == "test-model"
        assert len(sample_eval_run.results) == 2
        assert len(sample_eval_run.conversation) == 2
        
    def test_evalrun_to_dict(self, sample_eval_run):
        """Test conversion to dictionary."""
        eval_dict = sample_eval_run.to_dict()
        
        # Check dictionary values
        assert eval_dict["evalset_id"] == "test-evalset-id"
        assert eval_dict["evalset_name"] == "Test EvalSet"
        assert eval_dict["judge_model"] == "test-model"
        assert len(eval_dict["results"]) == 2
        assert len(eval_dict["conversation"]) == 2
        
    def test_evalrun_from_dict(self, sample_eval_run):
        """Test creation from dictionary."""
        # Convert to dict and back
        eval_dict = sample_eval_run.to_dict()
        eval2 = EvalRun.from_dict(eval_dict)
        
        # Check that the attributes match
        assert eval2.id == sample_eval_run.id
        assert eval2.evalset_id == "test-evalset-id"
        assert eval2.evalset_name == "Test EvalSet"
        assert eval2.judge_model == "test-model"
        
    def test_timestamp_formatting(self, sample_eval_run):
        """Test timestamp formatting."""
        timestamp_str = sample_eval_run.format_timestamp()
        
        # Check that the timestamp is formatted as a string
        assert isinstance(timestamp_str, str)
        assert len(timestamp_str) > 0


class TestEvalRunStorage:
    """Test the storage and retrieval of EvalRuns."""
    
    def test_save_eval_run(self, temp_eval_runs_dir, sample_eval_run, eval_run_cache_mock):
        """Test saving an EvalRun."""
        # Save the eval run
        result = save_eval_run(sample_eval_run)
        
        # Check that it was saved successfully
        assert result is True
        
        # Check that the file was created
        file_path = os.path.join(temp_eval_runs_dir, f"{sample_eval_run.id}.json")
        assert os.path.exists(file_path)
        
        # Verify the contents
        with open(file_path, "r") as f:
            saved_data = json.load(f)
        
        assert saved_data["id"] == sample_eval_run.id
        assert saved_data["evalset_id"] == "test-evalset-id"
        
        # Check that it was cached
        eval_run_cache_mock.put.assert_called_once_with(
            sample_eval_run.id, sample_eval_run
        )
    
    def test_get_eval_run(self, temp_eval_runs_dir, sample_eval_run, eval_run_cache_mock):
        """Test retrieving an EvalRun."""
        # First, save the eval run
        save_eval_run(sample_eval_run)
        
        # Get the eval run
        retrieved = get_eval_run(sample_eval_run.id)
        
        # Check that it was loaded correctly
        assert retrieved is not None
        assert retrieved.id == sample_eval_run.id
        assert retrieved.evalset_id == "test-evalset-id"
        assert retrieved.evalset_name == "Test EvalSet"
        
        # Test cache hit
        eval_run_cache_mock.get.return_value = sample_eval_run
        cached_run = get_eval_run(sample_eval_run.id)
        assert cached_run == sample_eval_run
        
        # Test getting a non-existent EvalRun
        eval_run_cache_mock.get.return_value = None
        assert get_eval_run("nonexistent-id") is None
    
    def test_get_eval_run_summary(self, temp_eval_runs_dir, sample_eval_run, eval_run_cache_mock):
        """Test retrieving a summary of an EvalRun."""
        # First, save the eval run
        save_eval_run(sample_eval_run)
        
        # Get the summary
        summary = get_eval_run_summary(sample_eval_run.id)
        
        # Check summary fields
        assert summary is not None
        assert summary["id"] == sample_eval_run.id
        assert summary["evalset_id"] == "test-evalset-id"
        assert summary["evalset_name"] == "Test EvalSet"
        assert summary["result_count"] == 2
        assert summary["conversation_length"] == 2
        
        # Test summary from cache
        eval_run_cache_mock.get.return_value = sample_eval_run
        cached_summary = get_eval_run_summary(sample_eval_run.id)
        assert cached_summary["id"] == sample_eval_run.id
        
        # Test getting a non-existent EvalRun
        eval_run_cache_mock.get.return_value = None
        assert get_eval_run_summary("nonexistent-id") is None
    
    def test_delete_eval_run(self, temp_eval_runs_dir, sample_eval_run, eval_run_cache_mock):
        """Test deleting an EvalRun."""
        # First, save the eval run
        save_eval_run(sample_eval_run)
        
        # Delete the eval run
        result = delete_eval_run(sample_eval_run.id)
        
        # Check that it was deleted successfully
        assert result is True
        
        # Check that the file is gone
        file_path = os.path.join(temp_eval_runs_dir, f"{sample_eval_run.id}.json")
        assert not os.path.exists(file_path)
        
        # Check that it was removed from the cache
        eval_run_cache_mock.remove.assert_called_once_with(sample_eval_run.id)
        
        # Test deleting a non-existent EvalRun
        assert delete_eval_run("nonexistent-id") is False


class TestEvalRunListing:
    """Test listing EvalRuns with pagination."""
    
    def test_list_eval_runs(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test listing EvalRuns."""
        # Save multiple eval runs
        save_eval_run(sample_eval_run)
        save_eval_run(sample_eval_run2)
        
        # List all eval runs
        eval_runs, total_count = list_eval_runs()
        
        # Check results
        assert total_count == 2
        assert len(eval_runs) == 2
        
        # Get IDs of the eval runs
        eval_run_ids = [run["id"] for run in eval_runs]
        assert sample_eval_run.id in eval_run_ids
        assert sample_eval_run2.id in eval_run_ids
    
    def test_list_pagination(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test pagination for list_eval_runs."""
        # Save multiple eval runs
        save_eval_run(sample_eval_run)
        save_eval_run(sample_eval_run2)
        
        # Test page 1, page size 1
        eval_runs_page1, total_count = list_eval_runs(page=1, page_size=1)
        assert total_count == 2
        assert len(eval_runs_page1) == 1
        
        # Test page 2, page size 1
        eval_runs_page2, total_count = list_eval_runs(page=2, page_size=1)
        assert total_count == 2
        assert len(eval_runs_page2) == 1
        
        # Check that page 1 and page 2 return different runs
        assert eval_runs_page1[0]["id"] != eval_runs_page2[0]["id"]
    
    def test_list_filter_by_evalset_id(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test filtering list_eval_runs by evalset_id."""
        # Save multiple eval runs
        save_eval_run(sample_eval_run)
        save_eval_run(sample_eval_run2)
        
        # Filter by evalset_id
        filtered_runs, total_count = list_eval_runs(evalset_id="test-evalset-id")
        
        # Check results
        assert total_count == 1
        assert len(filtered_runs) == 1
        assert filtered_runs[0]["evalset_id"] == "test-evalset-id"


class TestEvalRunMaintenance:
    """Test maintenance functions for EvalRuns."""
    
    def test_cleanup_old_eval_runs(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test cleanup of old EvalRuns."""
        # Create files with controlled modification times
        # First file (older)
        file_path1 = os.path.join(temp_eval_runs_dir, f"{sample_eval_run.id}.json")
        with open(file_path1, "w") as f:
            f.write(json.dumps(sample_eval_run.to_dict()))
        # Set modification time to be older
        os.utime(file_path1, (1000, 1000))
        
        # Second file (newer)
        file_path2 = os.path.join(temp_eval_runs_dir, f"{sample_eval_run2.id}.json")
        with open(file_path2, "w") as f:
            f.write(json.dumps(sample_eval_run2.to_dict()))
        # Set modification time to be newer
        os.utime(file_path2, (2000, 2000))
        
        # Verify we have two files
        assert len(os.listdir(temp_eval_runs_dir)) == 2
        
        # Patch the MAX_EVAL_RUNS constant
        with patch("agentoptim.evalrun.MAX_EVAL_RUNS", 1):
            # Run the cleanup
            removed_count = cleanup_old_eval_runs()
            
            # Check that one run was removed
            assert removed_count == 1
            
            # Check only one file remains
            assert len(os.listdir(temp_eval_runs_dir)) == 1
            
            # The remaining file should be the newer one
            assert os.path.exists(file_path2)
            assert not os.path.exists(file_path1)


class TestEvalRunFormatting:
    """Test formatting functions for EvalRuns."""
    
    def test_get_formatted_eval_run(self, sample_eval_run):
        """Test formatting an EvalRun for display."""
        formatted = get_formatted_eval_run(sample_eval_run)
        
        # Check the formatted result
        assert formatted["id"] == sample_eval_run.id
        assert formatted["evalset_id"] == "test-evalset-id"
        assert formatted["evalset_name"] == "Test EvalSet"
        
        # Check that the formatted_message field exists and contains expected sections
        assert "formatted_message" in formatted
        message = formatted["formatted_message"]
        assert "Test EvalSet" in message
        assert "Summary" in message
        assert "Detailed Results" in message
        assert "Conversation" in message


class TestManageEvalRuns:
    """Test the manage_eval_runs function."""
    
    def test_manage_eval_runs_list(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test manage_eval_runs with list action."""
        # Save multiple eval runs
        save_eval_run(sample_eval_run)
        save_eval_run(sample_eval_run2)
        
        # List all eval runs
        result = manage_eval_runs(action="list")
        
        # Check result format
        assert result["status"] == "success"
        assert "eval_runs" in result
        assert len(result["eval_runs"]) == 2
        
        # Check pagination info
        assert "pagination" in result
        pagination = result["pagination"]
        assert pagination["page"] == 1
        assert pagination["total_count"] == 2
        assert pagination["total_pages"] == 1
        assert pagination["has_next"] is False
        assert pagination["has_prev"] is False
        
        # Check formatted message
        assert "formatted_message" in result
        assert "Evaluation Runs" in result["formatted_message"]
    
    def test_manage_eval_runs_list_pagination(self, temp_eval_runs_dir, sample_eval_run, sample_eval_run2):
        """Test pagination in manage_eval_runs list action."""
        # Save multiple eval runs
        save_eval_run(sample_eval_run)
        save_eval_run(sample_eval_run2)
        
        # Test page 1, page size 1
        result_page1 = manage_eval_runs(action="list", page=1, page_size=1)
        assert len(result_page1["eval_runs"]) == 1
        assert result_page1["pagination"]["page"] == 1
        assert result_page1["pagination"]["total_pages"] == 2
        assert result_page1["pagination"]["has_next"] is True
        assert result_page1["pagination"]["has_prev"] is False
        
        # Test page 2, page size 1
        result_page2 = manage_eval_runs(action="list", page=2, page_size=1)
        assert len(result_page2["eval_runs"]) == 1
        assert result_page2["pagination"]["page"] == 2
        assert result_page2["pagination"]["total_pages"] == 2
        assert result_page2["pagination"]["has_next"] is False
        assert result_page2["pagination"]["has_prev"] is True
    
    def test_manage_eval_runs_get(self, temp_eval_runs_dir, sample_eval_run):
        """Test manage_eval_runs with get action."""
        # Save an eval run
        save_eval_run(sample_eval_run)
        
        # Get the eval run
        result = manage_eval_runs(action="get", eval_run_id=sample_eval_run.id)
        
        # Check result
        assert result["status"] == "success"
        assert "eval_run" in result
        assert result["eval_run"]["id"] == sample_eval_run.id
        assert result["eval_run"]["evalset_id"] == "test-evalset-id"
        assert "formatted_message" in result
        
        # Test getting a non-existent eval run
        error_result = manage_eval_runs(action="get", eval_run_id="nonexistent-id")
        assert "error" in error_result
    
    def test_manage_eval_runs_run(self):
        """Test manage_eval_runs with run action."""
        # The run action is implemented in server.py to avoid circular imports
        # This test just verifies that the function returns an error
        result = manage_eval_runs(
            action="run",
            evalset_id="test-evalset-id",
            conversation=[{"role": "user", "content": "Hello"}]
        )
        
        # Verify that we got some kind of error response
        assert isinstance(result, dict)
        assert "error" in result
        # Don't test the specific error message, as it might change
    
    def test_manage_eval_runs_invalid_action(self):
        """Test manage_eval_runs with an invalid action."""
        result = manage_eval_runs(action="invalid")
        
        # Verify that we got some kind of error response
        assert isinstance(result, dict)
        assert "error" in result
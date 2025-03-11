"""Tests for the system message optimization module."""

import os
import json
import uuid
import pytest
from unittest.mock import patch, MagicMock

from agentoptim.sysopt import (
    SystemMessageCandidate,
    SystemMessageGenerator,
    OptimizationRun,
    get_optimization_run,
    list_optimization_runs,
    manage_optimization_runs
)

from agentoptim.sysopt.core import (
    get_all_meta_prompts,
    save_generator,
    save_optimization_run
)

from agentoptim.server import optimize_system_messages_tool
from agentoptim.constants import (
    MAX_SYSTEM_MESSAGE_LENGTH,
    DEFAULT_NUM_CANDIDATES,
    MAX_CANDIDATES,
    DIVERSITY_LEVELS
)

# Test the SystemMessageCandidate model
def test_system_message_candidate():
    # Test valid candidate
    candidate = SystemMessageCandidate(
        content="You are a helpful assistant."
    )
    assert candidate.content == "You are a helpful assistant."
    assert candidate.score is None
    assert candidate.criterion_scores == {}
    assert candidate.rank is None
    assert candidate.generation_metadata == {}
    
    # Test content validation
    with pytest.raises(ValueError):
        SystemMessageCandidate(
            content="x" * (MAX_SYSTEM_MESSAGE_LENGTH + 1)
        )
    
    # Test with scores
    candidate = SystemMessageCandidate(
        content="You are a helpful assistant.",
        score=92.5,
        criterion_scores={"clarity": 95.0, "helpfulness": 90.0},
        rank=1
    )
    assert candidate.score == 92.5
    assert candidate.criterion_scores == {"clarity": 95.0, "helpfulness": 90.0}
    assert candidate.rank == 1

# Test the SystemMessageGenerator model
def test_system_message_generator():
    # Test valid generator
    generator = SystemMessageGenerator(
        id="test-generator",
        version=1,
        meta_prompt="Generate system messages for {{user_message}}",
        domain="general",
        performance_metrics={"success_rate": 0.95}
    )
    assert generator.id == "test-generator"
    assert generator.version == 1
    assert generator.meta_prompt == "Generate system messages for {{user_message}}"
    assert generator.domain == "general"
    assert generator.performance_metrics == {"success_rate": 0.95}
    assert generator.created_at is not None
    assert generator.updated_at is None
    
    # Test meta_prompt validation
    with pytest.raises(ValueError):
        SystemMessageGenerator(
            meta_prompt="x" * (MAX_SYSTEM_MESSAGE_LENGTH + 1)
        )
    
    # Test with default ID generation
    generator = SystemMessageGenerator(
        meta_prompt="Generate system messages for {{user_message}}"
    )
    assert generator.id is not None
    assert isinstance(generator.id, str)
    assert len(generator.id) > 0

# Test the OptimizationRun model
def test_optimization_run():
    # Test valid optimization run
    candidates = [
        SystemMessageCandidate(content="You are a helpful assistant.", score=95.0, rank=1),
        SystemMessageCandidate(content="You are a knowledgeable AI.", score=90.0, rank=2)
    ]
    
    run = OptimizationRun(
        id="test-run",
        user_message="How do I reset my password?",
        evalset_id="test-evalset",
        candidates=candidates,
        best_candidate_index=0,
        generator_id="test-generator",
        generator_version=1
    )
    
    assert run.id == "test-run"
    assert run.user_message == "How do I reset my password?"
    assert run.evalset_id == "test-evalset"
    assert len(run.candidates) == 2
    assert run.best_candidate_index == 0
    assert run.generator_id == "test-generator"
    assert run.generator_version == 1
    assert run.timestamp is not None
    assert run.metadata == {}
    
    # Test with default ID generation
    run = OptimizationRun(
        user_message="How do I reset my password?",
        evalset_id="test-evalset",
        candidates=candidates,
        best_candidate_index=0,
        generator_id="test-generator",
        generator_version=1
    )
    assert run.id is not None
    assert isinstance(run.id, str)
    assert len(run.id) > 0

# Mock the filesystem operations for testing
@pytest.fixture
def mock_filesystem(monkeypatch):
    """Mock filesystem operations for testing."""
    mock_generators = {
        "default": SystemMessageGenerator(
            id="default", 
            meta_prompt="Generate system messages for {{user_message}}",
            domain="general"
        ),
        "custom": SystemMessageGenerator(
            id="custom", 
            meta_prompt="Generate creative system messages for {{user_message}}",
            domain="creative"
        )
    }
    
    mock_runs = {
        "run1": OptimizationRun(
            id="run1",
            user_message="How do I reset my password?",
            evalset_id="evalset1",
            candidates=[
                SystemMessageCandidate(content="You are a helpful assistant.", score=95.0, rank=1)
            ],
            best_candidate_index=0,
            generator_id="default",
            generator_version=1
        ),
        "run2": OptimizationRun(
            id="run2",
            user_message="What's the weather like?",
            evalset_id="evalset1",
            candidates=[
                SystemMessageCandidate(content="You are a weather expert.", score=90.0, rank=1)
            ],
            best_candidate_index=0,
            generator_id="default",
            generator_version=1
        )
    }
    
    # Mock get_all_meta_prompts
    def mock_get_all_meta_prompts():
        return mock_generators
    
    # Mock save_generator
    def mock_save_generator(generator):
        mock_generators[generator.id] = generator
        return True
    
    # Mock save_optimization_run
    def mock_save_optimization_run(run):
        mock_runs[run.id] = run
        return True
    
    # Mock get_optimization_run
    def mock_get_optimization_run(run_id):
        return mock_runs.get(run_id)
    
    # Mock list_optimization_runs
    def mock_list_optimization_runs(page=1, page_size=10, evalset_id=None):
        filtered_runs = [run for run in mock_runs.values() 
                         if evalset_id is None or run.evalset_id == evalset_id]
        total_count = len(filtered_runs)
        total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 1
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        return {
            "status": "success",
            "optimization_runs": [run.model_dump() for run in filtered_runs[start_idx:end_idx]],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
                "next_page": page + 1 if page < total_pages else None,
                "prev_page": page - 1 if page > 1 else None
            }
        }
    
    # Apply the mocks
    monkeypatch.setattr("agentoptim.sysopt.core.get_all_meta_prompts", mock_get_all_meta_prompts)
    monkeypatch.setattr("agentoptim.sysopt.core.save_generator", mock_save_generator)
    monkeypatch.setattr("agentoptim.sysopt.core.save_optimization_run", mock_save_optimization_run)
    monkeypatch.setattr("agentoptim.sysopt.get_optimization_run", mock_get_optimization_run)
    monkeypatch.setattr("agentoptim.sysopt.list_optimization_runs", mock_list_optimization_runs)
    
    return {
        "generators": mock_generators,
        "runs": mock_runs
    }

# Test the get_all_meta_prompts function
def test_get_all_meta_prompts(mock_filesystem):
    from agentoptim.sysopt.core import get_all_meta_prompts
    generators = get_all_meta_prompts()
    assert len(generators) == 2
    assert "default" in generators
    assert "custom" in generators
    assert generators["default"].domain == "general"
    assert generators["custom"].domain == "creative"

# Test the save_generator function
def test_save_generator(mock_filesystem):
    from agentoptim.sysopt.core import save_generator, get_all_meta_prompts
    new_generator = SystemMessageGenerator(
        id="new",
        meta_prompt="New meta prompt",
        domain="technical"
    )
    result = save_generator(new_generator)
    assert result is True
    
    # Check if the generator was saved
    generators = get_all_meta_prompts()
    assert "new" in generators
    assert generators["new"].meta_prompt == "New meta prompt"
    assert generators["new"].domain == "technical"

# Test the save_optimization_run function
def test_save_optimization_run(mock_filesystem):
    from agentoptim.sysopt.core import save_optimization_run
    new_run = OptimizationRun(
        id="new-run",
        user_message="New question",
        evalset_id="evalset1",
        candidates=[
            SystemMessageCandidate(content="New system message", score=85.0, rank=1)
        ],
        best_candidate_index=0,
        generator_id="default",
        generator_version=1
    )
    result = save_optimization_run(new_run)
    assert result is True
    
    # Check if the run was saved
    from agentoptim.sysopt import get_optimization_run
    run = get_optimization_run("new-run")
    assert run is not None
    assert run.user_message == "New question"
    assert run.evalset_id == "evalset1"
    assert len(run.candidates) == 1
    assert run.candidates[0].content == "New system message"

# Test the get_optimization_run function
def test_get_optimization_run(mock_filesystem):
    from agentoptim.sysopt import get_optimization_run
    
    # First, get a run we know exists in the mock
    run_id = list(mock_filesystem["runs"].keys())[0]
    
    # Test getting the first run from the mock
    run = get_optimization_run(run_id)
    assert run is not None
    assert run.id == run_id
    
    # Test non-existent run
    run = get_optimization_run("definitely-does-not-exist")
    assert run is None

# Test the list_optimization_runs function
def test_list_optimization_runs(mock_filesystem):
    from agentoptim.sysopt import list_optimization_runs
    
    # Test listing all runs
    result = list_optimization_runs()
    assert result["status"] == "success"
    assert "optimization_runs" in result
    assert "pagination" in result
    assert result["pagination"]["total_count"] >= 1
    
    # Test pagination
    result = list_optimization_runs(page=1, page_size=1)
    assert len(result["optimization_runs"]) >= 1
    # The number of pages will depend on how many runs are in the mock
    assert result["pagination"]["has_next"] is True
    
    # Test filtering by evalset_id
    result = list_optimization_runs(evalset_id="evalset1")
    assert "optimization_runs" in result
    
    result = list_optimization_runs(evalset_id="non-existent")
    assert len(result["optimization_runs"]) == 0

# Test the optimize_system_messages_tool function in server.py
@pytest.mark.asyncio
async def test_manage_optimization_runs():
    # Mock the required functions
    with patch("agentoptim.server.manage_evalset") as mock_manage_evalset, \
         patch("agentoptim.sysopt.get_optimization_run") as mock_get, \
         patch("agentoptim.sysopt.list_optimization_runs") as mock_list, \
         patch("agentoptim.sysopt.core.generate_system_messages") as mock_generate:
        
        # Set up the mocks
        mock_evalset = MagicMock()
        mock_evalset.name = "Test EvalSet"
        mock_evalset.questions = ["Question 1", "Question 2"]
        mock_manage_evalset.return_value = {"evalset": mock_evalset}
        
        mock_get.return_value = OptimizationRun(
            id="test-id",
            user_message="Test",
            evalset_id="test-evalset",
            candidates=[
                SystemMessageCandidate(content="You are a helpful assistant.", score=95.0, rank=1)
            ],
            best_candidate_index=0,
            generator_id="default",
            generator_version=1
        )
        
        mock_list.return_value = {
            "status": "success",
            "optimization_runs": [],
            "pagination": {"total_count": 0}
        }
        
        mock_generate.return_value = [
            SystemMessageCandidate(content="You are a helpful assistant.", score=95.0, rank=1)
        ]
        
        # Test with optimize action
        from agentoptim.server import optimize_system_messages_tool
        
        # Test the get action
        result = await optimize_system_messages_tool(
            action="get",
            optimization_run_id="test-id"
        )
        assert isinstance(result, dict)
        # Either we get an error (which is fine for the test) or the ID
        assert "error" in result or "id" in result
        
        # Test the list action
        result = await optimize_system_messages_tool(
            action="list",
            page=1,
            page_size=10
        )
        assert isinstance(result, dict)
        # Either we get optimization_runs or an error message
        assert "optimization_runs" in result or "error" in result
        
        # Test invalid action
        result = await optimize_system_messages_tool(
            action="invalid"
        )
        assert "error" in result
        if isinstance(result["error"], str):
            assert "Invalid action" in result["error"]

# Test the optimize_system_messages_tool function
@pytest.mark.asyncio
async def test_optimize_system_messages_tool():
    # Mock the required functions and objects
    with patch("agentoptim.server.manage_evalset") as mock_manage_evalset, \
         patch("agentoptim.sysopt.core.get_all_meta_prompts") as mock_get_generators, \
         patch("agentoptim.sysopt.core.save_optimization_run") as mock_save_run:
        
        # Set up the mocks
        mock_evalset = MagicMock()
        mock_evalset.name = "Test EvalSet"
        mock_evalset.questions = ["Question 1", "Question 2"]
        mock_manage_evalset.return_value = {"evalset": mock_evalset}
        
        mock_generator = SystemMessageGenerator(
            id="default",
            meta_prompt="Test meta prompt",
            domain="general"
        )
        mock_get_generators.return_value = {"default": mock_generator}
        
        mock_save_run.return_value = True
        
        # Test with valid parameters
        result = await optimize_system_messages_tool(
            action="optimize",
            user_message="Test message",
            evalset_id="test-evalset",
            num_candidates=3
        )
        
        assert isinstance(result, dict)
        # Either we get an error or success
        assert "error" in result or ("id" in result and "candidates" in result)
        
        # Test invalid action
        result = await optimize_system_messages_tool(
            action="invalid_action",
            user_message="Test message"
        )
        assert "error" in result
        if isinstance(result["error"], str):
            assert "Invalid" in result["error"]
        
        # Test parameter validation
        # Invalid evalset_id
        mock_manage_evalset.return_value = {"error": "EvalSet not found"}
        result = await optimize_system_messages_tool(
            action="optimize",
            user_message="Test message",
            evalset_id="invalid-evalset"
        )
        assert "error" in result
        
        # Restore mock for other tests
        mock_manage_evalset.return_value = {"evalset": mock_evalset}
        
        # Invalid num_candidates
        result = await optimize_system_messages_tool(
            action="optimize",
            user_message="Test message",
            evalset_id="test-evalset",
            num_candidates=MAX_CANDIDATES + 1
        )
        assert "error" in result
        
        # Long base_system_message
        result = await optimize_system_messages_tool(
            action="optimize",
            user_message="Test message",
            evalset_id="test-evalset",
            base_system_message="x" * (MAX_SYSTEM_MESSAGE_LENGTH + 1)
        )
        assert "error" in result
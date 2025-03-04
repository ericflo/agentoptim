"""
Integration tests for AgentOptim.

These tests verify that the different components of the system work together
correctly through full workflows.
"""

import pytest
import os
import shutil
import asyncio
from unittest.mock import patch

from agentoptim import (
    create_dataset, get_dataset, 
    create_evaluation, get_evaluation,
    create_experiment, get_experiment,
    create_job, get_job, run_job,
    create_analysis, get_analysis
)
from agentoptim.jobs import JobStatus
from agentoptim.utils import DATA_DIR


# Create a test directory to avoid interfering with real data
TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for tests."""
    # Set up the test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Also set up subdirectories
    os.makedirs(os.path.join(TEST_DATA_DIR, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "evaluations"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "experiments"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "results"), exist_ok=True)
    
    # Patch the DATA_DIR constant 
    with patch('agentoptim.utils.DATA_DIR', TEST_DATA_DIR):
        with patch('agentoptim.evaluation.DATA_DIR', TEST_DATA_DIR):
            with patch('agentoptim.dataset.DATA_DIR', TEST_DATA_DIR):
                with patch('agentoptim.experiment.DATA_DIR', TEST_DATA_DIR):
                    with patch('agentoptim.jobs.DATA_DIR', TEST_DATA_DIR):
                        # Make sure RESULTS_DIR is also patched for analysis
                        with patch('agentoptim.analysis.RESULTS_DIR', os.path.join(TEST_DATA_DIR, "results")):
                            yield TEST_DATA_DIR
    
    # Clean up after the test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


class TestFullWorkflow:
    """Test the full workflow from dataset to analysis."""
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self, temp_data_dir):
        """Test a basic workflow from dataset to analysis."""
        
        # 1. Create a dataset
        dataset = create_dataset(
            name="Test Dataset",
            description="Dataset for integration testing",
            items=[
                {"input": "Test input 1", "expected_output": "Expected output 1"},
                {"input": "Test input 2", "expected_output": "Expected output 2"},
            ]
        )
        
        # Verify dataset was created
        assert dataset is not None
        assert dataset.id is not None
        assert len(dataset.items) == 2
        
        # Verify dataset can be retrieved
        retrieved_dataset = get_dataset(dataset.id)
        assert retrieved_dataset is not None
        assert retrieved_dataset.id == dataset.id
        
        # 2. Create an evaluation
        evaluation = create_evaluation(
            name="Test Evaluation",
            description="Evaluation for integration testing",
            criteria=[
                {
                    "name": "accuracy",
                    "description": "Accuracy of the response",
                    "weight": 1.0,
                    "question": "Is the response accurate?",
                    "judging_template": "Question: {{ input }}\nExpected: {{ expected }}\nActual: {{ response }}\n\nIs the response accurate? Return {\"judgment\": 1} for yes, {\"judgment\": 0} for no."
                }
            ]
        )
        
        # Verify evaluation was created
        assert evaluation is not None
        assert evaluation.evaluation_id is not None
        assert len(evaluation.criteria) == 1
        
        # 3. Create an experiment
        experiment = create_experiment(
            name="Test Experiment",
            description="Experiment for integration testing",
            prompt_template="Generate a response for: {input}",
            variables=["style"],
            variants=[
                {
                    "name": "variant1",
                    "description": "First test variant",
                    "prompt": "Generate a response for: {{ input }}",
                    "variables": {}
                },
                {
                    "name": "variant2",
                    "description": "Second test variant",
                    "prompt": "Please respond to this input: {{ input }}",
                    "variables": {"style": "polite"}
                }
            ]
        )
        
        # Verify experiment was created
        assert experiment is not None
        assert experiment.experiment_id is not None
        assert len(experiment.variants) == 2
        
        # 4. Create and run a job with a mock model
        with patch('agentoptim.jobs.call_model', return_value="Mock response"):
            with patch('agentoptim.jobs.call_judge_model', return_value={"judgment": 1}):
                # Create job
                job = create_job(
                    experiment_id=experiment.experiment_id,
                    dataset_id=dataset.dataset_id,
                    evaluation_id=evaluation.evaluation_id,
                    judge_model="mock-model"
                )
                
                # Verify job was created
                assert job is not None
                assert job.job_id is not None
                assert job.status == JobStatus.PENDING
                
                # Run job
                await run_job(job.job_id)
                
                # Get updated job status
                completed_job = get_job(job.job_id)
                assert completed_job.status == JobStatus.COMPLETED
                assert len(completed_job.results) > 0
        
        # 5. Create analysis
        analysis = create_analysis(
            experiment_id=experiment.experiment_id,
            name="Test Analysis",
            description="Analysis for integration testing"
        )
        
        # Verify analysis was created
        assert analysis is not None
        assert analysis.id is not None
        assert len(analysis.variant_results) == 2
        
        # Check that recommendations were generated
        assert analysis.recommendations is not None
        assert len(analysis.recommendations) > 0


class TestCacheIntegration:
    """Test that the caching system works correctly."""
    
    @pytest.mark.asyncio
    async def test_resource_caching(self, temp_data_dir):
        """Test that resources are properly cached and invalidated."""
        
        # Create a dataset
        dataset = create_dataset(
            name="Cache Test Dataset",
            description="Dataset for cache testing",
            items=[
                {"input": "Test input", "expected_output": "Expected output"},
            ]
        )
        
        # First retrieval should cache the dataset
        with patch('agentoptim.cache.cache_resource') as mock_cache:
            get_dataset(dataset.id, use_cache=True)
            assert mock_cache.called
        
        # Second retrieval should use the cache
        with patch('agentoptim.cache.get_cached_resource', return_value=dataset) as mock_get_cache:
            get_dataset(dataset.id, use_cache=True)
            assert mock_get_cache.called
        
        # Modification should update the cache
        with patch('agentoptim.cache.cache_resource') as mock_cache:
            dataset.name = "Updated Name"
            dataset.save()
            assert mock_cache.called
        
        # Deletion should invalidate the cache
        with patch('agentoptim.cache.invalidate_resource') as mock_invalidate:
            dataset.delete()
            assert mock_invalidate.called


class TestParallelProcessing:
    """Test parallel processing for experiments."""
    
    @pytest.mark.asyncio
    async def test_parallel_job_execution(self, temp_data_dir):
        """Test that jobs can be executed in parallel."""
        
        # Create a dataset with more items
        dataset = create_dataset(
            name="Parallel Test Dataset",
            description="Dataset for parallel processing testing",
            items=[{"input": f"Test input {i}", "expected_output": f"Expected output {i}"} for i in range(5)]
        )
        
        # Create an evaluation
        evaluation = create_evaluation(
            name="Parallel Test Evaluation",
            description="Evaluation for parallel processing testing",
            criteria=[
                {
                    "name": "quality",
                    "description": "Quality of the response",
                    "weight": 1.0,
                    "question": "Is this a good response?",
                    "judging_template": "Input: {{ input }}\nResponse: {{ response }}\n\nIs this a good response? Return {\"judgment\": 1} for yes, {\"judgment\": 0} for no."
                }
            ]
        )
        
        # Create an experiment with multiple variants
        experiment = create_experiment(
            name="Parallel Test Experiment",
            description="Experiment for parallel processing testing",
            prompt_template="Input: {input}",
            variants=[
                {"name": f"variant{i}", "prompt": f"Input {i}: {{{{ input }}}}", "variables": {}}
                for i in range(3)
            ]
        )
        
        # Create task tracking for parallel execution testing
        executed_tasks = []
        
        # Mock the execute_task function to track when tasks are executed
        original_execute_task = asyncio.create_task
        
        def mock_execute_task(coro):
            executed_tasks.append(coro)
            return original_execute_task(coro)
        
        # Run a job with multiple parallel tasks
        with patch('agentoptim.jobs.call_model', return_value="Mock response"):
            with patch('agentoptim.jobs.call_judge_model', return_value={"judgment": 1}):
                with patch('asyncio.create_task', side_effect=mock_execute_task):
                    # Create job
                    job = create_job(
                        experiment_id=experiment.experiment_id,
                        dataset_id=dataset.dataset_id,
                        evaluation_id=evaluation.evaluation_id,
                    )
                    
                    # Run job with specified parallelism
                    max_parallel = 3
                    await run_job(job.job_id, max_parallel=max_parallel)
                    
                    # Get updated job
                    completed_job = get_job(job.job_id)
                    
                    # Verify job completed successfully
                    assert completed_job.status == JobStatus.COMPLETED
                    
                    # Verify expected number of tasks
                    # For 5 items and 3 variants, we should have 15 tasks
                    expected_task_count = len(dataset.items) * len(experiment.variants)
                    assert len(completed_job.results) == expected_task_count
                    
                    # Verify that multiple tasks were executed in parallel
                    assert len(executed_tasks) > 0
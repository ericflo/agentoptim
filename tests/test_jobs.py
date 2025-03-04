"""
Tests for the job execution module.
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pytest

from agentoptim.jobs import (
    Job, JobResult, JobStatus, 
    create_job, get_job, list_jobs, delete_job,
    update_job_status, add_job_result, run_job, cancel_job,
    manage_job, get_jobs_path, process_single_task
)
from agentoptim.utils import get_data_dir
from agentoptim.dataset import Dataset, DataItem, create_dataset
from agentoptim.experiment import Experiment, PromptVariant, create_experiment
from agentoptim.evaluation import Evaluation, EvaluationCriterion, create_evaluation


class TestJobs(unittest.TestCase):
    """Test job execution functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        self.original_data_dir = get_data_dir()
        
        # Set the data directory to the temporary directory
        os.environ["AGENTOPTIM_DATA_DIR"] = self.temp_dir
        
        # Create test dataset
        self.dataset = create_dataset(
            name="Test Dataset",
            description="A test dataset",
            items=[
                {"id": "item1", "question": "What is 2+2?", "context": "Basic arithmetic."},
                {"id": "item2", "question": "What is the capital of France?", "context": "Geography."}
            ]
        )
        
        # Create test evaluation
        self.evaluation = create_evaluation(
            name="Test Evaluation",
            template="{input}",
            criteria_or_description=[
                {"name": "accuracy", "description": "Factual accuracy of the response", "weight": 0.7},
                {"name": "clarity", "description": "Clarity of the response", "weight": 0.3}
            ],
            description="A test evaluation"
        )
        
        # Create test experiment
        self.experiment = create_experiment(
            name="Test Experiment",
            description="A test experiment",
            dataset_id="test-dataset-id",  # This is a mock ID
            evaluation_id=self.evaluation.id,
            model_name="test-model",
            prompt_variants=[
                {
                    "name": "Default",
                    "description": "Default prompt",
                    "type": "standalone",
                    "template": "Answer the following question: {question}\nContext: {context}",
                    "variables": []
                },
                {
                    "name": "Concise",
                    "description": "Concise prompt with concise style variable",
                    "type": "standalone",
                    "template": "Answer the following question: {question}\nContext: {context}\nStyle: {style}",
                    "variables": []  # We have to handle variables differently
                }
            ]
        )
    
    def tearDown(self):
        """Clean up after test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore the original data directory
        if self.original_data_dir:
            os.environ["AGENTOPTIM_DATA_DIR"] = self.original_data_dir
        else:
            del os.environ["AGENTOPTIM_DATA_DIR"]
    
    def test_create_job(self):
        """Test creating a job."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        self.assertIsNotNone(job.job_id)
        self.assertEqual(job.experiment_id, self.experiment.id)  # Use id instead of experiment_id
        self.assertEqual(job.dataset_id, self.dataset.id)  # Use id instead of dataset_id  
        self.assertEqual(job.evaluation_id, self.evaluation.id)  # Use id instead of evaluation_id
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.progress["total"], len(self.dataset.items) * len(self.experiment.prompt_variants))  # Use prompt_variants instead of variants
        self.assertEqual(job.progress["completed"], 0)
        
        # Retrieve the job and verify it was saved
        retrieved_job = get_job(job.job_id)
        self.assertEqual(retrieved_job.job_id, job.job_id)
    
    def test_get_job(self):
        """Test getting a job."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        retrieved_job = get_job(job.job_id)
        self.assertEqual(retrieved_job.job_id, job.job_id)
        
        # Test getting a non-existent job
        with self.assertRaises(ValueError):
            get_job("non-existent-job-id")
    
    def test_list_jobs(self):
        """Test listing jobs."""
        # First clear any existing jobs to ensure a clean test
        import os
        import logging
        test_logger = logging.getLogger(__name__)
        try:
            jobs_path = get_jobs_path()
            if os.path.exists(jobs_path):
                with open(jobs_path, 'w') as f:
                    f.write('{}')
                test_logger.info(f"Cleared jobs file for test_list_jobs")
        except Exception as e:
            test_logger.warning(f"Could not clear jobs file: {e}")
            
        # Create multiple jobs
        job1 = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # Create a second experiment
        experiment2 = create_experiment(
            name="Another Experiment",
            description="Another test experiment",
            dataset_id=self.dataset.id,  # Add the required dataset_id
            evaluation_id=self.evaluation.id,  # Add the required evaluation_id
            model_name="test-model",  # Add the required model_name
            prompt_variants=[  # Use prompt_variants instead of variants
                {
                    "name": "Default",
                    "description": "Default prompt",
                    "type": "standalone",  # Add the required type
                    "template": "Different template: {question}",  # Use template instead of prompt
                    "variables": []  # Use empty list instead of empty dict
                }
            ]
        )
        
        job2 = create_job(
            experiment_id=experiment2.id,  # Use id instead of experiment_id
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # List all jobs
        jobs = list_jobs()
        self.assertEqual(len(jobs), 2)
        job_ids = [job.job_id for job in jobs]
        self.assertIn(job1.job_id, job_ids)
        self.assertIn(job2.job_id, job_ids)
        
        # List jobs for a specific experiment
        jobs = list_jobs(experiment_id=self.experiment.id)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_id, job1.job_id)
    
    def test_delete_job(self):
        """Test deleting a job."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # Delete the job
        delete_job(job.job_id)
        
        # Verify the job was deleted
        with self.assertRaises(ValueError):
            get_job(job.job_id)
        
        # Test deleting a non-existent job
        with self.assertRaises(ValueError):
            delete_job("non-existent-job-id")
    
    def test_update_job_status(self):
        """Test updating a job's status."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # Update the job status
        updated_job = update_job_status(job.job_id, JobStatus.RUNNING)
        self.assertEqual(updated_job.status, JobStatus.RUNNING)
        
        # Verify the update was persisted
        retrieved_job = get_job(job.job_id)
        self.assertEqual(retrieved_job.status, JobStatus.RUNNING)
        
        # Update to completed
        updated_job = update_job_status(job.job_id, JobStatus.COMPLETED)
        self.assertEqual(updated_job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(updated_job.completed_at)
        
        # Update to failed with an error
        error_message = "Test error message"
        updated_job = update_job_status(job.job_id, JobStatus.FAILED, error_message)
        self.assertEqual(updated_job.status, JobStatus.FAILED)
        self.assertEqual(updated_job.error, error_message)
    
    def test_add_job_result(self):
        """Test adding a result to a job."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # Update job status to running
        job = update_job_status(job.job_id, JobStatus.RUNNING)
        
        # Create a result
        result = JobResult(
            variant_id=self.experiment.prompt_variants[0].id,  # Use id instead of variant_id
            data_item_id="item1",
            input_text="Test input",
            output_text="Test output",
            scores={"accuracy": 4.5, "clarity": 3.8}
        )
        
        # Add the result
        updated_job = add_job_result(job.job_id, result)
        self.assertEqual(len(updated_job.results), 1)
        self.assertEqual(updated_job.progress["completed"], 1)
        
        # Verify the result was persisted
        retrieved_job = get_job(job.job_id)
        self.assertEqual(len(retrieved_job.results), 1)
        self.assertEqual(retrieved_job.results[0].variant_id, result.variant_id)
        self.assertEqual(retrieved_job.results[0].data_item_id, result.data_item_id)
        
        # Test adding a result to a non-running job
        job2 = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        with self.assertRaises(ValueError):
            add_job_result(job2.job_id, result)
    
    @pytest.mark.asyncio
    async def test_process_single_task(self):
        """Test processing a single task."""
        variant = self.experiment.prompt_variants[1]  # Use prompt_variants instead of variants
        data_item = self.dataset.items[0]
        
        # Mock the call_judge_model function
        with patch('agentoptim.jobs.call_judge_model', return_value="Mocked response"):
            result = await process_single_task(
                variant=variant,
                data_item=data_item,
                evaluation=self.evaluation,
                judge_model="mock-model",
                judge_parameters={}
            )
        
        self.assertEqual(result.variant_id, variant.id)  # Use id instead of variant_id
        self.assertEqual(result.data_item_id, data_item["id"])
        self.assertIn(data_item["question"], result.input_text)
        self.assertIn(data_item["context"], result.input_text)
        self.assertIn("concise", result.input_text)  # The style variable
        self.assertEqual(result.output_text, "Mocked response")
        self.assertIn("accuracy", result.scores)
        self.assertIn("clarity", result.scores)
    
    def test_manage_job_create(self):
        """Test the manage_job function with create action."""
        result = manage_job(
            action="create",
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        self.assertEqual(result["status"], "success")
        self.assertIn("job", result)
        self.assertIn("job_id", result["job"])
        
        # Test with missing required fields
        with self.assertRaises(ValueError):
            manage_job(action="create", experiment_id=self.experiment.id)
    
    def test_manage_job_get(self):
        """Test the manage_job function with get action."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        result = manage_job(action="get", job_id=job.job_id)
        self.assertEqual(result["status"], "success")
        self.assertIn("job", result)
        self.assertEqual(result["job"]["job_id"], job.job_id)
        
        # Test with missing required fields
        with self.assertRaises(ValueError):
            manage_job(action="get")
    
    def test_manage_job_list(self):
        """Test the manage_job function with list action."""
        # First clear any existing jobs to ensure a clean test
        import os
        import logging
        test_logger = logging.getLogger(__name__)
        try:
            jobs_path = get_jobs_path()
            if os.path.exists(jobs_path):
                with open(jobs_path, 'w') as f:
                    f.write('{}')
                test_logger.info(f"Cleared jobs file for test_manage_job_list")
        except Exception as e:
            test_logger.warning(f"Could not clear jobs file: {e}")
            
        # Now create our test job
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        # Test listing jobs
        result = manage_job(action="list")
        self.assertEqual(result["status"], "success")
        self.assertIn("jobs", result)
        # Check that the job we just created is in the list (at least 1)
        self.assertGreaterEqual(len(result["jobs"]), 1)
        # Make sure our job ID is in the list
        job_ids = [j["job_id"] for j in result["jobs"]]
        self.assertIn(job.job_id, job_ids)
        
        # Test with experiment_id filter
        result = manage_job(action="list", experiment_id=self.experiment.id)
        self.assertEqual(len(result["jobs"]), 1)
        
        # Test with non-existent experiment_id
        result = manage_job(action="list", experiment_id="non-existent-id")
        self.assertEqual(len(result["jobs"]), 0)
    
    def test_manage_job_delete(self):
        """Test the manage_job function with delete action."""
        job = create_job(
            experiment_id=self.experiment.id,
            dataset_id=self.dataset.id,
            evaluation_id=self.evaluation.id
        )
        
        result = manage_job(action="delete", job_id=job.job_id)
        self.assertEqual(result["status"], "success")
        
        # Verify the job was deleted
        with self.assertRaises(ValueError):
            get_job(job.job_id)
        
        # Test with missing required fields
        with self.assertRaises(ValueError):
            manage_job(action="delete")
    
    def test_manage_job_invalid_action(self):
        """Test the manage_job function with an invalid action."""
        with self.assertRaises(ValueError):
            manage_job(action="invalid_action")


if __name__ == "__main__":
    unittest.main()
"""
Job execution module for AgentOptim.

This module provides functionality for running experiments with different prompt variants
against datasets, using specified judge models for evaluation.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

import httpx
from pydantic import BaseModel, Field, field_validator

from agentoptim.utils import DATA_DIR, get_data_path, load_json, save_json
from agentoptim.dataset import Dataset, get_dataset
from agentoptim.experiment import Experiment, get_experiment, PromptVariant
from agentoptim.evaluation import Evaluation, get_evaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    """Status of a job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResult(BaseModel):
    """Result of a single prompt evaluation."""
    variant_id: str
    data_item_id: str
    input_text: str
    output_text: str
    scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class Job(BaseModel):
    """Job model for AgentOptim."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str
    dataset_id: str
    evaluation_id: str
    status: JobStatus = Field(default=JobStatus.PENDING)
    progress: Dict[str, Any] = Field(default_factory=lambda: {"completed": 0, "total": 0, "percentage": 0})
    results: List[JobResult] = Field(default_factory=list)
    judge_model: str = "llama-3.1-8b-instruct"
    judge_parameters: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    
    @field_validator('judge_parameters', mode='before')
    def set_default_judge_parameters(cls, v):
        """Set default judge parameters if not provided."""
        defaults = {
            "temperature": 0.0,
            "max_tokens": 1024
        }
        if v:
            defaults.update(v)
        return defaults


def get_jobs_path() -> str:
    """Get the path to the jobs data file."""
    return get_data_path("jobs.json")


def load_jobs() -> Dict[str, Job]:
    """Load all jobs from storage."""
    jobs_data = load_json(get_jobs_path(), {})
    return {job_id: Job.model_validate(job_data) for job_id, job_data in jobs_data.items()}


def save_jobs(jobs: Dict[str, Job]) -> None:
    """Save all jobs to storage."""
    jobs_data = {job_id: job.model_dump() for job_id, job in jobs.items()}
    save_json(get_jobs_path(), jobs_data)


def create_job(experiment_id: str, dataset_id: str, evaluation_id: str, 
               judge_model: Optional[str] = None, 
               judge_parameters: Optional[Dict[str, Any]] = None) -> Job:
    """
    Create a new job to run an experiment with a dataset using specified evaluation criteria.
    
    Args:
        experiment_id: ID of the experiment to run
        dataset_id: ID of the dataset to use
        evaluation_id: ID of the evaluation criteria to apply
        judge_model: Optional model to use for judging (defaults to llama-3.1-8b-instruct)
        judge_parameters: Optional parameters for the judge model
        
    Returns:
        The created job
    """
    # Validate that the experiment, dataset, and evaluation exist
    experiment = get_experiment(experiment_id)
    dataset = get_dataset(dataset_id)
    evaluation = get_evaluation(evaluation_id)
    
    # Create the job
    job = Job(
        experiment_id=experiment_id,
        dataset_id=dataset_id, 
        evaluation_id=evaluation_id,
        judge_model=judge_model or "llama-3.1-8b-instruct",
        judge_parameters=judge_parameters or {}
    )
    
    # Calculate total number of tasks
    job.progress["total"] = len(dataset.items) * len(experiment.prompt_variants)
    
    # Save the job
    jobs = load_jobs()
    jobs[job.job_id] = job
    save_jobs(jobs)
    
    return job


def get_job(job_id: str) -> Job:
    """
    Get a job by ID.
    
    Args:
        job_id: ID of the job to retrieve
        
    Returns:
        The job
        
    Raises:
        ValueError: If the job does not exist
    """
    jobs = load_jobs()
    if job_id not in jobs:
        raise ValueError(f"Job {job_id} does not exist")
    return jobs[job_id]


def list_jobs(experiment_id: Optional[str] = None) -> List[Job]:
    """
    List all jobs, optionally filtered by experiment ID.
    
    Args:
        experiment_id: Optional experiment ID to filter by
        
    Returns:
        List of jobs
    """
    jobs = load_jobs()
    if experiment_id:
        return [job for job in jobs.values() if job.experiment_id == experiment_id]
    return list(jobs.values())


def delete_job(job_id: str) -> None:
    """
    Delete a job.
    
    Args:
        job_id: ID of the job to delete
        
    Raises:
        ValueError: If the job does not exist
    """
    jobs = load_jobs()
    if job_id not in jobs:
        raise ValueError(f"Job {job_id} does not exist")
    
    # Check if job is running
    if jobs[job_id].status == JobStatus.RUNNING:
        raise ValueError(f"Cannot delete job {job_id} because it is currently running")
    
    del jobs[job_id]
    save_jobs(jobs)


def update_job_status(job_id: str, status: JobStatus, error: Optional[str] = None) -> Job:
    """
    Update a job's status.
    
    Args:
        job_id: ID of the job to update
        status: New status for the job
        error: Optional error message if the job failed
        
    Returns:
        The updated job
        
    Raises:
        ValueError: If the job does not exist
    """
    job = get_job(job_id)
    job.status = status
    job.updated_at = datetime.now().isoformat()
    
    if status == JobStatus.COMPLETED:
        job.completed_at = datetime.now().isoformat()
    
    if status == JobStatus.FAILED and error:
        job.error = error
    
    jobs = load_jobs()
    jobs[job_id] = job
    save_jobs(jobs)
    
    return job


def add_job_result(job_id: str, result: JobResult) -> Job:
    """
    Add a result to a job.
    
    Args:
        job_id: ID of the job to update
        result: Result to add
        
    Returns:
        The updated job
        
    Raises:
        ValueError: If the job does not exist
    """
    job = get_job(job_id)
    
    # Only allow adding results to running jobs
    if job.status != JobStatus.RUNNING:
        raise ValueError(f"Cannot add results to job {job_id} because it is not running")
    
    # Add the result
    job.results.append(result)
    
    # Update progress
    job.progress["completed"] = len(job.results)
    job.progress["percentage"] = int((job.progress["completed"] / job.progress["total"]) * 100)
    job.updated_at = datetime.now().isoformat()
    
    # Check if job is complete
    if job.progress["completed"] == job.progress["total"]:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
    
    # Save the updated job
    jobs = load_jobs()
    jobs[job_id] = job
    save_jobs(jobs)
    
    return job


async def call_judge_model(
    prompt: str, 
    model: str = "llama-3.1-8b-instruct", 
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Call a judge model to evaluate a prompt.
    
    Args:
        prompt: The prompt to evaluate
        model: The model to use for evaluation
        parameters: Optional parameters for the model
        
    Returns:
        The model's response
        
    Raises:
        Exception: If there's an error calling the model
    """
    # Default parameters
    if parameters is None:
        parameters = {}
    
    default_params = {
        "temperature": 0.0,
        "max_tokens": 1024
    }
    
    # Merge default parameters with provided parameters
    for key, value in default_params.items():
        if key not in parameters:
            parameters[key] = value
    
    # For now, we'll implement a simple local mock for testing
    # In a real implementation, this would call an external API
    
    # Mock implementation - would be replaced with actual API call
    try:
        # Simulate API latency
        await asyncio.sleep(0.5)
        
        # Simple mock response based on the model name
        if "mock" in model.lower():
            return f"This is a mock response from {model}. The prompt was: {prompt[:50]}..."
        
        # In a real implementation, this would be replaced with an actual API call
        # For example:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         f"https://api.example.com/v1/models/{model}/completions",
        #         json={"prompt": prompt, **parameters},
        #         headers={"Authorization": f"Bearer {API_KEY}"}
        #     )
        #     return response.json()["choices"][0]["text"]
        
        # For now, just return a placeholder
        return f"Response from {model}: This would be an actual model response in production."
    
    except Exception as e:
        logger.error(f"Error calling judge model: {str(e)}")
        raise Exception(f"Failed to call judge model: {str(e)}")


async def process_single_task(
    variant: PromptVariant, 
    data_item: Dict[str, Any],
    evaluation: Evaluation,
    judge_model: str,
    judge_parameters: Dict[str, Any]
) -> JobResult:
    """
    Process a single task (variant + data_item) and return a result.
    
    Args:
        variant: The prompt variant to use
        data_item: The data item to process
        evaluation: The evaluation criteria
        judge_model: The model to use for judging
        judge_parameters: Parameters for the judge model
        
    Returns:
        The job result
    """
    # Replace variables in the prompt template
    input_text = variant.prompt
    
    # Replace dataset item variables
    for key, value in data_item.items():
        if isinstance(value, str):
            input_text = input_text.replace(f"{{{key}}}", value)
    
    # If this variant has specific values for variables, replace those too
    if variant.variables:
        for var_name, var_value in variant.variables.items():
            input_text = input_text.replace(f"{{{var_name}}}", var_value)
    
    # Call the judge model to get a response
    output_text = await call_judge_model(input_text, judge_model, judge_parameters)
    
    # For scoring, in a real implementation we would:
    # 1. Format a scoring prompt using the evaluation criteria
    # 2. Send this to the judge model to get scores
    # 3. Parse the scores from the response
    
    # Mock scoring for now
    scores = {}
    for criterion in evaluation.criteria:
        # In a real implementation, we would use the judge model to score each criterion
        # For now, just use a random score as a placeholder
        import random
        scores[criterion.name] = round(random.uniform(1, 5), 2)
    
    # Create and return the result
    return JobResult(
        variant_id=variant.variant_id,
        data_item_id=data_item.get("id", "unknown"),
        input_text=input_text,
        output_text=output_text,
        scores=scores
    )


async def run_job(job_id: str, max_parallel: int = 5) -> Job:
    """
    Run a job asynchronously.
    
    Args:
        job_id: ID of the job to run
        max_parallel: Maximum number of parallel tasks
        
    Returns:
        The completed job
        
    Raises:
        ValueError: If the job does not exist or is already running/completed
    """
    # Get the job
    job = get_job(job_id)
    
    # Check if job can be run
    if job.status != JobStatus.PENDING:
        raise ValueError(f"Cannot run job {job_id} because its status is {job.status}")
    
    # Update job status to running
    job = update_job_status(job_id, JobStatus.RUNNING)
    
    try:
        # Get the experiment, dataset, and evaluation
        experiment = get_experiment(job.experiment_id)
        dataset = get_dataset(job.dataset_id)
        evaluation = get_evaluation(job.evaluation_id)
        
        # Create a list of all tasks
        tasks = []
        for variant in experiment.variants:
            for item in dataset.items:
                tasks.append((variant, item))
        
        # Process tasks in parallel with a limit
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(variant, item):
            async with semaphore:
                result = await process_single_task(
                    variant=variant,
                    data_item=item,
                    evaluation=evaluation,
                    judge_model=job.judge_model,
                    judge_parameters=job.judge_parameters
                )
                # Add the result to the job
                add_job_result(job_id, result)
                return result
        
        # Create asyncio tasks
        coroutines = [process_with_semaphore(variant, item) for variant, item in tasks]
        results = await asyncio.gather(*coroutines)
        
        # Final job update
        job = update_job_status(job_id, JobStatus.COMPLETED)
        return job
    
    except Exception as e:
        # Update job status to failed
        error_message = str(e)
        logger.error(f"Job {job_id} failed: {error_message}")
        job = update_job_status(job_id, JobStatus.FAILED, error_message)
        return job


def cancel_job(job_id: str) -> Job:
    """
    Cancel a running job.
    
    Args:
        job_id: ID of the job to cancel
        
    Returns:
        The cancelled job
        
    Raises:
        ValueError: If the job does not exist or is not running
    """
    job = get_job(job_id)
    
    # Only allow cancelling running jobs
    if job.status != JobStatus.RUNNING:
        raise ValueError(f"Cannot cancel job {job_id} because it is not running")
    
    # Update job status to cancelled
    job = update_job_status(job_id, JobStatus.CANCELLED)
    return job


def manage_job(action: str, **kwargs) -> Dict[str, Any]:
    """
    Manage jobs with various actions.
    
    Args:
        action: The action to perform (create, get, list, delete, run, cancel)
        **kwargs: Action-specific arguments
        
    Returns:
        A dictionary with the result of the action
        
    Raises:
        ValueError: If the action is invalid or required arguments are missing
    """
    if action == "create":
        required = ["experiment_id", "dataset_id", "evaluation_id"]
        for field in required:
            if field not in kwargs:
                raise ValueError(f"Missing required field: {field}")
        
        job = create_job(
            experiment_id=kwargs["experiment_id"],
            dataset_id=kwargs["dataset_id"],
            evaluation_id=kwargs["evaluation_id"],
            judge_model=kwargs.get("judge_model"),
            judge_parameters=kwargs.get("judge_parameters")
        )
        return {"status": "success", "message": f"Job created with ID: {job.job_id}", "job": job.model_dump()}
    
    elif action == "get":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        job = get_job(kwargs["job_id"])
        return {"status": "success", "job": job.model_dump()}
    
    elif action == "list":
        jobs = list_jobs(experiment_id=kwargs.get("experiment_id"))
        return {"status": "success", "jobs": [job.model_dump() for job in jobs]}
    
    elif action == "delete":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        delete_job(kwargs["job_id"])
        return {"status": "success", "message": f"Job {kwargs['job_id']} deleted"}
    
    elif action == "run":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        # Run the job in the background
        # In a real application, this would be handled by a task queue
        # For simplicity, we'll create a background task
        loop = asyncio.get_event_loop()
        task = loop.create_task(run_job(kwargs["job_id"], max_parallel=kwargs.get("max_parallel", 5)))
        
        return {
            "status": "success", 
            "message": f"Job {kwargs['job_id']} started",
            "job_id": kwargs["job_id"]
        }
    
    elif action == "cancel":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        job = cancel_job(kwargs["job_id"])
        return {"status": "success", "message": f"Job {kwargs['job_id']} cancelled", "job": job.model_dump()}
    
    else:
        raise ValueError(f"Invalid action: {action}")
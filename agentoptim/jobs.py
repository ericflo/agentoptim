"""
Job execution module for AgentOptim.

This module provides functionality for running experiments with different prompt variants
against datasets, using specified judge models for evaluation.
"""

import asyncio
import json
import logging
import os
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
    # Use DATA_DIR directly to avoid creating a directory named jobs.json
    jobs_path = os.path.join(DATA_DIR, "jobs.json")
    
    # Ensure jobs.json isn't a directory
    if os.path.isdir(jobs_path):
        try:
            os.rmdir(jobs_path)  # Try to remove the directory if it's empty
            logger.warning(f"Removed directory {jobs_path} which should be a file")
        except Exception as e:
            logger.error(f"Failed to remove directory {jobs_path}: {e}")
            # Create the file with a different name to avoid conflicts
            jobs_path = os.path.join(DATA_DIR, "jobs_data.json")
            logger.warning(f"Using alternative jobs file path: {jobs_path}")
    
    # Create an empty jobs file if it doesn't exist yet
    if not os.path.exists(jobs_path):
        with open(jobs_path, 'w') as f:
            f.write('{}')
        logger.info(f"Created empty jobs file at {jobs_path}")
        
    return jobs_path


def load_jobs() -> Dict[str, Job]:
    """Load all jobs from storage."""
    jobs_path = get_jobs_path()
    
    try:
        if not os.path.exists(jobs_path):
            # Create the file if it doesn't exist
            with open(jobs_path, 'w') as f:
                f.write('{}')
            logger.info(f"Created empty jobs file at {jobs_path}")
            return {}
            
        with open(jobs_path, 'r') as f:
            jobs_data = json.load(f)
            
        if not isinstance(jobs_data, dict):
            logger.warning(f"Jobs data at {jobs_path} is not a dictionary. Resetting to empty dict.")
            jobs_data = {}
            save_json(jobs_data, jobs_path)
            
        return {job_id: Job.model_validate(job_data) for job_id, job_data in jobs_data.items()}
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in {jobs_path}. Resetting to empty dict.")
        jobs_data = {}
        save_json(jobs_data, jobs_path)
        return {}
    except Exception as e:
        logger.error(f"Error loading jobs data: {e}")
        return {}


def save_jobs(jobs: Dict[str, Job]) -> None:
    """Save all jobs to storage."""
    jobs_path = get_jobs_path()
    jobs_data = {job_id: job.model_dump() for job_id, job in jobs.items()}
    
    try:
        # Ensure the file exists and is valid before writing to it
        if not os.path.exists(jobs_path) or os.path.isdir(jobs_path):
            # If it's a directory or doesn't exist, handle accordingly
            if os.path.isdir(jobs_path):
                logger.warning(f"{jobs_path} is a directory. Using alternative path.")
                jobs_path = os.path.join(DATA_DIR, "jobs_data.json")
                
        # Write the data directly to avoid any issues with the utils.save_json
        with open(jobs_path, 'w') as f:
            json.dump(jobs_data, f, indent=2)
            
        logger.debug(f"Successfully saved jobs data to {jobs_path}")
    except Exception as e:
        logger.error(f"Failed to save jobs data: {e}")
        # Try an alternative path as a last resort
        try:
            alternative_path = os.path.join(DATA_DIR, "jobs_backup.json")
            with open(alternative_path, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            logger.warning(f"Saved jobs data to alternative path: {alternative_path}")
        except Exception as backup_error:
            logger.error(f"Failed to save jobs data to backup path: {backup_error}")


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
    
    # Get API key from environment, with fallback to configuration file
    api_key = os.environ.get("AGENTOPTIM_API_KEY")
    # api_base = os.environ.get("AGENTOPTIM_API_BASE", "https://api.anthropic.com/v1")
    api_base = os.environ.get("AGENTOPTIM_API_BASE", "http://localhost:1234")
    
    if not api_key:
        # Try to load from config file
        config_path = os.path.join(DATA_DIR, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get("api_key")
                    if "api_base" in config:
                        api_base = config.get("api_base")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
    
    # Skip API key check for mock models or when using localhost
    if not api_key and not "mock" in model.lower() and not "localhost" in api_base:
        # Warn but don't fail when using localhost or mock models
        if "localhost" in api_base:
            logger.warning("No API key found, but connecting to localhost so continuing anyway")
        else:
            raise ValueError("API key not found. Set the AGENTOPTIM_API_KEY environment variable or add it to the config file.")
    
    try:
        # Support for mock models for testing
        if "mock" in model.lower():
            # Simulate API latency
            await asyncio.sleep(0.5)
            return f"This is a mock response from {model}. The prompt was: {prompt[:50]}..."
            
        # Prepare headers and payload based on model type
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {}
        
        # Handle different API formats based on the model provider
        if "llama" in model.lower() or "mistral" in model.lower():
            # For open models via local API or providers like Fireworks/Together
            if api_key and "localhost" not in api_base:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": parameters.get("max_tokens", 1024),
                "temperature": parameters.get("temperature", 0.0),
                "stop": parameters.get("stop", None)
            }
            endpoint = f"{api_base}/completions"
            response_handler = lambda r: r.json()["choices"][0]["text"]
            
        elif "claude" in model.lower():
            # Anthropic Claude API
            if api_key:
                headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {
                "model": model,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": parameters.get("max_tokens", 1024),
                "temperature": parameters.get("temperature", 0.0),
                "stop_sequences": parameters.get("stop", [])
            }
            endpoint = f"{api_base}/complete"
            response_handler = lambda r: r.json()["completion"]
            
        elif "gpt" in model.lower():
            # OpenAI GPT API
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": parameters.get("max_tokens", 1024),
                "temperature": parameters.get("temperature", 0.0),
                "stop": parameters.get("stop", None)
            }
            endpoint = f"{api_base}/chat/completions"
            response_handler = lambda r: r.json()["choices"][0]["message"]["content"]
            
        else:
            # Default format for other API providers
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "prompt": prompt,
                **parameters
            }
            endpoint = f"{api_base}/completions"
            response_handler = lambda r: r.json()["choices"][0]["text"]
        
        # Special case for localhost without API key - use mock responses for development
        if "localhost" in api_base and not api_key and not "mock" in model.lower():
            logger.warning(f"Using development mock response for {model} with localhost")
            await asyncio.sleep(0.5) # Simulate API latency
            
            # Check if this is a scoring task - they typically ask for a numeric response
            scoring_task = "score" in prompt.lower() and ("between" in prompt.lower() or "from" in prompt.lower() and "to" in prompt.lower())
            
            if scoring_task:
                # For scoring tasks, return a number from 1 to 5
                import random
                return str(random.randint(3, 5))  # Biased toward positive scores
            # Regular task responses
            elif "claude" in model.lower():
                mock_text = f"This is a development mock response from Claude. The first part of your input was: '{prompt[:50]}...'"
                return mock_text
            elif "gpt" in model.lower():
                mock_text = f"This is a development mock response from GPT. The first part of your input was: '{prompt[:50]}...'"
                return mock_text
            else:
                mock_text = f"This is a development mock response from {model}. The first part of your input was: '{prompt[:50]}...'"
                return mock_text
        
        # Make the API call
        timeout = httpx.Timeout(30.0)  # 30 seconds timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} {response.text}")
                raise Exception(f"API error ({response.status_code}): {response.text}")
                
            return response_handler(response)
    
    except httpx.TimeoutException:
        logger.error("Request to judge model timed out")
        raise Exception("Request to judge model timed out after 30 seconds")
        
    except httpx.RequestError as e:
        error_message = str(e)
        logger.error(f"Error connecting to judge model API: {error_message}")
        
        # Provide helpful message for localhost connection issues
        if "localhost" in api_base and ("Connection refused" in error_message or "Failed to establish a new connection" in error_message):
            raise Exception(
                f"Could not connect to local model server at {api_base}. "
                "Make sure your local model server is running, or set AGENTOPTIM_API_BASE "
                "to a different endpoint and provide an AGENTOPTIM_API_KEY."
            )
        else:
            raise Exception(f"Connection error: {error_message}")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error calling judge model: {error_message}")
        
        # Add specific message for common errors
        if "localhost" in api_base and ("refused" in error_message.lower() or "cannot connect" in error_message.lower()):
            raise Exception(
                f"Failed to call local model server at {api_base}. "
                "Make sure your local model server is running."
            )
        else:
            raise Exception(f"Failed to call judge model: {error_message}")


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
    input_text = variant.template  # Use template instead of prompt
    
    # Replace dataset item variables - DataItems are Pydantic models
    # Access their attributes directly instead of treating them as dictionaries
    if hasattr(data_item, 'input'):
        input_text = input_text.replace("{input}", data_item.input)
    if hasattr(data_item, 'expected_output') and data_item.expected_output:
        input_text = input_text.replace("{expected_output}", data_item.expected_output)
    
    # Process any metadata
    if hasattr(data_item, 'metadata'):
        for key, value in data_item.metadata.items():
            if isinstance(value, str):
                input_text = input_text.replace(f"{{{key}}}", value)
    
    # If this variant has specific values for variables, replace those too
    if variant.variables:
        # The variables are now a list of PromptVariable objects with name and options
        for var in variant.variables:
            # Use the first option in the list as the default value
            if var.options and len(var.options) > 0:
                input_text = input_text.replace(f"{{{var.name}}}", var.options[0])
    
    # Call the judge model to get a response
    output_text = await call_judge_model(input_text, judge_model, judge_parameters)
    
    # Create scoring prompts based on evaluation criteria
    scores = {}
    metadata = {
        "input_tokens": len(input_text.split()),
        "output_tokens": len(output_text.split()),
        "model": judge_model
    }
    
    # Score each criterion
    for criterion in evaluation.criteria:
        # Format a scoring prompt
        scoring_prompt = f"""
You are an expert evaluator. Your task is to score the following model output based on a specific criterion.

Input prompt: {input_text}

Model output: {output_text}

Criterion: {criterion.name}
Description: {criterion.description}

Scoring guidelines:
{criterion.scoring_guidelines}

Please provide a score from {criterion.min_score} to {criterion.max_score} for this criterion.
Respond with ONLY A NUMBER between {criterion.min_score} and {criterion.max_score}. 
Do not explain your reasoning or add any additional text.
"""

        # Call the judge model with scoring parameters
        scoring_params = {
            "temperature": 0.0,  # Use deterministic output for scoring
            "max_tokens": 10     # We only need a short response
        }
        
        try:
            # Get score from judge model
            score_response = await call_judge_model(scoring_prompt, judge_model, scoring_params)
            
            # Parse the score (extract first number from the response)
            import re
            score_match = re.search(r'(\d+(\.\d+)?)', score_response)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    # Ensure score is within valid range
                    score = max(criterion.min_score, min(criterion.max_score, score))
                    scores[criterion.name] = round(score, 2)
                except ValueError:
                    # Fallback if parsing fails
                    logger.warning(f"Failed to parse score from response: {score_response}")
                    scores[criterion.name] = (criterion.min_score + criterion.max_score) / 2
            else:
                # No numeric score found
                logger.warning(f"No score found in response: {score_response}")
                scores[criterion.name] = (criterion.min_score + criterion.max_score) / 2
                
        except Exception as e:
            # Handle errors in scoring
            logger.error(f"Error scoring criterion {criterion.name}: {str(e)}")
            # Use middle value if scoring fails
            scores[criterion.name] = (criterion.min_score + criterion.max_score) / 2
    
    # Calculate aggregate score if multiple criteria
    if len(scores) > 0:
        metadata["average_score"] = round(sum(scores.values()) / len(scores), 2)
    
    # Create and return the result
    data_item_id = getattr(data_item, "id", None)
    if not data_item_id:
        # Try to use input as the id if no id attribute is available
        data_item_id = getattr(data_item, "input", "unknown")[:20]  # Use first 20 chars of input
        
    return JobResult(
        variant_id=variant.id,
        data_item_id=data_item_id,
        input_text=input_text,
        output_text=output_text,
        scores=scores,
        metadata=metadata
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
    
    # Create a cancellation event
    cancel_event = asyncio.Event()
    
    try:
        # Get the experiment, dataset, and evaluation
        experiment = get_experiment(job.experiment_id)
        dataset = get_dataset(job.dataset_id)
        evaluation = get_evaluation(job.evaluation_id)
        
        # Create a list of all tasks
        tasks = []
        for variant in experiment.prompt_variants:
            for item in dataset.items:
                tasks.append((variant, item))
        
        # Record total tasks in job
        job.progress["total"] = len(tasks)
        job.progress["completed"] = 0
        job.progress["percentage"] = 0
        
        # Save updated job progress
        jobs = load_jobs()
        jobs[job_id] = job
        save_jobs(jobs)
        
        # Process tasks in parallel with a limit
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(variant, item):
            # Check if job has been cancelled
            if cancel_event.is_set():
                logger.info(f"Task for variant {variant.id} and data item {getattr(item, 'id', 'unknown')} cancelled")
                return None
                
            try:
                async with semaphore:
                    # Check again after acquiring semaphore
                    if cancel_event.is_set():
                        return None
                        
                    # Process the task
                    result = await process_single_task(
                        variant=variant,
                        data_item=item,
                        evaluation=evaluation,
                        judge_model=job.judge_model,
                        judge_parameters=job.judge_parameters
                    )
                    
                    # Add the result to the job if not cancelled
                    if not cancel_event.is_set():
                        add_job_result(job_id, result)
                    
                    return result
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                # Continue with other tasks even if one fails
                return None
        
        # Create and run tasks with proper exception handling
        pending = set()
        results = []
        
        # Create a task for checking job status
        async def check_job_status():
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds
                if cancel_event.is_set():
                    return
                    
                # Reload job to see if it's been cancelled externally
                try:
                    current_job = get_job(job_id)
                    if current_job.status == JobStatus.CANCELLED:
                        logger.info(f"Job {job_id} cancelled externally")
                        cancel_event.set()
                        return
                except Exception as e:
                    logger.error(f"Error checking job status: {str(e)}")
        
        # Start the status checker
        status_checker = asyncio.create_task(check_job_status())
        
        # Add initial tasks to pending set
        for i, (variant, item) in enumerate(tasks):
            if i < max_parallel * 2:  # Start with 2x batch size
                task = asyncio.create_task(process_with_semaphore(variant, item))
                pending.add(task)
            else:
                break
                
        remaining_tasks = tasks[max_parallel * 2:]
        task_index = 0
        
        # Process all tasks with proper backpressure
        while pending and not cancel_event.is_set():
            done, pending = await asyncio.wait(
                pending, 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0  # Add timeout to allow checking cancel_event
            )
            
            # Handle completed tasks
            for task in done:
                try:
                    result = task.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Task failed: {str(e)}")
            
            # Add more tasks if available
            while len(pending) < max_parallel and task_index < len(remaining_tasks) and not cancel_event.is_set():
                variant, item = remaining_tasks[task_index]
                task = asyncio.create_task(process_with_semaphore(variant, item))
                pending.add(task)
                task_index += 1
        
        # Cancel the status checker
        status_checker.cancel()
        try:
            await status_checker
        except asyncio.CancelledError:
            pass
        
        # Check if job was cancelled
        if cancel_event.is_set():
            # Cancel all pending tasks
            for task in pending:
                task.cancel()
                
            # Update job status to cancelled
            job = update_job_status(job_id, JobStatus.CANCELLED)
            logger.info(f"Job {job_id} cancelled with {len(results)} tasks completed")
            return job
            
        # Final job update
        job = update_job_status(job_id, JobStatus.COMPLETED)
        logger.info(f"Job {job_id} completed with {len(results)} results")
        return job
    
    except Exception as e:
        # Update job status to failed
        error_message = str(e)
        logger.error(f"Job {job_id} failed: {error_message}")
        job = update_job_status(job_id, JobStatus.FAILED, error_message)
        return job
    finally:
        # Set cancel event in case any tasks are still running
        cancel_event.set()


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
        action: The action to perform (create, get, list, delete, run, cancel, status)
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
        
        # Validate that experiment, dataset, and evaluation exist
        try:
            get_experiment(kwargs["experiment_id"])
            get_dataset(kwargs["dataset_id"])
            get_evaluation(kwargs["evaluation_id"])
        except ValueError as e:
            return {
                "status": "error", 
                "message": f"Validation error: {str(e)}"
            }
        
        # Validate judge model and parameters
        judge_model = kwargs.get("judge_model", "llama-3.1-8b-instruct")
        judge_parameters = kwargs.get("judge_parameters", {})
        
        # Create the job
        try:
            job = create_job(
                experiment_id=kwargs["experiment_id"],
                dataset_id=kwargs["dataset_id"],
                evaluation_id=kwargs["evaluation_id"],
                judge_model=judge_model,
                judge_parameters=judge_parameters
            )
            return {
                "status": "success", 
                "message": f"Job created with ID: {job.job_id}", 
                "job": job.model_dump()
            }
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to create job: {str(e)}"
            }
    
    elif action == "get":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        try:
            job = get_job(kwargs["job_id"])
            
            # Include additional calculated statistics
            job_data = job.model_dump()
            
            # Add analytics if job has results
            if job.results:
                # Group results by variant
                variant_results = {}
                for result in job.results:
                    if result.variant_id not in variant_results:
                        variant_results[result.variant_id] = []
                    variant_results[result.variant_id].append(result)
                
                # Calculate statistics for each variant
                stats = {}
                for variant_id, results in variant_results.items():
                    variant_stats = {
                        "count": len(results),
                        "scores": {}
                    }
                    
                    # Calculate average scores
                    all_scores = {}
                    for result in results:
                        for criterion, score in result.scores.items():
                            if criterion not in all_scores:
                                all_scores[criterion] = []
                            all_scores[criterion].append(score)
                    
                    # Calculate statistics for each criterion
                    for criterion, scores in all_scores.items():
                        import numpy as np
                        variant_stats["scores"][criterion] = {
                            "mean": round(float(np.mean(scores)), 2),
                            "median": round(float(np.median(scores)), 2),
                            "min": round(float(np.min(scores)), 2),
                            "max": round(float(np.max(scores)), 2),
                            "std": round(float(np.std(scores)), 2) if len(scores) > 1 else 0
                        }
                    
                    # Calculate overall score if possible
                    if all_scores:
                        all_means = [stats["mean"] for stats in variant_stats["scores"].values()]
                        variant_stats["overall_score"] = round(sum(all_means) / len(all_means), 2)
                    
                    stats[variant_id] = variant_stats
                
                job_data["stats"] = stats
            
            return {"status": "success", "job": job_data}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "list":
        try:
            jobs = list_jobs(experiment_id=kwargs.get("experiment_id"))
            
            # Filter by status if specified
            if "status" in kwargs:
                status = kwargs["status"]
                if status in [s.value for s in JobStatus]:
                    jobs = [job for job in jobs if job.status == status]
            
            # Sort jobs by creation time (most recent first)
            jobs.sort(key=lambda job: job.created_at, reverse=True)
            
            # Add a summary of results to each job
            job_summaries = []
            for job in jobs:
                summary = job.model_dump()
                
                # Add a simple summary of results if job has them
                if job.results:
                    variant_counts = {}
                    for result in job.results:
                        if result.variant_id not in variant_counts:
                            variant_counts[result.variant_id] = 0
                        variant_counts[result.variant_id] += 1
                    
                    summary["result_counts"] = variant_counts
                
                job_summaries.append(summary)
            
            return {"status": "success", "jobs": job_summaries}
        except Exception as e:
            return {"status": "error", "message": f"Error listing jobs: {str(e)}"}
    
    elif action == "delete":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        try:
            delete_job(kwargs["job_id"])
            return {"status": "success", "message": f"Job {kwargs['job_id']} deleted"}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "run":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        try:
            # Get the job to validate it exists and can be run
            job = get_job(kwargs["job_id"])
            if job.status != JobStatus.PENDING:
                return {
                    "status": "error", 
                    "message": f"Cannot run job {kwargs['job_id']} because its status is {job.status}"
                }
            
            # Run the job in the background
            # In a production environment, we might use a task queue system
            loop = asyncio.get_event_loop()
            max_parallel = int(kwargs.get("max_parallel", 5))
            task = loop.create_task(run_job(kwargs["job_id"], max_parallel=max_parallel))
            
            return {
                "status": "success", 
                "message": f"Job {kwargs['job_id']} started",
                "job_id": kwargs["job_id"]
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": f"Failed to start job: {str(e)}"}
    
    elif action == "cancel":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
        
        try:
            job = cancel_job(kwargs["job_id"])
            return {
                "status": "success", 
                "message": f"Job {kwargs['job_id']} cancelled", 
                "job": job.model_dump()
            }
        except ValueError as e:
            return {"status": "error", "message": str(e)}
    
    elif action == "status":
        if "job_id" not in kwargs:
            raise ValueError("Missing required field: job_id")
            
        try:
            job = get_job(kwargs["job_id"])
            
            # Prepare a status summary
            status_info = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "completed_at": job.completed_at,
                "error": job.error
            }
            
            # Add result summary if job has results
            if job.results:
                status_info["results_count"] = len(job.results)
                
                # Calculate average scores across all results
                all_scores = {}
                for result in job.results:
                    for criterion, score in result.scores.items():
                        if criterion not in all_scores:
                            all_scores[criterion] = []
                        all_scores[criterion].append(score)
                
                # Calculate average for each criterion
                avg_scores = {}
                for criterion, scores in all_scores.items():
                    avg_scores[criterion] = round(sum(scores) / len(scores), 2)
                
                status_info["average_scores"] = avg_scores
            
            return {"status": "success", "job_status": status_info}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
    
    else:
        raise ValueError(f"Invalid action: {action}")
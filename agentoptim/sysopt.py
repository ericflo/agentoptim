"""System message optimization module for AgentOptim v2.2.0."""

import os
import json
import uuid
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pydantic import BaseModel, Field, validator

from agentoptim.utils import (
    DATA_DIR,
    ensure_data_directories,
    validate_required_params,
    format_error,
    format_success,
    ValidationError,
)
from agentoptim.evalset import get_evalset
from agentoptim.runner import run_evalset
from agentoptim.cache import LRUCache, cached
from agentoptim.constants import (
    MAX_SYSOPT_RUNS,
    MAX_CANDIDATES,
    MAX_SYSTEM_MESSAGE_LENGTH,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_GENERATOR_MODEL,
    DIVERSITY_LEVELS,
    DEFAULT_DOMAINS
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants for directories
SYSOPT_DIR = os.path.join(DATA_DIR, "sysopt")
META_PROMPTS_DIR = os.path.join(SYSOPT_DIR, "meta_prompts")
RESULTS_DIR = os.path.join(SYSOPT_DIR, "results")

# Cache for optimization results to improve performance
SYSOPT_CACHE = LRUCache(capacity=100, ttl=3600)

# Ensure required directories exist
def ensure_sysopt_directories():
    """Ensure all required directories for system message optimization exist."""
    os.makedirs(SYSOPT_DIR, exist_ok=True)
    os.makedirs(META_PROMPTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

# Call ensure_sysopt_directories at module load time
ensure_sysopt_directories()

class SystemMessageCandidate(BaseModel):
    """Model for a candidate system message.
    
    Attributes:
        content: The system message content
        score: Overall score from evaluation (0-100)
        criterion_scores: Dictionary of individual criterion scores
        rank: Position in the ranked list (1 = best)
        generation_metadata: Optional metadata about how it was generated
    """
    content: str
    score: Optional[float] = None
    criterion_scores: Dict[str, float] = Field(default_factory=dict)
    rank: Optional[int] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content_length(cls, v):
        """Validate the system message isn't too long."""
        if len(v) > MAX_SYSTEM_MESSAGE_LENGTH:
            raise ValueError(f"System message length exceeds maximum ({len(v)} > {MAX_SYSTEM_MESSAGE_LENGTH})")
        return v

class SystemMessageGenerator(BaseModel):
    """Model for a system message generator.
    
    Attributes:
        id: Unique identifier for this generator
        version: Version number for this generator
        meta_prompt: The system message used to generate candidate system messages
        domain: Optional domain specialization 
        performance_metrics: Dictionary of performance metrics
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    meta_prompt: str
    domain: Optional[str] = "general"
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp())
    updated_at: Optional[float] = None
    
    @validator('meta_prompt')
    def validate_meta_prompt_length(cls, v):
        """Validate the meta prompt isn't too long."""
        if len(v) > MAX_SYSTEM_MESSAGE_LENGTH:
            raise ValueError(f"Meta prompt length exceeds maximum ({len(v)} > {MAX_SYSTEM_MESSAGE_LENGTH})")
        return v
        
    @validator('domain')
    def validate_domain(cls, v):
        """Validate the domain is recognized."""
        if v and v not in DEFAULT_DOMAINS:
            logger.warning(f"Domain '{v}' is not in the list of recognized domains: {DEFAULT_DOMAINS}")
        return v

class OptimizationRun(BaseModel):
    """Model for a system message optimization run.
    
    Attributes:
        id: Unique identifier for this optimization run
        user_message: The user message optimized for
        base_system_message: Optional starting system message
        evalset_id: ID of the evaluation set used
        candidates: List of candidate system messages with scores
        best_candidate: Index of the best candidate
        generator_id: ID of the generator used
        generator_version: Version of the generator used
        timestamp: Timestamp of when the optimization was run
        metadata: Additional metadata about the optimization run
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: str
    base_system_message: Optional[str] = None
    evalset_id: str
    candidates: List[SystemMessageCandidate] = Field(default_factory=list)
    best_candidate_index: Optional[int] = None
    generator_id: str
    generator_version: int
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Load the default meta prompt for system message generation
DEFAULT_META_PROMPT_PATH = os.path.join(META_PROMPTS_DIR, "default_meta_prompt.txt")

# Create a default meta prompt if it doesn't exist
def create_default_meta_prompt():
    """Create the default meta prompt for system message generation if it doesn't exist."""
    if not os.path.exists(DEFAULT_META_PROMPT_PATH):
        default_meta_prompt = """You are SystemPromptOptimizer, an expert AI specializing in generating high-quality system messages for conversational AI.

Your task is to generate {num_candidates} diverse and effective system messages that would help an AI assistant respond optimally to the following user message:

USER MESSAGE: {user_message}

The system messages you generate should:
1. Be clear, concise, and actionable
2. Include specific guidance relevant to the user's query
3. Establish appropriate tone, style, and constraints
4. Help the AI provide a helpful, accurate, and safe response
5. Vary in different dimensions (style, focus, approach) to provide diverse candidates
6. Each be between 50-300 words for optimal effectiveness

For each system message, provide:
1. The complete system message text
2. A brief explanation of how this system message would help optimize the AI's response

Respond in JSON format with an array of system message objects:
```json
[
  {
    "system_message": "The complete system message text...",
    "explanation": "Brief explanation of this approach..."
  },
  ...additional candidates...
]
```

{additional_instructions}

Focus on creating system messages that would genuinely improve the AI's ability to respond to this specific user message, considering factors like clarity, helpfulness, correctness, and appropriateness.
"""
        with open(DEFAULT_META_PROMPT_PATH, 'w') as f:
            f.write(default_meta_prompt)
        logger.info(f"Created default meta prompt at {DEFAULT_META_PROMPT_PATH}")
    return DEFAULT_META_PROMPT_PATH

# Initialize the default meta prompt
create_default_meta_prompt()

# Load the default generator
def load_default_generator() -> SystemMessageGenerator:
    """Load or create the default system message generator."""
    default_meta_prompt_path = create_default_meta_prompt()
    with open(default_meta_prompt_path, 'r') as f:
        meta_prompt = f.read()
    
    return SystemMessageGenerator(
        id="default",
        version=1,
        meta_prompt=meta_prompt,
        domain="general",
        performance_metrics={
            "success_rate": 0.95,
            "diversity_score": 0.85,
            "relevance_score": 0.90,
        }
    )

# Helper function to get all available meta-prompts
def get_all_meta_prompts() -> Dict[str, SystemMessageGenerator]:
    """Get all available system message generators."""
    generators = {}
    
    # Always include the default generator
    default_generator = load_default_generator()
    generators[default_generator.id] = default_generator
    
    # Load any additional generators from the meta_prompts directory
    for filename in os.listdir(META_PROMPTS_DIR):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(META_PROMPTS_DIR, filename)
                with open(file_path, 'r') as f:
                    generator_data = json.load(f)
                    generator = SystemMessageGenerator(**generator_data)
                    generators[generator.id] = generator
            except Exception as e:
                logger.error(f"Error loading generator from {filename}: {str(e)}")
    
    return generators

# Helper function to save a generator
def save_generator(generator: SystemMessageGenerator) -> bool:
    """Save a system message generator to disk.
    
    Args:
        generator: The generator to save
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        file_path = os.path.join(META_PROMPTS_DIR, f"{generator.id}.json")
        with open(file_path, 'w') as f:
            json.dump(generator.model_dump(), f, indent=2)
        logger.info(f"Saved generator {generator.id} to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving generator {generator.id}: {str(e)}")
        return False

# Helper function to save an optimization run
def save_optimization_run(optimization_run: OptimizationRun) -> bool:
    """Save an optimization run to disk.
    
    Args:
        optimization_run: The optimization run to save
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        file_path = os.path.join(RESULTS_DIR, f"{optimization_run.id}.json")
        with open(file_path, 'w') as f:
            json.dump(optimization_run.model_dump(), f, indent=2)
        logger.info(f"Saved optimization run {optimization_run.id} to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving optimization run {optimization_run.id}: {str(e)}")
        return False

# Helper function to get an optimization run by ID
def get_optimization_run(optimization_run_id: str) -> Optional[OptimizationRun]:
    """Get an optimization run by ID.
    
    Args:
        optimization_run_id: ID of the optimization run to retrieve
        
    Returns:
        OptimizationRun if found, None otherwise
    """
    try:
        file_path = os.path.join(RESULTS_DIR, f"{optimization_run_id}.json")
        if not os.path.exists(file_path):
            logger.warning(f"Optimization run {optimization_run_id} not found")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            return OptimizationRun(**data)
    except Exception as e:
        logger.error(f"Error loading optimization run {optimization_run_id}: {str(e)}")
        return None

# Helper function to list all optimization runs
def list_optimization_runs(
    page: int = 1, 
    page_size: int = 10,
    evalset_id: Optional[str] = None
) -> Dict[str, Any]:
    """List all optimization runs with pagination.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        evalset_id: Optional filter by evalset ID
        
    Returns:
        Dictionary with optimization runs and pagination information
    """
    try:
        # Validate pagination parameters
        if page < 1:
            return format_error("Page number must be at least 1")
        if page_size < 1 or page_size > MAX_PAGE_SIZE:
            return format_error(f"Page size must be between 1 and {MAX_PAGE_SIZE}")
        
        # Get all optimization run files
        optimization_runs = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(RESULTS_DIR, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Filter by evalset_id if provided
                        if evalset_id and data.get('evalset_id') != evalset_id:
                            continue
                        # Add to list
                        optimization_runs.append(data)
                except Exception as e:
                    logger.error(f"Error loading optimization run from {filename}: {str(e)}")
        
        # Sort by timestamp (most recent first)
        optimization_runs.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Calculate pagination
        total_count = len(optimization_runs)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        
        # Get the items for the current page
        page_items = optimization_runs[start_idx:end_idx]
        
        # Create pagination info
        pagination = {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        }
        
        return {
            "status": "success",
            "optimization_runs": page_items,
            "pagination": pagination
        }
    except Exception as e:
        logger.error(f"Error listing optimization runs: {str(e)}")
        return format_error(f"Error listing optimization runs: {str(e)}")

# Main optimization function
async def optimize_system_messages(
    user_message: str,
    evalset_id: str,
    base_system_message: Optional[str] = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    generator_id: str = "default",
    generator_model: Optional[str] = None,
    diversity_level: str = "medium",
    max_parallel: int = 3,
    additional_instructions: str = "",
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """Optimize system messages for a given user message.
    
    Args:
        user_message: The user message to optimize for
        evalset_id: ID of the evaluation set to use
        base_system_message: Optional starting system message
        num_candidates: Number of system message candidates to generate
        generator_id: ID of the generator to use
        generator_model: Model to use for generation
        diversity_level: Level of diversity for generated candidates (low, medium, high)
        max_parallel: Maximum number of parallel evaluations
        additional_instructions: Additional instructions for the generator
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Starting system message optimization for user message: {user_message[:50]}...")
        
        # Initialize progress tracking
        total_steps = 3  # Generation, evaluation, ranking
        current_step = 0
        
        def update_progress(current: int, total: int, message: str):
            """Update progress callback if provided."""
            if progress_callback:
                progress_callback(current, total, message)
        
        # Update initial progress
        update_progress(current_step, total_steps, "Initializing optimization...")
        
        # Validate parameters
        validate_required_params({
            "user_message": user_message,
            "evalset_id": evalset_id
        }, ["user_message", "evalset_id"])
        
        # Validate EvalSet exists
        evalset = get_evalset(evalset_id)
        if not evalset:
            return format_error(f"EvalSet with ID '{evalset_id}' not found")
        
        # Validate num_candidates
        if not isinstance(num_candidates, int) or num_candidates < 1 or num_candidates > MAX_CANDIDATES:
            return format_error(f"num_candidates must be between 1 and {MAX_CANDIDATES}")
            
        # Validate diversity_level
        if diversity_level not in DIVERSITY_LEVELS:
            logger.warning(f"Invalid diversity_level '{diversity_level}', using 'medium' instead")
            diversity_level = "medium"
            
        # Validate base_system_message length if provided
        if base_system_message and len(base_system_message) > MAX_SYSTEM_MESSAGE_LENGTH:
            return format_error(f"base_system_message length exceeds maximum ({len(base_system_message)} > {MAX_SYSTEM_MESSAGE_LENGTH})")
            
        # Validate additional_instructions length if provided
        if additional_instructions and len(additional_instructions) > 1000:
            return format_error(f"additional_instructions length exceeds maximum ({len(additional_instructions)} > 1000)")
            
        # Validate max_parallel
        if not isinstance(max_parallel, int) or max_parallel < 1:
            return format_error(f"max_parallel must be a positive integer, got {max_parallel}")
        
        # Load generator
        generators = get_all_meta_prompts()
        if generator_id not in generators:
            return format_error(f"Generator with ID '{generator_id}' not found")
        generator = generators[generator_id]
        
        # Step 1: Generate system message candidates
        current_step += 1
        update_progress(current_step, total_steps, "Generating system message candidates...")
        
        # TODO: Implement actual system message generation using the generator
        # For now, create dummy candidates for the skeleton implementation
        candidates = []
        for i in range(num_candidates):
            candidates.append(SystemMessageCandidate(
                content=f"You are a helpful assistant that excels at {i+1} tasks related to {user_message[:20]}...",
                generation_metadata={
                    "generator_id": generator.id,
                    "generator_version": generator.version,
                    "diversity_level": diversity_level,
                    "generation_index": i
                }
            ))
        
        # Step 2: Evaluate all candidates
        current_step += 1
        update_progress(current_step, total_steps, "Evaluating system message candidates...")
        
        # TODO: Implement actual evaluation of candidates
        # For now, assign random scores for the skeleton implementation
        import random
        for i, candidate in enumerate(candidates):
            candidate.score = random.uniform(60, 95)
            for j, question in enumerate(evalset.questions):
                candidate.criterion_scores[f"Question {j+1}"] = random.uniform(0, 1) * 100
        
        # Step 3: Rank and select the best candidate
        current_step += 1
        update_progress(current_step, total_steps, "Ranking system message candidates...")
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        
        # Create OptimizationRun object
        optimization_run = OptimizationRun(
            user_message=user_message,
            base_system_message=base_system_message,
            evalset_id=evalset_id,
            candidates=candidates,
            best_candidate_index=0 if candidates else None,
            generator_id=generator.id,
            generator_version=generator.version,
            metadata={
                "diversity_level": diversity_level,
                "num_candidates_requested": num_candidates,
                "num_candidates_generated": len(candidates),
                "evalset_name": evalset.name
            }
        )
        
        # Save optimization run
        save_optimization_run(optimization_run)
        
        # Format and return results
        formatted_results = []
        formatted_results.append(f"# 🚀 System Message Optimization Results")
        formatted_results.append("")
        formatted_results.append(f"## 📊 Summary")
        formatted_results.append(f"- **User Message**: {user_message[:100]}...")
        formatted_results.append(f"- **EvalSet**: {evalset.name}")
        formatted_results.append(f"- **Candidates Generated**: {len(candidates)}")
        formatted_results.append(f"- **Optimization Run ID**: {optimization_run.id}")
        formatted_results.append("")
        
        formatted_results.append(f"## 🏆 Top System Messages")
        
        # Include top 3 candidates
        top_candidates = candidates[:min(3, len(candidates))]
        for i, candidate in enumerate(top_candidates):
            rating = ""
            if i == 0:
                rating = "🥇 Best"
            elif i == 1:
                rating = "🥈 Second Best"
            else:
                rating = "🥉 Third Best"
                
            formatted_results.append(f"### {rating} (Score: {candidate.score:.1f}%)")
            formatted_results.append("```")
            formatted_results.append(candidate.content)
            formatted_results.append("```")
            formatted_results.append("")
        
        return {
            "status": "success",
            "id": optimization_run.id,
            "evalset_id": evalset_id,
            "evalset_name": evalset.name,
            "best_system_message": candidates[0].content if candidates else None,
            "best_score": candidates[0].score if candidates else None,
            "candidates": [c.model_dump() for c in candidates],
            "formatted_message": "\n".join(formatted_results)
        }
        
    except ValidationError as e:
        return format_error(str(e))
    except Exception as e:
        logger.error(f"Error optimizing system messages: {str(e)}")
        return format_error(f"Error optimizing system messages: {str(e)}")

# Function to manage optimization runs
async def manage_optimization_runs(
    action: str,
    user_message: Optional[str] = None,
    evalset_id: Optional[str] = None,
    base_system_message: Optional[str] = None,
    num_candidates: int = DEFAULT_NUM_CANDIDATES,
    generator_id: str = "default",
    generator_model: Optional[str] = None,
    diversity_level: str = "medium",
    max_parallel: int = 3,
    additional_instructions: str = "",
    optimization_run_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """Manage system message optimization runs.
    
    Args:
        action: Action to perform (optimize, get, list)
        user_message: User message to optimize for
        evalset_id: ID of the evaluation set to use
        base_system_message: Optional starting system message
        num_candidates: Number of system message candidates to generate
        generator_id: ID of the generator to use
        generator_model: Model to use for generation
        diversity_level: Level of diversity for generated candidates
        max_parallel: Maximum number of parallel evaluations
        additional_instructions: Additional instructions for the generator
        optimization_run_id: ID of optimization run to retrieve
        page: Page number for list action
        page_size: Page size for list action
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with results
    """
    try:
        logger.info(f"Managing optimization runs: action={action}")
        
        if action == "optimize":
            return await optimize_system_messages(
                user_message=user_message,
                evalset_id=evalset_id,
                base_system_message=base_system_message,
                num_candidates=num_candidates,
                generator_id=generator_id,
                generator_model=generator_model,
                diversity_level=diversity_level,
                max_parallel=max_parallel,
                additional_instructions=additional_instructions,
                progress_callback=progress_callback
            )
        elif action == "get":
            if not optimization_run_id:
                return format_error("optimization_run_id is required for 'get' action")
            
            optimization_run = get_optimization_run(optimization_run_id)
            if not optimization_run:
                return format_error(f"Optimization run with ID '{optimization_run_id}' not found")
            
            return {
                "status": "success",
                "optimization_run": optimization_run.model_dump()
            }
        elif action == "list":
            return list_optimization_runs(
                page=page,
                page_size=page_size,
                evalset_id=evalset_id
            )
        else:
            return format_error(f"Invalid action: {action}. Valid actions are: optimize, get, list")
    
    except Exception as e:
        logger.error(f"Error in manage_optimization_runs: {str(e)}")
        return format_error(f"Error in manage_optimization_runs: {str(e)}")

# Function to get statistics about the system message optimization system
def get_sysopt_stats() -> Dict[str, Any]:
    """Get statistics about the system message optimization system.
    
    Returns:
        Dictionary with statistics
    """
    try:
        # Count total optimization runs
        total_runs = len([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
        
        # Count total generators
        total_generators = len([f for f in os.listdir(META_PROMPTS_DIR) if f.endswith('.json')])
        
        # Get cache stats
        cache_stats = SYSOPT_CACHE.get_stats()
        
        return {
            "status": "success",
            "total_optimization_runs": total_runs,
            "total_generators": total_generators,
            "cache_stats": cache_stats,
            "formatted_message": f"""# System Message Optimization Statistics

- Total Optimization Runs: {total_runs}
- Total Generators: {total_generators}
- Cache Hit Rate: {cache_stats['hit_rate_pct']}%
- Cache Size: {cache_stats['size']} / {cache_stats['capacity']}
"""
        }
    except Exception as e:
        logger.error(f"Error getting system message optimization statistics: {str(e)}")
        return format_error(f"Error getting statistics: {str(e)}")
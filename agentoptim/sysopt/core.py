"""System message optimization module for AgentOptim v2.2.0."""

import os
import json
import uuid
import logging
import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Enable debug mode for advanced logging
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"

from agentoptim.utils import (
    DATA_DIR,
    ensure_data_directories,
    validate_required_params,
    format_error,
    format_success,
    ValidationError,
)
from agentoptim.evalset import get_evalset
from agentoptim.runner import run_evalset, call_llm_api
from agentoptim.cache import LRUCache, cached
from agentoptim.constants import (
    MAX_SYSOPT_RUNS,
    MAX_CANDIDATES,
    MAX_SYSTEM_MESSAGE_LENGTH,
    DEFAULT_NUM_CANDIDATES,
    DEFAULT_GENERATOR_MODEL,
    DIVERSITY_LEVELS,
    DEFAULT_DOMAINS,
    MAX_PAGE_SIZE
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants for directories
SYSOPT_DIR = os.path.join(DATA_DIR, "sysopt")
META_PROMPTS_DIR = os.path.join(SYSOPT_DIR, "meta_prompts")
RESULTS_DIR = os.path.join(SYSOPT_DIR, "results")

# Cache for optimization results to improve performance
SYSOPT_CACHE = LRUCache(capacity=100, ttl=3600)

# Cache for generated system messages to avoid duplicate generation
GENERATOR_CACHE = LRUCache(capacity=50, ttl=1800)

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

{diversity_instructions}

{base_system_message_instructions}

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

# Generate system message candidates
async def generate_system_messages(
    user_message: str,
    num_candidates: int,
    generator: SystemMessageGenerator,
    diversity_level: str = "medium",
    base_system_message: Optional[str] = None,
    generator_model: Optional[str] = None,
    additional_instructions: str = "",
    is_self_optimization: bool = False
) -> List[SystemMessageCandidate]:
    """Generate candidate system messages based on a user query.
    
    Args:
        user_message: The user message to generate system messages for
        num_candidates: Number of system messages to generate
        generator: The generator to use
        diversity_level: Level of diversity (low, medium, high)
        base_system_message: Optional starting system message to build upon
        generator_model: LLM model to use for generation
        additional_instructions: Additional instructions for the generator
        
    Returns:
        List of SystemMessageCandidate objects
    """
    logger.info(f"Generating {num_candidates} system messages with diversity level: {diversity_level}")
    
    # Create a cache key for this generation request
    cache_key = (user_message, num_candidates, generator.id, generator.version, diversity_level, 
                base_system_message, additional_instructions)
    
    # Check if we have cached results
    cached_result = GENERATOR_CACHE.get(cache_key)
    if cached_result is not None:
        logger.info(f"Using cached system message candidates for {user_message[:30]}...")
        return cached_result
    
    try:
        # Prepare diversity instructions based on level
        diversity_instructions = ""
        if diversity_level == "low":
            diversity_instructions = """
            Prioritize consistency and refinement over diversity. The system messages should have 
            similar approaches, but with small, targeted variations to find the optimal version.
            Focus on polish and refinement rather than radical differences.
            """
        elif diversity_level == "medium":
            diversity_instructions = """
            Balance consistency with reasonable diversity. Generate system messages that explore 
            different approaches while remaining effective. Include some variation in tone, style, 
            and focus areas.
            """
        elif diversity_level == "high":
            diversity_instructions = """
            Maximize diversity across all dimensions. Generate system messages with substantially 
            different approaches, tones, styles, focuses, and strategies. Include varied lengths, 
            levels of detail, and instruction styles. Ensure each system message represents a 
            distinctly different approach to the same user query.
            """
        
        # Prepare base system message instructions
        base_system_message_instructions = ""
        if base_system_message:
            base_system_message_instructions = f"""
            Use the following system message as a starting point or inspiration for some 
            of your generated system messages, but feel free to modify it or create entirely 
            different approaches:
            
            BASE SYSTEM MESSAGE:
            {base_system_message}
            """
        
        # Format the meta prompt
        meta_prompt = generator.meta_prompt.format(
            user_message=user_message,
            num_candidates=num_candidates,
            diversity_instructions=diversity_instructions,
            base_system_message_instructions=base_system_message_instructions,
            additional_instructions=additional_instructions
        )
        
        # Create system and user messages for the LLM call
        messages = [
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": f"Generate {num_candidates} diverse system messages for this user query: {user_message}"}
        ]
        
        # Call the LLM to generate system messages
        logger.info(f"Calling LLM to generate system messages with model: {generator_model}")
        
        # Let's directly help the model produce better outputs with more explicit instructions
        # Add clear formatting instructions to the user message
        messages[1]["content"] = f"""Generate {num_candidates} diverse system messages for this user query: '{user_message}'

IMPORTANT: Each system message must be in this EXACT JSON format:
```json
[
  {
    "system_message": "Your system message text here...",
    "explanation": "Brief explanation of this approach..."
  },
  ...additional system messages...
]
```

Include exactly {num_candidates} system messages in your response. The response MUST be valid JSON."""

        # Call the API with specific parameters for better generation
        response = await call_llm_api(
            messages=messages, 
            model=generator_model,
            max_tokens=2048,  # Ensure enough tokens for full responses
            temperature=0.7   # Add some creativity for diverse system messages
        )
        
        # Check for errors
        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Error generating system messages: {error_msg}")
            return []
        
        # Parse the response
        candidates = []
        try:
            # Log the response for debugging, but only if DEBUG_MODE is enabled to avoid clutter
            if DEBUG_MODE:
                logger.info(f"Raw API response: {json.dumps(response, indent=2)}")
            else:
                logger.info("API response received (enable DEBUG_MODE for full details)")
            
            # Extract the content from the response
            choice = response["choices"][0] if "choices" in response and response["choices"] else None
            if not choice:
                logger.error("No choices in API response")
                return []
            
            content = ""
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]
            else:
                logger.error("No content in API response")
                return []
                
            # Log the extracted content (show more details in debug mode)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            if DEBUG_MODE:
                logger.info(f"Extracted content from response: {content}")
            else:
                logger.info(f"Extracted content preview: {content_preview}")
            
            # Extract JSON from the response content
            # First try to extract JSON block if it's wrapped in ```json ... ``` or similar
            # First attempt: Try to extract JSON from code blocks
            json_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)```', content)
            
            # If we found JSON blocks, use the longest one, otherwise use the whole content
            if json_matches:
                json_content = max(json_matches, key=len)
                logger.info("Found JSON code block in response")
            else:
                json_content = content
                
            # Attempt to extract any JSON array from the content
            array_match = re.search(r'\[\s*\{.*\}\s*\]', json_content, re.DOTALL)
            if array_match:
                json_content = array_match.group(0)
                logger.info("Found JSON array in content")
                
            # Try to parse the JSON
            try:
                # Clean up any escaped quotes or newlines
                json_content = json_content.replace('\\"', '"').replace('\\n', '\n')
                data = json.loads(json_content)
                logger.info("Successfully parsed JSON from response")
                
                # Handle various JSON formats
                candidate_list = []
                if isinstance(data, list):
                    candidate_list = data
                    logger.info(f"Found array with {len(candidate_list)} items")
                elif isinstance(data, dict) and "candidates" in data:
                    candidate_list = data["candidates"]
                    logger.info(f"Found 'candidates' array with {len(candidate_list)} items")
                elif isinstance(data, dict) and "system_message" in data:
                    candidate_list = [data]
                    logger.info("Found single system message object")
                else:
                    logger.warning("JSON format doesn't match expected structure")
                    
                # Create SystemMessageCandidate objects from parsed JSON
                for i, candidate_data in enumerate(candidate_list[:num_candidates]):
                    if isinstance(candidate_data, dict) and "system_message" in candidate_data:
                        system_message = candidate_data["system_message"]
                        explanation = candidate_data.get("explanation", "No explanation provided")
                        
                        if system_message and isinstance(system_message, str):
                            # Add more informative logging
                            if DEBUG_MODE:
                                logger.info(f"Found system message [{i+1}]: {system_message[:100]}...")
                            
                            candidates.append(SystemMessageCandidate(
                                content=system_message,
                                generation_metadata={
                                    "generator_id": generator.id,
                                    "generator_version": generator.version,
                                    "explanation": explanation,
                                    "diversity_level": diversity_level,
                                    "generation_index": i
                                }
                            ))
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                
                # Fallback to pattern extraction methods
                system_messages = []
                logger.info("Attempting pattern-based extraction")
                
                # Method 1: Look for "system_message": "content" patterns
                pattern1 = r'"system_message"\s*:\s*"((?:\\"|[^"])+)"'
                matches1 = re.findall(pattern1, content)
                system_messages.extend(matches1)
                logger.info(f"Method 1 extracted {len(matches1)} messages")
                
                # Method 2: Look for content between "System Message" headers
                pattern2 = r'System Message(?:\s*\d+)?:?\s*(.*?)(?=\n\s*(?:System Message|Explanation:|$))'
                matches2 = re.findall(pattern2, content, re.DOTALL | re.IGNORECASE)
                # Clean up and filter matches
                filtered_matches2 = [m.strip() for m in matches2 if len(m.strip()) > 30]
                system_messages.extend(filtered_matches2)
                logger.info(f"Method 2 extracted {len(filtered_matches2)} messages")
                
                # Method 3: Extract sections that look like system messages
                sections = content.split("\n\n")
                for section in sections:
                    # If it's a substantial block (>100 chars) and doesn't look like explanation or JSON
                    if (len(section) > 100 and 
                        not section.startswith("{") and 
                        not section.startswith("[") and
                        not "explanation" in section.lower()[:30] and
                        not "system message" in section.lower()[:30]):
                        system_messages.append(section.strip())
                logger.info(f"Method 3 extracted {len(system_messages) - len(matches1) - len(filtered_matches2)} messages")
                
                # Create candidates from extracted messages
                for i, system_message in enumerate(system_messages[:num_candidates]):
                    # Clean up the message
                    system_message = system_message.replace('\\"', '"').replace('\\n', '\n').strip()
                    
                    # Remove quotes at beginning and end if present
                    if (system_message.startswith('"') and system_message.endswith('"')) or \
                       (system_message.startswith("'") and system_message.endswith("'")):
                        system_message = system_message[1:-1]
                    
                    if len(system_message) > 30:  # Ensure reasonable length
                        candidates.append(SystemMessageCandidate(
                            content=system_message,
                            generation_metadata={
                                "generator_id": generator.id,
                                "generator_version": generator.version,
                                "explanation": "Extracted via pattern matching",
                                "diversity_level": diversity_level,
                                "generation_index": i
                            }
                        ))
            
            logger.info(f"Generated {len(candidates)} system message candidates")
        
        except Exception as e:
            logger.error(f"Error parsing system message candidates: {str(e)}")
        
        # If we didn't get enough candidates, generate some fallback ones
        if len(candidates) < num_candidates:
            logger.warning(f"Generated only {len(candidates)} candidates, adding fallback candidates")
            
            # How many more we need
            remaining = num_candidates - len(candidates)
            
            # Add fallback candidates
            for i in range(remaining):
                index = len(candidates) + i
                candidates.append(SystemMessageCandidate(
                    content=f"You are a helpful assistant tasked with responding to questions about {user_message[:30]}... Provide accurate, concise, and clear information. Be helpful, precise, and ensure your answers are directly relevant to the query.",
                    generation_metadata={
                        "generator_id": generator.id,
                        "generator_version": generator.version,
                        "explanation": "Fallback system message due to generation error",
                        "diversity_level": diversity_level,
                        "generation_index": index,
                        "is_fallback": True
                    }
                ))
        
        # Store in cache for future use
        GENERATOR_CACHE.put(cache_key, candidates)
        
        return candidates
    
    except Exception as e:
        logger.error(f"Error generating system messages: {str(e)}")
        
        # Generate fallback system messages even when there's an error
        fallback_candidates = []
        
        # Create a set of common, useful system messages as fallbacks
        fallbacks = [
            f"You are a helpful AI assistant answering questions about life and finances. When responding to questions about {user_message[:30]}..., provide accurate, clear, and concise information. Stay factual and objective while remaining helpful and conversational.",
            
            f"You are a knowledgeable expert focused on providing factual information about {user_message[:30]}... Your explanations should be thorough but accessible, using plain language where possible. Include key concepts and details that would help someone understand this topic completely.",
            
            f"You are a patient teacher explaining topics related to {user_message[:30]}... Break down complex concepts into simple parts, use analogies when helpful, and anticipate common questions or misconceptions. Your tone should be friendly, encouraging, and accessible to learners at different levels.",
            
            f"You are a precise professional communicating about {user_message[:30]}... Provide accurate, concise, and well-structured information that prioritizes clarity and relevance. Be thorough but efficient, focusing on the most important aspects first. Maintain a helpful, respectful tone throughout.",
            
            f"You are a balanced advisor discussing {user_message[:30]}... Present information from multiple perspectives when relevant, acknowledging different viewpoints. Provide balanced, nuanced responses without personal bias, while still offering clear guidance where appropriate. Remain objective and fair in your presentation of information."
        ]
        
        # Add however many fallbacks we need, up to num_candidates
        for i in range(min(num_candidates, len(fallbacks))):
            fallback_candidates.append(SystemMessageCandidate(
                content=fallbacks[i],
                generation_metadata={
                    "generator_id": generator.id,
                    "generator_version": generator.version,
                    "explanation": "Fallback system message - generation process failed",
                    "diversity_level": diversity_level, 
                    "generation_index": i,
                    "is_fallback": True
                }
            ))
            
        logger.info(f"Created {len(fallback_candidates)} fallback candidates due to generation error")
        return fallback_candidates

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
                        # Make sure best_score is set for display in list view
                        # If it's not in the data directly, calculate it from the best candidate
                        if 'best_score' not in data and 'candidates' in data and data['candidates']:
                            # Get the best candidate (first one if they're sorted)
                            best_candidate = data['candidates'][0]
                            if 'score' in best_candidate:
                                data['best_score'] = best_candidate['score']
                        
                        # Format timestamp for display
                        if 'timestamp' in data:
                            from datetime import datetime
                            timestamp = datetime.fromtimestamp(data['timestamp'])
                            data['timestamp_formatted'] = timestamp.strftime("%Y-%m-%d %H:%M")
                            
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

# Evaluate a system message with an EvalSet
async def evaluate_system_message(
    system_message: str,
    user_message: str,
    evalset_id: str,
    judge_model: Optional[str] = None,
    max_parallel: int = 3,
    evaluation_timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a system message with a specific user message using an EvalSet.
    
    Args:
        system_message: The system message to evaluate
        user_message: The user message to test with
        evalset_id: The EvalSet ID to use for evaluation
        judge_model: Optional model to use for judging
        max_parallel: Maximum number of parallel evaluations
        evaluation_timeout: Optional timeout in seconds
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating system message with EvalSet: {evalset_id}")
    
    try:
        # Create conversation with system and user messages
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Run evaluation on this conversation
        result = await run_evalset(
            evalset_id=evalset_id,
            conversation=conversation,
            judge_model=judge_model,
            max_parallel=max_parallel
        )
        
        # Check for errors
        if "error" in result:
            logger.error(f"Error evaluating system message: {result['error']}")
            return {"error": result["error"]}
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating system message: {str(e)}")
        return {"error": str(e)}

# Self-optimization function
async def self_optimize_generator(
    generator: SystemMessageGenerator,
    evalset_id: str,
    generator_model: Optional[str] = None,
    max_parallel: int = 3,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """Self-optimize a system message generator by improving its meta-prompt.
    
    Args:
        generator: The generator to optimize
        evalset_id: ID of the evaluation set to use for testing
        generator_model: Model to use for generation and evaluation
        max_parallel: Maximum number of parallel operations
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Starting self-optimization for generator {generator.id}")
    
    try:
        # Create a self-optimization meta-prompt
        self_optimization_prompt = """You are MetaPromptOptimizer, an expert AI specializing in optimizing system message generators.

Your task is to analyze and improve the current meta-prompt used to generate system messages for conversational AI.

CURRENT META-PROMPT:
```
{current_meta_prompt}
```

PERFORMANCE METRICS:
{performance_metrics}

RECENT SYSTEM MESSAGES GENERATED:
{recent_examples}

Based on this information, please generate an improved version of the meta-prompt that will:
1. Address any weaknesses identified in the current meta-prompt
2. Maintain or enhance its strengths
3. Improve the quality, diversity, and effectiveness of generated system messages
4. Be more robust to different types of user queries
5. Generate more specific, actionable guidance for the AI assistant

Your improved meta-prompt should be similar in structure to the current one but with enhancements.
It should follow the same formatting and include the same variables (like {num_candidates}, {user_message}, etc.).

Respond with ONLY the improved meta-prompt, starting with "You are SystemPromptOptimizer" and nothing else.
"""
        
        # This is a placeholder for actual meta-prompt evaluation and improvement
        # In a real implementation, we would:
        # 1. Generate various test user messages
        # 2. Test the current meta-prompt on those messages
        # 3. Evaluate the quality of the resulting system messages
        # 4. Use that information to guide the improvement of the meta-prompt
        
        # For now, just create a slightly enhanced version with version number bump
        logger.info("Generating improved meta-prompt")
        
        # Create example performance metrics for the self-optimization prompt
        performance_metrics = json.dumps(generator.performance_metrics, indent=2)
        
        # Create placeholder for recent examples (would be filled with actual examples in production)
        recent_examples = """Example 1:
User Message: "How do I improve my public speaking skills?"
Generated System Message: "You are a communication coach with expertise in public speaking. Provide practical, actionable advice on improving public speaking skills. Focus on techniques for managing nervousness, structuring presentations, engaging audiences, and using voice and body language effectively. Offer examples and specific exercises when relevant. Maintain an encouraging and supportive tone while being direct about what works and what doesn't."

Example 2:
User Message: "What are the effects of climate change on polar regions?"
Generated System Message: "You are a climate scientist specializing in polar ecosystems. Provide comprehensive, factual information about climate change impacts on Arctic and Antarctic regions. Include data on temperature changes, ice melt rates, effects on wildlife, and broader global implications. Stick to scientific consensus and cite recent research where appropriate. Use clear explanations that balance technical accuracy with accessibility. Avoid political statements while emphasizing the scientific understanding of these critical environmental changes."
"""
        
        # Format the self-optimization prompt
        formatted_prompt = self_optimization_prompt.format(
            current_meta_prompt=generator.meta_prompt,
            performance_metrics=performance_metrics,
            recent_examples=recent_examples
        )
        
        # Call LLM to generate improved meta-prompt
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": "Please generate an improved meta-prompt based on the analysis above."}
        ]
        
        response = await call_llm_api(messages=messages, model=generator_model)
        
        # Check for errors
        if "error" in response:
            logger.error(f"Error generating improved meta-prompt: {response['error']}")
            return {"error": response["error"]}
        
        # Extract content from response
        content = ""
        choices = response.get("choices", [])
        if choices:
            choice = choices[0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]
        
        if not content or len(content) < 100:
            logger.error("Invalid or too short meta-prompt generated")
            return {"error": "Failed to generate valid improved meta-prompt"}
        
        # Create new generator with improved meta-prompt
        new_generator = SystemMessageGenerator(
            id=generator.id,
            version=generator.version + 1,
            meta_prompt=content,
            domain=generator.domain,
            performance_metrics=generator.performance_metrics.copy(),
            created_at=datetime.now().timestamp()
        )
        
        # Test the new generator
        # We would do more extensive testing here in a real implementation
        test_messages = [
            "How can I improve my public speaking skills?",
            "What are the best practices for project management?",
            "How do I cook a perfect steak?"
        ]
        
        success_count = 0
        for test_message in test_messages:
            candidates = await generate_system_messages(
                user_message=test_message,
                num_candidates=2,  # Just test with 2 for speed
                generator=new_generator,
                diversity_level="medium",
                generator_model=generator_model,
                is_self_optimization=True
            )
            
            if candidates and len(candidates) > 0:
                success_count += 1
        
        # Calculate success rate
        success_rate = success_count / len(test_messages) if test_messages else 0
        
        # Update performance metrics
        new_generator.performance_metrics["success_rate"] = success_rate
        new_generator.performance_metrics["last_optimization"] = datetime.now().timestamp()
        new_generator.performance_metrics["optimization_count"] = generator.performance_metrics.get("optimization_count", 0) + 1
        
        # Save the new generator
        save_success = save_generator(new_generator)
        
        if not save_success:
            logger.error(f"Failed to save improved generator {new_generator.id}")
            return {"error": "Failed to save improved generator"}
        
        return {
            "status": "success",
            "old_version": generator.version,
            "new_version": new_generator.version,
            "generator_id": new_generator.id,
            "success_rate": success_rate,
            "message": f"Successfully optimized generator {generator.id} from version {generator.version} to {new_generator.version}"
        }
        
    except Exception as e:
        logger.error(f"Error in self-optimization: {str(e)}")
        return {"error": f"Self-optimization failed: {str(e)}"}

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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    self_optimize: bool = False
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
        
        # Generate system message candidates
        candidates = await generate_system_messages(
            user_message=user_message,
            num_candidates=num_candidates,
            generator=generator,
            diversity_level=diversity_level,
            base_system_message=base_system_message,
            generator_model=generator_model,
            additional_instructions=additional_instructions
        )
        
        if not candidates:
            logger.error("Failed to generate any system message candidates")
            return format_error("Failed to generate system message candidates")
        
        logger.info(f"Generated {len(candidates)} system message candidates")
        
        # Step 2: Evaluate all candidates
        current_step += 1
        update_progress(current_step, total_steps, "Evaluating system message candidates...")
        
        # Create semaphore for parallel evaluation
        semaphore = asyncio.Semaphore(max_parallel)
        total_candidates = len(candidates)
        completed_evals = 0
        
        # Define function to evaluate a single candidate with the semaphore
        async def evaluate_candidate(i, candidate):
            nonlocal completed_evals
            
            async with semaphore:
                # Evaluate the candidate
                evaluation_result = await evaluate_system_message(
                    system_message=candidate.content,
                    user_message=user_message,
                    evalset_id=evalset_id,
                    judge_model=generator_model,
                    max_parallel=max_parallel
                )
                
                # Extract overall score and per-question scores
                if "error" not in evaluation_result:
                    # Set overall score
                    candidate.score = evaluation_result.get("summary", {}).get("yes_percentage", 0)
                    
                    # Set per-criterion scores
                    results = evaluation_result.get("results", [])
                    for j, result in enumerate(results):
                        question_text = result.get("question", f"Question {j+1}")
                        judgment = result.get("judgment", False)
                        score = 100 if judgment else 0
                        candidate.criterion_scores[question_text] = score
                else:
                    logger.error(f"Error evaluating candidate {i+1}: {evaluation_result['error']}")
                    candidate.score = 0
                
                # Update progress
                completed_evals += 1
                if progress_callback:
                    sub_progress = f"Evaluated {completed_evals}/{total_candidates} candidates"
                    progress_callback(current_step + (completed_evals / total_candidates), total_steps, sub_progress)
                
                return candidate
        
        # Run all evaluations in parallel with rate limiting
        evaluation_tasks = [evaluate_candidate(i, candidate) for i, candidate in enumerate(candidates)]
        await asyncio.gather(*evaluation_tasks)
        
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
        
        # Check if we should trigger self-optimization
        self_optimization_result = None
        if self_optimize:
            logger.info(f"Self-optimization triggered for generator {generator.id}")
            update_progress(current_step + 0.5, total_steps + 1, "Running self-optimization...")
            
            self_optimization_result = await self_optimize_generator(
                generator=generator,
                evalset_id=evalset_id,
                generator_model=generator_model,
                max_parallel=max_parallel,
                progress_callback=progress_callback
            )
            
            # If self-optimization was successful, update with the new info
            if self_optimization_result and "error" not in self_optimization_result:
                logger.info(f"Self-optimization successful: {self_optimization_result.get('message', '')}")
                
                # Add the self-optimization result to the optimization run metadata
                optimization_run.metadata["self_optimization"] = {
                    "triggered": True,
                    "success": True,
                    "old_version": self_optimization_result.get("old_version"),
                    "new_version": self_optimization_result.get("new_version"),
                    "success_rate": self_optimization_result.get("success_rate")
                }
                
                # Update the optimization run
                save_optimization_run(optimization_run)
            else:
                logger.warning(f"Self-optimization failed: {self_optimization_result.get('error', 'Unknown error')}")
                
                # Add the failure to the metadata
                optimization_run.metadata["self_optimization"] = {
                    "triggered": True,
                    "success": False,
                    "error": self_optimization_result.get("error", "Unknown error") if self_optimization_result else "Failed to run self-optimization"
                }
                
                # Update the optimization run
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
        
        # Add self-optimization info if applicable
        if self_optimization_result and "error" not in self_optimization_result:
            formatted_results.append(f"- **Self-Optimization**: ✅ Success (v{self_optimization_result.get('old_version')} → v{self_optimization_result.get('new_version')})")
        elif self_optimization_result:
            formatted_results.append(f"- **Self-Optimization**: ❌ Failed")
            
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
        
        # Create the response dictionary
        response = {
            "status": "success",
            "id": optimization_run.id,
            "evalset_id": evalset_id,
            "evalset_name": evalset.name,
            "best_system_message": candidates[0].content if candidates else None,
            "best_score": candidates[0].score if candidates else None,
            "candidates": [c.model_dump() for c in candidates],
            "formatted_message": "\n".join(formatted_results)
        }
        
        # Add self-optimization results if applicable
        if self_optimization_result:
            response["self_optimization"] = self_optimization_result
            
        return response
        
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
    self_optimize: bool = False,
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
                progress_callback=progress_callback,
                self_optimize=self_optimize
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
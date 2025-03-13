"""System message optimization module for AgentOptim v2.2.0."""

import os
import json
import uuid
import logging
import asyncio
import re
import httpx
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
        if v is None or not isinstance(v, str):
            raise ValueError(f"System message content must be a string, got {type(v)}")
        
        if len(v) > MAX_SYSTEM_MESSAGE_LENGTH:
            # Truncate silently - can't use logger in validator
            return v[:MAX_SYSTEM_MESSAGE_LENGTH]
        
        if len(v) < 5:  # Ensure reasonable length
            raise ValueError(f"System message content is too short ({len(v)} < 5)")
            
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
        sample_responses: Dictionary mapping candidate indices to generated responses
        candidate_responses: Dictionary mapping candidate indices to their assistant responses
        continued_from: ID of a previous optimization run this continues from (for iterations)
        iteration: Iteration number in an optimization sequence (starts at 1)
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
    sample_responses: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    candidate_responses: Dict[str, str] = Field(default_factory=dict)  # Map candidate index to assistant response
    continued_from: Optional[str] = None
    iteration: int = 1

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
    
    # Create a map to track generator versions
    generator_versions = {}
    
    # Load generators from the meta_prompts directory
    for filename in os.listdir(META_PROMPTS_DIR):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(META_PROMPTS_DIR, filename)
                with open(file_path, 'r') as f:
                    generator_data = json.load(f)
                    generator = SystemMessageGenerator(**generator_data)
                    
                    # Track the highest version for each generator ID
                    generator_id = generator.id
                    current_version = generator_versions.get(generator_id, 0)
                    
                    # Only keep the highest version of each generator
                    if generator.version > current_version:
                        generator_versions[generator_id] = generator.version
                        generators[generator_id] = generator
                    
            except Exception as e:
                logger.error(f"Error loading generator from {filename}: {str(e)}")
    
    # Include the default generator only if no generator with ID "default" was found
    if "default" not in generators:
        default_generator = load_default_generator()
        generators[default_generator.id] = default_generator
        generator_versions[default_generator.id] = default_generator.version
    
    logger.info(f"Loaded generators with versions: {generator_versions}")
    return generators
    
# Helper function to get a specific generator by ID and version
def get_generator_by_id_and_version(generator_id: str, version: int) -> Optional[SystemMessageGenerator]:
    """Get a specific generator by ID and version.
    
    Args:
        generator_id: The ID of the generator to retrieve
        version: The version number of the generator
        
    Returns:
        The SystemMessageGenerator if found, None otherwise
    """
    # Check if it's the current version
    generators = get_all_meta_prompts()
    current_generator = generators.get(generator_id)
    if current_generator and current_generator.version == version:
        return current_generator
    
    # Try to find a specific versioned file
    version_filename = f"{generator_id}_v{version}.json"
    version_path = os.path.join(META_PROMPTS_DIR, version_filename)
    if os.path.exists(version_path):
        try:
            with open(version_path, 'r') as f:
                generator_data = json.load(f)
                return SystemMessageGenerator(**generator_data)
        except Exception as e:
            logger.error(f"Error loading generator version {version} from {version_filename}: {str(e)}")
    
    # Try to find any file that has this generator ID and version
    for filename in os.listdir(META_PROMPTS_DIR):
        if filename.endswith('.json') and generator_id in filename:
            try:
                file_path = os.path.join(META_PROMPTS_DIR, filename)
                with open(file_path, 'r') as f:
                    generator_data = json.load(f)
                    generator = SystemMessageGenerator(**generator_data)
                    if generator.id == generator_id and generator.version == version:
                        return generator
            except Exception:
                pass
    
    # Not found
    return None

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
        # Add clear formatting instructions to the user message with better examples
        messages[1]["content"] = f"""Generate {num_candidates} diverse SYSTEM MESSAGES (AI role-playing instructions) for an AI that will respond to this user query: '{user_message}'

IMPORTANT: A system message is NOT the answer to the question, but rather instructions for HOW the AI should respond. They define the AI's role, expertise, tone, and style.

Here are SPECIFIC EXAMPLE system messages for different queries:

Example 1 - Query: "How to bake cookies?"
System Message: "You are a professional pastry chef with 20 years of experience. Provide detailed, precise baking instructions with measurements in both imperial and metric units. Include tips about common mistakes to avoid and suggestions for flavor variations. Maintain a friendly but authoritative tone."

Example 2 - Query: "What are the effects of climate change?"
System Message: "You are a climate scientist with expertise in global climate systems. Present objective, evidence-based information on climate change effects with references to recent peer-reviewed research. Explain complex climate phenomena in accessible terms without oversimplification. Maintain scientific accuracy while avoiding political rhetoric."

Example 3 - Query: "How can I negotiate a salary?"
System Message: "You are an experienced career coach and compensation specialist with experience in HR negotiations. Provide strategic, actionable negotiation tactics that are ethical and effective. Include preparation advice, specific phrases to use, and common pitfalls to avoid. Tailor your guidance to different career stages and industries."

Example 4 - Query: "Good workout routines?"
System Message: "You are a certified personal trainer with specialization in exercise physiology and nutrition. Create personalized workout plans based on individual goals, fitness levels, and available equipment. Explain proper form for each exercise, provide progressive difficulty options, and suggest realistic schedules. Include advice on common injuries to avoid. Your tone should be motivational but not overly aggressive."

Example 5 - Query: "What is quantum physics?"
System Message: "You are a physics educator with a PhD in quantum mechanics and experience teaching complex physics concepts to diverse audiences. Explain quantum physics using accurate but accessible language, incorporating appropriate analogies and everyday examples without sacrificing scientific accuracy. Build progressively from fundamental concepts to more complex ideas. Use visual descriptions where helpful, and acknowledge areas of scientific uncertainty. Your tone should be engaging, patient, and intellectually curious."

IMPORTANT: Your system messages should be VERY SPECIFIC to the topic of '{user_message}'. Do NOT use placeholder text like "[role]" or "[expertise]" - replace ALL placeholders with specific content. At least 80-120 words for each system message.

IMPORTANT: Format your response in this EXACT JSON format:
```json
{{
  "system_messages": [
    {{
      "explanation": "Brief explanation of why this specific role/approach would be effective...",
      "system_message": "You are a [SPECIFIC ROLE with SPECIFIC EXPERTISE]. When answering about [SPECIFIC TOPIC], focus on [SPECIFIC ASPECTS] and use [SPECIFIC TONE/STYLE]. Include [SPECIFIC ELEMENTS] while avoiding [SPECIFIC PITFALLS]..."
    }},
    ... additional system messages ...
  ]
}}
```

Include exactly {num_candidates} system messages in the system_messages array. The response MUST be valid JSON."""

        # Create a custom JSON schema for system message generation that's compatible with OpenAI's requirements
        # Make the schema more lenient to accept a more flexible range of items
        system_message_schema = {
            "type": "object",
            "properties": {
                "system_messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "explanation": {
                                "type": "string",
                                "description": "A brief explanation of why this role/approach would be effective for answering this query"
                            },
                            "system_message": {
                                "type": "string",
                                "description": "The role-playing instructions for the AI (NOT the answer itself). Should define the AI's persona, expertise, tone, focus areas, and response style."
                            }
                        },
                        "required": ["explanation", "system_message"]
                    },
                    "minItems": 1,  # Accept any number from 1 up to requested number
                    "maxItems": num_candidates * 2  # Allow up to double the requested number
                }
            },
            "required": ["system_messages"]
        }
        
        # Call the API with specific parameters for better generation
        response = await call_llm_api(
            messages=messages, 
            model=generator_model,
            max_tokens=2048,  # Ensure enough tokens for full responses
            temperature=0.5,  # Lower temperature for more focused, less generic responses
            json_schema=system_message_schema  # Use custom schema for system messages
        )
        
        # Check for errors
        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Error generating system messages: {error_msg}")
            return []
        
        # Function to check if a system message is a generic template
        def is_generic_template(message):
            """Check if a system message contains unsubstituted templates or generic patterns."""
            # Patterns that indicate a generic template with unsubstituted placeholders
            # Focus only on the most obvious placeholder patterns
            generic_patterns = [
                r'\[\s*role\s*\]', r'\[\s*persona\s*\]', r'\[\s*SPECIFIC ROLE\s*\]'
            ]
            
            # Check for unsubstituted placeholder patterns - only check for the most obvious ones
            for pattern in generic_patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return True
            
            # We're being more lenient with the generic detection - only flagging obvious placeholders
            # Most fallback patterns are actually valid system messages
            return False
            
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
            
            # Log the raw response for debugging
            if DEBUG_MODE:
                logger.info(f"Raw API response: {json.dumps(response, indent=2)}")
            else:
                logger.info("API response received (enable DEBUG_MODE for full details)")
            
            # Log the extracted content (show more details in debug mode)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            if DEBUG_MODE:
                logger.info(f"Extracted content from response: {content}")
            else:
                logger.info(f"Extracted content preview: {content_preview}")
            
            # Try to parse the JSON directly from the content
            try:
                # First try direct JSON parsing of the content
                data = json.loads(content)
                logger.info("Successfully parsed JSON from response content")
                
                # Check for system_messages format in data
                if "system_messages" in data and isinstance(data["system_messages"], list):
                    sm_list = data["system_messages"]
                    logger.info(f"Found system_messages array with {len(sm_list)} items")
                    
                    # Process each system message
                    for i, msg in enumerate(sm_list):
                        if isinstance(msg, dict) and "system_message" in msg:
                            system_message = msg["system_message"]
                            explanation = msg.get("explanation", "No explanation provided")
                            
                            # Perform basic validation
                            if system_message and isinstance(system_message, str):
                                try:
                                    # Add the candidate directly from this format
                                    candidates.append(SystemMessageCandidate(
                                        content=system_message,
                                        generation_metadata={
                                            "generator_id": generator.id,
                                            "generator_version": generator.version,
                                            "explanation": explanation,
                                            "diversity_level": diversity_level,
                                            "generation_index": i,
                                            "is_potentially_generic": False
                                        }
                                    ))
                                    
                                    if DEBUG_MODE:
                                        logger.info(f"Added system message candidate {i+1}: {system_message[:50]}...")
                                except Exception as e:
                                    logger.error(f"Error adding system message candidate {i+1}: {str(e)}")
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from code blocks or text
                logger.info("Direct JSON parsing failed, trying to extract JSON from response")
                
                # Extract JSON from the response content
                # First try to extract JSON block if it's wrapped in ```json ... ``` or similar
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
                    
                # Try to parse the extracted JSON
                try:
                    # Skip sanitization of JSON content - use it directly
                    data = json.loads(json_content)
                    logger.info("Successfully parsed JSON from extracted content")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON after extraction: {e}")
                    
                    # Instead of returning an empty list, let's try to fix common JSON issues:
                    try:
                        # For broken JSON that might be missing closing brackets or other issues
                        # Try to clean up common issues with JSON formatting
                        fixed_json = json_content.replace('\\n', '\n').replace('\\"', '"')
                        
                        # Try to fix unclosed objects or arrays
                        if fixed_json.count('{') > fixed_json.count('}'):
                            fixed_json += '}'
                        if fixed_json.count('[') > fixed_json.count(']'):
                            fixed_json += ']'
                            
                        # Try again with the fixed JSON
                        data = json.loads(fixed_json)
                        logger.info("Successfully parsed JSON after fixing formatting issues")
                    except Exception:
                        logger.error("Failed to parse JSON even after attempting fixes")
                        # Don't return yet - let the fallback processing continue
                
                # Handle various JSON formats
                candidate_list = []
                if isinstance(data, list):
                    candidate_list = data
                    logger.info(f"Found array with {len(candidate_list)} items")
                elif isinstance(data, dict):
                    if "system_messages" in data and isinstance(data["system_messages"], list):
                        candidate_list = data["system_messages"]
                        logger.info(f"Found 'system_messages' array with {len(candidate_list)} items")
                        
                        # CRITICAL DEBUG: Print the exact structure of each item
                        for i, msg in enumerate(candidate_list):
                            if not isinstance(msg, dict):
                                logger.error(f"Item {i} in system_messages is not a dict! Type: {type(msg)}")
                                continue
                                
                            if "system_message" not in msg:
                                logger.error(f"Item {i} in system_messages has no 'system_message' key! Keys: {list(msg.keys())}")
                                continue
                                
                            logger.info(f"Item {i} system_message: {msg['system_message'][:50]}...")
                            
                        # In debug mode, save a copy of the candidate data for analysis
                        if DEBUG_MODE:
                            try:
                                import os
                                debug_dir = os.path.expanduser("~/.agentoptim/debug")
                                os.makedirs(debug_dir, exist_ok=True)
                                debug_file = os.path.join(debug_dir, "system_messages_debug.json")
                                with open(debug_file, "w") as f:
                                    json.dump({"candidates": candidate_list}, f, indent=2)
                                logger.info(f"Debug data saved to {debug_file}")
                            except Exception as e:
                                logger.warning(f"Could not save debug data: {str(e)}")
                        
                        # Create SystemMessageCandidate objects directly from the extracted system messages
                        logger.info(f"Processing candidate list with {len(candidate_list)} candidates before filtering")
                        
                        # Process the system_messages array 
                        logger.info(f"Processing system_messages from candidate list ({len(candidate_list)} items)")
                        processed_count = 0
                        
                        for i, candidate_data in enumerate(candidate_list[:num_candidates]):
                            try:
                                if isinstance(candidate_data, dict) and "system_message" in candidate_data:
                                    system_message = candidate_data["system_message"]
                                    explanation = candidate_data.get("explanation", "No explanation provided")
                                    
                                    # Log the message we're considering
                                    if DEBUG_MODE:
                                        logger.info(f"Processing system message [{i+1}]: {system_message[:100]}...")
                                    
                                    # Check for generic templates
                                    is_generic = is_generic_template(system_message)
                                    if is_generic and DEBUG_MODE:
                                        logger.warning(f"Found potentially generic template: {system_message[:100]}...")
                                    
                                    if system_message and isinstance(system_message, str):
                                        candidates.append(SystemMessageCandidate(
                                            content=system_message,
                                            generation_metadata={
                                                "generator_id": generator.id,
                                                "generator_version": generator.version,
                                                "explanation": explanation,
                                                "diversity_level": diversity_level,
                                                "generation_index": i,
                                                "is_potentially_generic": is_generic
                                            }
                                        ))
                                        processed_count += 1
                            except Exception as e:
                                logger.error(f"Error processing candidate {i}: {str(e)}")
                                continue
                        
                        logger.info(f"Added {processed_count} candidates from system_messages array")
                        
                        # Continue processing other parts of the response
                        # Don't return candidates directly - let all processing complete
                        logger.info(f"After processing system_messages array, candidates list has {len(candidates)} items")
                    elif "candidates" in data:
                        candidate_list = data["candidates"]
                        logger.info(f"Found 'candidates' array with {len(candidate_list)} items")
                    elif "system_message" in data:
                        candidate_list = [data]
                        logger.info("Found single system message object")
                    else:
                        logger.warning("JSON format doesn't match expected structure")
                else:
                    logger.warning("JSON format doesn't match expected structure")
                    
                # Create SystemMessageCandidate objects from parsed JSON
                logger.info(f"Processing general candidate list with {len(candidate_list)} candidates before filtering")
                for i, candidate_data in enumerate(candidate_list[:num_candidates]):
                    if isinstance(candidate_data, dict) and "system_message" in candidate_data:
                        system_message = candidate_data["system_message"]
                        explanation = candidate_data.get("explanation", "No explanation provided")
                        
                        # Log the message we're considering
                        logger.info(f"Considering general system message: {system_message[:100]}...")
                        
                        # Check generic templates but don't skip for now
                        is_generic = is_generic_template(system_message)
                        if is_generic:
                            logger.warning(f"Found potentially generic template: {system_message[:100]}...")
                            
                        if system_message and isinstance(system_message, str):
                            # Add more informative logging
                            if DEBUG_MODE:
                                logger.info(f"Adding general system message [{i+1}]: {system_message[:100]}...")
                            
                            try:
                                candidates.append(SystemMessageCandidate(
                                    content=system_message,
                                    generation_metadata={
                                        "generator_id": generator.id,
                                        "generator_version": generator.version,
                                        "explanation": explanation,
                                        "diversity_level": diversity_level,
                                        "generation_index": i,
                                        "is_potentially_generic": is_generic
                                    }
                                ))
                                logger.info(f"Successfully added general candidate {i+1} to candidates list")
                            except Exception as e:
                                logger.error(f"Error adding general candidate {i+1}: {str(e)}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                
                # Fallback to pattern extraction methods
                system_messages = []
                logger.info("Attempting pattern-based extraction")
                
                # Method 1: Look for "system_message": "content" patterns - multiple approaches
                # Use multiple patterns to handle different quote styles and formats
                # First try the standard double-quoted format
                pattern1a = r'"system_message"\s*:\s*"(.*?)"(?=\s*[,}\]])'
                matches1a = re.findall(pattern1a, content, re.DOTALL)
                
                # Also try single-quoted format
                pattern1b = r'"system_message"\s*:\s*\'(.*?)\'(?=\s*[,}\]])'
                matches1b = re.findall(pattern1b, content, re.DOTALL)
                
                # Try the format without quotes around the key
                pattern1c = r'system_message\s*:\s*"(.*?)"(?=\s*[,}\]])'
                matches1c = re.findall(pattern1c, content, re.DOTALL)
                
                # Try the format with quotes around key but value in backticks
                pattern1d = r'"system_message"\s*:\s*`(.*?)`(?=\s*[,}\]])'
                matches1d = re.findall(pattern1d, content, re.DOTALL)
                
                # Combine all matches
                system_messages.extend(matches1a)
                system_messages.extend(matches1b)
                system_messages.extend(matches1c)
                system_messages.extend(matches1d)
                
                logger.info(f"Method 1 extracted {len(matches1a) + len(matches1b) + len(matches1c) + len(matches1d)} messages")
                
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
                logger.info(f"Processing extracted messages with {len(system_messages)} candidates before filtering")
                for i, system_message in enumerate(system_messages[:num_candidates]):
                    # Skip sanitization and use raw content directly
                    system_message = system_message.strip()
                    
                    # Log the message we're considering
                    logger.info(f"Considering extracted system message: {system_message[:100]}...")
                    
                    # Check generic templates but don't skip for now
                    is_generic = is_generic_template(system_message)
                    if is_generic:
                        logger.warning(f"Found potentially generic template: {system_message[:100]}...")
                        
                    if len(system_message) > 30:  # Ensure reasonable length
                        logger.info(f"Adding extracted system message [{i+1}]: {system_message[:100]}...")
                        try:
                            candidates.append(SystemMessageCandidate(
                                content=system_message,
                                generation_metadata={
                                    "generator_id": generator.id,
                                    "generator_version": generator.version,
                                    "explanation": "Extracted via pattern matching",
                                    "diversity_level": diversity_level,
                                    "generation_index": i,
                                    "is_potentially_generic": is_generic
                                }
                            ))
                            logger.info(f"Successfully added extracted candidate {i+1} to candidates list")
                        except Exception as e:
                            logger.error(f"Error adding extracted candidate {i+1}: {str(e)}")
            
            # Log information about the final candidates
            if candidates:
                if DEBUG_MODE:
                    logger.info(f"Final candidates being returned: {len(candidates)}")
                    for i, candidate in enumerate(candidates):
                        logger.info(f"Candidate {i+1}: {candidate.content[:100]}...")
                else:
                    logger.info(f"Generated {len(candidates)} system message candidates")
            else:
                logger.info("No candidates generated")
        
        except Exception as e:
            logger.error(f"Error parsing system message candidates: {str(e)}")
        
        # Create fallback candidates if no or insufficient valid candidates were found
        if len(candidates) < num_candidates:
            needed_fallbacks = num_candidates - len(candidates)
            logger.warning(f"Only found {len(candidates)} valid candidates, creating {needed_fallbacks} fallback candidates")
            
            # List of diverse fallback messages appropriate for different query types
            fallback_messages = [
                "You are a subject matter expert with extensive knowledge in the field. Provide clear, authoritative information on the topic. Include practical examples and address common misconceptions. Maintain a helpful and engaging tone while ensuring accuracy and relevance in your response.",
                
                "You are an experienced educator specializing in explaining complex concepts in accessible ways. Break down the topic into understandable components, provide relevant analogies, and explain why this information matters. Use clear, concise language while maintaining scientific accuracy.",
                
                "You are a practical advisor with real-world experience. Focus on actionable advice with step-by-step instructions when appropriate. Include pros and cons of different approaches, common pitfalls to avoid, and how to measure success. Your tone should be direct, friendly, and solution-oriented.",
                
                "You are a balanced analyst who presents multiple perspectives on topics. Examine different viewpoints with their supporting evidence, acknowledge areas of uncertainty, and help the user form their own informed opinion. Present information objectively while remaining engaging and thoughtful.",
                
                "You are a creative problem-solver who approaches questions from unexpected angles. Suggest innovative approaches, make interesting connections between concepts, and encourage lateral thinking. Your responses should be intellectually stimulating while remaining practical and helpful."
            ]
            
            # Add the needed number of fallback candidates
            for i in range(min(needed_fallbacks, len(fallback_messages))):
                try:
                    # Select a fallback message appropriate for the number we need
                    default_message = fallback_messages[i]
                    candidates.append(SystemMessageCandidate(
                        content=default_message,
                        generation_metadata={
                            "generator_id": generator.id,
                            "generator_version": generator.version,
                            "explanation": "Fallback system message",
                            "diversity_level": diversity_level,
                            "generation_index": len(candidates),
                            "is_potentially_generic": False,
                            "is_fallback": True
                        }
                    ))
                    logger.info(f"Created fallback candidate {i+1}")
                except Exception as e:
                    logger.error(f"Error creating fallback candidate {i+1}: {str(e)}")
                    
            logger.info(f"Created {min(needed_fallbacks, len(fallback_messages))} fallback candidates")
                
        # Log appropriate messages based on the number of candidates generated
        if len(candidates) < num_candidates:
            if len(candidates) > 0:
                error_msg = f"Generated fewer system message candidates than requested. Requested: {num_candidates}, Generated: {len(candidates)}."
                logger.warning(error_msg)
                logger.info(f"Proceeding with {len(candidates)} candidates")
            else:
                error_msg = f"Failed to generate any system message candidates. This is a critical error."
                logger.error(error_msg)
        
        # Store in cache for future use
        GENERATOR_CACHE.put(cache_key, candidates)
        
        return candidates
    
    except Exception as e:
        error_msg = f"Error generating system messages: {str(e)}"
        logger.error(error_msg)
        
        # Create default fallback messages even in case of errors
        fallback_candidates = []
        try:
            # Create generic fallback messages
            for i in range(min(num_candidates, 3)):  # Create up to 3 fallback messages
                message = f"You are a helpful assistant providing information about {user_message}. Offer accurate and relevant details, using examples where appropriate. Be clear and informative while maintaining a conversational tone."
                fallback_candidates.append(SystemMessageCandidate(
                    content=message,
                    generation_metadata={
                        "generator_id": generator.id,
                        "generator_version": generator.version,
                        "explanation": "Emergency fallback system message due to error",
                        "diversity_level": diversity_level,
                        "generation_index": i,
                        "is_potentially_generic": True,
                        "is_fallback": True,
                        "is_error_fallback": True
                    }
                ))
            logger.info(f"Created {len(fallback_candidates)} emergency fallback candidates due to error: {str(e)}")
            return fallback_candidates
        except Exception as inner_e:
            logger.error(f"Failed to create emergency fallback candidates: {str(inner_e)}")
            # Return empty list as last resort
            return []

# Helper function to save a generator
def save_generator(generator: SystemMessageGenerator) -> bool:
    """Save a system message generator to disk.
    
    Args:
        generator: The generator to save
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Ensure generator has valid version number
        if generator.version <= 0:
            logger.warning(f"Invalid version number {generator.version} for generator {generator.id}, setting to 1")
            generator.version = 1
        
        # Save as the current version (overwrites any existing file)
        file_path = os.path.join(META_PROMPTS_DIR, f"{generator.id}.json")
        with open(file_path, 'w') as f:
            json.dump(generator.model_dump(), f, indent=2)
        logger.info(f"Saved generator {generator.id} to {file_path}")
        
        # Also save as a specific version for history/comparison
        version_file_path = os.path.join(META_PROMPTS_DIR, f"{generator.id}_v{generator.version}.json")
        with open(version_file_path, 'w') as f:
            json.dump(generator.model_dump(), f, indent=2)
        logger.info(f"Saved generator {generator.id} (v{generator.version}) to {version_file_path}")
        
        # Delete any older default.json backup (if it exists)
        backup_path = os.path.join(META_PROMPTS_DIR, f"{generator.id}.json.bak")
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
                logger.debug(f"Removed old backup file {backup_path}")
            except Exception as e:
                logger.debug(f"Failed to remove backup file {backup_path}: {str(e)}")
        
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

# Generate an assistant response using a system message
async def generate_assistant_response(
    system_message: str,
    user_message: str,
    generator_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an assistant response using a system message and user message.
    
    Args:
        system_message: The system message to use
        user_message: The user message to respond to
        generator_model: Model to use for generation
        
    Returns:
        Dictionary with response content and metadata
    """
    logger.info(f"Generating assistant response with system message: {system_message[:50]}...")
    
    try:
        # Create messages for the API call
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Call the LLM API with more conservative parameters
        response = await call_llm_api(
            messages=messages,
            model=generator_model or DEFAULT_GENERATOR_MODEL,
            max_tokens=1024,
            temperature=0.4  # Lower temperature for more reliable output
            # Removed top_p parameter as it's not supported by the current API
        )
        
        # Check for errors
        if "error" in response:
            error_msg = response["error"]
            logger.error(f"Error generating assistant response: {error_msg}")
            return {"error": error_msg}
        
        # Extract content from response
        content = ""
        choice = response["choices"][0] if "choices" in response and response["choices"] else None
        if choice:
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]
        
        # Validate content
        if not content:
            logger.error("No content in assistant response")
            return {"error": "Failed to generate assistant response"}
            
        # Limit content length to avoid issues with extremely long responses
        if len(content) > 15000:
            logger.warning(f"Response content was very long ({len(content)} chars), truncating")
            content = content[:15000] + "... [truncated due to excessive length]"
            
        # Make sure there's meaningful content
        if len(content.strip()) < 10:
            logger.warning(f"Response content was too short: '{content}'")
            return {"error": "Generated response was too short or empty"}
        
        return {
            "content": content,
            "model": generator_model or DEFAULT_GENERATOR_MODEL,
            "timestamp": datetime.now().timestamp()
        }
    
    except Exception as e:
        logger.error(f"Error generating assistant response: {str(e)}")
        return {"error": str(e)}

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

# Evaluate a user-assistant conversation without system message
async def evaluate_user_assistant_conversation(
    user_message: str,
    assistant_message: str,
    evalset_id: str,
    judge_model: Optional[str] = None,
    max_parallel: int = 3,
    evaluation_timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a user-assistant conversation pair using an EvalSet.
    
    Args:
        user_message: The user message
        assistant_message: The assistant's response
        evalset_id: The EvalSet ID to use for evaluation
        judge_model: Optional model to use for judging
        max_parallel: Maximum number of parallel evaluations
        evaluation_timeout: Optional timeout in seconds
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating user-assistant conversation with EvalSet: {evalset_id}")
    
    try:
        # Create conversation with only user and assistant messages
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
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
            logger.error(f"Error evaluating user-assistant conversation: {result['error']}")
            return {"error": result["error"]}
        
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating user-assistant conversation: {str(e)}")
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
        # Debug what's in the cache
        if DEBUG_MODE:
            logger.info(f"Debugging: generator id={generator.id}, version={generator.version}")
            logger.info(f"Debugging: evalset_id={evalset_id}")
            logger.info(f"Debugging: generator_model={generator_model}")
            logger.info(f"Debugging: max_parallel={max_parallel}")
            if progress_callback:
                logger.info("Debugging: progress_callback is provided")
            else:
                logger.info("Debugging: progress_callback is None")
        
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
        
        # Format the self-optimization prompt with better error handling
        try:
            # Check if the meta prompt contains placeholders that might cause issues
            if "{num_candidates}" in self_optimization_prompt:
                logger.warning("self_optimization_prompt contains {num_candidates} which might cause conflicts")
                self_optimization_prompt = self_optimization_prompt.replace("{num_candidates}", "{{num_candidates}}")
                
            if "{user_message}" in self_optimization_prompt:
                logger.warning("self_optimization_prompt contains {user_message} which might cause conflicts")
                self_optimization_prompt = self_optimization_prompt.replace("{user_message}", "{{user_message}}")
                
            formatted_prompt = self_optimization_prompt.format(
                current_meta_prompt=generator.meta_prompt,
                performance_metrics=performance_metrics,
                recent_examples=recent_examples
            )
        except KeyError as e:
            logger.error(f"KeyError formatting self-optimization prompt: {str(e)}")
            # Try a simplified version without the problematic key
            try:
                formatted_prompt = "You are MetaPromptOptimizer. Please create an improved system message generator that follows the same format as the current one but with better performance.\n\n"
                formatted_prompt += f"The current meta-prompt is:\n\n```\n{generator.meta_prompt}\n```\n\n"
                formatted_prompt += "Your response must include all the original placeholders like {num_candidates}, {user_message}, {diversity_instructions}, etc."
            except Exception as e2:
                logger.error(f"Error in fallback formatting: {str(e2)}")
                return {"error": f"Error formatting self-optimization prompt: {str(e)} then {str(e2)}"}
        except Exception as e:
            logger.error(f"Error formatting self-optimization prompt: {str(e)}")
            return {"error": f"Error in self-optimization formatting: {str(e)}"}
        
        # Call LLM to generate improved meta-prompt
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": "Please generate an improved meta-prompt based on the analysis above."}
        ]
        
        # Create messages with clear instruction for meta-prompt generation
        fixed_messages = [
            {"role": "system", "content": "You are a system message optimization expert. Your task is to create improved meta-prompts that generate better system messages. IMPORTANT: Your meta-prompt MUST include these exact placeholders: {num_candidates}, {user_message}, {diversity_instructions}, {base_system_message_instructions}, and {additional_instructions}. These are essential for the template to work."},
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": "Please generate an improved meta-prompt based on the analysis above. Ensure that ALL required placeholders are included in your response."}
        ]
        
        # Define simple JSON schema for the meta-prompt
        meta_prompt_schema = {
            "type": "object",
            "properties": {
                "meta_prompt": {
                    "type": "string",
                    "description": "The full text of the improved meta-prompt. CRITICAL: You MUST include ALL of the following placeholders in your meta-prompt: {num_candidates}, {user_message}, {diversity_instructions}, {base_system_message_instructions}, and {additional_instructions}. These placeholders will be replaced with actual values at runtime."
                }
            },
            "required": ["meta_prompt"]
        }
        
        # Use more conservative parameters for meta-prompt generation with JSON schema
        response = await call_llm_api(
            messages=fixed_messages, 
            model=generator_model,
            temperature=0.3,  # Lower temperature for more reliable output
            max_tokens=4096,  # Allow more tokens for complete meta-prompt
            json_schema=meta_prompt_schema  # Use custom schema for meta-prompt
        )
        
        # Check for errors
        if "error" in response:
            logger.error(f"Error generating improved meta-prompt: {response['error']}")
            return {"error": response["error"]}
        
        # Extract content from response
        content = ""
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            choice = choices[0]
            if "message" in choice and "content" in choice["message"]:
                # Parse the JSON response to get the meta_prompt field
                try:
                    content_json = json.loads(choice["message"]["content"])
                    if "meta_prompt" in content_json:
                        content = content_json["meta_prompt"]
                        logger.info("Successfully extracted meta_prompt from JSON response")
                    else:
                        logger.warning("JSON response missing meta_prompt field")
                        content = choice["message"]["content"]  # Fallback to raw content
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response, using raw content")
                    content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]
        
        if not content or len(content) < 100:
            logger.error("Invalid or too short meta-prompt generated")
            return {"error": "Failed to generate valid improved meta-prompt"}
        
        # Check if all required placeholders are in the new meta prompt
        required_placeholders = [
            "{num_candidates}",
            "{user_message}",
            "{diversity_instructions}",
            "{base_system_message_instructions}",
            "{additional_instructions}"
        ]
        
        # Placeholder check
        missing_placeholders = []
        for placeholder in required_placeholders:
            if placeholder not in content:
                missing_placeholders.append(placeholder)
                
        if missing_placeholders:
            logger.warning(f"New meta prompt is missing placeholders: {missing_placeholders}")
            # Add the missing placeholders at the end
            content += "\n\n# Additional placeholders (auto-added):\n"
            for placeholder in missing_placeholders:
                content += f"\n{placeholder}"
                
        # Create new generator with improved meta-prompt
        new_generator = SystemMessageGenerator(
            id=generator.id,
            version=generator.version + 1,
            meta_prompt=content,
            domain=generator.domain,
            performance_metrics=generator.performance_metrics.copy(),
            created_at=datetime.now().timestamp()
        )
        
        # Set up comprehensive test data covering different domains and query types
        test_messages = [
            # Professional skills
            "How can I improve my public speaking skills?",
            "What are the best practices for project management?",
            
            # Technical/educational
            "How does photosynthesis work?",
            "Explain quantum computing to a beginner",
            
            # Practical/how-to
            "How do I cook a perfect steak?",
            "What's the best way to remove stains from carpet?",
            
            # Creative/abstract
            "How can I become more creative in my writing?",
            "What are some philosophical perspectives on happiness?",
            
            # Health/personal
            "How can I establish a good morning routine?",
            "What exercises are best for lower back pain?"
        ]
        
        # Store test results for reporting
        test_results = {
            "test_messages": test_messages.copy(),
            "success_count": 0,
            "failures": [],
            "sample_outputs": []
        }
        
        success_count = 0
        for test_message in test_messages:
            # Debug info for troubleshooting
            if DEBUG_MODE:
                logger.info(f"Testing generator with message: {test_message}")
                logger.info(f"Meta prompt keys: {new_generator.meta_prompt.count('{num_candidates}')}")
                
            # Use try-except to catch any errors during generation
            try:
                candidates = await generate_system_messages(
                    user_message=test_message,
                    num_candidates=2,  # Just test with 2 for speed
                    generator=new_generator,
                    diversity_level="medium",
                    generator_model=generator_model,
                    additional_instructions="",  # Add missing parameter
                    is_self_optimization=True
                )
                # If we get here, the generation worked
                if candidates and len(candidates) > 0:
                    success_count += 1
                    
                    # Store sample output for reporting
                    if len(candidates) > 0:
                        test_results["sample_outputs"].append({
                            "test_message": test_message,
                            "system_message": candidates[0].content,
                            "success": True
                        })
                        test_results["success_count"] += 1
                else:
                    # Record failure with empty candidates
                    test_results["failures"].append({
                        "test_message": test_message,
                        "error": "No candidates generated",
                        "success": False
                    })
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error testing generator with message '{test_message}': {error_message}")
                
                # Record failure for reporting
                test_results["failures"].append({
                    "test_message": test_message,
                    "error": error_message,
                    "success": False
                })
                
                # Continue testing with the next message despite the error
                continue
        
        # Calculate success rate
        success_rate = success_count / len(test_messages) if test_messages else 0
        
        # Calculate quality metrics for the candidates
        quality_score = 0.0
        diversity_score = 0.0
        
        # Group generated system messages by test message for diversity analysis
        test_message_results = {}
        for test_result in test_results.get("sample_outputs", []):
            test_msg = test_result.get("test_message", "")
            if test_msg not in test_message_results:
                test_message_results[test_msg] = []
            
            if test_result.get("system_message"):
                test_message_results[test_msg].append(test_result.get("system_message", ""))
        
        # Implement candidate quality evaluation
        quality_scores = []
        for test_result in test_results.get("sample_outputs", []):
            if test_result.get("system_message"):
                # Comprehensive quality heuristics for system messages
                message = test_result.get("system_message", "")
                test_message = test_result.get("test_message", "")
                
                # Length score (normalized between 0-1, optimal length ~200-300 chars)
                length = len(message)
                length_score = min(1.0, length / 300) if length < 300 else min(1.0, 500 / max(length, 1))
                
                # Specificity score (check for specific details vs generic language)
                specificity_words = ["specific", "detailed", "precise", "exactly", "particular"]
                specificity_score = sum(1 for word in specificity_words if word.lower() in message.lower()) / len(specificity_words)
                
                # Role clarity (check if the system message clearly defines a role)
                role_phrases = ["you are", "as a", "acting as", "your role", "specialist in", "expert in"]
                role_score = 0.0
                for phrase in role_phrases:
                    if phrase.lower() in message.lower():
                        role_score = 1.0
                        break
                
                # Topic relevance (check if system message mentions keywords from the test message)
                keywords = [word.lower() for word in test_message.split() if len(word) > 4]
                relevance_score = 0.0
                if keywords:
                    matches = sum(1 for keyword in keywords if keyword.lower() in message.lower())
                    relevance_score = min(1.0, matches / max(1, len(keywords) / 2))
                
                # Actionability (check for action-oriented language)
                action_words = ["guide", "explain", "provide", "offer", "describe", "analyze", "compare", "suggest"]
                action_score = min(1.0, sum(1 for word in action_words if word.lower() in message.lower()) / 3)
                
                # Lack of generic templates (penalize messages that seem like unfilled templates)
                generic_patterns = ["[", "]", "{", "}", "<role>", "<expertise>", "<topic>"]
                generic_penalty = sum(1 for pattern in generic_patterns if pattern in message) * 0.2
                
                # Combine into a quality score with weights
                message_quality = (
                    (length_score * 0.15) + 
                    (specificity_score * 0.2) + 
                    (role_score * 0.25) + 
                    (relevance_score * 0.25) + 
                    (action_score * 0.15) - 
                    generic_penalty
                )
                
                # Ensure score is between 0 and 1
                message_quality = max(0.0, min(1.0, message_quality))
                quality_scores.append(message_quality)
        
        # Overall quality is the average of individual scores
        if quality_scores:
            quality_score = sum(quality_scores) / len(quality_scores)
            
        # Calculate diversity score for each test message with multiple results
        diversity_scores = []
        for test_msg, messages in test_message_results.items():
            if len(messages) < 2:  # Need at least 2 messages to calculate diversity
                continue
                
            # Simple diversity measurement based on content differences
            total_diff = 0
            comparisons = 0
            
            # Compare each message pair
            for i in range(len(messages)):
                for j in range(i+1, len(messages)):
                    msg1 = messages[i]
                    msg2 = messages[j]
                    
                    # Jaccard similarity: intersection over union for word sets
                    words1 = set(msg1.lower().split())
                    words2 = set(msg2.lower().split())
                    
                    intersection = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    if union > 0:
                        similarity = intersection / union
                        difference = 1 - similarity  # Convert to difference (0-1)
                        total_diff += difference
                        comparisons += 1
            
            # Average difference score for this test message
            if comparisons > 0:
                diversity_scores.append(total_diff / comparisons)
        
        # Overall diversity is the average of individual test message diversity scores
        if diversity_scores:
            diversity_score = sum(diversity_scores) / len(diversity_scores)
            
        # Combined metric that weights success rate, quality, and diversity
        combined_score = (success_rate * 0.6) + (quality_score * 0.25) + (diversity_score * 0.15)
        
        # Compare to previous version to implement hill climbing
        previous_score = generator.performance_metrics.get("combined_score", 0.0)
        
        # Create a visual representation of the hill climbing process for logging
        score_diff = combined_score - previous_score
        hill_climb_viz = f"Hill Climbing: "
        
        if score_diff > 0.05:
            hill_climb_viz += f"↑↑↑ SIGNIFICANT IMPROVEMENT: {previous_score:.3f} → {combined_score:.3f} (+{score_diff:.3f})"
        elif score_diff > 0:
            hill_climb_viz += f"↑ Improvement: {previous_score:.3f} → {combined_score:.3f} (+{score_diff:.3f})"
        elif score_diff == 0:
            hill_climb_viz += f"→ No change: {previous_score:.3f} = {combined_score:.3f}"
        elif score_diff > -0.05:
            hill_climb_viz += f"↓ Worse: {previous_score:.3f} → {combined_score:.3f} ({score_diff:.3f})"
        else:
            hill_climb_viz += f"↓↓↓ SIGNIFICANT REGRESSION: {previous_score:.3f} → {combined_score:.3f} ({score_diff:.3f})"
            
        # Add component scores for clarity
        hill_climb_viz += f" [success={success_rate:.2f}, quality={quality_score:.2f}, diversity={diversity_score:.2f}]"
        
        logger.info(hill_climb_viz)
        
        # Only save the new generator if it's better than the previous one
        if combined_score >= previous_score:
            # Update performance metrics
            new_generator.performance_metrics["success_rate"] = success_rate
            new_generator.performance_metrics["quality_score"] = quality_score
            new_generator.performance_metrics["diversity_score"] = diversity_score
            new_generator.performance_metrics["combined_score"] = combined_score
            new_generator.performance_metrics["last_optimization"] = datetime.now().timestamp()
            new_generator.performance_metrics["optimization_count"] = generator.performance_metrics.get("optimization_count", 0) + 1
            
            # Store detailed metrics for tracking improvement over time
            if "improvement_history" not in new_generator.performance_metrics:
                new_generator.performance_metrics["improvement_history"] = []
                
            # Add this iteration's scores to the history
            new_generator.performance_metrics["improvement_history"].append({
                "version": new_generator.version,
                "timestamp": datetime.now().timestamp(),
                "success_rate": success_rate,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "combined_score": combined_score
            })
            
            # Save the improved generator
            save_success = save_generator(new_generator)
            
            if not save_success:
                logger.error(f"Failed to save improved generator {new_generator.id}")
                return {"error": "Failed to save improved generator"}
                
            logger.info(f"Hill climbing: Accepted new version with score {combined_score} (previous: {previous_score})")
            improvement = True
        else:
            # Reject the new version since it's not better
            logger.info(f"Hill climbing: Rejected new version with score {combined_score} (previous: {previous_score})")
            improvement = False
        
        # Add test results and diff information for better reporting
        old_meta_prompt_preview = generator.meta_prompt[:300] + "..." if len(generator.meta_prompt) > 300 else generator.meta_prompt
        new_meta_prompt_preview = new_generator.meta_prompt[:300] + "..." if len(new_generator.meta_prompt) > 300 else new_generator.meta_prompt
        
        # Return value now includes hill climbing information
        if improvement:
            return {
                "status": "success",
                "improved": True,
                "old_version": generator.version,
                "new_version": new_generator.version,
                "generator_id": new_generator.id,
                "success_rate": success_rate,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "combined_score": combined_score,
                "previous_score": previous_score,
                "test_results": test_results,
                "old_meta_prompt_preview": old_meta_prompt_preview,
                "new_meta_prompt_preview": new_meta_prompt_preview,
                "message": f"Successfully optimized generator {generator.id} from version {generator.version} to {new_generator.version} (score improved from {previous_score:.3f} to {combined_score:.3f})"
            }
        else:
            # Return information about the rejected optimization
            return {
                "status": "rejected",
                "improved": False,
                "old_version": generator.version,
                "generator_id": generator.id,
                "success_rate": success_rate,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "combined_score": combined_score,
                "previous_score": previous_score,
                "test_results": test_results,
                "message": f"Optimization rejected - new version did not improve over previous (score: {combined_score:.3f}, previous: {previous_score:.3f})"
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
    self_optimize: bool = False,
    continued_from: Optional[str] = None,
    iteration: int = 1
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
        self_optimize: Whether to trigger self-optimization of the generator
        continued_from: ID of a previous optimization run this continues from
        iteration: Iteration number in an optimization sequence (starts at 1)
        
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Starting system message optimization for user message: {user_message[:50]}...")
        
        # Initialize progress tracking
        total_steps = 4  # Generation, response generation, evaluation, ranking
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
        
        # Show warning if this is a high iteration
        if iteration > 2:
            logger.warning(f"Running high iteration optimization (iteration {iteration})")
            
        # Adjust diversity level for iterations to prevent degradation
        effective_diversity = diversity_level
        if iteration > 1:
            if diversity_level == "low":
                effective_diversity = "medium"
                logger.info(f"Increasing diversity from 'low' to 'medium' for iteration {iteration}")
            elif diversity_level == "medium":
                effective_diversity = "high"
                logger.info(f"Increasing diversity from 'medium' to 'high' for iteration {iteration}")
                
        # Generate system message candidates
        candidates = await generate_system_messages(
            user_message=user_message,
            num_candidates=num_candidates,
            generator=generator,
            diversity_level=effective_diversity,  # Use adjusted diversity level
            base_system_message=base_system_message,
            generator_model=generator_model,
            additional_instructions=additional_instructions
        )
        
        if not candidates:
            logger.error("Failed to generate any system message candidates")
            return format_error("Failed to generate system message candidates")
            
        # Add validation for the required fields in candidates
        invalid_candidates = [i for i, c in enumerate(candidates) if not c.content or len(c.content.strip()) < 20]
        if invalid_candidates:
            logger.warning(f"Found {len(invalid_candidates)} invalid candidates: indices {invalid_candidates}")
            # Remove invalid candidates
            candidates = [c for i, c in enumerate(candidates) if i not in invalid_candidates]
            
            # If we've removed all candidates, return an error
            if not candidates:
                logger.error("All generated candidates were invalid")
                return format_error("Failed to generate valid system message candidates")
                
            logger.info(f"Proceeding with {len(candidates)} valid candidates after filtering")
        else:
            logger.info(f"Generated {len(candidates)} valid system message candidates")
        
        # Step 2: Generate assistant responses for each candidate
        current_step += 1
        update_progress(current_step, total_steps, "Generating assistant responses...")

        # Create semaphore for parallel generation
        semaphore = asyncio.Semaphore(max_parallel)
        total_candidates = len(candidates)
        completed_generations = 0
        candidate_responses = {}

        # Define function to generate a response for a single candidate
        async def generate_response_for_candidate(i, candidate):
            nonlocal completed_generations
            
            async with semaphore:
                # Generate response with an added pre-system message to enforce non-JSON output
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that provides clear, natural language responses to user questions. Do NOT output JSON or structured formats unless explicitly requested."},
                    {"role": "system", "content": candidate.content},
                    {"role": "user", "content": user_message}
                ]
                
                # Call API directly instead of using generate_assistant_response
                api_response = await call_llm_api(
                    messages=messages,
                    model=generator_model or DEFAULT_GENERATOR_MODEL,
                    max_tokens=1024,
                    temperature=0.4
                )
                
                # Process the response
                response_result = {}
                if "error" in api_response:
                    response_result["error"] = api_response["error"]
                else:
                    # Extract content from the API response
                    content = ""
                    choice = api_response.get("choices", [{}])[0] if "choices" in api_response else {}
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    elif "text" in choice:
                        content = choice["text"]
                    
                    if content:
                        response_result["content"] = content
                        response_result["model"] = generator_model or DEFAULT_GENERATOR_MODEL
                        response_result["timestamp"] = datetime.now().timestamp()
                    else:
                        response_result["error"] = "No valid content in response"
                
                # Store the result
                if "error" not in response_result:
                    candidate_responses[str(i)] = response_result["content"]
                else:
                    logger.error(f"Error generating response for candidate {i+1}: {response_result['error']}")
                    candidate_responses[str(i)] = f"Error: {response_result['error']}"
                
                # Update progress
                completed_generations += 1
                if progress_callback:
                    sub_progress = f"Generated {completed_generations}/{total_candidates} responses"
                    progress_callback(current_step + (completed_generations / total_candidates), total_steps, sub_progress)
                
                return i, response_result

        # Run all response generations in parallel with rate limiting
        generation_tasks = [generate_response_for_candidate(i, candidate) for i, candidate in enumerate(candidates)]
        generation_results = await asyncio.gather(*generation_tasks)
        
        # Step 3: Evaluate the user-assistant conversation pairs
        current_step += 1
        update_progress(current_step, total_steps, "Evaluating assistant responses...")

        # Reset counters for this phase
        completed_evals = 0

        # Define function to evaluate a single user-assistant conversation
        async def evaluate_candidate_response(i, candidate):
            nonlocal completed_evals
            
            async with semaphore:
                # Get the assistant response for this candidate
                assistant_response = candidate_responses.get(str(i), "")
                if not assistant_response or assistant_response.startswith("Error:"):
                    logger.error(f"No valid response for candidate {i+1}")
                    candidate.score = 0
                    return candidate
                
                # Check for valid response
                if len(assistant_response.strip()) < 10:
                    logger.error(f"Response for candidate {i+1} is too short: '{assistant_response}'")
                    candidate.score = 0
                    candidate.criterion_scores["Response Quality"] = 0
                    return candidate

                try:
                    # Evaluate the user-assistant conversation
                    evaluation_result = await evaluate_user_assistant_conversation(
                        user_message=user_message,
                        assistant_message=assistant_response,
                        evalset_id=evalset_id,
                        judge_model=generator_model,
                        max_parallel=max_parallel
                    )
                    
                    # Extract overall score and per-question scores
                    if "error" not in evaluation_result:
                        # Set overall score
                        candidate.score = evaluation_result.get("summary", {}).get("yes_percentage", 0)
                        
                        # Make sure score is valid (between 0-100)
                        if candidate.score is None or not (0 <= candidate.score <= 100):
                            logger.warning(f"Invalid score for candidate {i+1}: {candidate.score}, resetting to 0")
                            candidate.score = 0
                        
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
                except Exception as e:
                    logger.error(f"Exception evaluating candidate {i+1}: {str(e)}")
                    candidate.score = 0
                
                # Update progress
                completed_evals += 1
                if progress_callback:
                    sub_progress = f"Evaluated {completed_evals}/{total_candidates} responses"
                    progress_callback(current_step + (completed_evals / total_candidates), total_steps, sub_progress)
                
                return candidate

        # Run all evaluations in parallel with rate limiting
        evaluation_tasks = [evaluate_candidate_response(i, candidate) for i, candidate in enumerate(candidates)]
        await asyncio.gather(*evaluation_tasks)
        
        # Step 4: Rank and select the best candidate
        current_step += 1
        update_progress(current_step, total_steps, "Ranking system message candidates...")
        
        # Sort candidates by score (descending)
        candidates.sort(key=lambda x: x.score or 0, reverse=True)
        
        # Assign ranks
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        
        # Verify ranking is correct
        # If somehow all scores are 0 or similar issues, redo scoring using diversity
        if all(c.score == 0 for c in candidates) or all(c.score is None for c in candidates):
            logger.warning("All candidates have zero or None scores, reassigning scores based on content diversity")
            # Assign some basic scores to avoid complete failure
            for i, candidate in enumerate(candidates):
                # Use reverse index as fallback score to ensure different ordering
                candidate.score = 50.0 - (i * 5)  # 50, 45, 40, etc.
                # Basic criterion score
                candidate.criterion_scores["Diversity"] = 100.0 - (i * 10)
            
            # Re-sort and re-rank
            candidates.sort(key=lambda x: x.score or 0, reverse=True)
            for i, candidate in enumerate(candidates):
                candidate.rank = i + 1
                
        # Convert previously generated responses into sample responses format
        update_progress(current_step + 0.5, total_steps, "Formatting sample responses...")
        
        # Store responses for each candidate
        sample_responses = {}
        
        # Use the top 3 candidates only (consistent with previous behavior)
        top_candidates = candidates[:min(3, len(candidates))]
        total_candidates = len(top_candidates)
        
        logger.info(f"Converting previously generated responses for {total_candidates} top candidates to sample format")
        
        for i, candidate in enumerate(top_candidates):
            # Update progress for each candidate
            candidate_progress = current_step + 0.5 + ((i+1) / total_candidates) * 0.5
            update_progress(candidate_progress, total_steps, f"Processing response {i+1}/{total_candidates}...")
            
            try:
                # Get the already generated response from candidate_responses
                content = candidate_responses.get(str(i), "")
                
                if not content or content.startswith("Error:"):
                    if DEBUG_MODE:
                        logger.warning(f"No valid response found for candidate {i+1}, using placeholder")
                    content = "No valid response was generated for this candidate."
                elif len(content.strip()) < 10:
                    if DEBUG_MODE:
                        logger.warning(f"Very short response for candidate {i+1}: '{content}', using placeholder")
                    content = "Response was too short or invalid."
                    
                # Clean content if it's very long (to avoid display issues)
                if len(content) > 10000:
                    content = content[:10000] + "... [truncated due to excessive length]"
                
                # Store the result with the same format as before
                sample_responses[str(i)] = {
                    "model": generator_model or DEFAULT_GENERATOR_MODEL,
                    "content": content,
                    "candidate_index": i,
                    "candidate_rank": candidate.rank,
                    "system_message_preview": candidate.content[:100] + "...",
                    "system_message_id": f"candidate_{i}",
                    "score": candidate.score
                }
                
                if DEBUG_MODE:
                    logger.info(f"Processed sample response {i+1}/{total_candidates}")
                
            except Exception as e:
                if DEBUG_MODE:
                    logger.error(f"Error processing sample response for candidate {i+1}: {str(e)}")
                sample_responses[str(i)] = {
                    "error": f"Failed to process response: {str(e)}",
                    "candidate_index": i,
                    "candidate_rank": candidate.rank
                }
        
        # Apply hill climbing if this is a continuation from a previous run
        if continued_from:
            previous_run = get_optimization_run(continued_from)
            current_best_score = candidates[0].score if candidates else 0
            hill_climbing_applied = False
            
            if previous_run:
                # Check if the previous run had valid candidates
                prev_candidates = previous_run.candidates
                if prev_candidates and len(prev_candidates) > 0:
                    # Get the best score from previous run
                    previous_best_score = max((c.score or 0) for c in prev_candidates)
                    previous_best_candidate = next((c for c in prev_candidates if c.score == previous_best_score), None)
                    
                    # Compare scores and apply hill climbing
                    if previous_best_score > current_best_score:
                        # The previous run was better - prepend the best previous candidate at the top
                        logger.info(f"Hill climbing: Previous best candidate ({previous_best_score:.2f}) outperforms current best ({current_best_score:.2f})")
                        
                        # Use the best candidate from previous run instead of current best
                        if previous_best_candidate:
                            # Insert at the beginning and re-rank
                            candidates.insert(0, previous_best_candidate)
                            
                            # Re-rank candidates after insertion
                            for i, candidate in enumerate(candidates):
                                candidate.rank = i + 1
                                
                            logger.info(f"Hill climbing: Promoted previous best candidate to rank 1 (score: {previous_best_score:.2f})")
                            hill_climbing_applied = True
                    else:
                        # Our current run is better - continue with current candidates
                        improvement = current_best_score - previous_best_score
                        logger.info(f"Hill climbing: Current best candidate ({current_best_score:.2f}) improves upon previous best ({previous_best_score:.2f}), gain: +{improvement:.2f}")
                        hill_climbing_applied = True
            
            # Create a visual representation of the hill climbing process
            if hill_climbing_applied:
                if previous_run:
                    previous_best = max((c.score or 0) for c in previous_run.candidates) if previous_run.candidates else 0
                    current_best = candidates[0].score if candidates else 0
                    improvement = current_best - previous_best
                    
                    hill_climb_viz = f"Hill Climbing: "
                    if improvement > 15:
                        hill_climb_viz += f"↑↑↑ SIGNIFICANT IMPROVEMENT: {previous_best:.2f} → {current_best:.2f} (+{improvement:.2f})"
                    elif improvement > 0:
                        hill_climb_viz += f"↑ Improvement: {previous_best:.2f} → {current_best:.2f} (+{improvement:.2f})"
                    elif improvement == 0:
                        hill_climb_viz += f"→ No change: {previous_best:.2f} = {current_best:.2f}"
                    elif improvement > -15:
                        hill_climb_viz += f"↓ Regression: {previous_best:.2f} → {current_best:.2f} ({improvement:.2f})"
                    else:
                        hill_climb_viz += f"↓↓↓ SIGNIFICANT REGRESSION: {previous_best:.2f} → {current_best:.2f} ({improvement:.2f})"
                    
                    logger.info(hill_climb_viz)
        
        # Calculate the best hill-climbing info before creating the run
        best_ever_score = 0
        best_ever_run_id = None
        hill_climbing_history = []
        
        # Set up initial values if this is a continuation
        if continued_from and previous_run:
            # Get the best score from previous run
            previous_best_score = max((c.score or 0) for c in previous_run.candidates) if previous_run.candidates else 0
            current_best_score = candidates[0].score if candidates else 0
            
            # Get existing history if available
            if "hill_climbing_history" in previous_run.metadata:
                hill_climbing_history = previous_run.metadata["hill_climbing_history"].copy()
            
            # Add current iteration to history (we'll update the run_id after creation)
            new_history_entry = {
                "iteration": iteration,
                "run_id": None,  # Will be updated after run creation
                "previous_best": previous_best_score,
                "current_best": current_best_score,
                "improvement": current_best_score - previous_best_score,
                "timestamp": datetime.now().timestamp()
            }
            hill_climbing_history.append(new_history_entry)
            
            # Calculate all-time best score and run
            best_ever_score = max(
                current_best_score,
                previous_run.metadata.get("best_ever_score", previous_best_score)
            )
            
            # Track which run had the best score
            if current_best_score >= best_ever_score:
                best_ever_run_id = None  # Will be set to current run after creation
            else:
                best_ever_run_id = previous_run.metadata.get("best_ever_run_id", continued_from)
        else:
            # For first run, the best is the current
            best_ever_score = candidates[0].score if candidates else 0
            best_ever_run_id = None  # Will be set to current run after creation
        
        # Create OptimizationRun object
        optimization_run = OptimizationRun(
            user_message=user_message,
            base_system_message=base_system_message,
            evalset_id=evalset_id,
            candidates=candidates,
            best_candidate_index=0 if candidates else None,
            generator_id=generator.id,
            generator_version=generator.version,
            sample_responses=sample_responses,
            candidate_responses=candidate_responses,  # Add the candidate responses
            continued_from=continued_from,
            iteration=iteration,
            metadata={
                "diversity_level": diversity_level,
                "num_candidates_requested": num_candidates,
                "num_candidates_generated": len(candidates),
                "evalset_name": evalset.name,
                "hill_climbing": True if continued_from else False,
                "best_ever_score": best_ever_score,
                "best_ever_run_id": best_ever_run_id if best_ever_run_id else optimization_run.id,
                "hill_climbing_history": hill_climbing_history,
                "current_best_score": candidates[0].score if candidates else 0
            }
        )
        
        # Update hill climbing history with actual run ID
        if hill_climbing_history and hill_climbing_history[-1]["run_id"] is None:
            hill_climbing_history[-1]["run_id"] = optimization_run.id
            optimization_run.metadata["hill_climbing_history"] = hill_climbing_history
            
        # Update best_ever_run_id with actual run ID if needed
        if optimization_run.metadata["best_ever_run_id"] is None:
            optimization_run.metadata["best_ever_run_id"] = optimization_run.id
            
        # Save optimization run
        save_optimization_run(optimization_run)
        
        # Log hill climbing summary if applicable
        if continued_from and previous_run:
            logger.info(f"Hill climbing summary: Best ever score is {best_ever_score:.2f} from run {optimization_run.metadata['best_ever_run_id']}")
        
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
        
        # Add hill climbing information if this is a continuation
        if continued_from and previous_run:
            previous_best_score = max((c.score or 0) for c in previous_run.candidates) if previous_run.candidates else 0
            current_best_score = candidates[0].score if candidates else 0
            improvement = current_best_score - previous_best_score
            
            if improvement > 15:
                hill_climb_status = f"↑↑↑ SIGNIFICANT IMPROVEMENT: +{improvement:.2f} points"
            elif improvement > 0:
                hill_climb_status = f"↑ Improvement: +{improvement:.2f} points"
            elif improvement == 0:
                hill_climb_status = f"→ No change (scores equal)"
            elif improvement > -15:
                hill_climb_status = f"↓ Previous iteration was better by {-improvement:.2f} points"
            else:
                hill_climb_status = f"↓↓↓ SIGNIFICANT REGRESSION: Previous was better by {-improvement:.2f} points"
                
            formatted_results.append(f"- **Hill Climbing**: {hill_climb_status}")
            formatted_results.append(f"- **Previous Best Score**: {previous_best_score:.2f}")
            formatted_results.append(f"- **Current Best Score**: {current_best_score:.2f}")
            formatted_results.append(f"- **Best Score Ever**: {best_ever_score:.2f}")
            formatted_results.append(f"- **Iteration**: {iteration}")
            
            # Add information about which run has the best results
            best_run_id = optimization_run.metadata["best_ever_run_id"]
            if best_run_id == optimization_run.id:
                formatted_results.append(f"- **Best Run**: Current run 🏆")
            else:
                formatted_results.append(f"- **Best Run**: {best_run_id} (previous)")
                # Add button to retrieve the best run's best candidate
                formatted_results.append(f"\nUse `python -m agentoptim.sysopt_cli get-best-candidate --run-id {best_run_id}` to see the best system message from all iterations.")
        
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
            "formatted_message": "\n".join(formatted_results),
            "continued_from": continued_from,
            "iteration": iteration
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
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    continued_from: Optional[str] = None,
    iteration: int = 1
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
                self_optimize=self_optimize,
                continued_from=continued_from,
                iteration=iteration
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
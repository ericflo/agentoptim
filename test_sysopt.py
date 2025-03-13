"""Test script for system message optimization."""

import asyncio
import json
import logging
import os

from agentoptim.sysopt.core import generate_system_messages, load_default_generator
from agentoptim.constants import DEFAULT_GENERATOR_MODEL

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_sysopt")
os.environ["AGENTOPTIM_DEBUG"] = "1"

async def test_generate_system_messages():
    """Test the generation of system messages."""
    logger.info("Starting test for generate_system_messages")
    
    # Load the default generator
    generator = load_default_generator()
    
    # Generate system messages
    user_message = "How does photosynthesis work?"
    num_candidates = 3
    
    logger.info(f"Generating {num_candidates} system messages for: {user_message}")
    
    candidates = await generate_system_messages(
        user_message=user_message,
        num_candidates=num_candidates,
        generator=generator,
        diversity_level="medium",
        generator_model=DEFAULT_GENERATOR_MODEL
    )
    
    logger.info(f"Generated {len(candidates)} candidates")
    
    # Print the candidates
    for i, candidate in enumerate(candidates):
        logger.info(f"Candidate {i+1}:")
        logger.info(f"Content: {candidate.content[:200]}...")
        logger.info(f"Metadata: {json.dumps(candidate.generation_metadata, indent=2)}")
        logger.info("-" * 50)
    
    return len(candidates) > 0

async def main():
    """Run the test."""
    success = await test_generate_system_messages()
    logger.info(f"Test {'passed' if success else 'failed'}")

if __name__ == "__main__":
    asyncio.run(main())
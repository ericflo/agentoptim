#!/usr/bin/env python3

"""
AgentOptim MCP Server

This module serves as the entry point for the AgentOptim MCP server when run as a module:
python -m agentoptim
"""

import os
import sys
import logging

# Configure logging first
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Ensure data directories exist
ensure_data_directories()

# Create a logger
log_file_path = os.path.join(DATA_DIR, "agentoptim.log")
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AGENTOPTIM_DEBUG", "0") == "1" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr to avoid breaking MCP's stdio transport
        logging.FileHandler(log_file_path),
    ],
)

# Create logger at the global level
logger = logging.getLogger("agentoptim")

# Log configuration info
logger.info("AgentOptim MCP Server starting")
logger.info(f"Logging to {log_file_path}")
logger.info(f"Debug mode: {os.environ.get('AGENTOPTIM_DEBUG', '0') == '1'}")
logger.info(f"LM Studio compatibility: {os.environ.get('AGENTOPTIM_LMSTUDIO_COMPAT', '1') == '1'}")

# Define main function to be called by entry point
def main():
    try:
        from agentoptim.server import main as server_main
        server_main()
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

# Run main function when module is executed directly
if __name__ == "__main__":
    main()
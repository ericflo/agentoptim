#!/usr/bin/env python3

"""
AgentOptim CLI - Optimize Commands
"""

import os
import sys
import logging

from agentoptim.sysopt_cli import optimize_setup_parser, handle_optimize

# Configure logging
logger = logging.getLogger(__name__)

def setup_optimize_parser(subparsers):
    """Set up the optimize CLI command parser."""
    try:
        # Use the optimize_setup_parser from sysopt_cli
        return optimize_setup_parser(subparsers)
    except Exception as e:
        logger.error(f"Error setting up optimize parser: {str(e)}")
        # Return a simple placeholder parser that shows an error
        parser = subparsers.add_parser(
            "optimize", 
            help="Optimize system messages for user queries (DISABLED)",
            description="System message optimization is currently unavailable due to an error."
        )
        # Add a dummy function that displays the error
        def error_handler(args):
            print(f"Error: System message optimization is unavailable. {str(e)}")
            return 1
        parser.set_defaults(func=error_handler)
        return parser
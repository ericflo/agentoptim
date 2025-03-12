#!/usr/bin/env python3

"""
AgentOptim CLI - System Message Optimization Commands
"""

import os
import sys
import asyncio
import importlib
import logging

from colorama import Fore, Style

logger = logging.getLogger(__name__)

def setup_optimize_parser(subparsers):
    """Set up the parser for the optimize command.
    
    This function is a bridge to the sysopt_cli implementation.
    
    Args:
        subparsers: The argparse subparsers object
        
    Returns:
        The configured optimize parser
    """
    try:
        # Import the optimize CLI implementation
        sysopt_cli = importlib.import_module("agentoptim.sysopt_cli")
        
        # Call the setup function from sysopt_cli
        optimize_parser = sysopt_cli.optimize_setup_parser(subparsers)
        
        # Set the handler function
        optimize_parser.set_defaults(func=handle_optimize_command)
        
        return optimize_parser
    except ImportError as e:
        logger.error(f"Failed to import sysopt_cli module: {e}")
        
        # Create a placeholder parser that shows an error
        optimize_parser = subparsers.add_parser(
            "optimize",
            help="Optimize system messages (module not available)",
            aliases=["opt", "o"],
        )
        optimize_parser.set_defaults(func=lambda args: print(
            f"{Fore.RED}System message optimization module is not available. "
            f"Please check your installation.{Style.RESET_ALL}"
        ))
        
        return optimize_parser
    except Exception as e:
        logger.error(f"Error setting up optimize parser: {e}")
        return None

def handle_optimize_command(args):
    """Handle the optimize command by delegating to the sysopt_cli module.
    
    Args:
        args: The argparse namespace with command arguments
        
    Returns:
        The exit code from the command
    """
    try:
        # Import the optimize CLI implementation
        sysopt_cli = importlib.import_module("agentoptim.sysopt_cli")
        
        # Call the handle_optimize function
        return asyncio.run(sysopt_cli.handle_optimize(args))
    except ImportError as e:
        logger.error(f"Failed to import sysopt_cli module: {e}")
        print(f"{Fore.RED}Error: System message optimization module is not available.{Style.RESET_ALL}")
        return 1
    except Exception as e:
        logger.error(f"Error handling optimize command: {e}")
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1
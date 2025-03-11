"""Hooks for integrating the system message optimization module with the CLI."""

import logging
import importlib

# Configure logging
logger = logging.getLogger(__name__)

def register_cli_commands(subparsers):
    """Register the system message optimization CLI commands with the main CLI.
    
    This function is called by the main CLI code during startup to register the
    'optimize' command and its subcommands.
    
    Args:
        subparsers: The subparsers object from argparse to add commands to
    
    Returns:
        True if registration was successful, False otherwise
    """
    try:
        # Import the optimize CLI implementation
        sysopt_cli = importlib.import_module("agentoptim.sysopt_cli")
        
        # Register the optimize command
        optimize_parser = sysopt_cli.optimize_setup_parser(subparsers)
        
        # Set the function to call when this command is used
        optimize_parser.set_defaults(func=handle_optimize_command)
        
        logger.info("System message optimization CLI commands registered successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to import sysopt_cli module: {e}")
        return False
    except Exception as e:
        logger.error(f"Error registering system message optimization CLI commands: {e}")
        return False

def handle_optimize_command(args):
    """Handle the optimize command by delegating to the sysopt_cli module.
    
    This function is called by the main CLI code when the 'optimize' command is used.
    
    Args:
        args: The argparse namespace with the parsed command arguments
    
    Returns:
        The exit code from the command (0 for success, non-zero for error)
    """
    try:
        # Import the optimize CLI implementation
        import asyncio
        sysopt_cli = importlib.import_module("agentoptim.sysopt_cli")
        
        # Call the handle_optimize function
        if asyncio._get_running_loop() is not None:
            # We're already in an async context
            task = sysopt_cli.handle_optimize(args)
            return asyncio.get_event_loop().run_until_complete(task)
        else:
            # We need to create a new event loop
            return asyncio.run(sysopt_cli.handle_optimize(args))
    except ImportError as e:
        logger.error(f"Failed to import sysopt_cli module: {e}")
        print(f"Error: Failed to load system message optimization module.")
        return 1
    except Exception as e:
        logger.error(f"Error handling optimize command: {e}")
        print(f"Error: {str(e)}")
        return 1
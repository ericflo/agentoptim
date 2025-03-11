"""CLI extension hooks for AgentOptim.

This module provides a registry and loader for CLI extension hooks,
which allow different modules to register their commands with the
main AgentOptim CLI.
"""

import logging
import importlib
from typing import Dict, Callable, Any, List

logger = logging.getLogger(__name__)

# Registry of CLI extension hooks
_CLI_EXTENSION_HOOKS = {}

def register_cli_extension(name: str, hook_func: Callable[[Any], bool]) -> None:
    """Register a CLI extension hook.
    
    Args:
        name: Name of the extension (for logging)
        hook_func: Function that will register the extension's commands
        
    Returns:
        None
    """
    logger.debug(f"Registering CLI extension hook: {name}")
    _CLI_EXTENSION_HOOKS[name] = hook_func

def load_builtin_extensions() -> None:
    """Load and register all built-in extensions.
    
    Returns:
        None
    """
    try:
        # Try to import and register the system message optimization extension
        logger.debug("Attempting to load system message optimization extension")
        sysopt = importlib.import_module("agentoptim.sysopt")
        register_cli_extension("sysopt", sysopt.register_cli_commands)
        logger.info("System message optimization extension loaded")
    except ImportError:
        logger.info("System message optimization module not available")
    except Exception as e:
        logger.warning(f"Error loading system message optimization extension: {e}")
    
    # Add more built-in extensions here as needed

def apply_extensions(subparsers: Any) -> List[str]:
    """Apply all registered CLI extensions to the given subparsers.
    
    Args:
        subparsers: The subparsers object from argparse
        
    Returns:
        List of extension names that were successfully applied
    """
    applied = []
    for name, hook_func in _CLI_EXTENSION_HOOKS.items():
        try:
            logger.debug(f"Applying CLI extension: {name}")
            if hook_func(subparsers):
                applied.append(name)
                logger.info(f"CLI extension applied: {name}")
            else:
                logger.warning(f"CLI extension failed to apply: {name}")
        except Exception as e:
            logger.error(f"Error applying CLI extension {name}: {e}")
    
    return applied
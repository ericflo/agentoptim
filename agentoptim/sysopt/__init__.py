"""System message optimization module for AgentOptim v2.2.0.

This module provides functionality for optimizing system messages for conversational AI.
"""

from agentoptim.sysopt.core import (
    SystemMessageCandidate,
    SystemMessageGenerator,
    OptimizationRun,
    manage_optimization_runs,
    get_optimization_run,
    list_optimization_runs,
    get_sysopt_stats,
    self_optimize_generator,
    MAX_CANDIDATES,
    DIVERSITY_LEVELS,
    DEFAULT_NUM_CANDIDATES,
)

from agentoptim.sysopt.hooks import (
    register_cli_commands,
    handle_optimize_command,
)

__all__ = [
    # Core functionality
    'SystemMessageCandidate',
    'SystemMessageGenerator',
    'OptimizationRun',
    'manage_optimization_runs',
    'get_optimization_run',
    'list_optimization_runs',
    'get_sysopt_stats',
    'self_optimize_generator',
    'MAX_CANDIDATES',
    'DIVERSITY_LEVELS',
    'DEFAULT_NUM_CANDIDATES',
    
    # CLI integration
    'register_cli_commands',
    'handle_optimize_command',
]
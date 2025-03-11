"""
AgentOptim CLI - Command-line interface for AgentOptim
"""

from agentoptim.cli.core import (
    VERSION,
    FancySpinner,
    format_box,
    get_random_tip,
    format_elapsed_time,
    show_welcome,
    show_success_animation,
    show_success_message,
    display_helpful_error,
    install_completion,
    create_cli_parser,
    run_cli,
)

__all__ = [
    'VERSION',
    'FancySpinner',
    'format_box',
    'get_random_tip',
    'format_elapsed_time',
    'show_welcome',
    'show_success_animation',
    'show_success_message',
    'display_helpful_error',
    'install_completion',
    'create_cli_parser',
    'run_cli',
]
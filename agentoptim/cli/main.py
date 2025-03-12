#!/usr/bin/env python3

"""
AgentOptim CLI Main Entry Point
"""

import os
import sys
import time
import asyncio
import random
import logging

from colorama import Fore, Style
from agentoptim.cli.core import (
    VERSION,
    show_welcome,
    show_success_animation,
    show_success_message,
    display_helpful_error,
    install_completion,
    create_cli_parser,
    get_random_tip,
    format_box,
)

# Create logger
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the CLI."""
    
    # Track command for better error handling
    command = " ".join(sys.argv[1:3]) if len(sys.argv) > 2 else (sys.argv[1] if len(sys.argv) > 1 else "")
    
    # Handle special "--install-completion" flag before parsing other args
    if len(sys.argv) == 2 and sys.argv[1] == "--install-completion":
        install_completion()
        return
    
    # Apply theme if specified
    theme = os.environ.get("AGENTOPTIM_THEME", "").lower()
    if theme in ["ocean", "sunset", "forest", "candy"]:
        theme_colors = {
            'ocean': {'primary': Fore.BLUE, 'secondary': Fore.CYAN},
            'sunset': {'primary': Fore.RED, 'secondary': Fore.YELLOW},
            'forest': {'primary': Fore.GREEN, 'secondary': Fore.CYAN},
            'candy': {'primary': Fore.MAGENTA, 'secondary': Fore.CYAN},
        }
        # Apply theme colors to global variables
        from agentoptim.cli.core import LOGO
        import agentoptim.cli.core as core
        
        theme_color = theme_colors[theme]['primary']
        secondary_color = theme_colors[theme]['secondary']
        core.LOGO = LOGO.replace(Fore.CYAN, theme_color).replace(Fore.YELLOW, secondary_color)
    
    # Show welcome message
    show_welcome()
    
    start_time = time.time()
    show_timer = os.environ.get("AGENTOPTIM_SHOW_TIMER", "0") == "1"
    celebrate = os.environ.get("AGENTOPTIM_CELEBRATE", "0") == "1"
    
    try:
        # Create the CLI parser
        parser, subparsers = create_cli_parser()
        
        # Import command modules and register them
        from agentoptim.cli.server_commands import setup_server_parser
        from agentoptim.cli.dev_commands import setup_dev_parser
        from agentoptim.cli.evalset_commands import setup_evalset_parser
        from agentoptim.cli.run_commands import setup_run_parser
        
        # Try to import the optimize_commands module
        try:
            from agentoptim.cli.optimize_commands import setup_optimize_parser
            has_optimize_commands = True
        except ImportError:
            has_optimize_commands = False
            logger.debug("optimize_commands module not available for direct import")
        
        # Register server commands
        setup_server_parser(subparsers)
        
        # Register dev commands
        setup_dev_parser(subparsers)
        
        # Register evalset commands
        setup_evalset_parser(subparsers)
        
        # Register run commands
        setup_run_parser(subparsers)
        
        # Register optimize commands if available
        if has_optimize_commands:
            setup_optimize_parser(subparsers)
            logger.debug("Registered optimize commands directly")
        
        # Import CLI hooks to register extensions
        from agentoptim import cli_hooks
        
        # For optimize commands in particular, we'll skip the hook registration if we registered directly
        if has_optimize_commands:
            # Register only non-optimize extensions
            logger.debug("Using direct optimize commands - skip sysopt extension")
            def filtered_apply_extensions(subparsers):
                """Apply extensions but skip sysopt that we registered directly."""
                all_extensions = cli_hooks._CLI_EXTENSION_HOOKS.copy()
                if 'sysopt' in all_extensions:
                    del all_extensions['sysopt']
                
                applied = []
                for name, hook_func in all_extensions.items():
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
            
            # Load built-in extensions (including sysopt, but we'll filter in apply)
            cli_hooks.load_builtin_extensions()
            
            # Apply extensions except sysopt
            applied_extensions = filtered_apply_extensions(subparsers)
        else:
            # Standard operation - load and apply all extensions
            cli_hooks.load_builtin_extensions()
            
            # Apply extensions to register commands
            applied_extensions = cli_hooks.apply_extensions(subparsers)
            
        logger.debug(f"Applied CLI extensions: {applied_extensions}")
        
        # Parse arguments
        args = parser.parse_args()
        
        # Handle case where no command is specified
        if not hasattr(args, 'command') or args.command is None:
            parser.print_help()
            return
        
        # Call the appropriate command handler
        if hasattr(args, 'func'):
            # Execute the command
            func = args.func
            result = func(args)
            
            # Show success message or timer
            if show_timer:
                elapsed_time = time.time() - start_time
                show_success_animation()
                show_success_message(command, elapsed_time)
                
                # Show a random tip occasionally (20% chance) after longer commands
                if elapsed_time > 2 and random.random() < 0.2:
                    tip = get_random_tip()
                    print(format_box(
                        "🌟 Pro Tip", 
                        tip,
                        style="single", 
                        border_color=Fore.YELLOW,
                        title_align="left"
                    ), file=sys.stderr)
            elif celebrate:
                # Show success message without timer
                show_success_animation()
                show_success_message(command)
        else:
            # No handler found
            parser.print_help()
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        print(f"\n{Fore.YELLOW}Operation canceled by user{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        # Show execution time even for errors if enabled
        if show_timer:
            elapsed_time = time.time() - start_time
            from agentoptim.cli.core import format_elapsed_time
            time_str = format_elapsed_time(elapsed_time)
            print(f"\n{Fore.RED}⚠️ Command failed after {time_str}{Style.RESET_ALL}", file=sys.stderr)
        
        # Log the error for debugging
        logger.error(f"Error in AgentOptim CLI: {str(e)}", exc_info=True)
        
        # Check for high contrast mode for better readability
        high_contrast = os.environ.get("AGENTOPTIM_HIGH_CONTRAST", "0") == "1"
        error_color = Fore.LIGHTRED_EX if high_contrast else Fore.RED
        
        # Show error message with a sad face emoji
        print(f"\n{error_color}😞 Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        
        # Provide helpful suggestions based on the error
        display_helpful_error(e, command)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
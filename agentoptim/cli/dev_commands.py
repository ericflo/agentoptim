"""
Development commands for the AgentOptim CLI.
These commands are primarily used by developers and advanced users.
"""

import os
import json
import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style

from agentoptim.cli.core import format_box
from agentoptim.utils import DATA_DIR

# Set up logger
logger = logging.getLogger(__name__)

def setup_dev_parser(subparsers):
    """Set up the development command parser."""
    # Add dev command (hidden from main help)
    dev_parser = subparsers.add_parser(
        "dev",
        help=argparse.SUPPRESS,
        description="Development commands for AgentOptim"
    )
    
    # Add dev subcommands
    dev_subparsers = dev_parser.add_subparsers(
        dest="subcommand",
        help="Subcommand to run"
    )
    
    # Cache command
    cache_parser = dev_subparsers.add_parser(
        "cache",
        help="View and manage cache statistics",
        description="View and manage caching statistics for AgentOptim."
    )
    cache_parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear all caches"
    )
    cache_parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Logs command
    logs_parser = dev_subparsers.add_parser(
        "logs",
        help="View application logs",
        description="View and analyze AgentOptim application logs."
    )
    logs_parser.add_argument(
        "--tail", "-t",
        type=int,
        default=20,
        help="Number of log lines to show (default: 20)"
    )
    logs_parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Follow log output in real-time"
    )
    logs_parser.add_argument(
        "--level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Minimum log level to show"
    )
    
    # Set the function to call when the dev command is used
    dev_parser.set_defaults(func=handle_dev_command)
    
    return dev_parser

def handle_dev_command(args):
    """Handle the development commands."""
    if not hasattr(args, 'subcommand') or not args.subcommand:
        print(f"{Fore.RED}Error: No subcommand specified{Style.RESET_ALL}")
        return 1
        
    if args.subcommand == "cache":
        return handle_cache_command(args)
    elif args.subcommand == "logs":
        return handle_logs_command(args)
    else:
        print(f"{Fore.RED}Error: Unknown subcommand '{args.subcommand}'{Style.RESET_ALL}")
        return 1

def handle_cache_command(args):
    """Handle the cache command."""
    try:
        # Import relevant modules
        from agentoptim.cache import get_evalset_cache_stats, get_api_cache_stats, clear_all_caches
        
        # Clear caches if requested
        if args.clear:
            clear_all_caches()
            print(f"{Fore.GREEN}✓ All caches cleared successfully{Style.RESET_ALL}")
            return 0
            
        # Get cache statistics
        evalset_stats = get_evalset_cache_stats()
        api_stats = get_api_cache_stats()
        
        # Format the output based on the requested format
        if args.format == "json":
            stats = {
                "evalset_cache": evalset_stats,
                "api_cache": api_stats
            }
            print(json.dumps(stats, indent=2))
        else:
            # Format as text with colored output
            evalset_info = (
                f"Size: {evalset_stats.get('size', 0)}\n"
                f"Max Size: {evalset_stats.get('maxsize', 0)}\n"
                f"Hit Rate: {evalset_stats.get('hit_rate', 0):.1f}%\n"
                f"Hits: {evalset_stats.get('hits', 0)}\n"
                f"Misses: {evalset_stats.get('misses', 0)}\n"
                f"Current Items: {evalset_stats.get('currsize', 0)}\n"
                f"Saved API Calls: {evalset_stats.get('hits', 0)}"
            )
            
            api_info = (
                f"Size: {api_stats.get('size', 0)}\n"
                f"Max Size: {api_stats.get('maxsize', 0)}\n"
                f"Hit Rate: {api_stats.get('hit_rate', 0):.1f}%\n"
                f"Hits: {api_stats.get('hits', 0)}\n"
                f"Misses: {api_stats.get('misses', 0)}\n"
                f"Current Items: {api_stats.get('currsize', 0)}\n"
                f"Saved API Calls: {api_stats.get('hits', 0)}"
            )
            
            # Calculate total estimated savings (assuming $0.001 per API call saved)
            saved_calls = evalset_stats.get('hits', 0) + api_stats.get('hits', 0)
            estimated_savings = saved_calls * 0.001  # $0.001 per API call saved
            
            savings_info = (
                f"Total Cached API Calls: {saved_calls}\n"
                f"Estimated Cost Savings: ${estimated_savings:.2f}\n"
                f"Enabled: {evalset_stats.get('enabled', True)}"
            )
            
            print(format_box("EvalSet Cache", evalset_info, style="rounded", border_color=Fore.CYAN))
            print()
            print(format_box("API Cache", api_info, style="rounded", border_color=Fore.GREEN))
            print()
            print(format_box("Summary", savings_info, style="rounded", border_color=Fore.MAGENTA))
            
            # Add a note about how to clear caches
            print(f"\n{Fore.YELLOW}ℹ️  To clear all caches, run:{Style.RESET_ALL}")
            print("   agentoptim dev cache --clear")
            
        return 0
        
    except Exception as e:
        logger.error(f"Error in cache command: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_logs_command(args):
    """Handle the logs command."""
    try:
        log_file = os.path.join(DATA_DIR, "agentoptim.log")
        
        if not os.path.exists(log_file):
            print(f"{Fore.YELLOW}ℹ️  No log file found at {log_file}{Style.RESET_ALL}")
            return 0
            
        # Function to filter log lines by level
        def filter_by_level(line, min_level):
            level_map = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50
            }
            
            # Try to detect the log level from the line
            for level, value in level_map.items():
                if f" - {level} - " in line:
                    line_level = value
                    break
            else:
                # If we can't detect the level, assume it's INFO
                line_level = 20
                
            # Filter based on the minimum level
            return line_level >= level_map.get(min_level, 0)
            
        # Function to colorize log line based on level
        def colorize_log_line(line):
            if " - DEBUG - " in line:
                level_color = Fore.BLUE
            elif " - INFO - " in line:
                level_color = Fore.GREEN
            elif " - WARNING - " in line:
                level_color = Fore.YELLOW
            elif " - ERROR - " in line:
                level_color = Fore.RED
            elif " - CRITICAL - " in line:
                level_color = Fore.RED + Style.BRIGHT
            else:
                level_color = ""
                
            if level_color:
                # Colorize just the level part
                parts = line.split(" - ", 3)
                if len(parts) >= 3:
                    return f"{parts[0]} - {parts[1]} - {level_color}{parts[2]}{Style.RESET_ALL} - {parts[3] if len(parts) > 3 else ''}"
            
            return line
            
        # Follow logs in real-time if requested
        if args.follow:
            import time
            
            print(f"{Fore.YELLOW}Following log file: {log_file}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Press Ctrl+C to stop{Style.RESET_ALL}")
            print()
            
            # Start from the end of the file
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)  # Go to the end of the file
                
                try:
                    while True:
                        line = f.readline()
                        if line:
                            if args.level is None or filter_by_level(line, args.level):
                                print(colorize_log_line(line.rstrip()))
                        else:
                            time.sleep(0.1)
                except KeyboardInterrupt:
                    print(f"\n{Fore.YELLOW}Log following stopped by user{Style.RESET_ALL}")
                    
            return 0
            
        # Read the last N lines of the log file
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            # Read the whole file for simplicity (this could be optimized for very large files)
            lines = f.readlines()
            
            # Filter by level if specified
            if args.level:
                lines = [line for line in lines if filter_by_level(line, args.level)]
                
            # Get the last N lines
            tail_lines = lines[-args.tail:] if args.tail > 0 else lines
            
            if not tail_lines:
                print(f"{Fore.YELLOW}ℹ️  No matching log entries found{Style.RESET_ALL}")
                return 0
                
            # Print the log file with colored output
            print(f"{Fore.YELLOW}Log file: {log_file}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Showing last {len(tail_lines)} entries{Style.RESET_ALL}")
            print()
            
            for line in tail_lines:
                print(colorize_log_line(line.rstrip()))
                
            # Show how to view more logs
            if len(lines) > args.tail:
                print()
                print(f"{Fore.YELLOW}ℹ️  To see more log entries, run:{Style.RESET_ALL}")
                print(f"   agentoptim dev logs --tail {args.tail * 2}")
                print(f"{Fore.YELLOW}ℹ️  To follow logs in real-time, run:{Style.RESET_ALL}")
                print(f"   agentoptim dev logs --follow")
                
        return 0
        
    except Exception as e:
        logger.error(f"Error in logs command: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1
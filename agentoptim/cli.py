#!/usr/bin/env python3

"""
AgentOptim CLI - Redesigned for developer joy
"""

import os
import sys
import json
import argparse
import logging
import uuid
import time
import textwrap
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import colorama
from colorama import Fore, Style

# Configure logging first
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Initialize colorama for cross-platform color support
colorama.init()

# Ensure data directories exist
ensure_data_directories()

# Create a logger
log_file_path = os.path.join(DATA_DIR, "agentoptim.log")
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AGENTOPTIM_DEBUG", "0") == "1" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file_path),
    ],
)

# Create logger at the global level
logger = logging.getLogger("agentoptim")

# Constants
VERSION = "2.1.1"  # Updated for CLI delight features
MAX_WIDTH = 100  # Maximum width for formatted output


def setup_parser():
    """Set up the argument parser for the CLI with resource-verb pattern."""
    parser = argparse.ArgumentParser(
        description=f"AgentOptim v{VERSION}: Evaluate and optimize AI conversation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Start the server
          agentoptim server
          
          # Create an evaluation set interactively (generates a unique ID)
          agentoptim evalset create --wizard
          
          # List all evaluation sets to get their IDs
          agentoptim evalset list
          
          # Run evaluation with file input (system auto-generates result ID)
          agentoptim run create <evalset-id> conversation.json
          
          # Create and evaluate a conversation interactively
          agentoptim run create <evalset-id> --interactive
          
          # Evaluate a text file as a single user message
          agentoptim run create <evalset-id> --text response.txt
          
          # Get the most recent evaluation result (no need to remember IDs)
          agentoptim run get latest
          
          # View all your evaluation runs
          agentoptim run list
          
          # Compare two evaluation runs
          agentoptim run compare latest latest-1
          
          # Export evaluation results to a file
          agentoptim run export latest --format html --output report.html
          
          # Install shell tab completion
          agentoptim --install-completion
          
        Environment variables:
          AGENTOPTIM_SHOW_TIMER=1   Show execution time for commands
          AGENTOPTIM_DEBUG=1        Enable detailed debug logging
        """)
    )
    
    parser.add_argument('--version', action='version', version=f'AgentOptim v{VERSION}')
    parser.add_argument('--install-completion', action='store_true', help='Install shell completion script')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress non-error output for scripting')
    
    # Create subparsers for resources
    subparsers = parser.add_subparsers(dest="resource", help="Resource to manage")
    
    # === SERVER RESOURCE ===
    server_parser = subparsers.add_parser("server", help="Start the MCP server")
    server_parser.add_argument("--port", type=int, help="Port to run the server on")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    server_parser.add_argument("--provider", choices=["local", "openai", "anthropic"], default="local", 
                         help="API provider (default: local)")
    
    # === EVALSET RESOURCE ===
    evalset_parser = subparsers.add_parser("evalset", help="Manage evaluation sets", aliases=["es"])
    evalset_subparsers = evalset_parser.add_subparsers(dest="action", help="Action to perform")
    
    # evalset list
    es_list_parser = evalset_subparsers.add_parser("list", help="List all evaluation sets")
    es_list_parser.add_argument("--format", choices=["table", "json", "yaml"], default="table", 
                            help="Output format (default: table)")
    es_list_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # evalset get
    es_get_parser = evalset_subparsers.add_parser("get", help="Get details about a specific evaluation set")
    es_get_parser.add_argument("evalset_id", help="ID of the evaluation set")
    es_get_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                           help="Output format (default: text)")
    es_get_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # evalset create
    es_create_parser = evalset_subparsers.add_parser("create", help="Create a new evaluation set")
    es_create_parser.add_argument("--wizard", action="store_true", help="Create interactively with step-by-step prompts")
    es_create_parser.add_argument("--name", help="Name of the evaluation set")
    es_create_parser.add_argument("--questions", help="File containing questions (one per line) or comma-separated list")
    es_create_parser.add_argument("--short-desc", help="Short description of the evaluation set")
    es_create_parser.add_argument("--long-desc", help="Long description of the evaluation set")
    es_create_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                              help="Output format (default: text)")
    es_create_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # evalset update
    es_update_parser = evalset_subparsers.add_parser("update", help="Update an existing evaluation set")
    es_update_parser.add_argument("evalset_id", help="ID of the evaluation set to update")
    es_update_parser.add_argument("--name", help="New name for the evaluation set")
    es_update_parser.add_argument("--questions", help="File containing questions or comma-separated list")
    es_update_parser.add_argument("--short-desc", help="New short description")
    es_update_parser.add_argument("--long-desc", help="New long description")
    es_update_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                              help="Output format (default: text)")
    es_update_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # evalset delete
    es_delete_parser = evalset_subparsers.add_parser("delete", help="Delete an evaluation set")
    es_delete_parser.add_argument("evalset_id", help="ID of the evaluation set to delete")
    es_delete_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                              help="Output format (default: text)")
    es_delete_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # === RUN RESOURCE ===
    run_parser = subparsers.add_parser("run", help="Manage evaluation runs", aliases=["r"])
    run_subparsers = run_parser.add_subparsers(dest="action", help="Action to perform")
    
    # run list
    run_list_parser = run_subparsers.add_parser("list", help="List all evaluation runs")
    run_list_parser.add_argument("--evalset", dest="evalset_id", help="Filter by evaluation set ID")
    run_list_parser.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    run_list_parser.add_argument("--limit", type=int, default=10, dest="page_size",
                              help="Number of items per page (default: 10, max: 100)")
    run_list_parser.add_argument("--format", choices=["table", "json", "yaml"], default="table",
                              help="Output format (default: table)")
    run_list_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # run get
    run_get_parser = run_subparsers.add_parser("get", help="Get a specific evaluation run")
    run_get_parser.add_argument("eval_run_id", help="ID of a specific run to retrieve, or simply use 'latest' to get most recent result")
    run_get_parser.add_argument("--format", choices=["text", "json", "yaml", "markdown", "html"], default="text",
                            help="Output format (default: text)")
    run_get_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # run export
    run_export_parser = run_subparsers.add_parser("export", help="Export evaluation results in various formats")
    run_export_parser.add_argument("eval_run_id", help="ID of a specific run to export, or 'latest' for most recent")
    run_export_parser.add_argument("--format", choices=["markdown", "csv", "html", "json", "pdf"], default="markdown",
                               help="Export format (default: markdown)")
    run_export_parser.add_argument("--output", type=str, help="Output file (required)")
    run_export_parser.add_argument("--title", type=str, help="Custom title for the export")
    run_export_parser.add_argument("--template", type=str, help="Custom template file for HTML/PDF export")
    run_export_parser.add_argument("--color", action="store_true", help="Include colors in HTML output")
    run_export_parser.add_argument("--charts", action="store_true", help="Include charts in HTML/PDF output")
    
    # run compare - Compare two evaluation runs
    run_compare_parser = run_subparsers.add_parser("compare", help="Compare two evaluation runs side by side")
    run_compare_parser.add_argument("first_run_id", help="ID of first run to compare (or 'latest')")
    run_compare_parser.add_argument("second_run_id", help="ID of second run to compare (or 'latest-1')")
    run_compare_parser.add_argument("--output", type=str, help="Output file to save the comparison")
    run_compare_parser.add_argument("--format", choices=["text", "html"], default="text", 
                                help="Output format (default: text)")
    run_compare_parser.add_argument("--color", action="store_true", help="Use colors in output")
    run_compare_parser.add_argument("--detailed", action="store_true", help="Show detailed reasoning")
    
    # run create
    run_create_parser = run_subparsers.add_parser("create", help="Run a new evaluation (result ID will be auto-generated)")
    run_create_parser.add_argument("evalset_id", help="ID of the evaluation set to use (get IDs with 'evalset list')")
    run_create_parser.add_argument("conversation", nargs="?", help="Conversation file (JSON format)")
    run_create_parser.add_argument("--interactive", action="store_true", help="Create conversation interactively")
    run_create_parser.add_argument("--text", help="Text file to evaluate (treated as a single user message)")
    run_create_parser.add_argument("--model", help="Judge model to use for evaluation")
    run_create_parser.add_argument("--provider", choices=["local", "openai", "anthropic"], default="local", 
                                help="API provider (default: local)")
    run_create_parser.add_argument("--concurrency", type=int, default=3, dest="max_parallel",
                                help="Maximum parallel evaluations (default: 3)")
    run_create_parser.add_argument("--brief", action="store_true", dest="omit_reasoning",
                                help="Omit detailed reasoning from results")
    run_create_parser.add_argument("--format", choices=["text", "json", "yaml", "csv"], default="text", 
                                help="Output format (default: text)")
    run_create_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # === DEV RESOURCE (Hidden from normal help) ===
    dev_parser = subparsers.add_parser("dev", help=argparse.SUPPRESS)
    dev_subparsers = dev_parser.add_subparsers(dest="action", help="Developer actions")
    
    # dev cache
    dev_cache_parser = dev_subparsers.add_parser("cache", help="View cache statistics")
    dev_cache_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                               help="Output format (default: text)")
    dev_cache_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # dev logs
    dev_logs_parser = dev_subparsers.add_parser("logs", help="View application logs")
    dev_logs_parser.add_argument("--lines", type=int, default=50, help="Number of lines to show (default: 50)")
    dev_logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    
    # No legacy commands - we've removed backward compatibility
    
    return parser


def handle_output(data: Any, format_type: str, output_file: Optional[str] = None):
    """Format and output data according to the specified format and destination."""
    # Check if we're in quiet mode
    quiet_mode = os.environ.get("AGENTOPTIM_QUIET", "0") == "1"
    
    if format_type == "json":
        formatted_data = json.dumps(data, indent=2)
    elif format_type == "yaml":
        import yaml
        formatted_data = yaml.dump(data, sort_keys=False)
    elif format_type == "csv" and isinstance(data, list):
        import pandas as pd
        df = pd.DataFrame(data)
        formatted_data = df.to_csv(index=False)
    elif format_type == "table" and isinstance(data, list):
        # Try to use rich tables if available
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.style import Style
            
            # Create a table with minimal styling that shows full content
            from rich.box import SIMPLE
            console = Console(width=None)  # No width limit to prevent truncation
            table = Table(show_header=True, header_style="bold cyan", box=SIMPLE)
            
            # Add columns
            if len(data) > 0:
                sample_row = data[0]
                for column in sample_row.keys():
                    # Don't set specific widths to allow full content display
                    if column.lower() == "id":
                        table.add_column(column.upper(), style="dim", no_wrap=True)
                    elif "description" in column.lower():
                        table.add_column(column.title(), style="green")
                    elif "name" in column.lower():
                        table.add_column(column.title(), style="yellow")
                    elif "questions" in column.lower() or "count" in column.lower():
                        table.add_column(column.title(), style="blue", justify="right")
                    elif "percentage" in column.lower() or "score" in column.lower():
                        table.add_column(column.title(), style="magenta", justify="right")
                    elif "date" in column.lower() or "time" in column.lower():
                        table.add_column(column.title(), style="cyan")
                    else:
                        table.add_column(column.title())
                
                # Add rows
                for row in data:
                    # Format certain columns specially
                    formatted_row = []
                    for col, val in row.items():
                        if isinstance(val, float) and ("percentage" in col.lower() or "score" in col.lower()):
                            formatted_row.append(f"{val:.1f}%")
                        elif isinstance(val, (dict, list)):
                            formatted_row.append(str(val)[:20] + "..." if len(str(val)) > 20 else str(val))
                        elif col.lower() == "id":
                            # Keep full IDs for usability
                            formatted_row.append(str(val))
                        else:
                            formatted_row.append(str(val))
                    
                    table.add_row(*formatted_row)
                
                # Render the table to a string without width constraints
                export_console = Console(record=True, width=None)  # No width limit to prevent truncation
                export_console.print(table)
                formatted_data = export_console.export_text()
            else:
                formatted_data = "No data available."
        except ImportError:
            # Fall back to pandas if rich isn't available
            import pandas as pd
            df = pd.DataFrame(data)
            formatted_data = df.to_string(index=False)
            formatted_data = f"{Fore.YELLOW}Tip: Install 'rich' for better table formatting: pip install rich{Style.RESET_ALL}\n\n{formatted_data}"
    else:
        # Default to text format (using the formatted_message if available)
        if isinstance(data, dict) and "formatted_message" in data:
            formatted_data = data["formatted_message"]
        elif isinstance(data, dict) and "result" in data:
            formatted_data = data["result"]
        elif isinstance(data, str):
            formatted_data = data
        else:
            formatted_data = str(data)
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_data)
        if not quiet_mode:
            print(f"{Fore.GREEN}✅ Output saved to: {output_file}{Style.RESET_ALL}")
    else:
        if not quiet_mode or format_type in ["json", "yaml", "csv"]:  # Always print data output in machine-readable formats
            print(formatted_data)


def parse_questions(questions_input: str) -> List[str]:
    """Parse questions from a file or comma-separated string."""
    if os.path.isfile(questions_input):
        with open(questions_input, 'r', encoding='utf-8') as f:
            # Split by lines and strip whitespace
            questions = [line.strip() for line in f.readlines() if line.strip()]
    else:
        # Assume comma-separated string
        questions = [q.strip() for q in questions_input.split(",") if q.strip()]
    
    return questions


def load_conversation(conversation_file: Optional[str], text_file: Optional[str]) -> List[Dict[str, str]]:
    """Load a conversation from a file or create one from a text file."""
    if conversation_file:
        with open(conversation_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"{Fore.RED}Error: {conversation_file} is not valid JSON{Style.RESET_ALL}")
                sys.exit(1)
    elif text_file:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
            # Create a simple conversation with one user message
            return [
                {"role": "user", "content": text_content}
            ]
    else:
        print(f"{Fore.RED}Error: Either conversation or text file must be provided{Style.RESET_ALL}")
        sys.exit(1)


def run_interactive_wizard(args):
    """Run the interactive wizard for creating an evaluation set."""
    print(f"{Fore.CYAN}===== AgentOptim Evaluation Set Creation Wizard ====={Style.RESET_ALL}")
    print("This wizard will guide you through creating a new evaluation set\n")
    
    # Prompt for name
    name = input(f"{Fore.GREEN}Name of the evaluation set: {Style.RESET_ALL}")
    
    # Prompt for short description
    short_desc = input(f"{Fore.GREEN}Short description (one sentence): {Style.RESET_ALL}")
    
    # Prompt for long description
    print(f"{Fore.GREEN}Long description (detailed, press Enter twice when done):{Style.RESET_ALL}")
    long_desc_lines = []
    while True:
        line = input()
        if not line and (not long_desc_lines or not long_desc_lines[-1]):
            # Empty line after another empty line or at the beginning, we're done
            break
        long_desc_lines.append(line)
    long_desc = "\n".join(long_desc_lines)
    
    # Prompt for questions
    print(f"{Fore.GREEN}Enter evaluation questions (one per line, press Enter twice when done):{Style.RESET_ALL}")
    questions = []
    while True:
        line = input().strip()
        if not line and (not questions or not questions[-1]):
            # Empty line after another empty line or at the beginning, we're done
            break
        if line:
            questions.append(line)
    
    return {
        "name": name,
        "short_description": short_desc,
        "long_description": long_desc,
        "questions": questions
    }


# Legacy command handling code removed - no backward compatibility


def get_suggestion(command, available_commands):
    """Get command suggestion for typos based on Levenshtein distance."""
    if not command or not available_commands:
        return None
        
    # Calculate Levenshtein distance (minimal string edit operations) 
    # between command and each available command
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Find the closest match
    closest = None
    min_distance = float('inf')
    
    for cmd in available_commands:
        distance = levenshtein_distance(command.lower(), cmd.lower())
        if distance < min_distance:
            min_distance = distance
            closest = cmd
            
    # Only suggest if the distance is reasonable (relative to command length)
    threshold = max(2, len(command) // 3)  # Allow more errors for longer commands
    if min_distance <= threshold:
        return closest
    return None


def run_cli():
    """Run the AgentOptim CLI based on the provided arguments."""
    parser = setup_parser()
    
    # First check for quiet flag
    quiet_mode = False
    for arg in sys.argv:
        if arg in ['--quiet', '-q']:
            quiet_mode = True
            os.environ["AGENTOPTIM_QUIET"] = "1"
            break
    
    # Custom print function that respects quiet mode
    def cli_print(*args, error=False, **kwargs):
        if error or not quiet_mode:
            print(*args, **kwargs)
    
    # Split into two stages: first check for mistyped commands without raising errors,
    # then parse properly for execution
    if len(sys.argv) > 1:
        resource = sys.argv[1]
        
        # Check for common resources
        main_resources = ["server", "evalset", "es", "run", "r", "dev"]
        if resource not in main_resources and not resource.startswith('-'):
            # Check for typos
            suggestion = get_suggestion(resource, main_resources)
            if suggestion:
                cli_print(f"{Fore.YELLOW}Command '{resource}' not found. Did you mean '{suggestion}'?{Style.RESET_ALL}", error=True)
                cli_print(f"Run {Fore.CYAN}agentoptim --help{Style.RESET_ALL} to see available commands.", error=True)
                sys.exit(1)
                
        # Check for action typos if appropriate
        if len(sys.argv) > 2 and resource in ["evalset", "es", "run", "r", "dev"]:
            action = sys.argv[2]
            
            # Define valid actions for each resource
            action_map = {
                "evalset": ["list", "get", "create", "update", "delete"],
                "es": ["list", "get", "create", "update", "delete"],
                "run": ["list", "get", "create"],
                "r": ["list", "get", "create"],
                "dev": ["cache", "logs"]
            }
            
            valid_actions = action_map.get(resource, [])
            
            if action not in valid_actions and not action.startswith('-'):
                # Check for typos in action
                suggestion = get_suggestion(action, valid_actions)
                if suggestion:
                    cli_print(f"{Fore.YELLOW}Action '{action}' not found for '{resource}'. Did you mean '{suggestion}'?{Style.RESET_ALL}", error=True)
                    cli_print(f"Valid actions for '{resource}': {', '.join(valid_actions)}", error=True)
                    sys.exit(1)
    
    args = parser.parse_args()
    
    # Handle no args case
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Handle commands based on resource/action pattern
    if args.resource == "server":
        # Set environment variables based on arguments
        if args.debug:
            os.environ["AGENTOPTIM_DEBUG"] = "1"
        if args.port:
            os.environ["AGENTOPTIM_PORT"] = str(args.port)
            
        # Import constants here to avoid circular imports
        from agentoptim.constants import (
            DEFAULT_LOCAL_API_BASE,
            DEFAULT_OPENAI_API_BASE,
            DEFAULT_ANTHROPIC_API_BASE,
            DEFAULT_LOCAL_MODEL,
            DEFAULT_OPENAI_MODEL,
            DEFAULT_ANTHROPIC_MODEL
        )
        
        # Configure provider settings
        if args.provider:
            # Only set API_BASE if not already set by user
            if "AGENTOPTIM_API_BASE" not in os.environ:
                if args.provider == "openai":
                    os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_OPENAI_API_BASE
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = DEFAULT_OPENAI_MODEL
                elif args.provider == "anthropic":
                    os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_ANTHROPIC_API_BASE
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = DEFAULT_ANTHROPIC_MODEL
                elif args.provider == "local":
                    os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_LOCAL_API_BASE
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = DEFAULT_LOCAL_MODEL
        
        # Log configuration info
        logger.info("AgentOptim MCP Server starting")
        logger.info(f"Logging to {log_file_path}")
        logger.info(f"Debug mode: {os.environ.get('AGENTOPTIM_DEBUG', '0') == '1'}")
        logger.info(f"Provider: {args.provider}")
        logger.info(f"API base: {os.environ.get('AGENTOPTIM_API_BASE')}")
        logger.info(f"Judge model: {os.environ.get('AGENTOPTIM_JUDGE_MODEL')}")
        
        # Start the server
        try:
            from agentoptim.server import main as server_main
            server_main()
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}", exc_info=True)
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
            sys.exit(1)
    
    elif args.resource in ["evalset", "es"]:
        # Import necessary modules
        from agentoptim.evalset import manage_evalset
        
        if not args.action:
            print(f"{Fore.RED}Error: No action specified for evalset resource{Style.RESET_ALL}")
            print("Available actions: list, get, create, update, delete")
            sys.exit(1)
        
        # Handle evalset actions
        if args.action == "list":
            result = manage_evalset(action="list")
            
            # Convert to a list format for table display
            if "evalsets" in result:
                formatted_result = []
                for evalset_id, evalset_data in result["evalsets"].items():
                    formatted_result.append({
                        "id": evalset_id,
                        "name": evalset_data["name"],
                        "description": evalset_data["short_description"],
                        "questions": len(evalset_data["questions"])
                    })
                
                # Add a pretty header if not in quiet mode and using table format
                quiet_mode = os.environ.get("AGENTOPTIM_QUIET", "0") == "1"
                if not quiet_mode and args.format == "table":
                    try:
                        # Try to use rich for a prettier header
                        from rich.console import Console
                        from rich.panel import Panel
                        console = Console()
                        
                        header = f"[bold cyan]Evaluation Sets[/bold cyan]\n[dim]{len(formatted_result)} sets available[/dim]"
                        console.print(Panel(header, expand=False))
                    except ImportError:
                        # Fall back to simple ANSI colors
                        print(f"{Fore.CYAN}===== Evaluation Sets ====={Style.RESET_ALL}")
                        print(f"{len(formatted_result)} sets available\n")
                
                # Sort the evalsets by name for better readability
                formatted_result.sort(key=lambda x: x["name"].lower())
                
                handle_output(formatted_result, args.format, args.output)
                
                # Add a helpful footer for table format
                if not quiet_mode and args.format == "table" and not args.output:
                    print(f"\n{Fore.CYAN}Tip:{Style.RESET_ALL} Use {Fore.GREEN}agentoptim evalset get <id>{Style.RESET_ALL} to view details of a specific evaluation set")
                    print(f"{Fore.CYAN}Tip:{Style.RESET_ALL} Use {Fore.GREEN}agentoptim run create <evalset-id> --interactive{Style.RESET_ALL} to run an evaluation")
            else:
                handle_output(result, args.format, args.output)
        
        elif args.action == "get":
            result = manage_evalset(action="get", evalset_id=args.evalset_id)
            
            # Enhanced output for text format
            if args.format == "text" and "evalset" in result:
                evalset = result["evalset"]
                
                try:
                    # Try to use rich for a prettier output
                    from rich.console import Console
                    from rich.panel import Panel
                    from rich.markdown import Markdown
                    from rich.table import Table
                    
                    console = Console(record=True, width=MAX_WIDTH)
                    
                    # Header with evalset info
                    title = f"[bold cyan]{evalset['name']}[/bold cyan]"
                    metadata = f"[dim]ID:[/dim] [yellow]{evalset['id']}[/yellow]"
                    console.print(Panel(f"{title}\n{metadata}", expand=False))
                    
                    # Description section
                    if evalset.get("short_description"):
                        console.print(f"\n[bold cyan]Description[/bold cyan]")
                        console.print(f"{evalset['short_description']}")
                    
                    # Detailed description if available
                    if evalset.get("long_description"):
                        console.print(f"\n[bold cyan]Detailed Description[/bold cyan]")
                        console.print(Panel(evalset['long_description'], border_style="blue"))
                    
                    # Questions section
                    console.print(f"\n[bold cyan]Evaluation Questions[/bold cyan] [dim]({len(evalset['questions'])} questions)[/dim]")
                    
                    # Create a table for questions
                    question_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 1, 0, 1))
                    question_table.add_column("#", style="dim", justify="right")
                    question_table.add_column("Question", style="green")
                    
                    # Add each question
                    for i, question in enumerate(evalset['questions'], 1):
                        question_table.add_row(str(i), question)
                    
                    console.print(question_table)
                    
                    # Add a helpful call to action
                    console.print(f"\n[bold cyan]Usage Example[/bold cyan]")
                    console.print(f"To evaluate a conversation with this evalset, run:")
                    console.print(f"[green]agentoptim run create {evalset['id']} --interactive[/green]")
                    
                    # Update the output
                    result["formatted_message"] = console.export_text()
                    
                except ImportError:
                    # If rich is not available, create a simple formatted text output
                    lines = []
                    lines.append(f"{Fore.CYAN}===== {evalset['name']} ====={Style.RESET_ALL}")
                    lines.append(f"ID: {evalset['id']}")
                    lines.append("")
                    
                    if evalset.get("short_description"):
                        lines.append(f"{Fore.CYAN}Description:{Style.RESET_ALL}")
                        lines.append(evalset["short_description"])
                        lines.append("")
                    
                    if evalset.get("long_description"):
                        lines.append(f"{Fore.CYAN}Detailed Description:{Style.RESET_ALL}")
                        lines.append(evalset["long_description"])
                        lines.append("")
                    
                    lines.append(f"{Fore.CYAN}Evaluation Questions ({len(evalset['questions'])} questions):{Style.RESET_ALL}")
                    for i, question in enumerate(evalset['questions'], 1):
                        lines.append(f"{i}. {question}")
                    
                    lines.append("")
                    lines.append(f"{Fore.CYAN}Usage Example:{Style.RESET_ALL}")
                    lines.append(f"To evaluate a conversation with this evalset, run:")
                    lines.append(f"{Fore.GREEN}agentoptim run create {evalset['id']} --interactive{Style.RESET_ALL}")
                    
                    result["formatted_message"] = "\n".join(lines)
            
            handle_output(result, args.format, args.output)
        
        elif args.action == "create":
            # Use interactive wizard if requested
            if args.wizard:
                wizard_results = run_interactive_wizard(args)
                questions = wizard_results["questions"]
                name = wizard_results["name"]
                short_desc = wizard_results["short_description"] 
                long_desc = wizard_results["long_description"]
            else:
                # Verify required params for non-interactive mode
                if not args.name or not args.questions or not args.short_desc or not args.long_desc:
                    print(f"{Fore.RED}Error: Non-interactive creation requires --name, --questions, --short-desc, and --long-desc{Style.RESET_ALL}")
                    print(f"Tip: Use {Fore.GREEN}--wizard{Style.RESET_ALL} for interactive creation")
                    sys.exit(1)
                
                # Parse questions from file or string
                questions = parse_questions(args.questions)
                name = args.name
                short_desc = args.short_desc
                long_desc = args.long_desc
            
            # Create the evalset
            result = manage_evalset(
                action="create",
                name=name,
                questions=questions,
                short_description=short_desc,
                long_description=long_desc
            )
            handle_output(result, args.format, args.output)
        
        elif args.action == "update":
            update_args = {"action": "update", "evalset_id": args.evalset_id}
            if args.name:
                update_args["name"] = args.name
            if args.questions:
                update_args["questions"] = parse_questions(args.questions)
            if args.short_desc:
                update_args["short_description"] = args.short_desc
            if args.long_desc:
                update_args["long_description"] = args.long_desc
                
            result = manage_evalset(**update_args)
            handle_output(result, args.format, args.output)
        
        elif args.action == "delete":
            result = manage_evalset(action="delete", evalset_id=args.evalset_id)
            handle_output(result, args.format, args.output)
    
    elif args.resource in ["run", "r"]:
        # Import necessary modules 
        import asyncio
        from agentoptim.evalrun import manage_eval_runs, EvalRun, save_eval_run
        from agentoptim.runner import run_evalset
        
        if not args.action:
            print(f"{Fore.RED}Error: No action specified for run resource{Style.RESET_ALL}")
            print("Available actions: list, get, create")
            sys.exit(1)
        
        # Handle run actions
        if args.action == "list":
            # Import constants here to avoid circular imports
            from agentoptim.evalrun import list_eval_runs
            
            # Get paginated list of evaluation runs
            eval_runs, total_count = list_eval_runs(
                page=args.page,
                page_size=args.page_size,
                evalset_id=args.evalset_id
            )
            
            # Calculate pagination metadata
            total_pages = (total_count + args.page_size - 1) // args.page_size
            has_next = args.page < total_pages
            has_prev = args.page > 1
            
            # Format the eval runs for table display with prettier output
            formatted_eval_runs = []
            for run in eval_runs:
                # Format time nicely
                timestamp = run.get("timestamp_formatted", "Unknown")
                
                # Format score nicely
                score = run.get("summary", {}).get("yes_percentage", 0)
                score_formatted = f"{score:.1f}%" if isinstance(score, (int, float)) else "N/A"
                
                # Format model nicely
                model = run.get("judge_model") or "default"
                if len(str(model)) > 15:
                    model = str(model)[:12] + "..."
                
                # Create a nice entry for the table with standardized fields - keep full ID
                formatted_run = {
                    "id": run.get("id", "Unknown"),
                    "date": timestamp,
                    "evalset": run.get("evalset_name", "Unknown"),
                    "model": model,
                    "score": score,
                    "questions": run.get("result_count", 0),
                    "conv_len": run.get("conversation_length", 0),
                }
                formatted_eval_runs.append(formatted_run)
            
            # If using table format, just output the formatted runs directly
            if args.format == "table":
                # Add a pretty header if not in quiet mode
                quiet_mode = os.environ.get("AGENTOPTIM_QUIET", "0") == "1"
                if not quiet_mode:
                    try:
                        # Try to use rich for a prettier header
                        from rich.console import Console
                        from rich.panel import Panel
                        console = Console()
                        
                        # Create pagination indicator
                        pagination_text = f"Page {args.page} of {total_pages} • {total_count} total runs"
                        if has_prev:
                            pagination_text = f"← Previous • {pagination_text}"
                        if has_next:
                            pagination_text = f"{pagination_text} • Next →"
                            
                        header = f"[bold cyan]Evaluation Runs[/bold cyan]\n[dim]{pagination_text}[/dim]"
                        console.print(Panel(header, expand=False))
                    except ImportError:
                        # Fall back to simple ANSI colors
                        print(f"{Fore.CYAN}===== Evaluation Runs ====={Style.RESET_ALL}")
                        pagination_text = f"Page {args.page} of {total_pages} • {total_count} total runs"
                        print(f"{pagination_text}\n")
                
                # Output the formatted table
                handle_output(formatted_eval_runs, args.format, args.output)
                
                # Add pagination footer if not in quiet mode and not outputting to a file
                if not quiet_mode and not args.output:
                    print(f"\n{Fore.CYAN}Page {args.page} of {total_pages}{Style.RESET_ALL}")
                    if has_prev:
                        print(f"{Fore.BLUE}Use 'agentoptim run list --page {args.page-1}' for previous page{Style.RESET_ALL}")
                    if has_next:
                        print(f"{Fore.GREEN}Use 'agentoptim run list --page {args.page+1}' for next page{Style.RESET_ALL}")
            else:
                # For other formats, create a structured response with all details
                response = {
                    "status": "success",
                    "eval_runs": eval_runs,
                    "pagination": {
                        "page": args.page,
                        "page_size": args.page_size,
                        "total_count": total_count,
                        "total_pages": total_pages,
                        "has_next": has_next,
                        "has_prev": has_prev,
                        "next_page": args.page + 1 if has_next else None,
                        "prev_page": args.page - 1 if has_prev else None
                    }
                }
                
                handle_output(response, args.format, args.output)
        
        elif args.action in ["get", "export"]:
            # Import run retrieval functions
            from agentoptim.evalrun import get_eval_run, get_formatted_eval_run, list_eval_runs
            
            # Check if user wants the latest run
            if args.eval_run_id.lower() == 'latest':
                # Get the most recent run
                recent_runs, _ = list_eval_runs(page=1, page_size=1)
                if not recent_runs:
                    print(f"{Fore.RED}Error: No evaluation runs found in the system{Style.RESET_ALL}", file=sys.stderr)
                    sys.exit(1)
                
                # Use the most recent run's ID
                latest_run_id = recent_runs[0]["id"]
                print(f"{Fore.CYAN}Using latest run: {latest_run_id}{Style.RESET_ALL}")
                eval_run = get_eval_run(latest_run_id)
            else:
                # Get the specified run
                eval_run = get_eval_run(args.eval_run_id)
            
            if eval_run is None:
                print(f"{Fore.RED}Error: Evaluation run with ID '{args.eval_run_id}' not found{Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)
                
            # Basic get command
            if args.action == "get":
                # Format the evaluation run for output
                formatted_run = get_formatted_eval_run(eval_run)
                
                # If using text format, enhance the display
                if args.format == "text":
                    try:
                        # Try to use rich for a prettier output
                        from rich.console import Console
                        from rich.panel import Panel
                        from rich.markdown import Markdown
                        from rich.table import Table
                        from rich.progress import BarColumn, TextColumn
                        from rich.layout import Layout
                        from rich.text import Text
                        from rich.console import Group
                        
                        console = Console(record=True, width=MAX_WIDTH)
                        
                        # Create a layout
                        layout = Layout()
                        layout.split(
                            Layout(name="header", size=4),
                            Layout(name="main")
                        )
                        
                        # Header with run info
                        title = f"[bold cyan]{eval_run.evalset_name}[/bold cyan]"
                        metadata = (
                            f"[dim]Run ID:[/dim] [yellow]{eval_run.id}[/yellow]  "
                            f"[dim]Date:[/dim] [blue]{eval_run.timestamp}[/blue]  "
                            f"[dim]Model:[/dim] [green]{eval_run.judge_model or 'default'}[/green]"
                        )
                        console.print(Panel(f"{title}\n{metadata}", expand=False))
                        
                        # Summary section
                        console.print("\n[bold cyan]Summary[/bold cyan]")
                        
                        # Create a table for summary stats
                        summary_table = Table(show_header=False, box=None, padding=(0, 1, 0, 1))
                        summary_table.add_column("Key", style="dim")
                        summary_table.add_column("Value")
                        
                        # Calculate summary metrics
                        total_questions = len(eval_run.results)
                        yes_count = sum(1 for r in eval_run.results if r.get("judgment") is True)
                        no_count = sum(1 for r in eval_run.results if r.get("judgment") is False)
                        error_count = sum(1 for r in eval_run.results if r.get("error") is not None)
                        
                        # Calculate yes percentage
                        if total_questions - error_count > 0:
                            yes_percentage = (yes_count / (total_questions - error_count)) * 100
                        else:
                            yes_percentage = 0
                            
                        # Add summary rows
                        summary_table.add_row("Total questions", f"[blue]{total_questions}[/blue]")
                        summary_table.add_row("Success rate", f"[cyan]{total_questions - error_count}[/cyan]/[blue]{total_questions}[/blue] questions evaluated successfully")
                        summary_table.add_row("Score", f"[magenta]{yes_percentage:.1f}%[/magenta]")
                        
                        # Create a simple text-based progress bar for the score
                        bar_width = 40
                        filled_width = int(bar_width * yes_percentage / 100)
                        bar_text = f"[magenta]{'█' * filled_width}{'░' * (bar_width - filled_width)}[/magenta] {yes_percentage:.1f}%"
                        summary_table.add_row("", bar_text)
                        
                        # More detailed stats
                        summary_table.add_row("Yes responses", f"[green]{yes_count}[/green] ({yes_percentage:.1f}%)")
                        summary_table.add_row("No responses", f"[red]{no_count}[/red] ({100-yes_percentage:.1f}%)")
                        
                        # Add confidence information if available
                        if eval_run.summary and "mean_confidence" in eval_run.summary:
                            summary_table.add_row("Mean confidence", f"{eval_run.summary['mean_confidence']:.2f}")
                            if "mean_yes_confidence" in eval_run.summary and yes_count > 0:
                                summary_table.add_row("Mean confidence in Yes", f"{eval_run.summary['mean_yes_confidence']:.2f}")
                            if "mean_no_confidence" in eval_run.summary and no_count > 0:
                                summary_table.add_row("Mean confidence in No", f"{eval_run.summary['mean_no_confidence']:.2f}")
                                
                        console.print(summary_table)
                        
                        # Results section
                        console.print("\n[bold cyan]Detailed Results[/bold cyan]")
                        
                        # Print each question with its judgment
                        for i, result in enumerate(eval_run.results, 1):
                            # Create a panel for each result
                            if result.get("error"):
                                panel_title = f"Q{i}: {result.get('question')}"
                                panel_content = f"[red]Error:[/red] {result.get('error')}"
                                panel_style = "red"
                            else:
                                judgment = result.get("judgment")
                                judgment_text = "[green]Yes[/green]" if judgment else "[red]No[/red]"
                                confidence = result.get("confidence")
                                confidence_text = f" (confidence: {confidence:.2f})" if confidence is not None else ""
                                
                                panel_title = f"Q{i}: {result.get('question')}"
                                panel_content = f"[bold]Judgment:[/bold] {judgment_text}{confidence_text}"
                                
                                # Add reasoning if available
                                if result.get("reasoning"):
                                    # Truncate reasoning if too long
                                    reasoning = result.get("reasoning")
                                    if len(reasoning) > 200:
                                        reasoning = reasoning[:197] + "..."
                                    panel_content += f"\n\n[bold]Reasoning:[/bold] {reasoning}"
                                
                                panel_style = "green" if judgment else "red"
                            
                            console.print(Panel(panel_content, title=panel_title, title_align="left", border_style=panel_style))
                        
                        # Conversation section
                        console.print("\n[bold cyan]Conversation[/bold cyan]")
                        
                        # Format the conversation
                        conversation_text = ""
                        for msg in eval_run.conversation:
                            role = msg.get("role", "").upper()
                            content = msg.get("content", "")
                            
                            # Color based on role
                            if role == "SYSTEM":
                                role_formatted = f"[yellow]{role}:[/yellow]"
                            elif role == "USER":
                                role_formatted = f"[blue]{role}:[/blue]"
                            elif role == "ASSISTANT":
                                role_formatted = f"[green]{role}:[/green]"
                            else:
                                role_formatted = f"[dim]{role}:[/dim]"
                                
                            # Truncate content if too long
                            if len(content) > 200:
                                content = content[:197] + "..."
                                
                            conversation_text += f"{role_formatted} {content}\n\n"
                        
                        console.print(Panel(conversation_text, border_style="cyan"))
                        
                        # Export the console output
                        formatted_run["formatted_message"] = console.export_text()
                        
                    except ImportError:
                        # If rich is not available, keep original formatting
                        pass
                    
                handle_output(formatted_run, args.format, args.output)
            
            # Compare two evaluation runs side by side
            elif args.action == "compare":
                # Import run retrieval functions
                from agentoptim.evalrun import list_eval_runs
                
                # Resolve latest or latest-N identifiers
                def resolve_run_id(run_id):
                    if run_id.lower() == 'latest':
                        recent_runs, _ = list_eval_runs(page=1, page_size=1)
                        if not recent_runs:
                            print(f"{Fore.RED}Error: No evaluation runs found in the system{Style.RESET_ALL}", file=sys.stderr)
                            sys.exit(1)
                        latest_id = recent_runs[0]["id"]
                        print(f"{Fore.CYAN}Using latest run: {latest_id}{Style.RESET_ALL}")
                        return latest_id
                    
                    if run_id.lower().startswith('latest-'):
                        try:
                            offset = int(run_id.lower().split('-')[1])
                            recent_runs, _ = list_eval_runs(page=1, page_size=offset + 1)
                            if not recent_runs or len(recent_runs) <= offset:
                                print(f"{Fore.RED}Error: Not enough evaluation runs found in the system for offset {offset}{Style.RESET_ALL}", file=sys.stderr)
                                sys.exit(1)
                            offset_id = recent_runs[offset]["id"]
                            print(f"{Fore.CYAN}Using run with offset {offset}: {offset_id}{Style.RESET_ALL}")
                            return offset_id
                        except (ValueError, IndexError):
                            print(f"{Fore.RED}Error: Invalid offset format in '{run_id}'. Use 'latest-N' where N is a number.{Style.RESET_ALL}", file=sys.stderr)
                            sys.exit(1)
                    
                    return run_id
                
                # Get actual run IDs
                first_id = resolve_run_id(args.first_run_id)
                second_id = resolve_run_id(args.second_run_id)
                
                # Get both eval runs
                first_run = get_eval_run(first_id)
                second_run = get_eval_run(second_id)
                
                if first_run is None:
                    print(f"{Fore.RED}Error: Evaluation run with ID '{first_id}' not found{Style.RESET_ALL}", file=sys.stderr)
                    sys.exit(1)
                
                if second_run is None:
                    print(f"{Fore.RED}Error: Evaluation run with ID '{second_id}' not found{Style.RESET_ALL}", file=sys.stderr)
                    sys.exit(1)
                
                # Generate comparison output based on format
                if args.format == "text":
                    comparison = generate_text_comparison(first_run, second_run, use_color=args.color, detailed=args.detailed)
                    
                    # Print to output file or stdout
                    if args.output:
                        with open(args.output, "w", encoding="utf-8") as f:
                            f.write(comparison)
                        print(f"{Fore.GREEN}✓ Comparison saved to {args.output}{Style.RESET_ALL}")
                    else:
                        print(comparison)
                        
                elif args.format == "html":
                    # Ensure output file is specified for HTML
                    if not args.output:
                        print(f"{Fore.RED}Error: --output is required for HTML format{Style.RESET_ALL}", file=sys.stderr)
                        sys.exit(1)
                        
                    # Generate HTML comparison
                    html_comparison = generate_html_comparison(first_run, second_run, use_color=args.color, detailed=args.detailed)
                    
                    # Write to file
                    with open(args.output, "w", encoding="utf-8") as f:
                        f.write(html_comparison)
                    print(f"{Fore.GREEN}✓ HTML comparison saved to {args.output}{Style.RESET_ALL}")
                    
                    # Try to open in browser
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(["open", args.output], check=False)
                        elif sys.platform == "win32":  # Windows
                            os.startfile(args.output)
                        elif sys.platform.startswith("linux"):  # Linux
                            subprocess.run(["xdg-open", args.output], check=False)
                    except Exception as e:
                        logger.warning(f"Failed to open comparison in browser: {str(e)}")
            
            # Enhanced export command with more format options
            elif args.action == "export":
                # Validate output file is specified
                if not args.output:
                    print(f"{Fore.RED}Error: --output is required for export{Style.RESET_ALL}", file=sys.stderr)
                    sys.exit(1)
                    
                # Get title from args or default
                title = args.title or f"Evaluation Results: {eval_run.evalset_name}"
                
                # Prepare export based on requested format
                if args.format == "markdown":
                    export_markdown(eval_run, args.output, title, args.charts)
                    
                elif args.format == "csv":
                    export_csv(eval_run, args.output)
                    
                elif args.format == "html":
                    export_html(eval_run, args.output, title, args.template, args.color, args.charts)
                    
                elif args.format == "pdf":
                    try:
                        # Check if we have the required dependencies
                        import weasyprint
                        # First generate HTML, then convert to PDF
                        html_content = generate_html_report(eval_run, title, args.template, args.color, args.charts)
                        weasyprint.HTML(string=html_content).write_pdf(args.output)
                        print(f"{Fore.GREEN}✓ PDF report exported to {args.output}{Style.RESET_ALL}")
                    except ImportError:
                        print(f"{Fore.RED}Error: PDF export requires the weasyprint package.{Style.RESET_ALL}", file=sys.stderr)
                        print(f"{Fore.YELLOW}Install with: pip install weasyprint{Style.RESET_ALL}")
                        sys.exit(1)
                        
                elif args.format == "json":
                    # Full JSON export
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(eval_run.__dict__, f, indent=2, default=str)
                    print(f"{Fore.GREEN}✓ JSON data exported to {args.output}{Style.RESET_ALL}")
                
                # Open the file automatically if on a desktop system
                try:
                    if sys.platform == "darwin":  # macOS
                        subprocess.run(["open", args.output], check=False)
                    elif sys.platform == "win32":  # Windows
                        os.startfile(args.output)
                    elif sys.platform.startswith("linux"):  # Linux
                        subprocess.run(["xdg-open", args.output], check=False)
                except Exception as e:
                    # Don't fail if we can't open the file, just log it
                    logger.warning(f"Failed to open exported file: {str(e)}")
        
        elif args.action == "create":
            # Check if evalset_id is provided
            if not args.evalset_id:
                print(f"{Fore.RED}Error: No evaluation set ID provided. Get an ID with 'agentoptim evalset list'{Style.RESET_ALL}")
                print(f"{Fore.RED}Usage: agentoptim run create <evalset-id> conversation.json{Style.RESET_ALL}")
                sys.exit(1)
                
            # Interactive conversation creation mode
            if args.interactive:
                try:
                    # Try to enhance the experience with rich if available
                    try:
                        from rich.console import Console
                        from rich.panel import Panel
                        from rich.markdown import Markdown
                        from rich.prompt import Prompt, Confirm
                        has_rich = True
                        console = Console()
                    except ImportError:
                        has_rich = False
                        
                    # Show welcome message
                    if has_rich:
                        console.print(Panel(
                            "[bold cyan]Interactive Conversation Creator[/bold cyan]\n"
                            "Create a conversation to evaluate by adding messages turn by turn.\n"
                            "Press Ctrl+D when finished to start evaluation.",
                            title="✨ AgentOptim", 
                            border_style="cyan"
                        ))
                    else:
                        print(f"{Fore.CYAN}=== Interactive Conversation Creator ==={Style.RESET_ALL}")
                        print("Create a conversation to evaluate by adding messages turn by turn.")
                        print("Press Ctrl+D when finished to start evaluation.\n")
                    
                    # Create conversation with system message option
                    conversation = []
                    
                    # Ask for system message first
                    if has_rich:
                        if Confirm.ask("Include a system message?", default=True):
                            system_content = console.input("[bold green]System message: [/bold green]")
                            conversation.append({"role": "system", "content": system_content})
                    else:
                        include_system = input(f"{Fore.GREEN}Include a system message? (Y/n): {Style.RESET_ALL}").lower()
                        if include_system in ["", "y", "yes"]:
                            system_content = input(f"{Fore.GREEN}System message: {Style.RESET_ALL}")
                            conversation.append({"role": "system", "content": system_content})
                    
                    # Start with user message
                    current_role = "user"
                    
                    # Collect messages until user signals they're done
                    try:
                        while True:
                            # Prompt based on current role
                            prompt_text = f"{current_role.capitalize()}: "
                            
                            if has_rich:
                                role_color = "blue" if current_role == "user" else "green"
                                content = console.input(f"[bold {role_color}]{prompt_text}[/bold {role_color}]")
                            else:
                                role_color = Fore.BLUE if current_role == "user" else Fore.GREEN
                                content = input(f"{role_color}{prompt_text}{Style.RESET_ALL}")
                            
                            # Add message to conversation
                            conversation.append({"role": current_role, "content": content})
                            
                            # Toggle role for next message
                            current_role = "assistant" if current_role == "user" else "user"
                    except EOFError:
                        # End of input
                        print("\n")
                    except KeyboardInterrupt:
                        # User cancelled
                        print("\n\nCancelled by user.")
                        sys.exit(0)
                    
                    # Show preview of conversation
                    if has_rich:
                        console.print("\n[bold cyan]Conversation Preview:[/bold cyan]")
                        for msg in conversation:
                            role_color = {
                                "system": "yellow",
                                "user": "blue",
                                "assistant": "green"
                            }.get(msg["role"], "white")
                            
                            console.print(f"[bold {role_color}]{msg['role'].upper()}:[/bold {role_color}] {msg['content']}")
                            
                        # Confirm evaluation
                        if not Confirm.ask("\nEvaluate this conversation?", default=True):
                            console.print("[yellow]Evaluation cancelled.[/yellow]")
                            sys.exit(0)
                    else:
                        print(f"\n{Fore.CYAN}Conversation Preview:{Style.RESET_ALL}")
                        for msg in conversation:
                            role_color = {
                                "system": Fore.YELLOW,
                                "user": Fore.BLUE,
                                "assistant": Fore.GREEN
                            }.get(msg["role"], Fore.WHITE)
                            
                            print(f"{role_color}{msg['role'].upper()}:{Style.RESET_ALL} {msg['content']}")
                        
                        # Confirm evaluation
                        confirm = input(f"\n{Fore.CYAN}Evaluate this conversation? (Y/n): {Style.RESET_ALL}").lower()
                        if confirm not in ["", "y", "yes"]:
                            print(f"{Fore.YELLOW}Evaluation cancelled.{Style.RESET_ALL}")
                            sys.exit(0)
                            
                except Exception as e:
                    print(f"{Fore.RED}Error in interactive mode: {str(e)}{Style.RESET_ALL}")
                    sys.exit(1)
            # File-based conversation
            elif args.conversation or args.text:
                # Load the conversation from file
                conversation = load_conversation(args.conversation, args.text)
            else:
                print(f"{Fore.RED}Error: No conversation provided. Use a file, --text, or --interactive mode.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Examples:{Style.RESET_ALL}")
                print(f"  agentoptim run create <evalset-id> conversation.json")
                print(f"  agentoptim run create <evalset-id> --text response.txt")
                print(f"  agentoptim run create <evalset-id> --interactive")
                sys.exit(1)
            
            # Import constants here to avoid circular imports
            from agentoptim.constants import (
                DEFAULT_LOCAL_API_BASE,
                DEFAULT_OPENAI_API_BASE,
                DEFAULT_ANTHROPIC_API_BASE,
                DEFAULT_LOCAL_MODEL,
                DEFAULT_OPENAI_MODEL,
                DEFAULT_ANTHROPIC_MODEL
            )
            
            # Configure provider settings
            if args.provider:
                # Only set API_BASE if not already set by user
                if "AGENTOPTIM_API_BASE" not in os.environ:
                    if args.provider == "openai":
                        os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_OPENAI_API_BASE
                    elif args.provider == "anthropic":
                        os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_ANTHROPIC_API_BASE
                    elif args.provider == "local":
                        os.environ["AGENTOPTIM_API_BASE"] = DEFAULT_LOCAL_API_BASE
                
                # Set default model based on provider if not explicitly specified
                if not args.model:
                    if args.provider == "openai":
                        args.model = DEFAULT_OPENAI_MODEL
                    elif args.provider == "anthropic":
                        args.model = DEFAULT_ANTHROPIC_MODEL
                    elif args.provider == "local":
                        args.model = DEFAULT_LOCAL_MODEL
            
            # Set environment variables for judge model and omit_reasoning if specified
            if args.model:
                os.environ["AGENTOPTIM_JUDGE_MODEL"] = args.model
            if args.omit_reasoning:
                os.environ["AGENTOPTIM_OMIT_REASONING"] = "1"
                
            # Run the evaluation
            try:
                # Try to import rich for progress display
                try:
                    from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn
                    from rich.console import Console
                    from rich.panel import Panel
                    has_rich = True
                except ImportError:
                    has_rich = False
                    print(f"{Fore.YELLOW}Tip: Install 'rich' for better progress visualization: pip install rich{Style.RESET_ALL}")
                
                if has_rich:
                    console = Console()
                    with console.status(f"[bold green]Initializing evaluation...", spinner="dots"):
                        # Define a callback to update progress
                        question_count = 0
                        
                        # Get the eval set first to know how many questions
                        from agentoptim.evalset import manage_evalset
                        evalset_info = manage_evalset(action="get", evalset_id=args.evalset_id)
                        if "evalset" in evalset_info and "questions" in evalset_info["evalset"]:
                            question_count = len(evalset_info["evalset"]["questions"])
                            
                    # Now run with progress tracking
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(complete_style="green", finished_style="green"),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeElapsedColumn(),
                    ) as progress:
                        task = progress.add_task(f"[green]Evaluating with {args.model or os.environ.get('AGENTOPTIM_JUDGE_MODEL', 'default model')}...", total=question_count)
                        
                        # Define a progress callback 
                        def progress_callback(completed, total):
                            progress.update(task, completed=completed)
                            
                        # Run the evaluation with progress tracking
                        run_result = asyncio.run(run_evalset(
                            evalset_id=args.evalset_id,
                            conversation=conversation,
                            judge_model=args.model,
                            max_parallel=args.max_parallel,
                            omit_reasoning=args.omit_reasoning,
                            progress_callback=progress_callback
                        ))
                else:
                    # Simple spinner for when rich is not available
                    print(f"{Fore.CYAN}Evaluating conversation...{Style.RESET_ALL}")
                    animation = "|/-\\"
                    idx = 0
                    start_time = time.time()
                    
                    def print_spinner():
                        nonlocal idx
                        elapsed = time.time() - start_time
                        mins, secs = divmod(int(elapsed), 60)
                        sys.stdout.write(f"\r{Fore.CYAN}Working {animation[idx % len(animation)]} {mins:02d}:{secs:02d} elapsed{Style.RESET_ALL}")
                        sys.stdout.flush()
                        idx += 1
                    
                    # Run evaluation with a simple spinning cursor
                    spinner_timer = None
                    try:
                        import threading
                        spinner_timer = threading.Timer(0.1, print_spinner)
                        spinner_timer.start()
                        
                        # Call the actual evaluation
                        run_result = asyncio.run(run_evalset(
                            evalset_id=args.evalset_id,
                            conversation=conversation,
                            judge_model=args.model,
                            max_parallel=args.max_parallel,
                            omit_reasoning=args.omit_reasoning
                        ))
                        
                        # Clear the spinner line
                        sys.stdout.write("\r" + " " * 50 + "\r")
                        sys.stdout.flush()
                    finally:
                        if spinner_timer:
                            spinner_timer.cancel()
                
                # Check for errors
                if "error" in run_result:
                    handle_output(run_result, args.format, args.output)
                    return
                
                # Create and save EvalRun
                eval_run = EvalRun(
                    evalset_id=run_result.get("evalset_id"),
                    evalset_name=run_result.get("evalset_name"),
                    judge_model=run_result.get("judge_model"),
                    results=run_result.get("results", []),
                    conversation=conversation,
                    summary=run_result.get("summary", {})
                )
                
                # Save to disk
                save_success = save_eval_run(eval_run)
                if not save_success:
                    print(f"{Fore.YELLOW}Warning: Failed to save evaluation run.{Style.RESET_ALL}", file=sys.stderr)
                
                # Add the run ID to the result for future reference
                run_result["id"] = eval_run.id
                
                # Format the output
                if "formatted_message" in run_result:
                    run_result["result"] = run_result.pop("formatted_message")
                
                # Add a message about the saved run ID
                if "result" in run_result and isinstance(run_result["result"], str):
                    run_result["result"] += f"\n\nEvaluation saved with ID: {eval_run.id}"
                
                handle_output(run_result, args.format, args.output)
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)
    
    elif args.resource == "dev":
        if not args.action:
            print(f"{Fore.RED}Error: No action specified for dev resource{Style.RESET_ALL}")
            print("Available actions: cache, logs")
            sys.exit(1)
        
        # Handle dev actions
        if args.action == "cache":
            # Use the server.py get_cache_stats function
            from agentoptim.server import get_cache_stats
            stats = get_cache_stats()
            handle_output(stats, args.format, args.output)
        
        elif args.action == "logs":
            # Simple log viewing functionality
            try:
                if args.follow:
                    import subprocess
                    # Use tail command to follow logs
                    subprocess.run(["tail", "-f", "-n", str(args.lines), log_file_path])
                else:
                    # Just read the last N lines
                    import subprocess
                    subprocess.run(["tail", "-n", str(args.lines), log_file_path])
            except Exception as e:
                print(f"{Fore.RED}Error reading logs: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
                with open(log_file_path, "r") as f:
                    # Fallback to Python implementation if tail fails
                    lines = f.readlines()
                    for line in lines[-args.lines:]:
                        print(line, end="")


# Export helper functions
def export_markdown(eval_run, output_file, title, include_charts=False):
    """Export evaluation run to a Markdown file."""
    # Create markdown content
    markdown_lines = []
    markdown_lines.append(f"# {title}")
    markdown_lines.append("")
    markdown_lines.append(f"**Evaluation Set**: {eval_run.evalset_name}  ")
    markdown_lines.append(f"**ID**: {eval_run.id}  ")
    markdown_lines.append(f"**Date**: {eval_run.timestamp}  ")
    markdown_lines.append(f"**Model**: {eval_run.judge_model}  ")
    markdown_lines.append("")
    
    # Summary section
    markdown_lines.append("## Summary")
    markdown_lines.append("")
    
    # Calculate summary statistics
    total_questions = len(eval_run.results)
    yes_count = sum(1 for r in eval_run.results if r.get("judgment") is True)
    no_count = sum(1 for r in eval_run.results if r.get("judgment") is False)
    error_count = sum(1 for r in eval_run.results if r.get("error") is not None)
    
    # Calculate yes percentage
    if total_questions - error_count > 0:
        yes_percentage = (yes_count / (total_questions - error_count)) * 100
    else:
        yes_percentage = 0
    
    # Add summary statistics
    markdown_lines.append(f"- **Total questions**: {total_questions}")
    markdown_lines.append(f"- **Success rate**: {total_questions - error_count}/{total_questions} questions evaluated successfully")
    markdown_lines.append(f"- **Score**: {round(yes_percentage, 1)}%")
    markdown_lines.append(f"- **Yes responses**: {yes_count} ({round(yes_percentage, 1)}%)")
    markdown_lines.append(f"- **No responses**: {no_count} ({round(100 - yes_percentage, 1)}%)")
    
    # Add confidence information
    if eval_run.summary and "mean_confidence" in eval_run.summary:
        markdown_lines.append(f"- **Mean confidence**: {eval_run.summary['mean_confidence']:.2f}")
        if "mean_yes_confidence" in eval_run.summary and yes_count > 0:
            markdown_lines.append(f"  - **Yes confidence**: {eval_run.summary['mean_yes_confidence']:.2f}")
        if "mean_no_confidence" in eval_run.summary and no_count > 0:
            markdown_lines.append(f"  - **No confidence**: {eval_run.summary['mean_no_confidence']:.2f}")
    
    # Add ASCII chart if requested
    if include_charts:
        markdown_lines.append("")
        markdown_lines.append("### Score Chart")
        markdown_lines.append("```")
        # Create bar chart for yes percentage
        yes_bar_length = min(40, int(yes_percentage / 2.5))  # Scale to max 40 chars for markdown
        no_bar_length = 40 - yes_bar_length
        yes_bar = "█" * yes_bar_length
        no_bar = "░" * no_bar_length
        markdown_lines.append(f"{yes_bar}{no_bar} {round(yes_percentage, 1)}%")
        markdown_lines.append("```")
    
    markdown_lines.append("")
    
    # Detailed results section
    markdown_lines.append("## Detailed Results")
    markdown_lines.append("")
    
    # Add each question, judgment, and reasoning
    for i, result in enumerate(eval_run.results, 1):
        if result.get("error"):
            markdown_lines.append(f"{i}. **Q**: {result.get('question')}")
            markdown_lines.append(f"   **Error**: {result.get('error')}")
        else:
            judgment_text = "Yes" if result.get("judgment") else "No"
            markdown_lines.append(f"{i}. **Q**: {result.get('question')}")
            
            # Add confidence if available
            confidence_display = ""
            if result.get("confidence") is not None:
                confidence_display = f" (confidence: {result.get('confidence'):.2f})"
            
            markdown_lines.append(f"   **A**: {judgment_text}{confidence_display}")
            
            # Add reasoning if available
            if result.get("reasoning"):
                markdown_lines.append(f"   **Reasoning**: {result.get('reasoning')}")
        
        markdown_lines.append("")  # Add blank line between results
    
    # Add conversation section
    markdown_lines.append("## Conversation")
    markdown_lines.append("")
    markdown_lines.append("```")
    for msg in eval_run.conversation:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        markdown_lines.append(f"{role}: {content}")
        markdown_lines.append("")
    markdown_lines.append("```")
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    
    print(f"{Fore.GREEN}✓ Markdown report exported to {output_file}{Style.RESET_ALL}")

def export_csv(eval_run, output_file):
    """Export evaluation run results to a CSV file."""
    import csv
    
    # Prepare data for CSV
    header = ["Question", "Judgment", "Confidence", "Reasoning", "Error"]
    rows = []
    
    for result in eval_run.results:
        rows.append([
            result.get("question", ""),
            "Yes" if result.get("judgment") is True else "No" if result.get("judgment") is False else "N/A",
            f"{result.get('confidence', 'N/A')}" if result.get("confidence") is not None else "N/A",
            result.get("reasoning", ""),
            result.get("error", "")
        ])
    
    # Write to CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"{Fore.GREEN}✓ CSV data exported to {output_file}{Style.RESET_ALL}")

def generate_html_report(eval_run, title, template_file=None, use_color=True, include_charts=False):
    """Generate HTML report content."""
    # Try to import jinja2 for templating
    try:
        import jinja2
    except ImportError:
        print(f"{Fore.YELLOW}Warning: jinja2 not found, using basic HTML template.{Style.RESET_ALL}")
        return generate_basic_html_report(eval_run, title, use_color, include_charts)
    
    # Calculate summary statistics
    total_questions = len(eval_run.results)
    yes_count = sum(1 for r in eval_run.results if r.get("judgment") is True)
    no_count = sum(1 for r in eval_run.results if r.get("judgment") is False)
    error_count = sum(1 for r in eval_run.results if r.get("error") is not None)
    
    # Calculate yes percentage
    if total_questions - error_count > 0:
        yes_percentage = (yes_count / (total_questions - error_count)) * 100
    else:
        yes_percentage = 0
    
    # Load template
    template_content = ''
    if template_file and os.path.exists(template_file):
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
    else:
        # Default HTML template with modern styling
        template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        :root {
            --primary-color: {% if use_color %}#3498db{% else %}#444{% endif %};
            --secondary-color: {% if use_color %}#2ecc71{% else %}#777{% endif %};
            --error-color: {% if use_color %}#e74c3c{% else %}#999{% endif %};
            --text-color: #333;
            --light-bg: #f9f9f9;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: var(--primary-color);
        }
        
        .metadata {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .metadata p {
            margin: 5px 0;
        }
        
        .summary {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .progress-bar {
            height: 25px;
            background-color: #f0f0f0;
            border-radius: 5px;
            margin: 15px 0;
            overflow: hidden;
        }
        
        .progress-bar .filled {
            height: 100%;
            background-color: var(--secondary-color);
            width: {{ yes_percentage }}%;
            transition: width 0.5s ease-in-out;
        }
        
        .results {
            margin-top: 20px;
        }
        
        .question {
            border: 1px solid #eee;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        
        .question h3 {
            margin-top: 0;
        }
        
        .yes {
            border-left: 5px solid var(--secondary-color);
        }
        
        .no {
            border-left: 5px solid var(--error-color);
        }
        
        .error {
            border-left: 5px solid var(--error-color);
            background-color: #ffeeee;
        }
        
        .conversation {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        
        .system {
            color: {% if use_color %}#f39c12{% else %}#555{% endif %};
        }
        
        .user {
            color: {% if use_color %}#3498db{% else %}#333{% endif %};
        }
        
        .assistant {
            color: {% if use_color %}#2ecc71{% else %}#555{% endif %};
        }
        
        {% if include_charts %}
        canvas {
            max-width: 500px;
            margin: 20px auto;
            display: block;
        }
        {% endif %}
        
        footer {
            margin-top: 30px;
            text-align: center;
            font-size: 0.8em;
            color: #888;
        }
    </style>
    {% if include_charts %}
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    {% endif %}
</head>
<body>
    <h1>{{ title }}</h1>
    
    <div class="metadata">
        <p><strong>Evaluation Set:</strong> {{ eval_run.evalset_name }}</p>
        <p><strong>ID:</strong> {{ eval_run.id }}</p>
        <p><strong>Date:</strong> {{ eval_run.timestamp }}</p>
        <p><strong>Model:</strong> {{ eval_run.judge_model }}</p>
    </div>
    
    <h2>Summary</h2>
    <div class="summary">
        <p><strong>Total questions:</strong> {{ total_questions }}</p>
        <p><strong>Success rate:</strong> {{ total_questions - error_count }}/{{ total_questions }} questions evaluated successfully</p>
        <p><strong>Score:</strong> {{ "%.1f"|format(yes_percentage) }}%</p>
        
        <div class="progress-bar">
            <div class="filled"></div>
        </div>
        
        <p><strong>Yes responses:</strong> {{ yes_count }} ({{ "%.1f"|format(yes_percentage) }}%)</p>
        <p><strong>No responses:</strong> {{ no_count }} ({{ "%.1f"|format(100 - yes_percentage) }}%)</p>
        
        {% if eval_run.summary and eval_run.summary.mean_confidence is defined %}
        <p><strong>Mean confidence:</strong> {{ "%.2f"|format(eval_run.summary.mean_confidence) }}</p>
            {% if eval_run.summary.mean_yes_confidence is defined and yes_count > 0 %}
            <p><strong>Yes confidence:</strong> {{ "%.2f"|format(eval_run.summary.mean_yes_confidence) }}</p>
            {% endif %}
            {% if eval_run.summary.mean_no_confidence is defined and no_count > 0 %}
            <p><strong>No confidence:</strong> {{ "%.2f"|format(eval_run.summary.mean_no_confidence) }}</p>
            {% endif %}
        {% endif %}
    </div>
    
    {% if include_charts %}
    <div>
        <canvas id="resultsChart"></canvas>
    </div>
    {% endif %}
    
    <h2>Detailed Results</h2>
    <div class="results">
        {% for i, result in results %}
            <div class="question {% if result.error %}error{% elif result.judgment %}yes{% else %}no{% endif %}">
                <h3>{{ i }}. {{ result.question }}</h3>
                
                {% if result.error %}
                    <p><strong>Error:</strong> {{ result.error }}</p>
                {% else %}
                    <p><strong>Answer:</strong> 
                        {% if result.judgment %}Yes{% else %}No{% endif %}
                        {% if result.confidence is defined %}
                            (confidence: {{ "%.2f"|format(result.confidence) }})
                        {% endif %}
                    </p>
                    
                    {% if result.reasoning %}
                        <p><strong>Reasoning:</strong> {{ result.reasoning }}</p>
                    {% endif %}
                {% endif %}
            </div>
        {% endfor %}
    </div>
    
    <h2>Conversation</h2>
    <div class="conversation">
        {% for msg in eval_run.conversation %}
            <div class="{{ msg.role }}">
                <strong>{{ msg.role|upper }}:</strong> {{ msg.content }}
            </div>
        {% endfor %}
    </div>
    
    {% if include_charts %}
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var ctx = document.getElementById('resultsChart').getContext('2d');
            var resultsChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Yes', 'No', 'Error'],
                    datasets: [{
                        data: [{{ yes_count }}, {{ no_count }}, {{ error_count }}],
                        backgroundColor: [
                            {% if use_color %}'#2ecc71', '#e74c3c', '#f39c12'{% else %}'#777', '#999', '#bbb'{% endif %}
                        ],
                        borderColor: '#ffffff',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Evaluation Results'
                        }
                    }
                }
            });
        });
    </script>
    {% endif %}
    
    <footer>
        Generated with AgentOptim on {{ timestamp }}
    </footer>
</body>
</html>
"""
    
    # Create Jinja2 environment and template
    template = jinja2.Template(template_content)
    
    # Prepare template variables
    import datetime
    now = datetime.datetime.now()
    
    # Create indexed results for the template
    indexed_results = [(i+1, r) for i, r in enumerate(eval_run.results)]
    
    # Render the template
    html_content = template.render(
        title=title,
        eval_run=eval_run,
        results=indexed_results,
        total_questions=total_questions,
        yes_count=yes_count,
        no_count=no_count,
        error_count=error_count,
        yes_percentage=yes_percentage,
        use_color=use_color,
        include_charts=include_charts,
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return html_content

def generate_basic_html_report(eval_run, title, use_color=True, include_charts=False):
    """Generate a basic HTML report without using Jinja."""
    # Calculate summary statistics
    total_questions = len(eval_run.results)
    yes_count = sum(1 for r in eval_run.results if r.get("judgment") is True)
    no_count = sum(1 for r in eval_run.results if r.get("judgment") is False)
    error_count = sum(1 for r in eval_run.results if r.get("error") is not None)
    
    # Calculate yes percentage
    if total_questions - error_count > 0:
        yes_percentage = (yes_count / (total_questions - error_count)) * 100
    else:
        yes_percentage = 0
    
    # Basic HTML with inline style
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: {"#3498db" if use_color else "#444"}; }}
        .result {{ margin-bottom: 15px; padding: 10px; border: 1px solid #eee; }}
        .yes {{ border-left: 5px solid {"#2ecc71" if use_color else "#777"}; }}
        .no {{ border-left: 5px solid {"#e74c3c" if use_color else "#999"}; }}
        .error {{ border-left: 5px solid {"#e74c3c" if use_color else "#999"}; background-color: #ffeeee; }}
        .progress {{ width: 100%; height: 20px; background-color: #f3f3f3; }}
        .progress-bar {{ width: {yes_percentage}%; height: 20px; background-color: {"#2ecc71" if use_color else "#777"}; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <p><strong>Evaluation Set:</strong> {eval_run.evalset_name}</p>
    <p><strong>ID:</strong> {eval_run.id}</p>
    <p><strong>Date:</strong> {eval_run.timestamp}</p>
    <p><strong>Model:</strong> {eval_run.judge_model}</p>
    
    <h2>Summary</h2>
    <p><strong>Total questions:</strong> {total_questions}</p>
    <p><strong>Success rate:</strong> {total_questions - error_count}/{total_questions} questions evaluated successfully</p>
    <p><strong>Score:</strong> {round(yes_percentage, 1)}%</p>
    
    <div class="progress">
        <div class="progress-bar"></div>
    </div>
    
    <p><strong>Yes responses:</strong> {yes_count} ({round(yes_percentage, 1)}%)</p>
    <p><strong>No responses:</strong> {no_count} ({round(100 - yes_percentage, 1)}%)</p>
    """
    
    # Add confidence info if available
    if eval_run.summary and "mean_confidence" in eval_run.summary:
        html += f"""
        <p><strong>Mean confidence:</strong> {eval_run.summary["mean_confidence"]:.2f}</p>
        """
        if "mean_yes_confidence" in eval_run.summary and yes_count > 0:
            html += f"""
            <p><strong>Yes confidence:</strong> {eval_run.summary["mean_yes_confidence"]:.2f}</p>
            """
        if "mean_no_confidence" in eval_run.summary and no_count > 0:
            html += f"""
            <p><strong>No confidence:</strong> {eval_run.summary["mean_no_confidence"]:.2f}</p>
            """
    
    # Detailed results
    html += """
    <h2>Detailed Results</h2>
    """
    
    for i, result in enumerate(eval_run.results, 1):
        if result.get("error"):
            html += f"""
            <div class="result error">
                <h3>{i}. {result.get("question")}</h3>
                <p><strong>Error:</strong> {result.get("error")}</p>
            </div>
            """
        else:
            judgment_text = "Yes" if result.get("judgment") else "No"
            result_class = "yes" if result.get("judgment") else "no"
            
            confidence_display = ""
            if result.get("confidence") is not None:
                confidence_display = f" (confidence: {result.get('confidence'):.2f})"
            
            html += f"""
            <div class="result {result_class}">
                <h3>{i}. {result.get("question")}</h3>
                <p><strong>Answer:</strong> {judgment_text}{confidence_display}</p>
                """
            
            if result.get("reasoning"):
                html += f"""
                <p><strong>Reasoning:</strong> {result.get("reasoning")}</p>
                """
            
            html += """
            </div>
            """
    
    # Conversation section
    html += """
    <h2>Conversation</h2>
    <pre>
    """
    
    for msg in eval_run.conversation:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        html += f"{role}: {content}\n\n"
    
    html += """
    </pre>
    
    <p><small>Generated with AgentOptim</small></p>
</body>
</html>
    """
    
    return html

def export_html(eval_run, output_file, title, template_file=None, use_color=True, include_charts=False):
    """Export evaluation run to an HTML file."""
    # Generate HTML content
    html_content = generate_html_report(eval_run, title, template_file, use_color, include_charts)
    
    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"{Fore.GREEN}✓ HTML report exported to {output_file}{Style.RESET_ALL}")

def generate_text_comparison(first_run, second_run, use_color=True, detailed=False):
    """Generate a text-based comparison between two evaluation runs."""
    lines = []
    
    # Title
    lines.append("=" * 80)
    title = f"EVALUATION COMPARISON: {first_run.evalset_name}"
    lines.append(title.center(80))
    lines.append("=" * 80)
    lines.append("")
    
    # Format run metadata
    lines.append("RUN INFORMATION:")
    lines.append("")
    lines.append(f"{'':4}{'RUN A':^36}{'RUN B':^36}")
    lines.append(f"{'':4}{'-' * 36}{'-' * 36}")
    lines.append(f"{'ID:':4}{first_run.id:36}{second_run.id:36}")
    lines.append(f"{'Date:':4}{str(first_run.timestamp):36}{str(second_run.timestamp):36}")
    lines.append(f"{'Model:':4}{str(first_run.judge_model):36}{str(second_run.judge_model):36}")
    lines.append("")
    
    # Calculate summary statistics
    first_total = len(first_run.results)
    second_total = len(second_run.results)
    
    first_yes = sum(1 for r in first_run.results if r.get("judgment") is True)
    second_yes = sum(1 for r in second_run.results if r.get("judgment") is True)
    
    first_no = sum(1 for r in first_run.results if r.get("judgment") is False)
    second_no = sum(1 for r in second_run.results if r.get("judgment") is False)
    
    first_errors = sum(1 for r in first_run.results if r.get("error") is not None)
    second_errors = sum(1 for r in second_run.results if r.get("error") is not None)
    
    # Calculate yes percentages
    first_yes_pct = (first_yes / (first_total - first_errors)) * 100 if first_total - first_errors > 0 else 0
    second_yes_pct = (second_yes / (second_total - second_errors)) * 100 if second_total - second_errors > 0 else 0
    
    # Determine score change
    score_change = second_yes_pct - first_yes_pct
    
    # Format scores with colors if requested
    if use_color:
        first_score = f"{Fore.CYAN}{first_yes_pct:.1f}%{Style.RESET_ALL}"
        
        if score_change > 0:
            change_str = f"{Fore.GREEN}+{score_change:.1f}%{Style.RESET_ALL}"
            second_score = f"{Fore.GREEN}{second_yes_pct:.1f}%{Style.RESET_ALL}"
        elif score_change < 0:
            change_str = f"{Fore.RED}{score_change:.1f}%{Style.RESET_ALL}"
            second_score = f"{Fore.RED}{second_yes_pct:.1f}%{Style.RESET_ALL}"
        else:
            change_str = f"{Fore.YELLOW}0.0%{Style.RESET_ALL}"
            second_score = f"{Fore.YELLOW}{second_yes_pct:.1f}%{Style.RESET_ALL}"
    else:
        first_score = f"{first_yes_pct:.1f}%"
        second_score = f"{second_yes_pct:.1f}%"
        change_str = f"{score_change:+.1f}%" if score_change != 0 else "0.0%"
    
    # Add summary comparison
    lines.append("SUMMARY COMPARISON:")
    lines.append("")
    lines.append(f"{'':20}{'RUN A':^20}{'RUN B':^20}{'CHANGE':^20}")
    lines.append(f"{'-' * 80}")
    lines.append(f"{'Overall Score:':20}{first_score:^20}{second_score:^20}{change_str:^20}")
    lines.append(f"{'Yes Responses:':20}{first_yes:^20}{second_yes:^20}{second_yes - first_yes:+^20}")
    lines.append(f"{'No Responses:':20}{first_no:^20}{second_no:^20}{second_no - first_no:+^20}")
    lines.append(f"{'Total Questions:':20}{first_total:^20}{second_total:^20}{'-':^20}")
    
    # Add confidence info if available
    if (first_run.summary and "mean_confidence" in first_run.summary and
        second_run.summary and "mean_confidence" in second_run.summary):
        first_conf = first_run.summary["mean_confidence"]
        second_conf = second_run.summary["mean_confidence"]
        conf_change = second_conf - first_conf
        
        lines.append(f"{'Mean Confidence:':20}{first_conf:.2f:^20}{second_conf:.2f:^20}{conf_change:+.2f:^20}")
    
    lines.append("")
    
    # Add ASCII charts if the terminal supports it
    first_bar_len = int(first_yes_pct / 5)  # 20 chars = 100%
    second_bar_len = int(second_yes_pct / 5)
    
    lines.append("SCORE VISUALIZATION:")
    lines.append("")
    lines.append(f"Run A: [{'█' * first_bar_len}{' ' * (20 - first_bar_len)}] {first_yes_pct:.1f}%")
    lines.append(f"Run B: [{'█' * second_bar_len}{' ' * (20 - second_bar_len)}] {second_yes_pct:.1f}%")
    lines.append("")
    
    # Question-by-question comparison
    lines.append("QUESTION-BY-QUESTION COMPARISON:")
    lines.append("")
    
    # Map for easier comparison
    first_results_map = {r.get("question"): r for r in first_run.results}
    second_results_map = {r.get("question"): r for r in second_run.results}
    
    # Get all questions from both runs
    all_questions = set()
    for r in first_run.results:
        all_questions.add(r.get("question"))
    for r in second_run.results:
        all_questions.add(r.get("question"))
    
    # Sort questions for consistent output
    sorted_questions = sorted(list(all_questions))
    
    # Generate comparison for each question
    for q_idx, question in enumerate(sorted_questions, 1):
        # Get results for this question from both runs
        first_result = first_results_map.get(question, {"error": "Question not in Run A"})
        second_result = second_results_map.get(question, {"error": "Question not in Run B"})
        
        # Determine if there's a change in judgment
        first_judgment = first_result.get("judgment")
        second_judgment = second_result.get("judgment")
        
        if first_judgment is None or second_judgment is None:
            change_indicator = "[ERROR]"
            if use_color:
                change_indicator = f"{Fore.RED}{change_indicator}{Style.RESET_ALL}"
        elif first_judgment == second_judgment:
            change_indicator = "[SAME]"
            if use_color:
                change_indicator = f"{Fore.BLUE}{change_indicator}{Style.RESET_ALL}"
        elif first_judgment is False and second_judgment is True:
            change_indicator = "[IMPROVED]"
            if use_color:
                change_indicator = f"{Fore.GREEN}{change_indicator}{Style.RESET_ALL}"
        else:
            change_indicator = "[REGRESSED]"
            if use_color:
                change_indicator = f"{Fore.RED}{change_indicator}{Style.RESET_ALL}"
        
        # Add question header with change indicator
        lines.append(f"Q{q_idx}: {question} {change_indicator}")
        
        # Format judgments with colors
        first_judge = format_judgment(first_judgment, use_color)
        second_judge = format_judgment(second_judgment, use_color)
        
        # Add judgments
        lines.append(f"  Run A: {first_judge}")
        lines.append(f"  Run B: {second_judge}")
        
        # Add reasoning if detailed mode
        if detailed:
            if "reasoning" in first_result and first_result["reasoning"]:
                lines.append(f"  Run A reasoning: {first_result['reasoning']}")
            if "reasoning" in second_result and second_result["reasoning"]:
                lines.append(f"  Run B reasoning: {second_result['reasoning']}")
        
        lines.append("")
    
    # Return the formatted text comparison
    return "\n".join(lines)

def format_judgment(judgment, use_color=True):
    """Format judgment with color if supported."""
    if judgment is None:
        result = "ERROR"
        return f"{Fore.RED}{result}{Style.RESET_ALL}" if use_color else result
    elif judgment is True:
        result = "Yes"
        return f"{Fore.GREEN}{result}{Style.RESET_ALL}" if use_color else result
    else:
        result = "No"
        return f"{Fore.RED}{result}{Style.RESET_ALL}" if use_color else result

def generate_html_comparison(first_run, second_run, use_color=True, detailed=False):
    """Generate HTML comparison between two evaluation runs."""
    # Try to import Jinja for templating
    try:
        import jinja2
    except ImportError:
        print(f"{Fore.YELLOW}Warning: jinja2 not found, using basic HTML template{Style.RESET_ALL}")
        return generate_basic_html_comparison(first_run, second_run, use_color, detailed)
    
    # Calculate summary statistics
    first_total = len(first_run.results)
    second_total = len(second_run.results)
    
    first_yes = sum(1 for r in first_run.results if r.get("judgment") is True)
    second_yes = sum(1 for r in second_run.results if r.get("judgment") is True)
    
    first_no = sum(1 for r in first_run.results if r.get("judgment") is False)
    second_no = sum(1 for r in second_run.results if r.get("judgment") is False)
    
    first_errors = sum(1 for r in first_run.results if r.get("error") is not None)
    second_errors = sum(1 for r in second_run.results if r.get("error") is not None)
    
    # Calculate yes percentages
    first_yes_pct = (first_yes / (first_total - first_errors)) * 100 if first_total - first_errors > 0 else 0
    second_yes_pct = (second_yes / (second_total - second_errors)) * 100 if second_total - second_errors > 0 else 0
    
    # Determine score change
    score_change = second_yes_pct - first_yes_pct
    score_change_class = "positive" if score_change > 0 else "negative" if score_change < 0 else "neutral"
    
    # Map for easier comparison
    first_results_map = {r.get("question"): r for r in first_run.results}
    second_results_map = {r.get("question"): r for r in second_run.results}
    
    # Get all questions from both runs
    all_questions = set()
    for r in first_run.results:
        all_questions.add(r.get("question"))
    for r in second_run.results:
        all_questions.add(r.get("question"))
    
    # Sort questions for consistent output
    sorted_questions = sorted(list(all_questions))
    
    # Prepare question comparisons
    comparisons = []
    for question in sorted_questions:
        first_result = first_results_map.get(question, {"error": "Question not in Run A"})
        second_result = second_results_map.get(question, {"error": "Question not in Run B"})
        
        # Determine status
        first_judgment = first_result.get("judgment")
        second_judgment = second_result.get("judgment")
        
        if first_judgment is None or second_judgment is None:
            status = "error"
            status_text = "ERROR"
        elif first_judgment == second_judgment:
            status = "same"
            status_text = "SAME"
        elif first_judgment is False and second_judgment is True:
            status = "improved"
            status_text = "IMPROVED"
        else:
            status = "regressed"
            status_text = "REGRESSED"
        
        comparisons.append({
            "question": question,
            "first_result": first_result,
            "second_result": second_result,
            "status": status,
            "status_text": status_text
        })
    
    # HTML Template
    template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentOptim Evaluation Comparison</title>
    <style>
        :root {
            --primary-color: {% if use_color %}#3498db{% else %}#444{% endif %};
            --secondary-color: {% if use_color %}#2ecc71{% else %}#777{% endif %};
            --negative-color: {% if use_color %}#e74c3c{% else %}#999{% endif %};
            --neutral-color: {% if use_color %}#f39c12{% else %}#aaa{% endif %};
            --text-color: #333;
            --light-bg: #f9f9f9;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3 {
            color: var(--primary-color);
        }
        
        .metadata {
            display: flex;
            justify-content: space-between;
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .run-info {
            flex: 1;
            padding: 10px;
        }
        
        .run-info h3 {
            margin-top: 0;
        }
        
        .summary {
            background-color: var(--light-bg);
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .comparison-table th {
            background-color: var(--primary-color);
            color: white;
            padding: 10px;
            text-align: left;
        }
        
        .comparison-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .positive {
            color: {% if use_color %}var(--secondary-color){% else %}inherit{% endif %};
        }
        
        .negative {
            color: {% if use_color %}var(--negative-color){% else %}inherit{% endif %};
        }
        
        .neutral {
            color: {% if use_color %}var(--neutral-color){% else %}inherit{% endif %};
        }
        
        .progress-bars {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .progress-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .progress-label {
            width: 60px;
            text-align: right;
        }
        
        .progress-bar {
            height: 25px;
            background-color: #f0f0f0;
            border-radius: 5px;
            flex-grow: 1;
            overflow: hidden;
        }
        
        .progress-bar .filled {
            height: 100%;
            background-color: var(--primary-color);
            width: 0%;
            transition: width 0.5s ease-in-out;
        }
        
        .run-a .filled {
            background-color: var(--primary-color);
            width: {{ first_yes_pct }}%;
        }
        
        .run-b .filled {
            background-color: var(--secondary-color);
            width: {{ second_yes_pct }}%;
        }
        
        .question-comparison {
            margin-bottom: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .question-header {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background-color: var(--light-bg);
            border-bottom: 1px solid #eee;
        }
        
        .question-content {
            padding: 15px;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .run-result {
            flex: 1;
            min-width: 300px;
        }
        
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            color: white;
        }
        
        .status-same {
            background-color: var(--neutral-color);
        }
        
        .status-improved {
            background-color: var(--secondary-color);
        }
        
        .status-regressed {
            background-color: var(--negative-color);
        }
        
        .status-error {
            background-color: #777;
        }
        
        .judgment {
            font-weight: bold;
        }
        
        .judgment-yes {
            color: {% if use_color %}var(--secondary-color){% else %}inherit{% endif %};
        }
        
        .judgment-no {
            color: {% if use_color %}var(--negative-color){% else %}inherit{% endif %};
        }
        
        .judgment-error {
            color: #777;
        }
        
        footer {
            margin-top: 30px;
            text-align: center;
            font-size: 0.8em;
            color: #888;
        }
    </style>
</head>
<body>
    <h1>AgentOptim Evaluation Comparison</h1>
    
    <div class="metadata">
        <div class="run-info">
            <h3>Run A</h3>
            <p><strong>ID:</strong> {{ first_run.id }}</p>
            <p><strong>Date:</strong> {{ first_run.timestamp }}</p>
            <p><strong>Model:</strong> {{ first_run.judge_model }}</p>
        </div>
        
        <div class="run-info">
            <h3>Run B</h3>
            <p><strong>ID:</strong> {{ second_run.id }}</p>
            <p><strong>Date:</strong> {{ second_run.timestamp }}</p>
            <p><strong>Model:</strong> {{ second_run.judge_model }}</p>
        </div>
    </div>
    
    <h2>Summary Comparison</h2>
    
    <div class="summary">
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Run A</th>
                <th>Run B</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Overall Score</td>
                <td>{{ "%.1f"|format(first_yes_pct) }}%</td>
                <td>{{ "%.1f"|format(second_yes_pct) }}%</td>
                <td class="{{ score_change_class }}">{{ "%+.1f"|format(score_change) }}%</td>
            </tr>
            <tr>
                <td>Yes Responses</td>
                <td>{{ first_yes }}</td>
                <td>{{ second_yes }}</td>
                <td class="{{ 'positive' if second_yes > first_yes else 'negative' if second_yes < first_yes else 'neutral' }}">{{ "%+d"|format(second_yes - first_yes) }}</td>
            </tr>
            <tr>
                <td>No Responses</td>
                <td>{{ first_no }}</td>
                <td>{{ second_no }}</td>
                <td class="{{ 'negative' if second_no > first_no else 'positive' if second_no < first_no else 'neutral' }}">{{ "%+d"|format(second_no - first_no) }}</td>
            </tr>
            <tr>
                <td>Errors</td>
                <td>{{ first_errors }}</td>
                <td>{{ second_errors }}</td>
                <td class="{{ 'negative' if second_errors > first_errors else 'positive' if second_errors < first_errors else 'neutral' }}">{{ "%+d"|format(second_errors - first_errors) }}</td>
            </tr>
            <tr>
                <td>Total Questions</td>
                <td>{{ first_total }}</td>
                <td>{{ second_total }}</td>
                <td>{{ second_total - first_total }}</td>
            </tr>
            {% if first_run.summary and first_run.summary.mean_confidence is defined and second_run.summary and second_run.summary.mean_confidence is defined %}
            <tr>
                <td>Mean Confidence</td>
                <td>{{ "%.2f"|format(first_run.summary.mean_confidence) }}</td>
                <td>{{ "%.2f"|format(second_run.summary.mean_confidence) }}</td>
                <td class="{{ 'positive' if second_run.summary.mean_confidence > first_run.summary.mean_confidence else 'negative' if second_run.summary.mean_confidence < first_run.summary.mean_confidence else 'neutral' }}">
                    {{ "%+.2f"|format(second_run.summary.mean_confidence - first_run.summary.mean_confidence) }}
                </td>
            </tr>
            {% endif %}
        </table>
    </div>
    
    <h3>Score Visualization</h3>
    
    <div class="progress-bars">
        <div class="progress-container">
            <div class="progress-label">Run A:</div>
            <div class="progress-bar run-a">
                <div class="filled"></div>
            </div>
            <div>{{ "%.1f"|format(first_yes_pct) }}%</div>
        </div>
        
        <div class="progress-container">
            <div class="progress-label">Run B:</div>
            <div class="progress-bar run-b">
                <div class="filled"></div>
            </div>
            <div>{{ "%.1f"|format(second_yes_pct) }}%</div>
        </div>
    </div>
    
    <h2>Question-by-Question Comparison</h2>
    
    {% for comparison in comparisons %}
    <div class="question-comparison">
        <div class="question-header">
            <div><strong>Q{{ loop.index }}:</strong> {{ comparison.question }}</div>
            <div>
                <span class="status-badge status-{{ comparison.status }}">{{ comparison.status_text }}</span>
            </div>
        </div>
        
        <div class="question-content">
            <div class="run-result">
                <h4>Run A</h4>
                {% if comparison.first_result.error %}
                    <p><strong>Error:</strong> {{ comparison.first_result.error }}</p>
                {% else %}
                    <p>
                        <strong>Judgment:</strong> 
                        <span class="judgment judgment-{{ 'yes' if comparison.first_result.judgment else 'no' }}">
                            {{ "Yes" if comparison.first_result.judgment else "No" }}
                        </span>
                        {% if comparison.first_result.confidence is defined %}
                            (confidence: {{ "%.2f"|format(comparison.first_result.confidence) }})
                        {% endif %}
                    </p>
                    
                    {% if detailed and comparison.first_result.reasoning %}
                        <p><strong>Reasoning:</strong><br>{{ comparison.first_result.reasoning }}</p>
                    {% endif %}
                {% endif %}
            </div>
            
            <div class="run-result">
                <h4>Run B</h4>
                {% if comparison.second_result.error %}
                    <p><strong>Error:</strong> {{ comparison.second_result.error }}</p>
                {% else %}
                    <p>
                        <strong>Judgment:</strong> 
                        <span class="judgment judgment-{{ 'yes' if comparison.second_result.judgment else 'no' }}">
                            {{ "Yes" if comparison.second_result.judgment else "No" }}
                        </span>
                        {% if comparison.second_result.confidence is defined %}
                            (confidence: {{ "%.2f"|format(comparison.second_result.confidence) }})
                        {% endif %}
                    </p>
                    
                    {% if detailed and comparison.second_result.reasoning %}
                        <p><strong>Reasoning:</strong><br>{{ comparison.second_result.reasoning }}</p>
                    {% endif %}
                {% endif %}
            </div>
        </div>
    </div>
    {% endfor %}
    
    <footer>
        Generated with AgentOptim on {{ timestamp }}
    </footer>
</body>
</html>
"""
    
    # Create template and render
    template = jinja2.Template(template_content)
    
    # Prepare variables
    import datetime
    now = datetime.datetime.now()
    
    # Render HTML
    html_content = template.render(
        first_run=first_run,
        second_run=second_run,
        first_yes_pct=first_yes_pct,
        second_yes_pct=second_yes_pct,
        first_yes=first_yes,
        second_yes=second_yes,
        first_no=first_no,
        second_no=second_no,
        first_errors=first_errors,
        second_errors=second_errors,
        first_total=first_total,
        second_total=second_total,
        score_change=score_change,
        score_change_class=score_change_class,
        comparisons=comparisons,
        use_color=use_color,
        detailed=detailed,
        timestamp=now.strftime("%Y-%m-%d %H:%M:%S")
    )
    
    return html_content

def generate_basic_html_comparison(first_run, second_run, use_color=True, detailed=False):
    """Generate basic HTML comparison without Jinja."""
    # Calculate statistics
    first_total = len(first_run.results)
    second_total = len(second_run.results)
    
    first_yes = sum(1 for r in first_run.results if r.get("judgment") is True)
    second_yes = sum(1 for r in second_run.results if r.get("judgment") is True)
    
    first_no = sum(1 for r in first_run.results if r.get("judgment") is False)
    second_no = sum(1 for r in second_run.results if r.get("judgment") is False)
    
    first_errors = sum(1 for r in first_run.results if r.get("error") is not None)
    second_errors = sum(1 for r in second_run.results if r.get("error") is not None)
    
    # Calculate yes percentages
    first_yes_pct = (first_yes / (first_total - first_errors)) * 100 if first_total - first_errors > 0 else 0
    second_yes_pct = (second_yes / (second_total - second_errors)) * 100 if second_total - second_errors > 0 else 0
    
    # Determine score change
    score_change = second_yes_pct - first_yes_pct
    
    # Simple HTML content
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AgentOptim Evaluation Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: {"#3498db" if use_color else "#444"}; }}
        .comparison {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
        .run {{ flex: 1; padding: 10px; border: 1px solid #eee; margin: 0 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background-color: {"#3498db" if use_color else "#eee"}; color: {"white" if use_color else "#333"}; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .positive {{ color: {"#2ecc71" if use_color else "#333"}; }}
        .negative {{ color: {"#e74c3c" if use_color else "#333"}; }}
        .neutral {{ color: {"#f39c12" if use_color else "#333"}; }}
        .yes {{ color: {"#2ecc71" if use_color else "#333"}; font-weight: bold; }}
        .no {{ color: {"#e74c3c" if use_color else "#333"}; font-weight: bold; }}
        .question {{ border: 1px solid #eee; margin-bottom: 15px; padding: 10px; }}
        .status {{ display: inline-block; padding: 3px 8px; border-radius: 10px; color: white; font-size: 12px; }}
        .improved {{ background-color: {"#2ecc71" if use_color else "#aaa"}; }}
        .regressed {{ background-color: {"#e74c3c" if use_color else "#aaa"}; }}
        .same {{ background-color: {"#f39c12" if use_color else "#aaa"}; }}
        .error {{ background-color: #777; }}
    </style>
</head>
<body>
    <h1>AgentOptim Evaluation Comparison</h1>
    
    <div class="comparison">
        <div class="run">
            <h3>Run A</h3>
            <p><strong>ID:</strong> {first_run.id}</p>
            <p><strong>Date:</strong> {first_run.timestamp}</p>
            <p><strong>Model:</strong> {first_run.judge_model}</p>
        </div>
        
        <div class="run">
            <h3>Run B</h3>
            <p><strong>ID:</strong> {second_run.id}</p>
            <p><strong>Date:</strong> {second_run.timestamp}</p>
            <p><strong>Model:</strong> {second_run.judge_model}</p>
        </div>
    </div>
    
    <h2>Summary Comparison</h2>
    
    <table>
        <tr>
            <th>Metric</th>
            <th>Run A</th>
            <th>Run B</th>
            <th>Change</th>
        </tr>
        <tr>
            <td>Overall Score</td>
            <td>{first_yes_pct:.1f}%</td>
            <td>{second_yes_pct:.1f}%</td>
            <td class="{"positive" if score_change > 0 else "negative" if score_change < 0 else "neutral"}">{score_change:+.1f}%</td>
        </tr>
        <tr>
            <td>Yes Responses</td>
            <td>{first_yes}</td>
            <td>{second_yes}</td>
            <td class="{"positive" if second_yes > first_yes else "negative" if second_yes < first_yes else "neutral"}">{second_yes - first_yes:+d}</td>
        </tr>
        <tr>
            <td>No Responses</td>
            <td>{first_no}</td>
            <td>{second_no}</td>
            <td class="{"negative" if second_no > first_no else "positive" if second_no < first_no else "neutral"}">{second_no - first_no:+d}</td>
        </tr>
        <tr>
            <td>Total Questions</td>
            <td>{first_total}</td>
            <td>{second_total}</td>
            <td>{second_total - first_total}</td>
        </tr>
    </table>
    
    <h2>Question-by-Question Comparison</h2>
    """
    
    # Map for easier comparison
    first_results_map = {r.get("question"): r for r in first_run.results}
    second_results_map = {r.get("question"): r for r in second_run.results}
    
    # Get all questions from both runs
    all_questions = set()
    for r in first_run.results:
        all_questions.add(r.get("question"))
    for r in second_run.results:
        all_questions.add(r.get("question"))
    
    # Sort questions for consistent output
    sorted_questions = sorted(list(all_questions))
    
    # Generate comparison for each question
    for q_idx, question in enumerate(sorted_questions, 1):
        # Get results for this question from both runs
        first_result = first_results_map.get(question, {"error": "Question not in Run A"})
        second_result = second_results_map.get(question, {"error": "Question not in Run B"})
        
        # Determine status
        first_judgment = first_result.get("judgment")
        second_judgment = second_result.get("judgment")
        
        if first_judgment is None or second_judgment is None:
            status = "error"
            status_text = "ERROR"
        elif first_judgment == second_judgment:
            status = "same"
            status_text = "SAME"
        elif first_judgment is False and second_judgment is True:
            status = "improved"
            status_text = "IMPROVED"
        else:
            status = "regressed"
            status_text = "REGRESSED"
        
        html += f"""
        <div class="question">
            <div>
                <strong>Q{q_idx}:</strong> {question}
                <span class="status {status}">{status_text}</span>
            </div>
            
            <div class="comparison">
                <div class="run">
                    <h4>Run A</h4>
                    """
        
        if "error" in first_result and first_result["error"]:
            html += f"<p><strong>Error:</strong> {first_result['error']}</p>"
        else:
            html += f"""
                    <p><strong>Judgment:</strong> 
                        <span class="{'yes' if first_judgment else 'no'}">
                            {"Yes" if first_judgment else "No"}
                        </span>
                        {f" (confidence: {first_result.get('confidence'):.2f})" if "confidence" in first_result and first_result["confidence"] is not None else ""}
                    </p>
                    """
            
            if detailed and "reasoning" in first_result and first_result["reasoning"]:
                html += f"<p><strong>Reasoning:</strong><br>{first_result['reasoning']}</p>"
        
        html += """
                </div>
                
                <div class="run">
                    <h4>Run B</h4>
                    """
        
        if "error" in second_result and second_result["error"]:
            html += f"<p><strong>Error:</strong> {second_result['error']}</p>"
        else:
            html += f"""
                    <p><strong>Judgment:</strong> 
                        <span class="{'yes' if second_judgment else 'no'}">
                            {"Yes" if second_judgment else "No"}
                        </span>
                        {f" (confidence: {second_result.get('confidence'):.2f})" if "confidence" in second_result and second_result["confidence"] is not None else ""}
                    </p>
                    """
            
            if detailed and "reasoning" in second_result and second_result["reasoning"]:
                html += f"<p><strong>Reasoning:</strong><br>{second_result['reasoning']}</p>"
        
        html += """
                </div>
            </div>
        </div>
        """
    
    # Close HTML
    html += """
    <p><small>Generated with AgentOptim</small></p>
</body>
</html>
    """
    
    return html

def generate_completion_script():
    """Generate a bash completion script for the agentoptim CLI."""
    return '''
# AgentOptim CLI bash completion script
_agentoptim_completion() {
    local cur prev opts resources actions
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Define main resources
    resources="server evalset es run r dev"
    # Define actions based on resource
    evalset_actions="list get create update delete"
    run_actions="list get create"
    dev_actions="cache logs"
    
    # Handle completion based on input position
    case ${COMP_CWORD} in
        1)
            # First argument should be a resource or help
            COMPREPLY=( $(compgen -W "${resources} --help --version" -- ${cur}) )
            return 0
            ;;
        2)
            # Second argument depends on the resource
            case ${prev} in
                evalset|es)
                    COMPREPLY=( $(compgen -W "${evalset_actions}" -- ${cur}) )
                    ;;
                run|r)
                    COMPREPLY=( $(compgen -W "${run_actions}" -- ${cur}) )
                    ;;
                dev)
                    COMPREPLY=( $(compgen -W "${dev_actions}" -- ${cur}) )
                    ;;
                server)
                    COMPREPLY=( $(compgen -W "--port --debug --provider" -- ${cur}) )
                    ;;
            esac
            return 0
            ;;
        3)
            # Third argument handles common parameters
            resource="${COMP_WORDS[1]}"
            action="${COMP_WORDS[2]}"
            
            if [[ "${resource}" == "evalset" || "${resource}" == "es" ]]; then
                if [[ "${action}" == "create" ]]; then
                    COMPREPLY=( $(compgen -W "--wizard --name --questions --short-desc --long-desc" -- ${cur}) )
                elif [[ "${action}" == "get" || "${action}" == "update" || "${action}" == "delete" ]]; then
                    # Try to complete with IDs from the list command output (simplified)
                    COMPREPLY=( $(compgen -W "latest" -- ${cur}) )
                fi
            elif [[ "${resource}" == "run" || "${resource}" == "r" ]]; then
                if [[ "${action}" == "get" ]]; then
                    COMPREPLY=( $(compgen -W "latest" -- ${cur}) )
                elif [[ "${action}" == "create" ]]; then
                    # Try to complete with IDs from evalset list
                    COMPREPLY=( $(compgen -W "--interactive --text --model --provider" -- ${cur}) )
                fi
            fi
            return 0
            ;;
    esac
    
    # Default to general options when no specific completion is available
    general_opts="--help --format --output"
    COMPREPLY=( $(compgen -W "${general_opts}" -- ${cur}) )
    return 0
}

# Register the completion function
complete -F _agentoptim_completion agentoptim
'''

def install_completion():
    """Attempt to install shell completion."""
    try:
        # Generate the completion script
        script = generate_completion_script()
        
        # Determine completion path
        user_home = os.path.expanduser("~")
        completion_file = os.path.join(user_home, ".agentoptim_completion.sh")
        
        # Write completion script to file
        with open(completion_file, "w") as f:
            f.write(script)
        
        # Determine shell
        shell_path = os.environ.get("SHELL", "")
        shell_name = os.path.basename(shell_path)
        
        # Generate instruction message for user
        print(f"{Fore.GREEN}Shell completion script created at: {completion_file}{Style.RESET_ALL}")
        
        # Add specific instructions based on shell
        if "bash" in shell_name:
            print(f"\nTo enable completion, add this line to your ~/.bashrc:")
            print(f"{Fore.YELLOW}source {completion_file}{Style.RESET_ALL}")
        elif "zsh" in shell_name:
            print(f"\nTo enable completion, add these lines to your ~/.zshrc:")
            print(f"{Fore.YELLOW}autoload -Uz compinit")
            print(f"compinit")
            print(f"source {completion_file}{Style.RESET_ALL}")
        else:
            print(f"\nTo enable completion, source the script in your shell's startup file:")
            print(f"{Fore.YELLOW}source {completion_file}{Style.RESET_ALL}")
        
        print(f"\nAfter adding the line, restart your shell or run: {Fore.CYAN}source ~/.bashrc{Style.RESET_ALL} (or equivalent)")
        
        return True
    except Exception as e:
        print(f"{Fore.RED}Error installing shell completion: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        return False


def main():
    """Main entry point for the module."""
    # Handle special "--install-completion" flag before parsing other args
    if len(sys.argv) == 2 and sys.argv[1] == "--install-completion":
        install_completion()
        return
    
    start_time = time.time()
    show_timer = os.environ.get("AGENTOPTIM_SHOW_TIMER", "0") == "1"
    
    try:
        # Execute the CLI command
        run_cli()
        
        # Show execution time if enabled
        if show_timer:
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.1:
                time_str = f"{elapsed_time * 1000:.0f}ms"
            elif elapsed_time < 1:
                time_str = f"{elapsed_time * 1000:.0f}ms"
            elif elapsed_time < 60:
                time_str = f"{elapsed_time:.2f}s"
            else:
                mins, secs = divmod(int(elapsed_time), 60)
                time_str = f"{mins}m {secs}s"
                
            # Print execution time in a subtle way
            print(f"\n{Fore.CYAN}⏱ Command completed in {time_str}{Style.RESET_ALL}", file=sys.stderr)
            
    except Exception as e:
        # Show execution time even for errors if enabled
        if show_timer:
            elapsed_time = time.time() - start_time
            if elapsed_time < 1:
                time_str = f"{elapsed_time * 1000:.0f}ms"
            elif elapsed_time < 60:
                time_str = f"{elapsed_time:.2f}s"
            else:
                mins, secs = divmod(int(elapsed_time), 60)
                time_str = f"{mins}m {secs}s"
            
            print(f"\n{Fore.CYAN}⏱ Command failed after {time_str}{Style.RESET_ALL}", file=sys.stderr)
        
        logger.error(f"Error in AgentOptim CLI: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
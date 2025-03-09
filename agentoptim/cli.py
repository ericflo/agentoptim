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
          
          # Install shell tab completion
          agentoptim --install-completion
        """)
    )
    
    parser.add_argument('--version', action='version', version=f'AgentOptim v{VERSION}')
    parser.add_argument('--install-completion', action='store_true', help='Install shell completion script')
    
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
    run_get_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text",
                            help="Output format (default: text)")
    run_get_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
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
        import pandas as pd
        df = pd.DataFrame(data)
        formatted_data = df.to_string(index=False)
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
        print(f"Output saved to: {output_file}")
    else:
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
                print(f"{Fore.YELLOW}Command '{resource}' not found. Did you mean '{suggestion}'?{Style.RESET_ALL}")
                print(f"Run {Fore.CYAN}agentoptim --help{Style.RESET_ALL} to see available commands.")
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
                    print(f"{Fore.YELLOW}Action '{action}' not found for '{resource}'. Did you mean '{suggestion}'?{Style.RESET_ALL}")
                    print(f"Valid actions for '{resource}': {', '.join(valid_actions)}")
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
                
                handle_output(formatted_result, args.format, args.output)
            else:
                handle_output(result, args.format, args.output)
        
        elif args.action == "get":
            result = manage_evalset(action="get", evalset_id=args.evalset_id)
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
            
            # For other formats, create a structured response
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
        
        elif args.action == "get":
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
                
            # Format the evaluation run for output
            formatted_run = get_formatted_eval_run(eval_run)
            handle_output(formatted_run, args.format, args.output)
        
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
    
    try:
        run_cli()
    except Exception as e:
        logger.error(f"Error in AgentOptim CLI: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
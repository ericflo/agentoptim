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
VERSION = "2.1.1"
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
          
          # Create an evaluation set
          agentoptim evalset create --name "Support Quality" --questions questions.txt
          
          # List all evaluation sets to get their IDs
          agentoptim evalset list
          
          # Run an evaluation (use an ID from evalset list output)
          agentoptim run create <evalset-id> conversation.json
          
          # Get the most recent evaluation results
          agentoptim run get latest
          
          # List all evaluation runs
          agentoptim run list
        """)
    )
    
    parser.add_argument('--version', action='version', version=f'AgentOptim v{VERSION}')
    
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
    run_get_parser.add_argument("eval_run_id", help="ID of the evaluation run to retrieve, or 'latest' for most recent run")
    run_get_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text",
                            help="Output format (default: text)")
    run_get_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # run create
    run_create_parser = run_subparsers.add_parser("create", help="Run a new evaluation")
    run_create_parser.add_argument("evalset_id", help="ID of the evaluation set to use")
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


def run_cli():
    """Run the AgentOptim CLI based on the provided arguments."""
    parser = setup_parser()
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
            # Interactive mode not implemented yet - placeholder
            if args.interactive:
                print(f"{Fore.YELLOW}Interactive conversation input mode not implemented yet.{Style.RESET_ALL}")
                print("Please provide a conversation file or --text file.")
                sys.exit(1)
            
            # Load the conversation
            conversation = load_conversation(args.conversation, args.text)
            
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
                # First call run_evalset
                run_result = asyncio.run(run_evalset(
                    evalset_id=args.evalset_id,
                    conversation=conversation,
                    judge_model=args.model,
                    max_parallel=args.max_parallel,
                    omit_reasoning=args.omit_reasoning
                ))
                
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


def main():
    """Main entry point for the module."""
    try:
        run_cli()
    except Exception as e:
        logger.error(f"Error in AgentOptim CLI: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
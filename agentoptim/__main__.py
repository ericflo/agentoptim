#!/usr/bin/env python3

"""
AgentOptim CLI and MCP Server

This module serves as the entry point for both:
1. The AgentOptim CLI: python -m agentoptim [command] [args]
2. The AgentOptim MCP Server: python -m agentoptim server

Run 'python -m agentoptim --help' for usage information.
"""

import os
import sys
import json
import argparse
import logging
import uuid
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml
import colorama
from colorama import Fore, Style
from pathlib import Path

# Configure logging first
from agentoptim.utils import DATA_DIR, ensure_data_directories

# Initialize colorama
colorama.init()

# Ensure data directories exist
ensure_data_directories()

# Create a logger
log_file_path = os.path.join(DATA_DIR, "agentoptim.log")
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("AGENTOPTIM_DEBUG", "0") == "1" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),  # Use stderr to avoid breaking MCP's stdio transport
        logging.FileHandler(log_file_path),
    ],
)

# Create logger at the global level
logger = logging.getLogger("agentoptim")

def setup_parser():
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="AgentOptim: Evaluate and optimize AI conversation quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the MCP server
  agentoptim server
  
  # List all available evaluation sets
  agentoptim list
  
  # Get details about a specific evaluation set
  agentoptim get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
  
  # Create a new evaluation set
  agentoptim create --name "Support Quality" --questions questions.txt
  
  # Evaluate a conversation against an evaluation set
  agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
  
  # Evaluate a text file (treated as a single user message)
  agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e --text response.txt
  
  # Get cache statistics
  agentoptim stats
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the MCP server")
    server_parser.add_argument("--port", type=int, help="Port to run the server on")
    server_parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    server_parser.add_argument("--provider", choices=["local", "openai", "anthropic"], default="local", 
                          help="API provider (default: local)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all available evaluation sets")
    list_parser.add_argument("--format", choices=["table", "json", "yaml", "text"], default="table", 
                          help="Output format (default: table)")
    list_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get details about a specific evaluation set")
    get_parser.add_argument("evalset_id", help="ID of the evaluation set")
    get_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                         help="Output format (default: text)")
    get_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new evaluation set")
    create_parser.add_argument("--name", required=True, help="Name of the evaluation set")
    create_parser.add_argument("--questions", required=True, 
                            help="File containing questions (one per line) or comma-separated list of questions")
    create_parser.add_argument("--short-desc", required=True, help="Short description of the evaluation set")
    create_parser.add_argument("--long-desc", required=True, help="Long description of the evaluation set")
    create_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                            help="Output format (default: text)")
    create_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update an existing evaluation set")
    update_parser.add_argument("evalset_id", help="ID of the evaluation set to update")
    update_parser.add_argument("--name", help="New name for the evaluation set")
    update_parser.add_argument("--questions", 
                            help="File containing questions (one per line) or comma-separated list of questions")
    update_parser.add_argument("--short-desc", help="New short description")
    update_parser.add_argument("--long-desc", help="New long description")
    update_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                            help="Output format (default: text)")
    update_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an evaluation set")
    delete_parser.add_argument("evalset_id", help="ID of the evaluation set to delete")
    delete_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                            help="Output format (default: text)")
    delete_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a conversation against an evaluation set")
    eval_parser.add_argument("evalset_id", help="ID of the evaluation set to use")
    eval_parser.add_argument("conversation", nargs="?", help="Conversation file (JSON format)")
    eval_parser.add_argument("--text", help="Text file to evaluate (treated as a single user message)")
    eval_parser.add_argument("--model", help="Judge model to use for evaluation")
    eval_parser.add_argument("--provider", choices=["local", "openai", "anthropic"], default="local", 
                          help="API provider (default: local)")
    eval_parser.add_argument("--parallel", type=int, default=3, help="Maximum parallel evaluations (default: 3)")
    eval_parser.add_argument("--no-reasoning", action="store_true", help="Omit reasoning from results")
    eval_parser.add_argument("--format", choices=["text", "json", "yaml", "csv"], default="text", 
                          help="Output format (default: text)")
    eval_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get cache statistics")
    stats_parser.add_argument("--format", choices=["text", "json", "yaml"], default="text", 
                           help="Output format (default: text)")
    stats_parser.add_argument("--output", type=str, help="Output file (default: stdout)")
    
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

def run_cli():
    """Run the AgentOptim CLI based on the provided arguments."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Import modules only when needed to reduce startup time
    if args.command == "server":
        # Set environment variables based on arguments
        if args.debug:
            os.environ["AGENTOPTIM_DEBUG"] = "1"
        if args.port:
            os.environ["AGENTOPTIM_PORT"] = str(args.port)
            
        # Configure provider settings
        if args.provider:
            # Only set API_BASE if not already set by user
            if "AGENTOPTIM_API_BASE" not in os.environ:
                if args.provider == "openai":
                    os.environ["AGENTOPTIM_API_BASE"] = "https://api.openai.com/v1"
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = "gpt-4o-mini"
                elif args.provider == "anthropic":
                    os.environ["AGENTOPTIM_API_BASE"] = "https://api.anthropic.com/v1"
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = "claude-3-5-haiku-20241022"
                elif args.provider == "local":
                    os.environ["AGENTOPTIM_API_BASE"] = "http://localhost:1234/v1"
                    # Only set default model if not explicitly specified
                    if "AGENTOPTIM_JUDGE_MODEL" not in os.environ:
                        os.environ["AGENTOPTIM_JUDGE_MODEL"] = "meta-llama-3.1-8b-instruct"
        
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
    else:
        # For all other commands, we need to interact with the agentoptim modules directly
        from agentoptim.evalset import manage_evalset, get_cache_statistics
        from agentoptim.runner import run_evalset, get_api_cache_stats
        import asyncio
        
        if args.command == "list":
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
                
                # Create custom formatted outputs
                evalset_count = len(formatted_result)
                
                if args.format == "table":
                    # Create a prettier, more readable table
                    title = f"{Fore.CYAN}╭──────────────────────────────────────────────────╮{Style.RESET_ALL}"
                    title += f"\n{Fore.CYAN}│    Available Evaluation Sets ({evalset_count})             │{Style.RESET_ALL}"
                    title += f"\n{Fore.CYAN}╰──────────────────────────────────────────────────╯{Style.RESET_ALL}"
                    
                    # Format table manually for better control
                    id_max_len = max(len(item["id"]) for item in formatted_result)
                    name_max_len = max(len(item["name"]) for item in formatted_result)
                    name_max_len = min(name_max_len, 30)  # Limit name length
                    
                    # Table header
                    table = [
                        f"{Fore.WHITE}┌{'─' * (id_max_len + 2)}┬{'─' * (name_max_len + 2)}┬{'─' * 12}┬{'─' * 12}┐{Style.RESET_ALL}",
                        f"{Fore.WHITE}│ {Fore.YELLOW}{'ID'.ljust(id_max_len)}{Style.RESET_ALL} │ {Fore.YELLOW}{'Name'.ljust(name_max_len)}{Style.RESET_ALL} │ {Fore.YELLOW}{'Questions'.ljust(10)}{Style.RESET_ALL} │ {Fore.YELLOW}{'Actions'.ljust(10)}{Style.RESET_ALL} │{Style.RESET_ALL}",
                        f"{Fore.WHITE}├{'─' * (id_max_len + 2)}┼{'─' * (name_max_len + 2)}┼{'─' * 12}┼{'─' * 12}┤{Style.RESET_ALL}"
                    ]
                    
                    # Table rows
                    for i, item in enumerate(formatted_result):
                        name = item["name"]
                        if len(name) > name_max_len:
                            name = name[:name_max_len-3] + "..."
                            
                        row = f"{Fore.WHITE}│ {Fore.GREEN}{item['id'].ljust(id_max_len)}{Style.RESET_ALL} │ "
                        row += f"{Fore.GREEN}{name.ljust(name_max_len)}{Style.RESET_ALL} │ "
                        row += f"{Fore.CYAN}{str(item['questions']).ljust(10)}{Style.RESET_ALL} │ "
                        row += f"{Fore.BLUE}{'get/eval'.ljust(10)}{Style.RESET_ALL} │"
                        table.append(row)
                        
                        # Add separator between rows except after the last row
                        if i < len(formatted_result) - 1:
                            table.append(f"{Fore.WHITE}├{'─' * (id_max_len + 2)}┼{'─' * (name_max_len + 2)}┼{'─' * 12}┼{'─' * 12}┤{Style.RESET_ALL}")
                    
                    # Table footer
                    table.append(f"{Fore.WHITE}└{'─' * (id_max_len + 2)}┴{'─' * (name_max_len + 2)}┴{'─' * 12}┴{'─' * 12}┘{Style.RESET_ALL}")
                    
                    # Add guidance
                    guidance = [
                        f"\n{Fore.YELLOW}Usage:{Style.RESET_ALL}",
                        f"  {Fore.GREEN}agentoptim get <id> {Style.RESET_ALL}- View details of an evaluation set",
                        f"  {Fore.GREEN}agentoptim eval <id> <conversation.json> {Style.RESET_ALL}- Evaluate a conversation"
                    ]
                    
                    # Print everything
                    table_str = "\n".join(table)
                    guidance_str = "\n".join(guidance)
                    print(f"{title}\n\n{table_str}\n{guidance_str}")
                    
                    if args.output:
                        # Create plain version without colors for file output
                        plain_title = f"Available Evaluation Sets ({evalset_count})"
                        
                        plain_table = [
                            f"┌{'─' * (id_max_len + 2)}┬{'─' * (name_max_len + 2)}┬{'─' * 12}┬{'─' * 12}┐",
                            f"│ {'ID'.ljust(id_max_len)} │ {'Name'.ljust(name_max_len)} │ {'Questions'.ljust(10)} │ {'Actions'.ljust(10)} │",
                            f"├{'─' * (id_max_len + 2)}┼{'─' * (name_max_len + 2)}┼{'─' * 12}┼{'─' * 12}┤"
                        ]
                        
                        for i, item in enumerate(formatted_result):
                            name = item["name"]
                            if len(name) > name_max_len:
                                name = name[:name_max_len-3] + "..."
                            
                            row = f"│ {item['id'].ljust(id_max_len)} │ {name.ljust(name_max_len)} │ "
                            row += f"{str(item['questions']).ljust(10)} │ {'get/eval'.ljust(10)} │"
                            plain_table.append(row)
                            
                            if i < len(formatted_result) - 1:
                                plain_table.append(f"├{'─' * (id_max_len + 2)}┼{'─' * (name_max_len + 2)}┼{'─' * 12}┼{'─' * 12}┤")
                        
                        plain_table.append(f"└{'─' * (id_max_len + 2)}┴{'─' * (name_max_len + 2)}┴{'─' * 12}┴{'─' * 12}┘")
                        
                        plain_guidance = [
                            "\nUsage:",
                            "  agentoptim get <id> - View details of an evaluation set",
                            "  agentoptim eval <id> <conversation.json> - Evaluate a conversation"
                        ]
                        
                        with open(args.output, "w", encoding="utf-8") as f:
                            plain_table_str = "\n".join(plain_table)
                            plain_guidance_str = "\n".join(plain_guidance)
                            f.write(f"{plain_title}\n\n{plain_table_str}\n{plain_guidance_str}")
                        
                        print(f"Output saved to: {args.output}")
                    
                    return  # Skip the default handle_output
                    
                elif args.format == "text":
                    # Create a more visually appealing text format with box drawing
                    width = 50  # Same fixed width as cards
                    
                    title_text = f"Available Evaluation Sets ({evalset_count})"
                    title_padding = width - 2 - len(title_text)  # -2 for the borders
                    left_title_pad = title_padding // 2
                    right_title_pad = title_padding - left_title_pad
                    
                    title = f"{Fore.CYAN}╭{'─' * (width-2)}╮{Style.RESET_ALL}"
                    title += f"\n{Fore.CYAN}│{' ' * left_title_pad}{title_text}{' ' * right_title_pad}│{Style.RESET_ALL}"
                    title += f"\n{Fore.CYAN}╰{'─' * (width-2)}╯{Style.RESET_ALL}\n"
                    
                    # Format each evalset as a card
                    cards = []
                    for i, item in enumerate(formatted_result):
                        card = []
                        
                        width = 50  # Set a fixed card width
                        
                        # Card header with name
                        name = item['name']
                        if len(name) > width - 8:  # Leave some buffer space
                            name = name[:width-11] + "..."
                            
                        name_padding = width - 2 - len(name)  # -2 for the borders
                        left_pad = name_padding // 2
                        right_pad = name_padding - left_pad
                        
                        card.append(f"{Fore.GREEN}┌{'─' * (width-2)}┐{Style.RESET_ALL}")
                        card.append(f"{Fore.GREEN}│{' ' * left_pad}{Fore.WHITE}{name}{Style.RESET_ALL}{Fore.GREEN}{' ' * right_pad}│{Style.RESET_ALL}")
                        card.append(f"{Fore.GREEN}├{'─' * (width-2)}┤{Style.RESET_ALL}")
                        
                        # Calculate how much space to leave for content after the label
                        content_width = width - 4  # -4 for "│ " at start and " │" at end
                        
                        # Card content
                        id_label = f"{Fore.YELLOW}ID:{Style.RESET_ALL} "
                        id_content_space = content_width - len("ID: ")
                        # Truncate ID if necessary
                        id_value = item['id']
                        if len(id_value) > id_content_space:
                            id_value = id_value[:id_content_space-3] + "..."
                        
                        card.append(f"{Fore.GREEN}│ {id_label}{id_value.ljust(id_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                        
                        # Description (with word wrap for longer descriptions)
                        if item['description']:
                            desc_label = f"{Fore.YELLOW}Description:{Style.RESET_ALL} "
                            desc_content_space = content_width - len("Description: ")
                            desc = item['description']
                            
                            # If description is too long, split into two lines
                            if len(desc) > desc_content_space:
                                # First line with label
                                first_line = desc[:desc_content_space]
                                card.append(f"{Fore.GREEN}│ {desc_label}{first_line.ljust(desc_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                                
                                # Second line (indented to align with first line content)
                                indent = len("Description: ")
                                remaining = desc[desc_content_space:]
                                if len(remaining) > desc_content_space:
                                    remaining = remaining[:desc_content_space-3] + "..."
                                card.append(f"{Fore.GREEN}│ {' ' * indent}{remaining.ljust(desc_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                            else:
                                # Single line for short descriptions
                                card.append(f"{Fore.GREEN}│ {desc_label}{desc.ljust(desc_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                        
                        # Question count
                        questions_label = f"{Fore.YELLOW}Questions:{Style.RESET_ALL} "
                        questions_content_space = content_width - len("Questions: ")
                        card.append(f"{Fore.GREEN}│ {questions_label}{str(item['questions']).ljust(questions_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                        
                        # Card footer with actions
                        card.append(f"{Fore.GREEN}├{'─' * (width-2)}┤{Style.RESET_ALL}")
                        actions_label = f"{Fore.BLUE}Actions:{Style.RESET_ALL} "
                        actions_content_space = content_width - len("Actions: ")
                        card.append(f"{Fore.GREEN}│ {actions_label}{'get, eval'.ljust(actions_content_space)}{Fore.GREEN} │{Style.RESET_ALL}")
                        card.append(f"{Fore.GREEN}└{'─' * (width-2)}┘{Style.RESET_ALL}")
                        
                        cards.append("\n".join(card))
                    
                    # Join all cards with a separator
                    all_cards = "\n\n".join(cards)
                    
                    # Add guidance
                    guidance = [
                        f"\n{Fore.YELLOW}Usage:{Style.RESET_ALL}",
                        f"  {Fore.WHITE}agentoptim get <id> {Style.RESET_ALL}- View details of an evaluation set",
                        f"  {Fore.WHITE}agentoptim eval <id> <conversation.json> {Style.RESET_ALL}- Evaluate a conversation"
                    ]
                    
                    # Print everything
                    guidance_str = "\n".join(guidance)
                    print(f"{title}\n{all_cards}\n{guidance_str}")
                    
                    # If output file is specified, write without ANSI colors
                    if args.output:
                        # Create plain title with matching width
                        width = 50  # Same fixed width as cards
                        title_text = f"Available Evaluation Sets ({evalset_count})"
                        title_padding = width - 2 - len(title_text)  # -2 for the borders
                        left_title_pad = title_padding // 2
                        right_title_pad = title_padding - left_title_pad
                        
                        plain_title = f"╭{'─' * (width-2)}╮\n"
                        plain_title += f"│{' ' * left_title_pad}{title_text}{' ' * right_title_pad}│\n"
                        plain_title += f"╰{'─' * (width-2)}╯\n"
                        
                        # Create plain version of cards
                        plain_cards = []
                        for item in formatted_result:
                            plain_card = []
                            
                            width = 50  # Same fixed card width as colored version
                            
                            # Card header with name
                            name = item['name']
                            if len(name) > width - 8:  # Leave some buffer space
                                name = name[:width-11] + "..."
                                
                            name_padding = width - 2 - len(name)  # -2 for the borders
                            left_pad = name_padding // 2
                            right_pad = name_padding - left_pad
                            
                            plain_card.append(f"┌{'─' * (width-2)}┐")
                            plain_card.append(f"│{' ' * left_pad}{name}{' ' * right_pad}│")
                            plain_card.append(f"├{'─' * (width-2)}┤")
                            
                            # Calculate how much space to leave for content after the label
                            content_width = width - 4  # -4 for "│ " at start and " │" at end
                            
                            # Card content - ID
                            id_content_space = content_width - len("ID: ")
                            id_value = item['id']
                            if len(id_value) > id_content_space:
                                id_value = id_value[:id_content_space-3] + "..."
                            plain_card.append(f"│ ID: {id_value.ljust(id_content_space)} │")
                            
                            # Description with word wrap
                            if item['description']:
                                desc_content_space = content_width - len("Description: ")
                                desc = item['description']
                                
                                # If description is too long, split into two lines
                                if len(desc) > desc_content_space:
                                    # First line with label
                                    first_line = desc[:desc_content_space]
                                    plain_card.append(f"│ Description: {first_line.ljust(desc_content_space)} │")
                                    
                                    # Second line (indented to align with first line content)
                                    indent = len("Description: ")
                                    remaining = desc[desc_content_space:]
                                    if len(remaining) > desc_content_space:
                                        remaining = remaining[:desc_content_space-3] + "..."
                                    plain_card.append(f"│ {' ' * indent}{remaining.ljust(desc_content_space)} │")
                                else:
                                    # Single line for short descriptions
                                    plain_card.append(f"│ Description: {desc.ljust(desc_content_space)} │")
                            
                            # Question count
                            questions_content_space = content_width - len("Questions: ")
                            plain_card.append(f"│ Questions: {str(item['questions']).ljust(questions_content_space)} │")
                            
                            # Card footer with actions
                            plain_card.append(f"├{'─' * (width-2)}┤")
                            actions_content_space = content_width - len("Actions: ")
                            plain_card.append(f"│ Actions: {'get, eval'.ljust(actions_content_space)} │")
                            plain_card.append(f"└{'─' * (width-2)}┘")
                            
                            plain_cards.append("\n".join(plain_card))
                        
                        # Join all cards with a separator
                        all_plain_cards = "\n\n".join(plain_cards)
                        
                        # Add plain guidance
                        plain_guidance = [
                            "\nUsage:",
                            "  agentoptim get <id> - View details of an evaluation set",
                            "  agentoptim eval <id> <conversation.json> - Evaluate a conversation"
                        ]
                        
                        with open(args.output, "w", encoding="utf-8") as f:
                            plain_guidance_str = "\n".join(plain_guidance)
                            f.write(f"{plain_title}\n{all_plain_cards}\n{plain_guidance_str}")
                        
                        print(f"Output saved to: {args.output}")
                    
                    return  # Skip the default handle_output
                else:
                    handle_output(formatted_result, args.format, args.output)
            else:
                handle_output(result, args.format, args.output)
                
        elif args.command == "get":
            result = manage_evalset(action="get", evalset_id=args.evalset_id)
            handle_output(result, args.format, args.output)
            
        elif args.command == "create":
            questions = parse_questions(args.questions)
            result = manage_evalset(
                action="create",
                name=args.name,
                questions=questions,
                short_description=args.short_desc,
                long_description=args.long_desc
            )
            handle_output(result, args.format, args.output)
            
        elif args.command == "update":
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
            
        elif args.command == "delete":
            result = manage_evalset(action="delete", evalset_id=args.evalset_id)
            handle_output(result, args.format, args.output)
            
        elif args.command == "eval":
            # Load the conversation
            conversation = load_conversation(args.conversation, args.text)
            
            # Configure provider settings
            if args.provider:
                # Only set API_BASE if not already set by user
                if "AGENTOPTIM_API_BASE" not in os.environ:
                    if args.provider == "openai":
                        os.environ["AGENTOPTIM_API_BASE"] = "https://api.openai.com/v1"
                    elif args.provider == "anthropic":
                        os.environ["AGENTOPTIM_API_BASE"] = "https://api.anthropic.com/v1"
                    elif args.provider == "local":
                        os.environ["AGENTOPTIM_API_BASE"] = "http://localhost:1234/v1"
                
                # Set default model based on provider if not explicitly specified
                if not args.model:
                    if args.provider == "openai":
                        args.model = "gpt-4o-mini"
                    elif args.provider == "anthropic":
                        args.model = "claude-3-5-haiku-20241022"
                    elif args.provider == "local":
                        args.model = "meta-llama-3.1-8b-instruct"
            
            # Set environment variables for judge model and omit_reasoning if specified
            if args.model:
                os.environ["AGENTOPTIM_JUDGE_MODEL"] = args.model
            if args.no_reasoning:
                os.environ["AGENTOPTIM_OMIT_REASONING"] = "1"
                
            # Run the evaluation
            try:
                # We need to run the async function in the asyncio event loop
                result = asyncio.run(run_evalset(
                    evalset_id=args.evalset_id,
                    conversation=conversation,
                    judge_model=args.model,
                    max_parallel=args.parallel,
                    omit_reasoning=args.no_reasoning
                ))
                
                # For CSV format, flatten the results
                if args.format == "csv":
                    # Extract just the results list for CSV output
                    if "results" in result:
                        flattened_results = []
                        for i, r in enumerate(result["results"]):
                            flattened_results.append({
                                "question_number": i + 1,
                                "question": r["question"],
                                "judgment": "Yes" if r["judgment"] else "No",
                                "confidence": r["confidence"],
                                "reasoning": r.get("reasoning", "")
                            })
                        handle_output(flattened_results, args.format, args.output)
                    else:
                        handle_output(result, args.format, args.output)
                else:
                    handle_output(result, args.format, args.output)
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
                sys.exit(1)
                
        elif args.command == "stats":
            evalset_stats = get_cache_statistics()
            api_stats = get_api_cache_stats()
            
            # Calculate overall stats
            total_hits = evalset_stats["hits"] + api_stats["hits"]
            total_misses = evalset_stats["misses"] + api_stats["misses"]
            total_requests = total_hits + total_misses
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                "evalset_cache": evalset_stats,
                "api_cache": api_stats,
                "overall": {
                    "hit_rate_pct": round(overall_hit_rate, 2),
                    "total_hits": total_hits,
                    "total_misses": total_misses,
                    "estimated_time_saved_seconds": round(total_hits * 0.5, 1)
                }
            }
            
            # Format a message with the statistics
            stats["formatted_message"] = "\n".join([
                "# Cache Performance Statistics",
                "",
                "## EvalSet Cache",
                f"- Size: {evalset_stats['size']} / {evalset_stats['capacity']} (current/max)",
                f"- Hit Rate: {evalset_stats['hit_rate_pct']}%",
                f"- Hits: {evalset_stats['hits']}",
                f"- Misses: {evalset_stats['misses']}",
                f"- Evictions: {evalset_stats['evictions']}",
                f"- Expirations: {evalset_stats['expirations']}",
                "",
                "## API Response Cache",
                f"- Size: {api_stats['size']} / {api_stats['capacity']} (current/max)",
                f"- Hit Rate: {api_stats['hit_rate_pct']}%",
                f"- Hits: {api_stats['hits']}",
                f"- Misses: {api_stats['misses']}",
                f"- Evictions: {api_stats['evictions']}",
                f"- Expirations: {api_stats['expirations']}",
                "",
                "## Overall Performance",
                f"- Combined Hit Rate: {round(overall_hit_rate, 2)}%",
                f"- Total Hits: {total_hits}",
                f"- Total Misses: {total_misses}",
                f"- Resource Savings: Approximately {round(total_hits * 0.5, 1)} seconds of API processing time saved"
            ])
            
            handle_output(stats, args.format, args.output)

def main():
    """Main entry point for the module."""
    try:
        run_cli()
    except Exception as e:
        logger.error(f"Error in AgentOptim CLI: {str(e)}", exc_info=True)
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

# Run main function when module is executed directly
if __name__ == "__main__":
    main()
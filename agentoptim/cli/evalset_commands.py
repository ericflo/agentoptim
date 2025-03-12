"""
EvalSet commands for the AgentOptim CLI
"""

import os
import json
import logging
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from colorama import Fore, Style

from agentoptim.cli.core import format_box, FancySpinner
from agentoptim.utils import DATA_DIR

# Set up logger
logger = logging.getLogger(__name__)

def setup_evalset_parser(subparsers):
    """Set up the evalset command parser."""
    # Add evalset command
    evalset_parser = subparsers.add_parser(
        "evalset",
        help="Manage evaluation sets",
        aliases=["es"],
        description="Create, list, get, update, and delete evaluation sets."
    )
    
    # Add evalset subcommands
    evalset_subparsers = evalset_parser.add_subparsers(
        dest="action",
        help="Action to perform"
    )
    
    # List command
    list_parser = evalset_subparsers.add_parser(
        "list",
        help="List all evaluation sets",
        description="List all available evaluation sets with their IDs."
    )
    list_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    list_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    
    # Get command
    get_parser = evalset_subparsers.add_parser(
        "get",
        help="Get details of an evaluation set",
        description="Get details of a specific evaluation set by ID."
    )
    get_parser.add_argument(
        "evalset_id",
        help="ID of the evaluation set to get"
    )
    get_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    get_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    get_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    
    # Create command
    create_parser = evalset_subparsers.add_parser(
        "create",
        help="Create a new evaluation set",
        description="Create a new evaluation set with questions and description."
    )
    create_parser.add_argument(
        "--wizard", "-w",
        action="store_true",
        help="Run interactive creation wizard"
    )
    create_parser.add_argument(
        "--name", "-n",
        help="Name of the evaluation set"
    )
    create_parser.add_argument(
        "--questions", "-q",
        help="Questions file or comma-separated list"
    )
    create_parser.add_argument(
        "--short-desc", "-s",
        help="Short description of the evaluation set"
    )
    create_parser.add_argument(
        "--long-desc", "-l",
        help="Long description of the evaluation set"
    )
    create_parser.add_argument(
        "--template", "-t",
        help="Template file to use for evaluations"
    )
    create_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    create_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    create_parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    
    # Update command
    update_parser = evalset_subparsers.add_parser(
        "update",
        help="Update an existing evaluation set",
        description="Update an existing evaluation set by ID."
    )
    update_parser.add_argument(
        "evalset_id",
        help="ID of the evaluation set to update"
    )
    update_parser.add_argument(
        "--name", "-n",
        help="New name for the evaluation set"
    )
    update_parser.add_argument(
        "--questions", "-q",
        help="New questions file or comma-separated list"
    )
    update_parser.add_argument(
        "--short-desc", "-s",
        help="New short description"
    )
    update_parser.add_argument(
        "--long-desc", "-l",
        help="New long description"
    )
    update_parser.add_argument(
        "--template", "-t",
        help="New template file"
    )
    update_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    update_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    update_parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    
    # Delete command
    delete_parser = evalset_subparsers.add_parser(
        "delete",
        help="Delete an evaluation set",
        description="Delete an evaluation set by ID."
    )
    delete_parser.add_argument(
        "evalset_id",
        help="ID of the evaluation set to delete"
    )
    delete_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)"
    )
    delete_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    delete_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    delete_parser.add_argument(
        "--confirm", "-c",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Set default function
    evalset_parser.set_defaults(func=handle_evalset_command)
    
    return evalset_parser

def handle_output(data, args):
    """Handle outputting data based on format and destination."""
    if args.format == "json":
        output = json.dumps(data, indent=2)
    elif args.format == "yaml":
        try:
            import yaml
            output = yaml.dump(data, sort_keys=False)
        except ImportError:
            output = json.dumps(data, indent=2)
            print(f"{Fore.YELLOW}Warning: PyYAML not installed. Falling back to JSON.{Style.RESET_ALL}",
                  file=sys.stderr)
    else:
        output = data  # Text format
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"{Fore.GREEN}Output written to {args.output}{Style.RESET_ALL}", file=sys.stderr)
    else:
        print(output)
    
    return 0

def parse_questions(questions_input):
    """Parse questions from file or comma-separated string."""
    # Check if input is a file path
    if os.path.exists(questions_input):
        try:
            with open(questions_input, 'r') as f:
                content = f.read().strip()
                # Check if file contains JSON
                if content.startswith('[') and content.endswith(']'):
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                # Otherwise, treat each line as a question
                return [q.strip() for q in content.split('\n') if q.strip()]
        except Exception as e:
            raise ValueError(f"Error reading questions file: {str(e)}")
    
    # Otherwise treat as comma-separated list
    return [q.strip() for q in questions_input.split(',') if q.strip()]

def read_file_if_exists(file_path):
    """Read the contents of a file if it exists."""
    if not file_path:
        return None
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")
    
    return file_path  # If not a file, return the input as-is

def run_interactive_wizard():
    """Run an interactive wizard to create an evaluation set."""
    print(f"{Fore.CYAN}=== EvalSet Creation Wizard ==={Style.RESET_ALL}")
    print("This wizard will help you create an evaluation set with questions.")
    print("Press Ctrl+C at any time to cancel.\n")
    
    # Get name
    name = input(f"{Fore.CYAN}Name of the evaluation set:{Style.RESET_ALL} ")
    if not name:
        print(f"{Fore.RED}Error: Name is required{Style.RESET_ALL}")
        return None
    
    # Get short description
    short_desc = input(f"{Fore.CYAN}Short description:{Style.RESET_ALL} ")
    if not short_desc:
        print(f"{Fore.YELLOW}Warning: Short description is recommended{Style.RESET_ALL}")
    
    # Get long description
    print(f"{Fore.CYAN}Long description (optional, press Enter twice to finish):{Style.RESET_ALL}")
    long_desc_lines = []
    while True:
        line = input()
        if not line and (not long_desc_lines or not long_desc_lines[-1]):
            break
        long_desc_lines.append(line)
    long_desc = "\n".join(long_desc_lines).strip()
    
    # Get questions
    questions = []
    print(f"{Fore.CYAN}Enter questions (one per line, press Enter twice to finish):{Style.RESET_ALL}")
    while True:
        question = input(f"{Fore.GREEN}Question {len(questions) + 1}:{Style.RESET_ALL} ")
        if not question:
            if not questions:
                print(f"{Fore.RED}Error: At least one question is required{Style.RESET_ALL}")
            else:
                break
        else:
            questions.append(question)
    
    # Get template (optional)
    print(f"{Fore.CYAN}Enter template file path (optional):{Style.RESET_ALL}")
    template_path = input("File path: ")
    template = None
    if template_path:
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    template = f.read()
            except Exception as e:
                print(f"{Fore.RED}Error reading template file: {str(e)}{Style.RESET_ALL}")
                return None
        else:
            print(f"{Fore.RED}Template file not found: {template_path}{Style.RESET_ALL}")
            return None
    
    # Show summary
    print(f"\n{Fore.CYAN}=== Summary ==={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Name:{Style.RESET_ALL} {name}")
    print(f"{Fore.GREEN}Short Description:{Style.RESET_ALL} {short_desc}")
    print(f"{Fore.GREEN}Long Description:{Style.RESET_ALL} {long_desc[:100]}{'...' if len(long_desc) > 100 else ''}")
    print(f"{Fore.GREEN}Questions:{Style.RESET_ALL}")
    for i, q in enumerate(questions):
        print(f"  {i+1}. {q}")
    if template:
        print(f"{Fore.GREEN}Template:{Style.RESET_ALL} {template[:100]}{'...' if len(template) > 100 else ''}")
    
    # Confirm
    confirm = input(f"\n{Fore.YELLOW}Create this evaluation set? (y/n):{Style.RESET_ALL} ")
    if confirm.lower() not in ['y', 'yes']:
        print(f"{Fore.RED}Cancelled{Style.RESET_ALL}")
        return None
    
    return {
        "name": name,
        "short_description": short_desc,
        "long_description": long_desc,
        "questions": questions,
        "template": template
    }

def handle_evalset_list(args):
    """Handle the evalset list command."""
    from agentoptim.evalset import manage_evalset
    
    # Get the evalsets
    result = manage_evalset(action="list")
    
    # Check for errors in the result
    if result.get("error"):
        print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        return 1
    
    # Get the evalsets from the items list
    evalsets = result.get("items", [])
    
    # Format the output based on the format
    if args.format in ["json", "yaml"]:
        return handle_output(evalsets, args)
    
    # Text output
    if not evalsets:
        if not args.quiet:
            print(f"{Fore.YELLOW}No evaluation sets found{Style.RESET_ALL}")
        return 0
    
    # Try to use rich for fancy tables if available
    try:
        if not args.quiet:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            
            table = Table(title="EvalSet List")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Questions", style="magenta")
            table.add_column("Short Description", style="yellow")
            
            for evalset in evalsets:
                # Get question count if available
                question_count = str(len(evalset.get("questions", []))) if "questions" in evalset else "?"
                
                table.add_row(
                    str(evalset["id"]),
                    str(evalset["name"]),
                    question_count,
                    str(evalset.get("short_description", ""))
                )
            
            console.print(table)
        else:
            # Quiet mode - just print IDs
            for evalset in evalsets:
                print(evalset["id"])
    except ImportError:
        # Fallback to simple table
        if not args.quiet:
            print(f"{Fore.CYAN}{'ID':<36} {'Name':<30} {'Questions':>9} {'Description':<30}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*36} {'='*30} {'='*9} {'='*30}{Style.RESET_ALL}")
            
            for evalset in evalsets:
                # Get question count if available
                question_count = str(len(evalset.get("questions", []))) if "questions" in evalset else "?"
                print(f"{evalset['id']:<36} {evalset['name']:<30} {question_count:>9} {evalset.get('short_description', ''):<30}")
        else:
            # Quiet mode - just print IDs
            for evalset in evalsets:
                print(evalset["id"])
    
    return 0

def handle_evalset_get(args):
    """Handle the evalset get command."""
    from agentoptim.evalset import manage_evalset
    
    # Get the evalset
    result = manage_evalset(action="get", evalset_id=args.evalset_id)
    
    if result.get("error"):
        print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        return 1
    
    # The evalset is in result["evalset"] for the get action
    evalset = result.get("evalset", {})
    
    # Format the output based on the format
    if args.format in ["json", "yaml"]:
        return handle_output(evalset, args)
    
    # Text output
    if args.quiet:
        print(evalset["id"])
        return 0
    
    # Try to use rich for fancy output if available
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        
        console = Console()
        
        # Basic info panel
        info = f"ID: {evalset['id']}\nName: {evalset['name']}\nQuestions: {len(evalset['questions'])}"
        console.print(Panel(info, title="EvalSet Details", border_style="cyan"))
        
        # Short description
        if evalset.get("short_description"):
            console.print(Panel(evalset["short_description"], title="Short Description", border_style="green"))
        
        # Long description (as markdown if it looks like markdown)
        if evalset.get("long_description"):
            long_desc = evalset["long_description"]
            if "##" in long_desc or "*" in long_desc:
                # Looks like markdown
                console.print(Panel(Markdown(long_desc), title="Long Description", border_style="yellow"))
            else:
                console.print(Panel(long_desc, title="Long Description", border_style="yellow"))
        
        # Questions
        questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(evalset["questions"])])
        console.print(Panel(questions, title="Questions", border_style="magenta"))
        
        # Template (if present)
        if evalset.get("template"):
            console.print(Panel(evalset["template"], title="Template", border_style="blue"))
    
    except ImportError:
        # Fallback to simple output
        print(format_box("EvalSet Details", 
                        f"ID:       {evalset['id']}\n"
                        f"Name:     {evalset['name']}\n"
                        f"Questions: {len(evalset['questions'])}", 
                        border_color=Fore.CYAN))
        
        if evalset.get("short_description"):
            print(format_box("Short Description", evalset["short_description"], border_color=Fore.GREEN))
        
        if evalset.get("long_description"):
            print(format_box("Long Description", evalset["long_description"], border_color=Fore.YELLOW))
        
        print(format_box("Questions", 
                        "\n".join([f"{i+1}. {q}" for i, q in enumerate(evalset["questions"])]), 
                        border_color=Fore.MAGENTA))
        
        if evalset.get("template"):
            print(format_box("Template", evalset["template"], border_color=Fore.BLUE))
    
    return 0

def handle_evalset_create(args):
    """Handle the evalset create command."""
    from agentoptim.evalset import manage_evalset
    
    wizard_data = None
    if args.wizard:
        try:
            wizard_data = run_interactive_wizard()
            if not wizard_data:
                return 1
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Cancelled{Style.RESET_ALL}")
            return 1
    
    # Prepare creation parameters
    if wizard_data:
        # Use data from the wizard
        name = wizard_data["name"]
        short_desc = wizard_data["short_description"]
        long_desc = wizard_data["long_description"]
        questions = wizard_data["questions"]
        template = wizard_data["template"]
    else:
        # Use command line arguments
        name = args.name
        if not name:
            print(f"{Fore.RED}Error: Name is required{Style.RESET_ALL}")
            return 1
        
        short_desc = args.short_desc or ""
        
        # Get long description from file or argument
        long_desc = args.long_desc or ""
        if long_desc and os.path.exists(long_desc):
            try:
                with open(long_desc, 'r') as f:
                    long_desc = f.read().strip()
            except Exception as e:
                print(f"{Fore.RED}Error reading long description file: {str(e)}{Style.RESET_ALL}")
                return 1
        
        # Get questions from file or argument
        if not args.questions:
            print(f"{Fore.RED}Error: Questions are required{Style.RESET_ALL}")
            return 1
        
        try:
            questions = parse_questions(args.questions)
        except Exception as e:
            print(f"{Fore.RED}Error parsing questions: {str(e)}{Style.RESET_ALL}")
            return 1
        
        # Get template from file or argument
        template = None
        if args.template:
            if os.path.exists(args.template):
                try:
                    with open(args.template, 'r') as f:
                        template = f.read()
                except Exception as e:
                    print(f"{Fore.RED}Error reading template file: {str(e)}{Style.RESET_ALL}")
                    return 1
            else:
                template = args.template
    
    # Create the evalset
    spinner = None
    if not args.quiet and sys.stdout.isatty():
        spinner = FancySpinner()
        spinner.start(f"Creating EvalSet '{name}'...")
    
    try:
        create_params = {
            "action": "create",
            "name": name,
            "questions": questions,
            "short_description": short_desc,
            "long_description": long_desc
        }
        
        if template:
            create_params["template"] = template
        
        result = manage_evalset(**create_params)
        
        if spinner:
            spinner.stop()
        
        if result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Result format is consistent for create action
        evalset = result.get("evalset", {})
        
        # Format the output based on the format
        if args.format in ["json", "yaml"]:
            return handle_output(evalset, args)
        
        # Text output
        if args.quiet:
            print(evalset["id"])
        else:
            print(f"{Fore.GREEN}EvalSet created successfully!{Style.RESET_ALL}")
            print(f"ID: {evalset['id']}")
            print(f"Name: {evalset['name']}")
            print(f"Questions: {len(evalset['questions'])}")
            
            # Show how to use the evalset
            print(f"\n{Fore.YELLOW}To view this evalset:{Style.RESET_ALL}")
            print(f"  agentoptim evalset get {evalset['id']}")
            print(f"\n{Fore.YELLOW}To use this evalset for evaluation:{Style.RESET_ALL}")
            print(f"  agentoptim run create {evalset['id']} conversation.json")
        
        return 0
    
    except Exception as e:
        if spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_evalset_update(args):
    """Handle the evalset update command."""
    from agentoptim.evalset import manage_evalset
    
    # Prepare update parameters
    update_params = {
        "action": "update",
        "evalset_id": args.evalset_id
    }
    
    # Add optional parameters if provided
    if args.name:
        update_params["name"] = args.name
    
    if args.short_desc:
        update_params["short_description"] = args.short_desc
    
    if args.long_desc:
        long_desc = args.long_desc
        if os.path.exists(long_desc):
            try:
                with open(long_desc, 'r') as f:
                    long_desc = f.read().strip()
            except Exception as e:
                print(f"{Fore.RED}Error reading long description file: {str(e)}{Style.RESET_ALL}")
                return 1
        update_params["long_description"] = long_desc
    
    if args.questions:
        try:
            questions = parse_questions(args.questions)
            update_params["questions"] = questions
        except Exception as e:
            print(f"{Fore.RED}Error parsing questions: {str(e)}{Style.RESET_ALL}")
            return 1
    
    if args.template:
        if os.path.exists(args.template):
            try:
                with open(args.template, 'r') as f:
                    template = f.read()
                update_params["template"] = template
            except Exception as e:
                print(f"{Fore.RED}Error reading template file: {str(e)}{Style.RESET_ALL}")
                return 1
        else:
            update_params["template"] = args.template
    
    # Update the evalset
    spinner = None
    if not args.quiet and sys.stdout.isatty():
        spinner = FancySpinner()
        spinner.start(f"Updating EvalSet {args.evalset_id}...")
    
    try:
        result = manage_evalset(**update_params)
        
        if spinner:
            spinner.stop()
        
        if result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Result format is consistent for update action
        evalset = result.get("evalset", {})
        
        # Format the output based on the format
        if args.format in ["json", "yaml"]:
            return handle_output(evalset, args)
        
        # Text output
        if args.quiet:
            print(evalset["id"])
        else:
            print(f"{Fore.GREEN}EvalSet updated successfully!{Style.RESET_ALL}")
            print(f"ID: {evalset['id']}")
            print(f"Name: {evalset['name']}")
            print(f"Questions: {len(evalset['questions'])}")
            
            # Show how to view the updated evalset
            print(f"\n{Fore.YELLOW}To view the updated evalset:{Style.RESET_ALL}")
            print(f"  agentoptim evalset get {evalset['id']}")
        
        return 0
    
    except Exception as e:
        if spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_evalset_delete(args):
    """Handle the evalset delete command."""
    from agentoptim.evalset import manage_evalset
    
    # Confirm deletion if not already confirmed
    if not args.confirm and sys.stdout.isatty():
        confirm = input(f"{Fore.YELLOW}Are you sure you want to delete evalset {args.evalset_id}? (y/n): {Style.RESET_ALL}")
        if confirm.lower() not in ['y', 'yes']:
            print(f"{Fore.RED}Deletion cancelled{Style.RESET_ALL}")
            return 0
    
    # Delete the evalset
    spinner = None
    if not args.quiet and sys.stdout.isatty():
        spinner = FancySpinner()
        spinner.start(f"Deleting EvalSet {args.evalset_id}...")
    
    try:
        result = manage_evalset(action="delete", evalset_id=args.evalset_id)
        
        if spinner:
            spinner.stop()
        
        if result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Format the output based on the format
        if args.format in ["json", "yaml"]:
            return handle_output(result, args)
        
        # Text output
        if not args.quiet:
            print(f"{Fore.GREEN}EvalSet {args.evalset_id} deleted successfully!{Style.RESET_ALL}")
        
        return 0
    
    except Exception as e:
        if spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_evalset_command(args):
    """Handle the evalset command."""
    # Route to the appropriate handler based on the action
    if not hasattr(args, 'action') or not args.action:
        print(f"{Fore.RED}Error: No action specified{Style.RESET_ALL}")
        return 1
    
    if args.action == "list":
        return handle_evalset_list(args)
    elif args.action == "get":
        return handle_evalset_get(args)
    elif args.action == "create":
        return handle_evalset_create(args)
    elif args.action == "update":
        return handle_evalset_update(args)
    elif args.action == "delete":
        return handle_evalset_delete(args)
    else:
        print(f"{Fore.RED}Error: Unknown action '{args.action}'{Style.RESET_ALL}")
        return 1
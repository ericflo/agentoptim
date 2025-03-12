"""
Evaluation run commands for the AgentOptim CLI
"""

import os
import json
import logging
import sys
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from colorama import Fore, Style

from agentoptim.cli.core import format_box, FancySpinner
from agentoptim.utils import DATA_DIR

# Set up logger
logger = logging.getLogger(__name__)

def setup_run_parser(subparsers):
    """Set up the run command parser."""
    # Add run command
    run_parser = subparsers.add_parser(
        "run",
        help="Manage evaluation runs",
        aliases=["r"],
        description="Create, list, get, compare, export, and visualize evaluation runs."
    )
    
    # Add run subcommands
    run_subparsers = run_parser.add_subparsers(
        dest="action",
        help="Action to perform"
    )
    
    # List command
    list_parser = run_subparsers.add_parser(
        "list",
        help="List evaluation runs",
        description="List available evaluation runs with pagination and filtering."
    )
    list_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml", "csv"],
        default="text",
        help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--evalset", "-e",
        help="Filter by EvalSet ID"
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
    list_parser.add_argument(
        "--page", "-p",
        type=int,
        default=1,
        help="Page number for pagination (default: 1)"
    )
    list_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Number of items per page (default: 10)"
    )
    
    # Get command
    get_parser = run_subparsers.add_parser(
        "get",
        help="Get details of an evaluation run",
        description="Get detailed results of a specific evaluation run."
    )
    get_parser.add_argument(
        "eval_run_id",
        help="ID of the evaluation run to get (use 'latest' for most recent)"
    )
    get_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml", "html", "csv", "markdown"],
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
    create_parser = run_subparsers.add_parser(
        "create",
        help="Create a new evaluation run",
        description="Evaluate a conversation using a specified EvalSet."
    )
    create_parser.add_argument(
        "evalset_id",
        help="ID of the EvalSet to use for evaluation"
    )
    create_parser.add_argument(
        "conversation_file",
        nargs="?",
        help="Path to conversation JSON file (optional with --interactive)"
    )
    create_parser.add_argument(
        "--model", "-m",
        help="Judge model to use for evaluation"
    )
    create_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Maximum number of parallel evaluations (default: 3)"
    )
    create_parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="Omit reasoning for faster evaluations"
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
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    create_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Create and evaluate conversations interactively"
    )
    create_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        help="Provider to use for evaluations (sets appropriate defaults)"
    )
    create_parser.add_argument(
        "--text", "-t",
        help="Text file containing a raw response to evaluate"
    )
    
    # Delete command
    delete_parser = run_subparsers.add_parser(
        "delete",
        help="Delete an evaluation run",
        description="Delete a specific evaluation run."
    )
    delete_parser.add_argument(
        "eval_run_id",
        help="ID of the evaluation run to delete (use 'latest' for most recent)"
    )
    delete_parser.add_argument(
        "--confirm", "-c",
        action="store_true",
        help="Skip confirmation prompt"
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
    
    # Export command
    export_parser = run_subparsers.add_parser(
        "export",
        help="Export evaluation results in various formats",
        description="Export evaluation results in various formats for reporting and analysis."
    )
    export_parser.add_argument(
        "eval_run_id",
        help="ID of the evaluation run to export (use 'latest' for most recent)"
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml", "html", "csv", "markdown", "pdf"],
        default="text",
        help="Export format (default: text)"
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    export_parser.add_argument(
        "--charts", "-c",
        action="store_true",
        help="Include visualizations in HTML export"
    )
    export_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    
    # Compare command
    compare_parser = run_subparsers.add_parser(
        "compare",
        help="Compare multiple evaluation runs",
        description="Compare results from multiple evaluation runs."
    )
    compare_parser.add_argument(
        "eval_run_ids",
        nargs="+",
        help="List of run IDs to compare (can use 'latest', 'latest-1', etc.)"
    )
    compare_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml", "html"],
        default="text",
        help="Output format (default: text)"
    )
    compare_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)"
    )
    compare_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress headers and formatting"
    )
    compare_parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Include detailed per-question comparisons"
    )
    
    # Set default function
    run_parser.set_defaults(func=handle_run_command)
    
    return run_parser

def debug_print_object(obj, prefix=""):
    """Helper function to print debug information about any object.
    
    This function is useful for debugging but is not called during normal operation.
    """
    pass

def handle_output(data, args):
    """Handle outputting data based on format and destination."""
    # If data is a tuple, it may be from a return value that should be unpacked
    if isinstance(data, tuple) and len(data) > 0:
        # Take the first element of the tuple, which is likely the data we want
        data = data[0]
    
    if args.format == "json":
        if isinstance(data, (dict, list)):
            output = json.dumps(data, indent=2)
        else:
            output = json.dumps({"data": str(data)}, indent=2)
            print(f"{Fore.YELLOW}Warning: Data is not JSON serializable. Wrapping in a string.{Style.RESET_ALL}",
                  file=sys.stderr)
    elif args.format == "yaml":
        try:
            import yaml
            if isinstance(data, (dict, list)):
                output = yaml.dump(data, sort_keys=False)
            else:
                output = yaml.dump({"data": str(data)}, sort_keys=False)
                print(f"{Fore.YELLOW}Warning: Data is not YAML serializable. Wrapping in a string.{Style.RESET_ALL}",
                      file=sys.stderr)
        except ImportError:
            if isinstance(data, (dict, list)):
                output = json.dumps(data, indent=2)
            else:
                output = json.dumps({"data": str(data)}, indent=2)
            print(f"{Fore.YELLOW}Warning: PyYAML not installed. Falling back to JSON.{Style.RESET_ALL}",
                  file=sys.stderr)
    elif args.format == "csv":
        try:
            import csv
            from io import StringIO
            
            output_buffer = StringIO()
            writer = csv.writer(output_buffer)
            
            # Handle different data structures for CSV output
            if isinstance(data, list):
                # List of items
                if data and isinstance(data[0], dict):
                    headers = data[0].keys()
                    writer.writerow(headers)
                    for item in data:
                        writer.writerow([item.get(h, "") for h in headers])
            elif isinstance(data, dict):
                # Dictionary - convert to rows
                writer.writerow(["Key", "Value"])
                for key, value in data.items():
                    writer.writerow([key, value])
            else:
                # Not a structure we can easily convert to CSV
                writer.writerow(["Data"])
                writer.writerow([str(data)])
                print(f"{Fore.YELLOW}Warning: Data is not easily convertible to CSV. Using simple format.{Style.RESET_ALL}",
                      file=sys.stderr)
            
            output = output_buffer.getvalue()
        except Exception as e:
            if isinstance(data, (dict, list)):
                output = json.dumps(data, indent=2)
            else:
                output = json.dumps({"data": str(data)}, indent=2)
            print(f"{Fore.YELLOW}Warning: Error creating CSV: {str(e)}. Falling back to JSON.{Style.RESET_ALL}",
                  file=sys.stderr)
    elif args.format == "html":
        try:
            # Basic HTML template
            html_start = """<!DOCTYPE html>
<html>
<head>
    <title>AgentOptim Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #0066cc; }
        .container { background: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .yes { color: green; }
        .no { color: red; }
        .score { font-weight: bold; font-size: 1.2em; color: #0066cc; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>AgentOptim Evaluation Results</h1>
"""
            html_end = """
</body>
</html>"""
            
            # Convert data to HTML (simple implementation - would be more complex in practice)
            if isinstance(data, (dict, list)):
                content = "<div class='container'><pre>" + json.dumps(data, indent=2) + "</pre></div>"
            else:
                content = f"<div class='container'><pre>{str(data)}</pre></div>"
                print(f"{Fore.YELLOW}Warning: Data is not JSON serializable. Using string representation.{Style.RESET_ALL}",
                      file=sys.stderr)
            
            output = html_start + content + html_end
        except Exception as e:
            if isinstance(data, (dict, list)):
                output = json.dumps(data, indent=2)
            else:
                output = json.dumps({"data": str(data)}, indent=2)
            print(f"{Fore.YELLOW}Warning: Error creating HTML: {str(e)}. Falling back to JSON.{Style.RESET_ALL}",
                  file=sys.stderr)
    elif args.format == "markdown":
        try:
            # Simple markdown conversion
            if isinstance(data, dict):
                if "eval_run" in data:
                    # If the data contains an eval_run key, use that instead
                    data = data["eval_run"]
                
                lines = ["# AgentOptim Evaluation Results", ""]
                
                # Add summary if available
                if "summary" in data:
                    lines.append("## Summary")
                    lines.append("")
                    summary = data["summary"]
                    for key, value in summary.items():
                        lines.append(f"**{key}**: {value}")
                    lines.append("")
                
                # Add details for results if available
                if "results" in data:
                    lines.append("## Detailed Results")
                    lines.append("")
                    for i, result in enumerate(data["results"]):
                        # Handle case where result might be a tuple instead of dict
                        if not isinstance(result, dict):
                            lines.append(f"### Result {i+1}: Data format error (type: {type(result)})")
                            lines.append(f"```\n{str(result)}\n```")
                            lines.append("")
                            continue
                            
                        lines.append(f"### Question {i+1}: {result.get('question', 'N/A')}")
                        lines.append("")
                        lines.append(f"**Judgment**: {'Yes' if result.get('judgment') else 'No'}")
                        lines.append("")
                        
                        # Safely handle confidence which might be None
                        confidence = result.get('confidence')
                        if confidence is not None:
                            lines.append(f"**Confidence**: {confidence:.2f}")
                        else:
                            lines.append("**Confidence**: N/A")
                        lines.append("")
                        
                        lines.append("**Reasoning**:")
                        lines.append("")
                        lines.append(result.get("reasoning", "N/A"))
                        lines.append("")
                
                output = "\n".join(lines)
            elif isinstance(data, list):
                lines = ["# AgentOptim Evaluation Results", "", "## List Results", ""]
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        lines.append(f"### Item {i+1}")
                        for key, value in item.items():
                            lines.append(f"**{key}**: {value}")
                        lines.append("")
                    else:
                        lines.append(f"### Item {i+1}: {str(item)}")
                        lines.append("")
                output = "\n".join(lines)
            else:
                # Fallback for other data types
                output = "# AgentOptim Evaluation Results\n\n```\n" + str(data) + "\n```"
        except Exception as e:
            import traceback
            print(f"{Fore.YELLOW}Warning: Error creating Markdown: {str(e)}\n{traceback.format_exc()}{Style.RESET_ALL}",
                  file=sys.stderr)
            if isinstance(data, (dict, list)):
                output = json.dumps(data, indent=2) 
            else:
                output = str(data)
    else:
        # Text format - return data as is
        output = data if isinstance(data, str) else json.dumps(data, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"{Fore.GREEN}Output written to {args.output}{Style.RESET_ALL}", file=sys.stderr)
        
        # If HTML and on macOS, try to open it
        if args.format == "html" and sys.platform == "darwin" and not args.quiet:
            os.system(f"open {args.output}")
    else:
        print(output)
    
    return 0

def resolve_run_id(run_id):
    """Resolve special run IDs like 'latest' and 'latest-N'."""
    from agentoptim.server import list_eval_runs
    
    if run_id == "latest":
        # Get the most recent run
        runs = list_eval_runs(page=1, page_size=1)
        
        if isinstance(runs, tuple) and len(runs) > 0:
            # We should take the first element which is likely the runs data
            runs = runs[0]
        
        # Handle the response structure based on what we got
        if isinstance(runs, dict):
            # Check for eval_runs in the response
            if "eval_runs" in runs and runs["eval_runs"]:
                if isinstance(runs["eval_runs"][0], dict) and "id" in runs["eval_runs"][0]:
                    return runs["eval_runs"][0]["id"]
            
            # Try other potential keys
            for key in ["items", "eval_runs", "optimization_runs"]:
                if key in runs and runs[key] and isinstance(runs[key][0], dict) and "id" in runs[key][0]:
                    return runs[key][0]["id"]
        
        # Handle when runs is already a list of eval runs
        if isinstance(runs, list) and runs:
            if isinstance(runs[0], dict) and "id" in runs[0]:
                return runs[0]["id"]
        
        raise ValueError("No evaluation runs found")
        
    elif run_id.startswith("latest-"):
        try:
            # Get the Nth most recent run
            n = int(run_id.split("-")[1])
            runs = list_eval_runs(page=1, page_size=n+1)
            
            if isinstance(runs, tuple) and len(runs) > 0:
                # We should take the first element which is likely the runs data
                runs = runs[0]
            
            # Handle when runs is already a list of eval runs
            if isinstance(runs, list):
                if len(runs) <= n:
                    raise ValueError(f"Not enough evaluation runs available for {run_id}")
                
                if isinstance(runs[n], dict) and "id" in runs[n]:
                    return runs[n]["id"]
            
            # Try to find the runs in various possible keys
            eval_runs = None
            if isinstance(runs, dict):
                for key in ["items", "eval_runs", "optimization_runs"]:
                    if key in runs and runs[key]:
                        eval_runs = runs[key]
                        break
            
            if not eval_runs or len(eval_runs) <= n:
                raise ValueError(f"Not enough evaluation runs available for {run_id}")
            
            return eval_runs[n]["id"]
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid run ID format: {run_id}. {str(e)}")
    
    # Regular run ID
    return run_id

def handle_run_list(args):
    """Handle the run list command."""
    from agentoptim.server import manage_eval_runs
    
    # Get the eval runs
    params = {
        "action": "list",
        "page": args.page,
        "page_size": args.limit
    }
    
    if args.evalset:
        params["evalset_id"] = args.evalset
    
    result = manage_eval_runs(**params)
    
    if result.get("error"):
        print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
        return 1
    
    # The eval runs are in result["eval_runs"]
    eval_runs = result.get("eval_runs", [])
    pagination = result.get("pagination", {})
    
    # Format the output based on the format
    if args.format in ["json", "yaml", "csv"]:
        return handle_output(eval_runs, args)
    
    # Text output
    if not eval_runs:
        if not args.quiet:
            print(f"{Fore.YELLOW}No evaluation runs found{Style.RESET_ALL}")
        return 0
    
    # Try to use rich for fancy tables if available
    try:
        if not args.quiet:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            
            table = Table(title=f"Evaluation Runs (Page {pagination.get('page', 1)} of {pagination.get('total_pages', 1)})")
            table.add_column("ID", style="cyan")
            table.add_column("EvalSet", style="green")
            table.add_column("Score", style="magenta")
            table.add_column("Date", style="yellow")
            
            for run in eval_runs:
                score = run.get("summary", {}).get("yes_percentage", 0)
                score_str = f"{score:.1f}%"
                
                timestamp = run.get("timestamp", "")
                if timestamp:
                    try:
                        # Format timestamp nicely
                        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = timestamp
                else:
                    date_str = ""
                
                table.add_row(
                    str(run.get("id", "")),
                    str(run.get("evalset_name", "")),
                    str(score_str),
                    str(date_str)
                )
            
            console.print(table)
            
            # Show pagination info
            if pagination.get("total_pages", 1) > 1:
                console.print(f"Page {pagination.get('page', 1)} of {pagination.get('total_pages', 1)} • {pagination.get('total_count', 0)} total runs")
                if pagination.get("page", 1) < pagination.get("total_pages", 1):
                    console.print(f"Use --page {pagination.get('page', 1) + 1} to see the next page")
        else:
            # Quiet mode - just print IDs
            for run in eval_runs:
                print(run.get("id", ""))
    except ImportError:
        # Fallback to simple table
        if not args.quiet:
            print(f"{Fore.CYAN}{'ID':<36} {'EvalSet':<20} {'Score':>7} {'Date':<19}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*36} {'='*20} {'='*7} {'='*19}{Style.RESET_ALL}")
            
            for run in eval_runs:
                score = run.get("summary", {}).get("yes_percentage", 0)
                score_str = f"{score:.1f}%"
                
                timestamp = run.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        date_str = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = timestamp
                else:
                    date_str = ""
                
                print(f"{run.get('id', ''):<36} {run.get('evalset_name', ''):<20} {score_str:>7} {date_str:<19}")
            
            # Show pagination info
            if pagination.get("total_pages", 1) > 1:
                print(f"\nPage {pagination.get('page', 1)} of {pagination.get('total_pages', 1)} • {pagination.get('total_count', 0)} total runs")
                if pagination.get("page", 1) < pagination.get("total_pages", 1):
                    print(f"Use --page {pagination.get('page', 1) + 1} to see the next page")
        else:
            # Quiet mode - just print IDs
            for run in eval_runs:
                print(run.get("id", ""))
    
    return 0

def handle_run_get(args):
    """Handle the run get command."""
    from agentoptim.server import manage_eval_runs
    
    try:
        # Resolve run ID (handles 'latest')
        run_id = resolve_run_id(args.eval_run_id)
        
        # Get the eval run
        result = manage_eval_runs(action="get", eval_run_id=run_id)
        
        # Handle the case where result might be a tuple (function returning multiple values)
        if isinstance(result, tuple) and len(result) > 0:
            result = result[0]
        
        if isinstance(result, dict) and result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Extract the eval run data from the response
        if isinstance(result, dict):
            eval_run = result.get("eval_run", result)  # Use result itself if no eval_run key
        else:
            eval_run = result  # Just use the result as is if it's not a dict
        
        # Format the output based on the format
        if args.format in ["json", "yaml", "csv", "html", "markdown"]:
            return handle_output(eval_run, args)
        
        # Text output
        if args.quiet:
            # Just print the ID in quiet mode
            print(eval_run.get("id", ""))
            return 0
        
        # Try to use rich for fancy output
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.markdown import Markdown
            
            console = Console()
            
            # Summary section
            summary = eval_run.get("summary", {})
            yes_percentage = summary.get("yes_percentage", 0)
            yes_count = summary.get("yes_count", 0)
            no_count = summary.get("no_count", 0)
            total = yes_count + no_count
            
            summary_text = f"Score: {yes_percentage:.1f}%\n"
            summary_text += f"Yes: {yes_count}/{total}\n"
            summary_text += f"No: {no_count}/{total}\n"
            summary_text += f"EvalSet: {eval_run.get('evalset_name', 'Unknown')}\n"
            summary_text += f"ID: {eval_run.get('id', 'Unknown')}"
            
            console.print(Panel(summary_text, title="Evaluation Summary", border_style="green"))
            
            # Results table
            results = eval_run.get("results", [])
            if results:
                table = Table(title="Evaluation Results", show_lines=True)
                table.add_column("Question", style="cyan")
                table.add_column("Judgment", style="green")
                table.add_column("Confidence", style="yellow")
                
                for result in results:
                    judgment = result.get("judgment", False)
                    judgment_str = "[green]Yes[/green]" if judgment else "[red]No[/red]"
                    confidence = result.get("confidence", 0)
                    confidence_str = f"{confidence:.2f}"
                    
                    table.add_row(
                        str(result.get("question", "")),
                        judgment_str,
                        str(confidence_str)
                    )
                
                console.print(table)
                
                # Detailed reasoning
                for i, result in enumerate(results):
                    if result.get("reasoning"):
                        console.print(Panel(
                            result.get("reasoning", ""),
                            title=f"Question {i+1}: {result.get('question', '')}",
                            border_style="cyan" if result.get("judgment", False) else "red"
                        ))
            
            # Conversation
            conversation = eval_run.get("conversation", [])
            if conversation:
                console.print(Panel("Conversation", title="Evaluated Conversation", border_style="blue"))
                
                for msg in conversation:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        console.print(f"[blue]System:[/blue] {content}")
                    elif role == "user":
                        console.print(f"[green]User:[/green] {content}")
                    elif role == "assistant":
                        console.print(f"[yellow]Assistant:[/yellow] {content}")
                    else:
                        console.print(f"[gray]{role}:[/gray] {content}")
        
        except ImportError:
            # Fallback to simple formatted output
            # Summary section
            summary = eval_run.get("summary", {})
            yes_percentage = summary.get("yes_percentage", 0)
            yes_count = summary.get("yes_count", 0)
            no_count = summary.get("no_count", 0)
            total = yes_count + no_count
            
            summary_text = f"Score: {yes_percentage:.1f}%\n"
            summary_text += f"Yes: {yes_count}/{total}\n"
            summary_text += f"No: {no_count}/{total}\n"
            summary_text += f"EvalSet: {eval_run.get('evalset_name', 'Unknown')}\n"
            summary_text += f"ID: {eval_run.get('id', 'Unknown')}"
            
            print(format_box("Evaluation Summary", summary_text, border_color=Fore.GREEN))
            
            # Results
            results = eval_run.get("results", [])
            if results:
                print(format_box("Evaluation Results", "", border_color=Fore.CYAN))
                
                for i, result in enumerate(results):
                    judgment = result.get("judgment", False)
                    judgment_str = f"{Fore.GREEN}Yes{Style.RESET_ALL}" if judgment else f"{Fore.RED}No{Style.RESET_ALL}"
                    confidence = result.get("confidence", 0)
                    
                    print(f"{i+1}. {result.get('question', '')}")
                    print(f"   Judgment: {judgment_str}")
                    print(f"   Confidence: {confidence:.2f}")
                    print(f"   Reasoning: {result.get('reasoning', '')}")
                    print()
            
            # Conversation
            conversation = eval_run.get("conversation", [])
            if conversation:
                print(format_box("Evaluated Conversation", "", border_color=Fore.BLUE))
                
                for msg in conversation:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        print(f"{Fore.BLUE}System:{Style.RESET_ALL} {content}")
                    elif role == "user":
                        print(f"{Fore.GREEN}User:{Style.RESET_ALL} {content}")
                    elif role == "assistant":
                        print(f"{Fore.YELLOW}Assistant:{Style.RESET_ALL} {content}")
                    else:
                        print(f"{role}: {content}")
                    print()
        
        return 0
    
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def create_conversation_from_text(text_path):
    """Create a conversation object from a text file."""
    try:
        with open(text_path, 'r') as f:
            content = f.read()
            
        # Create a simple conversation with the text as the assistant's response
        conversation = [
            {"role": "user", "content": "Please provide a response."},
            {"role": "assistant", "content": content}
        ]
        
        return conversation
    except Exception as e:
        raise ValueError(f"Error reading text file: {str(e)}")

def create_conversation_interactively():
    """Create a conversation interactively."""
    print(f"{Fore.CYAN}=== Interactive Conversation Creation ==={Style.RESET_ALL}")
    print("Create a conversation to evaluate.")
    print("Enter messages one at a time, press Ctrl+D (or Ctrl+Z on Windows) when done.\n")
    
    conversation = []
    
    # Ask about system message
    include_system = input(f"{Fore.CYAN}Include a system message? (y/n):{Style.RESET_ALL} ").lower() in ['y', 'yes']
    
    if include_system:
        print(f"{Fore.CYAN}Enter system message (press Enter when done):{Style.RESET_ALL}")
        system_lines = []
        while True:
            try:
                line = input()
                system_lines.append(line)
            except EOFError:
                break
            
        system_content = "\n".join(system_lines)
        conversation.append({"role": "system", "content": system_content})
    
    # Add user/assistant messages
    current_role = "user"  # Start with user
    message_num = 1
    
    while True:
        try:
            print(f"{Fore.GREEN if current_role == 'user' else Fore.YELLOW}{current_role.capitalize()} message {message_num if current_role == 'user' else message_num-1}:{Style.RESET_ALL}")
            content_lines = []
            while True:
                line = input()
                if not line and not content_lines:
                    continue  # Skip empty first line
                if not line and content_lines:
                    break  # End message on empty line after content
                content_lines.append(line)
            
            content = "\n".join(content_lines)
            conversation.append({"role": current_role, "content": content})
            
            # Switch roles
            if current_role == "user":
                current_role = "assistant"
            else:
                current_role = "user"
                message_num += 1
            
            # Check if done
            if current_role == "user":
                more = input(f"{Fore.CYAN}Add another message pair? (y/n):{Style.RESET_ALL} ").lower() in ['y', 'yes']
                if not more:
                    break
        except EOFError:
            break
    
    # Ensure conversation ends with assistant message
    if conversation and conversation[-1]["role"] == "user":
        print(f"{Fore.RED}Error: Conversation must end with an assistant message{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Enter final assistant message:{Style.RESET_ALL}")
        content_lines = []
        while True:
            try:
                line = input()
                if not line and content_lines:
                    break
                content_lines.append(line)
            except EOFError:
                break
            
        content = "\n".join(content_lines)
        conversation.append({"role": "assistant", "content": content})
    
    # Show summary
    print(f"\n{Fore.CYAN}=== Conversation Summary ==={Style.RESET_ALL}")
    for msg in conversation:
        role_color = Fore.BLUE if msg["role"] == "system" else (Fore.GREEN if msg["role"] == "user" else Fore.YELLOW)
        content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"{role_color}{msg['role'].capitalize()}:{Style.RESET_ALL} {content_preview}")
    
    # Confirm
    confirm = input(f"\n{Fore.CYAN}Evaluate this conversation? (y/n):{Style.RESET_ALL} ").lower() in ['y', 'yes']
    if not confirm:
        return None
    
    return conversation

def handle_run_create(args):
    """Handle the run create command."""
    from agentoptim.server import run_evalset, save_eval_run, EvalRun
    
    # Get the conversation
    conversation = None
    
    try:
        if args.interactive:
            # Create conversation interactively
            conversation = create_conversation_interactively()
            if conversation is None:
                print(f"{Fore.RED}Evaluation cancelled{Style.RESET_ALL}")
                return 1
        elif args.text:
            # Create conversation from text file
            conversation = create_conversation_from_text(args.text)
        elif args.conversation_file:
            # Read conversation from file
            with open(args.conversation_file, 'r') as f:
                conversation = json.load(f)
        else:
            print(f"{Fore.RED}Error: No conversation provided. Use --interactive, --text, or provide a conversation file.{Style.RESET_ALL}")
            return 1
        
        # Set up parameters
        evalset_id = args.evalset_id
        
        # Determine judge model
        judge_model = args.model
        # Configure environment variables for provider
        if args.provider:
            if args.provider == "openai" and not args.model:
                # If no model specified, use OpenAI default
                judge_model = "gpt-4o-mini"
            elif args.provider == "anthropic" and not args.model:
                # If no model specified, use Anthropic default
                judge_model = "claude-3-5-haiku-20240307"
        
        # Run the evaluation
        spinner = None
        if not args.quiet and sys.stdout.isatty():
            spinner = FancySpinner()
            spinner.start(f"Evaluating conversation with EvalSet {evalset_id}...")
        
        try:
            # This is an async function, so we need to run it in an event loop
            import asyncio
            # First, make sure the EvalSet exists
            from agentoptim.evalset import get_evalset
            evalset = get_evalset(evalset_id)
            if not evalset:
                print(f"{Fore.RED}Error: EvalSet with ID '{evalset_id}' not found{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Hint: Use 'agentoptim evalset list' to list available EvalSets{Style.RESET_ALL}")
                return 1
                
            # Now run the evaluation
            result = asyncio.run(run_evalset(
                evalset_id=evalset_id,
                conversation=conversation,
                judge_model=judge_model,
                max_parallel=args.concurrency,
                omit_reasoning=args.brief
            ))
            
            if spinner:
                spinner.stop()
            
            if result.get("error"):
                print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
                return 1
                
            # Create EvalRun object from the result
            eval_run = EvalRun(
                evalset_id=result.get("evalset_id"),
                evalset_name=result.get("evalset_name"),
                judge_model=result.get("judge_model"),
                results=result.get("results", []),
                conversation=conversation,
                summary=result.get("summary", {})
            )
            
            # Save to disk
            save_success = save_eval_run(eval_run)
            if not save_success:
                print(f"{Fore.YELLOW}Warning: Failed to save evaluation run{Style.RESET_ALL}")
                
            # Add the run ID to the result for future reference
            result["id"] = eval_run.id
            
        except Exception as e:
            if spinner:
                spinner.stop()
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return 1
        
        # Format the output based on the format
        if args.format in ["json", "yaml"]:
            return handle_output(result, args)
        
        # Text output for success
        if args.quiet:
            # Just print the ID in quiet mode
            print(result.get("id", ""))
        else:
            # Show summary info
            summary = result.get("summary", {})
            yes_percentage = summary.get("yes_percentage", 0)
            yes_count = summary.get("yes_count", 0)
            no_count = summary.get("no_count", 0)
            total = yes_count + no_count
            
            print(f"{Fore.GREEN}Evaluation complete!{Style.RESET_ALL}")
            print(f"Score: {yes_percentage:.1f}%")
            print(f"Yes: {yes_count}/{total}")
            print(f"No: {no_count}/{total}")
            print(f"ID: {result.get('id', 'Unknown')}")
            print(f"\n{Fore.YELLOW}To view detailed results:{Style.RESET_ALL}")
            print(f"  agentoptim run get {result.get('id', '')}")
            print(f"\n{Fore.YELLOW}To export as HTML report:{Style.RESET_ALL}")
            print(f"  agentoptim run export {result.get('id', '')} --format html --output report.html")
        
        return 0
    
    except Exception as e:
        if 'spinner' in locals() and spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_run_delete(args):
    """Handle the run delete command."""
    from agentoptim.server import manage_eval_runs
    
    try:
        # Resolve run ID (handles 'latest')
        run_id = resolve_run_id(args.eval_run_id)
        
        # Confirm deletion if not already confirmed
        if not args.confirm and sys.stdout.isatty():
            confirm = input(f"{Fore.YELLOW}Are you sure you want to delete evaluation run {run_id}? (y/n): {Style.RESET_ALL}")
            if confirm.lower() not in ['y', 'yes']:
                print(f"{Fore.RED}Deletion cancelled{Style.RESET_ALL}")
                return 0
        
        # Delete the eval run
        spinner = None
        if not args.quiet and sys.stdout.isatty():
            spinner = FancySpinner()
            spinner.start(f"Deleting evaluation run {run_id}...")
        
        result = manage_eval_runs(action="delete", eval_run_id=run_id)
        
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
            print(f"{Fore.GREEN}Evaluation run {run_id} deleted successfully!{Style.RESET_ALL}")
        
        return 0
    
    except Exception as e:
        if 'spinner' in locals() and spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_run_export(args):
    """Handle the run export command."""
    from agentoptim.server import manage_eval_runs
    
    try:
        # Resolve run ID (handles 'latest')
        run_id = resolve_run_id(args.eval_run_id)
        
        # Get the eval run
        result = manage_eval_runs(action="get", eval_run_id=run_id)
        
        if result.get("error"):
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Extract the eval run data from the response
        eval_run = result.get("eval_run", {})
        
        # Use the handle_output function to format and output the data
        # Format will determine the export type
        args.format = args.format or "text"
        
        # Handle special case of PDF export
        if args.format == "pdf":
            try:
                import tempfile
                
                # First export as HTML
                html_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                html_path = html_file.name
                html_file.close()
                
                # Create HTML version
                html_args = argparse.Namespace(
                    format="html",
                    output=html_path,
                    quiet=True,
                    charts=args.charts
                )
                handle_output(eval_run, html_args)
                
                # Convert HTML to PDF
                try:
                    # Try with weasyprint
                    from weasyprint import HTML
                    pdf_data = HTML(filename=html_path).write_pdf()
                    
                    if args.output:
                        with open(args.output, "wb") as f:
                            f.write(pdf_data)
                        print(f"{Fore.GREEN}PDF exported to {args.output}{Style.RESET_ALL}", file=sys.stderr)
                        
                        # Try to open the PDF on supported platforms
                        if sys.platform == "darwin" and not args.quiet:
                            os.system(f"open {args.output}")
                    else:
                        print(f"{Fore.RED}Error: PDF export requires an output file path.{Style.RESET_ALL}")
                        return 1
                except ImportError:
                    # Fallback to wkhtmltopdf if available
                    if args.output:
                        import subprocess
                        try:
                            subprocess.run(["wkhtmltopdf", html_path, args.output], check=True)
                            print(f"{Fore.GREEN}PDF exported to {args.output}{Style.RESET_ALL}", file=sys.stderr)
                            
                            # Try to open the PDF on supported platforms
                            if sys.platform == "darwin" and not args.quiet:
                                os.system(f"open {args.output}")
                        except FileNotFoundError:
                            print(f"{Fore.RED}Error: PDF export requires either weasyprint or wkhtmltopdf.{Style.RESET_ALL}")
                            print(f"{Fore.YELLOW}Install with: pip install weasyprint{Style.RESET_ALL}")
                            return 1
                        except subprocess.CalledProcessError as e:
                            print(f"{Fore.RED}Error generating PDF: {str(e)}{Style.RESET_ALL}")
                            return 1
                    else:
                        print(f"{Fore.RED}Error: PDF export requires an output file path.{Style.RESET_ALL}")
                        return 1
                
                # Clean up temp file
                try:
                    os.unlink(html_path)
                except:
                    pass
                
                return 0
            except Exception as e:
                print(f"{Fore.RED}Error exporting to PDF: {str(e)}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Try exporting as HTML instead:{Style.RESET_ALL}")
                print(f"  agentoptim run export {run_id} --format html --output report.html")
                return 1
        
        # Handle standard formats
        return handle_output(eval_run, args)
    
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def handle_run_compare(args):
    """Handle the run compare command."""
    from agentoptim.server import manage_eval_runs
    
    try:
        # Resolve all run IDs
        run_ids = [resolve_run_id(run_id) for run_id in args.eval_run_ids]
        
        if len(run_ids) < 2:
            print(f"{Fore.RED}Error: At least two evaluation run IDs are required for comparison{Style.RESET_ALL}")
            return 1
        
        # Create a spinner if appropriate
        spinner = None
        if not args.quiet and sys.stdout.isatty():
            spinner = FancySpinner()
            spinner.start(f"Comparing evaluation runs...")
        
        # Get all eval runs
        runs = []
        for run_id in run_ids:
            result = manage_eval_runs(action="get", eval_run_id=run_id)
            if result.get("error"):
                if spinner:
                    spinner.stop()
                print(f"{Fore.RED}Error retrieving run {run_id}: {result.get('error')}{Style.RESET_ALL}")
                return 1
            
            runs.append(result.get("eval_run", {}))
        
        if spinner:
            spinner.stop()
        
        # Prepare comparison data
        comparison = {
            "runs": runs,
            "summary": {}
        }
        
        # Calculate summary metrics
        for i, run in enumerate(runs):
            summary = run.get("summary", {})
            comparison["summary"][f"Run {i+1}"] = {
                "id": run.get("id", ""),
                "evalset_name": run.get("evalset_name", ""),
                "yes_percentage": summary.get("yes_percentage", 0),
                "yes_count": summary.get("yes_count", 0),
                "no_count": summary.get("no_count", 0)
            }
        
        # Add detailed comparison if requested
        if args.detailed:
            comparison["details"] = []
            
            # Get all questions from all runs
            questions = set()
            for run in runs:
                for result in run.get("results", []):
                    questions.add(result.get("question", ""))
            
            # Create a comparison for each question
            for question in sorted(questions):
                question_comparison = {
                    "question": question,
                    "judgments": []
                }
                
                for run in runs:
                    # Find the result for this question in this run
                    result = next((r for r in run.get("results", []) if r.get("question") == question), None)
                    
                    if result:
                        question_comparison["judgments"].append({
                            "run_id": run.get("id", ""),
                            "judgment": result.get("judgment", False),
                            "confidence": result.get("confidence", 0),
                            "reasoning": result.get("reasoning", "")
                        })
                    else:
                        question_comparison["judgments"].append({
                            "run_id": run.get("id", ""),
                            "judgment": None,
                            "confidence": 0,
                            "reasoning": "Question not evaluated in this run"
                        })
                
                comparison["details"].append(question_comparison)
        
        # Format the output based on the format
        if args.format in ["json", "yaml"]:
            return handle_output(comparison, args)
        elif args.format == "html":
            # Create a more detailed HTML comparison
            try:
                html_content = generate_html_comparison(comparison, with_charts=True)
                
                if args.output:
                    with open(args.output, "w") as f:
                        f.write(html_content)
                    print(f"{Fore.GREEN}Comparison exported to {args.output}{Style.RESET_ALL}", file=sys.stderr)
                    
                    # Try to open the HTML file on supported platforms
                    if sys.platform == "darwin" and not args.quiet:
                        os.system(f"open {args.output}")
                else:
                    print(html_content)
                
                return 0
            except Exception as e:
                print(f"{Fore.RED}Error generating HTML comparison: {str(e)}{Style.RESET_ALL}")
                # Fall back to text comparison
        
        # Text comparison (default)
        if not args.quiet:
            try:
                from rich.console import Console
                from rich.table import Table
                from rich.panel import Panel
                
                console = Console()
                
                # Create summary table
                table = Table(title="Evaluation Run Comparison")
                table.add_column("Metric", style="cyan")
                
                # Add columns for each run
                for i, run_id in enumerate(run_ids):
                    run = runs[i]
                    table.add_column(f"Run {i+1}: {run.get('evalset_name', '')}", style="green")
                
                # Add rows for metrics
                table.add_row(str("ID"), *[str(run.get("id", "")) for run in runs])
                table.add_row(str("EvalSet"), *[str(run.get("evalset_name", "")) for run in runs])
                table.add_row(
                    str("Score"),
                    *[str(f"{run.get('summary', {}).get('yes_percentage', 0):.1f}%") for run in runs]
                )
                table.add_row(
                    str("Yes/Total"),
                    *[str(f"{run.get('summary', {}).get('yes_count', 0)}/{run.get('summary', {}).get('yes_count', 0) + run.get('summary', {}).get('no_count', 0)}") for run in runs]
                )
                
                console.print(table)
                
                # Show detailed comparison if requested
                if args.detailed and "details" in comparison:
                    console.print("\nDetailed Question Comparison", style="bold cyan")
                    
                    for detail in comparison["details"]:
                        question = detail["question"]
                        judgments = detail["judgments"]
                        
                        # Create a panel for each question
                        content = []
                        for i, judgment in enumerate(judgments):
                            judgment_text = "[green]Yes[/green]" if judgment["judgment"] else "[red]No[/red]" if judgment["judgment"] is not None else "[yellow]N/A[/yellow]"
                            content.append(f"Run {i+1}: {judgment_text} (Confidence: {judgment['confidence']:.2f})")
                            
                            if args.detailed:
                                content.append(f"Reasoning: {judgment['reasoning']}\n")
                        
                        console.print(Panel("\n".join(content), title=question, border_style="cyan"))
            
            except ImportError:
                # Fallback to simple text format
                print(f"{Fore.CYAN}=== Evaluation Run Comparison ==={Style.RESET_ALL}")
                
                # Print summary table
                print(f"\n{Fore.CYAN}Summary:{Style.RESET_ALL}")
                print(f"{'Metric':<15} " + " ".join([f"{'Run ' + str(i+1):<30}" for i in range(len(runs))]))
                print("-" * (15 + 31 * len(runs)))
                
                # Print ID row
                print(f"{'ID':<15} " + " ".join([f"{run.get('id', '')[:28]:<30}" for run in runs]))
                
                # Print EvalSet row
                print(f"{'EvalSet':<15} " + " ".join([f"{run.get('evalset_name', '')[:28]:<30}" for run in runs]))
                
                # Print Score row
                scores = [f"{run.get('summary', {}).get('yes_percentage', 0):.1f}%" for run in runs]
                print(f"{'Score':<15} " + " ".join([f"{score:<30}" for score in scores]))
                
                # Print Yes/Total row
                yes_totals = [f"{run.get('summary', {}).get('yes_count', 0)}/{run.get('summary', {}).get('yes_count', 0) + run.get('summary', {}).get('no_count', 0)}" for run in runs]
                print(f"{'Yes/Total':<15} " + " ".join([f"{yt:<30}" for yt in yes_totals]))
                
                # Show detailed comparison if requested
                if args.detailed and "details" in comparison:
                    print(f"\n{Fore.CYAN}Detailed Question Comparison:{Style.RESET_ALL}")
                    
                    for detail in comparison["details"]:
                        question = detail["question"]
                        judgments = detail["judgments"]
                        
                        print(f"\n{Fore.CYAN}Question:{Style.RESET_ALL} {question}")
                        
                        for i, judgment in enumerate(judgments):
                            judgment_text = f"{Fore.GREEN}Yes{Style.RESET_ALL}" if judgment["judgment"] else f"{Fore.RED}No{Style.RESET_ALL}" if judgment["judgment"] is not None else f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
                            print(f"Run {i+1}: {judgment_text} (Confidence: {judgment['confidence']:.2f})")
                            
                            if args.detailed:
                                print(f"Reasoning: {judgment['reasoning']}")
                                print()
        else:
            # Quiet mode - just print the run IDs
            for run in runs:
                print(run.get("id", ""))
        
        return 0
    
    except Exception as e:
        if 'spinner' in locals() and spinner:
            spinner.stop()
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

def generate_html_comparison(comparison, with_charts=False):
    """Generate HTML comparison of evaluation runs."""
    runs = comparison.get("runs", [])
    summary = comparison.get("summary", {})
    details = comparison.get("details", [])
    
    # Basic HTML template
    html = """<!DOCTYPE html>
<html>
<head>
    <title>AgentOptim Evaluation Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        h1, h2, h3 { color: #0066cc; }
        .container { background: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .yes { color: green; }
        .no { color: red; }
        .na { color: #999; }
        .score { font-weight: bold; font-size: 1.2em; color: #0066cc; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; white-space: pre-wrap; }
        .question { background: #e6f7ff; padding: 10px; border-left: 4px solid #0066cc; margin: 15px 0; }
    </style>
"""

    # Add Chart.js for visualizations if needed
    if with_charts:
        html += """
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
"""
    
    html += """
</head>
<body>
    <h1>AgentOptim Evaluation Comparison</h1>
    
    <div class="container">
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
"""

    # Add headers for each run
    for i in range(len(runs)):
        run_name = f"Run {i+1}"
        html += f"                <th>{run_name}</th>\n"
    
    html += """
            </tr>
            <tr>
                <td>ID</td>
"""

    # Add IDs
    for run in runs:
        html += f"                <td>{run.get('id', '')}</td>\n"
    
    html += """
            </tr>
            <tr>
                <td>EvalSet</td>
"""

    # Add EvalSet names
    for run in runs:
        html += f"                <td>{run.get('evalset_name', '')}</td>\n"
    
    html += """
            </tr>
            <tr>
                <td>Score</td>
"""

    # Add scores
    for run in runs:
        score = run.get('summary', {}).get('yes_percentage', 0)
        html += f"                <td class='score'>{score:.1f}%</td>\n"
    
    html += """
            </tr>
            <tr>
                <td>Yes/Total</td>
"""

    # Add Yes/Total counts
    for run in runs:
        summary = run.get('summary', {})
        yes = summary.get('yes_count', 0)
        no = summary.get('no_count', 0)
        total = yes + no
        html += f"                <td>{yes}/{total}</td>\n"
    
    html += """
            </tr>
        </table>
    </div>
"""

    # Add chart if requested
    if with_charts:
        html += """
    <div class="container">
        <h2>Score Comparison</h2>
        <canvas id="scoreChart" width="400" height="200"></canvas>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const ctx = document.getElementById('scoreChart').getContext('2d');
                const myChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: [
"""
        
        # Add labels (run names)
        for i in range(len(runs)):
            run_name = f"Run {i+1}"
            html += f"                            '{run_name}',\n"
        
        html += """
                        ],
                        datasets: [{
                            label: 'Score (%)',
                            data: [
"""
        
        # Add data points (scores)
        for run in runs:
            score = run.get('summary', {}).get('yes_percentage', 0)
            html += f"                                {score:.1f},\n"
        
        html += """
                            ],
                            backgroundColor: [
"""
        
        # Add colors
        colors = ["'rgba(54, 162, 235, 0.2)'", "'rgba(255, 99, 132, 0.2)'", "'rgba(255, 206, 86, 0.2)'", "'rgba(75, 192, 192, 0.2)'"]
        for i in range(len(runs)):
            color_idx = i % len(colors)
            html += f"                                {colors[color_idx]},\n"
        
        html += """
                            ],
                            borderColor: [
"""
        
        # Add border colors
        border_colors = ["'rgba(54, 162, 235, 1)'", "'rgba(255, 99, 132, 1)'", "'rgba(255, 206, 86, 1)'", "'rgba(75, 192, 192, 1)'"]
        for i in range(len(runs)):
            color_idx = i % len(border_colors)
            html += f"                                {border_colors[color_idx]},\n"
        
        html += """
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            });
        </script>
    </div>
"""

    # Add detailed comparison if available
    if details:
        html += """
    <div class="container">
        <h2>Detailed Question Comparison</h2>
"""
        
        for detail in details:
            question = detail.get("question", "")
            judgments = detail.get("judgments", [])
            
            html += f"""
        <div class="question">
            <h3>{question}</h3>
            <table>
                <tr>
                    <th>Run</th>
                    <th>Judgment</th>
                    <th>Confidence</th>
                    <th>Reasoning</th>
                </tr>
"""
            
            for i, judgment in enumerate(judgments):
                judgment_val = judgment.get("judgment")
                judgment_class = "yes" if judgment_val else "no" if judgment_val is not None else "na"
                judgment_text = "Yes" if judgment_val else "No" if judgment_val is not None else "N/A"
                confidence = judgment.get("confidence", 0)
                reasoning = judgment.get("reasoning", "")
                
                html += f"""
                <tr>
                    <td>Run {i+1}</td>
                    <td class="{judgment_class}">{judgment_text}</td>
                    <td>{confidence:.2f}</td>
                    <td>{reasoning}</td>
                </tr>
"""
            
            html += """
            </table>
        </div>
"""
        
    html += """
    </div>
</body>
</html>
"""
    
    return html

def handle_run_command(args):
    """Handle the run command."""
    # Route to the appropriate handler based on the action
    if not hasattr(args, 'action') or not args.action:
        print(f"{Fore.RED}Error: No action specified{Style.RESET_ALL}")
        return 1
    
    if args.action == "list":
        return handle_run_list(args)
    elif args.action == "get":
        return handle_run_get(args)
    elif args.action == "create":
        return handle_run_create(args)
    elif args.action == "delete":
        return handle_run_delete(args)
    elif args.action == "export":
        return handle_run_export(args)
    elif args.action == "compare":
        return handle_run_compare(args)
    else:
        print(f"{Fore.RED}Error: Unknown action '{args.action}'{Style.RESET_ALL}")
        return 1
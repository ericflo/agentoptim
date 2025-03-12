"""Command-line interface for system message optimization."""

import os
import sys
import json
import asyncio
import argparse
import logging
import time
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path

from colorama import Fore, Back, Style, init as init_colorama
from agentoptim.cli.core import FancySpinner

from agentoptim.utils import DATA_DIR, ensure_data_directories
from agentoptim.sysopt.core import (
    manage_optimization_runs,
    get_optimization_run,
    list_optimization_runs,
    get_sysopt_stats,
    self_optimize_generator,
    SystemMessageCandidate,
    SystemMessageGenerator,
    OptimizationRun,
    MAX_CANDIDATES,
    DIVERSITY_LEVELS,
    DEFAULT_NUM_CANDIDATES,
)

# Initialize colorama
init_colorama()

# Configure logging
logger = logging.getLogger(__name__)

def optimize_setup_parser(subparsers):
    """Set up the parser for the optimize command."""
    # Create the optimize command
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize system messages for user queries",
        aliases=["opt", "o"],
        description="Optimize, evaluate, and rank system messages for optimal performance."
    )
    
    # Add subcommands to optimize
    optimize_subparsers = optimize_parser.add_subparsers(
        dest="action",
        help="Action to perform"
    )
    
    # Create (optimize a system message)
    create_parser = optimize_subparsers.add_parser(
        "create",
        help="Create a new system message optimization run",
        description="Generate and evaluate multiple system messages for a user query to find the best one."
    )
    create_parser.add_argument(
        "evalset_id",
        help="ID of the evaluation set to use for testing system messages"
    )
    create_parser.add_argument(
        "user_message",
        nargs="?",
        help="User message to optimize system messages for (if not provided, will prompt)"
    )
    create_parser.add_argument(
        "--base", "-b",
        help="Base system message to use as a starting point"
    )
    create_parser.add_argument(
        "--num-candidates", "-n",
        type=int,
        default=DEFAULT_NUM_CANDIDATES,
        help=f"Number of system message candidates to generate (default: {DEFAULT_NUM_CANDIDATES}, max: {MAX_CANDIDATES})"
    )
    create_parser.add_argument(
        "--diversity", "-d",
        choices=DIVERSITY_LEVELS,
        default="medium",
        help="Diversity level for generated candidates (default: medium)"
    )
    create_parser.add_argument(
        "--generator", "-g",
        default="default",
        help="ID of the generator to use (default: default)"
    )
    create_parser.add_argument(
        "--model", "-m",
        help="Model to use for generation and evaluation"
    )
    create_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        help="Provider to use for evaluations (sets appropriate defaults)"
    )
    create_parser.add_argument(
        "--instructions", "-i",
        help="Additional instructions for the generator"
    )
    create_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Maximum number of parallel operations (default: 3)"
    )
    create_parser.add_argument(
        "--output", "-o",
        help="Output file to write results to"
    )
    create_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format (default: text)"
    )
    create_parser.add_argument(
        "--self-optimize", "-s",
        action="store_true",
        help="Trigger self-optimization of the generator after optimization"
    )
    create_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode for inputting user message"
    )
    
    # Get (retrieve an optimization run)
    get_parser = optimize_subparsers.add_parser(
        "get",
        help="Get details of a specific optimization run",
        description="Retrieve and display details of a previous system message optimization run."
    )
    get_parser.add_argument(
        "optimization_run_id",
        help="ID of the optimization run to retrieve"
    )
    get_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "markdown", "html"],
        default="text",
        help="Output format (default: text)"
    )
    get_parser.add_argument(
        "--output", "-o",
        help="Output file to write results to"
    )
    get_parser.add_argument(
        "--generate-response", "-g",
        action="store_true",
        help="Generate a sample response using the best system message"
    )
    get_parser.add_argument(
        "--model", "-m",
        help="Model to use for generating sample response"
    )
    get_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        help="Provider to use for generating response (sets appropriate defaults)"
    )
    
    # List (list optimization runs)
    list_parser = optimize_subparsers.add_parser(
        "list",
        help="List all optimization runs",
        description="List all previous system message optimization runs with filtering and pagination."
    )
    list_parser.add_argument(
        "--evalset-id", "-e",
        help="Filter by evaluation set ID"
    )
    list_parser.add_argument(
        "--page", "-p",
        type=int,
        default=1,
        help="Page number (default: 1)"
    )
    list_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Number of items per page (default: 10)"
    )
    list_parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "markdown", "csv"],
        default="text",
        help="Output format (default: text)"
    )
    list_parser.add_argument(
        "--output", "-o",
        help="Output file to write results to"
    )
    list_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Print only the results, no headers or formatting"
    )
    
    # Meta-optimize (directly optimize the meta-prompt)
    meta_parser = optimize_subparsers.add_parser(
        "meta",
        help="Optimize the system message generator itself",
        description="Trigger self-optimization of the system message generator to improve its performance."
    )
    meta_parser.add_argument(
        "evalset_id",
        help="ID of the evaluation set to use for testing generator"
    )
    meta_parser.add_argument(
        "--generator", "-g",
        default="default",
        help="ID of the generator to optimize (default: default)"
    )
    meta_parser.add_argument(
        "--model", "-m",
        help="Model to use for optimization"
    )
    meta_parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=3,
        help="Maximum number of parallel operations (default: 3)"
    )
    meta_parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    return optimize_parser

def format_optimization_result(result, format_type="text", quiet=False):
    """Format optimization result based on the specified format."""
    # Extract best system message and score from candidates if not directly available
    best_score = result.get('best_score', 0)
    best_system_message = result.get('best_system_message', '')
    
    # If best_score or best_system_message is not in the result data, but there are candidates
    if (not best_score or not best_system_message) and result.get('candidates'):
        # Find the best candidate based on best_candidate_index or highest score
        best_candidate_index = result.get('best_candidate_index', 0)
        if best_candidate_index is not None and 0 <= best_candidate_index < len(result['candidates']):
            best_candidate = result['candidates'][best_candidate_index]
        else:
            # Fall back to finding the candidate with the highest score
            best_candidate = max(result['candidates'], key=lambda c: c.get('score', 0), default=None)
        
        # If we found a best candidate, extract its score and content
        if best_candidate:
            if not best_score:
                best_score = best_candidate.get('score', 0)
            if not best_system_message:
                best_system_message = best_candidate.get('content', '')
    
    # Add the extracted values back to the result for consistent access
    result['best_score'] = best_score
    result['best_system_message'] = best_system_message
    
    # Make sure evalset_name is set - this is sometimes missing in older records
    if 'evalset_id' in result and 'evalset_name' not in result:
        # Try to look up the evalset name from the ID
        evalset_id = result.get('evalset_id')
        if evalset_id:
            try:
                from agentoptim.evalset import get_evalset
                import asyncio
                evalset = asyncio.run(get_evalset(evalset_id))
                if evalset and 'name' in evalset:
                    result['evalset_name'] = evalset['name']
            except Exception:
                # If we can't get the evalset name, just use the ID as name
                result['evalset_name'] = f"Evalset {evalset_id[:8]}..."
    
    if format_type == "json":
        return json.dumps(result, indent=2)
    
    elif format_type == "markdown":
        # Generate a markdown formatted result
        md = "# System Message Optimization Results\n\n"
        md += f"**ID:** {result.get('id', 'N/A')}\n"
        md += f"**EvalSet:** {result.get('evalset_name', 'N/A')}\n"
        md += f"**User Query:** {result.get('user_message', 'N/A')}\n\n"
        
        md += f"## Best System Message (Score: {best_score:.1f}%)\n\n"
        md += f"```\n{best_system_message}\n```\n\n"
        
        md += "## All Candidates\n\n"
        for i, candidate in enumerate(result.get('candidates', [])):
            md += f"### Candidate {i+1} (Score: {candidate.get('score', 0):.1f}%)\n\n"
            md += f"```\n{candidate.get('content', 'N/A')}\n```\n\n"
            
            md += "#### Criterion Scores\n\n"
            for criterion, score in candidate.get('criterion_scores', {}).items():
                md += f"- {criterion}: {score:.1f}%\n"
            md += "\n"
        
        return md
    
    elif format_type == "html":
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>System Message Optimization Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #0066cc; }}
                .container {{ background: #f9f9f9; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .best {{ background: #e6f7ff; border-left: 5px solid #0066cc; }}
                .runnerup {{ background: #f0f7ff; border-left: 5px solid #66a3ff; }}
                .third {{ background: #f5faff; border-left: 5px solid #99c2ff; }}
                .system-message {{ background: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; margin: 10px 0; }}
                .score {{ font-weight: bold; color: #0066cc; }}
                .meta {{ color: #666; font-size: 0.9em; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>System Message Optimization Results</h1>
            
            <div class="container">
                <h2>Summary</h2>
                <p><strong>User Message:</strong> {result.get("user_message", "N/A")}</p>
                <p><strong>EvalSet:</strong> {result.get("evalset_name", "N/A")}</p>
                <p><strong>Candidates Generated:</strong> {len(result.get("candidates", []))}</p>
                <p><strong>Optimization Run ID:</strong> {result.get("id", "N/A")}</p>
                <p><strong>Best Score:</strong> {best_score:.1f}%</p>
        """
        
        # Add self-optimization info if available
        self_opt = result.get("self_optimization", {})
        if self_opt and "error" not in self_opt:
            html += f"""
                <p><strong>Self-Optimization:</strong> ✅ Success (v{self_opt.get("old_version")} → v{self_opt.get("new_version")})</p>
            """
        elif self_opt:
            html += f"""
                <p><strong>Self-Optimization:</strong> ❌ Failed</p>
            """
            
        html += """
            </div>
            
            <div class="container">
                <h2>Evaluation Criteria Performance</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Overall Score</th>
        """
        
        # Get first candidate to extract criterion names
        if result.get("candidates"):
            criteria = list(result["candidates"][0].get("criterion_scores", {}).keys())
            
            # Add table headers for each criterion
            for criterion in criteria:
                html += f"""
                        <th>{criterion}</th>
                """
            
            html += """
                    </tr>
            """
            
            # Add rows for each candidate
            for candidate in result.get("candidates", []):
                rank = candidate.get("rank", "N/A")
                score = candidate.get("score", 0)
                
                html += f"""
                    <tr>
                        <td>{rank}</td>
                        <td class="score">{score:.1f}%</td>
                """
                
                # Add scores for each criterion
                for criterion in criteria:
                    criterion_score = candidate.get("criterion_scores", {}).get(criterion, 0)
                    html += f"""
                        <td>{criterion_score:.1f}%</td>
                    """
                
                html += """
                    </tr>
                """
            
        html += """
                </table>
            </div>
        """
        
        # Add top candidates
        candidates = result.get("candidates", [])
        top_candidates = candidates[:min(3, len(candidates))]
        
        html += """
            <h2>Top System Messages</h2>
        """
        
        for i, candidate in enumerate(top_candidates):
            rank_class = "best" if i == 0 else ("runnerup" if i == 1 else "third")
            rank_name = "🥇 Best System Message" if i == 0 else ("🥈 Second Best" if i == 1 else "🥉 Third Best")
            
            html += f"""
            <div class="container {rank_class}">
                <h3>{rank_name} (Score: {candidate.get("score", 0):.1f}%)</h3>
                <div class="system-message">{candidate.get("content", "")}</div>
                <div class="meta">Rank: {candidate.get("rank", "N/A")}</div>
            </div>
            """
        
        # Add sample response if available
        sample_response = result.get('sample_response', {})
        if sample_response:
            html += """
            <div class="container">
                <h2>Sample Response</h2>
            """
            
            if 'content' in sample_response:
                model = sample_response.get('model', 'unknown')
                content = sample_response['content'].replace('\n', '<br>')
                
                html += f"""
                <p><strong>Model:</strong> {model}</p>
                <div class="system-message">{content}</div>
                """
            elif 'error' in sample_response:
                html += f"""
                <p style="color: red;">Error: {sample_response['error']}</p>
                """
                
            html += """
            </div>
            """
            
        html += """
        </body>
        </html>
        """
        
        return html
    
    else:  # text format (default)
        if quiet:
            return best_system_message
        
        formatted_text = []
        
        # Title
        formatted_text.append(f"{Fore.CYAN}╔{'═' * 76}╗{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}📋 System Message Optimization Results{Style.RESET_ALL}{' ' * 36}{Fore.CYAN}║{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}╚{'═' * 76}╝{Style.RESET_ALL}")
        formatted_text.append("")
        
        # Summary
        formatted_text.append(f"{Fore.GREEN}📊 Summary:{Style.RESET_ALL}")
        
        # Truncate user message if too long
        user_message = result.get("user_message", "")
        if len(user_message) > 60:
            user_message = user_message[:57] + "..."
            
        formatted_text.append(f"  {Fore.WHITE}User Message:{Style.RESET_ALL} {user_message}")
        formatted_text.append(f"  {Fore.WHITE}EvalSet:{Style.RESET_ALL} {result.get('evalset_name', 'Unknown')}")
        formatted_text.append(f"  {Fore.WHITE}Candidates Generated:{Style.RESET_ALL} {len(result.get('candidates', []))}")
        formatted_text.append(f"  {Fore.WHITE}Optimization Run ID:{Style.RESET_ALL} {result.get('id', 'Unknown')}")
        
        # Add self-optimization info if available
        self_opt = result.get("self_optimization", {})
        if self_opt and "error" not in self_opt:
            old_v = self_opt.get("old_version")
            new_v = self_opt.get("new_version")
            formatted_text.append(f"  {Fore.WHITE}Self-Optimization:{Style.RESET_ALL} {Fore.GREEN}✓ Success{Style.RESET_ALL} (v{old_v} → v{new_v})")
        elif self_opt:
            formatted_text.append(f"  {Fore.WHITE}Self-Optimization:{Style.RESET_ALL} {Fore.RED}✗ Failed{Style.RESET_ALL}")
            
        formatted_text.append("")
        
        # Best system message
        formatted_text.append(f"{Fore.GREEN}🏆 Best System Message (Score: {best_score:.1f}%):{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
        
        # Display the best system message, wrapping at 76 characters
        import textwrap
        for line in textwrap.wrap(best_system_message if best_system_message else "No system message available", width=76):
            formatted_text.append(line)
            
        formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
        formatted_text.append("")
        
        # Add all candidates comparison
        candidates = result.get('candidates', [])
        if len(candidates) > 1:
            formatted_text.append(f"{Fore.GREEN}📊 All Candidates Comparison:{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            
            # Table header
            formatted_text.append(f"{Fore.WHITE}{'Rank':^6} {'Score':^8} {'Type':^12} {'System Message (preview)':50}{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * 6:^6} {'─' * 8:^8} {'─' * 12:^12} {'─' * 50}{Style.RESET_ALL}")
            
            # Sort candidates by score
            sorted_candidates = sorted(candidates, key=lambda c: c.get('score', 0), reverse=True)
            
            # Display candidates
            for candidate in sorted_candidates:
                # Determine type
                if candidate.get('generation_metadata', {}).get('is_fallback', False):
                    type_str = f"{Fore.YELLOW}Fallback{Style.RESET_ALL}"
                else:
                    type_str = "Generated"
                
                # Get content preview
                content = candidate.get('content', '')
                content_preview = content[:47] + '...' if len(content) > 50 else content
                
                # Format row
                rank = candidate.get('rank', 'N/A')
                score = candidate.get('score', 0)
                score_str = f"{score:.1f}%"
                
                formatted_text.append(f"{rank:^6} {score_str:^8} {type_str:^12} {content_preview}")
            
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            formatted_text.append("")
        
        # Add sample response if available
        sample_response = result.get('sample_response', {})
        if sample_response:
            formatted_text.append(f"{Fore.GREEN}💬 Sample Response:{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            
            if 'content' in sample_response:
                model = sample_response.get('model', 'unknown')
                formatted_text.append(f"{Fore.WHITE}Model: {model}{Style.RESET_ALL}")
                formatted_text.append("")
                
                # Display the response, wrapping at 76 characters
                import textwrap
                for line in textwrap.wrap(sample_response['content'], width=76):
                    formatted_text.append(line)
            elif 'error' in sample_response:
                formatted_text.append(f"{Fore.RED}Error: {sample_response['error']}{Style.RESET_ALL}")
                
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            formatted_text.append("")
        
        # Add information about how to get more details
        formatted_text.append(f"{Fore.YELLOW}ℹ️  To get more details:{Style.RESET_ALL}")
        formatted_text.append(f"   agentoptim optimize get {result.get('id', 'ID')} --format html --output report.html")
        
        # Add tip about generating a sample response if not already present
        if not sample_response:
            formatted_text.append(f"{Fore.YELLOW}ℹ️  To generate a sample response:{Style.RESET_ALL}")
            formatted_text.append(f"   agentoptim optimize get {result.get('id', 'ID')} --generate-response")
        
        return "\n".join(formatted_text)

def format_optimization_list(result, format_type="text", quiet=False):
    """Format list of optimization runs based on the specified format."""
    if format_type == "json":
        return json.dumps(result, indent=2)
    
    elif format_type == "csv":
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["ID", "User Message", "EvalSet", "Best Score", "Candidates", "Timestamp"])
        
        # Write data
        for run in result.get("optimization_runs", []):
            writer.writerow([
                run.get("id", ""),
                run.get("user_message", "")[:50],
                run.get("evalset_name", ""),
                run.get("best_score", 0),
                len(run.get("candidates", [])),
                run.get("timestamp_formatted", "")
            ])
            
        return output.getvalue()
    
    elif format_type == "markdown":
        md = ["# System Message Optimization Runs", ""]
        
        # Table header
        md.append("| ID | User Message | EvalSet | Score | Date |")
        md.append("|----|-------------|---------|-------|------|")
        
        # Table rows
        for run in result.get("optimization_runs", []):
            user_msg = run.get("user_message", "")[:30] + "..." if len(run.get("user_message", "")) > 30 else run.get("user_message", "")
            md.append(f"| {run.get('id', '')} | {user_msg} | {run.get('evalset_name', '')} | {run.get('best_score', 0):.1f}% | {run.get('timestamp_formatted', '')} |")
            
        # Pagination info
        md.append("")
        pagination = result.get("pagination", {})
        md.append(f"Page {pagination.get('page', 1)} of {pagination.get('total_pages', 1)} • {pagination.get('total_count', 0)} total runs")
        
        return "\n".join(md)
    
    else:  # text format (default)
        if quiet:
            # Just print IDs, one per line
            return "\n".join([run.get("id", "") for run in result.get("optimization_runs", [])])
        
        formatted_text = []
        
        # Title
        formatted_text.append(f"{Fore.CYAN}╔{'═' * 76}╗{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}📋 System Message Optimization Runs{Style.RESET_ALL}{' ' * 36}{Fore.CYAN}║{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}╚{'═' * 76}╝{Style.RESET_ALL}")
        formatted_text.append("")
        
        # Handle empty results
        if not result.get("optimization_runs"):
            formatted_text.append(f"{Fore.YELLOW}No optimization runs found.{Style.RESET_ALL}")
            return "\n".join(formatted_text)
        
        # Table header
        formatted_text.append(f"{Fore.WHITE}{'ID':<36} {'User Message':<30} {'Score':>7} {'Date':<16}{Style.RESET_ALL}")
        formatted_text.append(f"{Fore.CYAN}{'─' * 36} {'─' * 30} {'─' * 7} {'─' * 16}{Style.RESET_ALL}")
        
        # Table rows
        for run in result.get("optimization_runs", []):
            # Truncate user message if too long
            user_msg = run.get("user_message", "")
            if len(user_msg) > 27:
                user_msg = user_msg[:24] + "..."
                
            # Format score with color based on value
            score = run.get("best_score", 0)
            if score >= 90:
                score_str = f"{Fore.GREEN}{score:.1f}%{Style.RESET_ALL}"
            elif score >= 70:
                score_str = f"{Fore.YELLOW}{score:.1f}%{Style.RESET_ALL}"
            else:
                score_str = f"{Fore.RED}{score:.1f}%{Style.RESET_ALL}"
                
            formatted_text.append(f"{run.get('id', ''):<36} {user_msg:<30} {score_str:>11} {run.get('timestamp_formatted', ''):<16}")
            
        formatted_text.append("")
        
        # Pagination info
        pagination = result.get("pagination", {})
        page = pagination.get("page", 1)
        total_pages = pagination.get("total_pages", 1)
        total_count = pagination.get("total_count", 0)
        
        formatted_text.append(f"{Fore.CYAN}Page {page} of {total_pages} • {total_count} total optimization runs{Style.RESET_ALL}")
        
        # Navigation help
        if pagination.get("has_next"):
            formatted_text.append(f"Use --page {page + 1} to see the next page")
            
        return "\n".join(formatted_text)

async def handle_optimize_create(args):
    """Handle the optimize create command."""
    # Get user message either from args or interactively
    user_message = args.user_message
    
    if not user_message or args.interactive:
        print(f"{Fore.CYAN}Enter user message to optimize system messages for:{Style.RESET_ALL}")
        user_message = input("> ")
        
    if not user_message:
        print(f"{Fore.RED}Error: No user message provided{Style.RESET_ALL}")
        return 1
        
    # Configure environment variables for provider and model
    if args.provider:
        if args.provider == "openai":
            os.environ["AGENTOPTIM_API_BASE"] = "https://api.openai.com/v1"
            if not args.model:
                args.model = "gpt-4o-mini"  # Default OpenAI model
        elif args.provider == "anthropic":
            os.environ["AGENTOPTIM_API_BASE"] = "https://api.anthropic.com/v1"
            if not args.model:
                args.model = "claude-3-5-haiku-20240307"  # Default Anthropic model
        elif args.provider == "local":
            os.environ["AGENTOPTIM_API_BASE"] = "http://localhost:1234/v1"
            if not args.model:
                args.model = "meta-llama-3.1-8b-instruct"  # Default local model
                
    # Set model if specified
    if args.model:
        os.environ["AGENTOPTIM_JUDGE_MODEL"] = args.model
        
    # Set up progress display
    spinner = None
    
    def update_progress(current, total, message):
        """Progress callback for optimization."""
        nonlocal spinner
        
        if spinner is None:
            # Initialize spinner on first progress update
            spinner = FancySpinner()
            spinner.start(f"Optimizing system messages for: {user_message[:40]}...")
            # Show debugging info - this will help diagnose issues
            print(f"{Fore.YELLOW}Debug: Set DEBUG_MODE=1 for more detailed logs{Style.RESET_ALL}")
            
        percent = int((current / total) * 100)
        spinner.update(percent=percent, message=message)
    
    try:
        # Call the manage_optimization_runs function
        result = await manage_optimization_runs(
            action="optimize",
            user_message=user_message,
            evalset_id=args.evalset_id,
            base_system_message=args.base,
            num_candidates=args.num_candidates,
            generator_id=args.generator,
            generator_model=args.model,
            diversity_level=args.diversity,
            max_parallel=args.concurrency,
            additional_instructions=args.instructions,
            self_optimize=args.self_optimize,
            progress_callback=update_progress
        )
        
        # Stop spinner if it was started
        if spinner:
            spinner.stop()
            
        # Check for errors
        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
            
        # Format and display the result
        formatted_result = format_optimization_result(result, args.format, quiet=False)
        
        # Handle output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_result)
                
            print(f"{Fore.GREEN}Results written to {args.output}{Style.RESET_ALL}")
            
            # If HTML and on macOS, try to open it
            if args.format == "html" and sys.platform == "darwin":
                os.system(f"open {args.output}")
        else:
            print(formatted_result)
            
        return 0
        
    except Exception as e:
        # Stop spinner if it was started
        if spinner:
            spinner.stop()
            
        # Provide more helpful error message
        error_msg = str(e)
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        
        # Add suggestions for common errors
        if "API" in error_msg or "model" in error_msg.lower():
            print(f"\n{Fore.YELLOW}Troubleshooting suggestions:{Style.RESET_ALL}")
            print(f"1. Check that you have set a valid API key for your provider:")
            print(f"   - For OpenAI: export OPENAI_API_KEY=your_key_here")
            print(f"   - For Anthropic: export ANTHROPIC_API_KEY=your_key_here")
            print(f"2. Verify the model is available with your API key")
            print(f"3. Try a different model with --model parameter")
            print(f"4. For local models, ensure your local API server is running")
        
        logger.exception("Error in handle_optimize_create")
        return 1

async def handle_optimize_get(args):
    """Handle the optimize get command."""
    try:
        # Call the manage_optimization_runs function
        result = await manage_optimization_runs(
            action="get",
            optimization_run_id=args.optimization_run_id
        )
        
        # Check for errors
        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
        
        # Extract optimization run data
        if "optimization_run" in result:
            optimization_run = result["optimization_run"]
        else:
            print(f"{Fore.RED}Error: No optimization run data in response{Style.RESET_ALL}")
            return 1
        
        # Generate a sample response if requested
        if args.generate_response:
            # Configure environment variables for provider and model
            if args.provider:
                if args.provider == "openai":
                    os.environ["AGENTOPTIM_API_BASE"] = "https://api.openai.com/v1"
                    if not args.model:
                        args.model = "gpt-4o-mini"  # Default OpenAI model
                elif args.provider == "anthropic":
                    os.environ["AGENTOPTIM_API_BASE"] = "https://api.anthropic.com/v1"
                    if not args.model:
                        args.model = "claude-3-5-haiku-20240307"  # Default Anthropic model
                elif args.provider == "local":
                    os.environ["AGENTOPTIM_API_BASE"] = "http://localhost:1234/v1"
                    if not args.model:
                        args.model = "meta-llama-3.1-8b-instruct"  # Default local model
            
            # Get system message and user message
            best_system_message = optimization_run.get("best_system_message", "")
            user_message = optimization_run.get("user_message", "")
            
            # Check both fields
            if not best_system_message:
                print(f"{Fore.RED}Error: Could not find system message in optimization run{Style.RESET_ALL}")
                return 1
                
            # For older optimization runs, the user message might be in the metadata
            if not user_message:
                # Try to get it from the candidate content which often includes the user query
                if "You are a helpful AI assistant answering questions about" in best_system_message:
                    try:
                        # Extract from format like "...answering questions about What is life insurance?..."
                        import re
                        match = re.search(r"questions about (.*?)\.\.\.?", best_system_message)
                        if match:
                            user_message = match.group(1).strip()
                    except Exception:
                        pass
                
                # If still no user message, use a placeholder
                if not user_message:
                    print(f"{Fore.YELLOW}Warning: Could not find user message in optimization run. Using a placeholder.{Style.RESET_ALL}")
                    user_message = "Please provide information about this topic."
            
            print(f"{Fore.YELLOW}Generating sample response...{Style.RESET_ALL}")
            
            # Get the model response
            try:
                from agentoptim.runner import call_llm_api
                
                # Create messages
                messages = [
                    {"role": "system", "content": best_system_message},
                    {"role": "user", "content": user_message}
                ]
                
                # Call the API
                response = await call_llm_api(
                    messages=messages,
                    model=args.model
                )
                
                # Extract the response content
                if "choices" in response and response["choices"]:
                    content = response["choices"][0].get("message", {}).get("content", "")
                    
                    # Add the generated response to the optimization run
                    optimization_run["sample_response"] = {
                        "model": args.model or response.get("model", "unknown"),
                        "content": content
                    }
                else:
                    # If there was an issue, add a note
                    optimization_run["sample_response"] = {
                        "error": "Failed to generate response",
                        "details": response
                    }
            except Exception as e:
                print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
                optimization_run["sample_response"] = {
                    "error": f"Failed to generate response: {str(e)}"
                }
        
        # Format and display the result
        formatted_result = format_optimization_result(optimization_run, args.format, quiet=False)
        
        # Handle output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_result)
                
            print(f"{Fore.GREEN}Results written to {args.output}{Style.RESET_ALL}")
            
            # If HTML and on macOS, try to open it
            if args.format == "html" and sys.platform == "darwin":
                os.system(f"open {args.output}")
        else:
            print(formatted_result)
            
        return 0
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        logger.exception("Error in handle_optimize_get")
        return 1

async def handle_optimize_list(args):
    """Handle the optimize list command."""
    try:
        # Call the manage_optimization_runs function
        result = await manage_optimization_runs(
            action="list",
            evalset_id=args.evalset_id,
            page=args.page,
            page_size=args.limit
        )
        
        # Check for errors
        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
            
        # Format and display the result
        formatted_result = format_optimization_list(result, args.format, quiet=args.quiet)
        
        # Handle output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_result)
                
            print(f"{Fore.GREEN}Results written to {args.output}{Style.RESET_ALL}")
        else:
            print(formatted_result)
            
        return 0
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        logger.exception("Error in handle_optimize_list")
        return 1

async def handle_optimize_meta(args):
    """Handle the optimize meta command (direct self-optimization)."""
    try:
        # Set up spinner for progress display
        spinner = FancySpinner()
        spinner.start(f"Self-optimizing generator '{args.generator}'...")
        
        # Import necessary functions
        from agentoptim.sysopt import self_optimize_generator, get_all_meta_prompts
        
        # Configure environment variables for provider and model
        if args.provider:
            if args.provider == "openai":
                os.environ["AGENTOPTIM_API_BASE"] = "https://api.openai.com/v1"
                if not args.model:
                    args.model = "gpt-4o-mini"  # Default OpenAI model
            elif args.provider == "anthropic":
                os.environ["AGENTOPTIM_API_BASE"] = "https://api.anthropic.com/v1"
                if not args.model:
                    args.model = "claude-3-5-haiku-20240307"  # Default Anthropic model
            elif args.provider == "local":
                os.environ["AGENTOPTIM_API_BASE"] = "http://localhost:1234/v1"
                if not args.model:
                    args.model = "meta-llama-3.1-8b-instruct"  # Default local model
                    
        # Set model if specified
        if args.model:
            os.environ["AGENTOPTIM_JUDGE_MODEL"] = args.model
            
        # Progress callback
        def update_progress(current, total, message):
            """Progress callback for optimization."""
            percent = int((current / total) * 100)
            spinner.update(percent=percent, message=message)
        
        # Get the generator
        generators = get_all_meta_prompts()
        if args.generator not in generators:
            spinner.stop()
            print(f"{Fore.RED}Error: Generator '{args.generator}' not found{Style.RESET_ALL}")
            return 1
            
        generator = generators[args.generator]
        
        # Run self-optimization
        result = await self_optimize_generator(
            generator=generator,
            evalset_id=args.evalset_id,
            generator_model=args.model,
            max_parallel=args.concurrency,
            progress_callback=update_progress
        )
        
        # Stop spinner
        spinner.stop()
        
        # Check for errors
        if "error" in result:
            print(f"{Fore.RED}Error: {result['error']}{Style.RESET_ALL}")
            return 1
            
        # Display result
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"{Fore.GREEN}✅ Self-optimization successful!{Style.RESET_ALL}")
            print(f"Generator: {result.get('generator_id')}")
            print(f"Version: {result.get('old_version')} → {result.get('new_version')}")
            print(f"Success rate: {result.get('success_rate', 0):.2f}")
            print(f"{Fore.YELLOW}The generator has been improved and will generate better system messages.{Style.RESET_ALL}")
            
        return 0
        
    except Exception as e:
        # Stop spinner if it was started
        if 'spinner' in locals() and spinner:
            spinner.stop()
            
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        logger.exception("Error in handle_optimize_meta")
        return 1

async def handle_optimize(args):
    """Handle the optimize command."""
    # Dispatch to the appropriate handler based on the action
    if args.action == "create":
        return await handle_optimize_create(args)
    elif args.action == "get":
        return await handle_optimize_get(args)
    elif args.action == "list":
        return await handle_optimize_list(args)
    elif args.action == "meta":
        return await handle_optimize_meta(args)
    else:
        print(f"{Fore.RED}Error: Unknown action '{args.action}'{Style.RESET_ALL}")
        return 1

# FancySpinner is now imported from agentoptim.cli.core
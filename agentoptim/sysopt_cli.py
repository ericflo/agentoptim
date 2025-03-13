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
import httpx

# Enable debug mode for advanced logging
DEBUG_MODE = os.environ.get("AGENTOPTIM_DEBUG", "0") == "1"

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
    
    # Set the handler function
    optimize_parser.set_defaults(func=handle_optimize)
    
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
    create_parser.add_argument(
        "--continue-from",
        help="Continue optimization from a previous run ID (uses best system message as base)"
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
        "--provider",
        choices=["openai", "anthropic", "local"],
        help="Provider to use for evaluations (sets appropriate defaults)"
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
                # Call directly without asyncio.run - the function is not async
                evalset = get_evalset(evalset_id)
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
            
        # Add sample responses if available
        sample_responses = result.get('sample_responses', {})
        
        # Also check the old format for backward compatibility
        single_sample_response = result.get('sample_response', {})
        if single_sample_response and not sample_responses:
            sample_responses = {"best": single_sample_response}
            
        if sample_responses:
            md += "## Sample Responses\n\n"
            
            for key, response in sample_responses.items():
                # Get candidate info if available
                candidate_info = ""
                if 'candidate_rank' in response:
                    rank = response.get('candidate_rank', 'N/A')
                    score = response.get('score', 0)
                    candidate_info = f" (Rank {rank}, Score: {score:.1f}%)"
                elif key == "best":
                    candidate_info = " (Best System Message)"
                elif key.isdigit():
                    # For backward compatibility with numerically indexed responses
                    candidate_info = f" (Candidate {int(key)+1})"
                
                md += f"### Sample Response{candidate_info}\n\n"
                
                # Add system message preview if available
                if 'system_message_preview' in response:
                    md += f"**System Message:** {response['system_message_preview']}\n\n"
                
                if 'content' in response:
                    model = response.get('model', 'unknown')
                    md += f"**Model:** {model}\n\n"
                    md += f"```\n{response['content']}\n```\n\n"
                elif 'error' in response:
                    md += f"**Error:** {response['error']}\n\n"
        
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
                .system-preview {{ font-family: monospace; color: #555; background-color: #f0f0f0; padding: 2px 5px; border-radius: 3px; display: inline-block; margin: 5px 0; }}
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
                
                {"<p><strong>Iteration:</strong> " + str(result.get('iteration', 1)) + "</p>" if result.get('iteration', 1) > 1 or result.get('continued_from') else ""}
                {"<p><strong>Continued From:</strong> " + str(result.get('continued_from', '')) + "</p>" if result.get('continued_from') else ""}
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
        
        # Add sample responses if available
        sample_responses = result.get('sample_responses', {})
        
        # Also check the old format for backward compatibility
        single_sample_response = result.get('sample_response', {})
        if single_sample_response and not sample_responses:
            sample_responses = {"best": single_sample_response}
        
        if sample_responses:
            html += """
            <div class="container">
                <h2>Sample Responses</h2>
            """
            
            for key, response in sample_responses.items():
                # Get candidate info if available
                candidate_info = ""
                candidate_class = ""
                
                if 'candidate_rank' in response:
                    rank = response.get('candidate_rank', 'N/A')
                    score = response.get('score', 0)
                    candidate_info = f" (Rank {rank}, Score: {score:.1f}%)"
                    
                    # Set CSS class based on rank
                    if rank == 1:
                        candidate_class = "best"
                    elif rank == 2:
                        candidate_class = "runnerup"
                    elif rank == 3:
                        candidate_class = "third"
                        
                elif key == "best":
                    candidate_info = " (Best System Message)"
                    candidate_class = "best"
                elif key.isdigit():
                    # For backward compatibility with numerically indexed responses
                    candidate_info = f" (Candidate {int(key)+1})"
                
                html += f"""
                <div class="container {candidate_class}">
                    <h3>Sample Response{candidate_info}</h3>
                """
                
                # Add system message preview if available
                if 'system_message_preview' in response:
                    html += f"""
                    <p><strong>System Message:</strong> <span class="system-preview">{response['system_message_preview']}</span></p>
                    """
                
                if 'content' in response:
                    model = response.get('model', 'unknown')
                    content = response['content'].replace('\n', '<br>')
                    
                    html += f"""
                    <p><strong>Model:</strong> {model}</p>
                    <div class="system-message">{content}</div>
                    """
                elif 'error' in response:
                    html += f"""
                    <p style="color: red;">Error: {response['error']}</p>
                    """
                    
                html += """
                </div>
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
        
        # Get terminal width for better formatting
        import shutil
        term_width = shutil.get_terminal_size().columns
        box_width = min(76, term_width - 4)
        
        # Simpler title for narrow terminals
        if term_width < 60:
            formatted_text.append(f"{Fore.CYAN}== System Message Optimization Results =={Style.RESET_ALL}")
        else:
            # Title with box - adjusted to terminal width
            formatted_text.append(f"{Fore.CYAN}╔{'═' * box_width}╗{Style.RESET_ALL}")
            title_text = "📋 System Message Optimization Results"
            padding = max(0, box_width - len(title_text) - 2)
            formatted_text.append(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}{title_text}{Style.RESET_ALL}{' ' * padding}{Fore.CYAN}║{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}╚{'═' * box_width}╝{Style.RESET_ALL}")
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
        
        # Show iteration information if this is part of a sequence
        if result.get('iteration', 1) > 1 or result.get('continued_from'):
            formatted_text.append(f"  {Fore.WHITE}Iteration:{Style.RESET_ALL} {result.get('iteration', 1)}")
            if result.get('continued_from'):
                formatted_text.append(f"  {Fore.WHITE}Continued From:{Style.RESET_ALL} {result.get('continued_from')}")
        
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
        
        # Use line width based on terminal width
        line_width = min(78, term_width - 2)
        formatted_text.append(f"{Fore.CYAN}{'─' * line_width}{Style.RESET_ALL}")
        
        # Display the best system message, wrapping based on terminal width
        import textwrap
        wrap_width = min(76, term_width - 4)  # Leave some margin
        
        # Always show the full system message
        # It's crucial information that shouldn't be truncated
        for line in textwrap.wrap(best_system_message if best_system_message else "No system message available", width=wrap_width):
            formatted_text.append(line)
            
        formatted_text.append(f"{Fore.CYAN}{'─' * line_width}{Style.RESET_ALL}")
        formatted_text.append("")
        
        # Add all candidates comparison
        candidates = result.get('candidates', [])
        if len(candidates) > 1:
            formatted_text.append(f"{Fore.GREEN}📊 Top Candidates Comparison:{Style.RESET_ALL}")
            
            # Use line width based on terminal width
            line_width = min(78, term_width - 2)
            formatted_text.append(f"{Fore.CYAN}{'─' * line_width}{Style.RESET_ALL}")
            
            # Get preview width based on terminal width
            preview_width = min(50, max(30, line_width - 30))
            
            # Table header - adjust based on terminal width
            if term_width < 80:
                # Compact header for narrow terminals
                formatted_text.append(f"{Fore.WHITE}{'#':^3} {'Score':^8} {'Preview':{preview_width}}{Style.RESET_ALL}")
                formatted_text.append(f"{Fore.CYAN}{'─' * 3:^3} {'─' * 8:^8} {'─' * preview_width}{Style.RESET_ALL}")
            else:
                # Full header for wider terminals
                formatted_text.append(f"{Fore.WHITE}{'Rank':^6} {'Score':^8} {'Type':^12} {'System Message (preview)':{preview_width}}{Style.RESET_ALL}")
                formatted_text.append(f"{Fore.CYAN}{'─' * 6:^6} {'─' * 8:^8} {'─' * 12:^12} {'─' * preview_width}{Style.RESET_ALL}")
            
            # Sort candidates by score
            sorted_candidates = sorted(candidates, key=lambda c: c.get('score', 0), reverse=True)
            
            # Display top candidates only (limit to 3 to avoid excessive output)
            top_candidates = sorted_candidates[:min(3, len(sorted_candidates))]
            for candidate in top_candidates:
                # Determine type
                if candidate.get('generation_metadata', {}).get('is_fallback', False):
                    type_str = f"{Fore.YELLOW}Fallback{Style.RESET_ALL}"
                else:
                    type_str = "Generated"
                
                # Get content preview
                content = candidate.get('content', '')
                preview_len = preview_width - 3  # Leave room for ellipsis
                # In the table, we still use a preview
                content_preview = content[:preview_len] + '...' if len(content) > preview_len else content
                
                # Format row
                rank = candidate.get('rank', 'N/A')
                score = candidate.get('score', 0)
                score_str = f"{score:.1f}%"
                
                # Adjust row format based on terminal width
                if term_width < 80:
                    formatted_text.append(f"{rank:^3} {score_str:^8} {content_preview}")
                else:
                    formatted_text.append(f"{rank:^6} {score_str:^8} {type_str:^12} {content_preview}")
            
            # Show count if we truncated the list
            if len(sorted_candidates) > len(top_candidates):
                remaining = len(sorted_candidates) - len(top_candidates)
                formatted_text.append(f"{Fore.YELLOW}... and {remaining} more candidates not shown{Style.RESET_ALL}")
                
            formatted_text.append(f"{Fore.CYAN}{'─' * line_width}{Style.RESET_ALL}")
            formatted_text.append("")
        
        # Add sample responses if available
        sample_responses = result.get('sample_responses', {})
        
        # Also check the old format for backward compatibility
        single_sample_response = result.get('sample_response', {})
        if single_sample_response and not sample_responses:
            sample_responses = {"best": single_sample_response}
            
        if sample_responses:
            formatted_text.append(f"{Fore.GREEN}💬 Sample Responses:{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            
            import textwrap
            for key, response in sample_responses.items():
                # Get candidate info if available
                candidate_info = ""
                candidate_color = Fore.YELLOW
                if 'candidate_rank' in response:
                    rank = response.get('candidate_rank', 'N/A')
                    score = response.get('score', 0)
                    candidate_info = f" (Rank {rank}, Score: {score:.1f}%)"
                    
                    # Set color based on rank
                    if rank == 1:
                        candidate_color = Fore.GREEN
                    elif rank == 2:
                        candidate_color = Fore.CYAN
                    elif rank == 3:
                        candidate_color = Fore.BLUE
                        
                elif key == "best":
                    candidate_info = " (Best System Message)"
                    candidate_color = Fore.GREEN
                elif key.isdigit():
                    # For backward compatibility with numerically indexed responses
                    candidate_info = f" (Candidate {int(key)+1})"
                
                formatted_text.append(f"{candidate_color}Sample Response{candidate_info}:{Style.RESET_ALL}")
                
                # Add system message preview if available
                if 'system_message_preview' in response:
                    formatted_text.append(f"{Fore.WHITE}System Message: {Fore.CYAN}{response['system_message_preview']}{Style.RESET_ALL}")
                
                if 'content' in response:
                    model = response.get('model', 'unknown')
                    formatted_text.append(f"{Fore.WHITE}Model: {model}{Style.RESET_ALL}")
                    formatted_text.append("")
                    
                    # Display the response, wrapping at 76 characters
                    for line in textwrap.wrap(response['content'], width=76):
                        formatted_text.append(line)
                    formatted_text.append("")
                elif 'error' in response:
                    formatted_text.append(f"{Fore.RED}Error: {response['error']}{Style.RESET_ALL}")
                    formatted_text.append("")
                    
                # Add a separator between responses
                formatted_text.append(f"{Fore.CYAN}{'─' * 40}{Style.RESET_ALL}")
                formatted_text.append("")
            
            formatted_text.append(f"{Fore.CYAN}{'─' * 78}{Style.RESET_ALL}")
            formatted_text.append("")
        
        # Add information about how to get more details
        formatted_text.append(f"{Fore.YELLOW}ℹ️  To get more details:{Style.RESET_ALL}")
        formatted_text.append(f"   agentoptim optimize get {result.get('id', 'ID')} --format html --output report.html")
        
        # Add tip about generating more sample responses
        formatted_text.append(f"{Fore.YELLOW}ℹ️  To generate additional sample responses:{Style.RESET_ALL}")
        formatted_text.append(f"   agentoptim optimize get {result.get('id', 'ID')} --generate-response")
        
        # Add tip about continuing optimization
        formatted_text.append(f"{Fore.YELLOW}ℹ️  To continue optimizing from this result:{Style.RESET_ALL}")
        formatted_text.append(f"   agentoptim optimize create {result.get('evalset_id', 'EVALSET')} \"{result.get('user_message', 'QUERY')}\" --continue-from {result.get('id', 'ID')}")
        
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
        
        # Get terminal width for better formatting
        import shutil
        term_width = shutil.get_terminal_size().columns
        box_width = min(76, term_width - 4)
        
        # Simpler title for narrow terminals
        if term_width < 60:
            formatted_text.append(f"{Fore.CYAN}== System Message Optimization Runs =={Style.RESET_ALL}")
        else:
            # Title with box - adjusted to terminal width
            formatted_text.append(f"{Fore.CYAN}╔{'═' * box_width}╗{Style.RESET_ALL}")
            title_text = "📋 System Message Optimization Runs"
            padding = max(0, box_width - len(title_text) - 2)
            formatted_text.append(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}{title_text}{Style.RESET_ALL}{' ' * padding}{Fore.CYAN}║{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}╚{'═' * box_width}╝{Style.RESET_ALL}")
        formatted_text.append("")
        
        # Handle empty results
        if not result.get("optimization_runs"):
            formatted_text.append(f"{Fore.YELLOW}No optimization runs found.{Style.RESET_ALL}")
            return "\n".join(formatted_text)
        
        # Calculate column widths based on terminal size
        if term_width < 80:
            # Compact mode for narrow terminals
            id_width = 8  # Show truncated IDs on narrow terminals
            msg_width = max(15, term_width - id_width - 15)  # Dynamic message width
            
            # Compact header
            formatted_text.append(f"{Fore.WHITE}{'ID':<{id_width}} {'Message':<{msg_width}} {'Score':>7}{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * id_width} {'─' * msg_width} {'─' * 7}{Style.RESET_ALL}")
        else:
            # Standard layout for wider terminals
            id_width = min(36, max(8, int(term_width * 0.3)))  # Dynamic width
            date_width = 16
            score_width = 7
            msg_width = max(15, term_width - id_width - score_width - date_width - 10)  # Dynamic message width
            
            # Full header
            formatted_text.append(f"{Fore.WHITE}{'ID':<{id_width}} {'User Message':<{msg_width}} {'Score':>7} {'Date':<{date_width}}{Style.RESET_ALL}")
            formatted_text.append(f"{Fore.CYAN}{'─' * id_width} {'─' * msg_width} {'─' * score_width} {'─' * date_width}{Style.RESET_ALL}")
        
        # Table rows
        for run in result.get("optimization_runs", []):
            # Format ID - truncate for narrow terminals
            run_id = run.get('id', '')
            if term_width < 80:
                run_id = run_id[:id_width-2] + ".." if len(run_id) > id_width else run_id
            
            # Truncate user message based on available width
            user_msg = run.get("user_message", "")
            if len(user_msg) > msg_width - 3:  # Leave room for ellipsis
                user_msg = user_msg[:msg_width-3] + "..."
                
            # Format score with color based on value
            score = run.get("best_score", 0)
            if score >= 90:
                score_str = f"{Fore.GREEN}{score:.1f}%{Style.RESET_ALL}"
            elif score >= 70:
                score_str = f"{Fore.YELLOW}{score:.1f}%{Style.RESET_ALL}"
            else:
                score_str = f"{Fore.RED}{score:.1f}%{Style.RESET_ALL}"
            
            # Format row based on terminal width
            if term_width < 80:
                # Compact row
                formatted_text.append(f"{run_id:<{id_width}} {user_msg:<{msg_width}} {score_str:>11}")
            else:
                # Full row
                formatted_text.append(f"{run_id:<{id_width}} {user_msg:<{msg_width}} {score_str:>11} {run.get('timestamp_formatted', ''):<{date_width}}")
            
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
        
    # Handle continuing from a previous optimization
    if args.continue_from:
        print(f"{Fore.CYAN}Continuing from previous optimization run: {args.continue_from}{Style.RESET_ALL}")
        
        # Get the previous optimization run (not an async function)
        previous_run = get_optimization_run(args.continue_from)
        if not previous_run:
            print(f"{Fore.RED}Error: Previous optimization run not found: {args.continue_from}{Style.RESET_ALL}")
            return 1
            
        # Find the best system message from the previous run
        previous_best = None
        
        # First try using the direct best_system_message field
        if hasattr(previous_run, 'best_system_message') and previous_run.best_system_message:
            previous_best = previous_run.best_system_message
        # Otherwise find the best candidate
        elif hasattr(previous_run, 'candidates') and previous_run.candidates:
            # Use best_candidate_index if available
            if hasattr(previous_run, 'best_candidate_index') and previous_run.best_candidate_index is not None:
                idx = previous_run.best_candidate_index
                if 0 <= idx < len(previous_run.candidates):
                    previous_best = previous_run.candidates[idx].content
            # Otherwise find the highest scored candidate
            else:
                best_candidate = max(previous_run.candidates, key=lambda c: getattr(c, 'score', 0) or 0)
                if best_candidate:
                    previous_best = best_candidate.content
                    
        if not previous_best:
            print(f"{Fore.RED}Error: Could not find best system message in previous run{Style.RESET_ALL}")
            return 1
            
        # Use the previous best as the base system message
        print(f"{Fore.GREEN}Using best system message from previous run as base{Style.RESET_ALL}")
        args.base = previous_best
        
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
    last_message = None  # Track last message to avoid repeating
    
    def update_progress(current, total, message):
        """Progress callback for optimization."""
        nonlocal spinner, last_message
        
        if spinner is None:
            # Initialize spinner on first progress update
            spinner = FancySpinner()
            spinner.start(f"Optimizing system messages for: {user_message[:40]}...")
            # Only show debugging tip if we hit an issue or in a terminal
            if DEBUG_MODE or os.environ.get("AGENTOPTIM_VERBOSE", "0") == "1":
                print(f"{Fore.YELLOW}Tip: Set DEBUG_MODE=1 for more detailed logs{Style.RESET_ALL}")
            
        # Calculate percent as a whole number
        percent = int((current / total) * 100)
        
        # Only update if the message changed or percent changed significantly
        # This prevents console spam
        if message != last_message or percent % 5 == 0:  # Update on 5% increments or message change
            spinner.update(percent=percent, message=message)
            last_message = message
    
    try:
        # Additional params for iteration tracking
        continued_from = args.continue_from
        iteration = 1
        
        # If continuing from a previous run, determine the iteration number
        if continued_from:
            # Get the previous run without awaiting (it's not an async function)
            previous_run = get_optimization_run(continued_from)
            if previous_run and hasattr(previous_run, 'iteration'):
                iteration = previous_run.iteration + 1
                
            # Ensure iteration doesn't go beyond a reasonable limit to prevent
            # trajectory optimization from getting worse over time
            if iteration > 3:
                print(f"{Fore.YELLOW}Warning: Limiting iteration to max of 3 (was {iteration}){Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Too many iterations can lead to optimization degradation.{Style.RESET_ALL}")
                iteration = 3  # Cap at 3 iterations to avoid degradation
                
            # Add diagnostic output about whether we're continuing correctly
            if DEBUG_MODE:
                print(f"{Fore.CYAN}Continuing from previous run: {continued_from}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Iteration: {iteration}{Style.RESET_ALL}")
        
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
            progress_callback=update_progress,
            continued_from=continued_from,
            iteration=iteration
        )
        
        # Stop spinner if it was started
        if spinner:
            spinner.stop()
            
        # Check for errors
        if "error" in result:
            error_msg = result['error']
            print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            
            # Add helpful suggestions for common errors
            if isinstance(error_msg, str):
                if "evalset" in error_msg.lower() and "not found" in error_msg.lower():
                    print(f"{Fore.YELLOW}Hint: Use 'agentoptim evalset list' to list available EvalSets{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}      or create a new one with 'agentoptim evalset create --wizard'{Style.RESET_ALL}")
                elif "generator" in error_msg.lower() and "not found" in error_msg.lower():
                    print(f"{Fore.YELLOW}Hint: The specified generator doesn't exist. Try using 'default'{Style.RESET_ALL}")
            
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
        
        # Add suggestions for common errors with more detailed help
        if isinstance(error_msg, str) and ("API" in error_msg or "model" in error_msg.lower()):
            print(f"\n{Fore.YELLOW}Troubleshooting suggestions:{Style.RESET_ALL}")
            
            # API key errors
            if "key" in error_msg.lower() or "authentication" in error_msg.lower() or "auth" in error_msg.lower():
                print(f"{Fore.GREEN}API Key Issue Detected:{Style.RESET_ALL}")
                print(f"1. Check that you have set a valid API key:")
                print(f"   - For OpenAI: {Fore.CYAN}export OPENAI_API_KEY=your_key_here{Style.RESET_ALL}")
                print(f"   - For Anthropic: {Fore.CYAN}export ANTHROPIC_API_KEY=your_key_here{Style.RESET_ALL}")
                print(f"2. Verify your API key has not expired or been revoked")
                
            # Model-specific errors
            elif "model" in error_msg.lower() or "not found" in error_msg.lower():
                print(f"{Fore.GREEN}Model Availability Issue Detected:{Style.RESET_ALL}")
                print(f"1. Try a different model with: {Fore.CYAN}--model gpt-4-turbo{Style.RESET_ALL} or {Fore.CYAN}--model claude-3-opus{Style.RESET_ALL}")
                print(f"2. For OpenAI models, try: {Fore.CYAN}--model gpt-4o-mini{Style.RESET_ALL} (newer model)")
                print(f"3. For Anthropic models, try: {Fore.CYAN}--model claude-3-5-sonnet{Style.RESET_ALL} (newer model)")
                
            # Connection errors
            elif "connect" in error_msg.lower() or "timeout" in error_msg.lower():
                print(f"{Fore.GREEN}Connection Issue Detected:{Style.RESET_ALL}")
                print(f"1. Check your internet connection")
                print(f"2. For local models, verify your local API server is running")
                print(f"3. The provider's API service may be experiencing issues - try again later")
            
            # Generic API errors
            else:
                print(f"1. Check that you have set a valid API key for your provider:")
                print(f"   - For OpenAI: {Fore.CYAN}export OPENAI_API_KEY=your_key_here{Style.RESET_ALL}")
                print(f"   - For Anthropic: {Fore.CYAN}export ANTHROPIC_API_KEY=your_key_here{Style.RESET_ALL}")
                print(f"2. Verify the model is available with your API key")
                print(f"3. Try a different model with {Fore.CYAN}--model parameter{Style.RESET_ALL}")
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
        
        # Generate additional sample responses if requested
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
            
            # Get system message and user message - using direct access to ensure we get the values
            best_system_message = optimization_run.get("best_system_message", None)
            if not best_system_message and "candidates" in optimization_run and optimization_run["candidates"]:
                # Try to get from the first candidate
                best_candidate_index = optimization_run.get("best_candidate_index", 0)
                if 0 <= best_candidate_index < len(optimization_run["candidates"]):
                    best_candidate = optimization_run["candidates"][best_candidate_index]
                    best_system_message = best_candidate.get("content", "")
            
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
                
                # Also ensure we have the full evalset name by looking it up if needed
                if 'evalset_id' in optimization_run and ('evalset_name' not in optimization_run or optimization_run.get('evalset_name', '').startswith('Evalset ')):
                    evalset_id = optimization_run.get('evalset_id')
                    if evalset_id:
                        try:
                            from agentoptim.evalset import get_evalset
                            # Call the function directly (not async)
                            evalset = get_evalset(evalset_id)
                            if evalset and 'name' in evalset:
                                optimization_run['evalset_name'] = evalset['name']
                        except Exception:
                            # If we can't get the evalset name, leave it as is
                            pass
            
            print(f"{Fore.YELLOW}Generating additional sample response...{Style.RESET_ALL}")
            
            # Get the model response
            try:
                from agentoptim.runner import call_llm_api
                
                # Create messages
                messages = [
                    {"role": "system", "content": best_system_message},
                    {"role": "user", "content": user_message}
                ]
                
                # Call the API - without a JSON schema for a more natural response
                # We need to modify the call directly to avoid using JSON schema
                from agentoptim.runner import get_api_base
                
                # Create basic payload for a normal text response
                payload = {
                    "model": args.model or "gpt-4o-mini",
                    "messages": [
                        # Add a strong system message to force natural language responses
                        {"role": "system", "content": "You are a helpful assistant that provides clear, natural language responses to user questions. Do NOT output JSON or structured formats unless explicitly requested. Just answer the question directly in plain, conversational language."},
                        # The original messages
                        {"role": "system", "content": best_system_message},
                        {"role": "user", "content": user_message}
                    ],
                    "temperature": 0.5,  # Lower temperature for more consistent responses
                    "max_tokens": 1024
                    # Removed top_p parameter as it's not supported by direct API calls
                }
                
                # Configure headers
                headers = {"Content-Type": "application/json"}
                
                # Add authentication
                openai_api_key = os.environ.get("OPENAI_API_KEY")
                anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
                api_base = get_api_base()
                
                if "openai.com" in api_base and openai_api_key:
                    headers["Authorization"] = f"Bearer {openai_api_key}"
                elif "anthropic.com" in api_base and anthropic_api_key:
                    headers["x-api-key"] = anthropic_api_key
                
                # Make direct API call
                import httpx
                import json
                
                # Only log API details in debug mode
                if DEBUG_MODE:
                    print(f"{Fore.YELLOW}Calling {api_base}/chat/completions directly...{Style.RESET_ALL}")
                
                # Make API request
                try:
                    with httpx.Client(timeout=30) as client:
                        response = client.post(
                            f"{api_base}/chat/completions",
                            json=payload,
                            headers=headers
                        )
                        response_data = response.json()
                except Exception as e:
                    response_data = {"error": f"API call failed: {str(e)}"}
                    
                # Convert to expected format
                response = response_data
                
                # Extract the response content
                if "choices" in response and response["choices"]:
                    content = response["choices"][0].get("message", {}).get("content", "")
                    
                    # Initialize sample_responses if it doesn't exist
                    if "sample_responses" not in optimization_run:
                        optimization_run["sample_responses"] = {}
                    
                    # Add the generated response to the optimization run - use additional to differentiate from auto-generated ones
                    response_index = f"additional_{int(time.time())}"
                    
                    # Create a system message preview for better context
                    system_message_preview = best_system_message[:100] + "..." if len(best_system_message) > 100 else best_system_message
                    
                    optimization_run["sample_responses"][response_index] = {
                        "model": args.model or response.get("model", "unknown"),
                        "content": content,
                        "generated_via": "additional request",
                        "system_message_preview": system_message_preview,
                        "system_message_id": "best",
                        "candidate_rank": 1  # Mark as from the best system message
                    }
                    
                    # For backward compatibility, also set sample_response
                    optimization_run["sample_response"] = {
                        "model": args.model or response.get("model", "unknown"),
                        "content": content
                    }
                else:
                    # If there was an issue, add a note
                    response_index = f"additional_{int(time.time())}"
                    
                    if "sample_responses" not in optimization_run:
                        optimization_run["sample_responses"] = {}
                        
                    # Create a system message preview for better context in error cases
                    system_message_preview = best_system_message[:100] + "..." if len(best_system_message) > 100 else best_system_message
                    
                    optimization_run["sample_responses"][response_index] = {
                        "error": "Failed to generate response",
                        "details": response,
                        "system_message_preview": system_message_preview,
                        "system_message_id": "best",
                        "candidate_rank": 1,  # Mark as from the best system message
                        "generated_via": "additional request (failed)"
                    }
                    
                    # For backward compatibility
                    optimization_run["sample_response"] = {
                        "error": "Failed to generate response",
                        "details": response
                    }
                
                # Save the updated optimization run
                from agentoptim.sysopt.core import save_optimization_run, OptimizationRun
                run_obj = OptimizationRun(**optimization_run)
                save_optimization_run(run_obj)
                
            except Exception as e:
                print(f"{Fore.RED}Error generating response: {str(e)}{Style.RESET_ALL}")
                
                # Initialize sample_responses if it doesn't exist
                if "sample_responses" not in optimization_run:
                    optimization_run["sample_responses"] = {}
                
                # Add error to sample_responses
                response_index = f"additional_{int(time.time())}"
                
                # Create a system message preview for better context even in error cases
                system_message_preview = best_system_message[:100] + "..." if len(best_system_message) > 100 else best_system_message
                
                optimization_run["sample_responses"][response_index] = {
                    "error": f"Failed to generate response: {str(e)}",
                    "system_message_preview": system_message_preview,
                    "system_message_id": "best",
                    "candidate_rank": 1,  # Mark as from the best system message
                    "generated_via": "additional request (failed)"
                }
                
                # For backward compatibility
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
        
        # Import necessary functions from correct location
        from agentoptim.sysopt.core import self_optimize_generator, get_all_meta_prompts
        
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
        
        # Check if the evalset exists
        from agentoptim.evalset import get_evalset
        evalset = get_evalset(args.evalset_id)
        if not evalset:
            spinner.stop()
            print(f"{Fore.RED}Error: EvalSet '{args.evalset_id}' not found. Please provide a valid evalset ID.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Hint: Use 'agentoptim evalset list' to see available evalsets.{Style.RESET_ALL}")
            return 1
            
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
            
        # Display result with detailed information
        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"\n{Fore.GREEN}✅ Meta-prompt Self-optimization{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}== OPTIMIZATION SUMMARY =={Style.RESET_ALL}")
            print(f"Generator ID: {result.get('generator_id')}")
            print(f"Version: {result.get('old_version')} → {result.get('new_version')}")
            print(f"Success rate: {result.get('success_rate', 0):.2f}")
            
            # Get the actual generators to show before/after
            try:
                from agentoptim.sysopt.core import get_generator_by_id_and_version
                
                old_generator = get_generator_by_id_and_version(
                    result.get('generator_id'), 
                    result.get('old_version')
                )
                
                new_generator = get_generator_by_id_and_version(
                    result.get('generator_id'), 
                    result.get('new_version')
                )
                
                if old_generator and new_generator:
                    # Show metrics comparison
                    print(f"\n{Fore.CYAN}== PERFORMANCE METRICS =={Style.RESET_ALL}")
                    for metric, value in new_generator.performance_metrics.items():
                        old_value = old_generator.performance_metrics.get(metric, "N/A")
                        if isinstance(old_value, (int, float)) and isinstance(value, (int, float)):
                            change = value - old_value
                            change_str = f"{change:+.2f}" if isinstance(change, float) else f"{change:+d}"
                            print(f"{metric}: {old_value} → {value} ({change_str})")
                        else:
                            print(f"{metric}: {old_value} → {value}")
                    
                    # Show prompt differences in a useful format
                    print(f"\n{Fore.CYAN}== META-PROMPT COMPARISON =={Style.RESET_ALL}")
                    
                    # Use previews from result if available, otherwise calculate them
                    old_preview = result.get('old_meta_prompt_preview')
                    new_preview = result.get('new_meta_prompt_preview')
                    
                    if not old_preview:
                        old_preview = old_generator.meta_prompt[:300] + "..." if len(old_generator.meta_prompt) > 300 else old_generator.meta_prompt
                    
                    if not new_preview:
                        new_preview = new_generator.meta_prompt[:300] + "..." if len(new_generator.meta_prompt) > 300 else new_generator.meta_prompt
                    
                    # Import rich for fancy table display
                    try:
                        from rich.console import Console
                        from rich.table import Table
                        from rich.syntax import Syntax
                        from rich.panel import Panel
                        from rich.text import Text
                        import difflib
                        
                        # Create a table showing side-by-side comparison
                        console = Console()
                        
                        # Generate a unified diff to identify changes
                        old_lines = old_generator.meta_prompt.splitlines()
                        new_lines = new_generator.meta_prompt.splitlines()
                        
                        # Generate diff for detailed comparison
                        diff = list(difflib.unified_diff(
                            old_lines,
                            new_lines,
                            lineterm='',
                            n=0  # Context lines
                        ))
                        
                        # Extract changes
                        additions = []
                        deletions = []
                        for line in diff:
                            if line.startswith('+') and not line.startswith('+++'):
                                additions.append(line[1:])
                            elif line.startswith('-') and not line.startswith('---'):
                                deletions.append(line[1:])
                        
                        # Create summary of changes
                        if additions or deletions:
                            print(f"\n{Fore.CYAN}== KEY CHANGES =={Style.RESET_ALL}")
                            if deletions:
                                print(f"\n{Fore.RED}Removed:{Style.RESET_ALL}")
                                for line in deletions:
                                    print(f"  - {line}")
                            if additions:
                                print(f"\n{Fore.GREEN}Added:{Style.RESET_ALL}")
                                for line in additions:
                                    print(f"  + {line}")

                        # Full meta-prompts
                        print(f"\n{Fore.YELLOW}Previous meta-prompt (v{old_generator.version}):{Style.RESET_ALL}")
                        print(f"{old_generator.meta_prompt}")
                        
                        print(f"\n{Fore.GREEN}New meta-prompt (v{new_generator.version}):{Style.RESET_ALL}")
                        print(f"{new_generator.meta_prompt}")
                        
                    except ImportError:
                        # Fallback to simpler output if rich is not available
                        print(f"\n{Fore.YELLOW}Previous meta-prompt (v{old_generator.version}):{Style.RESET_ALL}")
                        print(f"{old_generator.meta_prompt}")
                        
                        print(f"\n{Fore.GREEN}New meta-prompt (v{new_generator.version}):{Style.RESET_ALL}")
                        print(f"{new_generator.meta_prompt}")
                    
                    # Show test results if available
                    if "test_results" in result:
                        test_results = result["test_results"]
                        print(f"\n{Fore.CYAN}== TEST RESULTS =={Style.RESET_ALL}")
                        print(f"Test queries: {len(test_results.get('test_messages', []))}")
                        print(f"Successful generations: {test_results.get('success_count', 0)}")
                        print(f"Failed generations: {len(test_results.get('failures', []))}")
                        
                        # Show sample outputs
                        if test_results.get('sample_outputs'):
                            print(f"\n{Fore.CYAN}== SAMPLE SYSTEM MESSAGES =={Style.RESET_ALL}")
                            for i, sample in enumerate(test_results.get('sample_outputs', [])):
                                test_message = sample.get('test_message', '')
                                system_message = sample.get('system_message', '')
                                
                                # Format the system message with indentation for readability
                                formatted_message = "\n    ".join(system_message.split("\n"))
                                
                                print(f"\n{Fore.YELLOW}Test Query {i+1}:{Style.RESET_ALL} \"{test_message}\"")
                                print(f"{Fore.GREEN}Complete System Message:{Style.RESET_ALL}")
                                print(f"    {formatted_message}")
                        
                        # Show failures
                        if test_results.get('failures'):
                            print(f"\n{Fore.RED}== GENERATION FAILURES =={Style.RESET_ALL}")
                            for i, failure in enumerate(test_results.get('failures', [])):
                                print(f"\n{Fore.YELLOW}Failed Query {i+1}:{Style.RESET_ALL} \"{failure.get('test_message')}\"")
                                print(f"{Fore.RED}Error Details:{Style.RESET_ALL} {failure.get('error')}")
                    
                    # Get the full length for context
                    # Calculate various meta-prompt stats
                    old_len = len(old_generator.meta_prompt)
                    new_len = len(new_generator.meta_prompt)
                    len_change = new_len - old_len
                    len_change_str = f"{len_change:+d}"
                    
                    # Calculate word counts for more meaningful length comparisons
                    old_word_count = len(old_generator.meta_prompt.split())
                    new_word_count = len(new_generator.meta_prompt.split())
                    word_diff = new_word_count - old_word_count
                    
                    # Calculate line counts
                    old_line_count = len(old_generator.meta_prompt.splitlines())
                    new_line_count = len(new_generator.meta_prompt.splitlines())
                    line_diff = new_line_count - old_line_count
                    
                    print(f"\n{Fore.CYAN}== META-PROMPT STATISTICS =={Style.RESET_ALL}")
                    print(f"Characters: {old_len} → {new_len} ({len_change_str})")
                    print(f"Words: {old_word_count} → {new_word_count} ({word_diff:+d})")
                    print(f"Lines: {old_line_count} → {new_line_count} ({line_diff:+d})")
                    
                    # Add a way to see the full meta-prompts
                    print(f"\n{Fore.CYAN}== DETAILED VIEW =={Style.RESET_ALL}")
                    print(f"To view the full old meta-prompt: {Fore.YELLOW}cat ~/.agentoptim/sysopt/meta_prompts/{result.get('generator_id')}_v{result.get('old_version')}.json{Style.RESET_ALL}")
                    print(f"To view the full new meta-prompt: {Fore.YELLOW}cat ~/.agentoptim/sysopt/meta_prompts/{result.get('generator_id')}_v{result.get('new_version')}.json{Style.RESET_ALL}")
                
            except Exception as e:
                logger.error(f"Error retrieving generators for comparison: {str(e)}")
                # Show basic information even if detailed comparison fails
                pass
                
            print(f"\n{Fore.GREEN}The generator has been improved and will generate better system messages.{Style.RESET_ALL}")
            
        return 0
        
    except Exception as e:
        # Stop spinner if it was started
        if 'spinner' in locals() and spinner:
            spinner.stop()
            
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        logger.exception("Error in handle_optimize_meta")
        return 1

def handle_optimize(args):
    """Handle the optimize command."""
    # Handle optimize command without async functions
    try:
        # Dispatch to the appropriate handler based on the action
        if args.action == "create":
            asyncio.run(handle_optimize_create(args))
            return 0  # Success
        elif args.action == "get":
            asyncio.run(handle_optimize_get(args))
            return 0  # Success
        elif args.action == "list":
            asyncio.run(handle_optimize_list(args))
            return 0  # Success
        elif args.action == "meta":
            asyncio.run(handle_optimize_meta(args))
            return 0  # Success
        else:
            print(f"{Fore.RED}Error: Unknown action '{args.action}'{Style.RESET_ALL}")
            return 1
    except Exception as e:
        print(f"{Fore.RED}Error handling optimize command: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        return 1

# FancySpinner is now imported from agentoptim.cli.core
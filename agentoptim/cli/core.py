#!/usr/bin/env python3

"""
AgentOptim CLI Core - Core functionality for CLI commands
"""

import os
import sys
import json
import argparse
import logging
import uuid
import time
import textwrap
import random
import threading
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import colorama
from colorama import Fore, Style
import itertools
import datetime
import shutil
import asyncio

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
VERSION = "2.2.0"  # Updated with system message optimization tool
MAX_WIDTH = 100  # Maximum width for formatted output

# ASCII Art Logo
LOGO = f"""{Fore.CYAN}
  █████  ██████  ███████ ███    ██ ████████  ██████  ██████  ████████ ██ ███    ███ 
 ██   ██ ██      ██      ████   ██    ██    ██    ██ ██   ██    ██    ██ ████  ████ 
 ███████ ██  ███ █████   ██ ██  ██    ██    ██    ██ ██████     ██    ██ ██ ████ ██ 
 ██   ██ ██   ██ ██      ██  ██ ██    ██    ██    ██ ██         ██    ██ ██  ██  ██ 
 ██   ██  █████  ███████ ██   ████    ██     ██████  ██         ██    ██ ██      ██{Style.RESET_ALL}
                                                                      v{VERSION}
{Fore.YELLOW}📚 Your Complete Toolkit for AI Conversation Evaluation and Optimization{Style.RESET_ALL}
"""

# CLI Tips/Fortune Cookies
CLI_TIPS = [
    "💡 Use 'latest' to quickly access your most recent evaluation run",
    "💡 Try the --interactive flag to create and evaluate conversations in real-time",
    "💡 Set AGENTOPTIM_SHOW_TIMER=1 to see how long commands take to run",
    "💡 Install tab completion with 'agentoptim --install-completion'",
    "💡 Use 'agentoptim run export latest --format html' to generate beautiful reports",
    "💡 Try comparing runs with 'agentoptim run compare latest latest-1'",
    "💡 Use short aliases: 'es' for evalset, 'r' for run, and 'o' for optimize",
    "💡 Export to multiple formats: markdown, CSV, HTML, JSON, or PDF",
    "💡 Use --brief flag for faster evaluations without detailed reasoning",
    "💡 Try different judge models with --model <model_name>",
    "💡 Stuck? Add --help to any command for guidance",
    "💡 Set up a daily evaluation workflow with cron and agentoptim",
    "💡 Use 'agentoptim dev cache' to view caching statistics",
    "💡 Need scripting? Use --format json --quiet for machine-readable output",
    "💡 Combine runs with csvstack: 'agentoptim r export latest --format csv'",
    "💡 Want to run evaluations faster? Try the --concurrency flag",
    "💡 Use 'agentoptim run get latest -o report.md' to save results to a file",
    "💡 Remember eval run IDs with descriptive aliases in your shell",
    "💡 Try 'agentoptim server --provider openai' to switch to OpenAI models",
    "💡 Use emojis in your EvalSet names to make them easier to identify at a glance",
    "💡 Chain evaluations with shell scripts for automated testing",
    "💡 Set 'AGENTOPTIM_DEBUG=1' for detailed logging when troubleshooting",
    "💡 Create specialized EvalSets for different aspects of conversation quality",
    "💡 Try different score visualizations with the --charts flag on exports",
    "💡 Stay organized by tagging EvalSets with categories in their names",
    "💡 Use keyboard shortcuts when in interactive conversation mode",
    "💡 Benchmark different LLMs by changing the --model parameter",
    "💡 Share evaluation results by exporting to HTML and hosting online",
    "💡 Use the -q flag for quieter output in automation scripts",
    "💡 Evaluate conversations from logs by converting them to JSON format",
    "💡 Try AGENTOPTIM_THEME=(ocean|sunset|forest|candy) to customize colors",
    "💡 Filter evaluation runs by EvalSet with 'run list --evalset <id>'",
    "💡 Dark terminal? Use AGENTOPTIM_HIGH_CONTRAST=1 for better readability",
    "💡 Use 'latest-N' to get the Nth most recent evaluation run",
    "💡 Set AGENTOPTIM_CELEBRATE=1 for extra delight on successful commands",
    "💡 Generate optimized system messages with 'agentoptim optimize create'",
    "💡 List your optimization runs with 'agentoptim optimize list'",
    "💡 Export optimized system messages with 'agentoptim optimize get <id> -f html'",
    "💡 Customize system message generation with the --diversity flag",
    "💡 Improve the system message generator with 'agentoptim optimize meta'",
    "💡 Export pretty HTML reports for system message optimization runs",
]

# Success messages
SUCCESS_MESSAGES = [
    "Command completed successfully!",
    "Operation successful!",
    "Task completed successfully!",
    "Done!",
    "Success!",
    "Mission accomplished!",
    "All done!",
    "Finished successfully!",
    "Command executed successfully!",
    "Operation complete!",
]

# Box styles for formatted output
BOX_STYLES = {
    "single": {
        "top_left": "┌", "top_right": "┐", "bottom_left": "└", "bottom_right": "┘",
        "horizontal": "─", "vertical": "│", "title_left": "┤ ", "title_right": " ├"
    },
    "double": {
        "top_left": "╔", "top_right": "╗", "bottom_left": "╚", "bottom_right": "╝",
        "horizontal": "═", "vertical": "║", "title_left": "╡ ", "title_right": " ╞"
    },
    "rounded": {
        "top_left": "╭", "top_right": "╮", "bottom_left": "╰", "bottom_right": "╯",
        "horizontal": "─", "vertical": "│", "title_left": "┤ ", "title_right": " ├"
    },
    "bold": {
        "top_left": "┏", "top_right": "┓", "bottom_left": "┗", "bottom_right": "┛",
        "horizontal": "━", "vertical": "┃", "title_left": "┫ ", "title_right": " ┣"
    },
}

class FancySpinner:
    """A spinner class for showing progress with ETA calculation."""
    
    def __init__(self):
        """Initialize the spinner."""
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.message = ""
        self.current_frame = 0
        self.percent = 0
        self.start_time = None
        self.last_update_time = None
        
    def update(self, message=None, percent=None):
        """Update the spinner message and/or percent."""
        if message:
            self.message = message
        if percent is not None:
            self.percent = percent
        self.last_update_time = time.time()
        
    def _eta_text(self):
        """Calculate and format ETA based on progress."""
        if self.percent <= 0 or self.start_time is None:
            return "calculating..."
            
        elapsed = time.time() - self.start_time
        if self.percent >= 100:
            return f"{int(elapsed)}s total"
            
        total_estimated = elapsed / (self.percent / 100)
        remaining = total_estimated - elapsed
        
        if remaining < 60:
            return f"~{int(remaining)}s remaining"
        elif remaining < 3600:
            return f"~{int(remaining / 60)}m {int(remaining % 60)}s remaining"
        else:
            return f"~{int(remaining / 3600)}h {int((remaining % 3600) / 60)}m remaining"
    
    def _spin(self):
        """Spinner animation function that runs in a thread."""
        import time
        import sys
        
        self.start_time = time.time()
        self.last_update_time = self.start_time
        last_message = None  # Track last displayed message to avoid flicker
        last_percent = None  # Track last displayed percent 
        
        try:
            while self.running:
                # Only update the display if something changed or every 10 frames
                current_time = time.time()
                should_update = (
                    self.message != last_message or
                    self.percent != last_percent or
                    self.current_frame % 10 == 0 or
                    current_time - self.last_update_time < 2.0  # Always update during the first 2 seconds
                )
                
                if should_update:
                    progress_bar = ""
                    if self.percent > 0:
                        filled_len = int(20 * self.percent / 100)
                        progress_bar = f"[{'=' * filled_len}{' ' * (20 - filled_len)}] {self.percent}% {self._eta_text()}"
                    
                    # Construct the spinner line
                    frame = self.frames[self.current_frame]
                    line = f"{Fore.CYAN}{frame}{Style.RESET_ALL} {self.message} {Fore.YELLOW}{progress_bar}{Style.RESET_ALL}"
                    
                    # Clear line and print spinner
                    sys.stdout.write("\r\033[K" + line)
                    sys.stdout.flush()
                    
                    # Update tracking variables
                    last_message = self.message
                    last_percent = self.percent
                
                # Update frame
                self.current_frame = (self.current_frame + 1) % len(self.frames)
                
                # Sleep briefly
                time.sleep(0.1)
                
        except:
            # Handle any exceptions to prevent thread crashes
            pass
            
    def start(self, message="Processing..."):
        """Start the spinner with an initial message."""
        if self.running:
            return
            
        self.message = message
        self.running = True
        
        # Import in method to avoid circular import
        import threading
        import time
        
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop the spinner and clean up."""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
            
        # Clear the line and print a final message
        import sys
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


def format_box(title, content, style="single", border_color=Fore.CYAN, title_align="center", padding=1):
    """Create a nicely formatted box around the content with a title."""
    # Detect if we're running under MCP by checking specific environment variables
    is_mcp_environment = (
        "MODEL_CONTEXT_PROTOCOL_STDIO" in os.environ or 
        "MODEL_CONTEXT_PROTOCOL_VERSION" in os.environ or
        os.environ.get("AGENTOPTIM_IN_MCP", "0") == "1"
    )
    
    # For MCP mode, just return plain text without box formatting
    if is_mcp_environment:
        result = [title]
        result.append("")
        result.extend(content.strip().split("\n"))
        return "\n".join(result)
    
    # For terminal mode, create a fancy box
    
    # Get box style
    box = BOX_STYLES[style]
    
    # Split content into lines
    content_lines = content.strip().split("\n")
    
    # Find the maximum width
    max_content_width = max(len(line) for line in content_lines)
    max_width = max(max_content_width, len(title) + 4)  # Add some buffer for title
    
    # Create the box
    result = []
    
    # Top border
    if title and title_align == "center":
        title_padding = max(0, (max_width - len(title)) // 2)
        top_border = f"{border_color}{box['top_left']}{box['horizontal'] * title_padding}{box['title_left']}{Style.RESET_ALL}{title}{border_color}{box['title_right']}{box['horizontal'] * (max_width - title_padding - len(title) - 2)}{box['top_right']}{Style.RESET_ALL}"
    elif title and title_align == "left":
        top_border = f"{border_color}{box['top_left']}{box['title_left']}{Style.RESET_ALL}{title}{border_color}{box['title_right']}{box['horizontal'] * (max_width - len(title) - 2)}{box['top_right']}{Style.RESET_ALL}"
    else:
        top_border = f"{border_color}{box['top_left']}{box['horizontal'] * max_width}{box['top_right']}{Style.RESET_ALL}"
    
    result.append(top_border)
    
    # Add top padding
    for _ in range(padding):
        result.append(f"{border_color}{box['vertical']}{Style.RESET_ALL}{' ' * max_width}{border_color}{box['vertical']}{Style.RESET_ALL}")
    
    # Content
    for line in content_lines:
        padding_right = max_width - len(line)
        result.append(f"{border_color}{box['vertical']}{Style.RESET_ALL}{line}{' ' * padding_right}{border_color}{box['vertical']}{Style.RESET_ALL}")
    
    # Add bottom padding
    for _ in range(padding):
        result.append(f"{border_color}{box['vertical']}{Style.RESET_ALL}{' ' * max_width}{border_color}{box['vertical']}{Style.RESET_ALL}")
    
    # Bottom border
    result.append(f"{border_color}{box['bottom_left']}{box['horizontal'] * max_width}{box['bottom_right']}{Style.RESET_ALL}")
    
    return "\n".join(result)


def get_random_tip():
    """Return a random CLI tip."""
    return random.choice(CLI_TIPS)


def format_elapsed_time(seconds):
    """Format elapsed time in a human readable format."""
    if seconds < 0.1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 1:
        return f"{seconds:.1f}s"
    elif seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds_remainder = int(seconds % 60)
        return f"{minutes}m {seconds_remainder}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def show_welcome():
    """Show welcome message if not in quiet mode."""
    # Skip welcome in quiet mode
    if len(sys.argv) > 1 and (sys.argv[1] == "--quiet" or sys.argv[1] == "-q" or "--quiet" in sys.argv or "-q" in sys.argv):
        return
        
    # Skip welcome if not an interactive terminal and we're not running a server
    # This helps integration in pipelines
    if not sys.stdout.isatty() and (len(sys.argv) <= 1 or sys.argv[1] != "server"):
        return
        
    # Print logo
    print(LOGO, file=sys.stderr)


def show_success_animation():
    """Show a success animation."""
    if not sys.stdout.isatty():
        return
        
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Quick success animation
    for i in range(6):
        sys.stdout.write(f"\r{Fore.GREEN}{frames[i % len(frames)]} Success!{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.05)
    
    sys.stdout.write("\r\033[K")  # Clear the line
    sys.stdout.flush()


def show_success_message(command, elapsed_time=None):
    """Show a success message with elapsed time if provided."""
    # Pick a success message based on the command
    if command.startswith("run export") or command.startswith("r export"):
        message = "Export complete! Your data is ready."
    elif command.startswith("run compare") or command.startswith("r compare"):
        message = "Comparison complete! Here's the analysis."
    elif command.startswith("evalset create") or command.startswith("es create"):
        message = "EvalSet created successfully! Ready to use for evaluations."
    elif command.startswith("run create") or command.startswith("r create"):
        message = "Evaluation complete! Here are your results."
    elif command.startswith("optimize create") or command.startswith("opt create"):
        message = "System message optimization complete!"
    else:
        message = random.choice(SUCCESS_MESSAGES)
    
    # Add elapsed time if provided
    if elapsed_time:
        # Format the time
        time_str = format_elapsed_time(elapsed_time)
        message += f" ({time_str})"
    
    # Add command-specific flair
    if command.startswith("run create"):
        message += " Your evaluation is ready!"
    elif command.startswith("run export"):
        message += " Your report is ready!"
    elif command.startswith("server"):
        message += " Server is happily running!"
    
    # Show the message
    print(f"\n{Fore.GREEN}{message}{Style.RESET_ALL}", file=sys.stderr)


def display_helpful_error(error, command):
    """Display helpful error suggestions based on the type of error."""
    error_str = str(error).lower()
    
    if "connection" in error_str and "refused" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: The server appears to be unavailable. "
            f"Make sure to start it with:{Style.RESET_ALL}\n"
            f"   agentoptim server"
        )
        
    elif "no such file" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: The specified file could not be found. "
            f"Check the file path and try again.{Style.RESET_ALL}"
        )
        
    elif "permission" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: Permission error accessing a file. "
            f"Check file permissions or try running with elevated privileges.{Style.RESET_ALL}"
        )
        
    elif "json" in error_str and ("decode" in error_str or "parse" in error_str or "invalid" in error_str):
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: Invalid JSON format in the input file. "
            f"Check your JSON syntax or use the --interactive flag instead.{Style.RESET_ALL}"
        )
        
    elif "evalset not found" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: The specified EvalSet ID doesn't exist. "
            f"List available EvalSets with:{Style.RESET_ALL}\n"
            f"   agentoptim evalset list"
        )
        
    elif "eval_run not found" in error_str or "evalrun not found" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: The specified evaluation run ID doesn't exist. "
            f"List available runs with:{Style.RESET_ALL}\n"
            f"   agentoptim run list"
        )
        
    elif "api key" in error_str:
        if "openai" in error_str:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: OpenAI API key is missing or invalid. "
                f"Set your API key with:{Style.RESET_ALL}\n"
                f"   export OPENAI_API_KEY=your_key_here"
            )
        elif "anthropic" in error_str:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Anthropic API key is missing or invalid. "
                f"Set your API key with:{Style.RESET_ALL}\n"
                f"   export ANTHROPIC_API_KEY=your_key_here"
            )
        else:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: API key issue detected. "
                f"Check that you've set the appropriate API key for your provider.{Style.RESET_ALL}"
            )
            
    elif "timeout" in error_str:
        print(
            f"\n{Fore.YELLOW}💡 Suggestion: The operation timed out. "
            f"This might be due to network issues or high server load. "
            f"Try again or increase the timeout with --timeout option.{Style.RESET_ALL}"
        )
        
    # Command-specific errors
    elif "run create" in command or "r create" in command:
        if "conversation" in error_str:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Issue with conversation format. "
                f"Conversations should be valid JSON with 'role' and 'content' fields.{Style.RESET_ALL}\n"
                f"   Example: agentoptim run create <evalset_id> --interactive"
            )
        elif "model" in error_str:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Issue with the specified model. "
                f"Try using a different model:{Style.RESET_ALL}\n"
                f"   agentoptim run create <evalset_id> <file> --model gpt-4"
            )
    
    # Unknown errors - generic suggestions
    else:
        # Default to showing help for the command
        if command.startswith("evalset") or command.startswith("es"):
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Try viewing help for the evalset command:{Style.RESET_ALL}\n"
                f"   agentoptim evalset --help"
            )
        elif command.startswith("run") or command.startswith("r"):
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Try viewing help for the run command:{Style.RESET_ALL}\n"
                f"   agentoptim run --help"
            )
        elif command.startswith("optimize") or command.startswith("opt") or command.startswith("o"):
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Try viewing help for the optimize command:{Style.RESET_ALL}\n"
                f"   agentoptim optimize --help"
            )
        else:
            print(
                f"\n{Fore.YELLOW}💡 Suggestion: Try viewing the general help for available commands:{Style.RESET_ALL}\n"
                f"   agentoptim --help"
            )


def install_completion():
    """Install shell tab completion for the CLI."""
    shell = os.environ.get("SHELL", "")
    
    if "zsh" in shell:
        # Zsh completion
        completion_script = '''
# AgentOptim completion for zsh

_agentoptim_completion() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    response=("${(@f)$(agentoptim --generate-completion-script $words)}")

    for key desc in "${(kv)response}"; do
        if [[ -n $desc ]]; then
            completions_with_descriptions+=("$key:$desc")
        else
            completions+=("$key")
        fi
    done

    if [[ ${#completions_with_descriptions} -gt 0 ]]; then
        _describe -t agentoptim-option "agentoptim options" completions_with_descriptions
    fi

    if [[ ${#completions} -gt 0 ]]; then
        compadd -a completions
    fi
}

compdef _agentoptim_completion agentoptim
'''
        # Path for the completion script
        completion_path = os.path.expanduser("~/.zshrc")
        
        # Check if completion already installed
        try:
            with open(completion_path, "r") as f:
                content = f.read()
                if "AgentOptim completion for zsh" in content:
                    print(f"{Fore.GREEN}✓ Tab completion already installed for Zsh!{Style.RESET_ALL}")
                    return
        except:
            pass
        
        # Write the completion script
        try:
            with open(completion_path, "a") as f:
                f.write("\n" + completion_script + "\n")
            print(f"{Fore.GREEN}✓ Tab completion installed for Zsh!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}ℹ️  Restart your shell or run 'source ~/.zshrc' to activate{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error installing tab completion: {str(e)}{Style.RESET_ALL}")
    
    elif "bash" in shell:
        # Bash completion
        completion_script = '''
# AgentOptim completion for bash

_agentoptim_completion() {
    local cur prev words cword
    _init_completion || return

    local response
    response=$(agentoptim --generate-completion-script "${words[@]:1}" "$((cword-1))")

    if [[ -z "$response" ]]; then
        COMPREPLY=( $(compgen -W "" -- "$cur") )
        return
    fi

    COMPREPLY=( $(compgen -W "$response" -- "$cur") )
}

complete -F _agentoptim_completion agentoptim
'''
        # Path for the completion script
        completion_path = os.path.expanduser("~/.bashrc")
        
        # Check if completion already installed
        try:
            with open(completion_path, "r") as f:
                content = f.read()
                if "AgentOptim completion for bash" in content:
                    print(f"{Fore.GREEN}✓ Tab completion already installed for Bash!{Style.RESET_ALL}")
                    return
        except:
            pass
        
        # Write the completion script
        try:
            with open(completion_path, "a") as f:
                f.write("\n" + completion_script + "\n")
            print(f"{Fore.GREEN}✓ Tab completion installed for Bash!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}ℹ️  Restart your shell or run 'source ~/.bashrc' to activate{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error installing tab completion: {str(e)}{Style.RESET_ALL}")
    
    else:
        # Unsupported shell
        print(f"{Fore.YELLOW}⚠️  Unsupported shell: {shell}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}ℹ️  Tab completion is currently supported for Bash and Zsh.{Style.RESET_ALL}")


def create_cli_parser():
    """Create the CLI argument parser with all commands."""
    parser = argparse.ArgumentParser(
        description=f"AgentOptim v{VERSION} - AI Conversation Evaluation and Optimization Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''
        Examples:
          # Start the server
          agentoptim server
          
          # Create an EvalSet interactively
          agentoptim evalset create --wizard
          
          # Run an evaluation on a conversation
          agentoptim run create <evalset-id> conversation.json
          
          # View most recent evaluation results
          agentoptim run get latest
          
          # Compare two evaluation runs
          agentoptim run compare latest latest-1
          
          # Export evaluation results to HTML
          agentoptim run export latest --format html --output report.html
          
          # Optimize system messages for a user query
          agentoptim optimize create <evalset-id> "How do I reset my password?"
          
        For more examples and detailed documentation, visit:
        https://github.com/ericflo/agentoptim
        ''')
    )
    
    # Add global options
    parser.add_argument("-v", "--version", action="version", version=f"AgentOptim v{VERSION}")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-essential output")
    parser.add_argument(
        "--install-completion", action="store_true",
        help="Install tab completion for Bash or Zsh"
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    return parser, subparsers


def run_cli():
    """Run the CLI with the appropriate command."""
    # Create the parser
    parser, subparsers = create_cli_parser()
    
    # Import the cli_hooks module to register extensions
    from agentoptim import cli_hooks
    
    # Load built-in extensions
    cli_hooks.load_builtin_extensions()
    
    # Apply extensions to register commands
    applied_extensions = cli_hooks.apply_extensions(subparsers)
    logger.debug(f"Applied CLI extensions: {applied_extensions}")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle case where no command is specified
    if args.command is None:
        parser.print_help()
        return
    
    # Call the appropriate command handler
    if hasattr(args, 'func'):
        if asyncio.iscoroutinefunction(args.func):
            # Run async functions
            return asyncio.run(args.func(args))
        else:
            # Run regular functions
            return args.func(args)
    else:
        parser.print_help()
        return
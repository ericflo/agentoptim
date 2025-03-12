"""
Server commands for the AgentOptim CLI
"""

import os
import sys
import asyncio
import logging
import signal
import socket
import time
import subprocess
from colorama import Fore, Style
from typing import Optional

from agentoptim.cli.core import format_box
from agentoptim.utils import DATA_DIR

# Set up logger
logger = logging.getLogger(__name__)

def setup_server_parser(subparsers):
    """Set up the server command parser."""
    # Add server command
    server_parser = subparsers.add_parser(
        "server",
        help="Start the AgentOptim MCP server",
        description="Start the AgentOptim MCP server for evaluating AI conversations and optimizing system messages."
    )
    
    # Add server options
    server_parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.environ.get("AGENTOPTIM_PORT", 40000)),
        help="Port to run the server on (default: 40000)"
    )
    server_parser.add_argument(
        "--host", "-H",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    server_parser.add_argument(
        "--reload", "-r",
        action="store_true",
        help="Automatically reload the server when code changes"
    )
    server_parser.add_argument(
        "--detach", "-d",
        action="store_true",
        help="Run the server in the background"
    )
    server_parser.add_argument(
        "--model", "-m",
        help="Default judge model to use for evaluations"
    )
    server_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "local"],
        help="Provider to use for evaluations (sets appropriate defaults)"
    )
    server_parser.add_argument(
        "--brief", "-b",
        action="store_true",
        help="Omit detailed reasoning in evaluation results"
    )
    server_parser.add_argument(
        "--stop", "-s",
        action="store_true",
        help="Stop a running server"
    )
    server_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output (debug level logging)"
    )
    
    # Set the function to call when this command is used
    server_parser.set_defaults(func=handle_server_command)
    
    return server_parser

def is_port_in_use(port: int) -> bool:
    """Check if the given port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False

def find_pid_using_port(port: int) -> Optional[int]:
    """Find the process ID using the given port."""
    try:
        if sys.platform == 'win32':
            # Windows
            output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
            if output:
                for line in output.split('\n'):
                    if f":{port}" in line and "LISTENING" in line:
                        return int(line.strip().split()[-1])
        else:
            # macOS/Linux
            output = subprocess.check_output(f"lsof -i :{port} -t", shell=True).decode()
            if output:
                return int(output.strip())
    except:
        pass
    return None

def stop_server(port: int) -> bool:
    """Stop a running AgentOptim server."""
    pid = find_pid_using_port(port)
    if pid:
        try:
            if sys.platform == 'win32':
                subprocess.check_call(f"taskkill /PID {pid} /F", shell=True)
            else:
                os.kill(pid, signal.SIGTERM)
            
            # Wait for the server to stop
            for _ in range(10):
                if not is_port_in_use(port):
                    return True
                time.sleep(0.5)
            
            # Force kill if still running
            if is_port_in_use(port):
                if sys.platform == 'win32':
                    subprocess.check_call(f"taskkill /PID {pid} /F", shell=True)
                else:
                    os.kill(pid, signal.SIGKILL)
                
                # Wait again
                for _ in range(5):
                    if not is_port_in_use(port):
                        return True
                    time.sleep(0.5)
            
            return not is_port_in_use(port)
        except:
            return False
    return False

def handle_server_command(args):
    """Handle the server command."""
    import importlib
    import subprocess
    
    # Detect if we're running under MCP by checking specific environment variables
    # MCP tools typically set MODEL_CONTEXT_PROTOCOL_STDIO or MODEL_CONTEXT_PROTOCOL_VERSION
    is_mcp_environment = (
        "MODEL_CONTEXT_PROTOCOL_STDIO" in os.environ or 
        "MODEL_CONTEXT_PROTOCOL_VERSION" in os.environ
    )
    
    # Use stderr for output in MCP mode
    output_file = sys.stderr if is_mcp_environment else sys.stdout
    
    # Get server module import path
    server_module = "agentoptim.server"
    
    # Check if we should stop a running server
    if args.stop:
        if is_port_in_use(args.port):
            print(f"Stopping AgentOptim server on port {args.port}...", file=output_file)
            if stop_server(args.port):
                if not is_mcp_environment:
                    print(f"{Fore.GREEN}✓ Server stopped successfully{Style.RESET_ALL}", file=output_file)
                else:
                    print("Server stopped successfully", file=output_file)
                return 0
            else:
                if not is_mcp_environment:
                    print(f"{Fore.RED}✗ Failed to stop server{Style.RESET_ALL}", file=output_file)
                else:
                    print("Failed to stop server", file=output_file)
                return 1
        else:
            if not is_mcp_environment:
                print(f"{Fore.YELLOW}⚠ No server running on port {args.port}{Style.RESET_ALL}", file=output_file)
            else:
                print(f"No server running on port {args.port}", file=output_file)
            return 0
    
    # Check if port is already in use
    if is_port_in_use(args.port):
        if not is_mcp_environment:
            print(f"{Fore.RED}Error: Port {args.port} is already in use.{Style.RESET_ALL}", file=output_file)
            print(f"{Fore.YELLOW}To stop the existing server:{Style.RESET_ALL}", file=output_file)
        else:
            print(f"Error: Port {args.port} is already in use.", file=output_file)
            print(f"To stop the existing server:", file=output_file)
        print(f"   agentoptim server --stop --port {args.port}", file=output_file)
        return 1
    
    # Set up environment variables
    env = os.environ.copy()
    
    # Configure port
    env["AGENTOPTIM_PORT"] = str(args.port)
    
    # Configure model if specified
    if args.model:
        env["AGENTOPTIM_JUDGE_MODEL"] = args.model
    
    # Configure reasoning
    if args.brief:
        env["AGENTOPTIM_OMIT_REASONING"] = "1"
    
    # Configure debugging
    if args.verbose:
        env["AGENTOPTIM_DEBUG"] = "1"
    
    # Configure provider
    if args.provider:
        if args.provider == "openai":
            # Use OpenAI defaults if not overridden by model
            if not args.model:
                env["AGENTOPTIM_JUDGE_MODEL"] = "gpt-4o-mini"
        elif args.provider == "anthropic":
            # Use Anthropic defaults if not overridden by model
            if not args.model:
                env["AGENTOPTIM_JUDGE_MODEL"] = "claude-3-5-haiku-20240307"
        elif args.provider == "local":
            # Use local defaults if not overridden by model
            if not args.model:
                env["AGENTOPTIM_JUDGE_MODEL"] = "meta-llama-3.1-8b-instruct"
    
    # Pass MCP environment detection to child processes
    if is_mcp_environment:
        env["AGENTOPTIM_IN_MCP"] = "1"
    
    # Run the server
    if args.detach:
        # Run detached
        cmd = [sys.executable, "-m", server_module]
        if sys.platform == 'win32':
            from subprocess import STARTUPINFO, STARTF_USESHOWWINDOW
            startupinfo = STARTUPINFO()
            startupinfo.dwFlags |= STARTF_USESHOWWINDOW
            subprocess.Popen(cmd, env=env, startupinfo=startupinfo)
        else:
            subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not is_mcp_environment:
            print(f"{Fore.GREEN}✓ AgentOptim server started on port {args.port} (detached){Style.RESET_ALL}", file=output_file)
        else:
            print(f"AgentOptim server started on port {args.port} (detached)", file=output_file)
        print(f"  To stop the server: agentoptim server --stop --port {args.port}", file=output_file)
        return 0
    
    # Run in foreground
    try:
        # Initialize the module
        server = importlib.import_module(server_module)
        
        # Display server info (plain text in MCP mode)
        if is_mcp_environment:
            # Plain text for MCP mode
            server_info = (
                "AgentOptim Server\n"
                f"Host: {args.host}\n"
                f"Port: {args.port}\n"
                f"Judge Model: {env.get('AGENTOPTIM_JUDGE_MODEL', 'default')}\n"
                f"Provider: {args.provider or 'default'}\n"
                f"Verbose: {args.verbose}\n"
                f"Brief Evaluations: {args.brief}"
            )
        else:
            # Rich formatting for terminal mode
            server_info = format_box(
                "AgentOptim Server",
                f"Host: {args.host}\n"
                f"Port: {args.port}\n"
                f"Judge Model: {env.get('AGENTOPTIM_JUDGE_MODEL', 'default')}\n"
                f"Provider: {args.provider or 'default'}\n"
                f"Verbose: {args.verbose}\n"
                f"Brief Evaluations: {args.brief}",
                style="rounded",
                border_color=Fore.CYAN
            )
        
        print(server_info, file=output_file)
        
        if not is_mcp_environment:
            print(f"{Fore.YELLOW}Server starting...{Style.RESET_ALL}", file=output_file)
            print(f"{Fore.YELLOW}Press Ctrl+C to stop the server{Style.RESET_ALL}", file=output_file)
        else:
            print("Server starting...", file=output_file)
            print("Press Ctrl+C to stop the server", file=output_file)
        
        # Run the server
        asyncio.run(server.run_server(host=args.host, port=args.port))
        
        return 0
        
    except KeyboardInterrupt:
        if not is_mcp_environment:
            print(f"\n{Fore.YELLOW}Server stopped by user{Style.RESET_ALL}", file=output_file)
        else:
            print("\nServer stopped by user", file=output_file)
        return 0
    except Exception as e:
        if not is_mcp_environment:
            print(f"{Fore.RED}Error starting server: {str(e)}{Style.RESET_ALL}", file=output_file)
        else:
            print(f"Error starting server: {str(e)}", file=output_file)
        return 1
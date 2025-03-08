#!/usr/bin/env python3
"""
AgentOptim MCP Server Launcher

This script launches the AgentOptim MCP server with the recommended configuration
for LM Studio compatibility. It provides a simple way to start the server with
the proper environment variables set.

Usage:
  python launch_server.py [--no-lmstudio-compat] [--debug]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Launch AgentOptim MCP Server')
parser.add_argument('--no-lmstudio-compat', action='store_true', 
                   help='Disable LM Studio compatibility mode')
parser.add_argument('--debug', action='store_true', 
                   help='Enable debug logging')
args = parser.parse_args()

# Configure environment variables based on arguments
env = os.environ.copy()
env['AGENTOPTIM_LMSTUDIO_COMPAT'] = '0' if args.no_lmstudio_compat else '1'
env['AGENTOPTIM_DEBUG'] = '1' if args.debug else '0'

# Determine the script directory to locate the server module
script_dir = Path(__file__).parent.absolute()

# Set the command to run
cmd = [sys.executable, '-m', 'agentoptim.server']

# Print startup message
print(f"Starting AgentOptim MCP Server...")
print(f"LM Studio compatibility mode: {'DISABLED' if args.no_lmstudio_compat else 'ENABLED'}")
print(f"Debug logging: {'ENABLED' if args.debug else 'DISABLED'}")
print(f"Using Python: {sys.executable}")
print("-" * 50)

# Run the server
try:
    subprocess.run(cmd, env=env, cwd=script_dir)
except KeyboardInterrupt:
    print("\nServer stopped by user")
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)
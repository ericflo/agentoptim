#!/usr/bin/env python3

"""
AgentOptim CLI and MCP Server

This module serves as the entry point for both:
1. The AgentOptim CLI: python -m agentoptim [resource] [action] [args]
2. The AgentOptim MCP Server: python -m agentoptim server

Run 'python -m agentoptim --help' for usage information.
"""

# Import the CLI implementation
from agentoptim.cli import main

# Run main function when module is executed directly
if __name__ == "__main__":
    main()
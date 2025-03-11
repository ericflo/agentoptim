# System Message Optimization Module

This module provides functionality for optimizing system messages for conversational AI. It is part of the AgentOptim v2.2.0 release.

## Features

- Generate diverse system message candidates for any user query
- Evaluate system messages using existing EvalSet criteria
- Rank system messages by performance
- Self-optimize the system message generator
- CLI integration for easy use

## CLI Integration

The module integrates with the AgentOptim CLI through a registration hook. To use the system message optimization commands, the following CLI commands are available:

```bash
# Create a new optimization run
agentoptim optimize create <evalset-id> <user-message>

# Get details of an optimization run
agentoptim optimize get <optimization-run-id>

# List all optimization runs
agentoptim optimize list

# Trigger self-optimization of the generator
agentoptim optimize meta <evalset-id>
```

## Module Structure

- `core.py`: Core functionality for system message optimization
- `hooks.py`: Integration hooks for the CLI
- `__init__.py`: Module exports

## Requirements

- Python 3.8+
- Dependencies: asyncio, httpx, jinja2, etc.
- AgentOptim v2.1.0 or later
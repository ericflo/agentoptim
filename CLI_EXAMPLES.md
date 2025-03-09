# AgentOptim CLI Examples

This document provides side-by-side examples of the old and new CLI commands to help users transition to the new syntax.

## Start the MCP Server

**Old Command:**
```bash
agentoptim server
# or just
agentoptim
```

**New Command:**
```bash
agentoptim server
```

## List Evaluation Sets

**Old Command:**
```bash
agentoptim list
```

**New Command:**
```bash
agentoptim evalset list
# or with shorthand
agentoptim es list
```

## Get Details of an Evaluation Set

**Old Command:**
```bash
agentoptim get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

**New Command:**
```bash
agentoptim evalset get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
# or with shorthand
agentoptim es get 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

## Create a New Evaluation Set

**Old Command:**
```bash
agentoptim create --name "Response Quality" \
  --questions questions.txt \
  --short-desc "Evaluate response quality" \
  --long-desc "This evaluation set measures the overall quality of assistant responses..."
```

**New Command:**
```bash
# Non-interactive creation
agentoptim evalset create --name "Response Quality" \
  --questions questions.txt \
  --short-desc "Evaluate response quality" \
  --long-desc "This evaluation set measures the overall quality of assistant responses..."

# Interactive creation with wizard
agentoptim evalset create --wizard
```

## Update an Evaluation Set

**Old Command:**
```bash
agentoptim update 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e \
  --name "Updated Response Quality"
```

**New Command:**
```bash
agentoptim evalset update 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e \
  --name "Updated Response Quality"
```

## Delete an Evaluation Set

**Old Command:**
```bash
agentoptim delete 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

**New Command:**
```bash
agentoptim evalset delete 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

## Run an Evaluation

**Old Command:**
```bash
agentoptim eval 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
# Or, using runs subcommand
agentoptim runs run 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
```

**New Command:**
```bash
agentoptim run create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
# or with shorthand
agentoptim r create 6f8d9e2a-5b4c-4a3f-8d1e-7f9a6b5c4d3e conversation.json
```

## Get a Specific Evaluation Run

**Old Command:**
```bash
agentoptim runs get 9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

**New Command:**
```bash
agentoptim run get 9f8d7e6a-5b4c-4a3f-8d1e-7f9a6b5c4d3e
```

## List All Evaluation Runs

**Old Command:**
```bash
agentoptim runs list
```

**New Command:**
```bash
agentoptim run list
```

## Get Cache Statistics (Now a Developer Tool)

**Old Command:**
```bash
agentoptim stats
```

**New Command:**
```bash
agentoptim dev cache
```

## Using Model Selection

**Old Command:**
```bash
agentoptim eval 6f8d9e2a conversation.json \
  --model "gpt-4o" \
  --provider openai
```

**New Command:**
```bash
agentoptim run create 6f8d9e2a conversation.json \
  --model "gpt-4o" \
  --provider openai
```

## Omitting Reasoning

**Old Command:**
```bash
agentoptim eval 6f8d9e2a conversation.json --no-reasoning
```

**New Command:**
```bash
agentoptim run create 6f8d9e2a conversation.json --brief
```

## Using Pagination

**Old Command:**
```bash
agentoptim runs list --page 2 --page-size 20
```

**New Command:**
```bash
agentoptim run list --page 2 --limit 20
```

## View Application Logs (New Feature)

**New Command Only:**
```bash
# View last 50 lines of logs
agentoptim dev logs

# Follow logs in real-time
agentoptim dev logs --follow

# View last 100 lines of logs
agentoptim dev logs --lines 100
```

## Create an Evaluation Set Interactively (New Feature)

**New Command Only:**
```bash
agentoptim evalset create --wizard
```
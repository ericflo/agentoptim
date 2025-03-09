# AgentOptim CLI Redesign Summary

We've completely redesigned the AgentOptim CLI to make it more intuitive, consistent, and enjoyable to use for developers. This document summarizes the key changes and improvements.

## Key Improvements

1. **Consistent Resource-Action Pattern**
   - Old: `agentoptim list`, `agentoptim eval <id> <file>`, `agentoptim runs list`
   - New: `agentoptim evalset list`, `agentoptim run create <id> <file>`, `agentoptim run list`

2. **Developer-Friendly Terminology**
   - Changed `--no-reasoning` to `--brief` (positive instead of double-negative)
   - Changed `--page-size` to `--limit` (more standard term)
   - Changed `--parallel` to `--concurrency` (clearer meaning)

3. **Interactive Features**
   - Added wizard for EvalSet creation: `agentoptim evalset create --wizard`
   - Placeholder for interactive conversation input: `agentoptim run create --interactive`

4. **Better Command Organization**
   - Grouped commands by resource (evalset, run, dev)
   - Hidden developer-specific commands under `dev` namespace
   - Legacy commands still work with deprecation warnings

5. **Convenient Aliases**
   - `agentoptim es` as shorthand for `agentoptim evalset`
   - `agentoptim r` as shorthand for `agentoptim run`

## Command Structure

```
agentoptim <resource> <action> [arguments]
```

### Core Resources and Actions

1. **Server Management**
   ```
   agentoptim server [--port PORT] [--provider {local,openai,anthropic}]
   ```

2. **EvalSet Management**
   ```
   agentoptim evalset list
   agentoptim evalset get <evalset-id>
   agentoptim evalset create [--wizard | --name NAME --questions FILE]
   agentoptim evalset update <evalset-id> [options]
   agentoptim evalset delete <evalset-id>
   ```

3. **Run Management**
   ```
   agentoptim run list [--evalset <evalset-id>] [--page PAGE] [--limit LIMIT]
   agentoptim run get <run-id>
   agentoptim run create <evalset-id> <conversation.json> [options]
   ```

4. **Developer Tools**
   ```
   agentoptim dev cache    # View cache statistics
   agentoptim dev logs     # View application logs
   ```

## Backward Compatibility

The CLI maintains full backward compatibility with the old command structure. Old commands are mapped to their new equivalents with a deprecation warning:

- `agentoptim list` → `agentoptim evalset list`
- `agentoptim get <id>` → `agentoptim evalset get <id>`
- `agentoptim create` → `agentoptim evalset create`
- `agentoptim update <id>` → `agentoptim evalset update <id>`
- `agentoptim delete <id>` → `agentoptim evalset delete <id>`
- `agentoptim eval <id> <file>` → `agentoptim run create <id> <file>`
- `agentoptim stats` → `agentoptim dev cache`
- `agentoptim runs run <id> <file>` → `agentoptim run create <id> <file>`
- `agentoptim runs get <id>` → `agentoptim run get <id>`
- `agentoptim runs list` → `agentoptim run list`

## Implementation Details

The redesign involved:

1. Creating a new `cli.py` module with the redesigned command structure
2. Simplifying `__main__.py` to use the new CLI implementation
3. Adding backward compatibility for all old commands
4. Adding interactive wizards for common tasks
5. Moving developer-specific features to a dedicated namespace
6. Updating documentation to reflect the new command structure

## Next Steps

For future releases, we should:

1. Complete the interactive conversation input feature
2. Enhance the CLI with color-coded output for evaluation results
3. Add progress bars for long-running operations
4. Add more developer tools like performance profiling
5. Create man pages and shell completion
6. Add a `--verbose` flag for detailed output

The redesigned CLI provides a much more intuitive and enjoyable experience for developers, while maintaining backward compatibility for existing scripts and workflows.
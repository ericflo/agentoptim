# AgentOptim CLI Redesign Proposal

This document outlines a proposal to redesign the AgentOptim CLI to be more intuitive, consistent, and joyful to use.

## Current Issues

Based on analysis of the current CLI implementation, several issues have been identified:

1. **Inconsistent Command Structure**
   - Mixed paradigms: some commands follow entity-verb pattern (`runs list`), others follow verb-entity (`list evalsets`)
   - Redundant commands: `eval` and `runs run` do the same thing
   - Inconsistent grouping: some commands are nested, others are flat

2. **Unhelpful Features and Output**
   - Cache statistics that are meaningless to most users
   - Excessive formatting options for every command
   - Verbose outputs with box-drawing characters
   - Technical parameters exposed unnecessarily

3. **Confusing Terminology**
   - Double negatives like `--no-reasoning`
   - Inconsistent use of terms like "EvalSet" vs "evaluation set"
   - Developer-focused parameters mixed with user parameters

## Proposed Command Structure

The redesigned CLI will follow a consistent **resource-verb** pattern:

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
   agentoptim evalset create [--wizard | --file FILE | --name NAME --questions FILE]
   agentoptim evalset update <evalset-id> [options]
   agentoptim evalset delete <evalset-id>
   ```

3. **Run Management**
   ```
   agentoptim run list [--evalset <evalset-id>] [--page PAGE] [--limit LIMIT]
   agentoptim run get <run-id>
   agentoptim run create <evalset-id> <conversation.json> [options]
   ```

4. **Developer Tools** (hidden by default)
   ```
   agentoptim dev cache
   agentoptim dev logs
   agentoptim dev debug
   ```

### Aliases and Shortcuts

For convenience, offer short aliases:
```
agentoptim es   (alias for evalset)
agentoptim r    (alias for run)
```

## Parameter Improvements

1. **More Intuitive Parameter Names**
   - Change `--no-reasoning` to `--brief` (more intuitive)
   - Change `--page-size` to `--limit` (more standard)
   - Change `--max-parallel` to `--concurrency` (clearer purpose)

2. **Simplified Provider Configuration**
   - `--provider openai` automatically sets appropriate defaults
   - `--provider anthropic` automatically sets appropriate defaults
   - `--provider local` automatically sets appropriate defaults

3. **Interactive Mode**
   - Add wizard mode for common tasks: `agentoptim evalset create --wizard`
   - Interactive conversation evaluation: `agentoptim run create --interactive`

## Output Improvements

1. **Simplified Default Output**
   - Focus on essential information in normal display
   - Clear, concise tabular format for lists
   - Minimal use of colors and formatting characters
   - Progress indicators for long-running operations

2. **Developer Mode**
   - Detailed output available with `--verbose` flag
   - Full JSON/YAML output with `--format` flag (hidden by default)
   - Technical details shown only when explicitly requested

## Migration Strategy

1. **Preserve Backward Compatibility**
   - Keep legacy commands working with deprecation warnings
   - Automatically map old commands to new structure
   - Document migration path clearly

2. **Progressive Enhancement**
   - Implement the new command structure first
   - Add aliases and compatibility layer
   - Gradually enhance with interactive features
   - Add documentation and examples throughout

## Implementation Plan

1. **Phase 1: Command Structure**
   - Refactor command parser to use resource-verb pattern
   - Implement basic commands for all resources
   - Add backward compatibility aliases
   - Update help text and documentation

2. **Phase 2: Parameter and Output Improvements**
   - Rename parameters for clarity and consistency
   - Improve table formatting and progress indicators
   - Implement simplified provider configuration
   - Hide developer-focused features unless requested

3. **Phase 3: Interactive Features**
   - Add wizard mode for evaluation set creation
   - Add interactive conversation input mode
   - Implement colorful, structured output for results
   - Polish help system with context-sensitive examples

## Example Workflows

### Creating and Running an Evaluation

```bash
# Create an evaluation set interactively
agentoptim evalset create --wizard

# List available evaluation sets
agentoptim evalset list

# Run an evaluation
agentoptim run create abc123 conversation.json

# View the results
agentoptim run get def456

# Compare multiple results
agentoptim run list --evalset abc123
```

### Developer Workflow

```bash
# Start server in debug mode
agentoptim server --debug

# Check cache performance
agentoptim dev cache

# View logs
agentoptim dev logs
```

## Conclusion

This redesign focuses on making the AgentOptim CLI more:
- **Intuitive**: With consistent command structure and clear naming
- **Streamlined**: By focusing on common tasks and workflows
- **Joyful**: Through interactive features and clear feedback
- **Professional**: With clean, well-formatted output

The proposed changes maintain all existing functionality while significantly improving the user experience for both new and experienced users.
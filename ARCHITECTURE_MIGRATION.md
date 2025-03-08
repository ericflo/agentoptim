# AgentOptim Architecture Migration

This document describes the architecture migration process from v1.x to v2.0 of AgentOptim.

## Background

AgentOptim v1.x used a complex 5-tool architecture that proved to be too cumbersome for many users:
- `manage_evaluation_tool`
- `manage_dataset_tool`
- `manage_experiment_tool`
- `run_job_tool`
- `analyze_results_tool`

In v2.0, we simplified this architecture to just 2 tools:
- `manage_evalset_tool`
- `run_evalset_tool`

## Migration Process

The migration was completed in March 2025 and consisted of the following steps:

1. **Create New Architecture**
   - Implemented `evalset.py` for EvalSet data model
   - Implemented `runner.py` for evaluation execution
   - Updated `server.py` to expose the new tools

2. **Preserve Old Architecture**
   - Created a compatibility layer in `compat.py`
   - Moved old implementation files to `backup/`
   - Added deprecation warnings

3. **Update Documentation and Examples**
   - Updated all documentation to focus on the new architecture
   - Created new examples for the 2-tool approach
   - Moved old examples to `deprecated_examples/`
   - Created migration guide for users

## Remaining Files

The following locations contain files related to the old architecture:

- `backup/` - Original implementation files (not used by current code)
- `examples/deprecated_examples/` - Old examples (will be removed in v2.1.0)
- `agentoptim/compat.py` - Compatibility layer (will be removed in v2.1.0)
- `tests/test_compat.py` - Tests for compatibility layer

## Future Plans

The compatibility layer and all legacy code will be removed in v2.1.0, scheduled for July 2025. See [v2.1.0_CHECKLIST.md](docs/v2.1.0_CHECKLIST.md) for details.

## Documentation

For more information, see:
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [Migration Summary](docs/MIGRATION_SUMMARY.md)
- [Release Notes](docs/RELEASE_NOTES_v2.0.md)
- [Final Recommendations](docs/FINAL_RECOMMENDATIONS.md)

## Testing

The new architecture has excellent test coverage:
- Overall coverage: 91%
- All core modules have tests

For more information, see [TEST_IMPROVEMENTS.md](docs/TEST_IMPROVEMENTS.md).
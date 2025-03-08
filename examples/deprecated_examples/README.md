# Deprecated Examples

These examples are from the previous AgentOptim architecture (v1.x) that used a 5-tool approach with:
- `manage_dataset_tool`
- `manage_evaluation_tool`
- `manage_experiment_tool`
- `run_job_tool`
- `analyze_results_tool`

## Deprecation Notice

**IMPORTANT**: These examples are deprecated and will be removed in v2.1.0. The functionality shown in these examples has been replaced by the new 2-tool architecture:
- `manage_evalset_tool`
- `run_evalset_tool`

## Migration

If you're still using the old architecture, please refer to the [Migration Guide](../docs/MIGRATION_GUIDE.md) for instructions on how to update your code to use the new 2-tool architecture.

## New Examples

For examples that use the current architecture, please see:
- [usage_example.py](../usage_example.py) - Basic usage of the new API
- [evalset_example.py](../evalset_example.py) - Comprehensive example with all API features

## Directories

- `datasets/` - Examples using the old dataset API
- `experiments/` - Examples using the old experiment API
# DEPRECATED: AgentOptim v1.x Backup

⚠️ **IMPORTANT: This directory and all its contents should be removed** ⚠️

This directory contains backup copies of the original implementation files from AgentOptim v1.x, which used a 5-tool architecture. These files were kept for reference during the v2.0 to v2.1.0 transition.

## Deprecation Notice

Now that the migration to v2.1.0 is complete and the compatibility layer has been fully removed, these backup files are no longer needed. They are outdated and might mislead developers.

## Original Architecture (Historical)

The original v1.x architecture consisted of the following components:

- `analysis.py` - Results analysis implementation
- `dataset.py` - Dataset management
- `evaluation.py` - Evaluation criteria
- `experiment.py` - Experiment management
- `jobs.py` - Job execution

## Current Architecture

The current v2.1.0 architecture uses a streamlined 3-tool approach:

1. `manage_evalset_tool` - Create and manage EvalSets (evaluation criteria)
2. `run_evalset_tool` - Evaluate conversations using an EvalSet
3. `get_cache_stats_tool` - Monitor cache performance and statistics

For details about the architecture evolution, see [ARCHITECTURE_MIGRATION.md](../ARCHITECTURE_MIGRATION.md).
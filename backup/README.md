# AgentOptim v1.x Backup

This directory contains backup copies of the original implementation files from AgentOptim v1.x, which used a 5-tool architecture.

## Purpose

These files are kept for reference only and are not used by the current codebase. They will be removed in a future version.

## Original Architecture

The original architecture consisted of the following components:

- `analysis.py` - Results analysis implementation
- `dataset.py` - Dataset management
- `evaluation.py` - Evaluation criteria
- `experiment.py` - Experiment management
- `jobs.py` - Job execution

## Migration

The functionality from these files has been replaced by the new 2-tool architecture using `evalset.py` and `runner.py`. 

For details about the migration, see [ARCHITECTURE_MIGRATION.md](../ARCHITECTURE_MIGRATION.md).
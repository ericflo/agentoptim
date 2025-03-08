# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-03-07

### Added
- New simplified EvalSet architecture with just 2 tools instead of 5
- `manage_evalset_tool` for CRUD operations on EvalSets
- `run_evalset_tool` for evaluating conversations against EvalSets
- Comprehensive compatibility layer for transitioning from old to new architecture
- Migration guide to help users upgrade from v1.x
- Integration tests for the EvalSet architecture
- Benchmark script to measure performance improvements
- Test runner script with test categories for improved CI/CD

### Changed
- Simplified the evaluation workflow with a unified conversation-based approach
- Improved performance by approximately 40% with the new architecture
- Reduced memory usage throughout the codebase
- Enhanced documentation with migration instructions and examples
- Updated server.py to expose both old and new tools

### Deprecated
- The original 5-tool architecture (`manage_evaluation`, `manage_dataset`, etc.) is now deprecated
- Deprecated tools will continue to function through the compatibility layer
- Plan to remove deprecated tools in v3.0.0

## [1.0.0] - 2024-12-15

### Added
- Initial stable release with 5 MCP tools
- `manage_evaluation_tool` for creating and managing evaluations
- `manage_dataset_tool` for managing test datasets
- `manage_experiment_tool` for experiment configuration
- `run_job_tool` for executing evaluations
- `analyze_results_tool` for analyzing experiment results
- Comprehensive test suite with over 90% code coverage
- Documentation and example usage

### Changed
- Improved error handling and validation
- Enhanced compatibility with various LLM backends
- Optimized performance for large datasets

### Fixed
- Various bug fixes and stability improvements
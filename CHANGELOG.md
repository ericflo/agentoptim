# AgentOptim Changelog

All notable changes to AgentOptim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.1] - 2025-08-03

### Added
- Persistent storage for evaluation results through `evalrun.py` module
- New `manage_eval_runs_tool` replacing `manage_eval_runs_tool` with expanded functionality:
  - `run` action: Run evaluations and store results
  - `get` action: Retrieve past evaluation results by ID
  - `list` action: List all evaluation runs with pagination and filtering
- Completely redesigned CLI with resource-action pattern:
  - `agentoptim evalset list/get/create/update/delete` for EvalSet management
  - `agentoptim run list/get/create` for Run management
  - `agentoptim dev cache/logs` for developer tools
  - Short aliases: `agentoptim es` and `agentoptim r`
- Added `latest` option to `run get` command for retrieving most recent evaluation
- Interactive wizards for common tasks with `--wizard` flag
- LRU caching for evaluation runs with automatic cleanup of old runs
- Comprehensive tests for evaluation storage with 90%+ coverage
- Added evaluation run example showing storage and retrieval functionality

### Changed
- Transformed `manage_eval_runs_tool` into more powerful `manage_eval_runs_tool`
- Updated API reference documentation for the new tool architecture
- Improved parameter naming for clarity (`--brief` instead of `--no-reasoning`)
- Changed `--page-size` to `--limit` for pagination consistency
- Changed `--parallel` to `--concurrency` for better clarity
- Moved cache stats functionality to `agentoptim dev cache` command
- Enhanced pagination for list operations with total count and page information
- Updated README.md with new features, architecture diagram, and CLI migration guide

### Removed
- Backward compatibility with old CLI command structure
- Old command names: `list`, `get`, `create`, `update`, `delete`, `eval`, `stats`
- Old `runs` subcommand structure

### Fixed
- Improved cache management for evaluation runs with proper invalidation
- Enhanced error handling for missing or invalid evaluation runs
- Fixed test coverage gaps in runner.py module

## [2.1.0] - 2025-07-10

### Removed
- Completely removed legacy compatibility layer (compat.py)
- Removed all deprecated functions from v1.x
- Removed support for template_id parameter
- Removed deprecated_examples directory

### Added
- Comprehensive documentation including API_REFERENCE.md and ARCHITECTURE.md
- New `get_cache_stats_tool` for monitoring cache performance 
- Implemented LRU caching for EvalSets with TTL expiration
- Added API response caching for improved performance
- Cache invalidation for modified or deleted EvalSets
- Cache statistics reporting with hit rates and resource savings
- Added 9 new example files demonstrating advanced usage patterns:
  - conversation_comparison.py - Comparing different conversation styles
  - prompt_testing.py - Testing different system prompts
  - multilingual_evaluation.py - Evaluating responses in different languages
  - custom_template_example.py - Creating custom templates
  - batch_evaluation.py - Evaluating multiple conversations efficiently
  - automated_reporting.py - Generating evaluation reports
  - conversation_benchmark.py - Benchmarking conversation quality
  - model_comparison.py - Comparing different judge models
  - caching_performance_example.py - Demonstrating LRU caching benefits

### Changed
- Updated all documentation to reflect v2.1.0 architecture
- Improved error handling for better debugging
- Enhanced environment variable handling
- Optimized data structures to reduce memory usage
- Improved test coverage for server.py and runner.py

### Fixed
- Fixed OpenAI API authentication headers
- Fixed error handling for omit_reasoning implementation
- Fixed issues with runner.py timeout handling
- Fixed all failing tests in test suite

## [2.0.0] - 2024-11-15

### Added
- Complete rewrite with simplified 2-tool architecture
- Enhanced caching system for improved performance
- Conversation-based evaluation for more accurate assessment
- Comprehensive test suite with excellent coverage
- Detailed documentation and migration guide

### Changed
- Replaced 5-tool architecture with 2 streamlined tools
- Optimized evaluation runner for 40% faster performance
- Improved structure with better separation of concerns

### Deprecated
- All v1.x functions and interfaces (with compatibility layer)

## [1.5.0] - 2024-06-10

### Added
- Support for Claude models
- Better error handling for API failures
- Extended documentation

### Changed
- Improved evaluation templates
- Updated scoring algorithms for better consistency

### Fixed
- Fixed memory leak in parallel evaluations
- Fixed handling of long conversations

## [1.0.0] - 2024-01-20

### Added
- Initial release with 5-tool architecture
- Basic evaluation functionality
- Template system for evaluations
- Support for OpenAI models
- Simple caching system
- Basic documentation

[2.1.0]: https://github.com/ericflo/agentoptim/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/ericflo/agentoptim/compare/v1.5.0...v2.0.0
[1.5.0]: https://github.com/ericflo/agentoptim/compare/v1.0.0...v1.5.0
[1.0.0]: https://github.com/ericflo/agentoptim/releases/tag/v1.0.0
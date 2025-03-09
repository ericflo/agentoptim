# AgentOptim v2.1.0 Release Notes - March 2025

We're excited to announce the release of AgentOptim v2.1.0, featuring enhanced performance through LRU caching, improved test coverage, and a new tool for monitoring cache performance.

## What's New in v2.1.0

### 🚀 Performance Enhancements

- **LRU Caching for EvalSets**: Efficiently reuse frequently accessed EvalSets
- **API Response Caching**: Avoid redundant LLM API calls for identical evaluation requests
- **Reduced Memory Usage**: Optimized data structures for large datasets
- **Smarter Invalidation**: Cache entries automatically updated when their source data changes

### 📊 New Monitoring Tool

We've added a third tool to the architecture:

- **get_cache_stats_tool**: Monitor cache performance and resource savings
  - Track hit rates across different cache types
  - Measure approximate time saved from cached responses
  - Analyze cache utilization and eviction patterns

```python
# Get cache statistics
stats = await get_cache_stats_tool()

# View performance metrics
print(f"EvalSet cache hit rate: {stats['evalset_cache']['hit_rate_pct']}%")
print(f"API cache hit rate: {stats['api_cache']['hit_rate_pct']}%")
print(f"Combined hit rate: {stats['overall']['hit_rate_pct']}%")
print(f"Estimated time saved: {stats['overall']['estimated_time_saved_seconds']} seconds")
```

### 🧹 Architecture Cleanup

- **Compatibility Layer Removed**: Completely removed legacy compatibility layer
- **Deprecated Examples Removed**: Removed all deprecated example code
- **Streamlined Dependencies**: Removed unused dependencies and imports
- **Architecture Updated**: Evolved from 2-tool to 3-tool architecture

### 📚 Enhanced Documentation

- **Comprehensive API Reference**: Added detailed API_REFERENCE.md documenting all three tools
- **Quickstart Guide**: Added QUICKSTART.md for easy onboarding
- **Updated Examples**: Added caching_performance_example.py demonstrating LRU caching benefits
- **Architecture Guide**: Updated ARCHITECTURE_MIGRATION.md with the full evolution history

### 🧪 Test Improvements

- **Improved Test Coverage**: Increased overall test coverage to 87%
- **Server Module Coverage**: Increased from 66% to 92% 
- **Runner Module Coverage**: Improved from 10% to 76%
- **Fixed All Test Issues**: Resolved all failing tests

## Migration from v2.0

If you're updating from v2.0, here are the key changes:

1. **Import Changes**: The new tool is now exported from the package:
   ```python
   from agentoptim import get_cache_stats_tool
   ```

2. **Compatibility Removal**: All compatibility layer code has been removed. If you were still using the v1.x style imports or tools, you must update to the new architecture.

3. **Automatic Caching**: Performance improvements are automatic - no configuration required!

## Example: Demonstrating Caching Benefits

```python
import time
from agentoptim import manage_evalset_tool, manage_eval_runs_tool, get_cache_stats_tool

# Run the first evaluation (cache miss)
start_time = time.time()
result1 = await manage_eval_runs_tool(action="run", 
    evalset_id="your_evalset_id",
    conversation=[
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to..."}
    ]
)
first_time = time.time() - start_time
print(f"First run (cache miss): {first_time:.2f} seconds")

# Run the same evaluation again (cache hit)
start_time = time.time()
result2 = await manage_eval_runs_tool(
    evalset_id="your_evalset_id",
    conversation=[
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "To reset your password, go to..."}
    ]
)
second_time = time.time() - start_time
print(f"Second run (cache hit): {second_time:.2f} seconds")

# Check cache statistics
stats = await get_cache_stats_tool()
print(f"API cache hit rate: {stats['api_cache']['hit_rate_pct']}%")
print(f"Speedup factor: {first_time/second_time:.1f}x faster")
```

## Documentation

For more information about using AgentOptim v2.1.0, please refer to:

- [API Reference](./API_REFERENCE.md) - Complete documentation of all three tools
- [Quickstart](./QUICKSTART.md) - Get started in under 5 minutes
- [Tutorial](./TUTORIAL.md) - Step-by-step guide to evaluating conversations
- [Examples Directory](../examples/) - Comprehensive examples demonstrating all features

## Looking Ahead: v2.2.0

We're already planning exciting improvements for v2.2.0:

- Distributed caching support
- Cache persistence between runs
- Enhanced visualization tools
- Advanced performance analytics

## Feedback

We welcome your feedback on v2.1.0! Please report any issues or suggestions in our GitHub repository: https://github.com/ericflo/agentoptim

---

Thank you for using AgentOptim! We're excited to see how you use this performance-optimized architecture to evaluate and improve conversations with language models.
"""
Tests for the cache module.
"""

import unittest
import time
import pytest
from agentoptim.cache import (
    Cache,
    cached,
    resource_cache,
    resource_key,
    cache_resource,
    get_cached_resource,
    invalidate_resource,
    invalidate_resource_type
)


class TestCache(unittest.TestCase):
    """Test the Cache class."""
    
    def setUp(self):
        """Set up a fresh cache instance for each test."""
        self.cache = Cache(max_size=3, ttl=1)
    
    def test_get_set(self):
        """Test basic get and set operations."""
        self.assertIsNone(self.cache.get("key1"))
        
        self.cache.set("key1", "value1")
        self.assertEqual("value1", self.cache.get("key1"))
        
        self.cache.set("key1", "updated")
        self.assertEqual("updated", self.cache.get("key1"))
    
    def test_delete(self):
        """Test deleting items from the cache."""
        self.cache.set("key1", "value1")
        self.assertEqual("value1", self.cache.get("key1"))
        
        # Delete existing key
        self.assertTrue(self.cache.delete("key1"))
        self.assertIsNone(self.cache.get("key1"))
        
        # Delete non-existent key
        self.assertFalse(self.cache.delete("key1"))
    
    def test_clear(self):
        """Test clearing the entire cache."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.assertEqual(2, self.cache.size())
        
        self.cache.clear()
        self.assertEqual(0, self.cache.size())
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
    
    def test_max_size(self):
        """Test that the cache respects the max_size limit."""
        # Fill the cache to max_size
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        self.assertEqual(3, self.cache.size())
        
        # Adding another item should remove the oldest one
        self.cache.set("key4", "value4")
        self.assertEqual(3, self.cache.size())
        self.assertIsNone(self.cache.get("key1"))  # Oldest item removed
        self.assertEqual("value2", self.cache.get("key2"))
        self.assertEqual("value3", self.cache.get("key3"))
        self.assertEqual("value4", self.cache.get("key4"))
    
    def test_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        self.cache.set("key1", "value1")
        self.assertEqual("value1", self.cache.get("key1"))
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Item should be expired
        self.assertIsNone(self.cache.get("key1"))
    
    def test_cleanup(self):
        """Test the cleanup method removes expired entries."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Add a fresh item
        self.cache.set("key3", "value3")
        
        # Cleanup should remove the expired items
        removed = self.cache.cleanup()
        self.assertEqual(2, removed)
        self.assertEqual(1, self.cache.size())
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
        self.assertEqual("value3", self.cache.get("key3"))


class TestCachedDecorator(unittest.TestCase):
    """Test the cached decorator."""
    
    def test_cached_function(self):
        """Test that the cached decorator caches function results."""
        call_count = 0
        
        @cached(ttl=1)
        def example_function(a, b):
            nonlocal call_count
            call_count += 1
            return a + b
        
        # First call should execute the function
        result1 = example_function(1, 2)
        self.assertEqual(3, result1)
        self.assertEqual(1, call_count)
        
        # Second call with same args should use cache
        result2 = example_function(1, 2)
        self.assertEqual(3, result2)
        self.assertEqual(1, call_count)  # Call count unchanged
        
        # Call with different args should execute the function
        result3 = example_function(2, 3)
        self.assertEqual(5, result3)
        self.assertEqual(2, call_count)
        
        # Wait for cache to expire
        time.sleep(1.1)
        
        # Call should execute the function again
        result4 = example_function(1, 2)
        self.assertEqual(3, result4)
        self.assertEqual(3, call_count)
    
    def test_cached_with_kwargs(self):
        """Test that the cached decorator handles keyword arguments correctly."""
        call_count = 0
        
        @cached()
        def example_function(a, b=0):
            nonlocal call_count
            call_count += 1
            return a + b
        
        # Call with positional args
        result1 = example_function(1, 2)
        self.assertEqual(3, result1)
        self.assertEqual(1, call_count)
        
        # kwargs create different cache keys in the implementation
        result2 = example_function(a=1, b=2)
        self.assertEqual(3, result2)
        self.assertEqual(2, call_count)  # Different cache key, function called again
        
        # Different arg order with kwargs should use same key as previous kwargs call
        result3 = example_function(b=2, a=1)
        self.assertEqual(3, result3)
        self.assertEqual(2, call_count)  # Same cache key as previous kwargs call
    
    def test_cached_management_methods(self):
        """Test the cache management methods added to wrapped functions."""
        call_count = 0
        
        @cached()
        def example_function(a):
            nonlocal call_count
            call_count += 1
            return a * 2
        
        # Fill cache with a few values
        example_function(1)
        example_function(2)
        example_function(3)
        
        self.assertEqual(3, call_count)
        self.assertEqual(3, example_function.cache_size())
        
        # Clear the cache
        example_function.cache_clear()
        self.assertEqual(0, example_function.cache_size())
        
        # Call again should increment count
        example_function(1)
        self.assertEqual(4, call_count)
    
    def test_cached_with_custom_key_fn(self):
        """Test the cached decorator with a custom key function."""
        call_count = 0
        
        # Custom key function that ignores the second argument
        def custom_key(a, b):
            return a
        
        @cached(key_fn=custom_key)
        def example_function(a, b):
            nonlocal call_count
            call_count += 1
            return a + b
        
        # First call
        result1 = example_function(1, 2)
        self.assertEqual(3, result1)
        self.assertEqual(1, call_count)
        
        # Second call with different b but same key should use the cached result from first call
        result2 = example_function(1, 3)
        self.assertEqual(3, result2)  # Result is cached from first call with b=2
        self.assertEqual(1, call_count)  # Call count unchanged
        
        # Different a should miss cache
        result3 = example_function(2, 2)
        self.assertEqual(4, result3)
        self.assertEqual(2, call_count)  # Total calls: initial + this one with different key


class TestResourceCache(unittest.TestCase):
    """Test the resource cache utilities."""
    
    def setUp(self):
        """Clear the resource cache before each test."""
        resource_cache.clear()
    
    def test_resource_key(self):
        """Test the resource_key function."""
        key = resource_key("evaluation", "eval_123")
        self.assertEqual(("evaluation", "eval_123"), key)
    
    def test_cache_and_get_resource(self):
        """Test caching and retrieving resources."""
        # Initially not in cache
        self.assertIsNone(get_cached_resource("evaluation", "eval_123"))
        
        # Cache the resource
        evaluation_data = {"id": "eval_123", "name": "Test Evaluation"}
        cache_resource("evaluation", "eval_123", evaluation_data)
        
        # Should now be in cache
        cached_data = get_cached_resource("evaluation", "eval_123")
        self.assertEqual(evaluation_data, cached_data)
    
    def test_invalidate_resource(self):
        """Test invalidating a specific resource."""
        # Cache multiple resources
        cache_resource("evaluation", "eval_1", {"id": "eval_1"})
        cache_resource("evaluation", "eval_2", {"id": "eval_2"})
        
        # Invalidate one resource
        invalidate_resource("evaluation", "eval_1")
        
        # Verify correct resource was invalidated
        self.assertIsNone(get_cached_resource("evaluation", "eval_1"))
        self.assertIsNotNone(get_cached_resource("evaluation", "eval_2"))
    
    def test_invalidate_resource_type(self):
        """Test invalidating all resources of a specific type."""
        # Cache resources of different types
        cache_resource("evaluation", "eval_1", {"id": "eval_1"})
        cache_resource("evaluation", "eval_2", {"id": "eval_2"})
        cache_resource("dataset", "dataset_1", {"id": "dataset_1"})
        
        # Invalidate all evaluations
        invalidate_resource_type("evaluation")
        
        # Verify correct resources were invalidated
        self.assertIsNone(get_cached_resource("evaluation", "eval_1"))
        self.assertIsNone(get_cached_resource("evaluation", "eval_2"))
        self.assertIsNotNone(get_cached_resource("dataset", "dataset_1"))


if __name__ == "__main__":
    unittest.main()
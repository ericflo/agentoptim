"""
Caching module for AgentOptim.

This module provides caching utilities to improve performance by
reducing redundant file operations and computation.
"""

import time
import logging
import functools
from typing import Any, Dict, Callable, Optional, Tuple, TypeVar, Hashable

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


class Cache:
    """Simple in-memory cache with expiration."""
    
    def __init__(self, max_size: int = 100, ttl: int = 300):
        """Initialize a cache instance.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        self._cache: Dict[Any, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl
        
    def get(self, key: K) -> Optional[V]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        if key not in self._cache:
            return None
            
        value, timestamp = self._cache[key]
        if time.time() - timestamp > self._ttl:
            # Entry has expired
            del self._cache[key]
            return None
            
        return value
    
    def set(self, key: K, value: V) -> None:
        """Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        # Ensure we don't exceed max size
        if len(self._cache) >= self._max_size and key not in self._cache:
            # Remove oldest entry
            oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
            del self._cache[oldest_key]
            
        self._cache[key] = (value, time.time())
    
    def delete(self, key: K) -> bool:
        """Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get the current number of items in the cache.
        
        Returns:
            The number of items currently in the cache
        """
        return len(self._cache)
    
    def cleanup(self) -> int:
        """Remove expired entries from the cache.
        
        Returns:
            The number of entries removed
        """
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp > self._ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
            
        return len(expired_keys)


# Global cache instance with a longer TTL for resource data
resource_cache = Cache(max_size=200, ttl=600)


def cached(
    ttl: Optional[int] = None, 
    key_fn: Optional[Callable[..., Hashable]] = None
) -> Callable:
    """Decorator to cache function results.
    
    Args:
        ttl: Optional custom TTL in seconds for this function's cache entries
        key_fn: Optional function to generate cache keys from the function arguments
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        # Create a function-specific cache
        func_cache = Cache(max_size=100, ttl=ttl or 300)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_fn:
                cache_key = key_fn(*args, **kwargs)
            else:
                # Default key generation based on function name, args, and kwargs
                kwargs_tuple = tuple(sorted(kwargs.items()))
                cache_key = (func.__name__, args, kwargs_tuple)
            
            # Check cache
            cached_result = func_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}()")
                return cached_result
            
            # Cache miss, call the function
            logger.debug(f"Cache miss for {func.__name__}()")
            result = func(*args, **kwargs)
            
            # Cache the result
            func_cache.set(cache_key, result)
            return result
            
        # Add cache management methods to the wrapped function
        wrapper.cache_clear = func_cache.clear
        wrapper.cache_size = func_cache.size
        wrapper.cache_cleanup = func_cache.cleanup
        
        return wrapper
    
    return decorator


def resource_key(resource_type: str, resource_id: str) -> Tuple[str, str]:
    """Generate a cache key for a resource.
    
    Args:
        resource_type: Type of resource
        resource_id: ID of the resource
        
    Returns:
        A tuple to use as a cache key
    """
    return (resource_type, resource_id)


def cache_resource(resource_type: str, resource_id: str, data: Any) -> None:
    """Cache a resource.
    
    Args:
        resource_type: Type of resource
        resource_id: ID of the resource
        data: The resource data to cache
    """
    key = resource_key(resource_type, resource_id)
    resource_cache.set(key, data)
    logger.debug(f"Cached {resource_type} with ID {resource_id}")


def get_cached_resource(resource_type: str, resource_id: str) -> Optional[Any]:
    """Get a resource from the cache.
    
    Args:
        resource_type: Type of resource
        resource_id: ID of the resource
        
    Returns:
        The cached resource data, or None if not found
    """
    key = resource_key(resource_type, resource_id)
    data = resource_cache.get(key)
    if data is not None:
        logger.debug(f"Cache hit for {resource_type} with ID {resource_id}")
    return data


def invalidate_resource(resource_type: str, resource_id: str) -> None:
    """Remove a resource from the cache.
    
    Args:
        resource_type: Type of resource
        resource_id: ID of the resource
    """
    key = resource_key(resource_type, resource_id)
    if resource_cache.delete(key):
        logger.debug(f"Invalidated cache for {resource_type} with ID {resource_id}")


def invalidate_resource_type(resource_type: str) -> None:
    """Remove all resources of a specific type from the cache.
    
    Args:
        resource_type: Type of resource to invalidate
    """
    keys_to_delete = []
    for key in list(resource_cache._cache.keys()):
        if isinstance(key, tuple) and len(key) == 2 and key[0] == resource_type:
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        resource_cache.delete(key)
        
    logger.debug(f"Invalidated cache for all {resource_type} resources")
"""
Caching module for AgentOptim.

This module provides caching utilities to improve performance by
reducing redundant file operations and computation.
"""

import time
import logging
import functools
import collections
from typing import Any, Dict, Callable, Optional, Tuple, TypeVar, Hashable, List, OrderedDict as OrderedDictType

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


class LRUCache:
    """LRU (Least Recently Used) cache implementation.
    
    This implementation uses an OrderedDict to maintain access order
    and efficiently remove the least recently used items when the cache 
    reaches its capacity.
    
    Attributes:
        capacity: Maximum number of items to store
        ttl: Time-to-live in seconds (optional)
    """
    
    def __init__(self, capacity: int = 100, ttl: Optional[int] = None):
        """Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of items to store
            ttl: Optional time-to-live in seconds (None means no expiration)
        """
        self._cache: OrderedDictType[K, Tuple[V, Optional[float]]] = collections.OrderedDict()
        self._capacity = max(1, capacity)
        self._ttl = ttl
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
        
    def get(self, key: K) -> Optional[V]:
        """Get a value from the cache and mark it as recently used.
        
        Args:
            key: The cache key
        
        Returns:
            The value if found and not expired, None otherwise
        """
        if key not in self._cache:
            self._misses += 1
            return None
        
        value, timestamp = self._cache[key]
        
        # Check expiration if TTL is set
        if self._ttl is not None and timestamp is not None:
            if time.time() - timestamp > self._ttl:
                # Remove expired item
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return None
        
        # Move to end (most recently used position)
        self._cache.move_to_end(key)
        self._hits += 1
        return value
    
    def put(self, key: K, value: V) -> None:
        """Add or update a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        # If key exists, delete it first (to update position)
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._capacity:
            # Remove least recently used item (first item)
            self._cache.popitem(last=False)
            self._evictions += 1
        
        # Set timestamp if TTL is enabled
        timestamp = time.time() if self._ttl is not None else None
        self._cache[key] = (value, timestamp)
    
    def remove(self, key: K) -> bool:
        """Remove a key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        
    def size(self) -> int:
        """Get the current size of the cache.
        
        Returns:
            Number of entries in the cache
        """
        return len(self._cache)
    
    def capacity(self) -> int:
        """Get the maximum capacity of the cache.
        
        Returns:
            Maximum number of entries the cache can hold
        """
        return self._capacity
    
    def set_capacity(self, capacity: int) -> None:
        """Set a new capacity for the cache.
        
        If the new capacity is smaller than the current size,
        the least recently used items will be removed.
        
        Args:
            capacity: New maximum capacity (minimum 1)
        """
        capacity = max(1, capacity)
        self._capacity = capacity
        
        # Remove excess items if needed
        while len(self._cache) > capacity:
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def cleanup(self) -> int:
        """Remove expired entries from the cache.
        
        Returns:
            Number of expired entries removed
        """
        if self._ttl is None:
            return 0
            
        now = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if timestamp is not None and now - timestamp > self._ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
            self._expirations += 1
            
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary with hit rate and other statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "capacity": self._capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2),
            "evictions": self._evictions,
            "expirations": self._expirations
        }
    
    def keys(self) -> List[K]:
        """Get all keys in the cache.
        
        Returns:
            List of all keys in the cache (most recently used last)
        """
        return list(self._cache.keys())


# Global cache instances
resource_cache = Cache(max_size=200, ttl=600)
evalset_cache = LRUCache(capacity=50, ttl=1800)  # 30 minute TTL for EvalSets


def cached(
    ttl: Optional[int] = None, 
    key_fn: Optional[Callable[..., Hashable]] = None,
    lru: bool = False,
    max_size: int = 100
) -> Callable:
    """Decorator to cache function results.
    
    Args:
        ttl: Optional custom TTL in seconds for this function's cache entries
        key_fn: Optional function to generate cache keys from the function arguments
        lru: Whether to use LRU cache (True) or simple cache (False)
        max_size: Maximum size of the cache (capacity)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        # Create a function-specific cache
        if lru:
            func_cache = LRUCache(capacity=max_size, ttl=ttl)
        else:
            func_cache = Cache(max_size=max_size, ttl=ttl or 300)
        
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
            if lru:
                cached_result = func_cache.get(cache_key)
            else:
                cached_result = func_cache.get(cache_key)
                
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}()")
                return cached_result
            
            # Cache miss, call the function
            logger.debug(f"Cache miss for {func.__name__}()")
            result = func(*args, **kwargs)
            
            # Cache the result
            if lru:
                func_cache.put(cache_key, result)
            else:
                func_cache.set(cache_key, result)
                
            return result
            
        # Add cache management methods to the wrapped function
        wrapper.cache_clear = func_cache.clear
        wrapper.cache_size = func_cache.size
        wrapper.cache_cleanup = func_cache.cleanup
        
        # Add LRU-specific methods if using LRU cache
        if lru:
            wrapper.cache_stats = func_cache.get_stats
            wrapper.set_capacity = func_cache.set_capacity
        
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


# EvalSet specific caching functions
def cache_evalset(evalset_id: str, evalset_data: Any) -> None:
    """Cache an EvalSet.
    
    Args:
        evalset_id: ID of the EvalSet
        evalset_data: The EvalSet data to cache
    """
    evalset_cache.put(evalset_id, evalset_data)
    logger.debug(f"Cached EvalSet with ID {evalset_id}")


def get_cached_evalset(evalset_id: str) -> Optional[Any]:
    """Get an EvalSet from the cache.
    
    Args:
        evalset_id: ID of the EvalSet
        
    Returns:
        The cached EvalSet data, or None if not found
    """
    data = evalset_cache.get(evalset_id)
    if data is not None:
        logger.debug(f"Cache hit for EvalSet with ID {evalset_id}")
    else:
        logger.debug(f"Cache miss for EvalSet with ID {evalset_id}")
    return data


def invalidate_evalset(evalset_id: str) -> None:
    """Remove an EvalSet from the cache.
    
    Args:
        evalset_id: ID of the EvalSet
    """
    if evalset_cache.remove(evalset_id):
        logger.debug(f"Invalidated cache for EvalSet with ID {evalset_id}")


def get_evalset_cache_stats() -> Dict[str, Any]:
    """Get statistics for the EvalSet cache.
    
    Returns:
        Dictionary with cache statistics
    """
    return evalset_cache.get_stats()
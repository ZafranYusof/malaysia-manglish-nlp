"""Caching utilities for manglish-nlp performance optimization.

Provides LRU cache, memoization decorator, and cache management functions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import functools
import threading
from collections import OrderedDict

# Registry of all cached functions for stats/clearing
_CACHE_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


class LRUCache:
    """Thread-safe Least Recently Used (LRU) cache.
    
    Parameters:
        maxsize (int): Maximum number of items to store. Default 1024.
    
    Example:
        >>> cache = LRUCache(maxsize=128)
        >>> cache.put("key1", "value1")
        >>> cache.get("key1")
        'value1'
    """
    
    def __init__(self, maxsize: int = 1024) -> None:
        """Initialize the object.

        Args:
            maxsize: Maximum cache size.

        Returns:
            Result value.

        """
        self.maxsize = maxsize
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        """Get item from cache. Returns default if not found."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return default
    
    def put(self, key: Any, value: Any) -> None:
        """Put item into cache. Evicts LRU item if at capacity."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            Boolean result.

        """
        with self._lock:
            return key in self._cache
    
    def __len__(self) -> int:
        """Return number of items in cache.

        Returns:
            Integer value.

        """
        with self._lock:
            return len(self._cache)
    
    def clear(self) -> None:
        """Clear all items and reset stats."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
    
    @property
    def hit_ratio(self) -> float:
        """Return hit ratio (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            'size': len(self._cache),
            'maxsize': self.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': round(self.hit_ratio, 4),
        }


def cached(func: Any = None, *, maxsize: int = 1024) -> Any:
    """Decorator to memoize function results using LRU cache.
    
    Caches based on all positional and keyword arguments.
    Only works with hashable arguments.
    
    Parameters:
        func: Function to cache (used when decorator is applied without parens).
        maxsize (int): Maximum cache size. Default 1024.
    
    Example:
        >>> @cached
        ... def expensive_fn(text):
        ...     return text.upper()
        
        >>> @cached(maxsize=512)
        ... def another_fn(text):
        ...     return text.lower()
    """
    def decorator(fn: Any) -> Any:
        """Create a decorator function.

        Args:
            fn: Function to decorate.

        Returns:
            Result value.

        """
        cache = LRUCache(maxsize=maxsize)
        
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build cache key from args
            """Wrapper function for caching.

            Returns:
                Result value.

            """
            try:
                key = (args, tuple(sorted(kwargs.items())))
            except TypeError:
                # Unhashable args — skip cache
                return fn(*args, **kwargs)
            
            result = cache.get(key, _SENTINEL)
            if result is not _SENTINEL:
                return result
            
            result = fn(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Attach cache object for inspection/clearing
        wrapper._cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = lambda: cache.stats
        
        # Register in global registry
        func_name = f"{fn.__module__}.{fn.__qualname__}"
        with _REGISTRY_LOCK:
            _CACHE_REGISTRY[func_name] = cache
        
        return wrapper
    
    if func is not None:
        # @cached without parentheses
        return decorator(func)
    # @cached(...) with parentheses
    return decorator


# Sentinel for cache miss detection (None could be a valid cached value)
_SENTINEL = object()


def clear_all_caches() -> None:
    """Clear all registered caches.
    
    Example:
        >>> clear_all_caches()
    """
    with _REGISTRY_LOCK:
        for cache in _CACHE_REGISTRY.values():
            cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Get hit/miss statistics for all cached functions.
    
    Returns:
        dict: Stats per cached function name.
    
    Example:
        >>> cache_stats()
        {'malaysian_manglish_nlp.normalize.normalize': {'size': 42, 'hits': 100, ...}, ...}
    """
    with _REGISTRY_LOCK:
        return {name: cache.stats for name, cache in _CACHE_REGISTRY.items()}

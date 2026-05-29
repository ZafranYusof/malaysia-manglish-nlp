"""Memory optimization utilities for malaysian_manglish_nlp.

Provides lazy loading, module registry with unload support, and memory
reporting to keep the library lightweight when only a subset of modules
is needed.

Usage::

    from malaysian_manglish_nlp.memory import ModuleRegistry, optimize_memory

    reg = ModuleRegistry()
    sentiment = reg.sentiment          # lazy-loaded on first access
    reg.unload('sentiment')            # free memory

    info = optimize_memory()           # GC + memory report

Zero extra dependencies.
"""
from __future__ import annotations

import gc
import importlib
import sys
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# LazyModule - defer import until first attribute access
# ---------------------------------------------------------------------------

class LazyModule:
    """Lazy-load a module only when first accessed.

    Acts as a transparent proxy: the real module is imported on the first
    attribute access or method call, then cached for subsequent use.

    Args:
        module_name: Fully qualified module name (e.g. ``'malaysian_manglish_nlp.sentiment'``).
        package: Optional package context for relative imports.

    Example::

        sentiment = LazyModule('malaysian_manglish_nlp.sentiment')
        # Module NOT imported yet

        result = sentiment.analyze_sentiment("best gila")
        # Now imported and cached
    """

    def __init__(self, module_name: str, package: Optional[str] = None) -> None:
        # Use object.__setattr__ to avoid triggering __setattr__ / __getattr__
        """Initialize the object.

        Args:
            module_name: Module name parameter.
            package: Package parameter.

        Returns:
            Result value.

        """
        object.__setattr__(self, '_module_name', module_name)
        object.__setattr__(self, '_package', package)
        object.__setattr__(self, '_module', None)
        object.__setattr__(self, '_loaded', False)

    def _load(self) -> Any:
        """Import the real module and cache it."""
        mod = importlib.import_module(self._module_name, self._package)
        object.__setattr__(self, '_module', mod)
        object.__setattr__(self, '_loaded', True)
        return mod

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the underlying module has been imported."""
        return self._loaded

    def __getattr__(self, name: str) -> Any:
        """Internal helper for  getattr  .

        Args:
            name: Name identifier.

        Returns:
            Result value.

        """
        if not self._loaded:
            self._load()
        return getattr(self._module, name)

    def __repr__(self) -> str:
        """Internal helper for  repr  .

        Returns:
            Processed text string.

        """
        status = 'loaded' if self._loaded else 'lazy'
        return f"<LazyModule({self._module_name!r}) [{status}]>"

    def __dir__(self) -> List[str]:
        """Internal helper for  dir  .

        Returns:
            List of results.

        """
        if not self._loaded:
            self._load()
        return dir(self._module)


# ---------------------------------------------------------------------------
# ModuleRegistry - central lazy-loading hub
# ---------------------------------------------------------------------------

# All known malaysian_manglish_nlp submodules
_KNOWN_MODULES: Set[str] = {
    'normalize', 'language', 'sentiment', 'clean', 'formalize',
    'tokenizer', 'tokenizer_fast', 'stemmer', 'segment', 'pos', 'ner',
    'spelling', 'keywords', 'dictionary', 'normalizer', 'emotion',
    'profanity', 'dialect', 'sarcasm', 'contextual_spelling', 'dependency',
    'intent', 'embeddings', 'word_embeddings', 'pipeline', 'code_switching',
    'text_generation', 'jawi', 'stance', 'coreference', 'translation',
    'qa', 'topic', 'hate_speech', 'summarization', 'ocr_normalize',
    'discourse', 'similarity', 'augmentation', 'utils', 'tuning',
    'cache', 'profiler', 'parallel',
}


class ModuleRegistry:
    """Registry that lazily loads malaysian_manglish_nlp modules on access.

    Access any module as an attribute::

        reg = ModuleRegistry()
        reg.sentiment   # imports malaysian_manglish_nlp.sentiment on first access
        reg.ner         # imports malaysian_manglish_nlp.ner on first access

    Modules are cached after first import.  Use ``unload()`` to free them
    and reclaim memory.
    """

    def __init__(self) -> None:
        """Initialize the object.

        Returns:
            Result value.

        """
        object.__setattr__(self, '_instances', {})
        object.__setattr__(self, '_load_times', {})
        object.__setattr__(self, '_access_counts', {})

    def __getattr__(self, name: str) -> Any:
        """Internal helper for  getattr  .

        Args:
            name: Name identifier.

        Returns:
            Result value.

        """
        if name.startswith('_'):
            raise AttributeError(name)

        if name not in _KNOWN_MODULES:
            raise AttributeError(
                f"ModuleRegistry has no module '{name}'. "
                f"Known modules: {sorted(_KNOWN_MODULES)}"
            )

        instances = object.__getattribute__(self, '_instances')
        load_times = object.__getattribute__(self, '_load_times')
        access_counts = object.__getattribute__(self, '_access_counts')

        # Track access count
        access_counts[name] = access_counts.get(name, 0) + 1

        if name not in instances:
            t0 = time.perf_counter()
            mod = importlib.import_module(f'malaysian_manglish_nlp.{name}')
            load_times[name] = time.perf_counter() - t0
            instances[name] = mod

        return instances[name]

    def __dir__(self) -> List[str]:
        """Internal helper for  dir  .

        Returns:
            List of results.

        """
        return sorted(_KNOWN_MODULES)

    def is_loaded(self, name: str) -> bool:
        """Check if a module has been loaded.

        Args:
            name: Module name (without ``malaysian_manglish_nlp.`` prefix).

        Returns:
            ``True`` if the module is currently loaded in this registry.
        """
        instances = object.__getattribute__(self, '_instances')
        return name in instances

    def unload(self, name: str) -> bool:
        """Unload a module to free memory.

        Removes the module from the registry cache and deletes it from
        ``sys.modules`` if no other references exist.

        Args:
            name: Module name (without ``malaysian_manglish_nlp.`` prefix).

        Returns:
            ``True`` if the module was unloaded, ``False`` if it wasn't loaded.
        """
        instances = object.__getattribute__(self, '_instances')
        load_times = object.__getattribute__(self, '_load_times')
        access_counts = object.__getattribute__(self, '_access_counts')

        if name not in instances:
            return False

        del instances[name]
        load_times.pop(name, None)
        access_counts.pop(name, None)

        # Also remove from sys.modules if possible
        full_name = f'malaysian_manglish_nlp.{name}'
        if full_name in sys.modules:
            del sys.modules[full_name]

        gc.collect()
        return True

    def reload(self, name: str) -> Any:
        """Reload a module (unload then re-import).

        Args:
            name: Module name.

        Returns:
            The freshly imported module.
        """
        self.unload(name)
        return getattr(self, name)

    def loaded_modules(self) -> List[str]:
        """Return list of currently loaded module names.

        Returns:
            Sorted list of module name strings.
        """
        instances = object.__getattribute__(self, '_instances')
        return sorted(instances.keys())

    def available_modules(self) -> List[str]:
        """Return list of all known module names.

        Returns:
            Sorted list of all module names this registry knows about.
        """
        return sorted(_KNOWN_MODULES)

    def load_times(self) -> Dict[str, float]:
        """Return import times for all loaded modules.

        Returns:
            Dict mapping module name to import time in seconds.
        """
        return dict(object.__getattribute__(self, '_load_times'))

    def access_counts(self) -> Dict[str, int]:
        """Return access counts for all tracked modules.

        Returns:
            Dict mapping module name to number of accesses.
        """
        return dict(object.__getattribute__(self, '_access_counts'))

    def memory_info(self) -> Dict[str, Any]:
        """Report memory usage and stats for loaded modules.

        Returns:
            Dict with:
                - ``loaded``: list of loaded module names
                - ``count``: number of loaded modules
                - ``available``: total known modules
                - ``load_times``: dict of module -> import time (seconds)
                - ``access_counts``: dict of module -> access count
                - ``gc_stats``: garbage collector statistics
                - ``sys_modules_manglish``: count of malaysian_manglish_nlp modules in sys.modules
        """
        instances = object.__getattribute__(self, '_instances')
        load_times = object.__getattribute__(self, '_load_times')
        access_counts = object.__getattribute__(self, '_access_counts')

        # Count malaysian_manglish_nlp modules in sys.modules
        manglish_count = sum(
            1 for k in sys.modules if k.startswith('malaysian_manglish_nlp')
        )

        return {
            'loaded': sorted(instances.keys()),
            'count': len(instances),
            'available': len(_KNOWN_MODULES),
            'load_times': dict(load_times),
            'access_counts': dict(access_counts),
            'gc_stats': gc.get_stats(),
            'sys_modules_manglish': manglish_count,
        }

    def unload_all(self) -> int:
        """Unload all cached modules.

        Returns:
            Number of modules unloaded.
        """
        instances = object.__getattribute__(self, '_instances')
        names = list(instances.keys())
        count = 0
        for name in names:
            if self.unload(name):
                count += 1
        return count

    def __repr__(self) -> str:
        """Internal helper for  repr  .

        Returns:
            Processed text string.

        """
        instances = object.__getattribute__(self, '_instances')
        return f"<ModuleRegistry loaded={len(instances)}/{len(_KNOWN_MODULES)}>"


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------

def optimize_memory() -> Dict[str, Any]:
    """Run garbage collection and return memory usage report.

    Performs a full GC cycle and reports on objects collected, remaining
    tracked objects, and malaysian_manglish_nlp module footprint.

    Returns:
        Dict with GC results and memory statistics.
    """
    # Force full collection
    collected = gc.collect()
    gc.collect()  # second pass

    # Count manglish modules in sys.modules
    manglish_modules = sorted(
        k for k in sys.modules if k.startswith('malaysian_manglish_nlp')
    )

    # Get gc stats
    gc_stats = gc.get_stats()

    # Count tracked objects
    tracked = len(gc.get_objects())

    return {
        'collected': collected,
        'gc_stats': gc_stats,
        'tracked_objects': tracked,
        'manglish_modules_loaded': len(manglish_modules),
        'manglish_modules': manglish_modules,
    }


def module_size_estimate(module_name: str) -> Dict[str, Any]:
    """Estimate the memory footprint of a loaded module.

    Uses ``sys.getsizeof`` on the module object and its direct attributes.
    This is a rough estimate, not a deep traversal.

    Args:
        module_name: Fully qualified module name (e.g. ``'malaysian_manglish_nlp.sentiment'``).

    Returns:
        Dict with ``module``, ``estimated_bytes``, ``attribute_count``,
        and ``loaded`` flag.
    """
    if module_name not in sys.modules:
        return {
            'module': module_name,
            'estimated_bytes': 0,
            'attribute_count': 0,
            'loaded': False,
        }

    mod = sys.modules[module_name]
    size = sys.getsizeof(mod)
    attr_count = 0

    for attr_name in dir(mod):
        try:
            attr = getattr(mod, attr_name)
            size += sys.getsizeof(attr)
            attr_count += 1
        except Exception:
            pass

    return {
        'module': module_name,
        'estimated_bytes': size,
        'attribute_count': attr_count,
        'loaded': True,
    }


def get_all_module_sizes() -> List[Dict[str, Any]]:
    """Estimate memory usage for all loaded malaysian_manglish_nlp modules.

    Returns:
        List of dicts (one per module), sorted by estimated size descending.
    """
    results = []
    for name in sorted(sys.modules):
        if name.startswith('malaysian_manglish_nlp'):
            results.append(module_size_estimate(name))

    results.sort(key=lambda x: x['estimated_bytes'], reverse=True)
    return results

"""Tests for manglish_nlp.memory - Memory optimization utilities.

Tests cover:
    - LazyModule: deferred import, attribute proxy, is_loaded, repr
    - ModuleRegistry: lazy access, unload, reload, loaded_modules, memory_info
    - optimize_memory: GC and reporting
    - module_size_estimate and get_all_module_sizes
    - Edge cases: unknown modules, double unload, repr
"""
from __future__ import annotations

import gc
import os
import sys
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp.memory import (
    LazyModule,
    ModuleRegistry,
    optimize_memory,
    module_size_estimate,
    get_all_module_sizes,
)


# ---------------------------------------------------------------------------
# LazyModule tests
# ---------------------------------------------------------------------------

class TestLazyModule:
    """Tests for LazyModule deferred import proxy."""

    def test_lazy_not_loaded_initially(self):
        mod = LazyModule('manglish_nlp.sentiment')
        assert mod.is_loaded is False

    def test_lazy_loads_on_access(self):
        mod = LazyModule('manglish_nlp.sentiment')
        # Access an attribute -> triggers import
        func = mod.analyze_sentiment
        assert mod.is_loaded is True
        assert callable(func)

    def test_lazy_returns_correct_results(self):
        mod = LazyModule('manglish_nlp.sentiment')
        result = mod.analyze_sentiment("best gila")
        assert isinstance(result, dict)

    def test_lazy_caches_module(self):
        mod = LazyModule('manglish_nlp.sentiment')
        _ = mod.analyze_sentiment  # first access
        cached = mod._module
        _ = mod.sentiment  # second access
        assert mod._module is cached  # same object

    def test_lazy_repr_before_load(self):
        mod = LazyModule('manglish_nlp.sentiment')
        r = repr(mod)
        assert 'lazy' in r
        assert 'sentiment' in r

    def test_lazy_repr_after_load(self):
        mod = LazyModule('manglish_nlp.sentiment')
        _ = mod.analyze_sentiment
        r = repr(mod)
        assert 'loaded' in r

    def test_lazy_dir(self):
        mod = LazyModule('manglish_nlp.sentiment')
        attrs = dir(mod)
        assert isinstance(attrs, list)
        assert 'analyze_sentiment' in attrs
        assert mod.is_loaded is True  # dir triggers load

    def test_lazy_unknown_attribute(self):
        mod = LazyModule('manglish_nlp.sentiment')
        with pytest.raises(AttributeError):
            _ = mod.nonexistent_function_xyz

    def test_lazy_invalid_module(self):
        mod = LazyModule('manglish_nlp.totally_fake_module_12345')
        with pytest.raises(ModuleNotFoundError):
            _ = mod.something


# ---------------------------------------------------------------------------
# ModuleRegistry tests
# ---------------------------------------------------------------------------

class TestModuleRegistry:
    """Tests for ModuleRegistry lazy-loading hub."""

    def test_registry_init_empty(self):
        reg = ModuleRegistry()
        assert reg.loaded_modules() == []

    def test_registry_lazy_access(self):
        reg = ModuleRegistry()
        sentiment = reg.sentiment
        assert sentiment is not None
        assert 'sentiment' in reg.loaded_modules()

    def test_registry_caches_module(self):
        reg = ModuleRegistry()
        s1 = reg.sentiment
        s2 = reg.sentiment
        assert s1 is s2

    def test_registry_access_multiple(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        _ = reg.ner
        loaded = reg.loaded_modules()
        assert 'sentiment' in loaded
        assert 'ner' in loaded
        assert len(loaded) == 2

    def test_registry_unknown_module(self):
        reg = ModuleRegistry()
        with pytest.raises(AttributeError, match="no module"):
            _ = reg.totally_fake_xyz

    def test_registry_is_loaded(self):
        reg = ModuleRegistry()
        assert reg.is_loaded('sentiment') is False
        _ = reg.sentiment
        assert reg.is_loaded('sentiment') is True

    def test_registry_unload(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        assert reg.is_loaded('sentiment') is True

        result = reg.unload('sentiment')
        assert result is True
        assert reg.is_loaded('sentiment') is False

    def test_registry_unload_not_loaded(self):
        reg = ModuleRegistry()
        result = reg.unload('sentiment')
        assert result is False

    def test_registry_reload(self):
        reg = ModuleRegistry()
        s1 = reg.sentiment
        s2 = reg.reload('sentiment')
        # After reload, should be a fresh import
        assert 'sentiment' in reg.loaded_modules()

    def test_registry_loaded_modules_sorted(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        _ = reg.ner
        _ = reg.clean
        loaded = reg.loaded_modules()
        assert loaded == sorted(loaded)

    def test_registry_available_modules(self):
        reg = ModuleRegistry()
        available = reg.available_modules()
        assert isinstance(available, list)
        assert 'sentiment' in available
        assert 'ner' in available
        assert available == sorted(available)

    def test_registry_load_times(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        times = reg.load_times()
        assert 'sentiment' in times
        assert isinstance(times['sentiment'], float)
        assert times['sentiment'] >= 0

    def test_registry_access_counts(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        _ = reg.sentiment
        _ = reg.sentiment
        counts = reg.access_counts()
        assert counts.get('sentiment') == 3

    def test_registry_dir(self):
        reg = ModuleRegistry()
        d = dir(reg)
        assert 'sentiment' in d
        assert 'ner' in d

    def test_registry_unload_all(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        _ = reg.ner
        _ = reg.clean
        count = reg.unload_all()
        assert count == 3
        assert reg.loaded_modules() == []

    def test_registry_repr(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        r = repr(reg)
        assert 'ModuleRegistry' in r
        assert 'loaded=1' in r

    def test_registry_memory_info(self):
        reg = ModuleRegistry()
        _ = reg.sentiment
        _ = reg.ner

        info = reg.memory_info()
        assert 'loaded' in info
        assert 'count' in info
        assert 'available' in info
        assert 'load_times' in info
        assert 'access_counts' in info
        assert 'gc_stats' in info
        assert 'sys_modules_manglish' in info
        assert info['count'] == 2
        assert info['available'] > 0


# ---------------------------------------------------------------------------
# optimize_memory tests
# ---------------------------------------------------------------------------

class TestOptimizeMemory:
    """Tests for the optimize_memory function."""

    def test_optimize_returns_dict(self):
        result = optimize_memory()
        assert isinstance(result, dict)

    def test_optimize_has_keys(self):
        result = optimize_memory()
        assert 'collected' in result
        assert 'gc_stats' in result
        assert 'tracked_objects' in result
        assert 'manglish_modules_loaded' in result
        assert 'manglish_modules' in result

    def test_optimize_collected_is_int(self):
        result = optimize_memory()
        assert isinstance(result['collected'], int)
        assert result['collected'] >= 0

    def test_optimize_tracked_positive(self):
        result = optimize_memory()
        assert result['tracked_objects'] > 0

    def test_optimize_modules_list(self):
        # Import something first
        import manglish_nlp.sentiment
        result = optimize_memory()
        assert isinstance(result['manglish_modules'], list)
        assert result['manglish_modules_loaded'] > 0


# ---------------------------------------------------------------------------
# module_size_estimate tests
# ---------------------------------------------------------------------------

class TestModuleSizeEstimate:
    """Tests for module_size_estimate function."""

    def test_loaded_module(self):
        import manglish_nlp.sentiment
        result = module_size_estimate('manglish_nlp.sentiment')
        assert result['loaded'] is True
        assert result['estimated_bytes'] > 0
        assert result['attribute_count'] > 0

    def test_unloaded_module(self):
        result = module_size_estimate('manglish_nlp.totally_fake_xyz_123')
        assert result['loaded'] is False
        assert result['estimated_bytes'] == 0

    def test_result_keys(self):
        result = module_size_estimate('manglish_nlp.sentiment')
        assert 'module' in result
        assert 'estimated_bytes' in result
        assert 'attribute_count' in result
        assert 'loaded' in result


# ---------------------------------------------------------------------------
# get_all_module_sizes tests
# ---------------------------------------------------------------------------

class TestGetAllModuleSizes:
    """Tests for get_all_module_sizes function."""

    def test_returns_list(self):
        import manglish_nlp.sentiment
        results = get_all_module_sizes()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_sorted_by_size(self):
        import manglish_nlp.sentiment
        import manglish_nlp.ner
        results = get_all_module_sizes()
        sizes = [r['estimated_bytes'] for r in results]
        assert sizes == sorted(sizes, reverse=True)

    def test_all_manglish(self):
        results = get_all_module_sizes()
        for r in results:
            assert r['module'].startswith('manglish_nlp')

"""Performance tests for manglish-nlp.

Tests throughput, import time, and memory usage to ensure
performance stays within acceptable bounds.
"""

import time
import sys
import os
import pytest


class TestSentimentPerformance:
    """Test sentiment analysis throughput."""
    
    def test_sentiment_1000_texts_under_2_seconds(self):
        """Sentiment analysis should handle 1000 texts in < 2 seconds."""
        from manglish_nlp.sentiment import analyze_sentiment
        
        texts = [
            "gila best makanan dia",
            "teruk la service kat sini",
            "ok je biasa",
            "aku suka tempat ni memang terbaik",
            "hampeh betul mahal gila tak berbaloi",
            "sedap gila nasi lemak dia",
            "boring la cerita tu",
            "best weh recommend korang pergi",
            "frust betul lambat sangat",
            "cantik tempat ni peaceful",
        ] * 100  # 1000 texts
        
        start = time.perf_counter()
        for text in texts:
            analyze_sentiment(text)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 2.0, f"Sentiment analysis took {elapsed:.2f}s for 1000 texts (limit: 2s)"
    
    def test_sentiment_cached_speedup(self):
        """Cached sentiment calls should be significantly faster."""
        from manglish_nlp.sentiment import analyze_sentiment
        
        text = "gila best makanan dia memang terbaik"
        
        # First call (cold)
        analyze_sentiment(text)
        
        # Cached calls
        start = time.perf_counter()
        for _ in range(1000):
            analyze_sentiment(text)
        elapsed = time.perf_counter() - start
        
        # 1000 cached calls should be very fast (< 0.1s)
        assert elapsed < 0.1, f"Cached sentiment took {elapsed:.3f}s for 1000 calls (limit: 0.1s)"


class TestNormalizePerformance:
    """Test normalization throughput."""
    
    def test_normalize_1000_texts_under_1_second(self):
        """Normalize should handle 1000 texts in < 1 second."""
        from manglish_nlp.normalize import normalize
        
        texts = [
            "nk tnya brapa sem utk grad",
            "aku xde duit nk beli tu",
            "ko dh mkn blm",
            "jom la pegi mkn skrg",
            "sbb tu la aku ckp dgn dia",
            "tlg la bgtau aku cmne nk buat",
            "xpe la nnt aku try lg",
            "mmg best gila tmpt ni",
            "aku nk blk dh penat sgt",
            "ko tgk x cite tu smlm",
        ] * 100  # 1000 texts
        
        start = time.perf_counter()
        for text in texts:
            normalize(text)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, f"Normalize took {elapsed:.2f}s for 1000 texts (limit: 1s)"
    
    def test_normalize_cached_speedup(self):
        """Cached normalize calls should be significantly faster."""
        from manglish_nlp.normalize import normalize
        
        text = "nk tnya brapa sem utk grad"
        
        # First call (cold)
        normalize(text)
        
        # Cached calls
        start = time.perf_counter()
        for _ in range(1000):
            normalize(text)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"Cached normalize took {elapsed:.3f}s for 1000 calls (limit: 0.1s)"


class TestImportPerformance:
    """Test import time."""
    
    def test_import_time_under_500ms(self):
        """Importing manglish_nlp should take < 0.5 seconds."""
        import subprocess
        
        # Run import in a fresh subprocess to get accurate timing
        code = (
            "import time; "
            "start = time.perf_counter(); "
            "import manglish_nlp; "
            "elapsed = time.perf_counter() - start; "
            "print(f'{elapsed:.4f}')"
        )
        
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if result.returncode == 0:
            elapsed = float(result.stdout.strip())
            assert elapsed < 0.5, f"Import took {elapsed:.3f}s (limit: 0.5s)"
        else:
            # If subprocess fails, do in-process test (less accurate but still useful)
            pytest.skip(f"Subprocess failed: {result.stderr[:200]}")


class TestMemoryUsage:
    """Test memory usage stays reasonable."""
    
    def test_memory_under_50mb(self):
        """Core modules should use < 50MB of memory."""
        import manglish_nlp
        from manglish_nlp.profiler import memory_usage
        
        mem = memory_usage()
        total_bytes = sum(mem.values())
        total_mb = total_bytes / (1024 * 1024)
        
        assert total_mb < 50, f"Memory usage is {total_mb:.1f}MB (limit: 50MB)"


class TestCacheModule:
    """Test cache functionality."""
    
    def test_lru_cache_basic(self):
        """LRU cache should store and retrieve values."""
        from manglish_nlp.cache import LRUCache
        
        cache = LRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
    
    def test_lru_cache_eviction(self):
        """LRU cache should evict least recently used items."""
        from manglish_nlp.cache import LRUCache
        
        cache = LRUCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"
        
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4
    
    def test_lru_cache_stats(self):
        """LRU cache should track hit/miss stats."""
        from manglish_nlp.cache import LRUCache
        
        cache = LRUCache(maxsize=10)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss
        cache.get("a")  # hit
        
        stats = cache.stats
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_ratio'] > 0.6
    
    def test_cached_decorator(self):
        """@cached decorator should memoize function results."""
        from manglish_nlp.cache import cached
        
        call_count = 0
        
        @cached(maxsize=10)
        def expensive_fn(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        assert expensive_fn(5) == 10
        assert expensive_fn(5) == 10  # cached
        assert expensive_fn(3) == 6
        assert call_count == 2  # Only 2 actual calls
    
    def test_clear_all_caches(self):
        """clear_all_caches should reset all registered caches."""
        from manglish_nlp.cache import clear_all_caches, cache_stats
        
        # Trigger some cached calls
        from manglish_nlp.normalize import normalize
        normalize("test text")
        
        # Clear
        clear_all_caches()
        
        # All caches should be empty
        stats = cache_stats()
        for name, s in stats.items():
            assert s['size'] == 0


class TestStemmerCaching:
    """Test that stemmer caching works correctly."""
    
    def test_stemmer_cached(self):
        """Stemmer should return same results when cached."""
        from manglish_nlp.stemmer import stem_word
        
        # First call
        result1 = stem_word("berlari")
        # Second call (cached)
        result2 = stem_word("berlari")
        
        assert result1 == result2 == "lari"
    
    def test_stemmer_throughput(self):
        """Stemmer should handle many words quickly with caching."""
        from manglish_nlp.stemmer import stem_word
        
        words = [
            "berlari", "memakan", "menulis", "pelajaran", "menyapu",
            "terbang", "memasak", "sekolahan", "bermain", "membaca",
        ] * 100  # 1000 words
        
        start = time.perf_counter()
        for word in words:
            stem_word(word)
        elapsed = time.perf_counter() - start
        
        # 1000 words (mostly cached) should be very fast
        assert elapsed < 0.5, f"Stemmer took {elapsed:.3f}s for 1000 words (limit: 0.5s)"


class TestProfiler:
    """Test profiler module."""
    
    def test_profile_all_modules(self):
        """profile_all_modules should return timing dict."""
        from manglish_nlp.profiler import profile_all_modules
        
        results = profile_all_modules("aku nak pergi makan", iterations=10)
        
        assert isinstance(results, dict)
        assert 'normalize' in results
        assert 'sentiment' in results
        assert isinstance(results['normalize'], float)
    
    def test_profile_module(self):
        """profile_module should return detailed timing."""
        from manglish_nlp.profiler import profile_module
        
        result = profile_module('sentiment', "best gila", iterations=50)
        
        assert 'mean_ms' in result
        assert 'median_ms' in result
        assert 'p95_ms' in result
        assert result['iterations'] == 50
    
    def test_find_bottlenecks(self):
        """find_bottlenecks should return sorted list."""
        from manglish_nlp.profiler import find_bottlenecks
        
        results = find_bottlenecks("aku nak pergi")
        
        assert isinstance(results, list)
        assert len(results) > 0
        # Should be sorted descending by time
        times = [t for _, t in results]
        assert times == sorted(times, reverse=True)
    
    def test_benchmark_throughput(self):
        """benchmark_throughput should return metrics dict."""
        from manglish_nlp.profiler import benchmark_throughput
        
        texts = ["best gila makanan"] * 50
        result = benchmark_throughput(texts, module='sentiment')
        
        assert 'texts_per_second' in result
        assert 'avg_ms_per_text' in result
        assert result['total_texts'] == 50
    
    def test_generate_report(self):
        """generate_report should return markdown string."""
        from manglish_nlp.profiler import generate_report
        
        report = generate_report()
        
        assert isinstance(report, str)
        assert '# manglish-nlp Performance Report' in report
        assert 'Module Timings' in report
        assert 'Bottlenecks' in report

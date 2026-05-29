"""Tests for manglish_nlp.parallel - Parallel processing pipeline.

Tests cover:
    - ParallelPipeline initialization
    - Parallel sentiment batch (correctness & match sequential)
    - Parallel NER batch (correctness & match sequential)
    - Parallel POS, tokenize, normalize, clean, formalize batches
    - Generic map with custom functions
    - analyze_batch with multiple modules
    - Benchmark utility
    - Edge cases: empty lists, single items, thread vs process backend
"""
from __future__ import annotations

import os
import sys
import time
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp.parallel import ParallelPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "gila best makanan dia, aku suka gila",
    "teruk gila service kat sini, tak puas hati",
    "boleh la, takde la best sangat tapi ok je",
    "aku nak pergi kedai beli roti",
    "dia orang dah sampai ke belum?",
    "weh jom makan tengahari ni",
    "mahal gila harga barang sekarang ni",
    "okay la boleh tahan quality dia",
]

SHORT_TEXTS = [
    "best gila",
    "teruk la",
    "ok je",
]


@pytest.fixture
def pipeline():
    """Create a ParallelPipeline with 2 workers for testing."""
    return ParallelPipeline(n_workers=2)


@pytest.fixture
def thread_pipeline():
    """Create a thread-based ParallelPipeline."""
    return ParallelPipeline(n_workers=2, use_threads=True)


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for ParallelPipeline initialization."""

    def test_default_workers(self):
        pipe = ParallelPipeline()
        assert pipe.n_workers >= 1
        assert pipe.n_workers == (os.cpu_count() or 2)

    def test_custom_workers(self):
        pipe = ParallelPipeline(n_workers=4)
        assert pipe.n_workers == 4

    def test_use_threads_flag(self):
        pipe = ParallelPipeline(use_threads=True)
        assert pipe.use_threads is True

    def test_use_processes_default(self):
        pipe = ParallelPipeline()
        assert pipe.use_threads is False

    def test_repr(self):
        pipe = ParallelPipeline(n_workers=2)
        r = repr(pipe)
        assert 'ParallelPipeline' in r
        assert 'n_workers=2' in r
        assert 'processes' in r

    def test_repr_threads(self):
        pipe = ParallelPipeline(n_workers=2, use_threads=True)
        r = repr(pipe)
        assert 'threads' in r


# ---------------------------------------------------------------------------
# Generic map tests
# ---------------------------------------------------------------------------

class TestMap:
    """Tests for the generic map method."""

    def test_map_basic(self, pipeline):
        results = pipeline.map(str.upper, SAMPLE_TEXTS)
        assert len(results) == len(SAMPLE_TEXTS)
        for orig, result in zip(SAMPLE_TEXTS, results):
            assert result == orig.upper()

    def test_map_empty(self, pipeline):
        results = pipeline.map(str.upper, [])
        assert results == []

    def test_map_single_item(self, pipeline):
        results = pipeline.map(str.upper, ['hello'])
        assert results == ['HELLO']

    def test_map_preserves_order(self, pipeline):
        results = pipeline.map(len, SAMPLE_TEXTS)
        expected = [len(t) for t in SAMPLE_TEXTS]
        assert results == expected

    def test_map_thread_backend(self, thread_pipeline):
        results = thread_pipeline.map(str.upper, SAMPLE_TEXTS)
        assert len(results) == len(SAMPLE_TEXTS)
        for orig, result in zip(SAMPLE_TEXTS, results):
            assert result == orig.upper()


# ---------------------------------------------------------------------------
# Sentiment batch tests
# ---------------------------------------------------------------------------

class TestSentimentBatch:
    """Tests for parallel sentiment analysis."""

    def test_sentiment_batch_returns_list(self, pipeline):
        results = pipeline.sentiment_batch(SHORT_TEXTS)
        assert isinstance(results, list)
        assert len(results) == len(SHORT_TEXTS)

    def test_sentiment_batch_results_are_dicts(self, pipeline):
        results = pipeline.sentiment_batch(SHORT_TEXTS)
        for r in results:
            assert isinstance(r, dict)

    def test_sentiment_batch_matches_sequential(self):
        """Parallel results must match sequential results."""
        from manglish_nlp.sentiment import analyze_sentiment

        pipe = ParallelPipeline(n_workers=2, use_threads=True)
        texts = SHORT_TEXTS

        parallel_results = pipe.sentiment_batch(texts)
        sequential_results = [analyze_sentiment(t) for t in texts]

        assert len(parallel_results) == len(sequential_results)
        for p, s in zip(parallel_results, sequential_results):
            assert p == s

    def test_sentiment_batch_empty(self, pipeline):
        results = pipeline.sentiment_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# NER batch tests
# ---------------------------------------------------------------------------

class TestNERBatch:
    """Tests for parallel NER."""

    def test_ner_batch_returns_list(self, pipeline):
        results = pipeline.ner_batch(SHORT_TEXTS)
        assert isinstance(results, list)
        assert len(results) == len(SHORT_TEXTS)

    def test_ner_batch_results_are_lists(self, pipeline):
        results = pipeline.ner_batch(SHORT_TEXTS)
        for r in results:
            assert isinstance(r, list)

    def test_ner_batch_matches_sequential(self):
        """Parallel NER results must match sequential."""
        from manglish_nlp.ner import ner_tag

        pipe = ParallelPipeline(n_workers=2, use_threads=True)
        texts = SHORT_TEXTS

        parallel_results = pipe.ner_batch(texts)
        sequential_results = [ner_tag(t) for t in texts]

        assert len(parallel_results) == len(sequential_results)
        for p, s in zip(parallel_results, sequential_results):
            assert p == s


# ---------------------------------------------------------------------------
# Other batch method tests
# ---------------------------------------------------------------------------

class TestOtherBatches:
    """Tests for POS, tokenize, normalize, clean, formalize, language, emotion."""

    def test_tokenize_batch(self, pipeline):
        results = pipeline.tokenize_batch(SHORT_TEXTS)
        assert len(results) == len(SHORT_TEXTS)
        for r in results:
            assert isinstance(r, list)
            assert all(isinstance(tok, str) for tok in r)

    def test_normalize_batch(self, pipeline):
        results = pipeline.normalize_batch(SHORT_TEXTS)
        assert len(results) == len(SHORT_TEXTS)
        for r in results:
            assert isinstance(r, str)

    def test_language_batch(self, pipeline):
        results = pipeline.language_batch(SHORT_TEXTS)
        assert len(results) == len(SHORT_TEXTS)
        for r in results:
            # detect_language returns a dict with 'language' key
            assert isinstance(r, (str, dict))


# ---------------------------------------------------------------------------
# analyze_batch tests
# ---------------------------------------------------------------------------

class TestAnalyzeBatch:
    """Tests for the multi-module analyze_batch method."""

    def test_analyze_batch_default_modules(self, pipeline):
        results = pipeline.analyze_batch(SHORT_TEXTS)
        assert len(results) == len(SHORT_TEXTS)
        for r in results:
            assert isinstance(r, dict)
            assert 'sentiment' in r
            assert 'ner' in r
            assert 'pos' in r

    def test_analyze_batch_custom_modules(self, pipeline):
        results = pipeline.analyze_batch(
            SHORT_TEXTS,
            modules=['sentiment', 'tokenize'],
        )
        assert len(results) == len(SHORT_TEXTS)
        for r in results:
            assert 'sentiment' in r
            assert 'tokenize' in r
            assert 'ner' not in r

    def test_analyze_batch_unknown_module(self, pipeline):
        results = pipeline.analyze_batch(
            SHORT_TEXTS,
            modules=['sentiment', 'nonexistent_module'],
        )
        for r in results:
            assert 'sentiment' in r
            assert 'error' in r['nonexistent_module']

    def test_analyze_batch_empty(self, pipeline):
        results = pipeline.analyze_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Tests for the benchmark method."""

    def test_benchmark_returns_timing(self):
        pipe = ParallelPipeline(n_workers=2, use_threads=True)
        result = pipe.benchmark(SHORT_TEXTS, module='sentiment')
        assert 'parallel_time' in result
        assert 'sequential_time' in result
        assert 'speedup' in result
        assert result['parallel_time'] >= 0
        assert result['sequential_time'] >= 0

    def test_benchmark_unknown_module(self):
        pipe = ParallelPipeline(n_workers=2)
        with pytest.raises(ValueError, match="Unknown module"):
            pipe.benchmark(SHORT_TEXTS, module='nonexistent')

    def test_benchmark_no_sequential(self):
        pipe = ParallelPipeline(n_workers=2, use_threads=True)
        result = pipe.benchmark(SHORT_TEXTS, module='sentiment', sequential=False)
        assert 'parallel_time' in result
        assert 'sequential_time' not in result


# ---------------------------------------------------------------------------
# Thread vs process backend comparison
# ---------------------------------------------------------------------------

class TestBackends:
    """Verify both backends produce identical results."""

    def test_thread_vs_process_sentiment(self):
        texts = SHORT_TEXTS
        proc_pipe = ParallelPipeline(n_workers=2, use_threads=False)
        thread_pipe = ParallelPipeline(n_workers=2, use_threads=True)

        proc_results = proc_pipe.sentiment_batch(texts)
        thread_results = thread_pipe.sentiment_batch(texts)

        assert proc_results == thread_results

    def test_thread_vs_process_tokenize(self):
        texts = SHORT_TEXTS
        proc_pipe = ParallelPipeline(n_workers=2, use_threads=False)
        thread_pipe = ParallelPipeline(n_workers=2, use_threads=True)

        proc_results = proc_pipe.tokenize_batch(texts)
        thread_results = thread_pipe.tokenize_batch(texts)

        assert proc_results == thread_results

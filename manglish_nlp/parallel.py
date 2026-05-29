"""Parallel processing for batch NLP operations.

Provides ``ParallelPipeline`` for running sentiment analysis, NER, and
other manglish_nlp modules across multiple CPU cores using either
process-based or thread-based parallelism.

Usage::

    from manglish_nlp.parallel import ParallelPipeline

    pipe = ParallelPipeline(n_workers=4)
    results = pipe.sentiment_batch([
        "gila best makanan dia",
        "teruk gila service kat sini",
        "boleh la, takde la best sangat",
    ])

Zero extra dependencies -- uses only stdlib ``multiprocessing`` and
``concurrent.futures``.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import , Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_sentiment(text: str) -> Dict[str, Any]:
    """Run sentiment analysis on a single text (worker)."""
    from manglish_nlp.sentiment import analyze_sentiment
    return analyze_sentiment(text)


def _worker_ner(text: str) -> List[Dict[str, Any]]:
    """Run NER on a single text (worker)."""
    from manglish_nlp.ner import ner_tag
    return ner_tag(text)


def _worker_pos(text: str) -> List[Dict[str, Any]]:
    """Run POS tagging on a single text (worker)."""
    from manglish_nlp.pos import pos_tag
    return pos_tag(text)


def _worker_tokenize(text: str) -> List[str]:
    """Tokenize a single text (worker)."""
    from manglish_nlp.tokenizer_fast import tokenize
    return tokenize(text)


def _worker_normalize(text: str) -> str:
    """Normalize a single text (worker)."""
    from manglish_nlp.tokenizer_fast import normalize
    return normalize(text)


def _worker_clean(text: str) -> str:
    """Clean a single text (worker)."""
    from manglish_nlp.clean import clean
    return clean(text)


def _worker_formalize(text: str) -> str:
    """Formalize a single text (worker)."""
    from manglish_nlp.formalize import formalize
    return formalize(text)


def _worker_detect_language(text: str) -> str:
    """Detect language of a single text (worker)."""
    from manglish_nlp.language import detect_language
    return detect_language(text)


def _worker_detect_emotion(text: str) -> Dict[str, Any]:
    """Detect emotion of a single text (worker)."""
    from manglish_nlp.emotion import detect_emotion
    return detect_emotion(text)


def _worker_generic(text: str, module_name: str, func_name: str) -> Any:
    """Generic worker: import module and call func(text)."""
    import importlib
    mod = importlib.import_module(f'manglish_nlp.{module_name}')
    func = getattr(mod, func_name)
    return func(text)


# Module dispatch map for analyze_batch (must be top-level for pickling)
_MODULE_DISPATCH: Dict[str, Callable] = {
    'sentiment': _worker_sentiment,
    'ner': _worker_ner,
    'pos': _worker_pos,
    'tokenize': _worker_tokenize,
    'normalize': _worker_normalize,
    'clean': _worker_clean,
    'formalize': _worker_formalize,
    'language': _worker_detect_language,
    'emotion': _worker_detect_emotion,
}


def _worker_analyze(text_and_modules: tuple) -> Dict[str, Any]:
    """Worker for analyze_batch. Receives (text, modules_list) tuple.

    Must be top-level for ProcessPoolExecutor pickling.
    """
    text, modules = text_and_modules
    result: Dict[str, Any] = {}
    for mod_name in modules:
        if mod_name in _MODULE_DISPATCH:
            try:
                result[mod_name] = _MODULE_DISPATCH[mod_name](text)
            except Exception as exc:
                result[mod_name] = {'error': str(exc)}
        else:
            result[mod_name] = {'error': f'Unknown module: {mod_name}'}
    return result


# ---------------------------------------------------------------------------
# ParallelPipeline
# ---------------------------------------------------------------------------

class ParallelPipeline:
    """Process text batches using multiple CPU cores.

    Args:
        n_workers: Number of parallel workers. Defaults to CPU count.
        use_threads: If ``True``, use ``ThreadPoolExecutor`` instead of
            ``ProcessPoolExecutor``.  Threads share memory (no pickling
            overhead) but are limited by the GIL for CPU-bound work.
            Threads are better for I/O-bound tasks.
        chunk_size: Default number of items per chunk when splitting work.

    Example::

        pipe = ParallelPipeline(n_workers=4)
        sentiments = pipe.sentiment_batch(texts)
    """

    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False,
        chunk_size: int = 100,
    ) -> None:
        self.n_workers: int = n_workers or os.cpu_count() or 2
        self.use_threads: bool = use_threads
        self.chunk_size: int = chunk_size

    def _get_executor(self: Any) -> Dict[str, Any]:
        """Return the appropriate executor class."""
        if self.use_threads:
            return ThreadPoolExecutor(max_workers=self.n_workers)
        return ProcessPoolExecutor(max_workers=self.n_workers)

    # ------------------------------------------------------------------
    # Generic map
    # ------------------------------------------------------------------

    def map(
        self,
        func: Callable[[str], Any],
        texts: Sequence[str],
        chunk_size: Optional[int] = None,
    ) -> List[Any]:
        """Apply *func* to each text in *texts* in parallel.

        Args:
            func: A callable that takes a single string and returns a result.
            texts: Iterable of input texts.
            chunk_size: Items per chunk (currently used for doc; executor
                handles chunking internally).

        Returns:
            List of results in the same order as *texts*.
        """
        texts = list(texts)
        if not texts:
            return []

        # For single item or single worker, skip overhead
        if len(texts) == 1 or self.n_workers <= 1:
            return [func(t) for t in texts]

        with self._get_executor() as executor:
            results = list(executor.map(func, texts))
        return results

    # ------------------------------------------------------------------
    # Module-specific batch methods
    # ------------------------------------------------------------------

    def sentiment_batch(
        self,
        texts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Run sentiment analysis on a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of sentiment result dicts, one per input text.
        """
        return self.map(_worker_sentiment, texts)

    def ner_batch(
        self,
        texts: Sequence[str],
    ) -> List[List[Dict[str, Any]]]:
        """Run Named Entity Recognition on a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of NER result lists, one per input text.
        """
        return self.map(_worker_ner, texts)

    def pos_batch(
        self,
        texts: Sequence[str],
    ) -> List[List[Dict[str, Any]]]:
        """Run POS tagging on a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of POS tag results, one per input text.
        """
        return self.map(_worker_pos, texts)

    def tokenize_batch(
        self,
        texts: Sequence[str],
    ) -> List[List[str]]:
        """Tokenize a batch of texts in parallel.

        Uses the fast C tokenizer when available.

        Args:
            texts: List of input texts.

        Returns:
            List of token lists, one per input text.
        """
        return self.map(_worker_tokenize, texts)

    def normalize_batch(
        self,
        texts: Sequence[str],
    ) -> List[str]:
        """Normalize a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of normalized strings.
        """
        return self.map(_worker_normalize, texts)

    def clean_batch(
        self,
        texts: Sequence[str],
    ) -> List[str]:
        """Clean a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of cleaned strings.
        """
        return self.map(_worker_clean, texts)

    def formalize_batch(
        self,
        texts: Sequence[str],
    ) -> List[str]:
        """Formalize a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of formalized strings.
        """
        return self.map(_worker_formalize, texts)

    def language_batch(
        self,
        texts: Sequence[str],
    ) -> List[str]:
        """Detect language for a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of language label strings.
        """
        return self.map(_worker_detect_language, texts)

    def emotion_batch(
        self,
        texts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Detect emotion for a batch of texts in parallel.

        Args:
            texts: List of input texts.

        Returns:
            List of emotion result dicts.
        """
        return self.map(_worker_detect_emotion, texts)

    # ------------------------------------------------------------------
    # Generic multi-module batch
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        texts: Sequence[str],
        modules: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run multiple NLP modules on each text in parallel.

        For each text, runs all requested modules and returns a combined
        result dict.  Modules run sequentially per text but texts are
        processed in parallel.

        Args:
            texts: List of input texts.
            modules: List of module names to run. Supported values:
                ``'sentiment'``, ``'ner'``, ``'pos'``, ``'tokenize'``,
                ``'normalize'``, ``'clean'``, ``'formalize'``,
                ``'language'``, ``'emotion'``.
                Defaults to ``['sentiment', 'ner', 'pos']``.

        Returns:
            List of dicts with one key per module containing that module's
            result for the text.
        """
        if modules is None:
            modules = ['sentiment', 'ner', 'pos']

        # Create (text, modules) tuples for the top-level worker
        work_items = [(text, modules) for text in texts]

        texts_list = list(texts)
        if not texts_list:
            return []

        # For single item or single worker, skip overhead
        if len(texts_list) == 1 or self.n_workers <= 1:
            return [_worker_analyze(item) for item in work_items]

        with self._get_executor() as executor:
            results = list(executor.map(_worker_analyze, work_items))
        return results

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark(
        self,
        texts: Sequence[str],
        module: str = 'sentiment',
        sequential: bool = True,
    ) -> Dict[str, Any]:
        """Benchmark parallel vs sequential execution.

        Args:
            texts: Sample texts to process.
            module: Module name to benchmark.
            sequential: Also run sequential baseline if ``True``.

        Returns:
            Dict with timing results and speedup factor.
        """
        func = _MODULE_DISPATCH.get(module)
        if func is None:
            raise ValueError(f"Unknown module: {module}. Choose from: {list(_MODULE_DISPATCH.keys())}")

        results: Dict[str, Any] = {'module': module, 'n_texts': len(texts), 'n_workers': self.n_workers}

        # Parallel
        t0 = time.perf_counter()
        parallel_results = self.map(func, texts)
        results['parallel_time'] = time.perf_counter() - t0
        results['parallel_results'] = parallel_results

        # Sequential
        if sequential:
            t0 = time.perf_counter()
            seq_results = [func(t) for t in texts]
            results['sequential_time'] = time.perf_counter() - t0
            results['sequential_results'] = seq_results
            if results['parallel_time'] > 0:
                results['speedup'] = results['sequential_time'] / results['parallel_time']
            else:
                results['speedup'] = float('inf')

        return results

    def __repr__(self) -> str:
        backend = 'threads' if self.use_threads else 'processes'
        return (
            f"ParallelPipeline(n_workers={self.n_workers}, "
            f"backend={backend}, chunk_size={self.chunk_size})"
        )

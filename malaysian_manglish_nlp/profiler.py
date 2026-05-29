"""Performance profiling utilities for manglish-nlp.

Provides timing, memory usage, throughput benchmarking, and bottleneck detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import time
import sys
import os


def profile_all_modules(text: str, iterations: int = 100) -> Dict[str, Any]:
    """Profile all major modules with the given text.
    
    Args:
        text: Sample text to process.
        iterations: Number of iterations per module. Default 100.
    
    Returns:
        dict: Timing results per module (seconds per call).
    
    Example:
        >>> profile_all_modules("aku nak pergi makan la weh")
        {'normalize': 0.00012, 'sentiment': 0.00034, ...}
    """
    modules = {
        'normalize': lambda t: _call_module('normalize', t),
        'sentiment': lambda t: _call_module('sentiment', t),
        'language': lambda t: _call_module('language', t),
        'stemmer': lambda t: _call_module('stemmer', t),
        'tokenizer': lambda t: _call_module('tokenizer', t),
        'clean': lambda t: _call_module('clean', t),
        'ner': lambda t: _call_module('ner', t),
        'pos': lambda t: _call_module('pos', t),
        'segment': lambda t: _call_module('segment', t),
        'formalize': lambda t: _call_module('formalize', t),
        'keywords': lambda t: _call_module('keywords', t),
        'dictionary': lambda t: _call_module('dictionary', t),
    }
    
    results = {}
    for name, fn in modules.items():
        try:
            # Warmup
            fn(text)
            
            start = time.perf_counter()
            for _ in range(iterations):
                fn(text)
            elapsed = time.perf_counter() - start
            
            results[name] = round(elapsed / iterations, 6)
        except Exception as e:
            results[name] = f"error: {e}"
    
    return results


def profile_module(module_name: str, text: str, iterations: int = 1000) -> Dict[str, Any]:
    """Profile a specific module in detail.
    
    Args:
        module_name: Module name (e.g., 'sentiment', 'normalize').
        text: Sample text to process.
        iterations: Number of iterations. Default 1000.
    
    Returns:
        dict: Detailed timing with min, max, mean, median, std.
    
    Example:
        >>> profile_module('sentiment', "gila best makanan dia", iterations=500)
        {'module': 'sentiment', 'iterations': 500, 'mean_ms': 0.34, ...}
    """
    fn = lambda t: _call_module(module_name, t)
    
    # Warmup
    for _ in range(min(10, iterations)):
        fn(text)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    times.sort()
    n = len(times)
    mean = sum(times) / n
    median = times[n // 2]
    variance = sum((t - mean) ** 2 for t in times) / n
    std = variance ** 0.5
    
    return {
        'module': module_name,
        'iterations': iterations,
        'mean_ms': round(mean * 1000, 4),
        'median_ms': round(median * 1000, 4),
        'min_ms': round(times[0] * 1000, 4),
        'max_ms': round(times[-1] * 1000, 4),
        'std_ms': round(std * 1000, 4),
        'p95_ms': round(times[int(n * 0.95)] * 1000, 4),
        'p99_ms': round(times[int(n * 0.99)] * 1000, 4),
        'total_s': round(sum(times), 4),
    }


def memory_usage() -> Dict[str, Any]:
    """Get memory usage by each loaded manglish-nlp module.
    
    Returns:
        dict: Memory used by each module in bytes (approximate via sys.getsizeof).
    
    Example:
        >>> memory_usage()
        {'normalize': 45032, 'sentiment': 23456, ...}
    """
    results = {}
    
    for name, mod in sys.modules.items():
        if not name.startswith('malaysian_manglish_nlp'):
            continue
        if mod is None:
            continue
        
        short_name = name.replace('malaysian_manglish_nlp.', '').replace('malaysian_manglish_nlp', '__init__')
        
        total_size = 0
        try:
            for attr_name in dir(mod):
                if attr_name.startswith('__'):
                    continue
                try:
                    obj = getattr(mod, attr_name)
                    size = sys.getsizeof(obj)
                    # For containers, try to get deeper size
                    if isinstance(obj, (dict, set, list, tuple)):
                        size = _deep_getsizeof(obj)
                    total_size += size
                except: pass
        except Exception:
            pass
        
        if total_size > 0:
            results[short_name] = total_size
    
    return results


def benchmark_throughput(texts: List[str], module: str = 'sentiment') -> Dict[str, Any]:
    """Benchmark throughput (texts processed per second).
    
    Args:
        texts: List of text strings to process.
        module: Module to benchmark. Default 'sentiment'.
    
    Returns:
        dict: Throughput metrics.
    
    Example:
        >>> texts = ["best gila" * i for i in range(1, 101)]
        >>> benchmark_throughput(texts, module='sentiment')
        {'module': 'sentiment', 'texts_per_second': 5432.1, ...}
    """
    fn = lambda t: _call_module(module, t)
    
    # Warmup
    for t in texts[:5]:
        fn(t)
    
    start = time.perf_counter()
    for t in texts:
        fn(t)
    elapsed = time.perf_counter() - start
    
    count = len(texts)
    throughput = count / elapsed if elapsed > 0 else 0
    avg_len = sum(len(t) for t in texts) / max(count, 1)
    
    return {
        'module': module,
        'total_texts': count,
        'total_time_s': round(elapsed, 4),
        'texts_per_second': round(throughput, 1),
        'avg_ms_per_text': round((elapsed / count) * 1000, 4) if count > 0 else 0,
        'avg_text_length': round(avg_len, 1),
        'chars_per_second': round((sum(len(t) for t in texts)) / elapsed, 0) if elapsed > 0 else 0,
    }


def find_bottlenecks(text: str) -> Dict[str, Any]:
    """Find the slowest operations when processing text.
    
    Args:
        text: Text to analyze.
    
    Returns:
        list: Sorted list of (module_name, time_ms) tuples, slowest first.
    
    Example:
        >>> find_bottlenecks("aku nak pergi makan")
        [('ner', 1.23), ('sentiment', 0.89), ('normalize', 0.12), ...]
    """
    modules = [
        'normalize', 'sentiment', 'language', 'stemmer', 'tokenizer',
        'clean', 'ner', 'pos', 'segment', 'formalize', 'keywords',
        'dictionary',
    ]
    
    timings = []
    for name in modules:
        try:
            # Run multiple times for stability
            times = []
            for _ in range(50):
                start = time.perf_counter()
                _call_module(name, text)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            median = sorted(times)[len(times) // 2]
            timings.append((name, round(median * 1000, 4)))
        except Exception:
            pass
    
    # Sort by time descending (slowest first)
    timings.sort(key=lambda x: x[1], reverse=True)
    return timings


def generate_report(output_path: Optional[str] = None) -> str:
    """Generate a comprehensive performance report in markdown.
    
    Args:
        output_path: Path to write the report. If None, returns string.
    
    Returns:
        str: Markdown report content.
    
    Example:
        >>> report = generate_report()
        >>> print(report[:50])
        '# manglish-nlp Performance Report...'
    """
    sample_text = "aku nak pergi makan kat KL la weh, best gila makanan dia"
    
    lines = []
    lines.append("# manglish-nlp Performance Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Sample text: \"{sample_text}\"")
    lines.append("")
    
    # Module timings
    lines.append("## Module Timings (100 iterations)")
    lines.append("")
    lines.append("| Module | Avg Time (ms) |")
    lines.append("|--------|--------------|")
    
    timings = profile_all_modules(sample_text, iterations=100)
    for name, val in sorted(timings.items(), key=lambda x: x[1] if isinstance(x[1], float) else 999):
        if isinstance(val, float):
            lines.append(f"| {name} | {val * 1000:.4f} |")
        else:
            lines.append(f"| {name} | {val} |")
    
    lines.append("")
    
    # Bottlenecks
    lines.append("## Bottlenecks (slowest first)")
    lines.append("")
    bottlenecks = find_bottlenecks(sample_text)
    for i, (name, ms) in enumerate(bottlenecks[:5], 1):
        lines.append(f"{i}. **{name}**: {ms} ms")
    
    lines.append("")
    
    # Memory usage
    lines.append("## Memory Usage")
    lines.append("")
    lines.append("| Module | Size (KB) |")
    lines.append("|--------|----------|")
    
    mem = memory_usage()
    for name, size in sorted(mem.items(), key=lambda x: x[1], reverse=True)[:10]:
        lines.append(f"| {name} | {size / 1024:.1f} |")
    
    lines.append("")
    
    # Cache stats
    lines.append("## Cache Statistics")
    lines.append("")
    try:
        from malaysian_manglish_nlp.cache import cache_stats
        stats = cache_stats()
        if stats:
            lines.append("| Function | Size | Hits | Misses | Hit Ratio |")
            lines.append("|----------|------|------|--------|-----------|")
            for name, s in stats.items():
                short = name.split('.')[-1]
                lines.append(f"| {short} | {s['size']} | {s['hits']} | {s['misses']} | {s['hit_ratio']:.2%} |")
        else:
            lines.append("No cached functions registered yet.")
    except ImportError:
        lines.append("Cache module not available.")
    
    lines.append("")
    
    # Throughput
    lines.append("## Throughput Benchmark")
    lines.append("")
    test_texts = [sample_text] * 100
    for mod in ['normalize', 'sentiment', 'tokenizer']:
        try:
            tp = benchmark_throughput(test_texts, module=mod)
            lines.append(f"- **{mod}**: {tp['texts_per_second']:.0f} texts/sec ({tp['avg_ms_per_text']:.3f} ms/text)")
        except Exception:
            pass
    
    lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


def _call_module(module_name: str, text: str) -> Any:
    """Call a module's main function with text."""
    if module_name == 'normalize':
        from malaysian_manglish_nlp.normalize import normalize
        return normalize(text)
    elif module_name == 'sentiment':
        from malaysian_manglish_nlp.sentiment import analyze_sentiment
        return analyze_sentiment(text)
    elif module_name == 'language':
        from malaysian_manglish_nlp.language import detect_language
        return detect_language(text)
    elif module_name == 'stemmer':
        from malaysian_manglish_nlp.stemmer import stem
        return stem(text)
    elif module_name == 'tokenizer':
        from malaysian_manglish_nlp.tokenizer import tokenize
        return tokenize(text)
    elif module_name == 'clean':
        from malaysian_manglish_nlp.clean import clean
        return clean(text)
    elif module_name == 'ner':
        from malaysian_manglish_nlp.ner import ner_tag
        return ner_tag(text)
    elif module_name == 'pos':
        from malaysian_manglish_nlp.pos import pos_tag
        return pos_tag(text)
    elif module_name == 'segment':
        from malaysian_manglish_nlp.segment import segment
        return segment(text)
    elif module_name == 'formalize':
        from malaysian_manglish_nlp.formalize import formalize
        return formalize(text)
    elif module_name == 'keywords':
        from malaysian_manglish_nlp.keywords import extract_keywords
        return extract_keywords(text)
    elif module_name == 'dictionary':
        from malaysian_manglish_nlp.dictionary import is_malay
        return is_malay(text.split()[0] if text.split() else text)
    else:
        raise ValueError(f"Unknown module: {module_name}")


def _deep_getsizeof(obj: Any, seen: Optional[Any] = None) -> int:
    """Recursively get size of object and its contents."""
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(_deep_getsizeof(k, seen) + _deep_getsizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(_deep_getsizeof(item, seen) for item in obj)
    
    return size

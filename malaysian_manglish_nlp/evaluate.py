"""Evaluation framework — track performance, compare versions, detect regressions.

Usage:
    python -m malaysian_manglish_nlp.evaluate          # Run full eval
    python -m malaysian_manglish_nlp.evaluate --save   # Save results to history
    python -m malaysian_manglish_nlp.evaluate --compare  # Compare with last saved
"""

from __future__ import annotations

from typing import Any, Dict, List

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'eval_history.json')


def run_evaluation() -> Dict[str, Any]:
    """Run full evaluation suite and return structured results."""
    import importlib.util
    
    # Load benchmark_expanded from tests dir
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
    spec = importlib.util.spec_from_file_location('benchmark_expanded', os.path.join(tests_dir, 'benchmark_expanded.py'))
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    
    SENTIMENT_DATA = bm.SENTIMENT_DATA
    LANGUAGE_DATA = bm.LANGUAGE_DATA
    EMOTION_DATA = bm.EMOTION_DATA
    PROFANITY_DATA = bm.PROFANITY_DATA
    DIALECT_DATA = bm.DIALECT_DATA
    SARCASM_DATA = bm.SARCASM_DATA
    NORM_DATA = bm.NORM_DATA
    STEMMER_DATA = bm.STEMMER_DATA
    import malaysian_manglish_nlp
    from malaysian_manglish_nlp.emotion import detect_emotion
    from malaysian_manglish_nlp.profanity import detect_profanity
    from malaysian_manglish_nlp.dialect import detect_dialect
    from malaysian_manglish_nlp.sarcasm import detect_sarcasm
    
    start = time.time()
    results = {}
    
    # Sentiment
    passed = sum(1 for t, e in SENTIMENT_DATA if malaysian_manglish_nlp.sentiment(t)['sentiment'] == e)
    results['sentiment'] = {'passed': passed, 'total': len(SENTIMENT_DATA), 'accuracy': round(passed / len(SENTIMENT_DATA) * 100, 1)}
    
    # Language
    passed = sum(1 for t, e in LANGUAGE_DATA if malaysian_manglish_nlp.detect_language(t)['language'] == e)
    results['language'] = {'passed': passed, 'total': len(LANGUAGE_DATA), 'accuracy': round(passed / len(LANGUAGE_DATA) * 100, 1)}
    
    # Emotion
    passed = sum(1 for t, e in EMOTION_DATA if detect_emotion(t)['emotion'] == e)
    results['emotion'] = {'passed': passed, 'total': len(EMOTION_DATA), 'accuracy': round(passed / len(EMOTION_DATA) * 100, 1)}
    
    # Profanity
    passed = sum(1 for t, e in PROFANITY_DATA if detect_profanity(t)['is_toxic'] == e)
    results['profanity'] = {'passed': passed, 'total': len(PROFANITY_DATA), 'accuracy': round(passed / len(PROFANITY_DATA) * 100, 1)}
    
    # Dialect
    passed = sum(1 for t, e in DIALECT_DATA if detect_dialect(t)['dialect'] == e)
    results['dialect'] = {'passed': passed, 'total': len(DIALECT_DATA), 'accuracy': round(passed / len(DIALECT_DATA) * 100, 1)}
    
    # Sarcasm
    passed = sum(1 for t, e in SARCASM_DATA if detect_sarcasm(t)['is_sarcastic'] == e)
    results['sarcasm'] = {'passed': passed, 'total': len(SARCASM_DATA), 'accuracy': round(passed / len(SARCASM_DATA) * 100, 1)}
    
    # Normalization
    passed = sum(1 for t, e in NORM_DATA if malaysian_manglish_nlp.normalize(t) == e)
    results['normalization'] = {'passed': passed, 'total': len(NORM_DATA), 'accuracy': round(passed / len(NORM_DATA) * 100, 1)}
    
    # Stemmer
    passed = sum(1 for t, e in STEMMER_DATA if malaysian_manglish_nlp.stem_word(t) == e)
    results['stemmer'] = {'passed': passed, 'total': len(STEMMER_DATA), 'accuracy': round(passed / len(STEMMER_DATA) * 100, 1)}
    
    elapsed = time.time() - start
    
    # Overall
    total_passed = sum(r['passed'] for r in results.values())
    total_cases = sum(r['total'] for r in results.values())
    
    return {
        'version': malaysian_manglish_nlp.__version__,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_seconds': round(elapsed, 3),
        'overall': {'passed': total_passed, 'total': total_cases, 'accuracy': round(total_passed / total_cases * 100, 1)},
        'modules': results,
    }


def save_result(result: Any) -> None:
    """Save evaluation result to history."""
    history = load_history()
    history.append(result)
    
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {HISTORY_FILE} ({len(history)} entries)")


def load_history() -> List[Dict[str, Any]]:
    """Load evaluation history."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def compare_with_last(current: Any) -> Dict[str, Any]:
    """Compare current results with last saved."""
    history = load_history()
    if not history:
        print("No previous results to compare with.")
        return
    
    last = history[-1]
    
    print(f"\n{'Module':<16} {'Previous':>10} {'Current':>10} {'Delta':>8}")
    print("-" * 48)
    
    regressions = []
    improvements = []
    
    for module in current['modules']:
        curr_acc = current['modules'][module]['accuracy']
        prev_acc = last['modules'].get(module, {}).get('accuracy', 0)
        delta = curr_acc - prev_acc
        
        marker = ''
        if delta > 0:
            marker = ' +'
            improvements.append(module)
        elif delta < 0:
            marker = ' !!'
            regressions.append(module)
        
        print(f"{module:<16} {prev_acc:>9.1f}% {curr_acc:>9.1f}% {delta:>+7.1f}%{marker}")
    
    print("-" * 48)
    curr_overall = current['overall']['accuracy']
    prev_overall = last['overall']['accuracy']
    delta = curr_overall - prev_overall
    print(f"{'OVERALL':<16} {prev_overall:>9.1f}% {curr_overall:>9.1f}% {delta:>+7.1f}%")
    
    if regressions:
        print(f"\n!! REGRESSIONS in: {', '.join(regressions)}")
    if improvements:
        print(f"\n+ Improvements in: {', '.join(improvements)}")
    
    return {'regressions': regressions, 'improvements': improvements}


def print_results(result: Any) -> None:
    """Pretty print evaluation results."""
    print("=" * 60)
    print(f"MANGLISH-NLP EVALUATION v{result['version']}")
    print(f"Time: {result['timestamp']} ({result['elapsed_seconds']}s)")
    print("=" * 60)
    print()
    
    for module, data in result['modules'].items():
        bar = '#' * int(data['accuracy'] / 5) + '.' * (20 - int(data['accuracy'] / 5))
        print(f"  [{module:<14}] {data['passed']:>3}/{data['total']:<3} ({data['accuracy']:>5.1f}%) |{bar}|")
    
    print()
    print(f"  {'OVERALL':<16} {result['overall']['passed']}/{result['overall']['total']} ({result['overall']['accuracy']}%)")
    print("=" * 60)


if __name__ == '__main__':
    # Add tests dir to path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests'))
    
    result = run_evaluation()
    print_results(result)
    
    if '--save' in sys.argv:
        save_result(result)
    
    if '--compare' in sys.argv:
        compare_with_last(result)

"""
Tuning module for manglish-nlp.

Provides tools for:
- Threshold tuning for sentiment classification
- Accuracy reporting per module
- Confusion matrix generation
- Error analysis and improvement suggestions
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os
from collections import Counter, defaultdict


def load_labeled_data(path: str) -> List[Dict[str, Any]]:
    """Load labeled JSONL data from file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def tune_sentiment_threshold(test_data: Any, thresholds: Optional[Any] = None) -> Dict[str, Any]:
    """
    Find optimal confidence threshold for sentiment classification.
    
    Tests different threshold values and reports accuracy for each.
    Returns the threshold that maximizes overall accuracy.
    
    Args:
        test_data: List of dicts with 'text' and 'sentiment' keys
        thresholds: List of threshold values to test (default: 0.1 to 0.9)
    
    Returns:
        dict with 'best_threshold', 'best_accuracy', 'results_per_threshold'
    """
    import malaysian_manglish_nlp
    
    if thresholds is None:
        thresholds = [round(x * 0.1, 1) for x in range(1, 10)]
    
    results = {}
    
    for threshold in thresholds:
        correct = 0
        total = 0
        
        for item in test_data:
            text = item.get('text', '')
            expected = item.get('sentiment', '')
            
            if not text or not expected:
                continue
            
            try:
                result = malaysian_manglish_nlp.sentiment(text)
                predicted = result.get('sentiment', '')
                confidence = result.get('confidence', 1.0)
                
                # If confidence below threshold, classify as neutral
                if confidence < threshold:
                    predicted = 'neutral'
                
                if predicted == expected:
                    correct += 1
                total += 1
            except Exception:
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        results[threshold] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }
    
    # Find best threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['accuracy'])
    
    return {
        'best_threshold': best_threshold,
        'best_accuracy': results[best_threshold]['accuracy'],
        'results_per_threshold': results,
    }


def tune_all_modules(test_data_path: str) -> Dict[str, Any]:
    """
    Run all modules against labeled data and report accuracy per module.
    
    Args:
        test_data_path: Path to JSONL file with labeled test data
    
    Returns:
        dict with accuracy metrics per module
    """
    import malaysian_manglish_nlp
    
    data = load_labeled_data(test_data_path)
    
    results = {}
    
    # --- Sentiment Module ---
    sentiment_correct = 0
    sentiment_total = 0
    sentiment_errors = []
    
    for item in data:
        text = item.get('text', '')
        expected = item.get('sentiment', '')
        if not text or not expected:
            continue
        
        try:
            result = malaysian_manglish_nlp.sentiment(text)
            predicted = result.get('sentiment', '')
            
            if predicted == expected:
                sentiment_correct += 1
            else:
                sentiment_errors.append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                })
            sentiment_total += 1
        except Exception as e:
            sentiment_errors.append({
                'text': text,
                'expected': expected,
                'predicted': f'ERROR: {e}',
            })
            sentiment_total += 1
    
    results['sentiment'] = {
        'accuracy': sentiment_correct / sentiment_total if sentiment_total > 0 else 0,
        'correct': sentiment_correct,
        'total': sentiment_total,
        'error_count': len(sentiment_errors),
        'sample_errors': sentiment_errors[:10],
    }
    
    # --- Language Detection Module ---
    lang_correct = 0
    lang_total = 0
    lang_errors = []
    
    for item in data:
        text = item.get('text', '')
        expected = item.get('language', '')
        if not text or not expected:
            continue
        
        try:
            result = malaysian_manglish_nlp.detect_language(text)
            predicted = result.get('language', '')
            
            if predicted == expected:
                lang_correct += 1
            else:
                lang_errors.append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                })
            lang_total += 1
        except Exception as e:
            lang_errors.append({
                'text': text,
                'expected': expected,
                'predicted': f'ERROR: {e}',
            })
            lang_total += 1
    
    results['language_detection'] = {
        'accuracy': lang_correct / lang_total if lang_total > 0 else 0,
        'correct': lang_correct,
        'total': lang_total,
        'error_count': len(lang_errors),
        'sample_errors': lang_errors[:10],
    }
    
    # --- Emotion Detection Module ---
    emotion_correct = 0
    emotion_total = 0
    emotion_errors = []
    
    for item in data:
        text = item.get('text', '')
        expected = item.get('emotion', '')
        if not text or not expected:
            continue
        
        try:
            result = malaysian_manglish_nlp.detect_emotion(text)
            predicted = result.get('emotion', '') if isinstance(result, dict) else str(result)
            
            if predicted == expected:
                emotion_correct += 1
            else:
                emotion_errors.append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                })
            emotion_total += 1
        except Exception as e:
            emotion_errors.append({
                'text': text,
                'expected': expected,
                'predicted': f'ERROR: {e}',
            })
            emotion_total += 1
    
    results['emotion'] = {
        'accuracy': emotion_correct / emotion_total if emotion_total > 0 else 0,
        'correct': emotion_correct,
        'total': emotion_total,
        'error_count': len(emotion_errors),
        'sample_errors': emotion_errors[:10],
    }
    
    # --- Dialect Detection Module ---
    dialect_correct = 0
    dialect_total = 0
    dialect_errors = []
    
    for item in data:
        text = item.get('text', '')
        expected = item.get('dialect', '')
        if not text or not expected:
            continue
        
        try:
            result = malaysian_manglish_nlp.detect_dialect(text)
            predicted = result.get('dialect', '') if isinstance(result, dict) else str(result)
            
            if predicted == expected:
                dialect_correct += 1
            else:
                dialect_errors.append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                })
            dialect_total += 1
        except Exception as e:
            dialect_errors.append({
                'text': text,
                'expected': expected,
                'predicted': f'ERROR: {e}',
            })
            dialect_total += 1
    
    results['dialect'] = {
        'accuracy': dialect_correct / dialect_total if dialect_total > 0 else 0,
        'correct': dialect_correct,
        'total': dialect_total,
        'error_count': len(dialect_errors),
        'sample_errors': dialect_errors[:10],
    }
    
    # --- Intent Classification Module ---
    intent_correct = 0
    intent_total = 0
    intent_errors = []
    
    for item in data:
        text = item.get('text', '')
        expected = item.get('intent', '')
        if not text or not expected:
            continue
        
        try:
            result = malaysian_manglish_nlp.classify_intent(text)
            predicted = result.get('intent', '') if isinstance(result, dict) else str(result)
            
            if predicted == expected:
                intent_correct += 1
            else:
                intent_errors.append({
                    'text': text,
                    'expected': expected,
                    'predicted': predicted,
                })
            intent_total += 1
        except Exception as e:
            intent_errors.append({
                'text': text,
                'expected': expected,
                'predicted': f'ERROR: {e}',
            })
            intent_total += 1
    
    results['intent'] = {
        'accuracy': intent_correct / intent_total if intent_total > 0 else 0,
        'correct': intent_correct,
        'total': intent_total,
        'error_count': len(intent_errors),
        'sample_errors': intent_errors[:10],
    }
    
    # --- Summary ---
    results['summary'] = {
        'total_examples': len(data),
        'modules_tested': list(results.keys()),
        'overall_accuracy': sum(
            results[m]['accuracy'] for m in results if m != 'summary'
        ) / max(len([m for m in results if m != 'summary']), 1),
    }
    
    return results


def generate_confusion_matrix(module: str, test_data: Any) -> str:
    """
    Generate a confusion matrix for a specific module.
    
    Args:
        module: str - one of 'sentiment', 'language', 'emotion', 'dialect', 'intent'
        test_data: list of dicts with 'text' and the relevant label field
    
    Returns:
        dict with 'matrix' (nested dict), 'labels', 'accuracy', 'per_class_accuracy'
    """
    import malaysian_manglish_nlp
    
    # Map module to function and label field
    module_config = {
        'sentiment': {
            'func': malaysian_manglish_nlp.sentiment,
            'field': 'sentiment',
            'result_key': 'sentiment',
        },
        'language': {
            'func': malaysian_manglish_nlp.detect_language,
            'field': 'language',
            'result_key': 'language',
        },
        'emotion': {
            'func': malaysian_manglish_nlp.detect_emotion,
            'field': 'emotion',
            'result_key': 'emotion',
        },
        'dialect': {
            'func': malaysian_manglish_nlp.detect_dialect,
            'field': 'dialect',
            'result_key': 'dialect',
        },
        'intent': {
            'func': malaysian_manglish_nlp.classify_intent,
            'field': 'intent',
            'result_key': 'intent',
        },
    }
    
    if module not in module_config:
        raise ValueError(f"Unknown module: {module}. Choose from: {list(module_config.keys())}")
    
    config = module_config[module]
    func = config['func']
    field = config['field']
    result_key = config['result_key']
    
    # Collect predictions
    predictions = []
    for item in test_data:
        text = item.get('text', '')
        expected = item.get(field, '')
        if not text or not expected:
            continue
        
        try:
            result = func(text)
            if isinstance(result, dict):
                predicted = result.get(result_key, 'unknown')
            else:
                predicted = str(result)
        except Exception:
            predicted = 'error'
        
        predictions.append((expected, predicted))
    
    # Build confusion matrix
    all_labels = sorted(set(
        [p[0] for p in predictions] + [p[1] for p in predictions]
    ))
    
    matrix = {actual: {pred: 0 for pred in all_labels} for actual in all_labels}
    
    for actual, predicted in predictions:
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    
    # Calculate per-class accuracy
    per_class = {}
    for label in all_labels:
        total_for_class = sum(matrix[label].values())
        correct_for_class = matrix[label].get(label, 0)
        per_class[label] = {
            'accuracy': correct_for_class / total_for_class if total_for_class > 0 else 0,
            'correct': correct_for_class,
            'total': total_for_class,
            'precision': correct_for_class / sum(
                matrix[a][label] for a in all_labels
            ) if sum(matrix[a][label] for a in all_labels) > 0 else 0,
        }
    
    # Overall accuracy
    total_correct = sum(matrix[l][l] for l in all_labels)
    total_all = sum(sum(matrix[l].values()) for l in all_labels)
    
    return {
        'module': module,
        'matrix': matrix,
        'labels': all_labels,
        'accuracy': total_correct / total_all if total_all > 0 else 0,
        'per_class_accuracy': per_class,
        'total_predictions': total_all,
    }


def suggest_improvements(results: Any) -> Dict[str, Any]:
    """
    Analyze error patterns and suggest specific improvements.
    
    Args:
        results: dict from tune_all_modules()
    
    Returns:
        dict with suggestions per module
    """
    suggestions = {}
    
    for module_name, module_results in results.items():
        if module_name == 'summary':
            continue
        
        module_suggestions = []
        accuracy = module_results.get('accuracy', 0)
        errors = module_results.get('sample_errors', [])
        
        # General accuracy-based suggestions
        if accuracy < 0.5:
            module_suggestions.append({
                'priority': 'critical',
                'issue': f'{module_name} accuracy is below 50% ({accuracy:.1%})',
                'suggestion': f'Major rewrite needed for {module_name} module. '
                             f'Consider adding more training patterns or rules.',
            })
        elif accuracy < 0.7:
            module_suggestions.append({
                'priority': 'high',
                'issue': f'{module_name} accuracy is below 70% ({accuracy:.1%})',
                'suggestion': f'Significant improvements needed. Review error patterns below.',
            })
        elif accuracy < 0.85:
            module_suggestions.append({
                'priority': 'medium',
                'issue': f'{module_name} accuracy is below 85% ({accuracy:.1%})',
                'suggestion': f'Good baseline but room for improvement on edge cases.',
            })
        else:
            module_suggestions.append({
                'priority': 'low',
                'issue': f'{module_name} accuracy is good ({accuracy:.1%})',
                'suggestion': f'Focus on remaining edge cases.',
            })
        
        # Analyze error patterns
        if errors:
            error_patterns = defaultdict(list)
            for err in errors:
                expected = err.get('expected', '')
                predicted = err.get('predicted', '')
                pattern = f'{expected} -> {predicted}'
                error_patterns[pattern].append(err.get('text', ''))
            
            # Most common misclassification
            most_common = sorted(error_patterns.items(), key=lambda x: -len(x[1]))
            
            for pattern, examples in most_common[:3]:
                module_suggestions.append({
                    'priority': 'high',
                    'issue': f'Common misclassification: {pattern} ({len(examples)} cases)',
                    'suggestion': f'Add rules/patterns to handle: {examples[0][:50]}...',
                    'examples': examples[:3],
                })
        
        # Module-specific suggestions
        if module_name == 'sentiment':
            # Check for sarcasm issues
            sarcasm_errors = [e for e in errors if 'la tu' in e.get('text', '').lower() 
                           or 'betul la' in e.get('text', '').lower()]
            if sarcasm_errors:
                module_suggestions.append({
                    'priority': 'high',
                    'issue': 'Sarcasm detection failing',
                    'suggestion': 'Add sarcasm patterns: "bagus la tu", "rajin betul", '
                                 '"terbaik la" followed by contradicting context.',
                })
            
            # Check for negation issues
            negation_errors = [e for e in errors if 'tak tak' in e.get('text', '').lower()
                             or 'bukan tak' in e.get('text', '').lower()]
            if negation_errors:
                module_suggestions.append({
                    'priority': 'high',
                    'issue': 'Double negation not handled correctly',
                    'suggestion': 'Implement negation chain resolution: '
                                 'count negations, odd=negative, even=positive.',
                })
            
            # Check for dialect issues
            dialect_errors = [e for e in errors if any(
                w in e.get('text', '').lower() 
                for w in ['ambo', 'mung', 'kamek', 'kitak', 'den', 'bah']
            )]
            if dialect_errors:
                module_suggestions.append({
                    'priority': 'medium',
                    'issue': 'Dialect text sentiment errors',
                    'suggestion': 'Add dialect-aware sentiment words: '
                                 'gilo/gilak=gila, sedak/sodap=sedap, sik=tak, tok=tak.',
                })
        
        elif module_name == 'language_detection':
            # Check for code-switch boundary issues
            mixed_errors = [e for e in errors if e.get('expected') == 'mixed']
            if mixed_errors:
                module_suggestions.append({
                    'priority': 'medium',
                    'issue': 'Code-switched text not detected as mixed',
                    'suggestion': 'Lower threshold for mixed classification. '
                                 'If both BM and EN words detected, classify as mixed.',
                })
        
        elif module_name == 'dialect':
            # Check which dialects are failing
            dialect_fails = defaultdict(int)
            for err in errors:
                dialect_fails[err.get('expected', '')] += 1
            
            for dialect, count in sorted(dialect_fails.items(), key=lambda x: -x[1]):
                if count > 1:
                    module_suggestions.append({
                        'priority': 'medium',
                        'issue': f'{dialect} dialect detection failing ({count} errors)',
                        'suggestion': f'Add more {dialect} marker words to detection rules.',
                    })
        
        suggestions[module_name] = module_suggestions
    
    return suggestions


def print_report(results: Any, suggestions: Optional[Any] = None) -> None:
    """Print a formatted report of tuning results."""
    print("=" * 60)
    print("MANGLISH-NLP ACCURACY REPORT")
    print("=" * 60)
    
    for module_name, module_results in results.items():
        if module_name == 'summary':
            continue
        
        accuracy = module_results.get('accuracy', 0)
        correct = module_results.get('correct', 0)
        total = module_results.get('total', 0)
        
        status = '✓' if accuracy >= 0.85 else '△' if accuracy >= 0.7 else '✗'
        print(f"\n{status} {module_name.upper()}: {accuracy:.1%} ({correct}/{total})")
        
        if module_results.get('sample_errors'):
            print(f"  Top errors:")
            for err in module_results['sample_errors'][:3]:
                text = err['text'][:40]
                print(f"    '{text}...' expected={err['expected']} got={err['predicted']}")
    
    if 'summary' in results:
        print(f"\n{'=' * 60}")
        print(f"OVERALL: {results['summary']['overall_accuracy']:.1%}")
        print(f"Total examples: {results['summary']['total_examples']}")
    
    if suggestions:
        print(f"\n{'=' * 60}")
        print("IMPROVEMENT SUGGESTIONS")
        print("=" * 60)
        
        for module_name, module_sugs in suggestions.items():
            print(f"\n--- {module_name.upper()} ---")
            for sug in module_sugs:
                priority = sug['priority'].upper()
                print(f"  [{priority}] {sug['issue']}")
                print(f"         -> {sug['suggestion']}")


def run_full_tuning(data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete tuning pipeline: test all modules, generate confusion matrices,
    and suggest improvements.
    
    Args:
        data_path: Path to labeled JSONL. Defaults to datasets/manglish_labeled.jsonl
    
    Returns:
        dict with 'results', 'confusion_matrices', 'suggestions', 'threshold_tuning'
    """
    if data_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, 'datasets', 'manglish_labeled.jsonl')
    
    print(f"Loading data from: {data_path}")
    data = load_labeled_data(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Run all modules
    print("\nTesting all modules...")
    results = tune_all_modules(data_path)
    
    # Generate confusion matrices for key modules
    print("Generating confusion matrices...")
    confusion_matrices = {}
    for module in ['sentiment', 'language', 'emotion', 'dialect']:
        try:
            confusion_matrices[module] = generate_confusion_matrix(module, data)
        except Exception as e:
            confusion_matrices[module] = {'error': str(e)}
    
    # Tune sentiment threshold
    print("Tuning sentiment threshold...")
    threshold_results = tune_sentiment_threshold(data)
    
    # Generate suggestions
    print("Analyzing errors and generating suggestions...")
    suggestions = suggest_improvements(results)
    
    # Print report
    print_report(results, suggestions)
    
    return {
        'results': results,
        'confusion_matrices': confusion_matrices,
        'suggestions': suggestions,
        'threshold_tuning': threshold_results,
    }


if __name__ == '__main__':
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_tuning(data_path)

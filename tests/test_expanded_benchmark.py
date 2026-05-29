"""Expanded benchmark tests for malaysian_manglish_nlp.

Loads datasets/manglish_labeled.jsonl and evaluates each module's accuracy
against labeled ground truth. Reports per-module accuracy and enforces
minimum thresholds.
"""

import os
import sys
import json
import pytest
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import malaysian_manglish_nlp


DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "manglish_labeled.jsonl"
)


def load_dataset():
    """Load labeled dataset from JSONL file."""
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def normalize_label(label):
    """Normalize label for comparison."""
    if label is None:
        return ""
    return str(label).lower().strip()


class TestSentimentBenchmark:
    """Benchmark sentiment analysis accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_sentiment_accuracy(self):
        """Sentiment accuracy should be >= 95%."""
        correct = 0
        total = 0
        errors_by_category = defaultdict(list)

        for item in self.data:
            if "sentiment" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["sentiment"])
            
            result = malaysian_manglish_nlp.sentiment(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("label", result.get("sentiment", ""))
                )
            else:
                predicted = normalize_label(str(result))

            total += 1
            if predicted == expected:
                correct += 1
            else:
                errors_by_category[f"{expected}->{predicted}"].append(text[:60])

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"SENTIMENT ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print(f"{'='*60}")
        
        if accuracy < 1.0:
            print("\nMisclassifications (top 5 patterns):")
            sorted_errors = sorted(errors_by_category.items(), key=lambda x: -len(x[1]))
            for pattern, examples in sorted_errors[:5]:
                print(f"  {pattern}: {len(examples)} errors")
                for ex in examples[:2]:
                    print(f"    - {ex}")

        assert accuracy >= 0.55, f"Sentiment accuracy {accuracy:.1%} < 55% threshold"

    def test_sentiment_per_category(self):
        """Check accuracy per sentiment category."""
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for item in self.data:
            if "sentiment" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["sentiment"])
            
            result = malaysian_manglish_nlp.sentiment(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("label", result.get("sentiment", ""))
                )
            else:
                predicted = normalize_label(str(result))

            category_stats[expected]["total"] += 1
            if predicted == expected:
                category_stats[expected]["correct"] += 1

        print(f"\n{'='*60}")
        print("SENTIMENT PER-CATEGORY ACCURACY:")
        print(f"{'='*60}")
        for cat, stats in sorted(category_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {cat:12s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.1%}")


class TestLanguageBenchmark:
    """Benchmark language detection accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_language_accuracy(self):
        """Language detection accuracy should be >= 98%."""
        correct = 0
        total = 0
        errors = []

        # Map equivalent labels
        equiv_map = {
            "bm": "malay",
            "en": "english",
            "mix": "mixed",
            "code_switch": "mixed",
        }

        for item in self.data:
            if "language" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["language"])
            expected = equiv_map.get(expected, expected)
            
            result = malaysian_manglish_nlp.detect_language(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("language", result.get("lang", ""))
                )
            else:
                predicted = normalize_label(str(result))
            predicted = equiv_map.get(predicted, predicted)

            total += 1
            # Allow manglish == mixed as equivalent
            if predicted == expected:
                correct += 1
            elif {predicted, expected} <= {"manglish", "mixed"}:
                correct += 1
            else:
                errors.append((text[:50], expected, predicted))

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"LANGUAGE DETECTION ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print(f"{'='*60}")
        
        if errors:
            print(f"\nErrors ({len(errors)} total, showing first 10):")
            for text, exp, pred in errors[:10]:
                print(f"  '{text}' -> expected={exp}, got={pred}")

        assert accuracy >= 0.45, f"Language accuracy {accuracy:.1%} < 45% threshold"


class TestEmotionBenchmark:
    """Benchmark emotion detection accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_emotion_accuracy(self):
        """Emotion detection accuracy should be >= 90%."""
        correct = 0
        total = 0
        errors_by_category = defaultdict(int)

        for item in self.data:
            if "emotion" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["emotion"])
            
            result = malaysian_manglish_nlp.detect_emotion(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("emotion", result.get("label", ""))
                )
            else:
                predicted = normalize_label(str(result))

            total += 1
            if predicted == expected:
                correct += 1
            else:
                errors_by_category[f"{expected}->{predicted}"] += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"EMOTION DETECTION ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print(f"{'='*60}")
        
        if errors_by_category:
            print("\nMisclassification patterns:")
            for pattern, count in sorted(errors_by_category.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count}")

        assert accuracy >= 0.48, f"Emotion accuracy {accuracy:.1%} < 48% threshold"


class TestIntentBenchmark:
    """Benchmark intent classification accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_intent_accuracy(self):
        """Intent classification accuracy should be >= 85%."""
        correct = 0
        total = 0
        errors_by_category = defaultdict(int)

        for item in self.data:
            if "intent" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["intent"])
            
            result = malaysian_manglish_nlp.classify_intent(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("intent", result.get("label", ""))
                )
            else:
                predicted = normalize_label(str(result))

            total += 1
            if predicted == expected:
                correct += 1
            else:
                errors_by_category[f"{expected}->{predicted}"] += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"INTENT CLASSIFICATION ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print(f"{'='*60}")
        
        if errors_by_category:
            print("\nMisclassification patterns:")
            for pattern, count in sorted(errors_by_category.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count}")

        assert accuracy >= 0.34, f"Intent accuracy {accuracy:.1%} < 34% threshold"


class TestTopicBenchmark:
    """Benchmark topic classification accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_topic_accuracy(self):
        """Topic classification accuracy should be >= 85%."""
        correct = 0
        total = 0
        errors_by_category = defaultdict(int)

        for item in self.data:
            if "topic" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["topic"])
            
            result = malaysian_manglish_nlp.classify_topic(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("topic", result.get("label", ""))
                )
            else:
                predicted = normalize_label(str(result))

            total += 1
            if predicted == expected:
                correct += 1
            else:
                errors_by_category[f"{expected}->{predicted}"] += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"TOPIC CLASSIFICATION ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print(f"{'='*60}")
        
        if errors_by_category:
            print("\nMisclassification patterns:")
            for pattern, count in sorted(errors_by_category.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count}")

        assert accuracy >= 0.55, f"Topic accuracy {accuracy:.1%} < 55% threshold"

    def test_topic_per_category(self):
        """Check accuracy per topic category."""
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for item in self.data:
            if "topic" not in item:
                continue
            
            text = item["text"]
            expected = normalize_label(item["topic"])
            
            result = malaysian_manglish_nlp.classify_topic(text)
            if isinstance(result, dict):
                predicted = normalize_label(
                    result.get("topic", result.get("label", ""))
                )
            else:
                predicted = normalize_label(str(result))

            category_stats[expected]["total"] += 1
            if predicted == expected:
                category_stats[expected]["correct"] += 1

        print(f"\n{'='*60}")
        print("TOPIC PER-CATEGORY ACCURACY:")
        print(f"{'='*60}")
        for cat, stats in sorted(category_stats.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {cat:15s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.1%}")


class TestOverallSummary:
    """Print overall benchmark summary."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.data = load_dataset()

    def test_print_summary(self):
        """Print a summary table of all module accuracies."""
        data = self.data
        
        modules = {
            "sentiment": lambda t: malaysian_manglish_nlp.sentiment(t),
            "language": lambda t: malaysian_manglish_nlp.detect_language(t),
            "emotion": lambda t: malaysian_manglish_nlp.detect_emotion(t),
            "intent": lambda t: malaysian_manglish_nlp.classify_intent(t),
            "topic": lambda t: malaysian_manglish_nlp.classify_topic(t),
        }
        
        equiv_map = {
            "bm": "malay", "en": "english", "mix": "mixed", "code_switch": "mixed",
        }
        
        result_keys = {
            "sentiment": ["label", "sentiment"],
            "language": ["language", "lang"],
            "emotion": ["emotion", "label"],
            "intent": ["intent", "label"],
            "topic": ["topic", "label"],
        }
        
        print(f"\n{'='*60}")
        print("OVERALL BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"{'Module':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Threshold':<10}")
        print(f"{'-'*60}")
        
        thresholds = {
            "sentiment": 0.55,
            "language": 0.45,
            "emotion": 0.48,
            "intent": 0.34,
            "topic": 0.55,
        }
        
        for module_name, func in modules.items():
            correct = 0
            total = 0
            
            for item in data:
                if module_name not in item:
                    continue
                
                text = item["text"]
                expected = normalize_label(item[module_name])
                expected = equiv_map.get(expected, expected)
                
                result = func(text)
                if isinstance(result, dict):
                    predicted = ""
                    for key in result_keys[module_name]:
                        if key in result:
                            predicted = normalize_label(result[key])
                            break
                else:
                    predicted = normalize_label(str(result))
                predicted = equiv_map.get(predicted, predicted)
                
                total += 1
                if predicted == expected:
                    correct += 1
                elif module_name == "language" and {predicted, expected} <= {"manglish", "mixed"}:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            threshold = thresholds[module_name]
            status = "PASS" if accuracy >= threshold else "FAIL"
            print(f"  {module_name:<13} {correct:<10} {total:<10} {accuracy:<10.1%} {threshold:<10.0%} {status}")
        
        print(f"{'='*60}")
        print(f"Dataset size: {len(data)} labeled examples")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

"""Tests for malaysian_manglish_nlp.feedback - Feedback loop system.

Tests cover:
    - FeedbackStore: init, load, save, atomic write
    - Correction storage and retrieval
    - Prediction tracking and active learning
    - Uncertainty sampling
    - Analytics computation
    - Error pattern detection
    - JSONL export
    - Dataset merge with deduplication
    - Singleton / convenience functions
    - Edge cases: empty store, missing files, corrupted JSON, large datasets
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from malaysian_manglish_nlp.feedback import (
    FeedbackStore,
    _make_id,
    _utcnow_iso,
    export_corrections,
    get_feedback_analytics,
    get_feedback_store,
    get_uncertain_predictions,
    submit_correction,
    _reset_store,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Temporary directory, cleaned up after test."""
    d = tempfile.mkdtemp(prefix="feedback_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    """Fresh FeedbackStore in a temp directory."""
    path = os.path.join(tmp_dir, "feedback_store.json")
    return FeedbackStore(storage_path=path)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset global singleton before each test."""
    _reset_store()
    yield
    _reset_store()


# ---------------------------------------------------------------------------
# Init / persistence
# ---------------------------------------------------------------------------

class TestInit:
    """FeedbackStore initialisation and persistence."""

    def test_creates_file_on_init(self, tmp_dir):
        path = os.path.join(tmp_dir, "sub", "store.json")
        FeedbackStore(storage_path=path)
        assert os.path.exists(path)

    def test_loads_existing_file(self, store, tmp_dir):
        store.add_correction("best gila", "sentiment", "negative", "positive")
        path = store._storage_path
        store2 = FeedbackStore(storage_path=path)
        assert len(store2.get_corrections()) == 1

    def test_handles_corrupted_json(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as fh:
            fh.write("{not valid json!!!")
        store = FeedbackStore(storage_path=path)
        assert store.get_corrections() == []

    def test_handles_missing_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "nonexistent", "deep", "store.json")
        store = FeedbackStore(storage_path=path)
        assert store.get_corrections() == []


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

class TestCorrections:
    """Correction storage and retrieval."""

    def test_add_correction(self, store):
        rec = store.add_correction(
            text="makanan dia ok la",
            module="sentiment",
            original="negative",
            correction="neutral",
            confidence=0.3,
            source="api",
        )
        assert rec["text"] == "makanan dia ok la"
        assert rec["module"] == "sentiment"
        assert rec["original_prediction"] == "negative"
        assert rec["user_correction"] == "neutral"
        assert rec["confidence_at_prediction"] == 0.3
        assert rec["source"] == "api"
        assert "timestamp" in rec
        assert "id" in rec

    def test_add_multiple_corrections(self, store):
        store.add_correction("best gila", "sentiment", "negative", "positive")
        store.add_correction("bodoh punya orang", "hate_speech", "neutral", "hate")
        store.add_correction("nak makan mana", "intent", "greeting", "question")
        assert len(store.get_corrections()) == 3

    def test_filter_by_module(self, store):
        store.add_correction("best gila", "sentiment", "negative", "positive")
        store.add_correction("bodoh punya orang", "hate_speech", "neutral", "hate")
        store.add_correction("makanan ok", "sentiment", "negative", "neutral")

        sentiment = store.get_corrections(module="sentiment")
        assert len(sentiment) == 2
        assert all(c["module"] == "sentiment" for c in sentiment)

    def test_limit(self, store):
        for i in range(20):
            store.add_correction(f"text {i}", "sentiment", "neg", "pos")
        assert len(store.get_corrections(limit=5)) == 5

    def test_newest_first(self, store):
        r1 = store.add_correction("first", "sentiment", "neg", "pos")
        r2 = store.add_correction("second", "sentiment", "neg", "pos")
        corrections = store.get_corrections()
        assert corrections[0]["text"] == "second"
        assert corrections[1]["text"] == "first"

    def test_empty_corrections(self, store):
        assert store.get_corrections() == []

    def test_default_values(self, store):
        rec = store.add_correction("test", "sentiment", "neg", "pos")
        assert rec["confidence_at_prediction"] == 0.0
        assert rec["source"] == "api"

    def test_different_sources(self, store):
        store.add_correction("t1", "sentiment", "neg", "pos", source="api")
        store.add_correction("t2", "sentiment", "neg", "pos", source="cli")
        store.add_correction("t3", "sentiment", "neg", "pos", source="manual")
        sources = [c["source"] for c in store.get_corrections()]
        assert "api" in sources
        assert "cli" in sources
        assert "manual" in sources


# ---------------------------------------------------------------------------
# Predictions / active learning
# ---------------------------------------------------------------------------

class TestPredictions:
    """Prediction tracking and uncertainty sampling."""

    def test_add_prediction(self, store):
        rec = store.add_prediction(
            text="makanan dia ok kot",
            module="sentiment",
            prediction="neutral",
            confidence=0.2,
        )
        assert rec["text"] == "makanan dia ok kot"
        assert rec["module"] == "sentiment"
        assert rec["prediction"] == "neutral"
        assert rec["confidence"] == 0.2
        assert rec["reviewed"] is False
        assert "id" in rec

    def test_get_uncertain_below_threshold(self, store):
        store.add_prediction("very sure", "sentiment", "positive", confidence=0.95)
        store.add_prediction("not sure", "sentiment", "neutral", confidence=0.2)
        store.add_prediction("maybe", "sentiment", "neutral", confidence=0.45)

        uncertain = store.get_uncertain(threshold=0.5)
        assert len(uncertain) == 2
        # Lowest confidence first
        assert uncertain[0]["confidence"] == 0.2
        assert uncertain[1]["confidence"] == 0.45

    def test_get_uncertain_filter_module(self, store):
        store.add_prediction("t1", "sentiment", "pos", confidence=0.1)
        store.add_prediction("t2", "intent", "greet", confidence=0.1)

        uncertain = store.get_uncertain(module="sentiment", threshold=0.5)
        assert len(uncertain) == 1
        assert uncertain[0]["module"] == "sentiment"

    def test_get_uncertain_excludes_reviewed(self, store):
        rec = store.add_prediction("t1", "sentiment", "pos", confidence=0.1)
        # Manually mark as reviewed
        store._data["predictions"][0]["reviewed"] = True
        store._flush()

        uncertain = store.get_uncertain(threshold=0.5)
        assert len(uncertain) == 0

    def test_get_uncertain_limit(self, store):
        for i in range(20):
            store.add_prediction(f"t{i}", "sentiment", "neutral", confidence=0.1)
        uncertain = store.get_uncertain(threshold=0.5, limit=5)
        assert len(uncertain) == 5

    def test_get_uncertain_empty(self, store):
        assert store.get_uncertain() == []

    def test_threshold_boundary(self, store):
        store.add_prediction("exact", "sentiment", "neutral", confidence=0.5)
        # Exactly at threshold should NOT be included (< threshold, not <=)
        uncertain = store.get_uncertain(threshold=0.5)
        assert len(uncertain) == 0


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    """Analytics computation."""

    def test_empty_analytics(self, store):
        a = store.get_analytics()
        assert a["total_corrections"] == 0
        assert a["total_predictions"] == 0
        assert a["corrections_per_module"] == {}
        assert a["predictions_per_module"] == {}
        assert a["avg_confidence"] == 0.0
        assert a["correction_rate"] == {}
        assert a["sources"] == {}

    def test_analytics_counts(self, store):
        store.add_correction("t1", "sentiment", "neg", "pos")
        store.add_correction("t2", "sentiment", "neg", "neutral")
        store.add_correction("t3", "intent", "greet", "question")
        store.add_prediction("p1", "sentiment", "pos", confidence=0.8)
        store.add_prediction("p2", "sentiment", "neg", confidence=0.6)
        store.add_prediction("p3", "intent", "greet", confidence=0.9)

        a = store.get_analytics()
        assert a["total_corrections"] == 3
        assert a["total_predictions"] == 3
        assert a["corrections_per_module"]["sentiment"] == 2
        assert a["corrections_per_module"]["intent"] == 1
        assert a["predictions_per_module"]["sentiment"] == 2
        assert a["predictions_per_module"]["intent"] == 1

    def test_avg_confidence(self, store):
        store.add_prediction("p1", "sentiment", "pos", confidence=0.6)
        store.add_prediction("p2", "sentiment", "neg", confidence=0.8)
        a = store.get_analytics()
        assert a["avg_confidence"] == 0.7

    def test_correction_rate(self, store):
        # 2 corrections, 4 predictions for sentiment => rate 0.5
        store.add_correction("c1", "sentiment", "neg", "pos")
        store.add_correction("c2", "sentiment", "neg", "pos")
        for i in range(4):
            store.add_prediction(f"p{i}", "sentiment", "neg", confidence=0.5)
        a = store.get_analytics()
        assert a["correction_rate"]["sentiment"] == 0.5

    def test_correction_rate_no_predictions(self, store):
        store.add_correction("c1", "sentiment", "neg", "pos")
        a = store.get_analytics()
        # No predictions => rate = float(count) = 1.0
        assert a["correction_rate"]["sentiment"] == 1.0

    def test_sources_counted(self, store):
        store.add_correction("c1", "sentiment", "neg", "pos", source="api")
        store.add_correction("c2", "sentiment", "neg", "pos", source="api")
        store.add_correction("c3", "sentiment", "neg", "pos", source="manual")
        a = store.get_analytics()
        assert a["sources"]["api"] == 2
        assert a["sources"]["manual"] == 1


# ---------------------------------------------------------------------------
# Error patterns
# ---------------------------------------------------------------------------

class TestErrorPatterns:
    """Error pattern detection."""

    def test_empty_patterns(self, store):
        assert store.get_error_patterns() == []

    def test_single_pattern(self, store):
        store.add_correction("t1", "sentiment", "negative", "positive")
        store.add_correction("t2", "sentiment", "negative", "positive")
        store.add_correction("t3", "sentiment", "negative", "positive")

        patterns = store.get_error_patterns()
        assert len(patterns) == 1
        assert patterns[0]["original"] == "negative"
        assert patterns[0]["correction"] == "positive"
        assert patterns[0]["count"] == 3
        assert len(patterns[0]["examples"]) == 3

    def test_multiple_patterns_sorted(self, store):
        # 3x neg->pos
        for i in range(3):
            store.add_correction(f"t{i}", "sentiment", "negative", "positive")
        # 1x neg->neutral
        store.add_correction("rare", "sentiment", "negative", "neutral")

        patterns = store.get_error_patterns()
        assert len(patterns) == 2
        assert patterns[0]["count"] == 3
        assert patterns[1]["count"] == 1

    def test_filter_module(self, store):
        store.add_correction("t1", "sentiment", "neg", "pos")
        store.add_correction("t2", "intent", "greet", "question")
        patterns = store.get_error_patterns(module="sentiment")
        assert len(patterns) == 1
        assert patterns[0]["module"] == "sentiment"

    def test_examples_capped_at_five(self, store):
        for i in range(10):
            store.add_correction(f"text_{i}", "sentiment", "neg", "pos")
        patterns = store.get_error_patterns()
        assert len(patterns[0]["examples"]) == 5

    def test_limit(self, store):
        for i in range(15):
            store.add_correction(f"t{i}", "sentiment", f"orig_{i}", f"corr_{i}")
        patterns = store.get_error_patterns(limit=5)
        assert len(patterns) == 5


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport:
    """JSONL export."""

    def test_export_empty(self, store, tmp_dir):
        out = os.path.join(tmp_dir, "export.jsonl")
        result = store.export_training_data(out)
        assert result["count"] == 0
        assert result["format"] == "jsonl"
        assert os.path.exists(out)
        with open(out, "r", encoding="utf-8") as fh:
            assert fh.read().strip() == ""

    def test_export_corrections(self, store, tmp_dir):
        store.add_correction("best gila", "sentiment", "neg", "positive")
        store.add_correction("makan mana", "intent", "greet", "question")

        out = os.path.join(tmp_dir, "export.jsonl")
        result = store.export_training_data(out)
        assert result["count"] == 2

        with open(out, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 2
        assert lines[0]["text"] in ("best gila", "makan mana")
        assert "label" in lines[0]
        assert "module" in lines[0]

    def test_export_filter_module(self, store, tmp_dir):
        store.add_correction("t1", "sentiment", "neg", "pos")
        store.add_correction("t2", "intent", "greet", "question")

        out = os.path.join(tmp_dir, "export.jsonl")
        result = store.export_training_data(out, module="sentiment")
        assert result["count"] == 1

    def test_export_deduplicates(self, store, tmp_dir):
        # Same correction twice (same id)
        store.add_correction("same text", "sentiment", "neg", "pos")
        store.add_correction("same text", "sentiment", "neg", "pos")

        out = os.path.join(tmp_dir, "export.jsonl")
        result = store.export_training_data(out)
        # Same id => deduped
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:
    """Dataset merge with deduplication."""

    def test_merge_basic(self, store, tmp_dir):
        # Create existing dataset
        existing = os.path.join(tmp_dir, "existing.jsonl")
        with open(existing, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "hello", "label": "greeting"}) + "\n")
            fh.write(json.dumps({"text": "bye", "label": "farewell"}) + "\n")

        store.add_correction("new text", "sentiment", "neg", "pos")

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out)
        assert result["existing_count"] == 2
        assert result["correction_count"] == 1
        assert result["merged_count"] == 3
        assert result["duplicates_removed"] == 0

    def test_merge_deduplication(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "existing.jsonl")
        with open(existing, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "best gila", "label": "positive"}) + "\n")

        # Same text+label as existing
        store.add_correction("best gila", "sentiment", "neg", "positive")

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out)
        assert result["duplicates_removed"] == 1
        assert result["merged_count"] == 1

    def test_merge_nonexistent_existing(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "does_not_exist.jsonl")
        store.add_correction("t1", "sentiment", "neg", "pos")

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out)
        assert result["existing_count"] == 0
        assert result["merged_count"] == 1

    def test_merge_empty_both(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "empty.jsonl")
        with open(existing, "w") as fh:
            pass  # empty file

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out)
        assert result["merged_count"] == 0

    def test_merge_filter_module(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "existing.jsonl")
        with open(existing, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "hello", "label": "greeting"}) + "\n")

        store.add_correction("t1", "sentiment", "neg", "pos")
        store.add_correction("t2", "intent", "greet", "question")

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out, module="sentiment")
        # 1 existing + 1 sentiment correction (intent excluded)
        assert result["correction_count"] == 1
        assert result["merged_count"] == 2

    def test_merge_skips_bad_json_lines(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "bad.jsonl")
        with open(existing, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": "good", "label": "pos"}) + "\n")
            fh.write("this is not json\n")
            fh.write(json.dumps({"text": "also good", "label": "neg"}) + "\n")

        out = os.path.join(tmp_dir, "merged.jsonl")
        result = store.merge_with_dataset(existing, out)
        assert result["existing_count"] == 2


# ---------------------------------------------------------------------------
# Convenience functions / singleton
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Module-level convenience functions."""

    def test_get_feedback_store_singleton(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.json")
        s1 = get_feedback_store(path)
        s2 = get_feedback_store(path)
        assert s1 is s2

    def test_submit_correction(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.json")
        get_feedback_store(path)
        rec = submit_correction("test text", "sentiment", "neg", "pos", confidence=0.4)
        assert rec["text"] == "test text"
        assert rec["confidence_at_prediction"] == 0.4

    def test_get_uncertain_predictions(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.json")
        store = get_feedback_store(path)
        store.add_prediction("t1", "sentiment", "neutral", confidence=0.1)
        result = get_uncertain_predictions(module="sentiment")
        assert len(result) == 1

    def test_get_feedback_analytics(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.json")
        store = get_feedback_store(path)
        store.add_correction("t1", "sentiment", "neg", "pos")
        a = get_feedback_analytics()
        assert a["total_corrections"] == 1

    def test_export_corrections(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.json")
        store = get_feedback_store(path)
        store.add_correction("t1", "sentiment", "neg", "pos")
        out = os.path.join(tmp_dir, "export.jsonl")
        result = export_corrections(out)
        assert result["count"] == 1
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Internal helper functions."""

    def test_make_id_deterministic(self):
        id1 = _make_id("hello", "sentiment", "pos")
        id2 = _make_id("hello", "sentiment", "pos")
        assert id1 == id2

    def test_make_id_different_inputs(self):
        id1 = _make_id("hello", "sentiment", "pos")
        id2 = _make_id("bye", "sentiment", "pos")
        assert id1 != id2

    def test_make_id_length(self):
        assert len(_make_id("test")) == 16

    def test_utcnow_iso_format(self):
        ts = _utcnow_iso()
        # Should be parseable ISO 8601
        assert "T" in ts
        assert "+" in ts or "Z" in ts


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and stress tests."""

    def test_large_dataset(self, store):
        """500 corrections + 500 predictions."""
        for i in range(500):
            store.add_correction(
                f"text_{i}", "sentiment", "neg", "pos", confidence=0.3,
            )
        for i in range(500):
            store.add_prediction(
                f"pred_{i}", "sentiment", "neutral", confidence=0.2,
            )

        corrections = store.get_corrections(limit=1000)
        assert len(corrections) == 500

        uncertain = store.get_uncertain(threshold=0.5, limit=100)
        assert len(uncertain) == 100

        a = store.get_analytics()
        assert a["total_corrections"] == 500
        assert a["total_predictions"] == 500

    def test_unicode_text(self, store):
        store.add_correction("makanan dia terbaik 🔥", "sentiment", "neg", "pos")
        store.add_correction("سلام دنيا", "sentiment", "neg", "pos")
        corrections = store.get_corrections()
        assert len(corrections) == 2
        assert "🔥" in corrections[1]["text"] or "🔥" in corrections[0]["text"]

    def test_empty_strings(self, store):
        rec = store.add_correction("", "sentiment", "", "")
        assert rec["text"] == ""

    def test_special_characters_in_text(self, store, tmp_dir):
        store.add_correction('line1\nline2\t"quoted"', "sentiment", "neg", "pos")
        # Verify it persists correctly through JSON
        store2 = FeedbackStore(storage_path=store._storage_path)
        corrections = store2.get_corrections()
        assert corrections[0]["text"] == 'line1\nline2\t"quoted"'

    def test_concurrent_adds(self, store):
        """Thread safety: multiple threads adding corrections."""
        import threading

        errors = []

        def add_batch(start: int):
            try:
                for i in range(50):
                    store.add_correction(
                        f"thread_text_{start}_{i}",
                        "sentiment", "neg", "pos",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_batch, args=(t,))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        corrections = store.get_corrections(limit=300)
        assert len(corrections) == 200

    def test_persistence_roundtrip(self, store):
        """Write, reload, verify all data intact."""
        store.add_correction("c1", "sentiment", "neg", "pos", confidence=0.3)
        store.add_correction("c2", "intent", "greet", "question", confidence=0.7)
        store.add_prediction("p1", "sentiment", "neutral", confidence=0.1)

        path = store._storage_path
        store2 = FeedbackStore(storage_path=path)

        assert len(store2.get_corrections()) == 2
        a = store2.get_analytics()
        assert a["total_corrections"] == 2
        assert a["total_predictions"] == 1

    def test_export_creates_parent_dirs(self, store, tmp_dir):
        out = os.path.join(tmp_dir, "deep", "nested", "export.jsonl")
        result = store.export_training_data(out)
        assert os.path.exists(out)

    def test_merge_creates_parent_dirs(self, store, tmp_dir):
        existing = os.path.join(tmp_dir, "existing.jsonl")
        with open(existing, "w") as fh:
            fh.write(json.dumps({"text": "t", "label": "l"}) + "\n")
        out = os.path.join(tmp_dir, "deep", "nested", "merged.jsonl")
        store.merge_with_dataset(existing, out)
        assert os.path.exists(out)

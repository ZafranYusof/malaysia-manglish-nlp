"""Feedback loop system for human-in-the-loop model improvement.

Collects user corrections, tracks prediction confidence for active learning,
provides analytics on error patterns, and exports training-ready data.

Storage: lightweight JSON file, no database dependency.
Thread-safe via atomic write operations.

Usage::

    from malaysian_manglish_nlp.feedback import (
        submit_correction,
        get_uncertain_predictions,
        get_feedback_analytics,
        export_corrections,
    )

    # Record a correction
    submit_correction(
        text="makanan dia ok la",
        module="sentiment",
        original="negative",
        correction="neutral",
    )

    # Get low-confidence predictions for review
    uncertain = get_uncertain_predictions(module="sentiment", limit=5)

    # Export corrections as JSONL for retraining
    export_corrections("data/corrections.jsonl")

Zero extra dependencies (pure Python + json).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# FeedbackStore - core storage and logic
# ---------------------------------------------------------------------------

class FeedbackStore:
    """Persistent feedback store backed by a JSON file.

    Stores corrections and predictions for active learning, analytics,
    and training data export.  All file writes are atomic (write to
    temp file then rename) to avoid corruption under concurrent access.

    Args:
        storage_path: Path to the JSON storage file.  Parent directories
            are created automatically.

    Example::

        store = FeedbackStore("data/feedback_store.json")
        store.add_correction("best gila", "sentiment", "negative", "positive")
        analytics = store.get_analytics()
    """

    def __init__(self, storage_path: str = "data/feedback_store.json") -> None:
        self._storage_path = os.path.normpath(storage_path)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "corrections": [],
            "predictions": [],
            "metadata": {
                "created": _utcnow_iso(),
                "last_updated": _utcnow_iso(),
                "version": "1.0",
            },
        }
        self._load()

    # -- internal I/O -------------------------------------------------------

    def _load(self) -> None:
        """Load store from disk.  Creates file if missing."""
        if not os.path.exists(self._storage_path):
            self._ensure_dir()
            self._flush()
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            # Corrupted or unreadable: start fresh
            self._data = {
                "corrections": [],
                "predictions": [],
                "metadata": {
                    "created": _utcnow_iso(),
                    "last_updated": _utcnow_iso(),
                    "version": "1.0",
                },
            }
            self._flush()

    def _flush(self) -> None:
        """Atomic write: temp file then rename."""
        self._ensure_dir()
        self._data["metadata"]["last_updated"] = _utcnow_iso()
        dir_name = os.path.dirname(self._storage_path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="feedback_", dir=dir_name,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=2)
            # Atomic rename (POSIX).  On Windows, remove target first.
            if os.name == "nt" and os.path.exists(self._storage_path):
                os.remove(self._storage_path)
            os.replace(tmp_path, self._storage_path)
        except OSError:
            # Best-effort cleanup
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _ensure_dir(self) -> None:
        """Create parent directories if needed."""
        dir_name = os.path.dirname(self._storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    # -- corrections --------------------------------------------------------

    def add_correction(
        self,
        text: str,
        module: str,
        original: str,
        correction: str,
        confidence: float = 0.0,
        source: str = "api",
    ) -> dict:
        """Record a user correction.

        Args:
            text: The original input text.
            module: NLP module name (e.g. ``'sentiment'``, ``'intent'``).
            original: Model's original (wrong) prediction.
            correction: User-provided correct label.
            confidence: Model confidence at time of prediction (0-1).
            source: Where correction came from (``'api'``, ``'cli'``, ``'manual'``).

        Returns:
            dict: The stored correction record.
        """
        record = {
            "id": _make_id(text, module, original, correction),
            "text": text,
            "module": module,
            "original_prediction": original,
            "user_correction": correction,
            "confidence_at_prediction": float(confidence),
            "source": source,
            "timestamp": _utcnow_iso(),
        }
        with self._lock:
            self._data["corrections"].append(record)
            self._flush()
        return record

    def get_corrections(
        self, module: str = None, limit: int = None,
    ) -> list:
        """Get recorded corrections.

        Args:
            module: Filter by module name.  ``None`` returns all.
            limit: Max records to return.  ``None`` returns all.

        Returns:
            list[dict]: Correction records, newest first.
        """
        with self._lock:
            corrections = list(self._data.get("corrections", []))
        if module:
            corrections = [c for c in corrections if c.get("module") == module]
        corrections.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
        return corrections[:limit] if limit is not None else corrections

    # -- predictions (active learning) --------------------------------------

    def add_prediction(
        self,
        text: str,
        module: str,
        prediction: str,
        confidence: float,
    ) -> dict:
        """Record a prediction for active learning tracking.

        Args:
            text: Input text.
            module: NLP module name.
            prediction: Model's predicted label.
            confidence: Confidence score (0-1).  Lower = more uncertain.

        Returns:
            dict: The stored prediction record.
        """
        record = {
            "id": _make_id(text, module, prediction),
            "text": text,
            "module": module,
            "prediction": prediction,
            "confidence": float(confidence),
            "timestamp": _utcnow_iso(),
            "reviewed": False,
        }
        with self._lock:
            self._data["predictions"].append(record)
            self._flush()
        return record

    def get_uncertain(
        self,
        module: str = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list:
        """Get predictions with lowest confidence for human review.

        Uses uncertainty sampling: predictions near decision boundary
        (low confidence / score near 0) are prioritised.

        Args:
            module: Filter by module.  ``None`` returns all.
            limit: Max records to return.
            threshold: Only return predictions with confidence below this.

        Returns:
            list[dict]: Unreviewed prediction records, lowest confidence first.
        """
        with self._lock:
            preds = list(self._data.get("predictions", []))
        if module:
            preds = [p for p in preds if p.get("module") == module]
        # Only unreviewed
        preds = [p for p in preds if not p.get("reviewed", False)]
        # Below threshold
        preds = [p for p in preds if p.get("confidence", 1.0) < threshold]
        # Sort ascending by confidence (most uncertain first)
        preds.sort(key=lambda p: p.get("confidence", 1.0))
        return preds[:limit]

    # -- analytics ----------------------------------------------------------

    def get_analytics(self) -> dict:
        """Return correction statistics.

        Returns:
            dict with keys:
                - ``total_corrections``: int
                - ``total_predictions``: int
                - ``corrections_per_module``: dict[str, int]
                - ``predictions_per_module``: dict[str, int]
                - ``avg_confidence``: float (average prediction confidence)
                - ``correction_rate``: dict[str, float] (corrections / predictions per module)
                - ``sources``: dict[str, int] (corrections per source)
        """
        with self._lock:
            corrections = list(self._data.get("corrections", []))
            predictions = list(self._data.get("predictions", []))

        corr_per_mod: Counter = Counter()
        source_counts: Counter = Counter()
        for c in corrections:
            corr_per_mod[c.get("module", "unknown")] += 1
            source_counts[c.get("source", "unknown")] += 1

        pred_per_mod: Counter = Counter()
        confidences: List[float] = []
        for p in predictions:
            pred_per_mod[p.get("module", "unknown")] += 1
            confidences.append(p.get("confidence", 0.0))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Correction rate: corrections / predictions per module
        all_modules = set(corr_per_mod.keys()) | set(pred_per_mod.keys())
        correction_rate: Dict[str, float] = {}
        for mod in all_modules:
            p_count = pred_per_mod.get(mod, 0)
            c_count = corr_per_mod.get(mod, 0)
            correction_rate[mod] = c_count / p_count if p_count > 0 else float(c_count)

        return {
            "total_corrections": len(corrections),
            "total_predictions": len(predictions),
            "corrections_per_module": dict(corr_per_mod),
            "predictions_per_module": dict(pred_per_mod),
            "avg_confidence": round(avg_conf, 4),
            "correction_rate": correction_rate,
            "sources": dict(source_counts),
        }

    def get_error_patterns(
        self, module: str = None, limit: int = 10,
    ) -> list:
        """Identify common misprediction patterns.

        Groups corrections by (original_prediction -> user_correction) pair
        and returns the most frequent confusion patterns.

        Args:
            module: Filter by module.  ``None`` returns all.
            limit: Max patterns to return.

        Returns:
            list[dict]: Each dict has ``original``, ``correction``, ``count``,
            ``examples`` (list of texts), ``module``.
        """
        with self._lock:
            corrections = list(self._data.get("corrections", []))
        if module:
            corrections = [c for c in corrections if c.get("module") == module]

        pattern_map: Dict[tuple, Dict[str, Any]] = {}
        for c in corrections:
            key = (c.get("original_prediction", ""), c.get("user_correction", ""))
            if key not in pattern_map:
                pattern_map[key] = {
                    "original": key[0],
                    "correction": key[1],
                    "module": c.get("module", "unknown"),
                    "count": 0,
                    "examples": [],
                }
            pattern_map[key]["count"] += 1
            if len(pattern_map[key]["examples"]) < 5:
                pattern_map[key]["examples"].append(c.get("text", ""))

        patterns = sorted(
            pattern_map.values(), key=lambda p: p["count"], reverse=True,
        )
        return patterns[:limit]

    # -- export -------------------------------------------------------------

    def export_training_data(
        self,
        output_path: str,
        module: str = None,
        format: str = "jsonl",
    ) -> dict:
        """Export corrections as training-ready data.

        Args:
            output_path: Destination file path.
            module: Filter by module.  ``None`` exports all.
            format: Output format.  Currently only ``'jsonl'`` supported.

        Returns:
            dict: ``{'count': int, 'path': str, 'format': str}``.
        """
        corrections = self.get_corrections(module=module)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        seen_ids: set = set()
        count = 0
        with open(output_path, "w", encoding="utf-8") as fh:
            for c in corrections:
                rid = c.get("id", "")
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                record = {
                    "text": c.get("text", ""),
                    "label": c.get("user_correction", ""),
                    "module": c.get("module", ""),
                    "source": "correction",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        return {"count": count, "path": output_path, "format": format}

    def merge_with_dataset(
        self,
        existing_path: str,
        output_path: str,
        module: str = None,
    ) -> dict:
        """Merge corrections into existing JSONL training dataset with dedup.

        Reads existing JSONL file (one JSON object per line with ``text``
        and ``label`` fields), appends corrections, removes duplicates
        based on (text, label) pairs, writes merged output.

        Args:
            existing_path: Path to existing JSONL dataset.
            output_path: Destination for merged dataset.
            module: Filter corrections by module.  ``None`` merges all.

        Returns:
            dict: ``{'existing_count': int, 'correction_count': int,
                     'merged_count': int, 'duplicates_removed': int, 'path': str}``.
        """
        # Read existing dataset
        existing_records: List[Dict[str, Any]] = []
        if os.path.exists(existing_path):
            with open(existing_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            existing_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # Get corrections as training records
        corrections = self.get_corrections(module=module)
        correction_records = []
        for c in corrections:
            correction_records.append({
                "text": c.get("text", ""),
                "label": c.get("user_correction", ""),
                "module": c.get("module", ""),
                "source": "correction",
            })

        # Merge with dedup on (text, label)
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        total_input = len(existing_records) + len(correction_records)

        for record in existing_records + correction_records:
            key = (record.get("text", ""), record.get("label", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)

        duplicates_removed = total_input - len(merged)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for record in merged:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            "existing_count": len(existing_records),
            "correction_count": len(correction_records),
            "merged_count": len(merged),
            "duplicates_removed": duplicates_removed,
            "path": output_path,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _make_id(*parts: str) -> str:
    """Deterministic short ID from string parts."""
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Singleton & convenience functions
# ---------------------------------------------------------------------------

_feedback_store: Optional[FeedbackStore] = None
_store_lock = threading.Lock()


def get_feedback_store(storage_path: str = None) -> FeedbackStore:
    """Get or create singleton FeedbackStore.

    Args:
        storage_path: Override default storage path.  Only used on first
            call; subsequent calls return the cached instance.

    Returns:
        FeedbackStore: The singleton instance.
    """
    global _feedback_store
    with _store_lock:
        if _feedback_store is None:
            path = storage_path or "data/feedback_store.json"
            _feedback_store = FeedbackStore(storage_path=path)
        return _feedback_store


def _reset_store() -> None:
    """Reset singleton (for testing only)."""
    global _feedback_store
    with _store_lock:
        _feedback_store = None


def submit_correction(
    text: str,
    module: str,
    original: str,
    correction: str,
    **kwargs: Any,
) -> dict:
    """Quick-submit a correction to the singleton store.

    Args:
        text: Input text.
        module: NLP module name.
        original: Model's wrong prediction.
        correction: Correct label.
        **kwargs: Forwarded to ``FeedbackStore.add_correction``
            (``confidence``, ``source``).

    Returns:
        dict: Stored correction record.
    """
    store = get_feedback_store()
    return store.add_correction(
        text=text,
        module=module,
        original=original,
        correction=correction,
        **kwargs,
    )


def get_uncertain_predictions(
    module: str = None, limit: int = 10,
) -> list:
    """Get uncertain predictions from singleton store for review.

    Args:
        module: Filter by module.
        limit: Max results.

    Returns:
        list[dict]: Low-confidence unreviewed predictions.
    """
    store = get_feedback_store()
    return store.get_uncertain(module=module, limit=limit)


def get_feedback_analytics() -> dict:
    """Get analytics from singleton store.

    Returns:
        dict: Correction and prediction statistics.
    """
    store = get_feedback_store()
    return store.get_analytics()


def export_corrections(
    output_path: str = "data/corrections.jsonl",
    **kwargs: Any,
) -> dict:
    """Export corrections from singleton store for training.

    Args:
        output_path: Destination file path.
        **kwargs: Forwarded to ``FeedbackStore.export_training_data``
            (``module``, ``format``).

    Returns:
        dict: ``{'count': int, 'path': str, 'format': str}``.
    """
    store = get_feedback_store()
    return store.export_training_data(output_path=output_path, **kwargs)

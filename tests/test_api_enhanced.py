"""
Tests for enhanced REST API endpoints.

Covers:
- WebSocket streaming (/ws/analyze)
- Async batch (/batch/async, /batch/status, /batch/cancel)
- New NLP endpoints (/aspect-sentiment, /multi-emotion)
- Feedback endpoints (/feedback, /feedback/stats)
- Active learning (/active-learning/uncertain)
- Backward compatibility (existing endpoints still work)

Run:
    pytest tests/test_api_enhanced.py -v
"""

import json
import time
import asyncio
import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI test client not available", allow_module_level=True)

from malaysian_manglish_nlp.rest_api import (
    app,
    batch_store,
    _feedback_store,
    ws_manager,
    ConnectionManager,
    BatchJobStore,
    _resolve_modules,
    ALL_MODULES,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_stores():
    """Clean in-memory stores between tests."""
    _feedback_store.clear()
    batch_store._jobs.clear()
    yield
    _feedback_store.clear()
    batch_store._jobs.clear()


# --- Helper Tests ---

class TestHelpers:
    def test_resolve_modules_all(self):
        result = _resolve_modules(["all"])
        assert result == ALL_MODULES

    def test_resolve_modules_specific(self):
        result = _resolve_modules(["sentiment", "emotion"])
        assert result == ["sentiment", "emotion"]

    def test_resolve_modules_mixed(self):
        result = _resolve_modules(["all", "sentiment"])
        assert result == ALL_MODULES

    def test_all_modules_list(self):
        assert "sentiment" in ALL_MODULES
        assert "normalize" in ALL_MODULES
        assert "emotion" in ALL_MODULES
        assert len(ALL_MODULES) >= 10


# --- Connection Manager Tests ---

class TestConnectionManager:
    def test_init(self):
        mgr = ConnectionManager()
        assert mgr.active_connections == []

    def test_rate_limit_allows(self):
        mgr = ConnectionManager()

        class FakeWS:
            pass

        ws = FakeWS()
        # First 10 should be allowed
        for _ in range(10):
            assert mgr.is_rate_allowed(ws, max_msgs=10, window=1.0) is True
        # 11th should be denied
        assert mgr.is_rate_allowed(ws, max_msgs=10, window=1.0) is False

    def test_rate_limit_different_connections(self):
        mgr = ConnectionManager()

        class FakeWS:
            pass

        ws1 = FakeWS()
        ws2 = FakeWS()
        # Exhaust ws1
        for _ in range(10):
            mgr.is_rate_allowed(ws1, max_msgs=10)
        # ws2 should still work
        assert mgr.is_rate_allowed(ws2, max_msgs=10) is True


# --- Batch Job Store Tests ---

class TestBatchJobStore:
    def test_create_job(self):
        store = BatchJobStore()
        job_id = store.create_job(["hello", "world"], ["sentiment"])
        assert job_id
        job = store.get_job(job_id)
        assert job is not None
        assert job["status"] == "queued"
        assert job["total"] == 2
        assert job["completed"] == 0

    def test_update_progress(self):
        store = BatchJobStore()
        job_id = store.create_job(["hello"], ["sentiment"])
        store.update_progress(job_id, {"text": "hello", "sentiment": "positive"})
        job = store.get_job(job_id)
        assert job["completed"] == 1
        assert job["status"] == "completed"
        assert len(job["results"]) == 1

    def test_cancel_job(self):
        store = BatchJobStore()
        job_id = store.create_job(["hello"], ["sentiment"])
        assert store.cancel_job(job_id) is True
        job = store.get_job(job_id)
        assert job["status"] == "cancelled"

    def test_cancel_completed_job(self):
        store = BatchJobStore()
        job_id = store.create_job(["hello"], ["sentiment"])
        store.update_progress(job_id, {"text": "hello", "sentiment": "positive"})
        assert store.cancel_job(job_id) is False

    def test_get_nonexistent_job(self):
        store = BatchJobStore()
        assert store.get_job("nonexistent") is None

    def test_cleanup_old(self):
        store = BatchJobStore()
        job_id = store.create_job(["hello"], ["sentiment"])
        store._jobs[job_id]["finished_at"] = time.time() - 7200  # 2 hours ago
        removed = store.cleanup_old(max_age=3600)
        assert removed == 1
        assert store.get_job(job_id) is None


# --- Backward Compatibility Tests ---

class TestBackwardCompat:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "modules_loaded" in data

    def test_modules_list(self, client):
        resp = client.get("/modules")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = [m["name"] for m in data]
        assert "sentiment" in names
        assert "ner" in names

    def test_sentiment(self, client):
        resp = client.post("/sentiment", json={"text": "Best lah makan sini!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        assert "processing_time_ms" in data

    def test_analyze(self, client):
        resp = client.post("/analyze", json={"text": "Wah sedap gila!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        result = data["result"]
        assert "sentiment" in result
        assert "normalized" in result

    def test_batch_sync(self, client):
        resp = client.post("/batch", json={
            "texts": ["Best lah!", "Tak suka"],
            "modules": ["sentiment"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_all_modules_shortcut(self, client):
        resp = client.post("/batch", json={
            "texts": ["Test text"],
            "modules": ["all"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        # Should have many module results
        result = data["results"][0]
        assert "sentiment" in result
        assert "normalize" in result

    def test_batch_max_100(self, client):
        texts = [f"Text {i}" for i in range(100)]
        resp = client.post("/batch", json={"texts": texts, "modules": ["sentiment"]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 100

    def test_batch_over_100_rejected(self, client):
        texts = [f"Text {i}" for i in range(101)]
        resp = client.post("/batch", json={"texts": texts, "modules": ["sentiment"]})
        assert resp.status_code == 422  # Validation error


# --- Async Batch Tests ---

class TestAsyncBatch:
    def test_submit_async_batch(self, client):
        resp = client.post("/batch/async", json={
            "texts": ["Hello", "World"],
            "modules": ["sentiment"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["total"] == 2

    def test_check_batch_status(self, client):
        # Submit
        submit_resp = client.post("/batch/async", json={
            "texts": ["Hello"],
            "modules": ["sentiment"],
        })
        job_id = submit_resp.json()["job_id"]

        # Wait a moment for background processing
        time.sleep(0.5)

        # Check status
        resp = client.get(f"/batch/status/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "processing", "completed")
        assert "progress_pct" in data

    def test_cancel_batch(self, client):
        # Submit with many texts to have time to cancel
        texts = [f"Text {i}" for i in range(100)]
        submit_resp = client.post("/batch/async", json={
            "texts": texts,
            "modules": ["sentiment", "emotion", "ner"],
        })
        job_id = submit_resp.json()["job_id"]

        # Cancel immediately
        resp = client.get(f"/batch/cancel/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id

    def test_nonexistent_job(self, client):
        resp = client.get("/batch/status/nonexistent-id")
        assert resp.status_code == 404

    def test_cancel_nonexistent(self, client):
        resp = client.get("/batch/cancel/nonexistent-id")
        assert resp.status_code == 404


# --- WebSocket Tests ---

class TestWebSocket:
    def test_ws_basic_flow(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"text": "Best lah!", "modules": ["sentiment"]})

            # Should get module result
            msg1 = ws.receive_json()
            assert msg1["type"] in ("module_result", "module_error")

            # Should get completion
            msg2 = ws.receive_json()
            if msg1["type"] == "module_result":
                assert msg2["type"] in ("complete", "module_result")

    def test_ws_ping_pong(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"type": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_ws_multiple_modules(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({
                "text": "Wah sedap gila!",
                "modules": ["sentiment", "emotion", "language"],
            })

            messages = []
            for _ in range(10):  # Max 10 messages
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("type") == "complete":
                    break

            # Should have results + complete
            types = [m["type"] for m in messages]
            assert "complete" in types

    def test_ws_all_modules_shortcut(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"text": "Test", "modules": ["all"]})

            messages = []
            for _ in range(30):  # Many modules
                msg = ws.receive_json()
                messages.append(msg)
                if msg.get("type") == "complete":
                    break

            complete_msg = [m for m in messages if m["type"] == "complete"][0]
            assert complete_msg["total_modules"] > 5  # Should process many modules

    def test_ws_invalid_message(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"no_text_field": True})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_ws_empty_text(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"text": "", "modules": ["sentiment"]})
            resp = ws.receive_json()
            assert resp["type"] == "error"

    def test_ws_unknown_module(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"text": "Test", "modules": ["nonexistent_module"]})
            msg = ws.receive_json()
            assert msg["type"] == "module_error"


# --- New NLP Endpoint Tests ---

class TestNewNLPEndpoints:
    def test_aspect_sentiment(self, client):
        resp = client.post("/aspect-sentiment", json={
            "text": "Nasi lemak sedap tapi servis lambat",
            "domain": "food",
        })
        # May be 501 if module not implemented yet
        assert resp.status_code in (200, 501)
        if resp.status_code == 501:
            assert "not yet available" in resp.json()["detail"]

    def test_multi_emotion(self, client):
        resp = client.post("/multi-emotion", json={
            "text": "Aku marah gila tapi sedih jugak",
            "threshold": 0.2,
        })
        # May be 501 if module not implemented yet
        assert resp.status_code in (200, 501)
        if resp.status_code == 501:
            assert "not yet available" in resp.json()["detail"]

    def test_aspect_sentiment_default_domain(self, client):
        resp = client.post("/aspect-sentiment", json={
            "text": "Test text",
        })
        assert resp.status_code in (200, 501)

    def test_multi_emotion_default_threshold(self, client):
        resp = client.post("/multi-emotion", json={
            "text": "Test text",
        })
        assert resp.status_code in (200, 501)


# --- Feedback Endpoint Tests ---

class TestFeedback:
    def test_submit_feedback(self, client):
        resp = client.post("/feedback", json={
            "text": "Best lah!",
            "module": "sentiment",
            "prediction": "negative",
            "correction": "positive",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data

    def test_feedback_stats_empty(self, client):
        resp = client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_feedback"] == 0
        assert data["by_module"] == {}

    def test_feedback_stats_after_submissions(self, client):
        # Submit 2 sentiment feedbacks
        for _ in range(2):
            client.post("/feedback", json={
                "text": "Test",
                "module": "sentiment",
                "prediction": "pos",
                "correction": "neg",
            })
        # Submit 1 emotion feedback
        client.post("/feedback", json={
            "text": "Test",
            "module": "emotion",
            "prediction": "joy",
            "correction": "anger",
        })

        resp = client.get("/feedback/stats")
        data = resp.json()
        assert data["total_feedback"] == 3
        assert data["by_module"]["sentiment"] == 2
        assert data["by_module"]["emotion"] == 1

    def test_feedback_requires_fields(self, client):
        resp = client.post("/feedback", json={
            "text": "Test",
            "module": "sentiment",
            # Missing prediction and correction
        })
        assert resp.status_code == 422


# --- Active Learning Tests ---

class TestActiveLearning:
    def test_uncertain_empty(self, client):
        resp = client.get("/active-learning/uncertain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["uncertain"] == []
        assert data["total"] == 0

    def test_uncertain_with_params(self, client):
        resp = client.get("/active-learning/uncertain?limit=5&module=emotion")
        assert resp.status_code == 200
        data = resp.json()
        assert data["module"] == "emotion"

    def test_uncertain_default_params(self, client):
        resp = client.get("/active-learning/uncertain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["module"] == "sentiment"  # Default


# --- Module Info Endpoint Tests ---

class TestModuleInfo:
    def test_modules_include_new_endpoints(self, client):
        resp = client.get("/modules")
        data = resp.json()
        names = [m["name"] for m in data]
        assert "aspect_sentiment" in names
        assert "multi_emotion" in names


# --- Validation Tests ---

class TestValidation:
    def test_empty_text_rejected(self, client):
        resp = client.post("/sentiment", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_text_rejected(self, client):
        resp = client.post("/sentiment", json={})
        assert resp.status_code == 422

    def test_long_text_accepted(self, client):
        resp = client.post("/sentiment", json={"text": "Hello " * 1000})
        assert resp.status_code == 200

    def test_text_too_long_rejected(self, client):
        resp = client.post("/sentiment", json={"text": "x" * 10001})
        assert resp.status_code == 422

    def test_batch_empty_list_rejected(self, client):
        resp = client.post("/batch", json={"texts": [], "modules": ["sentiment"]})
        assert resp.status_code == 422


# --- Dialect Endpoint Test ---

class TestDialect:
    def test_dialect_endpoint(self, client):
        resp = client.post("/dialect", json={"text": "Ambo nak gi pasar"})
        # May be 501 if not available
        assert resp.status_code in (200, 501)


# --- Error Handling Tests ---

class TestErrorHandling:
    def test_unknown_module_in_batch(self, client):
        resp = client.post("/batch", json={
            "texts": ["Hello"],
            "modules": ["nonexistent"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data["results"][0]["nonexistent"]

"""
manglish-nlp REST API — FastAPI service for Malaysian Manglish NLP.

Run with:
    uvicorn malaysian_manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000

Or:
    python -m malaysian_manglish_nlp.rest_api
"""

from __future__ import annotations

import time
import uuid
import asyncio
import json
from typing import Any, Dict, List, Optional
from functools import wraps

try:
    from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    raise ImportError(
        "FastAPI is required for the REST API. Install with: pip install manglish-nlp[api]"
    )

try:
    import websockets  # noqa: F401
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False

import malaysian_manglish_nlp

# --- App Setup ---

app = FastAPI(
    title="manglish-nlp API",
    description="Full NLP toolkit for Malaysian Manglish — sentiment, NER, POS, translation, and more.",
    version=getattr(malaysian_manglish_nlp, '__version__', '3.0.0'),
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rate Limiting (simple in-memory) ---

class RateLimiter:
    """Simple in-memory rate limiter per IP."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return False

        self._requests[client_ip].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next: Any) -> Any:
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return Response(
            content='{"error": "Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )
    response = await call_next(request)
    return response

# --- WebSocket Connection Manager ---

class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self._rate_buckets: Dict[str, List[float]] = {}

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active_connections:
            self.active_connections.remove(ws)
        client_id = id(ws)
        self._rate_buckets.pop(str(client_id), None)

    def is_rate_allowed(self, ws: WebSocket, max_msgs: int = 10, window: float = 1.0) -> bool:
        """Per-connection rate limit: max_msgs per window seconds."""
        client_id = str(id(ws))
        now = time.time()
        if client_id not in self._rate_buckets:
            self._rate_buckets[client_id] = []
        bucket = self._rate_buckets[client_id]
        self._rate_buckets[client_id] = [t for t in bucket if now - t < window]
        if len(self._rate_buckets[client_id]) >= max_msgs:
            return False
        self._rate_buckets[client_id].append(now)
        return True

    async def broadcast(self, message: dict) -> None:
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)

ws_manager = ConnectionManager()

# --- Async Batch Job Storage ---

class BatchJobStore:
    """In-memory storage for async batch jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, texts: List[str], modules: List[str], options: Optional[Dict] = None) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "texts": texts,
            "modules": modules,
            "options": options,
            "total": len(texts),
            "completed": 0,
            "results": [],
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "cancelled": False,
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, result: Dict[str, Any]) -> None:
        job = self._jobs.get(job_id)
        if job:
            job["results"].append(result)
            job["completed"] += 1
            if job["completed"] >= job["total"]:
                job["status"] = "completed"
                job["finished_at"] = time.time()

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job["status"] in ("queued", "processing"):
            job["cancelled"] = True
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
            return True
        return False

    def cleanup_old(self, max_age: float = 3600) -> int:
        """Remove jobs older than max_age seconds. Returns count removed."""
        now = time.time()
        to_remove = [
            jid for jid, j in self._jobs.items()
            if j.get("finished_at") and now - j["finished_at"] > max_age
        ]
        for jid in to_remove:
            del self._jobs[jid]
        return len(to_remove)

batch_store = BatchJobStore()

# --- Request/Response Models ---

class TextRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "Wah sedap gila nasi lemak ni!", "options": None}})
    text: str = Field(..., description="Input text to process", min_length=1, max_length=10000)
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional processing options")

class TranslateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "I'm going to the mamak lah", "target": "en"}})
    text: str = Field(..., description="Input text to translate", min_length=1, max_length=10000)
    target: str = Field(default="en", description="Target language: en, bm, ms, formal")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional options")

# All known module names
ALL_MODULES = [
    "sentiment", "normalize", "ner", "pos", "translate",
    "emotion", "keywords", "language", "formalize", "summarize", "dialect",
]

class BatchRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"texts": ["Best lah!", "Tak suka", "Okay je"], "modules": ["sentiment", "emotion"]}})
    texts: List[str] = Field(..., description="List of texts to process", min_length=1, max_length=100)
    modules: List[str] = Field(
        default=["sentiment"],
        description='Modules to run: sentiment, normalize, ner, pos, translate, emotion, keywords, language, formalize, summarize, dialect. Use ["all"] for all modules.',
    )
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional options")

class BatchAsyncRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process", min_length=1, max_length=100)
    modules: List[str] = Field(
        default=["sentiment"],
        description='Modules to run. Use ["all"] for all modules.',
    )
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional options")

class BatchAsyncResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "status": "queued", "total": 50, "message": "Batch job submitted. Poll GET /batch/status/{job_id} for results."}})
    job_id: str
    status: str
    total: int
    message: str

class BatchStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    progress_pct: float
    results: List[Dict[str, Any]]
    processing_time_ms: Optional[float] = None

class NLPResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"result": {"label": "positive", "score": 0.95}, "processing_time_ms": 12.3}})
    result: Any
    processing_time_ms: float

class BatchResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time_ms: float
    count: int

class HealthResponse(BaseModel):
    status: str
    version: str
    modules_loaded: int

class ModuleInfo(BaseModel):
    name: str
    description: str
    endpoint: str

class AspectSentimentRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "Nasi lemak sedap tapi servis lambat", "domain": "food"}})
    text: str = Field(..., description="Input text", min_length=1, max_length=10000)
    domain: str = Field(default="general", description="Domain: general, food, tech, service")

class MultiEmotionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "Aku marah gila tapi sedih jugak", "threshold": 0.2}})
    text: str = Field(..., description="Input text", min_length=1, max_length=10000)
    threshold: float = Field(default=0.2, description="Min confidence threshold", ge=0.0, le=1.0)

class FeedbackRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"text": "Best lah!", "module": "sentiment", "prediction": "negative", "correction": "positive"}})
    text: str = Field(..., description="Original text", min_length=1, max_length=10000)
    module: str = Field(..., description="Module name: sentiment, emotion, etc.")
    prediction: str = Field(..., description="Model's prediction")
    correction: str = Field(..., description="Human correction")

# --- Helper ---

def process_with_timing(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a function and return result with timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms

def _resolve_modules(modules: List[str]) -> List[str]:
    """Expand 'all' shortcut to full module list."""
    if "all" in modules:
        return list(ALL_MODULES)
    return modules

def _get_module_func(name: str) -> Optional[Any]:
    """Get module function by name. Returns None if not found."""
    module_map = {
        "sentiment": malaysian_manglish_nlp.sentiment,
        "normalize": malaysian_manglish_nlp.normalize,
        "ner": malaysian_manglish_nlp.ner_tag,
        "pos": malaysian_manglish_nlp.pos_tag,
        "translate": malaysian_manglish_nlp.to_english,
        "emotion": malaysian_manglish_nlp.detect_emotion,
        "keywords": malaysian_manglish_nlp.extract_keywords,
        "language": malaysian_manglish_nlp.detect_language,
        "formalize": malaysian_manglish_nlp.formalize,
        "summarize": malaysian_manglish_nlp.summarize,
        "dialect": getattr(malaysian_manglish_nlp, "detect_dialect", None),
    }
    return module_map.get(name)

# --- Feedback Storage (in-memory) ---
_feedback_store: List[Dict[str, Any]] = []
_uncertain_store: Dict[str, List[Dict[str, Any]]] = {}  # module -> list of uncertain predictions

# --- System Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint.

    Returns service health, version, and module count.

    **Example response:**
    ```json
    {"status": "healthy", "version": "3.0.0", "modules_loaded": 51}
    ```
    """
    return HealthResponse(
        status="healthy",
        version=getattr(malaysian_manglish_nlp, '__version__', 'unknown'),
        modules_loaded=len([t for t in malaysian_manglish_nlp.available_tasks()]) if hasattr(malaysian_manglish_nlp, 'available_tasks') else 51,
    )

@app.get("/modules", response_model=List[ModuleInfo], tags=["System"])
async def list_modules() -> List[str]:
    """List available NLP modules and their endpoints.

    **Example response:**
    ```json
    [{"name": "sentiment", "description": "Sentiment analysis", "endpoint": "/sentiment"}]
    ```
    """
    modules = [
        ModuleInfo(name="analyze", description="Full analysis pipeline", endpoint="/analyze"),
        ModuleInfo(name="sentiment", description="Sentiment analysis (positive/negative/neutral)", endpoint="/sentiment"),
        ModuleInfo(name="normalize", description="Expand Manglish shortforms", endpoint="/normalize"),
        ModuleInfo(name="translate", description="Translate between BM/EN/Manglish", endpoint="/translate"),
        ModuleInfo(name="ner", description="Named Entity Recognition (9 types)", endpoint="/ner"),
        ModuleInfo(name="pos", description="Part-of-Speech tagging (15 tags)", endpoint="/pos"),
        ModuleInfo(name="summarize", description="Text summarization", endpoint="/summarize"),
        ModuleInfo(name="emotion", description="Emotion detection", endpoint="/emotion"),
        ModuleInfo(name="keywords", description="Keyword extraction", endpoint="/keywords"),
        ModuleInfo(name="language", description="Language detection (BM/EN/Manglish)", endpoint="/language"),
        ModuleInfo(name="formalize", description="Informal to formal BM", endpoint="/formalize"),
        ModuleInfo(name="dialect", description="Dialect detection and normalization", endpoint="/dialect"),
        ModuleInfo(name="aspect_sentiment", description="Aspect-based sentiment analysis", endpoint="/aspect-sentiment"),
        ModuleInfo(name="multi_emotion", description="Multi-label emotion detection", endpoint="/multi-emotion"),
    ]
    return modules

# --- NLP Endpoints ---

@app.post("/analyze", response_model=NLPResponse, tags=["NLP"])
async def analyze_text(req: TextRequest) -> Dict[str, Any]:
    """Run full analysis pipeline on text.

    Runs sentiment, normalization, language detection, POS, NER, emotion, and keywords in one call.

    **Example:**
    ```json
    // POST /analyze
    {"text": "Wah sedap gila nasi lemak ni!"}

    // Response
    {"result": {"sentiment": {"label": "positive", ...}, ...}, "processing_time_ms": 45.2}
    ```
    """
    start = time.perf_counter()

    result = {
        "normalized": malaysian_manglish_nlp.normalize(req.text),
        "sentiment": malaysian_manglish_nlp.sentiment(req.text),
        "language": malaysian_manglish_nlp.detect_language(req.text),
        "pos_tags": malaysian_manglish_nlp.pos_tag(req.text),
        "entities": malaysian_manglish_nlp.ner_tag(req.text),
        "emotion": malaysian_manglish_nlp.detect_emotion(req.text),
        "keywords": malaysian_manglish_nlp.extract_keywords(req.text),
    }

    elapsed_ms = (time.perf_counter() - start) * 1000
    return NLPResponse(result=result, processing_time_ms=round(elapsed_ms, 2))

@app.post("/sentiment", response_model=NLPResponse, tags=["NLP"])
async def sentiment_analysis(req: TextRequest) -> Dict[str, Any]:
    """Analyze sentiment of text.

    **Example:**
    ```json
    // POST /sentiment
    {"text": "Best lah makan sini!"}

    // Response
    {"result": {"label": "positive", "score": 0.92}, "processing_time_ms": 8.1}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.sentiment, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/normalize", response_model=NLPResponse, tags=["NLP"])
async def normalize_text(req: TextRequest) -> Dict[str, Any]:
    """Normalize Manglish shortforms.

    **Example:**
    ```json
    // POST /normalize
    {"text": "xpyh la, nnti kte g"}

    // Response
    {"result": "tak payah lah, nanti kita pergi", "processing_time_ms": 2.3}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.normalize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/translate", response_model=NLPResponse, tags=["NLP"])
async def translate_text(req: TranslateRequest) -> Dict[str, Any]:
    """Translate text to target language.

    **Example:**
    ```json
    // POST /translate
    {"text": "I'm going to the mamak lah", "target": "en"}

    // Response
    {"result": "I am going to the mamak stall", "processing_time_ms": 5.4}
    ```
    """
    target = req.target.lower()

    if target == 'en':
        result, ms = process_with_timing(malaysian_manglish_nlp.to_english, req.text)
    elif target in ('bm', 'ms', 'malay'):
        result, ms = process_with_timing(malaysian_manglish_nlp.to_malay, req.text)
    elif target == 'formal':
        result, ms = process_with_timing(malaysian_manglish_nlp.to_formal, req.text)
    else:
        result, ms = process_with_timing(malaysian_manglish_nlp.translate, req.text)

    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/ner", response_model=NLPResponse, tags=["NLP"])
async def named_entity_recognition(req: TextRequest) -> Dict[str, Any]:
    """Extract named entities from text.

    **Example:**
    ```json
    // POST /ner
    {"text": "Ali pergi KLCC dengan Siti"}

    // Response
    {"result": [{"text": "Ali", "label": "PERSON"}, {"text": "KLCC", "label": "LOCATION"}], "processing_time_ms": 6.7}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.ner_tag, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/pos", response_model=NLPResponse, tags=["NLP"])
async def pos_tagging(req: TextRequest) -> Dict[str, Any]:
    """Part-of-Speech tagging.

    **Example:**
    ```json
    // POST /pos
    {"text": "Aku nak makan nasi"}

    // Response
    {"result": [{"token": "Aku", "tag": "PRON"}, ...], "processing_time_ms": 4.2}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.pos_tag, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/summarize", response_model=NLPResponse, tags=["NLP"])
async def summarize_text(req: TextRequest) -> Dict[str, Any]:
    """Summarize text.

    **Example:**
    ```json
    // POST /summarize
    {"text": "Long paragraph about Malaysian politics..."}

    // Response
    {"result": "Key summary points...", "processing_time_ms": 120.5}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.summarize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/emotion", response_model=NLPResponse, tags=["NLP"])
async def detect_emotion(req: TextRequest) -> Dict[str, Any]:
    """Detect emotion in text.

    **Example:**
    ```json
    // POST /emotion
    {"text": "Aku happy gila hari ni!"}

    // Response
    {"result": {"label": "joy", "score": 0.88}, "processing_time_ms": 7.3}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.detect_emotion, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/keywords", response_model=NLPResponse, tags=["NLP"])
async def extract_keywords(req: TextRequest) -> Dict[str, Any]:
    """Extract keywords from text.

    **Example:**
    ```json
    // POST /keywords
    {"text": "Malaysia economy growth outlook 2025"}

    // Response
    {"result": ["malaysia", "economy", "growth", "outlook"], "processing_time_ms": 5.1}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.extract_keywords, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/language", response_model=NLPResponse, tags=["NLP"])
async def detect_language(req: TextRequest) -> Dict[str, Any]:
    """Detect language of text.

    **Example:**
    ```json
    // POST /language
    {"text": "Jom pergi mamak"}

    // Response
    {"result": {"language": "manglish", "confidence": 0.91}, "processing_time_ms": 3.8}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.detect_language, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/formalize", response_model=NLPResponse, tags=["NLP"])
async def formalize_text(req: TextRequest) -> Dict[str, Any]:
    """Convert informal text to formal BM.

    **Example:**
    ```json
    // POST /formalize
    {"text": "xpyh la, nnti kte g"}

    // Response
    {"result": "Tidak payahlah, nanti kita pergi", "processing_time_ms": 4.5}
    ```
    """
    result, ms = process_with_timing(malaysian_manglish_nlp.formalize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/dialect", response_model=NLPResponse, tags=["NLP"])
async def detect_dialect(req: TextRequest) -> Dict[str, Any]:
    """Detect dialect and normalize.

    **Example:**
    ```json
    // POST /dialect
    {"text": "Ambo nak gi pasar"}

    // Response
    {"result": {"dialect": "kelantan", "normalized": "Saya nak pergi pasar"}, "processing_time_ms": 5.0}
    ```
    """
    func = getattr(malaysian_manglish_nlp, "detect_dialect", None)
    if func is None:
        raise HTTPException(status_code=501, detail="Dialect detection not available in this version")
    result, ms = process_with_timing(func, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

# --- New NLP Endpoints ---

@app.post("/aspect-sentiment", response_model=NLPResponse, tags=["NLP"])
async def aspect_sentiment_analysis(req: AspectSentimentRequest) -> Dict[str, Any]:
    """Aspect-based sentiment analysis.

    Extracts sentiments for individual aspects within text (e.g., food quality vs service).

    **Example:**
    ```json
    // POST /aspect-sentiment
    {"text": "Nasi lemak sedap tapi servis lambat", "domain": "food"}

    // Response
    {"result": {"aspects": [{"aspect": "food", "sentiment": "positive"}, {"aspect": "service", "sentiment": "negative"}]}, "processing_time_ms": 15.2}
    ```
    """
    func = getattr(malaysian_manglish_nlp, "analyze_aspect_sentiment", None)
    if func is None:
        raise HTTPException(
            status_code=501,
            detail="Aspect-based sentiment analysis not yet available. Coming in a future update."
        )
    result, ms = process_with_timing(func, req.text, domain=req.domain)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

@app.post("/multi-emotion", response_model=NLPResponse, tags=["NLP"])
async def multi_emotion_detection(req: MultiEmotionRequest) -> Dict[str, Any]:
    """Multi-label emotion detection.

    Returns all emotions above the confidence threshold (text can express multiple emotions).

    **Example:**
    ```json
    // POST /multi-emotion
    {"text": "Aku marah gila tapi sedih jugak", "threshold": 0.2}

    // Response
    {"result": {"emotions": [{"label": "anger", "score": 0.78}, {"label": "sadness", "score": 0.45}]}, "processing_time_ms": 9.1}
    ```
    """
    func = getattr(malaysian_manglish_nlp, "detect_multi_emotion", None)
    if func is None:
        raise HTTPException(
            status_code=501,
            detail="Multi-label emotion detection not yet available. Coming in a future update."
        )
    result, ms = process_with_timing(func, req.text, threshold=req.threshold)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))

# --- Feedback Endpoints ---

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """Submit a prediction correction for model improvement.

    Stores feedback for active learning pipeline.

    **Example:**
    ```json
    // POST /feedback
    {"text": "Best lah!", "module": "sentiment", "prediction": "negative", "correction": "positive"}

    // Response
    {"status": "recorded", "feedback_id": "fb-001"}
    ```
    """
    feedback_entry = {
        "feedback_id": f"fb-{len(_feedback_store) + 1:04d}",
        "text": req.text,
        "module": req.module,
        "prediction": req.prediction,
        "correction": req.correction,
        "timestamp": time.time(),
    }
    _feedback_store.append(feedback_entry)
    return {"status": "recorded", "feedback_id": feedback_entry["feedback_id"]}

@app.get("/feedback/stats", tags=["Feedback"])
async def feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics.

    **Example response:**
    ```json
    {"total_feedback": 42, "by_module": {"sentiment": 30, "emotion": 12}}
    ```
    """
    by_module: Dict[str, int] = {}
    for fb in _feedback_store:
        mod = fb["module"]
        by_module[mod] = by_module.get(mod, 0) + 1
    return {
        "total_feedback": len(_feedback_store),
        "by_module": by_module,
    }

@app.get("/active-learning/uncertain", tags=["Feedback"])
async def get_uncertain_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Max results to return"),
    module: str = Query(default="sentiment", description="Module to query"),
) -> Dict[str, Any]:
    """Get texts where the model is least confident.

    Returns predictions with low confidence scores for human review.

    **Example:**
    ```
    GET /active-learning/uncertain?limit=5&module=sentiment
    ```
    ```json
    {"uncertain": [{"text": "...", "prediction": "...", "confidence": 0.51}], "total": 5}
    ```
    """
    entries = _uncertain_store.get(module, [])
    # Sort by confidence ascending (least confident first)
    sorted_entries = sorted(entries, key=lambda x: x.get("confidence", 1.0))
    limited = sorted_entries[:limit]
    return {"uncertain": limited, "total": len(limited), "module": module}

# --- Batch Endpoints ---

@app.post("/batch", response_model=BatchResponse, tags=["Batch"])
async def batch_process(req: BatchRequest) -> Dict[str, Any]:
    """Batch process multiple texts synchronously.

    Processes up to 100 texts with specified modules. Use `["all"]` for all modules.

    **Example:**
    ```json
    // POST /batch
    {"texts": ["Best lah!", "Tak suka"], "modules": ["sentiment", "emotion"]}

    // Response
    {"results": [{"text": "Best lah!", "sentiment": {...}, "emotion": {...}}, ...], "processing_time_ms": 25.0, "count": 2}
    ```
    """
    start = time.perf_counter()
    resolved_modules = _resolve_modules(req.modules)

    results = []
    for text in req.texts:
        text_result = {"text": text}
        for module_name in resolved_modules:
            func = _get_module_func(module_name)
            if func:
                try:
                    text_result[module_name] = func(text)
                except Exception as e:
                    text_result[module_name] = {"error": str(e)}
            else:
                text_result[module_name] = {"error": f"Unknown module: {module_name}"}
        results.append(text_result)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return BatchResponse(
        results=results,
        processing_time_ms=round(elapsed_ms, 2),
        count=len(results),
    )

@app.post("/batch/async", response_model=BatchAsyncResponse, tags=["Batch"])
async def batch_async_submit(req: BatchAsyncRequest) -> Dict[str, Any]:
    """Submit a batch job for async processing.

    Returns a job_id. Poll `GET /batch/status/{job_id}` for progress and results.

    **Example:**
    ```json
    // POST /batch/async
    {"texts": ["text1", "text2", ...], "modules": ["sentiment"]}

    // Response
    {"job_id": "uuid-here", "status": "queued", "total": 50, "message": "..."}
    ```
    """
    resolved_modules = _resolve_modules(req.modules)
    job_id = batch_store.create_job(req.texts, resolved_modules, req.options)

    # Schedule background processing
    asyncio.create_task(_run_batch_job(job_id))

    return BatchAsyncResponse(
        job_id=job_id,
        status="queued",
        total=len(req.texts),
        message="Batch job submitted. Poll GET /batch/status/{job_id} for results.",
    )

async def _run_batch_job(job_id: str) -> None:
    """Background task to process a batch job."""
    job = batch_store.get_job(job_id)
    if not job:
        return

    job["status"] = "processing"
    job["started_at"] = time.time()

    for text in job["texts"]:
        if job["cancelled"]:
            job["status"] = "cancelled"
            job["finished_at"] = time.time()
            return

        text_result = {"text": text}
        for module_name in job["modules"]:
            func = _get_module_func(module_name)
            if func:
                try:
                    text_result[module_name] = func(text)
                except Exception as e:
                    text_result[module_name] = {"error": str(e)}
            else:
                text_result[module_name] = {"error": f"Unknown module: {module_name}"}

        batch_store.update_progress(job_id, text_result)
        # Yield control to event loop periodically
        await asyncio.sleep(0)

    job = batch_store.get_job(job_id)
    if job and not job["cancelled"]:
        job["status"] = "completed"
        job["finished_at"] = time.time()


@app.get("/batch/status/{job_id}", response_model=BatchStatusResponse, tags=["Batch"])
async def batch_status(job_id: str) -> Dict[str, Any]:
    """Check async batch job progress and get results.

    **Example:**
    ```
    GET /batch/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
    ```
    ```json
    {"job_id": "...", "status": "processing", "total": 50, "completed": 25, "progress_pct": 50.0, "results": [...]}
    ```
    """
    job = batch_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    processing_time_ms = None
    if job["started_at"] and job["finished_at"]:
        processing_time_ms = round((job["finished_at"] - job["started_at"]) * 1000, 2)
    elif job["started_at"]:
        processing_time_ms = round((time.time() - job["started_at"]) * 1000, 2)

    progress_pct = round((job["completed"] / job["total"]) * 100, 1) if job["total"] > 0 else 0.0

    return BatchStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        total=job["total"],
        completed=job["completed"],
        progress_pct=progress_pct,
        results=job["results"],
        processing_time_ms=processing_time_ms,
    )


@app.get("/batch/cancel/{job_id}", tags=["Batch"])
async def batch_cancel(job_id: str) -> Dict[str, Any]:
    """Cancel a running async batch job.

    **Example:**
    ```
    GET /batch/cancel/a1b2c3d4-e5f6-7890-abcd-ef1234567890
    ```
    ```json
    {"job_id": "...", "status": "cancelled"}
    ```
    """
    job = batch_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if batch_store.cancel_job(job_id):
        return {"job_id": job_id, "status": "cancelled"}
    else:
        return {"job_id": job_id, "status": job["status"], "message": "Job cannot be cancelled (already completed or cancelled)"}


# --- WebSocket Endpoints ---

@app.websocket("/ws/analyze")
async def ws_analyze(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time streaming text analysis.

    **Protocol:**
    1. Client connects to `ws://host:port/ws/analyze`
    2. Client sends JSON: `{"text": "...", "modules": ["sentiment", "emotion", "ner"]}`
    3. Server streams back results per module as they complete
    4. Client can send `{"type": "ping"}` for keepalive
    5. Server responds with `{"type": "pong"}`

    **Rate limit:** 10 messages per second per connection.

    **Example flow:**
    ```
    >> {"text": "Best lah!", "modules": ["sentiment", "emotion"]}
    << {"type": "module_result", "module": "sentiment", "result": {...}, "processing_time_ms": 5.2}
    << {"type": "module_result", "module": "emotion", "result": {...}, "processing_time_ms": 3.1}
    << {"type": "complete", "text": "Best lah!", "total_modules": 2, "processing_time_ms": 8.3}
    ```
    """
    await ws_manager.connect(ws)
    try:
        while True:
            try:
                data = await ws.receive_json()
            except Exception:
                break

            # Handle ping/pong keepalive
            if isinstance(data, dict) and data.get("type") == "ping":
                await ws.send_json({"type": "pong"})
                continue

            # Rate limit check
            if not ws_manager.is_rate_allowed(ws):
                await ws.send_json({
                    "type": "error",
                    "error": "Rate limit exceeded. Max 10 messages/second.",
                })
                continue

            # Validate message
            text = data.get("text") if isinstance(data, dict) else None
            if not text or not isinstance(text, str):
                await ws.send_json({
                    "type": "error",
                    "error": "Missing or invalid 'text' field. Send {\"text\": \"...\", \"modules\": [...]}",
                })
                continue

            modules = data.get("modules", ["sentiment"]) if isinstance(data, dict) else ["sentiment"]
            if "all" in modules:
                modules = list(ALL_MODULES)

            # Process each module and stream results
            overall_start = time.perf_counter()
            processed = 0

            for module_name in modules:
                func = _get_module_func(module_name)
                if not func:
                    await ws.send_json({
                        "type": "module_error",
                        "module": module_name,
                        "error": f"Unknown module: {module_name}",
                    })
                    continue

                try:
                    result, ms = process_with_timing(func, text)
                    await ws.send_json({
                        "type": "module_result",
                        "module": module_name,
                        "result": result,
                        "processing_time_ms": round(ms, 2),
                    })
                    processed += 1
                except Exception as e:
                    await ws.send_json({
                        "type": "module_error",
                        "module": module_name,
                        "error": str(e),
                    })

            # Send completion marker
            overall_ms = (time.perf_counter() - overall_start) * 1000
            await ws.send_json({
                "type": "complete",
                "text": text[:100],  # Truncate for response
                "total_modules": processed,
                "processing_time_ms": round(overall_ms, 2),
            })

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.disconnect(ws)


# --- Error Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Any:
    return Response(
        content=f'{{"error": "{str(exc)}", "type": "{type(exc).__name__}"}}',
        status_code=500,
        media_type="application/json",
    )


# --- Run directly ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
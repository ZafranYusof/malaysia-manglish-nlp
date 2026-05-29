"""
manglish-nlp REST API — FastAPI service for Malaysian Manglish NLP.

Run with:
    uvicorn malaysian_manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000

Or:
    python -m malaysian_manglish_nlp.rest_api
"""

from __future__ import annotations

import time
import asyncio
from typing import Any, Dict, List, Optional
from functools import wraps

try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI is required for the REST API. Install with: pip install manglish-nlp[api]"
    )

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
        """Initialize the object.

        Args:
            max_requests: Max requests parameter.
            window_seconds: Window seconds parameter.

        Returns:
            Result value.

        """
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if allowed.

        Args:
            client_ip: Client ip parameter.

        Returns:
            Boolean result.

        """
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
    """Rate limit middleware.

    Args:
        request: Request parameter.
        call_next: Next middleware or handler.

    Returns:
        Result value.

    """
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return Response(
            content='{"error": "Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )
    response = await call_next(request)
    return response


# --- Request/Response Models ---

class TextRequest(BaseModel):
    text: str = Field(..., description="Input text to process", min_length=1, max_length=10000)
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional processing options")


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Input text to translate", min_length=1, max_length=10000)
    target: str = Field(default="en", description="Target language: en, bm, ms, formal")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional options")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to process", min_length=1, max_length=50)
    modules: List[str] = Field(
        default=["sentiment"],
        description="Modules to run: sentiment, normalize, ner, pos, translate, emotion, keywords",
    )
    options: Optional[Dict[str, Any]] = Field(default=None, description="Optional options")


class NLPResponse(BaseModel):
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


# --- Helper ---

def process_with_timing(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a function and return result with timing."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=getattr(malaysian_manglish_nlp, '__version__', 'unknown'),
        modules_loaded=len([t for t in malaysian_manglish_nlp.available_tasks()]) if hasattr(malaysian_manglish_nlp, 'available_tasks') else 51,
    )


@app.get("/modules", response_model=List[ModuleInfo], tags=["System"])
async def list_modules() -> List[str]:
    """List available NLP modules."""
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
    ]
    return modules


@app.post("/analyze", response_model=NLPResponse, tags=["NLP"])
async def analyze_text(req: TextRequest) -> Dict[str, Any]:
    """Run full analysis pipeline on text."""
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
    """Analyze sentiment of text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.sentiment, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/normalize", response_model=NLPResponse, tags=["NLP"])
async def normalize_text(req: TextRequest) -> Dict[str, Any]:
    """Normalize Manglish shortforms."""
    result, ms = process_with_timing(malaysian_manglish_nlp.normalize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/translate", response_model=NLPResponse, tags=["NLP"])
async def translate_text(req: TranslateRequest) -> Dict[str, Any]:
    """Translate text to target language."""
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
    """Extract named entities from text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.ner_tag, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/pos", response_model=NLPResponse, tags=["NLP"])
async def pos_tagging(req: TextRequest) -> Dict[str, Any]:
    """Part-of-Speech tagging."""
    result, ms = process_with_timing(malaysian_manglish_nlp.pos_tag, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/summarize", response_model=NLPResponse, tags=["NLP"])
async def summarize_text(req: TextRequest) -> Dict[str, Any]:
    """Summarize text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.summarize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/emotion", response_model=NLPResponse, tags=["NLP"])
async def detect_emotion(req: TextRequest) -> Dict[str, Any]:
    """Detect emotion in text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.detect_emotion, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/keywords", response_model=NLPResponse, tags=["NLP"])
async def extract_keywords(req: TextRequest) -> Dict[str, Any]:
    """Extract keywords from text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.extract_keywords, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/language", response_model=NLPResponse, tags=["NLP"])
async def detect_language(req: TextRequest) -> Dict[str, Any]:
    """Detect language of text."""
    result, ms = process_with_timing(malaysian_manglish_nlp.detect_language, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/formalize", response_model=NLPResponse, tags=["NLP"])
async def formalize_text(req: TextRequest) -> Dict[str, Any]:
    """Convert informal text to formal BM."""
    result, ms = process_with_timing(malaysian_manglish_nlp.formalize, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/dialect", response_model=NLPResponse, tags=["NLP"])
async def detect_dialect(req: TextRequest) -> Dict[str, Any]:
    """Detect dialect and normalize."""
    result, ms = process_with_timing(malaysian_manglish_nlp.detect_dialect, req.text)
    return NLPResponse(result=result, processing_time_ms=round(ms, 2))


@app.post("/batch", response_model=BatchResponse, tags=["NLP"])
async def batch_process(req: BatchRequest) -> Dict[str, Any]:
    """Batch process multiple texts."""
    start = time.perf_counter()

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
    }

    results = []
    for text in req.texts:
        text_result = {"text": text}
        for module_name in req.modules:
            func = module_map.get(module_name)
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


# --- Error Handlers ---

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> Dict[str, Any]:
    """Global exception handler.

    Args:
        request: Request parameter.
        exc: Exc parameter.

    Returns:
        Dictionary with results.

    """
    return Response(
        content=f'{{"error": "{str(exc)}", "type": "{type(exc).__name__}"}}',
        status_code=500,
        media_type="application/json",
    )


# --- Run directly ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

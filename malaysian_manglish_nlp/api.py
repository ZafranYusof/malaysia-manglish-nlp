"""
manglish-nlp REST API server.

Lightweight FastAPI wrapper for all manglish-nlp modules.
Run: python -m malaysian_manglish_nlp.api
Or:  uvicorn malaysian_manglish_nlp.api:app --port 8000

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import Dict, List, Optional
except ImportError:
    raise ImportError("REST API requires fastapi and uvicorn. Install: pip install fastapi uvicorn")

import malaysian_manglish_nlp
from malaysian_manglish_nlp.emotion import detect_emotion, emotion_summary
from malaysian_manglish_nlp.profanity import detect_profanity, censor, is_safe
from malaysian_manglish_nlp.dialect import detect_dialect, normalize_dialect, available_dialects
from malaysian_manglish_nlp.sarcasm import detect_sarcasm

app = FastAPI(
    title="manglish-nlp API",
    description="NLP toolkit for Malaysian Manglish text processing",
    version=malaysian_manglish_nlp.__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---

class TextRequest(BaseModel):
    text: str

class TextsRequest(BaseModel):
    texts: List[str]

class SimilarityRequest(BaseModel):
    text1: str
    text2: str
    method: Optional[str] = "semantic"

class CensorRequest(BaseModel):
    text: str
    level: Optional[int] = 1
    replacement: Optional[str] = "*"

class KeywordsRequest(BaseModel):
    text: str
    top_n: Optional[int] = 10
    method: Optional[str] = "frequency"

class CorrectRequest(BaseModel):
    text: str
    max_distance: Optional[int] = 1

class BatchRequest(BaseModel):
    texts: List[str]
    tasks: Optional[List[str]] = ["normalize", "sentiment", "lang"]


# --- Endpoints ---

@app.get("/")
def root() -> Dict[str, Any]:
    """Get API root information and available endpoints.

    Returns:
        Dictionary with results.

    """
    return {
        "name": "manglish-nlp",
        "version": malaysian_manglish_nlp.__version__,
        "modules": 22,
        "shortforms": 626,
        "endpoints": [
            "/normalize", "/sentiment", "/language", "/tokenize", "/stem",
            "/pos", "/ner", "/segment", "/formalize", "/clean",
            "/emotion", "/profanity", "/censor", "/dialect", "/sarcasm",
            "/keywords", "/similarity", "/correct", "/batch",
        ]
    }


@app.post("/normalize")
def api_normalize(req: TextRequest) -> Dict[str, Any]:
    """Normalize Malaysian text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"result": malaysian_manglish_nlp.normalize(req.text), "original": req.text}


@app.post("/sentiment")
def api_sentiment(req: TextRequest) -> Dict[str, Any]:
    """Analyze sentiment of text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return malaysian_manglish_nlp.sentiment(req.text)


@app.post("/language")
def api_language(req: TextRequest) -> Dict[str, Any]:
    """Detect language of text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return malaysian_manglish_nlp.detect_language(req.text)


@app.post("/tokenize")
def api_tokenize(req: TextRequest) -> Dict[str, Any]:
    """Tokenize text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"tokens": malaysian_manglish_nlp.tokenize(req.text)}


@app.post("/stem")
def api_stem(req: TextRequest) -> Dict[str, Any]:
    """Stem Malay words via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"result": malaysian_manglish_nlp.stem(req.text), "original": req.text}


@app.post("/pos")
def api_pos(req: TextRequest) -> Dict[str, Any]:
    """Tag parts of speech via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    result = malaysian_manglish_nlp.pos_tag(req.text)
    return {"tags": [{"word": w, "tag": t} for w, t in result]}


@app.post("/ner")
def api_ner(req: TextRequest) -> Dict[str, Any]:
    """Extract named entities via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"entities": malaysian_manglish_nlp.ner_tag(req.text)}


@app.post("/segment")
def api_segment(req: TextRequest) -> Dict[str, Any]:
    """Segment text into sentences via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return malaysian_manglish_nlp.segment(req.text)


@app.post("/formalize")
def api_formalize(req: TextRequest) -> Dict[str, Any]:
    """Formalize informal text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"result": malaysian_manglish_nlp.formalize(req.text), "original": req.text}


@app.post("/clean")
def api_clean(req: TextRequest) -> Dict[str, Any]:
    """Clean text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {
        "cleaned": malaysian_manglish_nlp.clean(req.text),
        "cleaned_nlp": malaysian_manglish_nlp.clean_for_nlp(req.text),
        "original": req.text,
    }


@app.post("/emotion")
def api_emotion(req: TextRequest) -> Dict[str, Any]:
    """Detect emotion in text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return detect_emotion(req.text)


@app.post("/profanity")
def api_profanity(req: TextRequest) -> Dict[str, Any]:
    """Detect profanity in text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return detect_profanity(req.text)


@app.post("/censor")
def api_censor(req: CensorRequest) -> Dict[str, Any]:
    """Censor profanity in text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"result": censor(req.text, req.replacement, req.level), "original": req.text}


@app.post("/dialect")
def api_dialect(req: TextRequest) -> Dict[str, Any]:
    """Detect Malay dialect via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    detection = detect_dialect(req.text)
    if detection['dialect'] != 'standard':
        norm = normalize_dialect(req.text)
        detection['normalized'] = norm['normalized']
    return detection


@app.post("/sarcasm")
def api_sarcasm(req: TextRequest) -> Dict[str, Any]:
    """Detect sarcasm in text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return detect_sarcasm(req.text)


@app.post("/keywords")
def api_keywords(req: KeywordsRequest) -> Dict[str, Any]:
    """Extract keywords from text via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return {"keywords": malaysian_manglish_nlp.extract_keywords(req.text, req.top_n, req.method)}


@app.post("/similarity")
def api_similarity(req: SimilarityRequest) -> float:
    """Compute text similarity via API.

    Args:
        req: Request object.

    Returns:
        Floating-point score or value.

    """
    if req.method == "semantic":
        return malaysian_manglish_nlp.similarity.semantic_similarity(req.text1, req.text2)
    elif req.method == "jaccard":
        return {"score": malaysian_manglish_nlp.similarity.jaccard(req.text1, req.text2)}
    elif req.method == "cosine":
        return {"score": malaysian_manglish_nlp.similarity.cosine(req.text1, req.text2)}
    else:
        return malaysian_manglish_nlp.similarity.semantic_similarity(req.text1, req.text2)


@app.post("/correct")
def api_correct(req: CorrectRequest) -> Dict[str, Any]:
    """Correct spelling via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return malaysian_manglish_nlp.correct(req.text, req.max_distance)


@app.post("/normalize_all")
def api_normalize_all(req: TextRequest) -> Dict[str, Any]:
    """Apply all normalizations via API.

    Args:
        req: Request object.

    Returns:
        Dictionary with results.

    """
    return malaysian_manglish_nlp.normalize_all(req.text)


@app.post("/batch")
def api_batch(req: BatchRequest) -> List[Dict[str, Any]]:
    """Process multiple texts in batch via API.

    Args:
        req: Request object.

    Returns:
        List of results.

    """
    results = []
    for text in req.texts:
        item = {"original": text}
        if "normalize" in req.tasks:
            item["normalized"] = malaysian_manglish_nlp.normalize(text)
        if "sentiment" in req.tasks:
            item["sentiment"] = malaysian_manglish_nlp.sentiment(text)
        if "lang" in req.tasks:
            item["language"] = malaysian_manglish_nlp.detect_language(text)
        if "emotion" in req.tasks:
            item["emotion"] = detect_emotion(text)
        if "profanity" in req.tasks:
            item["profanity"] = detect_profanity(text)
        if "dialect" in req.tasks:
            item["dialect"] = detect_dialect(text)
        results.append(item)
    return {"results": results, "count": len(results)}


@app.get("/dialects")
def api_dialects() -> Dict[str, Any]:
    """List available Malay dialects.

    Returns:
        Dictionary with results.

    """
    return {"dialects": available_dialects()}


@app.get("/health")
def api_health() -> Dict[str, Any]:
    """Check API health status.

    Returns:
        Dictionary with results.

    """
    return {"status": "ok", "version": malaysian_manglish_nlp.__version__}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

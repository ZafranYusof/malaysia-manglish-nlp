"""
Load and run inference with the fine-tuned Manglish multi-task model.

Provides sentiment, emotion, and intent predictions for Manglish text.
Falls back to rule-based prediction if model weights are not available.

Usage:
    from malaysian_manglish_nlp.transformers.manglish_model import load_model, predict

    model = load_model()
    result = predict("weh best gila makanan dia")
    # {'sentiment': {'label': 'positive', 'confidence': 0.92}, ...}

    # If no fine-tuned weights exist:
    result = demo_predict("teruk la service dia")
    # Uses rule-based fallback from existing modules
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os
from pathlib import Path

# Default model directory (relative to package root)
_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = str(_PACKAGE_DIR / 'resources' / 'manglish_finetuned')

# Label definitions (must match finetune.py)
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'love', 'neutral']
INTENT_LABELS = ['question', 'statement', 'request', 'complaint', 'greeting', 'opinion']


def _check_torch() -> bool:
    """Check if torch/transformers are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def load_model(model_dir: Optional[str] = None) -> Any:
    """Load the fine-tuned multi-task model for inference.
    
    Args:
        model_dir: Path to saved model directory.
            Default: resources/manglish_finetuned/
    
    Returns:
        dict: Model bundle with 'model', 'tokenizer', 'config', 'device'.
    
    Raises:
        ImportError: If torch/transformers not installed.
        FileNotFoundError: If model weights not found.
    """
    if not _check_torch():
        raise ImportError(
            "torch and transformers required for model inference. "
            "Install: pip install torch transformers"
        )
    
    import torch
    from transformers import AutoTokenizer
    from malaysian_manglish_nlp.transformers.finetune import ManglishMultiTaskModel
    
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_path = os.path.join(model_dir, 'model.pt')
    config_path = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}\n"
            f"Train first: python -m malaysian_manglish_nlp.transformers.finetune\n"
            f"Or use demo_predict() for rule-based fallback."
        )
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder_name = config.get('model_name', 'distilbert-base-multilingual-cased')
    
    model = ManglishMultiTaskModel(encoder_name)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'device': device,
    }


# Module-level cache for loaded model
_cached_model = None


def _get_model(model_dir: Optional[str] = None) -> Any:
    """Get or load cached model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model(model_dir)
    return _cached_model


def predict(text: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Predict sentiment, emotion, and intent for a single text.
    
    Args:
        text: Input Manglish text.
        model_dir: Optional model directory override.
    
    Returns:
        dict: Predictions with confidence scores.
            {
                'sentiment': {'label': str, 'confidence': float},
                'emotion': {'label': str, 'confidence': float},
                'intent': {'label': str, 'confidence': float},
            }
    """
    import torch
    import torch.nn.functional as F
    
    bundle = _get_model(model_dir)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    device = bundle['device']
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Convert logits to predictions with confidence
    results = {}
    
    label_lists = {
        'sentiment': SENTIMENT_LABELS,
        'emotion': EMOTION_LABELS,
        'intent': INTENT_LABELS,
    }
    
    for task, labels in label_lists.items():
        probs = F.softmax(logits[task], dim=1).squeeze(0)
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
        
        results[task] = {
            'label': labels[top_idx],
            'confidence': round(confidence, 4),
        }
    
    return results


def predict_batch(texts: List[str], model_dir: Optional[str] = None, batch_size: int = 32) -> List[Dict[str, Any]]:
    """Batch prediction for multiple texts.
    
    Args:
        texts: Input texts.
        model_dir: Optional model directory override.
        batch_size: Batch size for inference.
    
    Returns:
        list[dict]: List of prediction dicts (same format as predict()).
    """
    import torch
    import torch.nn.functional as F
    
    bundle = _get_model(model_dir)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    device = bundle['device']
    
    all_results = []
    
    label_lists = {
        'sentiment': SENTIMENT_LABELS,
        'emotion': EMOTION_LABELS,
        'intent': INTENT_LABELS,
    }
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        encoding = tokenizer(
            batch_texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        
        # Process each sample in batch
        for j in range(len(batch_texts)):
            result = {}
            for task, labels in label_lists.items():
                probs = F.softmax(logits[task][j], dim=0)
                top_idx = probs.argmax().item()
                confidence = probs[top_idx].item()
                result[task] = {
                    'label': labels[top_idx],
                    'confidence': round(confidence, 4),
                }
            all_results.append(result)
    
    return all_results


def demo_predict(text: str) -> Dict[str, Any]:
    """Rule-based fallback prediction when model weights are not available.
    
    Uses existing malaysian_manglish_nlp sentiment and emotion modules for a
    best-effort prediction without requiring fine-tuned weights.
    
    Args:
        text: Input Manglish text.
    
    Returns:
        dict: Predictions (same format as predict(), but rule-based).
    """
    result = {
        'sentiment': {'label': 'neutral', 'confidence': 0.5},
        'emotion': {'label': 'neutral', 'confidence': 0.5},
        'intent': {'label': 'statement', 'confidence': 0.5},
        '_fallback': True,
    }
    
    # Try using existing sentiment module
    try:
        from malaysian_manglish_nlp.sentiment import analyze as sentiment_analyze
        sent_result = sentiment_analyze(text)
        if isinstance(sent_result, dict):
            label = sent_result.get('sentiment', sent_result.get('label', 'neutral'))
            score = sent_result.get('score', sent_result.get('confidence', 0.5))
            if label in SENTIMENT_LABELS:
                result['sentiment'] = {'label': label, 'confidence': round(float(score), 4)}
    except: # Fallback: simple keyword-based sentiment
        result['sentiment'] = _keyword_sentiment(text)
    
    # Try using existing emotion module
    try:
        from malaysian_manglish_nlp.emotion import detect as emotion_detect
        emo_result = emotion_detect(text)
        if isinstance(emo_result, dict):
            label = emo_result.get('emotion', emo_result.get('label', 'neutral'))
            score = emo_result.get('score', emo_result.get('confidence', 0.5))
            if label in EMOTION_LABELS:
                result['emotion'] = {'label': label, 'confidence': round(float(score), 4)}
    except: # Fallback: simple keyword-based emotion
        result['emotion'] = _keyword_emotion(text)
    
    # Intent detection (rule-based)
    result['intent'] = _keyword_intent(text)
    
    return result


def _keyword_sentiment(text: str) -> Dict[str, Any]:
    """Simple keyword-based sentiment detection."""
    text_lower = text.lower()
    
    positive_words = [
        'best', 'gila best', 'terbaik', 'bagus', 'cantik', 'sedap', 'awesome',
        'nice', 'good', 'great', 'love', 'suka', 'mantap', 'power', 'syok',
        'happy', 'gembira', 'puas', 'satisfied', 'recommend',
    ]
    negative_words = [
        'teruk', 'bodoh', 'stupid', 'bad', 'worst', 'hate', 'benci', 'marah',
        'annoying', 'boring', 'sucks', 'terrible', 'horrible', 'sampah',
        'rubbish', 'disappointed', 'kecewa', 'fail', 'hancur',
    ]
    
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    
    if pos_count > neg_count:
        confidence = min(0.5 + pos_count * 0.15, 0.9)
        return {'label': 'positive', 'confidence': round(confidence, 4)}
    elif neg_count > pos_count:
        confidence = min(0.5 + neg_count * 0.15, 0.9)
        return {'label': 'negative', 'confidence': round(confidence, 4)}
    
    return {'label': 'neutral', 'confidence': 0.5}


def _keyword_emotion(text: str) -> Dict[str, Any]:
    """Simple keyword-based emotion detection."""
    text_lower = text.lower()
    
    emotion_keywords = {
        'happy': ['happy', 'gembira', 'seronok', 'syok', 'best', 'yeay', 'yay', 'haha'],
        'sad': ['sad', 'sedih', 'kesian', 'malang', 'cry', 'nangis', 'down'],
        'angry': ['angry', 'marah', 'geram', 'bengang', 'pissed', 'wtf', 'bodoh'],
        'fear': ['takut', 'scared', 'fear', 'nervous', 'cuak', 'gerun', 'seram'],
        'surprise': ['surprise', 'terkejut', 'wow', 'omg', 'gila', 'eh', 'wah'],
        'disgust': ['disgusting', 'geli', 'eww', 'yucks', 'menyampah', 'jijik'],
        'love': ['love', 'sayang', 'cinta', 'rindu', 'miss', 'dear', 'heart'],
    }
    
    scores = {}
    for emotion, keywords in emotion_keywords.items():
        count = sum(1 for k in keywords if k in text_lower)
        if count > 0:
            scores[emotion] = count
    
    if scores:
        top_emotion = max(scores, key=scores.get)
        confidence = min(0.5 + scores[top_emotion] * 0.15, 0.85)
        return {'label': top_emotion, 'confidence': round(confidence, 4)}
    
    return {'label': 'neutral', 'confidence': 0.5}


def _keyword_intent(text: str) -> Dict[str, Any]:
    """Simple rule-based intent detection."""
    text_lower = text.strip().lower()
    
    # Question markers
    question_markers = ['?', 'apa', 'kenapa', 'macam mana', 'bila', 'siapa',
                        'mana', 'berapa', 'what', 'why', 'how', 'when', 'where',
                        'who', 'which', 'ke?', 'tak?', 'kan?']
    
    # Greeting markers
    greeting_markers = ['hi', 'hello', 'hey', 'assalamualaikum', 'salam',
                        'morning', 'evening', 'weh', 'yo', 'sup']
    
    # Request markers
    request_markers = ['tolong', 'please', 'boleh', 'can you', 'help',
                       'minta', 'nak', 'want', 'need', 'perlukan']
    
    # Complaint markers
    complaint_markers = ['complaint', 'aduan', 'teruk', 'tak puas',
                         'disappointed', 'kecewa', 'unacceptable', 'worst']
    
    # Check question first (highest priority)
    if text_lower.endswith('?') or any(m in text_lower for m in question_markers[:10]):
        return {'label': 'question', 'confidence': 0.75}
    
    # Check greetings (short messages)
    words = text_lower.split()
    if len(words) <= 3 and any(m in text_lower for m in greeting_markers):
        return {'label': 'greeting', 'confidence': 0.8}
    
    # Check requests
    if any(m in text_lower for m in request_markers):
        return {'label': 'request', 'confidence': 0.7}
    
    # Check complaints
    if any(m in text_lower for m in complaint_markers):
        return {'label': 'complaint', 'confidence': 0.65}
    
    # Check opinion markers
    opinion_markers = ['i think', 'aku rasa', 'pada aku', 'in my opinion',
                       'honestly', 'tbh', 'imo', 'personally']
    if any(m in text_lower for m in opinion_markers):
        return {'label': 'opinion', 'confidence': 0.65}
    
    # Default: statement
    return {'label': 'statement', 'confidence': 0.5}

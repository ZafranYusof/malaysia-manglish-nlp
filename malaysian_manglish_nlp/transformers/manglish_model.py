"""
Load and run inference with the fine-tuned Manglish multi-task model.

Provides sentiment, emotion, and intent predictions for Manglish text.
Supports v1 (distilbert) and v2 (xlm-roberta) models.
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
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = str(_PACKAGE_DIR / 'resources' / 'manglish_finetuned_v2')
DEFAULT_MODEL_DIR_V1 = str(_PACKAGE_DIR / 'resources' / 'manglish_finetuned')

# Label definitions (must match finetune.py)
SENTIMENT_LABELS = ['positive', 'negative', 'neutral']
EMOTION_LABELS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'love', 'neutral']
INTENT_LABELS = ['question', 'statement', 'request', 'complaint', 'greeting', 'opinion']

# Ensemble settings
ENSEMBLE_ENABLED = True
CONFIDENCE_THRESHOLD = 0.60


def _check_torch() -> bool:
    """Check if torch/transformers are available."""
    try:
        import torch
        import transformers
        return True
    except ImportError:
        return False


def _download_model_from_hf(model_dir: str) -> bool:
    """Download fine-tuned model from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[manglish_model] huggingface_hub not installed. Cannot auto-download.")
        return False
    
    repo_id = "vexccz/manglish-nlp-sentiment"
    files = ["model.pt", "config.json", "tokenizer.json", "tokenizer_config.json"]
    
    os.makedirs(model_dir, exist_ok=True)
    print(f"[manglish_model] Downloading model from HuggingFace ({repo_id})...")
    
    for fname in files:
        dest = os.path.join(model_dir, fname)
        if os.path.exists(dest):
            continue
        try:
            print(f"  Downloading {fname}...")
            hf_path = hf_hub_download(repo_id=repo_id, filename=fname)
            import shutil
            shutil.copy2(hf_path, dest)
            print(f"  Done: {fname}")
        except Exception as e:
            print(f"  Failed to download {fname}: {e}")
            return False
    
    return True


def load_model(model_dir: Optional[str] = None) -> Any:
    """Load the fine-tuned multi-task model for inference.
    
    Supports both v1 (distilbert) and v2 (xlm-roberta) architectures.
    
    Args:
        model_dir: Path to saved model directory.
    
    Returns:
        dict: Model bundle with 'model', 'tokenizer', 'config', 'device', 'version'.
    """
    if not _check_torch():
        raise ImportError(
            "torch and transformers required for model inference. "
            "Install: pip install torch transformers"
        )
    
    import torch
    from transformers import AutoTokenizer
    
    # Try v2 first, then v1
    if model_dir is None:
        if os.path.exists(os.path.join(DEFAULT_MODEL_DIR, 'model.pt')):
            model_dir = DEFAULT_MODEL_DIR
        elif os.path.exists(os.path.join(DEFAULT_MODEL_DIR_V1, 'model.pt')):
            model_dir = DEFAULT_MODEL_DIR_V1
        else:
            model_dir = DEFAULT_MODEL_DIR
    
    model_path = os.path.join(model_dir, 'model.pt')
    config_path = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(model_path):
        if _download_model_from_hf(model_dir):
            pass
        else:
            raise FileNotFoundError(
                f"Model weights not found at: {model_path}\n"
                f"Options:\n"
                f"  1. pip install huggingface_hub\n"
                f"  2. Train: python -m malaysian_manglish_nlp.transformers.finetune_v2\n"
                f"  3. Use demo_predict() for rule-based fallback."
            )
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    version = config.get('version', 'v1')
    encoder_name = config.get('model_name', 'distilbert-base-multilingual-cased')
    
    # Load appropriate model class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if version == 'v2':
        from malaysian_manglish_nlp.transformers.finetune_v2 import ManglishMultiTaskModelV2
        model = ManglishMultiTaskModelV2(encoder_name)
    else:
        from malaysian_manglish_nlp.transformers.finetune import ManglishMultiTaskModel
        model = ManglishMultiTaskModel(encoder_name)
    
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'device': device,
        'version': version,
    }


# Module-level cache
_cached_model = None

def _get_model(model_dir: Optional[str] = None) -> Any:
    """Get or load cached model."""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model(model_dir)
    return _cached_model


def predict(text: str, model_dir: Optional[str] = None,
            use_ensemble: bool = True) -> Dict[str, Any]:
    """Predict sentiment, emotion, and intent for a single text.
    
    Args:
        text: Input Manglish text.
        model_dir: Optional model directory override.
        use_ensemble: If True, applies ensemble with rule-based fallback.
    
    Returns:
        dict: Predictions with confidence scores.
    """
    import torch
    import torch.nn.functional as F
    
    bundle = _get_model(model_dir)
    model = bundle['model']
    tokenizer = bundle['tokenizer']
    device = bundle['device']
    
    # Tokenize (use raw text - minimal preprocessing)
    import re
    clean_text = re.sub(r'\s+', ' ', text.strip())
    
    encoding = tokenizer(
        clean_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
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
    
    # Apply ensemble if enabled
    if use_ensemble and ENSEMBLE_ENABLED:
        try:
            from malaysian_manglish_nlp.transformers.ensemble import ensemble_predict
            results = ensemble_predict(text, results, threshold=CONFIDENCE_THRESHOLD)
        except ImportError:
            pass
    
    return results


def predict_batch(texts: List[str], model_dir: Optional[str] = None,
                  batch_size: int = 32, use_ensemble: bool = True) -> List[Dict[str, Any]]:
    """Batch prediction for multiple texts."""
    import torch
    import torch.nn.functional as F
    import re
    
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
    
    for i in range(0, len(texts), batch_size):
        batch_texts = [re.sub(r'\s+', ' ', t.strip()) for t in texts[i:i + batch_size]]
        
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
    
    # Apply ensemble if enabled
    if use_ensemble and ENSEMBLE_ENABLED:
        try:
            from malaysian_manglish_nlp.transformers.ensemble import ensemble_predict_batch
            all_results = ensemble_predict_batch(texts, all_results, threshold=CONFIDENCE_THRESHOLD)
        except ImportError:
            pass
    
    return all_results


def demo_predict(text: str) -> Dict[str, Any]:
    """Rule-based fallback prediction when model weights are not available."""
    result = {
        'sentiment': {'label': 'neutral', 'confidence': 0.5},
        'emotion': {'label': 'neutral', 'confidence': 0.5},
        'intent': {'label': 'statement', 'confidence': 0.5},
        '_fallback': True,
    }
    
    try:
        from malaysian_manglish_nlp.sentiment import analyze as sentiment_analyze
        sent_result = sentiment_analyze(text)
        if isinstance(sent_result, dict):
            label = sent_result.get('sentiment', sent_result.get('label', 'neutral'))
            score = sent_result.get('score', sent_result.get('confidence', 0.5))
            if label in SENTIMENT_LABELS:
                result['sentiment'] = {'label': label, 'confidence': round(float(score), 4)}
    except:
        result['sentiment'] = _keyword_sentiment(text)
    
    try:
        from malaysian_manglish_nlp.emotion import detect as emotion_detect
        emo_result = emotion_detect(text)
        if isinstance(emo_result, dict):
            label = emo_result.get('emotion', emo_result.get('label', 'neutral'))
            score = emo_result.get('score', emo_result.get('confidence', 0.5))
            if label in EMOTION_LABELS:
                result['emotion'] = {'label': label, 'confidence': round(float(score), 4)}
    except:
        result['emotion'] = _keyword_emotion(text)
    
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
    
    question_markers = ['?', 'apa', 'kenapa', 'macam mana', 'bila', 'siapa',
                        'mana', 'berapa', 'what', 'why', 'how', 'when', 'where',
                        'who', 'which', 'ke?', 'tak?', 'kan?']
    
    greeting_markers = ['hi', 'hello', 'hey', 'assalamualaikum', 'salam',
                        'morning', 'evening', 'weh', 'yo', 'sup']
    
    request_markers = ['tolong', 'please', 'boleh', 'can you', 'help',
                       'minta', 'nak', 'want', 'need', 'perlukan']
    
    complaint_markers = ['complaint', 'aduan', 'teruk', 'tak puas',
                         'disappointed', 'kecewa', 'unacceptable', 'worst']
    
    if text_lower.endswith('?') or any(m in text_lower for m in question_markers[:10]):
        return {'label': 'question', 'confidence': 0.75}
    
    words = text_lower.split()
    if len(words) <= 3 and any(m in text_lower for m in greeting_markers):
        return {'label': 'greeting', 'confidence': 0.8}
    
    if any(m in text_lower for m in request_markers):
        return {'label': 'request', 'confidence': 0.7}
    
    if any(m in text_lower for m in complaint_markers):
        return {'label': 'complaint', 'confidence': 0.65}
    
    opinion_markers = ['i think', 'aku rasa', 'pada aku', 'in my opinion',
                       'honestly', 'tbh', 'imo', 'personally']
    if any(m in text_lower for m in opinion_markers):
        return {'label': 'opinion', 'confidence': 0.65}
    
    return {'label': 'statement', 'confidence': 0.5}

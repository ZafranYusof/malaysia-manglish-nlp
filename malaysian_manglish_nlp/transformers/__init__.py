"""
Transformer-based NLP models for Malaysian text.

Optional module — requires `torch` and `transformers` packages.
Install: pip install torch transformers

Usage:
    from malaysian_manglish_nlp.transformers import (
        sentiment_model,
        ner_model,
        translation_model,
        summarization_model,
        text_classification_model,
    )

    # Sentiment
    model = sentiment_model()
    model.predict("gila best makanan dia")

    # Translation
    model = translation_model()
    model.translate("saya suka makan nasi goreng", target='en')
"""

from __future__ import annotations

from typing import Any, Optional

from malaysian_manglish_nlp.transformers.base import check_dependencies
from malaysian_manglish_nlp.transformers.sentiment import sentiment_model
from malaysian_manglish_nlp.transformers.ner import ner_model
from malaysian_manglish_nlp.transformers.translation import translation_model
from malaysian_manglish_nlp.transformers.summarization import summarization_model
from malaysian_manglish_nlp.transformers.classification import text_classification_model
from malaysian_manglish_nlp.transformers.pos import pos_model

# Multi-task fine-tuned model (lazy imports to avoid torch dependency at package level)
def manglish_multitask_model(model_dir: Optional[str] = None) -> Any:
    """Load the fine-tuned multi-task Manglish model.
    
    Returns a bundle for use with predict()/predict_batch().
    Falls back to demo_predict() if weights are unavailable.
    """
    from malaysian_manglish_nlp.transformers.manglish_model import load_model
    return load_model(model_dir)

__all__ = [
    'sentiment_model',
    'ner_model',
    'translation_model',
    'summarization_model',
    'text_classification_model',
    'pos_model',
    'manglish_multitask_model',
    'check_dependencies',
]

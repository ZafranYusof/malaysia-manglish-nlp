"""Transformer-based POS tagging."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from malaysian_manglish_nlp.transformers.base import TokenClassificationModel

AVAILABLE_MODELS = {
    'mesolitica/pos-bert-base-bahasa-cased': 'BERT base POS for BM',
    'vblagoje/bert-english-uncased-finetuned-pos': 'BERT POS for EN',
    'QCRI/bert-base-multilingual-cased-pos-english': 'Multilingual POS',
}

DEFAULT_MODEL = 'QCRI/bert-base-multilingual-cased-pos-english'


class POSModel(TokenClassificationModel):
    """Transformer-based POS tagging model.
    
    Example:
        >>> model = pos_model()
        >>> model.predict("saya makan nasi goreng")
        [{'word': 'saya', 'tag': 'PRON', 'score': 0.99}, ...]
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the object.

        Args:
            model_name: Model name parameter.

        Returns:
            Result value.

        """
        model = model_name or DEFAULT_MODEL
        super().__init__(model)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """POS tag text using transformer model.
        
        Args:
            text: Input text.
        
        Returns:
            list[dict]: Tokens with 'word', 'tag', 'score'.
        """
        results = self.pipe(text)
        
        output = []
        for r in results:
            output.append({
                'word': r['word'],
                'tag': r['entity_group'],
                'score': round(r['score'], 4),
                'start': r['start'],
                'end': r['end'],
            })
        
        return output
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """POS tag multiple texts.
        
        Args:
            texts: Input texts.
        
        Returns:
            list[list[dict]]: Tags per text.
        """
        return [self.predict(text) for text in texts]


def pos_model(model: Any = None) -> Any:
    """Load a transformer POS model.
    
    Args:
        model: HuggingFace model name.
    
    Returns:
        POSModel: Model instance.
    
    Available models:
        - QCRI/bert-base-multilingual-cased-pos-english (default)
        - mesolitica/pos-bert-base-bahasa-cased (BM-specific)
    """
    return POSModel(model)

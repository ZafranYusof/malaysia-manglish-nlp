"""Transformer-based Named Entity Recognition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from malaysian_manglish_nlp.transformers.base import TokenClassificationModel

AVAILABLE_MODELS = {
    'mesolitica/ner-bert-base-bahasa-cased': 'BERT base NER for BM',
    'cahya/bert-base-indonesian-NER': 'Indonesian BERT NER (works for BM)',
    'dslim/bert-base-NER': 'English BERT NER (multilingual capable)',
}

DEFAULT_MODEL = 'dslim/bert-base-NER'


class NERModel(TokenClassificationModel):
    """Transformer-based NER model.
    
    Example:
        >>> model = ner_model()
        >>> model.predict("Ahmad tinggal di Kuala Lumpur")
        [{'entity': 'PER', 'word': 'Ahmad', 'score': 0.99},
         {'entity': 'LOC', 'word': 'Kuala Lumpur', 'score': 0.98}]
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
        """Predict named entities in text.
        
        Parameters:
            text (str): Input text.
        
        Returns:
            list[dict]: Entities with 'entity', 'word', 'score', 'start', 'end'.
        """
        results = self.pipe(text)
        
        entities = []
        for r in results:
            entity_type = r['entity_group'].replace('B-', '').replace('I-', '')
            entities.append({
                'entity': entity_type,
                'word': r['word'],
                'score': round(r['score'], 4),
                'start': r['start'],
                'end': r['end'],
            })
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict NER for multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
        
        Returns:
            list[list[dict]]: Entities per text.
        """
        return [self.predict(text) for text in texts]


def ner_model(model: Any = None) -> Any:
    """Load a transformer NER model.
    
    Parameters:
        model (str): HuggingFace model name.
    
    Returns:
        NERModel: Model instance.
    """
    return NERModel(model)

"""Transformer-based text classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from malaysian_manglish_nlp.transformers.base import BaseModel

AVAILABLE_MODELS = {
    'mesolitica/bert-base-bahasa-cased': 'BERT base for BM (fine-tune yourself)',
    'cardiffnlp/twitter-roberta-base-emotion': 'Emotion detection (6 classes)',
    'j-hartmann/emotion-english-distilroberta-base': 'Emotion (7 classes)',
    'papluca/xlm-roberta-base-language-detection': 'Language detection (20 langs)',
}

DEFAULT_MODEL = 'cardiffnlp/twitter-roberta-base-emotion'


class TextClassificationModel:
    """Transformer-based text classification.
    
    Example:
        >>> model = text_classification_model()
        >>> model.predict("aku gembira sangat hari ni")
        {'label': 'joy', 'score': 0.89}
    """
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the object.

        Args:
            model_name: Model name parameter.

        Returns:
            Result value.

        """
        from malaysian_manglish_nlp.transformers.base import check_dependencies
        check_dependencies()
        
        self.model_name = model_name or DEFAULT_MODEL
        self._pipe = None
    
    def _load(self) -> None:
        """Internal helper for load.

        Returns:
            Result value.

        """
        if self._pipe is None:
            from transformers import pipeline
            print(f"Loading classification model: {self.model_name}...")
            self._pipe = pipeline("text-classification", model=self.model_name, top_k=None)
            print("Model loaded.")
    
    def predict(self, text: str, top_k: int = 1) -> Dict[str, Any]:
        """Classify text.
        
        Parameters:
            text (str): Input text.
            top_k (int): Number of top predictions to return.
        
        Returns:
            dict or list[dict]: Prediction(s) with 'label' and 'score'.
        """
        self._load()
        results = self._pipe(text)
        
        if isinstance(results[0], list):
            results = results[0]
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        formatted = [{'label': r['label'].lower(), 'score': round(r['score'], 4)} for r in results[:top_k]]
        
        if top_k == 1:
            return formatted[0]
        return formatted
    
    def predict_batch(self, texts: List[str], top_k: int = 1, batch_size: int = 16) -> List[Dict[str, Any]]:
        """Classify multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
            top_k (int): Top predictions per text.
            batch_size (int): Batch size.
        
        Returns:
            list: Predictions per text.
        """
        self._load()
        results = self._pipe(texts, batch_size=batch_size)
        
        output = []
        for result in results:
            if isinstance(result, list):
                result.sort(key=lambda x: x['score'], reverse=True)
                formatted = [{'label': r['label'].lower(), 'score': round(r['score'], 4)} for r in result[:top_k]]
            else:
                formatted = [{'label': result['label'].lower(), 'score': round(result['score'], 4)}]
            
            output.append(formatted[0] if top_k == 1 else formatted)
        
        return output


def text_classification_model(model: Any = None) -> Any:
    """Load a text classification model.
    
    Parameters:
        model (str): HuggingFace model name.
    
    Returns:
        TextClassificationModel: Model instance.
    
    Available models:
        - cardiffnlp/twitter-roberta-base-emotion (default, emotion detection)
        - j-hartmann/emotion-english-distilroberta-base (7 emotions)
        - papluca/xlm-roberta-base-language-detection (language ID)
    """
    return TextClassificationModel(model)

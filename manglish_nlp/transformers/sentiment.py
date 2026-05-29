"""Transformer-based sentiment analysis."""

from manglish_nlp.transformers.base import BaseModel

# Available models (mesolitica/HuggingFace)
AVAILABLE_MODELS = {
    'w11wo/indonesian-roberta-base-sentiment-classifier': 'RoBERTa, BM/ID/Manglish (recommended)',
    'distilbert/distilbert-base-uncased-finetuned-sst-2-english': 'DistilBERT, EN-only (fast)',
    'mesolitica/sentiment-analysis-nanot5-small-malaysian-cased': 'NanoT5, BM (needs custom loading)',
    'cardiffnlp/twitter-roberta-base-sentiment-latest': 'Twitter RoBERTa, multilingual',
}

DEFAULT_MODEL = 'w11wo/indonesian-roberta-base-sentiment-classifier'


class SentimentModel(BaseModel):
    """Transformer-based sentiment analysis model.
    
    Example:
        >>> model = sentiment_model()
        >>> model.predict("gila best makanan dia")
        {'label': 'positive', 'score': 0.95}
        >>> model.predict_batch(["best gila", "teruk la"])
        [{'label': 'positive', ...}, {'label': 'negative', ...}]
    """
    
    def __init__(self, model_name=None):
        model = model_name or DEFAULT_MODEL
        super().__init__(model, task='sentiment')
        self._pipe = None
    
    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            print(f"Loading sentiment model: {self.model_name}...")
            self._pipe = pipeline("sentiment-analysis", model=self.model_name)
            print("Model loaded.")
    
    def predict(self, text):
        """Predict sentiment for a single text.
        
        Parameters:
            text (str): Input text.
        
        Returns:
            dict: {'label': str, 'score': float}
        """
        self._load()
        result = self._pipe(text)[0]
        
        # Normalize label
        label = result['label'].lower()
        if 'pos' in label or label == 'label_2':
            label = 'positive'
        elif 'neg' in label or label == 'label_0':
            label = 'negative'
        else:
            label = 'neutral'
        
        return {'label': label, 'score': round(result['score'], 4)}
    
    def predict_batch(self, texts, batch_size=16):
        """Predict sentiment for multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
            batch_size (int): Batch size for inference.
        
        Returns:
            list[dict]: List of predictions.
        """
        self._load()
        results = self._pipe(texts, batch_size=batch_size)
        
        output = []
        for r in results:
            label = r['label'].lower()
            if 'pos' in label or label == 'label_2':
                label = 'positive'
            elif 'neg' in label or label == 'label_0':
                label = 'negative'
            else:
                label = 'neutral'
            output.append({'label': label, 'score': round(r['score'], 4)})
        
        return output


def sentiment_model(model=None):
    """Load a transformer sentiment model.
    
    Parameters:
        model (str): HuggingFace model name. Default: twitter-roberta-base.
    
    Returns:
        SentimentModel: Model instance.
    
    Available models:
        - cardiffnlp/twitter-roberta-base-sentiment-latest (default, multilingual)
        - mesolitica/sentiment-bert-base-bahasa-cased (BM-specific)
        - w11wo/indonesian-roberta-base-sentiment-classifier (ID/BM)
    """
    return SentimentModel(model)

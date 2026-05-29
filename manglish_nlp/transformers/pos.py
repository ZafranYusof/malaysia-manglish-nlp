"""Transformer-based POS tagging."""

from manglish_nlp.transformers.base import TokenClassificationModel

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
    
    def __init__(self, model_name=None):
        model = model_name or DEFAULT_MODEL
        super().__init__(model)
    
    def predict(self, text):
        """POS tag text using transformer model.
        
        Parameters:
            text (str): Input text.
        
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
    
    def predict_batch(self, texts):
        """POS tag multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
        
        Returns:
            list[list[dict]]: Tags per text.
        """
        return [self.predict(text) for text in texts]


def pos_model(model=None):
    """Load a transformer POS model.
    
    Parameters:
        model (str): HuggingFace model name.
    
    Returns:
        POSModel: Model instance.
    
    Available models:
        - QCRI/bert-base-multilingual-cased-pos-english (default)
        - mesolitica/pos-bert-base-bahasa-cased (BM-specific)
    """
    return POSModel(model)

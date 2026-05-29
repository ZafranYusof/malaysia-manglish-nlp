"""Transformer-based text summarization."""

from manglish_nlp.transformers.base import Seq2SeqModel

AVAILABLE_MODELS = {
    'mesolitica/summarization-t5-small-standard-bahasa-cased': 'T5 small summarizer for BM',
    'mesolitica/summarization-t5-base-standard-bahasa-cased': 'T5 base summarizer for BM',
    'facebook/bart-large-cnn': 'BART large (EN, but works ok for Manglish)',
    'Falconsai/text_summarization': 'Lightweight T5 summarizer',
}

DEFAULT_MODEL = 'Falconsai/text_summarization'


class SummarizationModel(Seq2SeqModel):
    """Transformer-based summarization model.
    
    Example:
        >>> model = summarization_model()
        >>> model.summarize("Kerajaan Malaysia hari ini mengumumkan...")
        'Kerajaan umum dasar baru...'
    """
    
    def __init__(self, model_name=None):
        model = model_name or DEFAULT_MODEL
        super().__init__(model)
    
    def summarize(self, text, max_length=150, min_length=30):
        """Summarize text.
        
        Parameters:
            text (str): Input text (long).
            max_length (int): Max summary length in tokens.
            min_length (int): Min summary length in tokens.
        
        Returns:
            str: Summarized text.
        """
        import torch
        
        # Prefix for T5 models
        if 't5' in self.model_name.lower():
            text = "summarize: " + text
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def summarize_batch(self, texts, max_length=150, min_length=30):
        """Summarize multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
        
        Returns:
            list[str]: Summaries.
        """
        return [self.summarize(t, max_length, min_length) for t in texts]


def summarization_model(model=None):
    """Load a summarization model.
    
    Parameters:
        model (str): HuggingFace model name.
    
    Returns:
        SummarizationModel: Model instance.
    
    Available models:
        - Falconsai/text_summarization (default, lightweight)
        - mesolitica/summarization-t5-small-standard-bahasa-cased (BM)
        - facebook/bart-large-cnn (EN, heavy)
    """
    return SummarizationModel(model)

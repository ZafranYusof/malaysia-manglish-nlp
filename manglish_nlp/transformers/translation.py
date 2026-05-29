"""Transformer-based translation (BM <-> EN)."""

from manglish_nlp.transformers.base import Seq2SeqModel

AVAILABLE_MODELS = {
    'mesolitica/translation-t5-tiny-standard-bahasa-cased': 'T5 tiny BM<->EN (fastest)',
    'mesolitica/translation-t5-small-standard-bahasa-cased': 'T5 small BM<->EN',
    'mesolitica/translation-t5-base-standard-bahasa-cased': 'T5 base BM<->EN',
    'mesolitica/nanot5-small-malaysian-translation-v2': 'NanoT5 small v2',
}

DEFAULT_MODEL = 'mesolitica/translation-t5-tiny-standard-bahasa-cased'


class TranslationModel(Seq2SeqModel):
    """Transformer-based translation model (BM <-> EN).
    
    Uses mesolitica T5 models which handle both directions via prefix.
    
    Example:
        >>> model = translation_model()
        >>> model.translate("saya suka makan nasi goreng", target='en')
        'I like to eat fried rice'
        >>> model.translate("I want to go home", target='bm')
        'Saya mahu pulang ke rumah'
    """
    
    def __init__(self, model_name=None):
        model = model_name or DEFAULT_MODEL
        super().__init__(model)
    
    def translate(self, text, target='en', max_length=256):
        """Translate text between BM and EN.
        
        Parameters:
            text (str): Input text.
            target (str): Target language - 'en' or 'bm'/'ms' (default: 'en').
            max_length (int): Max output length.
        
        Returns:
            str: Translated text.
        """
        if target == 'en':
            prefix = "terjemah Melayu ke Inggeris: "
        elif target in ('bm', 'ms', 'malay'):
            prefix = "terjemah Inggeris ke Melayu: "
        else:
            raise ValueError(f"Unsupported target: {target}. Use 'en' or 'bm'.")
        
        input_text = prefix + text
        return self.generate(input_text, max_length=max_length)
    
    def translate_batch(self, texts, target='en', max_length=256):
        """Translate multiple texts.
        
        Parameters:
            texts (list[str]): Input texts.
            target (str): Target language.
        
        Returns:
            list[str]: Translated texts.
        """
        return [self.translate(t, target, max_length) for t in texts]


def translation_model(model=None):
    """Load a translation model.
    
    Parameters:
        model (str): HuggingFace model name.
    
    Returns:
        TranslationModel: Model instance.
    
    Available models:
        - mesolitica/translation-t5-tiny-standard-bahasa-cased (default, fastest)
        - mesolitica/translation-t5-small-standard-bahasa-cased
        - mesolitica/translation-t5-base-standard-bahasa-cased (best quality)
        - mesolitica/nanot5-small-malaysian-translation-v2
    """
    return TranslationModel(model)

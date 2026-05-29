"""Base utilities for transformer models."""

import importlib


def check_dependencies():
    """Check if torch and transformers are installed.
    
    Returns:
        dict: Status of dependencies.
    
    Raises:
        ImportError: If required packages are missing.
    """
    status = {'torch': False, 'transformers': False}
    
    try:
        import torch
        status['torch'] = True
        status['torch_version'] = torch.__version__
        status['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status['gpu'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    
    try:
        import transformers
        status['transformers'] = True
        status['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    
    if not status['torch'] or not status['transformers']:
        missing = []
        if not status['torch']:
            missing.append('torch')
        if not status['transformers']:
            missing.append('transformers')
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )
    
    return status


class BaseModel:
    """Base class for all transformer models."""
    
    def __init__(self, model_name, task=None):
        check_dependencies()
        self.model_name = model_name
        self.task = task
        self._model = None
        self._tokenizer = None
    
    def _load(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Loading model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            print(f"Model loaded.")
    
    @property
    def model(self):
        self._load()
        return self._model
    
    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer


class Seq2SeqModel:
    """Base class for sequence-to-sequence models (translation, summarization)."""
    
    def __init__(self, model_name):
        check_dependencies()
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
    
    def _load(self):
        """Lazy load model and tokenizer."""
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"Loading model: {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
            print(f"Model loaded.")
    
    @property
    def model(self):
        self._load()
        return self._model
    
    @property
    def tokenizer(self):
        self._load()
        return self._tokenizer
    
    def generate(self, text, max_length=256, **kwargs):
        """Generate output from input text."""
        import torch
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                **kwargs,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Clean up extra whitespace from tokenizer
        result = ' '.join(result.split())
        return result


class TokenClassificationModel:
    """Base class for token classification (NER, POS)."""
    
    def __init__(self, model_name):
        check_dependencies()
        self.model_name = model_name
        self._pipeline = None
    
    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            print(f"Loading model: {self.model_name}...")
            self._pipeline = pipeline("token-classification", model=self.model_name, aggregation_strategy="simple")
            print(f"Model loaded.")
    
    @property
    def pipe(self):
        self._load()
        return self._pipeline

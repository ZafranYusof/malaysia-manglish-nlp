# Pretrained Models

malaysian-manglish-nlp ships with pretrained models trained on real Malaysian social media data.

---

## Word Embeddings

| Model | Method | Dimensions | Vocab | Training Data |
|-------|--------|-----------|-------|---------------|
| `manglish-word2vec` | Word2Vec CBOW | 100 | 518 | 50k+ tweets |
| `manglish-fasttext` | FastText skip-gram | 100 | 518 | 50k+ tweets |

### Loading word embeddings

```python
from malaysian_manglish_nlp import word_embeddings

# Load Word2Vec
w2v = word_embeddings.load_word2vec()
w2v.most_similar("makan")
# [('nasi', 0.82), ('roti', 0.79), ('minum', 0.74), ...]

# Load FastText
ft = word_embeddings.load_fasttext()
ft.most_similar("best")
# [('gempak', 0.78), ('power', 0.76), ('padu', 0.73), ...]
```

---

## Fine-tuned Multi-Task Model (v3.1.0)

A DistilBERT model fine-tuned on **7,884 labeled Manglish examples** for multi-task classification.

| Task | Accuracy | Classes |
|------|----------|----------|
| Sentiment | 88.5% | positive, negative, neutral |
| Emotion | 83.6% | happy, sad, angry, fear, surprise, disgust, love, neutral |
| Intent | 94.5% | question, statement, request, complaint, greeting, opinion |
| **Average** | **88.9%** | |

### Usage

```python
from malaysian_manglish_nlp.transformers.manglish_model import load_model, predict

# Load model (auto-downloads from HuggingFace on first use)
model = load_model()

# Predict
result = predict("gila best servis ni")
# {'sentiment': {'label': 'positive', 'confidence': 0.96},
#  'emotion':    {'label': 'happy',    'confidence': 0.85},
#  'intent':     {'label': 'opinion',  'confidence': 1.00}}

# Batch prediction
results = predict_batch(["best gila", "teruk la", "ok je"])
```

### Model details

- **Base**: `distilbert-base-multilingual-cased`
- **Architecture**: Shared encoder + 3 task-specific heads (256 hidden units each)
- **Fine-tuned on**: 6,307 train / 1,577 validation (from 7,884 total)
- **Training**: 5 epochs, lr=2e-5 (encoder) / 2e-4 (heads), batch_size=16
- **Optimizer**: AdamW with linear warmup scheduler
- **Hardware**: NVIDIA RTX 2070 8GB VRAM
- **Model size**: ~541MB (PyTorch state dict)

### Training history

| Epoch | Train Loss | Val Loss | Sentiment | Emotion | Intent | Avg Acc |
|-------|-----------|----------|-----------|---------|--------|---------|
| 1 | 1.258 | 0.928 | 63.0% | 50.6% | 75.6% | 63.1% |
| 2 | 0.743 | 0.599 | 78.9% | 71.1% | 87.9% | 79.3% |
| 3 | 0.462 | 0.441 | 86.1% | 78.2% | 92.1% | 85.5% |
| 4 | 0.316 | 0.390 | 87.4% | 82.8% | 94.4% | 88.2% |
| **5** | **0.243** | **0.375** | **88.5%** | **83.6%** | **94.5%** | **88.9%** |

### Download from HuggingFace

```python
# Auto-download (built into load_model())
from malaysian_manglish_nlp.transformers.manglish_model import load_model
model = load_model()

# Or manual download
from huggingface_hub import hf_hub_download
hf_hub_download("vexccz/manglish-nlp-sentiment", "model.pt")
hf_hub_download("vexccz/manglish-nlp-sentiment", "config.json")
hf_hub_download("vexccz/manglish-nlp-sentiment", "tokenizer.json")
hf_hub_download("vexccz/manglish-nlp-sentiment", "tokenizer_config.json")
```

### Rule-based fallback

If the fine-tuned model is not available (no `[transformers]` extra), use the built-in rule-based modules:

```python
from malaysian_manglish_nlp import sentiment, detect_emotion, classify_intent

# Rule-based sentiment (no model needed)
result = sentiment("Best lah movie ni, memang power!")
# {'sentiment': 'positive', 'score': 0.94}

# Rule-based emotion
emotion = detect_emotion("sedih doh tak dapat tiket")
# {'emotion': 'sad', 'confidence': 0.82}
```

### Comparison with previous model

| | v3.0.0 (561 examples) | v3.1.0 (7,884 examples) |
|---|---|---|
| Sentiment | 69% | **88.5%** |
| Emotion | 63% | **83.6%** |
| Intent | 69% | **94.5%** |
| Average | 67% | **88.9%** |
| Tasks | Single (sentiment) | Multi-task (3 tasks) |

### Comparison with other models

| Model | Accuracy | Notes |
|-------|---------|-------|
| `manglish-finetuned` v3.1.0 | **88.9%** | Multi-task, best for Manglish |
| Mesolitica NanoT5 (tiny) | 86.1% | Malay-only base |
| huseinzol05 sentiment | 84.7% | Broader Malay coverage |
| DistilBERT multilingual (zero-shot) | 62.3% | No fine-tuning |

---

## Model Storage

Models are stored locally or downloaded from HuggingFace:

```
~/.agents/skills/manglish-nlp/malaysian_manglish_nlp/resources/
├── manglish_finetuned/
│   ├── model.pt          # 541MB
│   ├── config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── word_embeddings/
    ├── word2vec.model
    └── fasttext.model
```

---

## Citation

```bibtex
@software{malaysian_manglish_nlp_2026,
  title  = {malaysian-manglish-nlp: A Comprehensive NLP Toolkit for Malaysian Manglish},
  author = {Yusof, Zafran},
  year   = {2026},
  version = {3.1.0},
  url    = {https://github.com/ZafranYusof/malaysia-manglish-nlp}
}
```

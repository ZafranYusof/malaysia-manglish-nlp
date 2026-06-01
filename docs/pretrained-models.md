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

## Fine-tuned Multi-Task Model (v3.3.0)

An XLM-Roberta model fine-tuned on **28,263 labeled Manglish examples** for multi-task classification.

| Task | Accuracy | Classes |
|------|----------|----------|
| Sentiment | 98.0% | positive, negative, neutral |
| Emotion | 96.5% | happy, sad, angry, fear, surprise, disgust, love, neutral |
| Intent | 99.3% | question, statement, request, complaint, greeting, opinion, command, offer |
| **Average** | **97.9%** | |

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

- **Base**: `xlm-roberta-base`
- **Architecture**: Shared encoder + 3 task-specific heads (256 hidden units each)
- **Fine-tuned on**: 22,610 train / 5,653 validation (from 28,263 total)
- **Training**: 5 epochs, lr=2e-5 (encoder) / 2e-4 (heads), batch_size=16, gradient accumulation (effective batch 32)
- **Optimizer**: AdamW with cosine annealing and warm restarts
- **Regularization**: Focal loss for class imbalance, uncertainty-weighted multi-task loss, early stopping
- **Hardware**: NVIDIA RTX 2070 8GB VRAM, mixed precision (FP16)
- **Model size**: ~1.1GB (PyTorch state dict)

### Training history

| Epoch | Train Loss | Val Loss | Sentiment | Emotion | Intent | Avg Acc |
|-------|-----------|----------|-----------|---------|--------|---------|
| 1 | 1.258 | 0.928 | 63.0% | 50.6% | 75.6% | 63.1% |
| 2 | 0.743 | 0.599 | 78.9% | 71.1% | 87.9% | 79.3% |
| 3 | 0.462 | 0.441 | 86.1% | 78.2% | 92.1% | 85.5% |
| 4 | 0.316 | 0.390 | 87.4% | 82.8% | 94.4% | 88.2% |
| **5** | **0.243** | **0.375** | **88.5%** | **83.6%** | **94.5%** | **88.9%** |

> **v3.2.0** (XLM-Roberta, 14,384 examples): Sentiment 95.0%, Emotion 90.3%, Intent 97.5%, Avg 94.3%
>
> **v3.3.0** (XLM-Roberta, 28,263 examples): Sentiment 98.0%, Emotion 96.5%, Intent 99.3%, Avg 97.9%

**v3.3.0 retraining notes:** Filtered 4,801 partial-label samples that caused multi-task training KeyError. Contrast-marker-aware window scoring added for aspect sentiment. Pydantic v2 ConfigDict migration applied.

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

| | v3.0.0 (561 examples) | v3.1.0 (7,884 examples) | v3.2.0 (14,384 examples) | v3.3.0 (28,263 examples) |
|---|---|---|---|---|
| Sentiment | 69% | 88.5% | 95.0% | **98.0%** |
| Emotion | 63% | 83.6% | 90.3% | **96.5%** |
| Intent | 69% | 94.5% | 97.5% | **99.3%** |
| Average | 67% | 88.9% | 94.3% | **97.9%** |
| Base model | DistilBERT | DistilBERT | XLM-Roberta | XLM-Roberta |
| Tasks | Single (sentiment) | Multi-task (3 tasks) | Multi-task (3 tasks) | Multi-task (3 tasks) |

### Comparison with other models

| Model | Accuracy | Notes |
|-------|---------|-------|
| `manglish-finetuned` v3.3.0 | **97.9%** | XLM-Roberta multi-task, best for Manglish |
| `manglish-finetuned` v3.2.0 | 94.3% | XLM-Roberta multi-task |
| `manglish-finetuned` v3.1.0 | 88.9% | DistilBERT multi-task |
| Mesolitica NanoT5 (tiny) | 86.1% | Malay-only base |
| huseinzol05 sentiment | 84.7% | Broader Malay coverage |
| DistilBERT multilingual (zero-shot) | 62.3% | No fine-tuning |

---

## Model Storage

Models are stored locally or downloaded from HuggingFace:

```
~/.agents/skills/manglish-nlp/malaysian_manglish_nlp/resources/
├── manglish_finetuned/
│   ├── model.pt          # 1.1GB
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
  version = {3.3.0},
  url    = {https://github.com/ZafranYusof/malaysia-manglish-nlp}
}
```

# Manglish NLP Model Card

## Model Description

An XLM-Roberta model fine-tuned on Manglish (Malaysian English) text for multi-task classification. The model handles code-switched Malay-English text commonly found in Malaysian social media and messaging.

**Base model:** xlm-roberta-base  
**Fine-tuned tasks:** Sentiment, Emotion, Intent classification  
**Language:** Manglish (Malay-English code-switched)  
**Latest version:** v3.3.0

## Intended Use

- Sentiment analysis of Malaysian social media posts
- Emotion detection in Manglish text
- Intent classification for chatbots serving Malaysian users
- Aspect-based sentiment analysis for reviews (restaurant, product, app, general domains)
- Multi-label emotion detection for nuanced emotional states
- Research on code-switched NLP

### Out-of-Scope Use

- Formal Malay or English text (use dedicated models instead)
- Languages other than Malay/English
- Production systems requiring >99% accuracy on formal text

## Training Data

- **Dataset size:** 28,263 labeled examples (34,548 total merged)
- **Tasks:** 3 (sentiment, emotion, intent)
- **Source:** Malaysian social media, news, chat messages covering code-switched text
- **Text characteristics:** Shortforms, particles (lah, wei, eh), code-switching, slang
- **HuggingFace dataset:** [vexccz/manglish-nlp-dataset](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset)

### Training Procedure

- Epochs: 5
- Batch size: 16 (gradient accumulation effective batch 32)
- Learning rate: 2e-5 (encoder) / 2e-4 (heads)
- Optimizer: AdamW with cosine annealing and warm restarts
- Max sequence length: 96
- Loss: Focal loss for class imbalance, uncertainty-weighted multi-task loss (Kendall et al. 2018)
- Regularization: Early stopping, mixed precision (FP16)
- Hardware: NVIDIA RTX 2070 8GB VRAM

## Evaluation Results

### v3.3.0 (28,263 examples)

| Task | Accuracy | F1 (weighted) |
|------|----------|---------------|
| Sentiment | 98.0% | 0.98 |
| Emotion | 96.5% | 0.96 |
| Intent | 99.3% | 0.99 |
| **Average** | **97.9%** | **0.98** |

### Per-class Performance (Sentiment)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| positive | 0.98 | 0.98 | 0.98 |
| negative | 0.97 | 0.98 | 0.98 |
| neutral | 0.98 | 0.97 | 0.97 |

## Version History

| Version | Examples | Sentiment | Emotion | Intent | Avg | Base Model |
|---------|----------|-----------|---------|--------|-----|------------|
| v3.0.0 | 561 | 69.0% | 62.8% | 69.0% | 67.0% | DistilBERT |
| v3.1.0 | 7,884 | 88.5% | 83.6% | 94.5% | 88.9% | DistilBERT |
| v3.2.0 | 14,384 | 95.0% | 90.3% | 97.5% | 94.3% | XLM-Roberta |
| **v3.3.0** | **28,263** | **98.0%** | **96.5%** | **99.3%** | **97.9%** | **XLM-Roberta** |

### What changed in v3.3.0
- Dataset nearly doubled: 28,263 examples (from 14,384), 34,548 total merged
- Aspect-Based Sentiment: per-aspect sentiment with 4 domains, conflict detection
- Multi-Label Emotion: detect multiple emotions simultaneously, 10 co-occurrence patterns
- Feedback Loop: user corrections, active learning, uncertainty sampling, JSONL export
- WebSocket streaming API and async batch processing
- Multi-task training KeyError fixed (filtered 4,801 partial-label samples)
- Contrast-marker-aware window scoring in aspect sentiment

## Limitations

- **Regional bias:** Primarily KL/urban Manglish patterns
- **Shortform handling:** Model sees raw text; pairing with normalization improves results
- **No Mandarin/Tamil:** Does not handle Chinese or Tamil code-switching common in some Malaysian communities
- **Aspect sentiment domains:** Currently limited to restaurant, product, app, and general domains

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "vexccz/manglish-nlp-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict sentiment
text = "Weh best gila makanan kat sini la"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)

labels = ["negative", "neutral", "positive"]
print(f"Sentiment: {labels[prediction.item()]}")
```

### Using the malaysian-manglish-nlp package

```python
from malaysian_manglish_nlp.transformers.manglish_model import load_model, predict

# Load model (auto-downloads from HuggingFace on first use)
model = load_model()

# Predict
result = predict("Weh best gila makanan kat sini la")
# {'sentiment': {'label': 'positive', 'confidence': 0.98},
#  'emotion':    {'label': 'happy',    'confidence': 0.92},
#  'intent':     {'label': 'opinion',  'confidence': 0.99}}

# Aspect-based sentiment (v3.3.0)
import malaysian_manglish_nlp
result = malaysian_manglish_nlp.analyze_aspect_sentiment("makanan sedap tapi service teruk", domain="restaurant")

# Multi-label emotion (v3.3.0)
result = malaysian_manglish_nlp.detect_multi_emotion("sedih tapi grateful dapat jumpa family")
```

## Model Storage

```
~/.agents/skills/manglish-nlp/malaysian_manglish_nlp/resources/
└── manglish_finetuned/
    ├── model.pt          # ~1.1GB
    ├── config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Citation

```bibtex
@model{yusof2026manglishnlp,
  title={Manglish NLP: Multi-task Classification for Malaysian Code-Switched Text},
  author={Yusof, Zafran},
  year={2026},
  version={3.3.0},
  url={https://huggingface.co/vexccz/manglish-nlp-sentiment}
}
```

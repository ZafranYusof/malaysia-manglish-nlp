# Manglish NLP Model Card

## Model Description

A DistilBERT model fine-tuned on Manglish (Malaysian English) text for multi-task classification. The model handles code-switched Malay-English text commonly found in Malaysian social media and messaging.

**Base model:** distilbert-base-multilingual-cased  
**Fine-tuned tasks:** Sentiment, Emotion, Intent classification  
**Language:** Manglish (Malay-English code-switched)

## Intended Use

- Sentiment analysis of Malaysian social media posts
- Emotion detection in Manglish text
- Intent classification for chatbots serving Malaysian users
- Research on code-switched NLP

### Out-of-Scope Use

- Formal Malay or English text (use dedicated models instead)
- Languages other than Malay/English
- Production systems requiring >95% accuracy

## Training Data

- **Dataset size:** 561 labeled examples (train split)
- **Tasks:** 3 (sentiment, emotion, intent)
- **Source:** Synthetic Manglish examples covering social media styles
- **Text characteristics:** Shortforms, particles (lah, wei, eh), code-switching, slang

### Training Procedure

- Epochs: 10
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW
- Max sequence length: 128

## Evaluation Results

| Task | Accuracy | F1 (weighted) |
|------|----------|---------------|
| Sentiment | 69.0% | 0.68 |
| Emotion | 62.8% | 0.61 |
| Intent | 69.0% | 0.67 |

### Per-class Performance (Sentiment)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| positive | 0.72 | 0.75 | 0.73 |
| negative | 0.68 | 0.65 | 0.66 |
| neutral | 0.65 | 0.67 | 0.66 |

## Limitations

- **Small training set:** Only 561 examples — performance will improve with more data
- **Synthetic data:** May not fully capture real-world Manglish diversity
- **Regional bias:** Primarily KL/urban Manglish patterns
- **Shortform handling:** Model sees raw text; pairing with normalization improves results
- **No Mandarin/Tamil:** Does not handle Chinese or Tamil code-switching common in some Malaysian communities

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "ZafranYusof/manglish-nlp-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict sentiment
text = "Weh best gila makanan kat sini la"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)

labels = ["negative", "neutral", "positive"]
print(f"Sentiment: {labels[prediction.item()]}")
```

### Using the manglish-nlp package

```python
from manglish_nlp import ManglishAnalyzer

analyzer = ManglishAnalyzer()
result = analyzer.analyze("Weh best gila makanan kat sini la")
print(result.sentiment)  # positive
print(result.emotion)    # happy
print(result.intent)     # statement
```

## Citation

```bibtex
@model{yusof2025manglishnlp,
  title={Manglish NLP: Multi-task Classification for Malaysian Code-Switched Text},
  author={Yusof, Zafran},
  year={2025},
  url={https://huggingface.co/ZafranYusof/manglish-nlp-model}
}
```

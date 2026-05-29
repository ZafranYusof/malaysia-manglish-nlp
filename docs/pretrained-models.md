# Pretrained Models

manglish-nlp ships with pretrained models trained on real Malaysian social media data.

---

## Word Embeddings

| Model | Method | Dimensions | Vocab | Training Data |
|-------|--------|-----------|-------|---------------|
| `manglish-word2vec` | Word2Vec CBOW | 100 | 518 | 50k+ tweets |
| `manglish-fasttext` | FastText skip-gram | 100 | 518 | 50k+ tweets |

### Loading word embeddings

```python
from manglish_nlp import word_embeddings

# Load Word2Vec
w2v = word_embeddings.load_word2vec()
w2v.most_similar("makan")
# [('nasi', 0.82), ('roti', 0.79), ('minum', 0.74), ...]

# Load FastText
ft = word_embeddings.load_fasttext()
ft.most_similar("best")
# [('gempak', 0.78), ('power', 0.76), ('padu', 0.73), ...]
```

### How they were trained

Both models were trained on 50,000+ Malaysian tweets collected between 2023–2025. Text was preprocessed using manglish-nlp's normalisation pipeline (slang → standard Malay/English) before training.

- **Word2Vec**: Gensim CBOW, window=5, min_count=2, epochs=50
- **FastText**: Gensim skip-gram, window=5, min_count=2, epochs=50

### Limitations

- Vocabulary is small (518 tokens) — reflects curated Manglish lexicon, not full Malay/English
- Best used for similarity lookups and as input features, not standalone NLU

---

## Fine-tuned Sentiment Model

| Model | Architecture | Accuracy | F1 | Training Data |
|-------|-------------|----------|----|---------------|
| `manglish-finetuned` | DistilBERT (multilingual) | 89.1% | 0.89 | 1,139 labeled examples |

### Usage

```python
from manglish_nlp import sentiment

# Rule-based (no model needed)
result = sentiment.analyse("Best lah movie ni, memang power!")
# SentimentResult(label='positive', score=0.78)

# Fine-tuned model (requires transformers extra)
from manglish_nlp.advanced import FinetunedSentimentClassifier
classifier = FinetunedSentimentClassifier()
classifier.predict("Best lah movie ni, memang power!")
# 'positive'
```

### Model details

- **Base**: `distilbert-base-multilingual-cased`
- **Fine-tuned on**: 912 train / 227 test Manglish examples
- **Labels**: positive, negative, neutral
- **Training**: 8 epochs, lr=2e-5, batch_size=16
- **Hardware**: Trained on Google Colab (T4 GPU)

### Comparison with other models

| Model | Accuracy | Notes |
|-------|---------|-------|
| `manglish-finetuned` | 89.1% | Best for Manglish |
| Mesolitica NanoT5 (tiny) | 86.1% | Malay-only base |
| huseinzol05 sentiment | 84.7% | Broader Malay coverage |
| DistilBERT multilingual (zero-shot) | 62.3% | No fine-tuning |

---

## Model Storage

Models are bundled with the package or downloaded on first use:

```
~/.manglish_nlp/models/
├── word2vec/
│   └── manglish-word2vec.model
├── fasttext/
│   └── manglish-fasttext.model
└── sentiment/
    └── manglish-finetuned/
        ├── config.json
        ├── model.safetensors
        └── tokenizer/
```

---

## Training Scripts

All models were trained using scripts in `scripts/train/`:

| Script | Output |
|--------|--------|
| `scripts/train/train_word2vec.py` | Word2Vec model |
| `scripts/train/train_fasttext.py` | FastText model |
| `scripts/train/train_sentiment.py` | DistilBERT fine-tuned classifier |
| `scripts/train/train_sentiment_notebook.ipynb` | Colab training notebook |

---

## Citation

If you use these models in your research:

```bibtex
@software{manglish_nlp_2025,
  title  = {manglish-nlp: A Comprehensive NLP Toolkit for Malaysian Manglish},
  author = {Yusof, Zafran},
  year   = {2025},
  url    = {https://github.com/ZafranYusof/manglish-nlp}
}
```

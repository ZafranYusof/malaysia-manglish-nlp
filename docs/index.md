# Welcome to Manglish documentation!

<div align="center" markdown="1">

[![PyPI version](https://badge.fury.io/py/malaysian-manglish-nlp.svg)](https://pypi.org/project/malaysian-manglish-nlp/)
[![Python versions](https://img.shields.io/pypi/pyversions/malaysian-manglish-nlp.svg)](https://pypi.org/project/malaysian-manglish-nlp/)
[![License: MIT](https://img.shields.io/github/license/ZafranYusof/malaysia-manglish-nlp.svg?color=blue)](https://github.com/ZafranYusof/malaysia-manglish-nlp/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/manglish-nlp/badge/?version=latest)](https://manglish-nlp.readthedocs.io/en/latest/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97-Model-yellow)](https://huggingface.co/vexccz/manglish-nlp-sentiment)
[![HuggingFace Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97-Space-yellow)](https://huggingface.co/spaces/vexccz/manglish-nlp-demo)
[![GitHub stars](https://img.shields.io/github/stars/ZafranYusof/malaysia-manglish-nlp)](https://github.com/ZafranYusof/malaysia-manglish-nlp/stargazers)

</div>

---

**malaysian-manglish-nlp** is a comprehensive Natural-Language-Processing toolkit for Malaysian Manglish - the code-switching mix of Malay, English, and local slang spoken by millions of Malaysians online.

It provides 54 modules covering sentiment analysis, named entity recognition, translation, normalisation, text generation, graph analysis, and more. Zero external dependencies for core modules.

## HuggingFace

| | Link |
|---|---|
| Model | [vexccz/manglish-nlp-sentiment](https://huggingface.co/vexccz/manglish-nlp-sentiment) - XLM-Roberta multi-task (sentiment 98.0%, emotion 96.5%, intent 99.3%) |
| Dataset | [vexccz/manglish-nlp-dataset](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset) - 28,263 labeled Manglish examples |
| Demo | [vexccz/manglish-nlp-demo](https://huggingface.co/spaces/vexccz/manglish-nlp-demo) - Gradio interactive demo (7 tabs) |

## Documentation

Proper documentation is available at [https://manglish-nlp.readthedocs.io/en/latest/](https://manglish-nlp.readthedocs.io/en/latest/)

## Installing from PyPI

```bash
pip install malaysian-manglish-nlp
```

Only **Python >= 3.8.0** is required.

### Extras

```bash
pip install malaysian-manglish-nlp[transformers]   # HuggingFace models
pip install malaysian-manglish-nlp[embeddings]     # Word2Vec / FastText
pip install malaysian-manglish-nlp[api]            # FastAPI REST server
pip install malaysian-manglish-nlp[all]            # Everything
```

## Development Release

Install from master branch:

```bash
pip install git+https://github.com/ZafranYusof/malaysia-manglish-nlp.git
```

## Pretrained Models

malaysian-manglish-nlp ships with pretrained Malaysian models. See [Pretrained Models](pretrained-models.md).

| Model | Type | Details |
|-------|------|---------|
| `manglish-word2vec` | Word Embedding | 100-dim, 518 vocab, trained on 50k+ tweets |
| `manglish-fasttext` | Word Embedding | 100-dim, 518 vocab, trained on 50k+ tweets |
| `manglish-finetuned` v3.2.0 | Multi-Task Classifier | XLM-Roberta, sentiment 95.0% / emotion 90.3% / intent 97.5% (14,384 examples) |
| `manglish-finetuned` v3.3.0 | Multi-Task Classifier | XLM-Roberta, sentiment 98.0% / emotion 96.5% / intent 99.3% (28,263 examples) |

## Datasets

Training data is bundled with the package. See [Datasets](datasets.md).

- **Multi-task**: 28,263 labeled examples (sentiment + emotion + intent) - [HuggingFace](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset)
- **Normalisation**: 638+ slang to standard mappings
- **NER**: 2,250+ annotated sentences (11 entity types)
- **Translation**: 1,000+ BM-EN word pairs

## Features

54 modules across 8 categories:

| Category | Modules | Examples |
|----------|---------|---------|
| Text Processing | 9 | `normalize`, `tokenize`, `sentence_split` |
| Analysis | 10 | `sentiment`, `emotion`, `aspect_sentiment`, `multi_emotion` |
| Extraction | 7 | `ner`, `keyword`, `entity_linking` |
| Advanced | 7 | `FinetunedSentimentClassifier`, `fewshot`, `llm` |
| Generation | 6 | `translate`, `paraphrase`, `augment` |
| Data & Embeddings | 5 | `word2vec`, `fasttext`, `load_sentiment` |
| Tools & Utilities | 7 | `pipeline`, `batch`, `benchmark`, `feedback` |
| Integrations | 4 | `to_spacy`, `to_huggingface`, `api_server` |

## Quick Start

```python
import malaysian_manglish_nlp

# Sentiment analysis
result = malaysian_manglish_nlp.sentiment.analyse("Best lah movie ni, memang power!")
# SentimentResult(label='positive', score=0.78)

# Normalisation
normal = malaysian_manglish_nlp.normalize("sy xnak g sbb hujan lebat")
# "saya tidak mahu pergi sebab hujan lebat"

# NER
entities = malaysian_manglish_nlp.ner.extract("Najib Razak mengumumkan dasar baharu di Kuala Lumpur")
# [('Najib Razak', 'PER'), ('Kuala Lumpur', 'LOC')]

# Translation
translated = malaysian_manglish_nlp.translate("Apa khabar hari ini?", source="ms", target="en")
# "How are you today?"

# Fine-tuned model (requires [transformers] extra)
from malaysian_manglish_nlp.transformers.manglish_model import load_model, predict
model = load_model()  # Auto-downloads from HuggingFace
result = predict("gila best servis ni")
# {'sentiment': {'label': 'positive', 'confidence': 0.96},
#  'emotion':    {'label': 'happy',    'confidence': 0.85},
#  'intent':     {'label': 'opinion',  'confidence': 1.00}}

# Aspect-based sentiment (v3.3.0)
result = malaysian_manglish_nlp.analyze_aspect_sentiment("makanan sedap tapi service teruk", domain="restaurant")
# {'aspects': [{'aspect': 'food', 'sentiment': 'positive', ...}, {'aspect': 'service', 'sentiment': 'negative', ...}], ...}

# Multi-label emotion (v3.3.0)
result = malaysian_manglish_nlp.detect_multi_emotion("sedih tapi grateful dapat jumpa family")
# {'emotions': [{'emotion': 'happy', 'confidence': 0.6}, {'emotion': 'sad', 'confidence': 0.4}], 'dominant': 'happy', 'is_multi': True}

# Feedback loop (v3.3.0)
from malaysian_manglish_nlp.feedback import submit_correction
submit_correction("text here", "sentiment", "positive", "negative")
```

## Running on Windows

See [Running on Windows](running-on-windows.md) for common issues and recommended setup.

## Benchmarks

See [Benchmarks](benchmarks.md) for accuracy comparisons against Malaya, Mesolitica, and other models.

## Contributing

Pull requests are welcome. See [Contributing](contributing.md) for guidelines.

## Acknowledgement

Heavily inspired by [Malaya](https://github.com/huseinzol05/Malaya) by Hussein Zolkepli. See [Acknowledgement](acknowledgement.md) for the full list of tools, data sources, and contributors.

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

## License

[MIT License](https://github.com/ZafranYusof/malaysia-manglish-nlp/blob/main/LICENSE) - free for commercial and non-commercial use.

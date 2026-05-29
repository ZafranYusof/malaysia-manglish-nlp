# Welcome to manglish-nlp's documentation!

<div align="center">

[![PyPI version](https://badge.fury.io/py/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![Python versions](https://img.shields.io/pypi/pyversions/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![License: MIT](https://img.shields.io/github/license/ZafranYusof/manglish-nlp.svg?color=blue)](https://github.com/ZafranYusof/manglish-nlp/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/manglish-nlp/badge/?version=latest)](https://manglish-nlp.readthedocs.io/)
[![GitHub stars](https://img.shields.io/github/stars/ZafranYusof/manglish-nlp)](https://github.com/ZafranYusof/manglish-nlp/stargazers)

</div>

---

**manglish-nlp** is a comprehensive Natural-Language-Processing toolkit for Malaysian Manglish — the code-switching mix of Malay, English, and local slang spoken by millions of Malaysians online.

It provides 51 modules covering sentiment analysis, named entity recognition, translation, normalisation, text generation, graph analysis, and more. Zero external dependencies for core modules.

## Documentation

Proper documentation is available at [https://manglish-nlp.readthedocs.io/](https://manglish-nlp.readthedocs.io/)

## Installing from PyPI

```bash
pip install manglish-nlp
```

Only **Python >= 3.8.0** is required.

### Extras

```bash
pip install manglish-nlp[transformers]   # HuggingFace models
pip install manglish-nlp[embeddings]     # Word2Vec / FastText
pip install manglish-nlp[api]            # FastAPI REST server
pip install manglish-nlp[all]            # Everything
```

## Development Release

Install from master branch:

```bash
pip install git+https://github.com/ZafranYusof/manglish-nlp.git
```

## Pretrained Models

manglish-nlp ships with pretrained Malaysian models. See [Pretrained Models](pretrained-models.md).

| Model | Type | Details |
|-------|------|---------|
| `manglish-word2vec` | Word Embedding | 100-dim, 518 vocab, trained on 50k+ tweets |
| `manglish-fasttext` | Word Embedding | 100-dim, 518 vocab, trained on 50k+ tweets |
| `manglish-finetuned` | Sentiment Classifier | DistilBERT multilingual, 89.1% accuracy |

## Datasets

Training data is bundled with the package. See [Datasets](datasets.md).

- **Sentiment**: 1,139 labeled examples (positive / negative / neutral)
- **Normalisation**: 259 slang → standard pairs
- **NER**: 2,250 annotated sentences (PER, ORG, LOC, MISC)
- **Translation**: 600+ EN↔MY parallel pairs

## Features

51 modules across 8 categories:

| Category | Modules | Examples |
|----------|---------|---------|
| Text Processing | 9 | `normalize`, `tokenize`, `sentence_split` |
| Analysis | 7 | `sentiment`, `emotion`, `subjectivity` |
| Extraction | 7 | `ner`, `keyword`, `entity_linking` |
| Advanced | 7 | `FinetunedSentimentClassifier`, `fewshot`, `llm` |
| Generation | 6 | `translate`, `paraphrase`, `augment` |
| Data & Embeddings | 5 | `word2vec`, `fasttext`, `load_sentiment` |
| Tools & Utilities | 6 | `pipeline`, `batch`, `benchmark` |
| Integrations | 4 | `to_spacy`, `to_huggingface`, `api_server` |

## Quick Start

```python
import manglish_nlp

# Sentiment analysis
result = manglish_nlp.sentiment.analyse("Best lah movie ni, memang power!")
# SentimentResult(label='positive', score=0.78)

# Normalisation
normal = manglish_nlp.normalize("sy xnak g sbb hujan lebat")
# "saya tidak mahu pergi sebab hujan lebat"

# NER
entities = manglish_nlp.ner.extract("Najib Razak mengumumkan dasar baharu di Kuala Lumpur")
# [('Najib Razak', 'PER'), ('Kuala Lumpur', 'LOC')]

# Translation
translated = manglish_nlp.translate("Apa khabar hari ini?", source="ms", target="en")
# "How are you today?"
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
@software{manglish_nlp_2025,
  title  = {manglish-nlp: A Comprehensive NLP Toolkit for Malaysian Manglish},
  author = {Yusof, Zafran},
  year   = {2025},
  url    = {https://github.com/ZafranYusof/manglish-nlp}
}
```

## License

[MIT License](https://github.com/ZafranYusof/manglish-nlp/blob/main/LICENSE) — free for commercial and non-commercial use.

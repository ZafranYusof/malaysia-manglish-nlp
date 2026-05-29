# manglish-nlp

[![PyPI version](https://img.shields.io/pypi/v/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![Python versions](https://img.shields.io/pypi/pyversions/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![Tests](https://img.shields.io/github/actions/workflow/status/ZafranYusof/manglish-nlp/test.yml?label=tests)](https://github.com/ZafranYusof/manglish-nlp/actions)
[![Docs](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://manglish-nlp.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Full NLP toolkit for Malaysian Manglish — 51 modules, zero dependencies for core.**

Built for real-world Malaysian text: social media, news, chat messages, code-switched Malay-English content.

## Installation

```bash
pip install manglish-nlp
```

### Extras

```bash
pip install manglish-nlp[transformers]   # HuggingFace transformer models
pip install manglish-nlp[embeddings]     # Word2Vec/FastText embeddings
pip install manglish-nlp[spacy]          # spaCy integration
pip install manglish-nlp[api]            # FastAPI REST API
pip install manglish-nlp[langchain]      # LangChain tools
pip install manglish-nlp[all]            # Everything
```

## Quick Start

```python
from manglish_nlp import sentiment, normalize, ner, detect_language

# Sentiment analysis
result = sentiment("Weh best gila makanan dia!")
print(result)
# {'sentiment': 'positive', 'score': 0.94, 'raw_score': 2.5}

# Text normalization
clean = normalize("xpe la bro, aku ok je")
print(clean)  # "tidak apa la bro, aku okay sahaja"

# Named Entity Recognition
entities = ner("Ali pergi Pavilion KL semalam")
print(entities)
# [{'text': 'Ali', 'type': 'PERSON', 'start': 0, 'end': 3},
#  {'text': 'Pavilion KL', 'type': 'LOCATION', 'start': 10, 'end': 21}]

# Language detection
lang = detect_language("Eh jom makan, I'm hungry gila")
print(lang)
# {'language': 'manglish', 'confidence': 0.87}
```

## Features (51 Modules)

### Text Processing
- **normalize** — Expand shortforms (638+ mappings: nk→nak, mcm→macam, sbb→sebab)
- **clean** — Remove URLs, mentions, repeated chars, HTML
- **formalize** — Convert informal to formal Malay (aku→saya, ko→anda)
- **tokenize** — Malaysian-aware tokenizer (handles URLs, hashtags, emoticons)
- **stemmer** — Malay stemmer with nasal assimilation (250+ roots)
- **segment** — Sentence segmentation for code-switched text
- **spelling** — Spell checking with Malaysian dictionary

### Analysis
- **sentiment** — Sentiment analysis with aspect-based (food, service, price, etc.)
- **emotion** — 8 emotion categories (happy, sad, angry, fear, surprise, disgust, love, neutral)
- **sarcasm** — Sarcasm detection for Malaysian text
- **hate_speech** — Hate speech detection (6 categories, severity levels)
- **intent** — 8 intent types (question, request, complaint, greeting, opinion, statement, command, offer)
- **topic** — 12 topic classification (food, politics, sports, tech, education, etc.)
- **stance** — Stance detection (support/oppose/neutral)
- **profanity** — Profanity detection with leetspeak evasion handling

### Entity & Structure
- **ner** — Named Entity Recognition (11 types: PERSON, ORG, LOC, PRODUCT, EVENT, MONEY, PHONE, EMAIL, DATE, TIME, PERCENT)
- **pos_tag** — Part-of-speech tagging (15 tags)
- **dependency** — Dependency parsing (SVO extraction)
- **coreference** — Pronoun resolution with Malaysian gender heuristics
- **keywords** — Keyword extraction (frequency, RAKE, TF-IDF, TextRank)

### Language Detection & Code-Switching
- **language** — Language identification (Malay/English/Manglish/Mixed)
- **code_switching** — Code-switching point detection, switch ratio, segmentation by language
- **dialect** — 6 Malay dialects (Standard, Kelantan, Terengganu, N9, Kedah, Sarawak, Sabah) with normalization

### Semantic & Similarity
- **similarity** — Text similarity (Jaccard, cosine, overlap, semantic)
- **embeddings** — Word2Vec/FastText trained on Malaysian social media (518 vocab, 100d)
- **augmentation** — Text augmentation (synonym replacement, shortform variation)

### Generation & Understanding
- **translation** — Rule-based BM↔EN translation (1000+ word pairs, phrase translation)
- **summarization** — Extractive summarization using TextRank algorithm
- **text_generation** — N-gram based text generation and autocomplete
- **qa** — Extractive question answering with TF-IDF retrieval
- **discourse** — Argument mining and fallacy detection
- **ocr_normalize** — OCR text correction for Malaysian documents

### Preprocessing & Utilities
- **normalizer** — Advanced normalization (money, dates, times, elongated text)
- **dictionary** — Malay-English dictionary lookup
- **similarity** — Multiple similarity metrics
- **pipeline** — Chain multiple modules together
- **calibration** — Confidence scoring for predictions
- **hybrid_ml** — Feature extraction + logistic classifier
- **evaluate** — Model evaluation and regression tracking
- **cache** — LRU caching for performance
- **profiler** — Performance benchmarking tools
- **tuning** — Hyperparameter tuning and threshold optimization

### Integration
- **spacy_integration** — Custom spaCy Language class and pipeline components
- **rest_api** — FastAPI REST API with rate limiting and CORS
- **langchain_tool** — LangChain tool wrappers
- **CLI** — Command-line interface with subcommands

## Performance

- **23,000+ texts/sec** sentiment analysis throughput
- **<0.5s** import time for core modules
- **Zero dependencies** for core text processing
- LRU caching on heavy operations (stemmer, normalize, sentiment, language detection)
- Lazy loading — only imports what you use
- Pre-compiled regex patterns across 6 modules

## Comparison with Malaya

| Feature | manglish-nlp | Malaya |
|---------|-------------|--------|
| Core dependencies | None | TensorFlow/PyTorch required |
| Import time | <0.5s | 10-30s |
| Manglish-first | ✅ Built for informal MY text | Formal BM focus |
| Modules | 51 | ~40 |
| Throughput | 23k+ texts/sec | Varies (GPU recommended) |
| Python support | 3.8-3.12 | 3.8+ |
| Aspect sentiment | ✅ | ❌ |
| Code-switching detection | ✅ | ❌ |
| Hate speech detection | ✅ | Limited |
| Discourse analysis | ✅ | ❌ |
| OCR normalization | ✅ | ❌ |
| Translation (rule-based) | ✅ | ❌ |
| Text generation | ✅ | ❌ |

Both are solid choices. Malaya excels at formal Bahasa Melayu with deep learning models. manglish-nlp is optimized for informal, code-switched Malaysian text with minimal overhead and advanced NLP features.

## CLI Usage

```bash
# Full analysis
manglish analyze "Weh best gila makanan dia!"

# Sentiment
manglish sentiment "Teruk la service kat sini"

# Normalize shortforms
manglish normalize "xpe la bro aku otw"

# Translate
manglish translate "Aku nak pergi makan" --to en

# NER
manglish ner "Ahmad kerja kat Google Malaysia"

# Summarize file
manglish summarize --file article.txt

# Run benchmarks
manglish benchmark

# Profile performance
manglish profile "Sample text here"
```

## REST API

```bash
# Start API server
uvicorn manglish_nlp.rest_api:app --port 8000

# Or with Docker
docker-compose up -d
```

Endpoints:
- `POST /analyze` — Full analysis
- `POST /sentiment` — Sentiment only
- `POST /normalize` — Normalize text
- `POST /translate` — Translate text
- `POST /ner` — Named entities
- `POST /pos` — POS tags
- `POST /summarize` — Summarize text
- `POST /batch` — Batch process multiple texts
- `GET /health` — Health check
- `GET /modules` — List available modules

## Testing

```bash
# Run all tests (900+ tests)
python -m pytest tests/ -q

# Run specific test file
python -m pytest tests/test_sentiment.py -v

# Run with coverage
python -m pytest tests/ --cov=manglish_nlp

# Run heavy tests (requires gensim)
RUN_HEAVY_TESTS=1 python -m pytest tests/test_word_embeddings.py -v
```

## Documentation

Full documentation available at [manglish-nlp.readthedocs.io](https://manglish-nlp.readthedocs.io)

Includes:
- Module reference for all 51 modules
- API documentation with examples
- Performance benchmarks
- Comparison with Malaya
- Contributing guide
- Changelog

## Contributing

Contributions welcome! Areas where help is needed:

1. **More training data** — Manglish text samples from social media
2. **Dialect support** — More regional variants and normalization rules
3. **Benchmarks** — Comparative benchmarks on Malaysian NLP datasets
4. **Documentation** — More usage examples and tutorials

```bash
git clone https://github.com/ZafranYusof/manglish-nlp.git
cd manglish-nlp
pip install -e ".[all]"
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use manglish-nlp in your research, please cite:

```bibtex
@software{manglish_nlp,
  author = {Zafran},
  title = {manglish-nlp: Full NLP toolkit for Malaysian Manglish},
  year = {2026},
  url = {https://github.com/ZafranYusof/manglish-nlp},
  version = {3.0.0}
}
```

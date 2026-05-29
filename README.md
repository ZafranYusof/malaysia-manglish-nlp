# manglish-nlp

[![PyPI version](https://img.shields.io/pypi/v/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![Python versions](https://img.shields.io/pypi/pyversions/manglish-nlp.svg)](https://pypi.org/project/manglish-nlp/)
[![Tests](https://img.shields.io/github/actions/workflow/status/zafran/manglish-nlp/test.yml?label=tests)](https://github.com/zafran/manglish-nlp/actions)
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
pip install manglish-nlp[embeddings]     # Sentence embeddings
pip install manglish-nlp[spacy]          # spaCy integration
pip install manglish-nlp[all]            # Everything
```

## Quick Start

```python
from manglish_nlp import sentiment, normalize, ner

# Sentiment analysis
result = sentiment.analyze("Weh best gila makanan dia!")
print(result)  # {'label': 'positive', 'score': 0.94}

# Text normalization
clean = normalize.text("xpe la bro, aku ok je")
print(clean)  # "takpe la bro, aku ok je"

# Named Entity Recognition
entities = ner.extract("Zafran pergi Pavilion KL semalam")
print(entities)  # [('Zafran', 'PERSON'), ('Pavilion KL', 'LOCATION')]
```

## Features (51 Modules)

### Text Processing
- **normalize** — Manglish text normalization (slang, abbreviations, spelling)
- **tokenize** — Malaysian-aware tokenization
- **segment** — Sentence segmentation for code-switched text
- **stemmer** — Malay stemmer with prefix/suffix handling
- **lemmatize** — Context-aware lemmatization
- **syllable** — Malay syllable splitting
- **phonetic** — Phonetic encoding for Malay words

### Analysis
- **sentiment** — Sentiment analysis (positive/negative/neutral)
- **emotion** — Emotion detection (8 categories)
- **sarcasm** — Sarcasm detection for Malaysian text
- **toxicity** — Toxicity and hate speech detection
- **intent** — Intent classification
- **topic** — Topic modeling and classification

### Entity & Structure
- **ner** — Named Entity Recognition (PERSON, ORG, LOC, etc.)
- **pos_tag** — Part-of-speech tagging
- **dependency** — Dependency parsing
- **chunker** — Noun/verb phrase chunking
- **coref** — Coreference resolution
- **relation** — Relation extraction

### Language Detection & Code-Switching
- **lang_detect** — Language identification (BM/EN/Manglish/others)
- **code_switch** — Code-switching point detection
- **script_detect** — Script detection (Latin, Jawi, etc.)

### Semantic
- **similarity** — Text similarity scoring
- **paraphrase** — Paraphrase detection
- **embeddings** — Text embeddings (word & sentence level)
- **keyword** — Keyword extraction
- **summarize** — Extractive & abstractive summarization
- **qa** — Question answering

### Generation & Transformation
- **translate** — BM↔EN translation
- **augment** — Text augmentation for training data
- **backtranslate** — Back-translation augmentation
- **fill_mask** — Masked language model predictions
- **generate** — Text generation

### Preprocessing
- **clean** — HTML/URL/emoji removal and cleaning
- **dedup** — Near-duplicate detection
- **spell** — Spell checking with Malaysian dictionary
- **number** — Number/currency normalization
- **date_parse** — Malaysian date format parsing
- **emoji_sentiment** — Emoji sentiment mapping

### Social Media
- **hashtag** — Hashtag segmentation
- **mention** — @mention extraction and resolution
- **url_expand** — URL expansion and metadata
- **chat_normalize** — Chat/SMS abbreviation expansion

### Corpus & Resources
- **stopwords** — Malaysian stopword lists
- **dictionary** — Malay-English dictionary lookup
- **wordnet** — Malaysian WordNet interface
- **collocation** — Collocation detection
- **frequency** — Word frequency lists

### Pipeline & Utilities
- **pipeline** — Composable NLP pipeline
- **batch** — Batch processing with progress
- **cache** — Result caching layer
- **benchmark** — Performance benchmarking tools
- **export** — Export to common formats (CoNLL, JSON, CSV)

## Performance

- **23,000+ texts/sec** throughput on standard hardware
- **<0.5s** import time for core modules
- **Zero dependencies** for core text processing
- Lazy loading — only imports what you use

## Comparison with Malaya

| Feature | manglish-nlp | Malaya |
|---------|-------------|--------|
| Core dependencies | None | TensorFlow/PyTorch required |
| Import time | <0.5s | 10-30s |
| Manglish-first | ✅ | Formal BM focus |
| Modules | 51 | ~40 |
| Throughput | 23k+ texts/sec | Varies (GPU recommended) |
| Python support | 3.8-3.12 | 3.8+ |

Both are solid choices. Malaya excels at formal Bahasa Melayu with deep learning models. manglish-nlp is optimized for informal, code-switched Malaysian text with minimal overhead.

## CLI Usage

```bash
# Analyze sentiment
manglish sentiment "Best gila movie tu!"

# Normalize text
manglish normalize "xpe la bro aku otw"

# Detect language
manglish lang-detect "Eh jom makan, I'm hungry gila"
```

## Contributing

Contributions welcome! Areas where help is needed:

1. **More training data** — Manglish text samples from social media
2. **Dialect support** — Kelantan, Terengganu, Sabah/Sarawak variants
3. **Benchmarks** — Comparative benchmarks on Malaysian datasets
4. **Documentation** — Usage examples and tutorials

```bash
git clone https://github.com/zafran/manglish-nlp.git
cd manglish-nlp
pip install -e ".[all]"
python -m pytest tests/ -q
```

## License

MIT — see [LICENSE](LICENSE) for details.

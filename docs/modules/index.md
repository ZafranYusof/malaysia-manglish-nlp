# Module Reference

**51 production-ready NLP modules for Malaysian text  -  zero configuration, one import.**

---

## Overview

malaysian-manglish-nlp organises every module into eight functional groups. All follow a consistent API: `mnlp.<module>(text, **options)`. Core modules run with zero external dependencies; advanced/generation modules need the optional `[ml]` extra.

```
pip install malaysian-manglish-nlp        # core only (text processing, analysis, extraction)
pip install malaysian-manglish-nlp[ml]    # + transformer models (generation, advanced)
pip install malaysian-manglish-nlp[all]   # everything including spaCy & LangChain
```

---

## Module Grid

### Text Processing

<div class="grid cards" markdown>

- **[`normalize`](text-processing.md#normalize)**  -  Informal → standard spelling (12k+ shortform mappings)
- **[`clean`](text-processing.md#clean)**  -  Strip noise: URLs, mentions, emojis, repeated chars
- **[`formalize`](text-processing.md#formalize)**  -  Casual Manglish → formal Bahasa Melayu
- **[`tokenize`](text-processing.md#tokenizer)**  -  Malaysian-aware tokeniser (word / sentence / subword)
- **[`stemmer`](text-processing.md#stemmer)**  -  Rule-based Malay affix stripping (me-/ber-/di-/-kan/-an/-i)
- **[`segment`](text-processing.md#segment)**  -  Split concatenated text, hashtags, URLs
- **[`spelling`](text-processing.md#spelling)**  -  Context-aware spelling correction

</div>

### Analysis

<div class="grid cards" markdown>

- **[`sentiment`](analysis.md#sentiment)**  -  Positive / negative / neutral with aspect-based option
- **[`emotion`](analysis.md#emotion)**  -  8 emotion labels + intensity scoring
- **[`language`](analysis.md#language)**  -  Language & dialect detection (BM, EN, Manglish, Kelantan, Kedah…)
- **[`profanity`](analysis.md#profanity)**  -  Profanity filter with severity levels & censor modes
- **[`sarcasm`](analysis.md#sarcasm)**  -  Sarcasm / irony detection with cue explanation

</div>

### Extraction

<div class="grid cards" markdown>

- **[`ner`](extraction.md#ner)**  -  7 entity types including Malaysian names, places, currency
- **[`pos`](extraction.md#pos)**  -  UD-based POS tagging adapted for Malay grammar
- **[`dependency`](extraction.md#dependency)**  -  Dependency parsing with tree visualisation
- **[`coreference`](extraction.md#coreference)**  -  Pronoun & mention resolution
- **[`keywords`](extraction.md#keywords)**  -  TF-IDF / TextRank / YAKE keyword extraction

</div>

### Advanced

<div class="grid cards" markdown>

- **[`code_switching`](advanced.md#code_switching)**  -  Detect language switch points & patterns
- **[`intent`](advanced.md#intent)**  -  8 intent categories + slot filling for chatbots
- **[`topic`](advanced.md#topic)**  -  Topic classification & unsupervised topic modelling
- **[`hate_speech`](advanced.md#hate_speech)**  -  3 severity levels across 6 target categories
- **[`stance`](advanced.md#stance)**  -  Support / oppose / neutral stance detection
- **[`discourse`](advanced.md#discourse)**  -  Rhetorical relation parsing (cause, contrast, concession…)
- **[`coreference`](advanced.md#coreference)**  -  Cross-sentence entity linking

</div>

### Generation

<div class="grid cards" markdown>

- **[`translation`](generation.md#translation)**  -  BM ↔ EN ↔ Manglish with entity preservation
- **[`summarization`](generation.md#summarization)**  -  Extractive & abstractive summaries
- **[`text_generation`](generation.md#text_generation)**  -  Controlled text generation (style, format, temperature)
- **[`qa`](generation.md#qa)**  -  Extractive & generative QA with conversational sessions

</div>

### Data & Embeddings

<div class="grid cards" markdown>

- **[`word_embeddings`](data.md#word_embeddings)**  -  300-dim Word2Vec trained on 10M+ Malaysian texts
- **[`embeddings`](data.md#embeddings)**  -  768-dim sentence/document embeddings (fast & accurate modes)
- **[`similarity`](data.md#similarity)**  -  Cosine / Jaccard / WMD semantic similarity
- **[`augmentation`](data.md#augmentation)**  -  6 augmentation strategies for Malaysian text
- **[`dictionary`](data.md#dictionary)**  -  Lexical resource with definitions, slang, frequency data
- **[`spelling`](data.md#spelling)**  -  Context-aware spelling correction with informal preservation

</div>

### Tools & Utilities

<div class="grid cards" markdown>

- **[`ocr_normalize`](tools.md#ocr_normalize)**  -  Fix OCR artefacts in Malay documents
- **[`pipeline`](tools.md#pipeline)**  -  Chain modules into reusable, serialisable workflows
- **[`calibration`](tools.md#calibration)**  -  Calibrate confidence scores (Platt / isotonic / temperature)
- **[`evaluate`](tools.md#evaluate)**  -  Accuracy, F1, cross-validation, error analysis
- **[`hybrid_ml`](tools.md#hybrid_ml)**  -  Rule-first routing with ML fallback
- **[`tuning`](tools.md#tuning)**  -  Grid / random / Bayesian hyperparameter search
- **[`profiler`](tools.md#profiler)**  -  Latency, memory, and throughput benchmarking
- **[`cache`](tools.md#cache)**  -  Memory / disk / Redis caching with TTL & warm-up

</div>

### Integrations

<div class="grid cards" markdown>

- **[`spacy`](integrations.md#spacy_integration)**  -  Drop-in spaCy pipeline components
- **[`rest_api`](integrations.md#rest-api)**  -  FastAPI server with Swagger docs
- **[`cli`](integrations.md#cli)**  -  Full CLI for every module, file processing, pipelines
- **[`langchain`](integrations.md#langchain)**  -  LangChain tool wrappers for agent usage

</div>

---

## Universal API Pattern

Every module follows the same call convention:

```python
import malaysian_manglish_nlp as mnlp

# Single text
result = mnlp.<module>(text)

# With options
result = mnlp.<module>(text, lang="ms", detailed=True)

# Batch (list input → list output)
results = mnlp.<module>(["text1", "text2", "text3"])
```

---

## Dependency Tiers

```
┌──────────────────────────────────────────────────────┐
│  Tier 0  -  Core (zero external deps)                  │
│  normalize, clean, tokenize, stem, segment, spelling │
├──────────────────────────────────────────────────────┤
│  Tier 1  -  Analysis (lightweight models)              │
│  sentiment, ner, pos, keywords, language, profanity  │
├──────────────────────────────────────────────────────┤
│  Tier 2  -  Advanced (optional ML)                     │
│  code_switching, intent, topic, hate_speech, stance  │
├──────────────────────────────────────────────────────┤
│  Tier 3  -  Generation (requires [ml])                 │
│  translate, summarize, generate, qa, embeddings      │
├──────────────────────────────────────────────────────┤
│  Tier 4  -  Integrations (requires [spacy]/[langchain])│
│  spacy, rest_api, langchain                          │
└──────────────────────────────────────────────────────┘
```

!!! info "Optional Dependencies"
    Core modules have **zero** external dependencies. Install extras only when needed:
    ```
    pip install malaysian-manglish-nlp[ml]       # Tier 2-3
    pip install malaysian-manglish-nlp[spacy]    # spaCy integration
    pip install malaysian-manglish-nlp[langchain]# LangChain tools
    pip install malaysian-manglish-nlp[all]      # Everything
    ```

# Module Overview

manglish-nlp ships with **51 modules** organized into 8 categories. All modules work out of the box with zero configuration.

---

## Categories

| Category | Modules | Description |
|----------|---------|-------------|
| [Text Processing](text-processing.md) | 6 | Normalize, clean, tokenize, stem, segment |
| [Analysis](analysis.md) | 5 | Sentiment, emotion, language detection, profanity, sarcasm |
| [Extraction](extraction.md) | 4 | NER, POS tagging, keywords, dependency parsing |
| [Advanced](advanced.md) | 7 | Code-switching, intent, topic, hate speech, stance, coreference, discourse |
| [Generation](generation.md) | 4 | Text generation, translation, summarization, QA |
| [Data & Embeddings](data.md) | 6 | Word embeddings, similarity, augmentation, dictionary, spelling |
| [Tools & Utilities](tools.md) | 8 | OCR, pipeline, calibration, evaluation, ML, tuning, profiler, cache |
| [Integrations](integrations.md) | 3 | spaCy, FastAPI, CLI |

---

## Quick Usage Pattern

All modules follow a consistent API pattern:

```python
import manglish_nlp as mnlp

# Direct function call
result = mnlp.<module_name>(text)

# With options
result = mnlp.<module_name>(text, lang="ms", detailed=True)

# Batch processing
results = mnlp.<module_name>(list_of_texts)
```

---

## Module Dependencies

```
┌─────────────────────────────────────────────┐
│              Core (zero deps)                │
│  normalize, clean, tokenize, stem, segment  │
├─────────────────────────────────────────────┤
│           Analysis & Extraction             │
│  sentiment, ner, pos, keywords, language    │
├─────────────────────────────────────────────┤
│         Advanced (optional ML)              │
│  code_switching, intent, topic, hate_speech │
├─────────────────────────────────────────────┤
│        Generation (requires [ml])           │
│  translate, summarize, generate, qa         │
└─────────────────────────────────────────────┘
```

!!! info "Optional Dependencies"
    Core modules (text processing, basic analysis) have zero external dependencies. Advanced and generation modules may require the `[ml]` extra: `pip install manglish-nlp[ml]`

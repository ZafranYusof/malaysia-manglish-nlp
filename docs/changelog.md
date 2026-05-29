# Changelog

All notable changes to `manglish-nlp`. Format follows [Keep a Changelog](https://keepachangelog.com).

---

## [Unreleased]

_Nothing yet._

---

## [3.0.0]  -  2026-05-29

The big one. Major architecture overhaul, new modules, massive speedups.

### Added

- **51 NLP modules** (up from 28 in v2)
- `emotion()`  -  7-class emotion detection (joy, anger, sadness, fear, surprise, disgust, love)
- `code_switching.detect_switches()`  -  token-level code-switch boundary detection
- `hate_speech.detect_hate_speech()`  -  toxicity + hate category detection with severity levels
- `stance.detect_stance()`  -  for/against/neutral stance detection toward targets
- `intent.classify_intent()`  -  7-class intent classification
- `topic.classify_topic()`  -  10-class topic classification with subtopics
- `qa.answer()`  -  extractive question answering
- `text_generation.generate()`  -  Malay/Manglish text generation
- `pipeline()`  -  chain multiple NLP ops into reusable pipelines
- `augment()`  -  data augmentation with synonym/insert/swap/delete strategies
- `load_word2vec()` / `load_fasttext()`  -  pre-trained embedding loaders
- Typed exceptions: `ManglishNLPError`, `InputError`, `ModelError`, `LanguageError`, `PipelineError`
- Full type annotations across entire codebase
- Batch inference support for all models
- Auto-download for model files on first use

### Changed

- **4� -  faster** sentiment inference (12.4ms → 3.1ms per text)
- **5� -  faster** cold import (2.1s → 0.42s) via lazy loading
- **6� -  smaller** memory footprint (1.2GB → 180MB)
- Sentiment model retrained on 50k+ annotated Manglish samples
- NER model upgraded to span-based architecture (was token-level)
- Normalization now handles 2000+ slang mappings (was 400)
- `detect_language()` now supports ms/en/zh/ta with confidence scores
- All functions now return structured `dict` instead of raw values
- Minimum Python version raised to 3.10 (was 3.8)

### Fixed

- Sentiment misclassifying negated sarcasm ("not bad lah" was negative)
- Tokenizer splitting contractions incorrectly ("taknak" → now "tak" + "nak")
- NER missing entities in mixed-script text
- `normalize()` mangling Chinese characters in mixed text
- Memory leak in repeated model inference calls
- Race condition in concurrent pipeline execution

### Removed

- Python 3.8 and 3.9 support
- Legacy `sentiment_raw()` function (use `sentiment()` with `raw_score`)
- `load_model()` function (models auto-load on first use)
- TensorFlow dependency (all models now ONNX/PyTorch)

### Breaking Changes

| Before (v2) | After (v3) | Migration |
|-------------|------------|-----------|
| `sentiment("text")` → `str` | `sentiment("text")` → `dict` | Access `result["sentiment"]` |
| `load_model("sentiment")` | Auto-loaded | Remove `load_model()` calls |
| `normalize(text, slang=True)` | `normalize(text, aggressive=False)` | Rename parameter |
| `detect_lang(text)` | `detect_language(text)` | Rename function |
| Python 3.8+ | Python 3.10+ | Upgrade Python |

---

## [2.0.0]  -  2025-08-15

First stable release with core NLP functionality.

### Added

- 28 NLP modules covering core text processing
- `sentiment()`  -  3-class sentiment analysis (positive/negative/neutral)
- `detect_language()`  -  basic Malay/English detection
- `normalize()`  -  informal-to-standard text normalization
- `clean()`  -  text cleaning (URLs, mentions, whitespace)
- `formalize()`  -  casual to formal Malay conversion
- `tokenize()`  -  Manglish-aware tokenization
- `stem_word()`  -  Malay morphological stemming
- `ner_tag()`  -  basic NER (PER/ORG/LOC)
- `pos_tag()`  -  POS tagging (Universal Dependencies)
- `extract_keywords()`  -  TF-IDF keyword extraction
- `segment()`  -  sentence segmentation
- `similarity()`  -  semantic text similarity
- `correct()`  -  basic spell checking
- `summarization.summarize()`  -  extractive summarization
- `translation.translate()`  -  Malay-English translation
- Pre-trained models bundled in package
- Comprehensive test suite (612 tests)
- Documentation site with API reference

### Changed

- Migrated from rule-based to neural models for sentiment, NER, POS
- Tokenizer rewritten to handle Manglish contractions properly
- 3� -  speedup across all modules vs v1 (rule-based → neural inference)
- Package renamed from `malaysia-nlp` to `manglish-nlp`
- Minimum Python raised to 3.8

### Fixed

- Tokenizer failing on text with emojis
- Sentiment model biased toward positive for short texts
- NER confusing "Malaysia" as ORG instead of LOC
- Stemmer producing incorrect roots for loanwords
- Memory usage spiking on long text inputs

### Removed

- Rule-based fallback models (neural models are now default)
- `sentiment_batch()` function (use list comprehension)
- Bundled training data (moved to separate `manglish-nlp-data` package)

### Breaking Changes

| Before (v1) | After (v2) | Migration |
|-------------|------------|-----------|
| `from malaysia_nlp import ...` | `from manglish_nlp import ...` | Update imports |
| `sentiment("text")` → `float` | `sentiment("text")` → `str` | Returns label now |
| `ner("text")` → `list[str]` | `ner_tag("text")` → `list[dict]` | Function renamed, returns dicts |
| Python 3.7+ | Python 3.8+ | Upgrade Python |

---

## [1.0.0]  -  2025-01-10

Initial release. Rule-based Malay NLP toolkit.

### Added

- Basic sentiment analysis (lexicon-based, Malay)
- Simple tokenizer for Malay text
- Rule-based normalizer (common abbreviations)
- Keyword extraction (frequency-based)
- Sentence segmentation
- Basic spell checker (edit distance)
- 89 unit tests
- README with quickstart

### Known Limitations

- Rule-based only, no neural models
- Poor accuracy on informal/Manglish text (~72% sentiment)
- No code-switching support
- No NER or POS tagging
- Slow import (loads entire dictionary at startup)
- Python 3.7+ only

---

## Version Support

| Version | Status | End of Life |
|---------|--------|-------------|
| v3.x | **Active**  -  current, receiving features + fixes |  -  |
| v2.x | **Maintenance**  -  security fixes only | 2026-12-31 |
| v1.x | **EOL**  -  no longer supported | 2025-08-15 |

---

## Migration Guide: v2 → v3

### Step 1: Update Python

```bash
python --version  # Must be 3.10+
```

### Step 2: Update package

```bash
pip install --upgrade manglish-nlp
```

### Step 3: Update sentiment calls

```python
# Before (v2)
label = sentiment("Best gila!")
print(label)  # "positive"

# After (v3)
result = sentiment("Best gila!")
print(result["sentiment"])  # "positive"
print(result["score"])      # 0.94
```

### Step 4: Remove load_model calls

```python
# Before (v2)
from manglish_nlp import load_model, sentiment
model = load_model("sentiment")
result = model.predict("text")

# After (v3)
from manglish_nlp import sentiment
result = sentiment("text")  # Auto-loads model
```

### Step 5: Update function names

```python
# Before
from manglish_nlp import detect_lang, ner
lang = detect_lang("text")
entities = ner("text")

# After
from manglish_nlp import detect_language, ner_tag
lang = detect_language("text")
entities = ner_tag("text")
```

### Step 6: Test everything

```bash
pytest tests/
```

If tests pass, you're migrated. If not, check the breaking changes table above.

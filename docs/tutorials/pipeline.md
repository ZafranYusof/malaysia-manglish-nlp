# Pipeline Usage

**Chain multiple NLP modules into a single call — custom workflows, batch processing, and serializable pipelines.**

---

## Why pipelines?

Running 5 separate modules on the same text means 5 function calls, 5 result dicts to merge, and repeated boilerplate. Pipelines let you define a sequence of modules once and run them in a single call. Results are automatically merged into one structured output.

---

## Load module

```python
import manglish_nlp as mnlp

# Default pipeline (normalize + sentiment + language + emotion)
result = mnlp.pipeline("gila best mkn dia")
print(result)
# {
#     'original': 'gila best mkn dia',
#     'normalized': 'gila best makan dia',
#     'sentiment': {'label': 'positive', 'score': 0.92},
#     'language': {'primary': 'manglish', ...},
#     'emotion': {'primary': 'joy', 'score': 0.85},
# }
```

---

## Basic usage

### Default pipeline

The default pipeline runs 4 modules: `normalize`, `sentiment`, `language`, `emotion`.

```python
result = mnlp.pipeline("Weh best gila kedai tu!")

print(result['original'])
# Weh best gila kedai tu!

print(result['normalized'])
# Weh best gila kedai tu!

print(result['sentiment']['label'])
# positive

print(result['emotion']['primary'])
# joy
```

### Custom steps

Choose which modules to run:

```python
result = mnlp.pipeline(
    "Ahmad kerja kat Petronas KL",
    steps=['normalize', 'ner', 'pos']
)

print(result['ner'])
# [('Ahmad', 'PERSON'), ('Petronas', 'ORG'), ('KL', 'LOCATION')]

print(result['pos'])
# [('Ahmad', 'PROPN'), ('kerja', 'VERB'), ...]
```

---

## Available steps

| Step | Module | What it does |
|------|--------|-------------|
| `normalize` | `mnlp.normalize()` | Expand shortforms |
| `sentiment` | `mnlp.sentiment()` | Sentiment analysis |
| `language` | `mnlp.detect_language()` | Language detection |
| `emotion` | `mnlp.detect_emotion()` | Emotion detection |
| `profanity` | `mnlp.detect_profanity()` | Profanity filter |
| `dialect` | `mnlp.detect_dialect()` | Dialect detection |
| `sarcasm` | `mnlp.detect_sarcasm()` | Sarcasm detection |
| `tokenize` | `mnlp.tokenize()` | Tokenization |
| `stem` | `mnlp.stem()` | Malay stemming |
| `pos` | `mnlp.pos_tag()` | POS tagging |
| `ner` | `mnlp.ner_tag()` | Named entities |
| `segment` | `mnlp.segment()` | Text segmentation |
| `formalize` | `mnlp.formalize()` | Formal BM |
| `clean` | `mnlp.clean()` | Noise removal |
| `keywords` | `mnlp.extract_keywords()` | Keyword extraction |
| `correct` | `mnlp.correct()` | Spelling correction |
| `dependency` | `mnlp.parse_dependencies()` | Dependency parsing |
| `all` | everything | Run all available steps |

---

## Run all modules

```python
result = mnlp.pipeline("Weh best gila kedai tu!", steps=['all'])

# All module results in one dict:
# result['original']
# result['normalized']
# result['sentiment']
# result['language']
# result['emotion']
# result['profanity']
# result['dialect']
# result['sarcasm']
# result['pos']
# result['ner']
# result['keywords']
# ... and more
```

!!! warning "Performance with 'all'"
    Running all modules takes ~5ms per text. For high-throughput scenarios, only include the modules you need.

---

## Normalize first

By default, text is normalized before other modules run. This improves accuracy for sentiment, NER, and language detection on informal text.

```python
# Normalize first (default)
result = mnlp.pipeline("xpe la best gila mkn dia", normalize_first=True)
# normalized text used for sentiment, language, emotion

# Skip normalization
result = mnlp.pipeline("xpe la best gila mkn dia", normalize_first=False)
# raw text used for all modules
```

---

## The analyze function

`mnlp.analyze()` is a shortcut for the full pipeline:

```python
result = mnlp.analyze("Weh best gila kedai tu, tapi service slow sikit")

# Returns comprehensive analysis:
# - normalized text
# - sentiment (with aspect detection for mixed)
# - language detection
# - POS tags
# - named entities
# - emotion
# - keywords
```

---

## Batch pipeline

Process multiple texts through the same pipeline:

```python
texts = [
    "Best gila movie tu!",
    "Teruk la service dia",
    "Ahmad kerja kat Petronas KL",
    "Ambo nok make nasi kerabu",
]

results = mnlp.batch_pipeline(
    texts,
    steps=['normalize', 'sentiment', 'ner']
)

for text, result in zip(texts, results):
    sent = result['sentiment']['label']
    entities = result.get('ner', [])
    print(f"{text[:30]:30s} → {sent:8s} | {entities}")

# Best gila movie tu!              → positive | []
# Teruk la service dia             → negative | []
# Ahmad kerja kat Petronas KL      → neutral  | [('Ahmad', 'PERSON'), ...]
# Ambo nok make nasi kerabu        → neutral  | []
```

---

## Custom pipeline workflow

Build a custom analysis function:

```python
def analyze_review(text):
    """Custom review analysis pipeline."""
    result = mnlp.pipeline(text, steps=['normalize', 'sentiment', 'emotion', 'keywords'])
    
    return {
        'text': result['normalized'],
        'sentiment': result['sentiment']['label'],
        'confidence': result['sentiment']['score'],
        'emotion': result['emotion']['primary'],
        'keywords': result['keywords'][:5],
        'is_positive': result['sentiment']['label'] == 'positive',
    }

# Use on a batch of reviews
reviews = [
    "Best gila makanan dia, sedap!",
    "Mahal sangat, tak berbaloi",
    "Service ok, food average je",
]

for review in reviews:
    analysis = analyze_review(review)
    print(f"{analysis['sentiment']:8s} | {analysis['emotion']:12s} | {analysis['text'][:30]}")
```

---

## Content moderation pipeline

```python
def moderate(text):
    """Content moderation pipeline."""
    result = mnlp.pipeline(text, steps=['normalize', 'profanity', 'sarcasm', 'sentiment'])
    
    flags = []
    if result.get('profanity', {}).get('is_profanity'):
        flags.append('profanity')
    if result.get('sarcasm', {}).get('is_sarcastic'):
        flags.append('sarcasm')
    if result.get('sentiment', {}).get('label') == 'negative':
        flags.append('negative')
    
    return {
        'approved': len(flags) == 0,
        'flags': flags,
        'clean_text': result['normalized'],
    }
```

---

## CLI usage

```bash
# Full analysis (default pipeline)
$ mnlp analyze "Weh best gila kedai tu"

# With JSON output
$ mnlp analyze "Sedap nasi lemak" --json

# Pipe input
$ echo "Best gila!" | mnlp analyze

# Specific modules
$ mnlp sentiment "Best gila!" && mnlp ner "Ahmad kat KL"

# Benchmark
$ mnlp benchmark
```

---

## Performance

| Pipeline | Latency | Throughput |
|----------|---------|------------|
| Default (4 modules) | ~2ms | 15,000 texts/sec |
| Minimal (2 modules) | ~1ms | 30,000 texts/sec |
| All modules | ~5ms | 6,000 texts/sec |
| Batch (100 texts, default) | ~150ms | 12,000 texts/sec |

!!! tip "Optimization"
    - Only include modules you need
    - Use batch_pipeline for multiple texts
    - Normalize first improves accuracy but adds ~0.2ms

---

## See also

- [REST API](rest-api.md) — serve pipelines over HTTP
- [Normalization](normalization.md) — understand the normalize step
- [Sentiment Analysis](sentiment.md) — understand sentiment output
- [API Reference](../api-reference.md#pipeline) — full function signature

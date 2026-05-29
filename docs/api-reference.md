# API Reference

Full function signatures for the most-used manglish-nlp functions.

---

## Text Processing

### `mnlp.normalize(text, **kwargs)`

Normalize informal Manglish spelling to standard form.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text or list of texts |
| `preserve_slang` | `bool` | `False` | Keep common slang terms unchanged |
| `aggressive` | `bool` | `False` | Normalize particles and fillers too |
| `custom_dict` | `dict` | `None` | Custom normalization mappings |

**Returns:** `str | list[str]` — Normalized text

```python
mnlp.normalize("xpe la, aku nk g mkn")
# "takpe la, aku nak pergi makan"
```

---

### `mnlp.clean(text, **kwargs)`

Remove noise from text (URLs, mentions, emojis, repeated chars).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `keep_emoji` | `bool` | `False` | Preserve emoji characters |
| `keep_hashtags` | `bool` | `False` | Keep hashtag text (remove #) |
| `keep_mentions` | `bool` | `False` | Keep @mentions |
| `max_repeat` | `int` | `1` | Max consecutive repeated chars |

**Returns:** `str | list[str]` — Cleaned text

---

### `mnlp.tokenize(text, **kwargs)`

Tokenize text with Malaysian language awareness.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `level` | `str` | `"word"` | Tokenization level: `word`, `sentence`, `subword` |
| `split_particles` | `bool` | `True` | Split particles (la, je, kot) as separate tokens |

**Returns:** `list[str] | list[list[str]]` — Token list

---

### `mnlp.stem(word, **kwargs)`

Stem a Malay word to its root form.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | `str` | required | Word to stem |
| `detailed` | `bool` | `False` | Return affix information |
| `conservative` | `bool` | `False` | Fewer reductions |

**Returns:** `str | dict` — Root word or detailed result

---

### `mnlp.formalize(text, **kwargs)`

Convert casual Manglish to formal Bahasa Melayu.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `level` | `int` | `5` | Formality level (1-5) |
| `keep_english` | `bool` | `False` | Preserve English terms |

**Returns:** `str | list[str]` — Formalized text

---

### `mnlp.segment(text, **kwargs)`

Split unsegmented/concatenated text into words.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Unsegmented text |
| `lang` | `str` | `"auto"` | Language hint: `ms`, `en`, `auto` |
| `scores` | `bool` | `False` | Return confidence scores |

**Returns:** `str | list[tuple]` — Segmented text

---

## Analysis

### `mnlp.sentiment(text, **kwargs)`

Analyze sentiment of Malaysian text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `detailed` | `bool` | `False` | Return all class scores |
| `aspect` | `bool` | `False` | Aspect-based sentiment |
| `cache` | `bool` | `False` | Cache results |

**Returns:** `dict | list[dict]`

```python
# Standard
{'label': 'positive', 'score': 0.94}

# Detailed
{'label': 'positive', 'scores': {'positive': 0.94, 'neutral': 0.04, 'negative': 0.02}}

# Aspect-based
[{'aspect': 'makanan', 'label': 'positive', 'score': 0.92}]
```

---

### `mnlp.emotion(text, **kwargs)`

Detect emotions in text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `multi` | `bool` | `False` | Multi-label detection |
| `intensity` | `bool` | `False` | Include intensity (1-5) |

**Returns:** `dict | list[dict]`

```python
{'primary': 'anger', 'score': 0.88, 'secondary': 'frustration'}
```

---

### `mnlp.language(text, **kwargs)`

Detect language composition.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `per_token` | `bool` | `False` | Per-token language labels |
| `dialect` | `bool` | `False` | Detect regional dialect |

**Returns:** `dict | list[dict]`

```python
{'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

---

### `mnlp.sarcasm(text, **kwargs)`

Detect sarcasm and irony.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `explain` | `bool` | `False` | Include explanation |

**Returns:** `dict | list[dict]`

```python
{'is_sarcastic': True, 'confidence': 0.78, 'cues': ['wah', 'memang']}
```

---

### `mnlp.profanity(text, **kwargs)`

Detect and filter profanity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `censor` | `bool` | `False` | Return censored text |
| `char` | `str` | `"*"` | Censor character |
| `min_severity` | `str` | `"low"` | Minimum severity: `low`, `medium`, `high` |
| `leetspeak` | `bool` | `False` | Detect leetspeak variants |

**Returns:** `dict | str`

---

## Extraction

### `mnlp.ner(text, **kwargs)`

Extract named entities.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `spans` | `bool` | `False` | Return character positions |
| `types` | `list[str]` | `None` | Filter by entity types |
| `threshold` | `float` | `0.5` | Confidence threshold |

**Returns:** `list[tuple] | list[dict]`

```python
[('Siti', 'PERSON'), ('Low Yat Plaza', 'LOCATION')]
```

---

### `mnlp.pos(text, **kwargs)`

Part-of-speech tagging.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `detailed` | `bool` | `False` | Fine-grained morphological features |
| `format` | `str` | `"tuple"` | Output format: `tuple`, `dict` |

**Returns:** `list[tuple] | list[dict]`

---

### `mnlp.keywords(text, **kwargs)`

Extract keywords and key phrases.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `top_k` | `int` | `10` | Number of keywords |
| `scores` | `bool` | `False` | Include relevance scores |
| `method` | `str` | `"textrank"` | Algorithm: `tfidf`, `textrank`, `yake` |
| `ngram_range` | `tuple` | `(1, 3)` | N-gram range for phrases |

**Returns:** `list[str] | list[tuple]`

---

### `mnlp.dependency(text, **kwargs)`

Dependency parsing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input sentence |
| `format` | `str` | `"list"` | Output: `list`, `tree`, `dict` |
| `detailed` | `bool` | `False` | Include POS and head index |
| `visualize` | `bool` | `False` | Generate visualization |
| `output` | `str` | `None` | Visualization output path |

**Returns:** `list[tuple] | str | list[dict]`

---

## Generation

### `mnlp.translate(text, **kwargs)`

Translate between BM, English, and Manglish.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `target` | `str` | required | Target: `en`, `ms`, `ms_formal`, `manglish` |
| `preserve_entities` | `bool` | `False` | Keep names/entities unchanged |
| `informal` | `bool` | `False` | Informal translation style |
| `alternatives` | `int` | `1` | Number of alternative translations |

**Returns:** `str | list[str]`

---

### `mnlp.summarize(text, **kwargs)`

Summarize Malaysian text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `max_length` | `int` | `None` | Max words in summary |
| `ratio` | `float` | `0.3` | Summary length as ratio of original |
| `method` | `str` | `"abstractive"` | Method: `extractive`, `abstractive` |
| `format` | `str` | `"text"` | Output: `text`, `bullets` |
| `lang` | `str` | `"auto"` | Summary language |

**Returns:** `str | list[str]`

---

### `mnlp.generate(prompt, **kwargs)`

Generate Malaysian text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | Generation prompt |
| `max_length` | `int` | `200` | Max tokens to generate |
| `style` | `str` | `"manglish"` | Style: `formal`, `manglish`, `mixed` |
| `temperature` | `float` | `0.8` | Creativity (0.0-1.0) |
| `format` | `str` | `None` | Format: `tweet`, `review`, `caption` |
| `mode` | `str` | `"generate"` | Mode: `generate`, `continue` |

**Returns:** `str`

---

### `mnlp.qa(question, **kwargs)`

Question answering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | Question text |
| `context` | `str` | `None` | Context passage (None for open-domain) |
| `top_k` | `int` | `1` | Number of answers |

**Returns:** `dict | list[dict]`

```python
{'answer': '2002', 'confidence': 0.95, 'span': (46, 50)}
```

---

## Embeddings & Similarity

### `mnlp.embeddings(text, **kwargs)`

Get sentence/document embeddings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | required | Input text |
| `model` | `str` | `"fast"` | Model: `fast`, `accurate` |
| `normalize` | `bool` | `False` | L2-normalize vectors |
| `pooling` | `str` | `"mean"` | Pooling: `mean`, `cls`, `max` |

**Returns:** `numpy.ndarray` — Shape `(dim,)` or `(n, dim)`

---

### `mnlp.similarity(text_a, text_b=None, **kwargs)`

Compute semantic similarity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_a` | `str \| list[str]` | required | First text(s) |
| `text_b` | `str` | `None` | Second text (None for matrix mode) |
| `method` | `str` | `"cosine"` | Method: `cosine`, `jaccard`, `wmd` |
| `mode` | `str` | `"pair"` | Mode: `pair`, `matrix` |
| `candidates` | `list[str]` | `None` | Find most similar from list |
| `top_k` | `int` | `5` | Top results for candidate mode |

**Returns:** `float | numpy.ndarray | list[tuple]`

---

## Pipeline

### `Pipeline(steps)`

Create a reusable processing pipeline.

```python
from manglish_nlp import Pipeline

pipe = Pipeline(['clean', 'normalize', 'sentiment'])
result = pipe("text here")
results = pipe.batch(texts, n_jobs=4)
pipe.save("pipeline.json")
```

| Method | Description |
|--------|-------------|
| `pipe(text)` | Process single text |
| `pipe.batch(texts, n_jobs=1)` | Batch process with parallelism |
| `pipe.save(path)` | Save pipeline config |
| `Pipeline.load(path)` | Load saved pipeline |

---

## Common Patterns

### Batch Processing

All functions accept `list[str]` for batch processing:

```python
texts = ["text1", "text2", "text3"]
results = mnlp.sentiment(texts)       # Returns list of results
results = mnlp.ner(texts)             # Returns list of entity lists
results = mnlp.normalize(texts)       # Returns list of strings
```

### Error Handling

```python
from manglish_nlp.exceptions import (
    ModelNotFoundError,    # ML model not installed
    LanguageError,         # Unsupported language
    InputError,            # Invalid input
    CacheError             # Cache backend issue
)

try:
    result = mnlp.generate("prompt")
except ModelNotFoundError:
    print("Install ML extra: pip install manglish-nlp[ml]")
```

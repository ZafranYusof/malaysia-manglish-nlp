# API Reference

Complete reference for all public functions in `manglish-nlp`.

---

## Core Analysis

### `sentiment(text)`

Analyze sentiment of Manglish/Malay text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text to analyze |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| sentiment | `str` | `"positive"`, `"negative"`, or `"neutral"` |
| score | `float` | Confidence score (0–1) |
| raw_score | `float` | Unnormalized logit score |

**Example:**
```python
from manglish_nlp import sentiment

result = sentiment("Best gila makanan kat sini!")
# {'sentiment': 'positive', 'score': 0.94, 'raw_score': 2.5}

result = sentiment("Boring la movie ni, waste time je")
# {'sentiment': 'negative', 'score': 0.87, 'raw_score': -1.8}
```

---

### `emotion(text)`

Detect fine-grained emotions in text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| emotion | `str` | Primary emotion label |
| scores | `dict[str, float]` | All emotion probabilities |

Supported emotions: `joy`, `anger`, `sadness`, `fear`, `surprise`, `disgust`, `love`.

**Example:**
```python
from manglish_nlp import emotion

result = emotion("Sumpah marah gila aku kat dia!")
# {'emotion': 'anger', 'scores': {'anger': 0.82, 'disgust': 0.09, ...}}
```

---

### `detect_language(text)`

Identify language(s) present in text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| primary | `str` | Dominant language code (e.g., `"ms"`, `"en"`, `"zh"`) |
| languages | `list[dict]` | All detected languages with confidence |
| is_mixed | `bool` | Whether code-switching detected |

**Example:**
```python
from manglish_nlp import detect_language

result = detect_language("I pergi kedai beli nasi lemak")
# {'primary': 'ms', 'languages': [{'lang': 'ms', 'conf': 0.72}, {'lang': 'en', 'conf': 0.28}], 'is_mixed': True}
```

---

## Text Processing

### `normalize(text)`

Normalize informal Manglish text to standard form.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| aggressive | `bool` | `False` | Apply aggressive normalization |

**Returns:** `str` — Normalized text.

**Example:**
```python
from manglish_nlp import normalize

normalize("xpe la, sy ok je")
# 'tidak apa lah, saya okay sahaja'

normalize("xpe la, sy ok je", aggressive=True)
# 'tidak apa, saya baik sahaja'
```

---

### `clean(text)`

Remove noise from text (URLs, mentions, extra whitespace, special chars).

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| remove_urls | `bool` | `True` | Strip URLs |
| remove_mentions | `bool` | `True` | Strip @mentions |
| lowercase | `bool` | `False` | Lowercase output |

**Returns:** `str` — Cleaned text.

**Example:**
```python
from manglish_nlp import clean

clean("@user check this out https://example.com  !!!")
# 'check this out !!!'
```

---

### `formalize(text)`

Convert casual/colloquial text to formal Malay.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `str` — Formal text.

**Example:**
```python
from manglish_nlp import formalize

formalize("aku nak pi kedai jap, nak beli rokok")
# 'saya hendak pergi ke kedai sebentar, hendak membeli rokok'
```

---

### `tokenize(text)`

Tokenize text into words, handling Manglish contractions and particles.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `list[str]` — Token list.

**Example:**
```python
from manglish_nlp import tokenize

tokenize("taknak lah pergi sana")
# ['tak', 'nak', 'lah', 'pergi', 'sana']
```

---

### `stem_word(word)`

Stem a Malay/Manglish word to its root form.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| word | `str` | — | Single word |

**Returns:** `str` — Root/stem form.

**Example:**
```python
from manglish_nlp import stem_word

stem_word("berlari")   # 'lari'
stem_word("memasak")   # 'masak'
stem_word("diperbaiki") # 'baiki'
```

---

### `ner_tag(text)`

Named entity recognition for Malay/Manglish text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `list[dict]`

| Key | Type | Description |
|-----|------|-------------|
| text | `str` | Entity surface form |
| label | `str` | Entity type (`PER`, `ORG`, `LOC`, `MISC`) |
| start | `int` | Start character offset |
| end | `int` | End character offset |

**Example:**
```python
from manglish_nlp import ner_tag

ner_tag("Mahathir pergi Kuala Lumpur semalam")
# [
#   {'text': 'Mahathir', 'label': 'PER', 'start': 0, 'end': 8},
#   {'text': 'Kuala Lumpur', 'label': 'LOC', 'start': 15, 'end': 27}
# ]
```

---

### `pos_tag(text)`

Part-of-speech tagging.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `list[tuple[str, str]]` — (word, tag) pairs.

Tags follow Universal Dependencies: `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `CONJ`, `PART`, etc.

**Example:**
```python
from manglish_nlp import pos_tag

pos_tag("Aku suka makan nasi lemak")
# [('Aku', 'PRON'), ('suka', 'VERB'), ('makan', 'VERB'), ('nasi', 'NOUN'), ('lemak', 'ADJ')]
```

---

### `extract_keywords(text, top_n=5)`

Extract top keywords from text using TF-IDF weighting.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| top_n | `int` | `5` | Number of keywords to return |

**Returns:** `list[dict]`

| Key | Type | Description |
|-----|------|-------------|
| keyword | `str` | Keyword text |
| score | `float` | Relevance score |

**Example:**
```python
from manglish_nlp import extract_keywords

extract_keywords("Harga minyak naik lagi, rakyat suffer gila", top_n=3)
# [{'keyword': 'minyak', 'score': 0.42}, {'keyword': 'harga', 'score': 0.38}, {'keyword': 'rakyat', 'score': 0.31}]
```

---

## Advanced NLP

### `segment(text)`

Segment continuous text into sentences, handling abbreviations common in Manglish.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `list[str]` — Sentence list.

**Example:**
```python
from manglish_nlp import segment

segment("Eh btw kau tau tak. Semalam aku jumpa dia. Best gila")
# ['Eh btw kau tau tak.', 'Semalam aku jumpa dia.', 'Best gila']
```

---

### `similarity(text_a, text_b)`

Compute semantic similarity between two texts.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text_a | `str` | — | First text |
| text_b | `str` | — | Second text |

**Returns:** `float` — Similarity score (0–1).

**Example:**
```python
from manglish_nlp import similarity

similarity("Aku lapar", "Perut dah kosong ni")
# 0.78

similarity("Aku lapar", "Kereta baru dia cantik")
# 0.12
```

---

### `augment(text, n=3)`

Generate augmented variants of text for data augmentation.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| n | `int` | `3` | Number of variants |
| method | `str` | `"mixed"` | Strategy: `"synonym"`, `"insert"`, `"swap"`, `"delete"`, `"mixed"` |

**Returns:** `list[str]` — Augmented texts.

**Example:**
```python
from manglish_nlp import augment

augment("Makanan sedap gila kat kedai tu", n=2)
# ['Makanan lazat gila kat kedai tu', 'Makanan sedap gila dekat kedai itu']
```

---

### `correct(text)`

Spell-check and correct Manglish/Malay text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| corrected | `str` | Corrected text |
| changes | `list[dict]` | List of corrections made |

**Example:**
```python
from manglish_nlp import correct

result = correct("Saya mau mkan nsi grng")
# {'corrected': 'saya mahu makan nasi goreng',
#  'changes': [{'original': 'mau', 'corrected': 'mahu'}, ...]}
```

---

## Code-Switching

### `code_switching.detect_switches(text)`

Detect and annotate code-switching boundaries in mixed-language text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| segments | `list[dict]` | Each segment with language label and span |
| switch_count | `int` | Number of language switches |
| pattern | `str` | Dominant switching pattern (e.g., `"ms-en"`) |

**Example:**
```python
from manglish_nlp import code_switching

result = code_switching.detect_switches("I nak pergi market beli fish")
# {
#   'segments': [
#     {'text': 'I', 'lang': 'en', 'start': 0, 'end': 1},
#     {'text': 'nak pergi', 'lang': 'ms', 'start': 2, 'end': 11},
#     {'text': 'market', 'lang': 'en', 'start': 12, 'end': 18},
#     {'text': 'beli', 'lang': 'ms', 'start': 19, 'end': 23},
#     {'text': 'fish', 'lang': 'en', 'start': 24, 'end': 28}
#   ],
#   'switch_count': 4,
#   'pattern': 'en-ms'
# }
```

---

## Intent & Classification

### `intent.classify_intent(text)`

Classify user intent from text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| intent | `str` | Primary intent label |
| confidence | `float` | Confidence (0–1) |
| all_intents | `list[dict]` | Top-N intents with scores |

Supported intents: `query`, `command`, `complaint`, `greeting`, `request`, `feedback`, `other`.

**Example:**
```python
from manglish_nlp import intent

result = intent.classify_intent("Macam mana nak refund barang ni?")
# {'intent': 'query', 'confidence': 0.91, 'all_intents': [...]}
```

---

### `topic.classify_topic(text)`

Classify text into topic categories.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| topic | `str` | Primary topic |
| confidence | `float` | Confidence (0–1) |
| subtopic | `str` or `None` | Subtopic if applicable |

Topics: `politics`, `sports`, `entertainment`, `business`, `technology`, `health`, `education`, `lifestyle`, `religion`, `other`.

**Example:**
```python
from manglish_nlp import topic

topic.classify_topic("Harga saham FGV naik mendadak hari ni")
# {'topic': 'business', 'confidence': 0.88, 'subtopic': 'finance'}
```

---

## Safety & Moderation

### `hate_speech.detect_hate_speech(text)`

Detect hate speech and toxicity in text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| is_hate_speech | `bool` | Whether hate speech detected |
| severity | `str` | `"none"`, `"mild"`, `"moderate"`, `"severe"` |
| categories | `list[str]` | Hate categories detected |
| score | `float` | Toxicity probability (0–1) |

**Example:**
```python
from manglish_nlp import hate_speech

hate_speech.detect_hate_speech("Kau ni memang [slur]")
# {'is_hate_speech': True, 'severity': 'severe', 'categories': ['ethnic'], 'score': 0.93}
```

---

### `stance.detect_stance(text, target=None)`

Detect author's stance toward a topic or entity.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| target | `str` or `None` | `None` | Target entity/topic |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| stance | `str` | `"for"`, `"against"`, `"neutral"` |
| confidence | `float` | Confidence (0–1) |

**Example:**
```python
from manglish_nlp import stance

stance.detect_stance("Dasar kerajaan ni teruk, menyusahkan rakyat je", target="kerajaan")
# {'stance': 'against', 'confidence': 0.89}
```

---

## Generative

### `summarization.summarize(text, max_length=100)`

Summarize long text.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| max_length | `int` | `100` | Max output words |
| style | `str` | `"extractive"` | `"extractive"` or `"abstractive"` |

**Returns:** `str` — Summary.

**Example:**
```python
from manglish_nlp import summarization

summarization.summarize(long_article, max_length=50)
# 'Perdana Menteri mengumumkan pakej rangsangan ekonomi...'
```

---

### `translation.translate(text, target_lang="en")`

Translate between Malay/Manglish and other languages.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| text | `str` | — | Input text |
| target_lang | `str` | `"en"` | Target language code |
| source_lang | `str` or `None` | `None` | Source language (auto-detect if `None`) |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| translated | `str` | Translated text |
| source_lang | `str` | Detected/source language |
| confidence | `float` | Translation confidence |

**Example:**
```python
from manglish_nlp import translation

translation.translate("Aku dah sampai rumah", target_lang="en")
# {'translated': "I've arrived home", 'source_lang': 'ms', 'confidence': 0.91}
```

---

### `qa.answer(question, context)`

Extract answer from context given a question.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| question | `str` | — | Question text |
| context | `str` | — | Context paragraph |

**Returns:** `dict`

| Key | Type | Description |
|-----|------|-------------|
| answer | `str` | Extracted answer |
| score | `float` | Confidence (0–1) |
| start | `int` | Start char offset in context |
| end | `int` | End char offset in context |

**Example:**
```python
from manglish_nlp import qa

qa.answer(
    "Siapa PM Malaysia?",
    "Perdana Menteri Malaysia ke-10 ialah Anwar Ibrahim sejak 2022."
)
# {'answer': 'Anwar Ibrahim', 'score': 0.95, 'start': 40, 'end': 53}
```

---

### `text_generation.generate(prompt, max_length=50)`

Generate text continuation from a prompt.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| prompt | `str` | — | Input prompt |
| max_length | `int` | `50` | Max tokens to generate |
| temperature | `float` | `0.7` | Sampling temperature |

**Returns:** `str` — Generated text.

**Example:**
```python
from manglish_nlp import text_generation

text_generation.generate("Cuaca hari ni memang", max_length=20)
# 'panas gila, rasa macam nak duduk dalam fridge je'
```

---

## Pipeline & Models

### `pipeline(steps)`

Chain multiple NLP operations into a reusable pipeline.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| steps | `list[str]` | — | Ordered list of operation names |

**Returns:** `Pipeline` object with `.run(text)` method.

Available steps: `"clean"`, `"normalize"`, `"tokenize"`, `"sentiment"`, `"ner"`, `"pos"`, `"keywords"`.

**Example:**
```python
from manglish_nlp import pipeline

pipe = pipeline(["clean", "normalize", "sentiment"])
result = pipe.run("@user sumpah best gila movie ni!!!")
# {'cleaned': 'sumpah best gila movie ni!!!',
#  'normalized': 'sumpah best gila filem ini',
#  'sentiment': {'sentiment': 'positive', 'score': 0.94}}
```

---

### `load_word2vec(model_path=None)`

Load pre-trained Word2Vec embeddings for Malay/Manglish.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| model_path | `str` or `None` | `None` | Path to model file (auto-downloads if `None`) |

**Returns:** `Word2VecModel`

| Method | Description |
|--------|-------------|
| `.get_vector(word)` | Get embedding vector |
| `.most_similar(word, n=10)` | Get nearest neighbors |
| `.similarity(word_a, word_b)` | Cosine similarity |

**Example:**
```python
from manglish_nlp import load_word2vec

w2v = load_word2vec()
w2v.most_similar("makan", n=5)
# [('minum', 0.82), ('masak', 0.76), ('nasi', 0.71), ...]
```

---

### `load_fasttext(model_path=None)`

Load pre-trained FastText embeddings with subword support.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| model_path | `str` or `None` | `None` | Path to model file (auto-downloads if `None`) |

**Returns:** `FastTextModel`

| Method | Description |
|--------|-------------|
| `.get_vector(word)` | Get embedding (handles OOV via subwords) |
| `.most_similar(word, n=10)` | Get nearest neighbors |
| `.get_sentence_vector(text)` | Get averaged sentence embedding |

**Example:**
```python
from manglish_nlp import load_fasttext

ft = load_fasttext()
ft.get_vector("lah")  # Works even for particles
# array([0.12, -0.34, 0.56, ...], dtype=float32)
```

---

## Error Handling

All functions raise typed exceptions:

| Exception | When |
|-----------|------|
| `ManglishNLPError` | Base exception for all errors |
| `InputError` | Invalid or empty input |
| `ModelError` | Model loading or inference failure |
| `LanguageError` | Unsupported language detected |
| `PipelineError` | Invalid pipeline step or chain |

**Example:**
```python
from manglish_nlp import sentiment, ManglishNLPError, InputError

try:
    sentiment("")
except InputError as e:
    print(e)  # "Input text cannot be empty"
```

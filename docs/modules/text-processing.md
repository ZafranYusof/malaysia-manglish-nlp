# Text Processing

**Clean, normalise, tokenise, and stem Malaysian text  -  the preprocessing foundation.**

---

## Overview

Text processing modules transform raw, messy Malaysian text into clean, structured input ready for analysis or generation. They handle SMS-speak, mixed scripts, repeated characters, and Malay-specific morphology.

All modules in this group have **zero external dependencies** and run in under 1 ms per sentence.

```python
import manglish_nlp as mnlp
```

---

## Quick Start

```python
import manglish_nlp as mnlp

raw = "Wehh xpe la bro, aku nk g mkn jap lg 🔥🔥🔥 bestttt!!!"

# Chain processing
cleaned   = mnlp.clean(raw)                        # strip noise
normalised = mnlp.normalize(cleaned)               # expand shortforms
tokens    = mnlp.tokenize(normalised)              # split into tokens

print(tokens)
# ['takpe', 'la', 'bro', 'aku', 'nak', 'pergi', 'makan', 'jap', 'lagi', 'best']
```

---

## Module Details

### `normalize`

Converts informal Manglish spelling to standard form. Ships with **12,000+ shortform mappings** covering SMS-speak, social media abbreviations, and common misspellings.

```python
import manglish_nlp as mnlp

text = "xpe la bro, aku nk g mkn jap lg"
result = mnlp.normalize(text)
print(result)
# "takpe la bro, aku nak pergi makan jap lagi"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `preserve_slang` | `bool` | `False` | Keep slang terms (e.g. "gempak") untouched |
| `custom_dict` | `dict` | `{}` | Additional shortform → standard mappings |
| `aggressive` | `bool` | `False` | Normalise particles too (la→lah, je→sahaja) |

!!! tip "Custom Dictionaries"
    Extend with domain-specific terms:
    ```python
    mnlp.normalize(text, custom_dict={"gais": "guys", "member": "kawan"})
    ```

!!! example "Aggressive Mode"
    ```python
    mnlp.normalize("xpe la je", aggressive=True)
    # "takpe lah sahaja"
    ```

---

### `clean`

Removes noise from text  -  URLs, mentions, hashtags, emojis, repeated characters, and HTML artefacts.

```python
import manglish_nlp as mnlp

text = "Weh @ahmad check ni https://t.co/abc 🔥🔥🔥 bestttt"
result = mnlp.clean(text)
print(result)
# "Weh check ni best"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `keep_emoji` | `bool` | `False` | Preserve emoji characters |
| `keep_hashtags` | `bool` | `False` | Keep hashtag text (strip `#` symbol only) |
| `keep_mentions` | `bool` | `False` | Preserve `@mentions` |
| `max_repeat` | `int` | `1` | Max allowed consecutive repeated chars |

!!! example "Preserving Emojis"
    ```python
    mnlp.clean("Best gila 🔥🔥🔥", keep_emoji=True)
    # "Best gila 🔥"
    ```

---

### `formalize`

Converts casual Manglish to formal Bahasa Melayu suitable for official documents, reports, or academic writing.

```python
import manglish_nlp as mnlp

text = "Aku rasa mcm nak apply kerja kat situ la"
result = mnlp.formalize(text)
print(result)
# "Saya rasa seperti ingin memohon pekerjaan di situ"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `level` | `int` | `5` | Formality level (1 = semi-casual, 5 = full formal BM) |
| `keep_english` | `bool` | `False` | Preserve English loanwords as-is |

!!! warning "Review Critical Output"
    Formalisation may shift meaning in ambiguous sentences. Always review output for official documents.

---

### `tokenizer`

Malaysian-aware tokeniser that correctly handles mixed scripts, particles (`la`, `je`, `kot`), compound words, and code-switched text.

```python
import manglish_nlp as mnlp

text = "Tak boleh la macam tu, it's not fair"
tokens = mnlp.tokenize(text)
print(tokens)
# ['Tak', 'boleh', 'la', 'macam', 'tu', ',', "it's", 'not', 'fair']
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `level` | `str` | `"word"` | Tokenisation level: `"word"`, `"sentence"`, `"subword"` |
| `split_particles` | `bool` | `True` | Separate particles from host words |

!!! tip "Malaysian Patterns"
    The tokeniser correctly splits:
    - Contractions: `"takde"` → `["tak", "ada"]`
    - Reduplication: `"budak-budak"` kept as one token
    - Code-switch boundaries: `"I rasa"` split cleanly

---

### `stemmer`

Rule-based Malay stemmer handling prefixes (`me-`, `ber-`, `di-`, `ke-`, `memper-`) and suffixes (`-kan`, `-an`, `-i`).

```python
import manglish_nlp as mnlp

words = ["memakan", "berlari", "ditulis", "permainan"]
stems = [mnlp.stem(w) for w in words]
print(stems)
# ['makan', 'lari', 'tulis', 'main']
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | `str` | *required* | Input word |
| `detailed` | `bool` | `False` | Return affix breakdown dict |
| `conservative` | `bool` | `False` | Fewer, safer reductions |

!!! example "Detailed Affix Analysis"
    ```python
    mnlp.stem("memperkenalkan", detailed=True)
    # {'stem': 'kenal', 'prefix': 'memper-', 'suffix': '-kan', 'original': 'memperkenalkan'}
    ```

---

### `segment`

Splits unsegmented text into words. Useful for hashtags, concatenated URLs, and OCR artefacts.

```python
import manglish_nlp as mnlp

text = "nakpergimanasatumalam"
result = mnlp.segment(text)
print(result)
# "nak pergi mana satu malam"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Concatenated input |
| `lang` | `str` | `"ms"` | Language hint for segmentation model |
| `scores` | `bool` | `False` | Return per-word confidence scores |

!!! example "Hashtag Segmentation"
    ```python
    mnlp.segment("#MalaysiaBoleh")
    # "Malaysia Boleh"

    mnlp.segment("goodmorningmalaysia", lang="en")
    # "good morning malaysia"
    ```

---

### `spelling`

Context-aware spelling correction that distinguishes intentional abbreviations (`nk`, `kat`, `mcm`) from actual typos.

```python
import manglish_nlp as mnlp

text = "Aku nk prgi mkn kat keday tu"
corrected = mnlp.spelling(text)
print(corrected)
# "Aku nak pergi makan kat kedai tu"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `candidates` | `bool` | `False` | Return top correction candidates with scores |
| `preserve_informal` | `bool` | `False` | Keep intentional abbreviations, fix only real typos |
| `context` | `bool` | `False` | Use surrounding words to disambiguate corrections |
| `whitelist` | `list[str]` | `[]` | Words to never correct |

!!! warning "Informal vs Typo"
    `preserve_informal=True` keeps `nk`, `kat`, `mcm` intact but still fixes `keday` → `kedai`. Without this flag, all non-standard forms are corrected.

!!! example "Context-Aware Correction"
    ```python
    mnlp.spelling("Dia bgi aku bku", context=True)
    # "Dia bagi aku buku"
    ```

---

## Chaining Modules

The typical preprocessing pipeline chains modules in order:

```
raw text → clean → normalize → tokenize → [stem | spell] → ready for analysis
```

Use the [`pipeline`](tools.md#pipeline) module to make this reusable:

```python
from manglish_nlp import Pipeline

preprocess = Pipeline([
    'clean',
    'normalize',
    'tokenize'
])

result = preprocess("Wehh xpe la bro!! 🔥 bestttt gila")
# ['takpe', 'la', 'bro', 'best', 'gila']
```

---

## See Also

- [Analysis](analysis.md)  -  sentiment, emotion, and language detection on cleaned text
- [Pipeline](tools.md#pipeline)  -  chain preprocessing steps into reusable workflows
- [Cache](tools.md#cache)  -  cache expensive normalisation for repeated inputs

# Language Detection

**Detect whether text is Bahasa Melayu, English, Manglish, or a mix  -  including dialect detection.**

---

## Why language detection?

Malaysian text is rarely monolingual. A single sentence might switch between BM, English, and local slang. Knowing the language composition helps with routing to the right downstream models, content filtering, and sociolinguistic analysis.

Standard language detectors (like langdetect or fastText) classify Manglish as "Malay" or "English" and miss the code-switching entirely.

---

## Load module

```python
import manglish_nlp as mnlp

result = mnlp.detect_language("Eh jom la we go makan, I lapar gila already")
print(result)
# {'primary': 'manglish', 'scores': {'ms': 0.45, 'en': 0.55}}
```

---

## Basic usage

### Pure BM

```python
mnlp.detect_language("Saya hendak pergi ke pasar")
# {'primary': 'ms', 'scores': {'ms': 0.95, 'en': 0.05}}
```

### Pure English

```python
mnlp.detect_language("I want to go to the market")
# {'primary': 'en', 'scores': {'ms': 0.05, 'en': 0.95}}
```

### Manglish (code-mixed)

```python
mnlp.detect_language("Weh jom la makan, I lapar gila ni")
# {'primary': 'manglish', 'scores': {'ms': 0.40, 'en': 0.60}}
```

### Short Manglish

```python
mnlp.detect_language("Best gila!")
# {'primary': 'manglish', 'scores': {'ms': 0.55, 'en': 0.45}}
```

---

## Language scores

The `scores` dict shows the proportion of each language in the text:

```python
result = mnlp.detect_language("I nak order satu nasi lemak with extra sambal")
print(result)
# {'primary': 'manglish', 'scores': {'ms': 0.50, 'en': 0.50}}

result = mnlp.detect_language("Saya suka makan nasi lemak every morning")
print(result)
# {'primary': 'manglish', 'scores': {'ms': 0.65, 'en': 0.35}}
```

!!! tip "Interpreting scores"
    - `ms` > 0.8 → mostly BM
    - `en` > 0.8 → mostly English
    - Both between 0.3–0.7 → code-mixed / Manglish
    - `primary: 'manglish'` → significant mixing detected

---

## Code-switching detection

For detailed analysis of where language switches happen, use the code-switching module:

```python
import manglish_nlp as mnlp

result = mnlp.code_switching.detect("I nak pergi kedai sebab I lapar gila")
print(result)
# {
#     'segments': [
#         {'text': 'I', 'lang': 'en'},
#         {'text': 'nak pergi kedai sebab', 'lang': 'ms'},
#         {'text': 'I', 'lang': 'en'},
#         {'text': 'lapar gila', 'lang': 'ms'},
#     ],
#     'switch_ratio': 0.6,
#     'switch_count': 3,
# }
```

See [Code-Switching Detection](code-switching.md) for the full tutorial.

---

## Dialect detection

Detect regional Malay dialects  -  6 supported:

```python
# Kelantan
mnlp.detect_dialect("Ambo nok make nasi kerabu")
# {'dialect': 'kelantan', 'confidence': 0.91}

# Kedah
mnlp.detect_dialect("Cheq nak pi makan satgi")
# {'dialect': 'kedah', 'confidence': 0.88}

# Terengganu
mnlp.detect_dialect("Ambe nok makan nasi dagang")
# {'dialect': 'terengganu', 'confidence': 0.87}

# Negeri Sembilan
mnlp.detect_dialect("Ehden nak makan lomak cili api")
# {'dialect': 'negeri_sembilan', 'confidence': 0.82}

# Perak
mnlp.detect_dialect("Teme nak gi kedai kejap")
# {'dialect': 'perak', 'confidence': 0.79}

# Standard (no dialect)
mnlp.detect_dialect("Saya nak pergi makan")
# {'dialect': 'standard', 'confidence': 0.93}
```

### Available dialects

```python
mnlp.available_dialects()
# ['standard', 'kelantan', 'kedah', 'terengganu', 'negeri_sembilan', 'perak', 'sabah', 'sarawak']
```

### Normalize dialect to standard BM

```python
mnlp.normalize_dialect("Ambo nok make nasi kerabu", dialect="kelantan")
# "Saya hendak makan nasi kerabu"

mnlp.normalize_dialect("Cheq nak pi satgi", dialect="kedah")
# "Saya hendak pergi sebentar"
```

---

## Real examples from social media

```python
# Twitter/X style
mnlp.detect_language("weh korang dh try ke burger abang burn tu? serious sedap gila")
# {'primary': 'manglish', 'scores': {'ms': 0.70, 'en': 0.30}}

# WhatsApp style
mnlp.detect_language("Bro i already at the mamak, jom la cepat sikit")
# {'primary': 'manglish', 'scores': {'ms': 0.35, 'en': 0.65}}

# Reddit r/malaysia style
mnlp.detect_language("Anyone tried the new LRT line? Macam ok je tapi crowded gila peak hour")
# {'primary': 'manglish', 'scores': {'ms': 0.40, 'en': 0.60}}

# Formal news
mnlp.detect_language("Perdana Menteri mengumumkan bantuan khas RM500")
# {'primary': 'ms', 'scores': {'ms': 0.92, 'en': 0.08}}
```

---

## Batch processing

```python
texts = [
    "Saya suka makan nasi lemak",
    "I love eating coconut rice",
    "Weh jom makan nasi lemak",
]

for text in texts:
    result = mnlp.detect_language(text)
    print(f"{text:35s} → {result['primary']}")

# Saya suka makan nasi lemak        → ms
# I love eating coconut rice        → en
# Weh jom makan nasi lemak          → manglish
```

---

## CLI usage

```bash
# Language detection
$ mnlp language "Weh jom la makan"
manglish (ms: 0.50, en: 0.50)

# Dialect detection
$ mnlp dialect "Ambo nok make nasi kerabu"
kelantan (0.91)

# JSON output
$ mnlp language "I nak pergi kedai" --json
{"primary": "manglish", "scores": {"ms": 0.45, "en": 0.55}}
```

---

## How it works

1. **Word-level classification**  -  each word classified as BM, EN, or shared
2. **Particle detection**  -  Malaysian particles ("la", "lah", "kan", "weh") signal Manglish
3. **Shortform recognition**  -  "nk", "brp", "sy" identified as BM shortforms
4. **Ratio calculation**  -  proportion of each language computed
5. **Dialect matching**  -  dialect-specific vocabulary and pronouns checked

---

## Performance

| Metric | Score |
|--------|-------|
| BM vs EN accuracy | 96.2% |
| Manglish detection F1 | 89.5% |
| Dialect detection accuracy | 83.7% |
| Throughput | 35,000 texts/sec |
| Latency (single) | < 0.3ms |

---

## See also

- [Code-Switching](code-switching.md)  -  detailed switch point detection
- [Normalization](normalization.md)  -  normalize Manglish for downstream processing
- [Translation](translation.md)  -  translate detected language pairs
- [Pipeline](pipeline.md)  -  include language detection in pipelines
- [API Reference](../api-reference.md#language)  -  full function signature

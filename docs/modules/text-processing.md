# Text Processing

Core text processing modules for cleaning, normalizing, and tokenizing Malaysian text.

---

## normalize

Converts informal Manglish spelling to standard form. Handles SMS-speak, abbreviations, and common misspellings.

```python
import manglish_nlp as mnlp

text = "xpe la bro, aku nk g mkn jap lg"
result = mnlp.normalize(text)
print(result)
# "takpe la bro, aku nak pergi makan jap lagi"
```

### Options

```python
# Preserve certain slang terms
mnlp.normalize(text, preserve_slang=True)

# Custom dictionary
mnlp.normalize(text, custom_dict={"gais": "guys"})

# Aggressive mode (normalize everything including particles)
mnlp.normalize(text, aggressive=True)
```

!!! tip "Custom Dictionaries"
    You can extend the normalization dictionary with domain-specific terms using `custom_dict`. This is useful for brand names or technical jargon.

---

## clean

Removes noise from text — URLs, mentions, hashtags, emojis, repeated characters.

```python
text = "Weh @ahmad check ni https://t.co/abc 🔥🔥🔥 bestttt"
result = mnlp.clean(text)
print(result)
# "Weh check ni best"
```

### Options

```python
# Keep emojis
mnlp.clean(text, keep_emoji=True)

# Keep hashtags (remove only the # symbol)
mnlp.clean(text, keep_hashtags=True)

# Keep mentions
mnlp.clean(text, keep_mentions=True)

# Remove repeated chars but keep doubles
mnlp.clean(text, max_repeat=2)
# "Weh check ni bestt"
```

---

## formalize

Converts casual Manglish to formal Bahasa Melayu suitable for official documents.

```python
text = "Aku rasa mcm nak apply kerja kat situ la"
result = mnlp.formalize(text)
print(result)
# "Saya rasa seperti ingin memohon pekerjaan di situ"
```

### Options

```python
# Target formality level (1-5)
mnlp.formalize(text, level=3)  # Semi-formal
mnlp.formalize(text, level=5)  # Full formal BM

# Preserve English terms
mnlp.formalize(text, keep_english=True)
```

!!! warning "Context Sensitivity"
    Formalization may change meaning in ambiguous cases. Always review output for critical documents.

---

## tokenizer

Malaysian-aware tokenizer that handles mixed scripts, particles, and compound words.

```python
text = "Tak boleh la macam tu, it's not fair"
tokens = mnlp.tokenize(text)
print(tokens)
# ['Tak', 'boleh', 'la', 'macam', 'tu', ',', "it's", 'not', 'fair']
```

### Options

```python
# Word-level (default)
mnlp.tokenize(text, level="word")

# Sentence-level
mnlp.tokenize(text, level="sentence")

# Subword (BPE)
mnlp.tokenize(text, level="subword")

# Keep particles attached
mnlp.tokenize(text, split_particles=False)
# ['Tak', 'boleh la', 'macam tu', ',', "it's", 'not', 'fair']
```

---

## stemmer

Malay-aware stemmer that handles prefixes (me-, ber-, di-, ke-) and suffixes (-kan, -an, -i).

```python
words = ["memakan", "berlari", "ditulis", "permainan"]
stems = [mnlp.stem(w) for w in words]
print(stems)
# ['makan', 'lari', 'tulis', 'main']
```

### Options

```python
# Return affix information
result = mnlp.stem("memperkenalkan", detailed=True)
print(result)
# {'stem': 'kenal', 'prefix': 'memper-', 'suffix': '-kan', 'original': 'memperkenalkan'}

# Conservative mode (fewer reductions)
mnlp.stem("permainan", conservative=True)
# 'main'
```

!!! note "Stemmer vs Lemmatizer"
    The stemmer reduces words to root form using rule-based affix stripping. For context-aware lemmatization, use the `[ml]` extra which provides a neural lemmatizer.

---

## segment

Splits unsegmented text into words. Useful for hashtags, URLs, and concatenated text.

```python
text = "nakpergimanasatumalam"
result = mnlp.segment(text)
print(result)
# "nak pergi mana satu malam"
```

### Options

```python
# Segment hashtags
mnlp.segment("#MalaysiaBoLeh")
# "Malaysia Boleh"

# Segment with language hint
mnlp.segment("goodmorningmalaysia", lang="en")
# "good morning malaysia"

# Return confidence scores
mnlp.segment(text, scores=True)
# [('nak', 0.98), ('pergi', 0.95), ('mana', 0.91), ...]
```

---

## See Also

- [Analysis modules](analysis.md) — use after text processing
- [Pipeline](tools.md#pipeline) — chain text processing steps together

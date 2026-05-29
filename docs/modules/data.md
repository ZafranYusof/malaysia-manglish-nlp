# Data & Embeddings

Modules for word representations, similarity, data augmentation, and lexical resources.

---

## word_embeddings

Pre-trained word embeddings for Malaysian vocabulary including Manglish terms.

```python
import manglish_nlp as mnlp

# Load embeddings
emb = mnlp.word_embeddings()

# Get vector for a word
vector = emb["makan"]
print(vector.shape)
# (300,)

# Find similar words
emb.most_similar("sedap")
# [('lazat', 0.87), ('enak', 0.82), ('best', 0.79), ('power', 0.71)]
```

### Options

```python
# Different embedding sizes
emb = mnlp.word_embeddings(dim=100)   # Smaller, faster
emb = mnlp.word_embeddings(dim=300)   # Default, balanced

# Analogy queries
emb.analogy("raja", "perempuan", "lelaki")
# 'ratu'

# OOV handling (subword fallback)
emb["xsedap"]  # Returns approximation from subword components

# Check vocabulary
"lepak" in emb  # True
emb.vocab_size  # ~500,000 tokens
```

!!! info "Embedding Sources"
    Trained on 10M+ Malaysian social media posts, news articles, and forum discussions. Covers formal BM, informal Manglish, and common code-switched terms.

---

## embeddings

Sentence and document-level embeddings for semantic representation.

```python
# Sentence embedding
vec = mnlp.embeddings("Aku nak pergi makan nasi lemak")
print(vec.shape)
# (768,)

# Batch embeddings
vecs = mnlp.embeddings(["text1", "text2", "text3"])
print(vecs.shape)
# (3, 768)
```

### Options

```python
# Model selection
mnlp.embeddings(text, model="fast")     # Lightweight, ~5ms/text
mnlp.embeddings(text, model="accurate") # Transformer-based, ~50ms/text

# Normalize vectors
mnlp.embeddings(text, normalize=True)

# Pooling strategy
mnlp.embeddings(text, pooling="mean")   # Default
mnlp.embeddings(text, pooling="cls")    # CLS token
mnlp.embeddings(text, pooling="max")    # Max pooling
```

---

## similarity

Compute semantic similarity between texts.

```python
score = mnlp.similarity(
    "Aku lapar gila",
    "I'm so hungry right now"
)
print(score)
# 0.91
```

### Options

```python
# Pairwise similarity matrix
texts = ["Nak makan", "Lapar sangat", "Cuaca panas", "Hari ni hot"]
matrix = mnlp.similarity(texts, mode="matrix")
# [[1.0, 0.88, 0.12, 0.15],
#  [0.88, 1.0, 0.10, 0.13],
#  [0.12, 0.10, 1.0, 0.85],
#  [0.15, 0.13, 0.85, 1.0]]

# Method selection
mnlp.similarity(a, b, method="cosine")     # Default
mnlp.similarity(a, b, method="jaccard")    # Token overlap
mnlp.similarity(a, b, method="wmd")        # Word Mover's Distance

# Find most similar from candidates
mnlp.similarity("Nak makan", candidates=["Food options", "Weather today", "Hungry"], top_k=1)
# [('Hungry', 0.89)]
```

---

## augmentation

Data augmentation techniques tailored for Malaysian text.

```python
text = "Makanan kat sini memang sedap"
augmented = mnlp.augment(text, n=5)
print(augmented)
# ['Makanan dekat sini memang sedap',
#  'Makanan kat situ mmg sedap',
#  'Food kat sini memang best',
#  'Makanan kat sini confirm sedap',
#  'Mknn kat sini mmg sedap']
```

### Augmentation Methods

```python
# Synonym replacement
mnlp.augment(text, method="synonym", n=3)

# Code-switch injection (add English/BM alternatives)
mnlp.augment(text, method="code_switch", n=3)

# Spelling variation (informal variants)
mnlp.augment(text, method="spelling", n=3)

# Back-translation
mnlp.augment(text, method="backtranslate", n=3)

# Random insertion/deletion/swap
mnlp.augment(text, method="random", n=3)

# Combined (mix of all methods)
mnlp.augment(text, method="combined", n=10)
```

!!! tip "Training Data"
    Use augmentation to expand small Malaysian NLP datasets. The code-switch and spelling variation methods are particularly effective for improving model robustness on informal text.

---

## dictionary

Malaysian lexical dictionary with definitions, examples, and relationships.

```python
entry = mnlp.dictionary("lepak")
print(entry)
# {'word': 'lepak', 'pos': 'verb',
#  'definitions': ['to hang out', 'to relax', 'to loiter'],
#  'examples': ['Jom lepak kat mamak', 'Aku lepak rumah je hari ni'],
#  'synonyms': ['hangout', 'chill', 'relax'],
#  'register': 'informal'}
```

### Options

```python
# Get all senses
mnlp.dictionary("set", all_senses=True)
# [{'sense': 1, 'definition': 'confirmed/agreed', 'register': 'informal'},
#  {'sense': 2, 'definition': 'a set/group', 'register': 'neutral'}]

# Reverse lookup (English to Manglish)
mnlp.dictionary("hangout", reverse=True)
# ['lepak', 'yumcha', 'mamak']

# Slang dictionary
mnlp.dictionary("gempak", include_slang=True)
# {'word': 'gempak', 'definitions': ['awesome', 'impressive'], 'era': '2000s'}

# Word frequency
mnlp.dictionary("makan", freq=True)
# {'word': 'makan', 'frequency_rank': 45, 'per_million': 2340}
```

---

## spelling

Spelling correction for Malaysian text with support for informal variants.

```python
text = "Aku nk prgi mkn kat keday tu"
corrected = mnlp.spelling(text)
print(corrected)
# "Aku nak pergi makan kat kedai tu"
```

### Options

```python
# Get correction candidates
mnlp.spelling("mkn", candidates=True)
# [('makan', 0.95), ('main', 0.12), ('min', 0.08)]

# Preserve intentional informal spelling
mnlp.spelling(text, preserve_informal=True)
# "Aku nak pergi makan kat kedai tu"  (keeps "nak" as-is)

# Context-aware correction
mnlp.spelling("Dia bgi aku bku", context=True)
# "Dia bagi aku buku"  (uses context to disambiguate)

# Custom vocabulary (don't correct these)
mnlp.spelling(text, whitelist=["nk", "kat"])
```

!!! warning "Informal vs Misspelling"
    Use `preserve_informal=True` to distinguish intentional abbreviations (nk, kat, mcm) from actual typos. The model knows common Malaysian abbreviation patterns.

---

## See Also

- [Text Processing](text-processing.md) — normalize before embedding
- [Tools](tools.md) — caching for expensive embedding operations
- [Extraction](extraction.md) — use embeddings for entity disambiguation

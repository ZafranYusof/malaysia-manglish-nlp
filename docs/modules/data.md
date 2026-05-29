# Data & Embeddings

**Vector representations, semantic similarity, data augmentation, and lexical resources for Malaysian text.**

---

## Overview

Data modules provide the numerical backbone for NLP: word vectors, sentence embeddings, similarity metrics, augmentation strategies, and a comprehensive Malaysian dictionary. Use these for semantic search, clustering, classification features, and training data expansion.

```python
import malaysian_manglish_nlp as mnlp
```

---

## Quick Start

```python
import malaysian_manglish_nlp as mnlp

# Word similarity
emb = mnlp.word_embeddings()
emb.most_similar("sedap")
# [('lazat', 0.87), ('enak', 0.82), ('best', 0.79), ('power', 0.71)]

# Sentence similarity
mnlp.similarity("Aku lapar gila", "I'm so hungry right now")
# 0.91

# Augment training data
mnlp.augment("Makanan kat sini memang sedap", n=3)
# ['Makanan dekat sini memang sedap',
#  'Makanan kat sini mmg sedap',
#  'Food kat sini memang best']
```

---

## Module Details

### `word_embeddings`

Pre-trained Word2Vec embeddings for Malaysian vocabulary. Trained on **10M+ Malaysian social media posts, news articles, and forum discussions**. Covers formal BM, informal Manglish, and code-switched terms.

```python
import malaysian_manglish_nlp as mnlp

emb = mnlp.word_embeddings()

# Lookup vector
emb["makan"].shape
# (300,)

# Similarity
emb.most_similar("sedap")
# [('lazat', 0.87), ('enak', 0.82), ('best', 0.79), ('power', 0.71)]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | `int` | `300` | Embedding dimension: `100`, `200`, or `300` |

#### Methods

| Method | Description | Example |
|--------|-------------|---------|
| `most_similar(word, top_k=10)` | Find nearest words | `emb.most_similar("sedap")` |
| `analogy(positive, negative, word)` | Word analogy | `emb.analogy("raja", "perempuan", "lelaki")` → `"ratu"` |
| `similarity(w1, w2)` | Pairwise cosine | `emb.similarity("makan", "minum")` → `0.72` |

#### Properties

| Property | Value |
|----------|-------|
| `vocab_size` | ~500,000 tokens |
| `dimensions` | 100 / 200 / 300 |
| `oov_handling` | Subword fallback (character n-gram averaging) |

!!! example "Analogy Queries"
    ```python
    emb.analogy("raja", "perempuan", "lelaki")
    # 'ratu'

    emb.analogy("KL", "Malaysia", "Thailand")
    # 'Bangkok'
    ```

!!! tip "OOV Words"
    Unknown words fall back to subword averaging:
    ```python
    emb["xsedap"]  # approximated from sub-words, not zero vector
    "lepak" in emb  # True  -  covered in training data
    ```

---

### `embeddings`

Sentence and document-level embeddings for semantic representation. Two model tiers available: fast (lightweight, ~5 ms/text) and accurate (transformer-based, ~50 ms/text).

```python
import malaysian_manglish_nlp as mnlp

# Single sentence
vec = mnlp.embeddings("Aku nak pergi makan nasi lemak")
vec.shape
# (768,)

# Batch
vecs = mnlp.embeddings(["text1", "text2", "text3"])
vecs.shape
# (3, 768)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text(s) |
| `model` | `str` | `"accurate"` | Model tier: `"fast"` or `"accurate"` |
| `normalize` | `bool` | `False` | L2-normalise output vectors |
| `pooling` | `str` | `"mean"` | Pooling strategy: `"mean"`, `"cls"`, `"max"` |

#### Model Comparison

| Model | Dimensions | Speed | Use Case |
|-------|-----------|-------|----------|
| `fast` | 384 | ~5 ms/text | Real-time search, clustering |
| `accurate` | 768 | ~50 ms/text | Semantic similarity, classification features |

!!! tip "Choosing a Model"
    - **`fast`**: real-time applications, large-scale retrieval, quick prototyping
    - **`accurate`**: production similarity, classification features, QA retrieval

---

### `similarity`

Compute semantic similarity between texts. Cross-lingual (BM ↔ EN ↔ Manglish) by default.

```python
import malaysian_manglish_nlp as mnlp

mnlp.similarity("Aku lapar gila", "I'm so hungry right now")
# 0.91

mnlp.similarity("Cuaca panas hari ni", "Hari ni memang hot gila")
# 0.93
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_a` | `str` | *required* | First text |
| `text_b` | `str \| list[str]` | *required* | Second text or candidate list |
| `method` | `str` | `"cosine"` | Similarity method: `"cosine"`, `"jaccard"`, `"wmd"` |
| `mode` | `str` | `"pair"` | `"pair"` (two texts) or `"matrix"` (all pairwise) |
| `top_k` | `int` | `None` | Return top-k most similar from candidates |

#### Method Comparison

| Method | Speed | Best For |
|--------|-------|----------|
| `cosine` | Fast | General semantic similarity (embedding-based) |
| `jaccard` | Fastest | Token overlap, near-duplicate detection |
| `wmd` | Slow | Fine-grained semantic distance (Word Mover's Distance) |

!!! example "Similarity Matrix"
    ```python
    texts = ["Nak makan", "Lapar sangat", "Cuaca panas", "Hari ni hot"]
    mnlp.similarity(texts, mode="matrix")
    # [[1.00, 0.88, 0.12, 0.15],
    #  [0.88, 1.00, 0.10, 0.13],
    #  [0.12, 0.10, 1.00, 0.85],
    #  [0.15, 0.13, 0.85, 1.00]]
    ```

!!! example "Find Best Match"
    ```python
    mnlp.similarity("Nak makan",
                    candidates=["Food options", "Weather today", "I'm hungry"],
                    top_k=1)
    # [('I\'m hungry', 0.89)]
    ```

---

### `augmentation`

Data augmentation strategies tailored for Malaysian text. Generates synthetic variants for training data expansion.

```python
import malaysian_manglish_nlp as mnlp

mnlp.augment("Makanan kat sini memang sedap", n=5)
# ['Makanan dekat sini memang sedap',        # synonym
#  'Makanan kat situ mmg sedap',             # spelling variation
#  'Food kat sini memang best',              # code-switch
#  'Makanan kat sini confirm sedap',         # synonym
#  'Mknn kat sini mmg sedap']                # abbreviation
```

#### Augmentation Strategies

| Method | Description | Example Output |
|--------|-------------|----------------|
| `synonym` | Replace words with synonyms | "sedap" → "lazat", "best" |
| `code_switch` | Inject BM/EN alternatives | "makanan" → "food" |
| `spelling` | Generate informal variants | "memang" → "mmg" |
| `backtranslate` | Translate away and back | BM → EN → BM variant |
| `random` | Random insert/delete/swap | Token-level perturbation |
| `combined` | Mix all strategies | Diverse output |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `n` | `int` | `3` | Number of augmented variants |
| `method` | `str` | `"combined"` | Strategy: see table above |

!!! tip "Training Data Expansion"
    For Malaysian NLP models, `code_switch` and `spelling` augmentation are the most effective  -  they mirror real-world input variation. Use `combined` for maximum diversity.

!!! warning "Label Preservation"
    Augmentation preserves meaning for classification tasks but may not suit sequence labelling. For NER, prefer `spelling` method which doesn't alter entity boundaries.

---

### `dictionary`

Malaysian lexical dictionary with definitions, examples, synonyms, register information, and word frequency data.

```python
import malaysian_manglish_nlp as mnlp

mnlp.dictionary("lepak")
# {'word': 'lepak', 'pos': 'verb',
#  'definitions': ['to hang out', 'to relax', 'to loiter'],
#  'examples': ['Jom lepak kat mamak', 'Aku lepak rumah je hari ni'],
#  'synonyms': ['hangout', 'chill', 'relax'],
#  'register': 'informal'}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `word` | `str` | *required* | Lookup word |
| `all_senses` | `bool` | `False` | Return all word senses |
| `reverse` | `bool` | `False` | English → Manglish reverse lookup |
| `include_slang` | `bool` | `False` | Include slang definitions and era info |
| `freq` | `bool` | `False` | Include corpus frequency statistics |

!!! example "Reverse Lookup"
    ```python
    mnlp.dictionary("hangout", reverse=True)
    # ['lepak', 'yumcha', 'mamak']
    ```

!!! example "Frequency Data"
    ```python
    mnlp.dictionary("makan", freq=True)
    # {'word': 'makan', 'frequency_rank': 45, 'per_million': 2340}
    ```

!!! example "Multi-Sense Words"
    ```python
    mnlp.dictionary("set", all_senses=True)
    # [{'sense': 1, 'definition': 'confirmed/agreed', 'register': 'informal'},
    #  {'sense': 2, 'definition': 'a set/group', 'register': 'neutral'}]
    ```

---

### `spelling`

Context-aware spelling correction that distinguishes intentional Malaysian abbreviations from actual typos.

```python
import malaysian_manglish_nlp as mnlp

mnlp.spelling("Aku nk prgi mkn kat keday tu")
# "Aku nak pergi makan kat kedai tu"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `candidates` | `bool` | `False` | Return top correction candidates with scores |
| `preserve_informal` | `bool` | `False` | Keep intentional abbreviations |
| `context` | `bool` | `False` | Use surrounding words for disambiguation |
| `whitelist` | `list[str]` | `[]` | Words to never correct |

!!! example "Correction Candidates"
    ```python
    mnlp.spelling("mkn", candidates=True)
    # [('makan', 0.95), ('main', 0.12), ('min', 0.08)]
    ```

!!! tip "Also in Text Processing"
    This module appears in both [Text Processing](text-processing.md#spelling) and here. The same `mnlp.spelling()` call works from either context.

---

## See Also

- [Text Processing](text-processing.md)  -  normalise text before embedding
- [Analysis](analysis.md)  -  use embeddings as classification features
- [Cache](tools.md#cache)  -  cache expensive embedding computations
- [Similarity + Pipeline](tools.md#pipeline)  -  build semantic search pipelines

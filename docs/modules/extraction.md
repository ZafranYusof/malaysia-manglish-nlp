# Extraction

Modules for extracting structured information from Malaysian text.

---

## NER (Named Entity Recognition)

Extract named entities from Manglish text with support for Malaysian-specific entities.

```python
import manglish_nlp as mnlp

text = "Siti beli iPhone kat Low Yat Plaza semalam"
entities = mnlp.ner(text)
print(entities)
# [('Siti', 'PERSON'), ('iPhone', 'PRODUCT'), ('Low Yat Plaza', 'LOCATION')]
```

### Supported Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| `PERSON` | Person names | Siti, Ahmad, Dr. Mahathir |
| `LOCATION` | Places, addresses | KL, Bukit Bintang, Pahang |
| `ORG` | Organizations | Petronas, UMP, Grab |
| `PRODUCT` | Products, brands | iPhone, Myvi, Milo |
| `EVENT` | Events | Hari Raya, Merdeka Day |
| `DATE` | Dates, times | semalam, 15 Mei, next week |
| `MONEY` | Monetary values | RM50, 3 ringgit |

### Options

```python
# Return spans with positions
mnlp.ner(text, spans=True)
# [{'text': 'Siti', 'label': 'PERSON', 'start': 0, 'end': 4}]

# Filter by entity type
mnlp.ner(text, types=["PERSON", "LOCATION"])

# Confidence threshold
mnlp.ner(text, threshold=0.8)

# Batch processing
mnlp.ner(["Text 1", "Text 2"])
```

!!! tip "Malaysian Names"
    The NER model is trained on Malaysian name patterns including Malay, Chinese, Indian, and mixed names with titles (Dato', Tan Sri, etc.).

---

## POS (Part-of-Speech Tagging)

Tag parts of speech with awareness of Malay grammar and code-switching.

```python
text = "Aku nak pergi makan kat kedai tu"
tags = mnlp.pos(text)
print(tags)
# [('Aku', 'PRON'), ('nak', 'AUX'), ('pergi', 'VERB'), ('makan', 'VERB'),
#  ('kat', 'ADP'), ('kedai', 'NOUN'), ('tu', 'DET')]
```

### Tag Set

Uses Universal Dependencies (UD) tag set adapted for Malay:

`NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `ADP`, `AUX`, `CONJ`, `PART`, `NUM`, `PUNCT`, `INTJ`

### Options

```python
# Detailed tags (fine-grained)
mnlp.pos(text, detailed=True)
# [('Aku', 'PRON', {'person': 1, 'number': 'sing'}), ...]

# Return as dict
mnlp.pos(text, format="dict")
# [{'word': 'Aku', 'tag': 'PRON', 'confidence': 0.97}, ...]
```

---

## keywords

Extract keywords and key phrases from Malaysian text.

```python
text = """
Kerajaan Malaysia umumkan pakej rangsangan ekonomi bernilai RM50 bilion
untuk membantu rakyat dan perniagaan kecil yang terjejas akibat pandemik.
"""
keywords = mnlp.keywords(text)
print(keywords)
# ['pakej rangsangan ekonomi', 'RM50 bilion', 'perniagaan kecil', 'pandemik']
```

### Options

```python
# Limit number of keywords
mnlp.keywords(text, top_k=5)

# With relevance scores
mnlp.keywords(text, scores=True)
# [('pakej rangsangan ekonomi', 0.92), ('RM50 bilion', 0.87), ...]

# Algorithm selection
mnlp.keywords(text, method="tfidf")     # TF-IDF based
mnlp.keywords(text, method="textrank")  # Graph-based
mnlp.keywords(text, method="yake")      # YAKE algorithm

# N-gram range
mnlp.keywords(text, ngram_range=(1, 3))
```

!!! info "Stopwords"
    The keyword extractor uses a Malaysian-specific stopword list that includes Malay particles (la, je, kot) and common code-switch words.

---

## dependency

Dependency parsing for Malay and Manglish sentences.

```python
text = "Ali bagi buku tu kat Siti semalam"
deps = mnlp.dependency(text)
print(deps)
# [('Ali', 'nsubj', 'bagi'),
#  ('bagi', 'ROOT', 'ROOT'),
#  ('buku', 'obj', 'bagi'),
#  ('tu', 'det', 'buku'),
#  ('kat', 'case', 'Siti'),
#  ('Siti', 'obl', 'bagi'),
#  ('semalam', 'obl:tmod', 'bagi')]
```

### Options

```python
# Return as tree structure
tree = mnlp.dependency(text, format="tree")
print(tree)
# bagi (ROOT)
# ├── Ali (nsubj)
# ├── buku (obj)
# │   └── tu (det)
# ├── Siti (obl)
# │   └── kat (case)
# └── semalam (obl:tmod)

# Return detailed token info
mnlp.dependency(text, detailed=True)
# [{'word': 'Ali', 'dep': 'nsubj', 'head': 'bagi', 'head_idx': 1, 'pos': 'PROPN'}, ...]

# Visualize (requires matplotlib)
mnlp.dependency(text, visualize=True, output="dep_tree.png")
```

!!! note "Relation Labels"
    Uses Universal Dependencies relation labels. See [UD documentation](https://universaldependencies.org/u/dep/) for full label descriptions.

---

## See Also

- [Text Processing](text-processing.md) — preprocess text before extraction
- [Advanced modules](advanced.md) — coreference resolution, discourse parsing
- [spaCy Integration](integrations.md) — use extraction modules via spaCy pipeline

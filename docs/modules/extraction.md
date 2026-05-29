# Extraction

**Pull structured data from Malaysian text — entities, grammar, keywords, and references.**

---

## Overview

Extraction modules convert unstructured text into structured representations: named entities, part-of-speech tags, dependency trees, keyword lists, and coreference chains. All modules handle Manglish and code-switched input natively.

```python
import manglish_nlp as mnlp
```

---

## Quick Start

```python
import manglish_nlp as mnlp

text = "Siti beli iPhone 15 kat Low Yat Plaza semalam, harga RM4,200"

mnlp.ner(text)
# [('Siti', 'PERSON'), ('iPhone 15', 'PRODUCT'),
#  ('Low Yat Plaza', 'LOCATION'), ('semalam', 'DATE'), ('RM4,200', 'MONEY')]

mnlp.pos(text)
# [('Siti', 'PROPN'), ('beli', 'VERB'), ('iPhone', 'PROPN'), ('15', 'NUM'),
#  ('kat', 'ADP'), ('Low', 'PROPN'), ('Yat', 'PROPN'), ('Plaza', 'PROPN'),
#  ('semalam', 'NOUN'), (',', 'PUNCT'), ('harga', 'NOUN'), ('RM4,200', 'NUM')]

mnlp.keywords(text, top_k=3)
# ['iPhone 15', 'Low Yat Plaza', 'RM4,200']
```

---

## Module Details

### `ner`

Named Entity Recognition with 7 entity types trained on Malaysian text. Handles Malay, Chinese, Indian, and mixed names with honorifics (Dato', Tan Sri, Dr.).

```python
import manglish_nlp as mnlp

mnlp.ner("Dato' Sri Ismail Sabri umum bantuan RM500 di Putrajaya")
# [('Dato' Sri Ismail Sabri', 'PERSON'), ('RM500', 'MONEY'),
#  ('Putrajaya', 'LOCATION')]
```

#### Entity Types

| Entity | Description | Examples |
|--------|-------------|---------|
| `PERSON` | Person names with titles | Siti Nurhaliza, Dr. Mahathir, Dato' Seri Anwar |
| `LOCATION` | Places, cities, addresses | KL, Bukit Bintang, Pahang, Low Yat Plaza |
| `ORG` | Organisations | Petronas, UMP, Grab, Shopee |
| `PRODUCT` | Products, brands, models | iPhone 15, Myvi, Milo, Samsung Galaxy |
| `EVENT` | Events, holidays | Hari Raya, Merdeka Day, PRU15 |
| `DATE` | Dates, times, durations | semalam, 15 Mei 2024, next week, 2 jam |
| `MONEY` | Monetary values | RM50, 3 ringgit, USD100 |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text |
| `spans` | `bool` | `False` | Return character offsets with each entity |
| `types` | `list[str]` | `None` | Filter to specific entity types only |
| `threshold` | `float` | `0.5` | Minimum confidence to include entity |

!!! example "Span Positions"
    ```python
    mnlp.ner("Ahmad kerja kat Petronas", spans=True)
    # [{'text': 'Ahmad', 'label': 'PERSON', 'start': 0, 'end': 5},
    #  {'text': 'Petronas', 'label': 'ORG', 'start': 14, 'end': 22}]
    ```

!!! tip "Malaysian Names"
    The NER model recognises Malay patronymic patterns (bin/binti), Chinese multi-character names, and Indian names. Honorifics like Tan Sri, Dato', and Tun are grouped with the name.

---

### `pos`

Part-of-speech tagging using Universal Dependencies (UD) adapted for Malay grammar. Correctly handles particles, code-switched words, and informal auxiliaries (`nak`, `dah`, `boleh`).

```python
import manglish_nlp as mnlp

mnlp.pos("Aku nak pergi makan kat kedai tu")
# [('Aku', 'PRON'), ('nak', 'AUX'), ('pergi', 'VERB'), ('makan', 'VERB'),
#  ('kat', 'ADP'), ('kedai', 'NOUN'), ('tu', 'DET')]
```

#### UD Tag Set

| Tag | Description | Malay Examples |
|-----|-------------|----------------|
| `NOUN` | Common noun | rumah, kereta, makanan |
| `VERB` | Action/state verb | makan, pergi, tidur |
| `ADJ` | Adjective | cantik, besar, sedap |
| `ADV` | Adverb | sangat, betul, cepat |
| `PRON` | Pronoun | aku, dia, mereka |
| `DET` | Determiner | tu, ini, semua |
| `ADP` | Preposition | kat, dari, untuk |
| `AUX` | Auxiliary | nak, dah, boleh, akan |
| `CONJ` | Conjunction | dan, tapi, atau |
| `PART` | Particle | la, je, kot, kan |
| `NUM` | Number | satu, 100, RM50 |
| `PUNCT` | Punctuation | ., !, ? |
| `INTJ` | Interjection | weh, eh, alamak |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `detailed` | `bool` | `False` | Include morphological features (person, number, tense) |
| `format` | `str` | `"list"` | Output format: `"list"`, `"dict"` |

!!! example "Detailed Morphological Features"
    ```python
    mnlp.pos("Mereka sedang bermain", detailed=True)
    # [('Mereka', 'PRON', {'person': 3, 'number': 'plur'}),
    #  ('sedang', 'AUX', {'aspect': 'prog'}),
    #  ('bermain', 'VERB', {'voice': 'act', 'root': 'main'})]
    ```

---

### `dependency`

Dependency parsing for Malay and Manglish sentences. Returns syntactic relations between words in Universal Dependencies format.

```python
import manglish_nlp as mnlp

mnlp.dependency("Ali bagi buku tu kat Siti semalam")
# [('Ali', 'nsubj', 'bagi'),
#  ('bagi', 'ROOT', 'ROOT'),
#  ('buku', 'obj', 'bagi'),
#  ('tu', 'det', 'buku'),
#  ('kat', 'case', 'Siti'),
#  ('Siti', 'obl', 'bagi'),
#  ('semalam', 'obl:tmod', 'bagi')]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input sentence |
| `format` | `str` | `"list"` | Output format: `"list"`, `"tree"`, `"conllu"` |
| `detailed` | `bool` | `False` | Include POS tags and head indices |
| `visualize` | `bool` | `False` | Generate dependency tree image (requires matplotlib) |

!!! example "Tree Visualisation"
    ```python
    tree = mnlp.dependency("Ali bagi buku tu kat Siti", format="tree")
    print(tree)
    # bagi (ROOT)
    # ├── Ali (nsubj)
    # ├── buku (obj)
    # │   └── tu (det)
    # └── Siti (obl)
    #     └── kat (case)
    ```

!!! note "Relation Labels"
    Uses Universal Dependencies v2 relation labels. Common ones: `nsubj`, `obj`, `obl`, `det`, `amod`, `advmod`, `case`. Full reference: [UD docs](https://universaldependencies.org/u/dep/).

---

### `coreference`

Resolve pronouns and repeated mentions across sentences. Handles gender-neutral Malay pronouns (`dia`, `ia`, `mereka`).

```python
import manglish_nlp as mnlp

text = "Ahmad jumpa Siti kat mall. Dia cakap dia nak balik awal."
mnlp.coreference(text)
# {'clusters': [
#   [('Ahmad', 0, 5), ('Dia', 26, 29)],
#   [('Siti', 12, 16), ('dia', 34, 37)]
# ]}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Multi-sentence input |
| `resolve` | `bool` | `False` | Replace pronouns with resolved entities |
| `chains` | `bool` | `False` | Return mention chains instead of clusters |

!!! example "Pronoun Resolution"
    ```python
    mnlp.coreference(text, resolve=True)
    # "Ahmad jumpa Siti kat mall. Ahmad cakap Siti nak balik awal."
    ```

!!! warning "Ambiguous Pronouns"
    Malay `dia` is gender-neutral. The model uses contextual cues (recency, topic, verb semantics) to resolve. For high-ambiguity texts, `chains=True` gives all candidates.

---

### `keywords`

Extract keywords and key phrases using multiple algorithms. Ships with a Malaysian-specific stopword list that includes particles (`la`, `je`, `kot`) and common code-switch fillers.

```python
import manglish_nlp as mnlp

article = """Kerajaan Malaysia umumkan pakej rangsangan ekonomi bernilai
RM50 bilion untuk membantu rakyat dan perniagaan kecil yang terjejas."""

mnlp.keywords(article, top_k=3, scores=True)
# [('pakej rangsangan ekonomi', 0.92),
#  ('RM50 bilion', 0.87),
#  ('perniagaan kecil', 0.79)]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `top_k` | `int` | `5` | Number of keywords to return |
| `scores` | `bool` | `False` | Include relevance scores |
| `method` | `str` | `"textrank"` | Algorithm: `"textrank"`, `"tfidf"`, `"yake"` |
| `ngram_range` | `tuple` | `(1, 3)` | Min/max n-gram size for key phrases |

#### Method Comparison

| Method | Speed | Best For |
|--------|-------|----------|
| `textrank` | Medium | General-purpose, graph-based ranking |
| `tfidf` | Fast | Large corpora, statistical weighting |
| `yake` | Slow | Single documents, unsupervised extraction |

!!! tip "N-gram Range"
    Use `ngram_range=(2, 4)` for longer key phrases; `(1, 1)` for single-word keywords only.

---

## See Also

- [Text Processing](text-processing.md) — preprocess text before extraction
- [Advanced](advanced.md) — coreference resolution, discourse parsing
- [spaCy Integration](integrations.md#spacy_integration) — use extraction modules inside a spaCy pipeline
- [Embeddings](data.md#embeddings) — use sentence embeddings for semantic keyword extraction

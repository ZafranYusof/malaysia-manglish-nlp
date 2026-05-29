# Word Embeddings

**Dense vector representations trained on 10M+ Malaysian texts  -  Word2Vec and FastText.**

---

## Why word embeddings?

Semantic search, similarity computation, text classification features, and linguistic analysis. Standard embeddings (GloVe, fastText English) don't capture Malaysian word semantics  -  "sedap" and "lazat" should be close vectors, but they're absent from English-trained models.

manglish-nlp provides embeddings trained specifically on Malaysian text including Manglish, social media, and news.

---

## Load module

```python
from manglish_nlp import word_embeddings

# Load Word2Vec model
model = word_embeddings.load_word2vec()

# Or load FastText model
model = word_embeddings.load_fasttext()
```

!!! warning "Model download"
    Embedding models are large (~200MB). First load will download automatically. Subsequent loads use cache.

---

## Basic usage

### Get word vector

```python
model = word_embeddings.load_word2vec()

vec = model.get_vector("makan")
print(vec.shape)
# (300,)

print(vec[:5])
# [ 0.234, -0.156,  0.089,  0.412, -0.067]
```

### Check if word exists

```python
model.has_word("sedap")
# True

model.has_word("xyzzyplugh")
# False
```

---

## Similarity search

### Find most similar words

```python
model.most_similar("makan", topn=10)
# [('minum', 0.82), ('nasi', 0.79), ('masak', 0.76),
#  ('sedap', 0.74), ('kedai', 0.71), ('restoran', 0.69),
#  ('lauk', 0.67), ('lemak', 0.65), ('goreng', 0.63),
#  ('roti', 0.61)]

model.most_similar("sedap", topn=5)
# [('lazat', 0.88), ('enak', 0.84), ('delicious', 0.76),
#  ('nyaman', 0.72), ('best', 0.69)]

model.most_similar("kereta", topn=5)
# [('Myvi', 0.81), ('Proton', 0.78), ('Honda', 0.76),
#  ('Toyota', 0.74), ('Perodua', 0.72)]
```

### Word similarity score

```python
model.similarity("makan", "minum")
# 0.82

model.similarity("sedap", "lazat")
# 0.88

model.similarity("makan", "kereta")
# 0.12
```

---

## Word arithmetic

Classic embedding operations:

```python
# king - man + woman = queen equivalent in Malay
model.analogy("raja", "lelaki", "perempuan")
# 'ratu'

# Positive analogy
model.analogy("sedap", "makanan", "minuman")
# 'segat'

# Vector math
vec = model.get_vector("nasi") + model.get_vector("lemak")
model.most_similar_vector(vec, topn=3)
# [('nasi_lemak', 0.91), ('rendang', 0.72), ('sambal', 0.68)]
```

---

## Sentence and document embeddings

For sentence-level embeddings, use the embeddings module:

```python
from manglish_nlp import embeddings

# Fast mode (averaged word vectors)
vec = embeddings.encode("Saya suka makan nasi lemak")
print(vec.shape)
# (768,)

# Accurate mode (transformer-based, requires [ml])
vec = embeddings.encode("Saya suka makan nasi lemak", mode="accurate")
```

### Sentence similarity

```python
from manglish_nlp import similarity

similarity.cosine(
    "Saya suka makan nasi lemak",
    "I like eating coconut rice"
)
# 0.78

similarity.jaccard(
    "Sedap gila makanan tu",
    "Makanan tu sedap gila"
)
# 0.83
```

---

## Visualization

Plot word embeddings in 2D:

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model = word_embeddings.load_word2vec()

words = ["makan", "minum", "nasi", "roti", "kereta", "motor", "bas", "sedap", "lazat"]
vectors = [model.get_vector(w) for w in words]

pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors)

plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(reduced[i][0], reduced[i][1])
    plt.annotate(word, (reduced[i][0], redecud[i][1]))
plt.title("Word Embeddings (PCA)")
plt.savefig("embeddings.png")
```

!!! tip "Dependencies for visualization"
    ```bash
    pip install matplotlib scikit-learn
    ```

---

## CLI usage

Word embeddings are primarily a Python API. For CLI similarity:

```bash
# Text similarity (uses embeddings internally)
$ mnlp similarity "makan nasi" "minum air"
cosine: 0.72

# Keywords (uses TF-IDF, related)
$ mnlp keywords "Nasi lemak sedap gila kat kedai mamak"
```

---

## How it works

1. **Word2Vec**  -  CBOW architecture, 300 dimensions, trained on 10M+ Malaysian texts
2. **FastText**  -  subword model, handles OOV words through character n-grams
3. **Sentence embeddings**  -  averaged word vectors (fast) or transformer (accurate)
4. **Similarity**  -  cosine distance between vectors

Training data includes: Malay news, social media posts, Wikipedia BM, forum posts, and Manglish text.

---

## Performance

| Metric | Word2Vec | FastText |
|--------|----------|----------|
| Dimensions | 300 | 300 |
| Vocabulary | 150,000 | 200,000+ |
| Analogy accuracy | 62.3% | 68.7% |
| Similarity correlation | 0.74 | 0.79 |
| OOV handling | No | Yes |
| Load time | ~2s | ~4s |

---

## See also

- [Summarization](summarization.md)  -  uses embeddings for semantic similarity
- [Question Answering](qa.md)  -  uses embeddings for context matching
- [Code-Switching](code-switching.md)  -  embeddings help detect language boundaries
- [API Reference](../api-reference.md#embeddings)  -  full function signature

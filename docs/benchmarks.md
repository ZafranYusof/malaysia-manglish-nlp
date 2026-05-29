# Benchmarks

Performance benchmarks for manglish-nlp on Malaysian text datasets.

---

## Throughput

Measured on Intel i7-12700H, 16GB RAM, Python 3.11. Single-threaded unless noted.

| Module | Texts/sec | Latency (ms) | Notes |
|--------|-----------|--------------|-------|
| `clean` | 89,000 | 0.01 | Regex-based |
| `normalize` | 45,000 | 0.02 | Dictionary lookup |
| `tokenize` | 67,000 | 0.01 | Rule-based |
| `stem` | 52,000 | 0.02 | Rule-based |
| `sentiment` (fast) | 23,400 | 0.04 | Statistical model |
| `sentiment` (accurate) | 1,200 | 0.83 | Transformer |
| `ner` (fast) | 15,800 | 0.06 | CRF model |
| `ner` (accurate) | 890 | 1.12 | Transformer |
| `language` | 38,000 | 0.03 | N-gram based |
| `embeddings` (fast) | 8,500 | 0.12 | Lightweight encoder |
| `embeddings` (accurate) | 650 | 1.54 | Transformer |
| `translate` | 180 | 5.56 | Seq2seq model |
| `summarize` | 95 | 10.5 | Abstractive |

!!! info "Batch Processing"
    Throughput increases significantly with batching. The `sentiment` module achieves 23,400 texts/sec at batch_size=32 vs 2,300 texts/sec at batch_size=1.

---

## Accuracy

### Sentiment Analysis

Evaluated on Malaysian social media test set (5,000 samples).

| Model | Accuracy | F1 (macro) | F1 (weighted) |
|-------|----------|------------|---------------|
| manglish-nlp (fast) | 84.2% | 82.1% | 84.0% |
| manglish-nlp (accurate) | 89.7% | 88.3% | 89.5% |
| Malaya (sentiment) | 87.1% | 85.4% | 86.9% |
| TextBlob (baseline) | 52.3% | 41.2% | 49.8% |

### Named Entity Recognition

Evaluated on Malaysian NER dataset (2,000 sentences).

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| manglish-nlp (fast) | 79.4% | 76.8% | 78.1% |
| manglish-nlp (accurate) | 86.2% | 84.1% | 85.1% |
| Malaya (NER) | 83.5% | 81.9% | 82.7% |
| spaCy ms (blank) | 61.2% | 58.7% | 59.9% |

### Language Detection

Evaluated on code-switched Malaysian text (3,000 samples).

| Model | Accuracy | Notes |
|-------|----------|-------|
| manglish-nlp | 94.8% | Handles code-switching |
| langdetect | 67.3% | Fails on mixed text |
| fasttext | 78.1% | Better but still struggles |

---

## Comparison with Malaya

[Malaya](https://github.com/huseinzol05/malaya) is the most established Malaysian NLP library. Here's how manglish-nlp compares:

| Aspect | manglish-nlp | Malaya |
|--------|-------------|--------|
| **Focus** | Manglish (informal) | Formal BM |
| **Dependencies** | Zero (core) | Heavy (TensorFlow/PyTorch) |
| **Install size** | ~15MB (core) | ~500MB+ |
| **Startup time** | <1s | 5-15s |
| **Throughput** | 23k+ texts/sec | ~500 texts/sec |
| **Code-switching** | Native support | Limited |
| **Informal text** | Optimized | Struggles |
| **Formal BM** | Good | Excellent |
| **Model variety** | Focused | Extensive |
| **API style** | Simple functions | Class-based |

!!! tip "When to Use Which"
    - **manglish-nlp**: Social media, chat data, informal text, lightweight deployment, code-switched content
    - **Malaya**: Formal documents, news articles, when you need maximum model variety, research applications

---

## Memory Usage

| Operation | Peak RAM | Notes |
|-----------|----------|-------|
| Core import | 12 MB | Zero-dep modules only |
| Full import | 45 MB | All modules loaded |
| Sentiment (fast) | 28 MB | Statistical model |
| Sentiment (accurate) | 380 MB | Transformer model |
| Embeddings (fast) | 95 MB | Lightweight encoder |
| Embeddings (accurate) | 420 MB | Full transformer |
| Word embeddings (300d) | 1.2 GB | Full vocabulary |
| Word embeddings (100d) | 400 MB | Reduced dimensions |

---

## How to Run Benchmarks

```bash
# Install benchmark dependencies
pip install manglish-nlp[benchmark]

# Run all benchmarks
mnlp benchmark --all

# Specific module
mnlp benchmark sentiment --samples 10000

# Compare models
mnlp benchmark sentiment --models fast,accurate --samples 5000

# Output formats
mnlp benchmark --all --format json --output results.json
mnlp benchmark --all --format table
```

### Custom Benchmark

```python
from manglish_nlp import profiler

# Benchmark your own data
texts = load_your_data()

results = profiler.benchmark(
    mnlp.sentiment,
    texts,
    batch_sizes=[1, 8, 32, 64, 128],
    warmup=100
)

print(results)
# batch_size=1:   2,340 texts/sec
# batch_size=8:   12,400 texts/sec
# batch_size=32:  23,100 texts/sec
# batch_size=64:  24,800 texts/sec
# batch_size=128: 25,100 texts/sec
```

### Hardware Scaling

```bash
# Multi-core benchmark
mnlp benchmark sentiment --workers 1,2,4,8

# GPU benchmark (requires [ml])
mnlp benchmark sentiment --device cuda --batch-size 64
```

---

## Reproducibility

All benchmarks use:
- **Dataset**: Malaysian Social Media Benchmark v2.1
- **Hardware**: Intel i7-12700H, 16GB DDR5, no GPU
- **Python**: 3.11.7
- **manglish-nlp**: latest stable release

To reproduce:
```bash
git clone https://github.com/ZafranYusof/manglish-nlp
cd manglish-nlp
pip install -e .[benchmark]
python benchmarks/run_all.py
```

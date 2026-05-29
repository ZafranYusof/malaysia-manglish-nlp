# Tools & Utilities

Infrastructure modules for building production NLP pipelines.

---

## ocr_normalize

Post-process OCR output from Malaysian documents — fix common OCR errors in Malay text.

```python
import manglish_nlp as mnlp

# Raw OCR output with typical errors
ocr_text = "Kerajaan Ma1aysia te1ah mengumumkan po1isi baru"
result = mnlp.ocr_normalize(ocr_text)
print(result)
# "Kerajaan Malaysia telah mengumumkan polisi baru"
```

### Options

```python
# Specify OCR engine for targeted fixes
mnlp.ocr_normalize(text, engine="tesseract")
mnlp.ocr_normalize(text, engine="easyocr")

# Fix common substitutions (1→l, 0→o, rn→m)
mnlp.ocr_normalize(text, fix_substitutions=True)

# Reconstruct broken lines
mnlp.ocr_normalize(text, fix_linebreaks=True)

# Confidence-based correction (only fix likely errors)
mnlp.ocr_normalize(text, threshold=0.8)
```

---

## pipeline

Chain multiple modules into a reusable processing pipeline.

```python
from manglish_nlp import Pipeline

# Define pipeline
pipe = Pipeline([
    'clean',
    'normalize',
    'tokenize',
    'sentiment'
])

# Process single text
result = pipe("Weh @ahmad best gila mknn tu!! 🔥🔥")
print(result)
# {'tokens': ['weh', 'best', 'gila', 'makanan', 'tu'],
#  'sentiment': {'label': 'positive', 'score': 0.93}}
```

### Options

```python
# Pipeline with custom config per step
pipe = Pipeline([
    ('clean', {'keep_emoji': False}),
    ('normalize', {'aggressive': True}),
    ('sentiment', {'detailed': True})
])

# Batch processing with parallelism
results = pipe.batch(texts, n_jobs=4)

# Save/load pipeline
pipe.save("my_pipeline.json")
loaded = Pipeline.load("my_pipeline.json")

# Conditional steps
pipe = Pipeline([
    'clean',
    ('normalize', {'condition': lambda x: len(x) > 10}),
    'sentiment'
])

# Get intermediate results
result = pipe(text, return_all=True)
# {'clean': '...', 'normalize': '...', 'sentiment': {...}}
```

!!! tip "Performance"
    Pipelines avoid redundant computation by passing results between steps efficiently. Use `pipe.batch()` with `n_jobs` for parallel processing of large datasets.

---

## calibration

Calibrate model confidence scores to produce reliable probability estimates.

```python
# Calibrate sentiment model
calibrator = mnlp.calibration("sentiment", method="platt")

# Apply calibration
raw_result = mnlp.sentiment(text)
# {'label': 'positive', 'score': 0.99}  (overconfident)

calibrated = calibrator.calibrate(raw_result)
# {'label': 'positive', 'score': 0.82}  (realistic)
```

### Options

```python
# Calibration methods
mnlp.calibration(model, method="platt")          # Platt scaling
mnlp.calibration(model, method="isotonic")       # Isotonic regression
mnlp.calibration(model, method="temperature")    # Temperature scaling

# Evaluate calibration quality
calibrator.ece()  # Expected Calibration Error
# 0.03 (lower is better)

# Calibrate on custom data
calibrator.fit(texts, labels)
```

---

## evaluate

Evaluate NLP model performance with Malaysian-specific metrics.

```python
from manglish_nlp import evaluate

# Evaluate sentiment model
results = evaluate.sentiment(
    texts=test_texts,
    labels=test_labels,
    model=mnlp.sentiment
)
print(results)
# {'accuracy': 0.87, 'f1_macro': 0.85, 'f1_weighted': 0.87,
#  'per_class': {'positive': 0.89, 'negative': 0.84, 'neutral': 0.81}}
```

### Options

```python
# NER evaluation
evaluate.ner(texts, gold_entities, model=mnlp.ner)
# {'precision': 0.82, 'recall': 0.79, 'f1': 0.80, 'per_type': {...}}

# Cross-validation
evaluate.cross_validate(texts, labels, model=mnlp.sentiment, folds=5)

# Error analysis
errors = evaluate.errors(texts, labels, model=mnlp.sentiment)
# [{'text': '...', 'predicted': 'positive', 'actual': 'negative', 'confidence': 0.51}]

# Generate classification report
evaluate.report(texts, labels, model=mnlp.sentiment, output="report.html")
```

---

## hybrid_ml

Combine rule-based and ML approaches for optimal accuracy-speed tradeoff.

```python
from manglish_nlp import hybrid_ml

# Create hybrid model
model = hybrid_ml.create(
    task="sentiment",
    rules=my_rules,          # Fast rule-based for clear cases
    ml_model=transformer,    # ML for ambiguous cases
    threshold=0.7            # Use ML when rules are uncertain
)

result = model("Best gila!")
# Uses rules (clear positive signal) → fast

result = model("Hmm ok la tu...")
# Falls through to ML (ambiguous) → accurate
```

### Options

```python
# Custom routing logic
model = hybrid_ml.create(
    task="sentiment",
    router=lambda text, rule_conf: rule_conf < 0.7,  # When to use ML
    rules=my_rules,
    ml_model=transformer
)

# Performance stats
model.stats()
# {'rule_hits': 7823, 'ml_hits': 2177, 'avg_latency_ms': 3.2}
```

---

## tuning

Hyperparameter tuning and model selection for Malaysian NLP tasks.

```python
from manglish_nlp import tuning

# Auto-tune sentiment model
best_config = tuning.optimize(
    task="sentiment",
    train_data=train_texts,
    train_labels=train_labels,
    eval_data=eval_texts,
    eval_labels=eval_labels,
    n_trials=50
)
print(best_config)
# {'model': 'transformer', 'lr': 2e-5, 'batch_size': 32, 'epochs': 3}
```

### Options

```python
# Grid search
tuning.grid_search(task, param_grid={...}, data=data)

# Random search
tuning.random_search(task, param_distributions={...}, n_trials=100)

# Bayesian optimization
tuning.bayesian(task, search_space={...}, n_trials=50)

# Early stopping
tuning.optimize(task, data=data, early_stopping=True, patience=5)
```

---

## profiler

Profile NLP pipeline performance — identify bottlenecks and optimize throughput.

```python
from manglish_nlp import profiler

# Profile a pipeline
with profiler.trace() as p:
    for text in texts[:100]:
        mnlp.sentiment(text)

p.report()
# ┌─────────────┬──────────┬─────────┬──────────┐
# │ Step        │ Avg (ms) │ Total   │ % Time   │
# ├─────────────┼──────────┼─────────┼──────────┤
# │ tokenize    │ 0.3      │ 30ms    │ 12%      │
# │ encode      │ 1.8      │ 180ms   │ 72%      │
# │ classify    │ 0.4      │ 40ms    │ 16%      │
# └─────────────┴──────────┴─────────┴──────────┘
```

### Options

```python
# Memory profiling
with profiler.memory() as p:
    mnlp.embeddings(large_corpus)
p.peak_mb  # 245.3

# Throughput benchmark
profiler.benchmark(mnlp.sentiment, texts, batch_sizes=[1, 8, 32, 64])
# {1: '2,300 texts/sec', 8: '12,400 texts/sec',
#  32: '23,100 texts/sec', 64: '24,800 texts/sec'}

# Compare models
profiler.compare([model_a, model_b], texts)
```

---

## cache

Cache expensive NLP operations for repeated processing.

```python
from manglish_nlp import cache

# Enable caching for embeddings
@cache.memoize(ttl=3600)
def get_embedding(text):
    return mnlp.embeddings(text)

# Or use built-in cache
mnlp.sentiment(text, cache=True)
mnlp.sentiment(text, cache=True)  # Returns cached result instantly
```

### Options

```python
# Configure cache backend
cache.configure(
    backend="disk",          # 'memory', 'disk', 'redis'
    max_size="1GB",
    ttl=86400               # 24 hours
)

# Cache stats
cache.stats()
# {'hits': 4521, 'misses': 892, 'hit_rate': 0.84, 'size_mb': 123}

# Clear cache
cache.clear()
cache.clear(module="sentiment")  # Clear only sentiment cache

# Warm cache
cache.warm(texts, modules=["sentiment", "ner"])
```

!!! tip "When to Cache"
    Cache is most effective for embedding and generation operations. Lightweight operations (tokenize, clean) are fast enough that caching overhead isn't worth it.

---

## See Also

- [Integrations](integrations.md) — deploy pipelines as APIs
- [Benchmarks](../benchmarks.md) — performance numbers

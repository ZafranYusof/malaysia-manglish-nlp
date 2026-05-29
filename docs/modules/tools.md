# Tools & Utilities

**Infrastructure for production NLP  -  pipelines, caching, profiling, evaluation, and hybrid routing.**

---

## Overview

Tools modules handle the engineering side: chaining modules into reusable pipelines, caching expensive operations, profiling latency and memory, evaluating model accuracy, tuning hyperparameters, and combining rule-based and ML approaches.

```python
import manglish_nlp as mnlp
from manglish_nlp import Pipeline, cache, profiler, evaluate
```

---

## Quick Start

```python
from manglish_nlp import Pipeline

# Build a reusable pipeline
pipe = Pipeline([
    'clean',
    'normalize',
    'sentiment'
])

# Process
result = pipe("Weh @ahmad best gila mknn tu!! 🔥🔥")
# {'sentiment': {'label': 'positive', 'score': 0.93}}

# Batch with parallelism
results = pipe.batch(texts, n_jobs=4)

# Save for later
pipe.save("sentiment_pipeline.json")
```

---

## Module Details

### `ocr_normalize`

Post-process OCR output from Malaysian documents. Fixes common OCR artefacts: character substitutions (`1`→`l`, `0`→`o`, `rn`→`m`), broken line breaks, and Malay-specific patterns.

```python
import manglish_nlp as mnlp

ocr_text = "Kerajaan Ma1aysia te1ah mengumumkan po1isi baru"
mnlp.ocr_normalize(ocr_text)
# "Kerajaan Malaysia telah mengumumkan polisi baru"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Raw OCR output |
| `engine` | `str` | `"auto"` | OCR engine hint: `"tesseract"`, `"easyocr"`, `"auto"` |
| `fix_substitutions` | `bool` | `True` | Fix common character substitutions |
| `fix_linebreaks` | `bool` | `False` | Reconstruct broken line wraps |
| `threshold` | `float` | `0.5` | Only correct tokens below this OCR confidence |

!!! tip "OCR Engine Targeting"
    Specify `engine="tesseract"` or `"easyocr"` for engine-specific fix patterns. Tesseract commonly confuses `rn`/`m` and `cl`/`d`; EasyOCR tends to merge adjacent words.

---

### `pipeline`

Chain multiple modules into a reusable, serialisable processing pipeline. Supports custom config per step, conditional steps, and batch parallelism.

```python
from manglish_nlp import Pipeline

pipe = Pipeline([
    'clean',
    'normalize',
    'tokenize',
    'sentiment'
])

result = pipe("Weh @ahmad best gila mknn tu!! 🔥🔥")
# {'tokens': ['weh', 'best', 'gila', 'makanan', 'tu'],
#  'sentiment': {'label': 'positive', 'score': 0.93}}
```

#### Parameters (Pipeline constructor)

| Parameter | Type | Description |
|-----------|------|-------------|
| `steps` | `list` | Module names or `(name, config)` tuples |

#### Parameters (Pipeline methods)

| Method | Parameters | Description |
|--------|-----------|-------------|
| `__call__(text)` | `return_all=False` | Process single text |
| `batch(texts)` | `n_jobs=1` | Parallel batch processing |
| `save(path)` |  -  | Serialise to JSON |
| `Pipeline.load(path)` |  -  | Load from JSON |

!!! example "Custom Config Per Step"
    ```python
    pipe = Pipeline([
        ('clean', {'keep_emoji': False}),
        ('normalize', {'aggressive': True}),
        ('sentiment', {'detailed': True})
    ])
    ```

!!! example "Conditional Steps"
    ```python
    pipe = Pipeline([
        'clean',
        ('normalize', {'condition': lambda x: len(x) > 10}),
        'sentiment'
    ])
    ```

!!! example "Intermediate Results"
    ```python
    result = pipe(text, return_all=True)
    # {'clean': '...', 'normalize': '...', 'sentiment': {...}}
    ```

!!! tip "Performance"
    Pipelines avoid redundant computation by passing results between steps. `pipe.batch(texts, n_jobs=4)` uses process-level parallelism for CPU-bound modules.

---

### `calibration`

Calibrate model confidence scores to produce reliable probability estimates. Essential for production systems that use confidence thresholds for routing or escalation.

```python
from manglish_nlp import calibration

calibrator = calibration("sentiment", method="platt")

# Before calibration (overconfident)
mnlp.sentiment("Best gila!")
# {'label': 'positive', 'score': 0.99}

# After calibration (realistic)
calibrator.calibrate(mnlp.sentiment("Best gila!"))
# {'label': 'positive', 'score': 0.82}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Module name to calibrate |
| `method` | `str` | `"platt"` | Method: `"platt"`, `"isotonic"`, `"temperature"` |

#### Methods

| Method | Description |
|--------|-------------|
| `calibrate(result)` | Apply calibration to a model output |
| `fit(texts, labels)` | Fit calibrator on labelled data |
| `ece()` | Expected Calibration Error (lower = better) |

!!! example "Evaluate Calibration Quality"
    ```python
    calibrator.ece()
    # 0.03  (near-perfect calibration)
    ```

---

### `evaluate`

Evaluate NLP model performance with Malaysian-specific metrics. Covers classification, NER, and cross-validation with error analysis.

```python
from manglish_nlp import evaluate

results = evaluate.sentiment(
    texts=test_texts,
    labels=test_labels,
    model=mnlp.sentiment
)
# {'accuracy': 0.87, 'f1_macro': 0.85, 'f1_weighted': 0.87,
#  'per_class': {'positive': 0.89, 'negative': 0.84, 'neutral': 0.81}}
```

#### Methods

| Method | Description |
|--------|-------------|
| `evaluate.sentiment(texts, labels, model)` | Classification metrics |
| `evaluate.ner(texts, gold, model)` | NER precision/recall/F1 per entity type |
| `evaluate.cross_validate(texts, labels, model, folds)` | K-fold cross-validation |
| `evaluate.errors(texts, labels, model)` | Misclassified samples with confidence |
| `evaluate.report(texts, labels, model, output)` | Generate HTML classification report |

!!! example "Error Analysis"
    ```python
    errors = evaluate.errors(test_texts, test_labels, model=mnlp.sentiment)
    # [{'text': 'Hmm ok la tu...', 'predicted': 'positive',
    #   'actual': 'negative', 'confidence': 0.51}]
    ```

---

### `hybrid_ml`

Combine rule-based and ML approaches. Routes clear cases to fast rules, ambiguous cases to ML models. Ideal for latency-sensitive production systems.

```python
from manglish_nlp import hybrid_ml

model = hybrid_ml.create(
    task="sentiment",
    rules=my_rules,
    ml_model=transformer,
    threshold=0.7
)

# Clear case → rule fires (fast)
model("Best gila!")
# Uses rules → 0.2ms

# Ambiguous → ML fallback (accurate)
model("Hmm ok la tu...")
# Falls through to ML → 15ms
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `str` | NLP task name |
| `rules` | `callable` | Fast rule-based function |
| `ml_model` | `callable` | ML model for fallback |
| `threshold` | `float` | Use ML when rule confidence < threshold |
| `router` | `callable` | Custom routing function (overrides threshold) |

!!! example "Performance Stats"
    ```python
    model.stats()
    # {'rule_hits': 7823, 'ml_hits': 2177, 'avg_latency_ms': 3.2}
    ```

---

### `tuning`

Hyperparameter tuning for Malaysian NLP tasks. Supports grid search, random search, and Bayesian optimisation.

```python
from manglish_nlp import tuning

best_config = tuning.optimize(
    task="sentiment",
    train_data=train_texts,
    train_labels=train_labels,
    eval_data=eval_texts,
    eval_labels=eval_labels,
    n_trials=50
)
# {'model': 'transformer', 'lr': 2e-5, 'batch_size': 32, 'epochs': 3}
```

#### Methods

| Method | Speed | Best For |
|--------|-------|----------|
| `grid_search(task, param_grid, data)` | Slow | Small search spaces, exhaustive coverage |
| `random_search(task, distributions, n_trials)` | Medium | Large search spaces, quick exploration |
| `bayesian(task, search_space, n_trials)` | Fast | Production tuning, expensive evaluations |
| `optimize(...)` | Adaptive | Auto-selects best strategy |

!!! tip "Early Stopping"
    ```python
    tuning.optimize(task, data=data, early_stopping=True, patience=5)
    # Stops when eval metric doesn't improve for 5 trials
    ```

---

### `profiler`

Profile NLP pipeline performance  -  identify latency bottlenecks, memory usage, and throughput limits.

```python
from manglish_nlp import profiler

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

#### Methods

| Method | Description |
|--------|-------------|
| `profiler.trace()` | Context manager for latency profiling |
| `profiler.memory()` | Context manager for peak memory tracking |
| `profiler.benchmark(fn, data, batch_sizes)` | Throughput at various batch sizes |
| `profiler.compare(models, data)` | Side-by-side model comparison |

!!! example "Memory Profiling"
    ```python
    with profiler.memory() as p:
        mnlp.embeddings(large_corpus)
    p.peak_mb
    # 245.3
    ```

!!! example "Throughput Benchmark"
    ```python
    profiler.benchmark(mnlp.sentiment, texts, batch_sizes=[1, 8, 32, 64])
    # {1: '2,300 texts/sec', 8: '12,400 texts/sec',
    #  32: '23,100 texts/sec', 64: '24,800 texts/sec'}
    ```

---

### `cache`

Cache expensive NLP operations. Supports memory, disk, and Redis backends with TTL, warm-up, and per-module clearing.

```python
from manglish_nlp import cache

# Built-in cache flag
mnlp.sentiment("Best gila!", cache=True)    # computes
mnlp.sentiment("Best gila!", cache=True)    # cached  -  instant

# Decorator for custom functions
@cache.memoize(ttl=3600)
def get_embedding(text):
    return mnlp.embeddings(text)
```

#### Parameters (cache.configure)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"memory"` | `"memory"`, `"disk"`, or `"redis"` |
| `max_size` | `str` | `"512MB"` | Maximum cache size |
| `ttl` | `int` | `3600` | Time-to-live in seconds |

#### Methods

| Method | Description |
|--------|-------------|
| `cache.stats()` | Hit rate, size, hit/miss counts |
| `cache.clear()` | Clear entire cache |
| `cache.clear(module=)` | Clear specific module cache |
| `cache.warm(texts, modules=)` | Pre-populate cache |

!!! example "Cache Stats"
    ```python
    cache.stats()
    # {'hits': 4521, 'misses': 892, 'hit_rate': 0.84, 'size_mb': 123}
    ```

!!! tip "When to Cache"
    - **Always**: embeddings, generation, translation (expensive, repeated)
    - **Sometimes**: sentiment, NER (moderate cost, high repeat rate)
    - **Skip**: tokenize, clean (sub-ms, caching overhead exceeds savings)

---

## See Also

- [Integrations](integrations.md)  -  deploy pipelines as REST APIs
- [Analysis](analysis.md)  -  modules commonly used inside pipelines
- [Benchmarks](../benchmarks.md)  -  full performance numbers across modules

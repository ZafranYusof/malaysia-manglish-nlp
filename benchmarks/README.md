# Benchmarks: manglish-nlp vs Malaya

## Purpose

Head-to-head comparison of **manglish-nlp** (rule-based, zero-dep) vs **Malaya** (deep learning, transformer-based) across 7 core NLP tasks on Malaysian text.

This benchmark exists to:
- Quantify where manglish-nlp excels (speed, Manglish-specific text, zero setup)
- Acknowledge where Malaya excels (formal BM, deep understanding, research-grade)
- Help users choose the right tool for their use case

## Tasks Compared

| # | Task | Test Cases | What We Measure |
|---|------|-----------|-----------------|
| 1 | Sentiment Analysis | 52 | Positive/negative/neutral classification |
| 2 | POS Tagging | 51 | Part-of-speech tag accuracy per token |
| 3 | NER | 51 | Named entity detection (PERSON, ORG, LOC) |
| 4 | Stemming | 51 | Root word extraction accuracy |
| 5 | Normalization | 51 | Slang/abbreviation expansion |
| 6 | Language Detection | 51 | ms/en/manglish classification |
| 7 | Tokenization | 51 | Token boundary accuracy |

## How to Run

```bash
# Run all benchmarks
python benchmarks/malaya_comparison.py

# Run specific tasks
python benchmarks/malaya_comparison.py --tasks sentiment ner stemming

# Run and save results
python benchmarks/malaya_comparison.py --save

# Custom output path
python benchmarks/malaya_comparison.py --save --output results/my_run.md
```

## Prerequisites

**Required:**
- Python 3.8+
- manglish-nlp (this package)

**Optional (for comparison):**
- malaya (`pip install malaya`)
- PyTorch or TensorFlow (for Malaya's transformer models)

If Malaya is not installed, the benchmark runs manglish-nlp only and marks Malaya columns as "N/A (not installed)".

## Methodology

### Test Case Design
- 50+ test cases per task, hand-crafted
- Mix of formal Malay, informal Malay, and Manglish (code-switched)
- Emphasis on real-world Malaysian internet text (social media, chat, forums)
- Expected outputs verified by native speakers

### Metrics
- **Accuracy:** Percentage of correct predictions vs expected output
- **Speed:** Total and per-item time using `time.perf_counter()`
- **Memory:** Peak memory via `tracemalloc`

### Execution
- Each library processes the same test cases in the same order
- Timing excludes model loading for Malaya (first call warms up)
- Errors are caught gracefully — a crash counts as incorrect, not a benchmark failure

## Fairness Disclaimer

This benchmark is designed to be transparent, not to declare a "winner":

### Where Malaya is expected to be better:
- **Formal Malay text** — trained on news, Wikipedia, formal documents
- **Complex sentences** — transformer models understand context better
- **Rare words** — subword tokenization handles unseen words
- **Research tasks** — more comprehensive API, more models available

### Where manglish-nlp is expected to be better:
- **Speed** — rule-based, no model loading, instant inference
- **Memory** — zero dependencies, no GPU needed
- **Manglish text** — specifically designed for code-switched MY text
- **Slang/abbreviations** — hand-crafted normalization rules
- **Dialect awareness** — handles regional variations
- **Startup time** — no model download, no warm-up

### Test set bias:
The test cases lean toward informal/Manglish text (social media, chat). This inherently favors manglish-nlp. A formal BM news corpus would likely favor Malaya. Both are valid use cases.

### Different design philosophies:
- **Malaya:** Research-grade, comprehensive, GPU-accelerated, formal BM focus
- **manglish-nlp:** Lightweight, fast, practical, informal MY text focus

Choose based on your use case, not benchmark numbers alone.

## Results

See [RESULTS.md](./RESULTS.md) for the latest benchmark run.

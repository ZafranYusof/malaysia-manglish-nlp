# manglish-nlp

**Full NLP toolkit for Malaysian Manglish** — built for the way Malaysians actually write and speak.

---

## What is manglish-nlp?

`manglish-nlp` is a comprehensive Natural Language Processing library designed specifically for Malaysian Manglish — the unique blend of Bahasa Melayu, English, Chinese, Tamil, and local slang that Malaysians use daily in texts, social media, and casual conversation.

Unlike general-purpose NLP tools that struggle with code-switching and local dialects, manglish-nlp understands Malaysian linguistic patterns natively.

## Key Features

- **51 modules** covering text processing, sentiment analysis, NER, translation, and more
- **Zero external dependencies** for core functionality — lightweight and fast
- **23,000+ texts/sec** processing throughput on standard hardware
- **Code-switching aware** — handles BM/English/Chinese mixing naturally
- **CLI included** — use from terminal without writing Python
- **Pipeline API** — chain modules together for complex workflows

## Quick Install

```bash
pip install manglish-nlp
```

## Quick Example

```python
import manglish_nlp as mnlp

# Sentiment analysis
result = mnlp.sentiment("Weh best gila makanan kat sini!")
print(result)
# {'label': 'positive', 'score': 0.94}

# Normalize Manglish text
clean = mnlp.normalize("xpe la bro, aku ok je")
print(clean)
# "takpe la bro, aku ok je"

# Named Entity Recognition
entities = mnlp.ner("Ahmad kerja kat Petronas Tower KL")
print(entities)
# [('Ahmad', 'PERSON'), ('Petronas Tower', 'ORG'), ('KL', 'LOCATION')]
```

## Documentation Sections

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Installation, first steps, CLI usage |
| [Modules](modules/index.md) | All 51 modules grouped by category |
| [API Reference](api-reference.md) | Full function signatures and parameters |
| [Benchmarks](benchmarks.md) | Performance numbers and comparisons |
| [Contributing](contributing.md) | How to contribute to the project |

---

!!! tip "New to manglish-nlp?"
    Start with the [Getting Started](getting-started.md) guide for a quick walkthrough of installation and basic usage.

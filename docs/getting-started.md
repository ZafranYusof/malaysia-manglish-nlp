# Getting Started

Get up and running with manglish-nlp in under 5 minutes.

---

## Installation

### Basic Install

```bash
pip install manglish-nlp
```

### With Extras

```bash
# Include ML model support (transformers backend)
pip install manglish-nlp[ml]

# Include spaCy integration
pip install manglish-nlp[spacy]

# Include API server (FastAPI)
pip install manglish-nlp[api]

# Everything
pip install manglish-nlp[all]
```

!!! note "Python Version"
    manglish-nlp requires Python 3.9 or higher.

---

## First Example

```python
import manglish_nlp as mnlp

# Analyze sentiment
text = "Sedap gila nasi lemak kat kedai tu!"
result = mnlp.sentiment(text)
print(result)
# {'label': 'positive', 'score': 0.96}
```

### Text Normalization

```python
# Normalize informal Manglish spelling
text = "xpe la bro, aku nk g mkn jap lg"
normalized = mnlp.normalize(text)
print(normalized)
# "takpe la bro, aku nak pergi makan jap lagi"
```

### Language Detection

```python
# Detect language mix in text
text = "Eh jom la we go makan, I lapar gila already"
lang = mnlp.language(text)
print(lang)
# {'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

### Named Entity Recognition

```python
entities = mnlp.ner("Siti beli iPhone kat Low Yat Plaza semalam")
print(entities)
# [('Siti', 'PERSON'), ('iPhone', 'PRODUCT'), ('Low Yat Plaza', 'LOCATION')]
```

---

## CLI Usage

manglish-nlp includes a command-line interface for quick processing without writing Python.

### Basic Commands

```bash
# Sentiment analysis
mnlp sentiment "Best gila movie tu!"
# positive (0.92)

# Normalize text
mnlp normalize "aku xfhm ape ko ckp"
# "aku tak faham apa kau cakap"

# NER
mnlp ner "Ali kerja kat Grab Malaysia"
# Ali (PERSON), Grab Malaysia (ORG)

# Process a file
mnlp sentiment --input tweets.txt --output results.json
```

### Pipeline via CLI

```bash
# Chain multiple operations
echo "xpe la bro, best gila" | mnlp normalize | mnlp sentiment
# "takpe la bro, best gila" → positive (0.89)
```

### Batch Processing

```bash
# Process entire directory
mnlp batch sentiment ./data/tweets/ --output ./results/ --format json
```

---

## What's Next?

- Browse the [Module Overview](modules/index.md) to see all available capabilities
- Check the [API Reference](api-reference.md) for detailed function signatures
- Run [Benchmarks](benchmarks.md) to see performance on your hardware

!!! tip "Pro Tip"
    Use the pipeline API to chain modules together for complex workflows:
    ```python
    from manglish_nlp import Pipeline

    pipe = Pipeline([
        'normalize',
        'tokenize',
        'sentiment'
    ])
    results = pipe("xpe la, mmg best gila tempat ni")
    ```

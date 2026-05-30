# Getting Started

Get up and running with malaysian-manglish-nlp in under 5 minutes.

---

## Installation

=== "pip (recommended)"

    ```bash
    pip install malaysian-manglish-nlp
    ```

    Core install has **zero external dependencies**. Lightweight, fast, works everywhere.

=== "With extras"

    ```bash
    # ML model backend (transformers)
    pip install malaysian-manglish-nlp[ml]

    # spaCy integration
    pip install malaysian-manglish-nlp[spacy]

    # FastAPI REST server
    pip install malaysian-manglish-nlp[api]

    # Everything
    pip install malaysian-manglish-nlp[all]
    ```

=== "From source"

    ```bash
    git clone https://github.com/ZafranYusof/malaysia-manglish-nlp.git
    cd malaysian-manglish-nlp
    pip install -e .
    ```

    Useful for development or if you want to modify modules directly.

=== "Docker"

    ```bash
    docker pull zafranyusof/malaysian-manglish-nlp:latest
    docker run -it malaysian-manglish-nlp mnlp --help
    ```

    Pre-built image with all extras included.

!!! note "Python Version"
    malaysian-manglish-nlp requires **Python 3.9+**. Verify with:
    ```bash
    python --version
    ```

---

## Verify Installation

```bash
mnlp --version
# malaysian-manglish-nlp 1.0.0

mnlp doctor
# ✓ Python 3.11.4
# ✓ Core modules loaded (51/51)
# ✓ Optional: ml not installed (install with [ml])
# ✓ Optional: spacy not installed (install with [spacy])
```

---

## First Example Walkthrough

Let's analyse a real Manglish sentence step by step.

```python
import malaysian_manglish_nlp as mnlp

text = "Sedap gila nasi lemak kat kedai tu! Confirm repeat lagi."
```

### Step 1: Sentiment Analysis

```python
result = mnlp.sentiment(text)
print(result)
# {'label': 'positive', 'score': 0.96, 'aspects': ['nasi lemak']}
```

The model correctly identifies this as positive  -  "sedap gila" (insanely delicious) and "confirm repeat" are strong positive signals in Manglish.

### Step 2: Normalise Spelling

```python
text2 = "xpe la bro, aku nk g mkn jap lg"
normalized = mnlp.normalize(text2)
print(normalized)
# "takpe la bro, aku nak pergi makan jap lagi"
```

Handles common Manglish abbreviations: `xpe` → `takpe`, `nk` → `nak`, `g` → `pergi`, `mkn` → `makan`.

### Step 3: Detect Language Mix

```python
lang = mnlp.language("Eh jom la we go makan, I lapar gila already")
print(lang)
# {'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

Code-switching detection shows the BM/English blend ratio.

### Step 4: Named Entity Recognition

```python
entities = mnlp.ner("Siti beli iPhone kat Low Yat Plaza semalam")
print(entities)
# [('Siti', 'PERSON'), ('iPhone', 'PRODUCT'), ('Low Yat Plaza', 'LOCATION')]
```

NER works on Malaysian entities  -  local places, brands, and names.

### Step 5: Chain with Pipeline

```python
from malaysian_manglish_nlp import Pipeline

pipe = Pipeline([
    'normalize',    # Clean up informal spelling
    'tokenize',     # Split into tokens
    'sentiment'     # Analyse sentiment
])

results = pipe("xpe la, mmg best gila tempat ni")
print(results)
# {
#   'normalized': 'takpe la, memang best gila tempat ni',
#   'tokens': ['takpe', 'la', ',', 'memang', 'best', 'gila', 'tempat', 'ni'],
#   'sentiment': {'label': 'positive', 'score': 0.89}
# }
```

Pipelines pass output from one module to the next automatically.

---

## CLI Usage Guide

The `mnlp` CLI lets you process text without writing Python.

### Basic Commands

```bash
# Sentiment analysis
$ mnlp sentiment "Best gila movie tu!"
positive (0.92)

# Normalise text
$ mnlp normalize "aku xfhm ape ko ckp"
"aku tak faham apa kau cakap"

# Named Entity Recognition
$ mnlp ner "Ali kerja kat Grab Malaysia"
Ali (PERSON), Grab Malaysia (ORG)

# Language detection
$ mnlp language "Let's go makan at the mamak"
primary: manglish | ms: 0.40 | en: 0.60

# Tokenize
$ mnlp tokenize "Aku pergi kedai beli roti canai"
Aku | pergi | kedai | beli | roti | canai
```

### All Subcommands

| Command | Description |
|---|---|
| `mnlp sentiment` | Sentiment analysis (positive/negative/neutral) |
| `mnlp normalize` | Normalise informal Manglish spelling |
| `mnlp ner` | Named Entity Recognition |
| `mnlp language` | Language and code-switching detection |
| `mnlp tokenize` | Tokenisation |
| `mnlp summarize` | Text summarisation |
| `mnlp translate` | Manglish ↔ standard Malay/English |
| `mnlp stem` | Stemming and lemmatisation |
| `mnlp topics` | Topic extraction |
| `mnlp keywords` | Keyword extraction |
| `mnlp --help` | Full list with descriptions |

### File & Batch Processing

```bash
# Process a file line-by-line
$ mnlp sentiment --input tweets.txt --output results.json

# Batch process an entire directory
$ mnlp batch sentiment ./data/tweets/ --output ./results/ --format json

# Specify output columns
$ mnlp sentiment --input reviews.csv --output scored.csv --column text --append
```

### Piping

Chain commands via stdin/stdout:

```bash
# Normalise then analyse
$ echo "xpe la bro, best gila" | mnlp normalize | mnlp sentiment
"takpe la bro, best gila" → positive (0.89)

# Process file through pipeline
$ cat data/raw.txt | mnlp normalize | mnlp ner --json
```

---

## Common Recipes

### Recipe: Social Media Sentiment Dashboard

```python
import malaysian_manglish_nlp as mnlp

tweets = [
    "Weh sedap gila burger ni, confirm datang lagi!",
    "Aduh mahalnya, baik aku masak sendiri",
    "Ok la, not bad for the price",
]

for tweet in tweets:
    result = mnlp.sentiment(tweet)
    print(f"{result['label']:>10} ({result['score']:.2f})  -  {tweet}")
#   positive (0.95)  -  Weh sedap gila burger ni, confirm datang lagi!
#   negative (0.78)  -  Aduh mahalnya, baik aku masak sendiri
#    neutral (0.61)  -  Ok la, not bad for the price
```

### Recipe: Normalise + NER Pipeline

```python
from malaysian_manglish_nlp import Pipeline

pipe = Pipeline(['normalize', 'ner'])
result = pipe("ali keje kat mcd ss15 subang")
print(result)
# {
#   'normalized': 'ali kerja kat mcd ss15 subang',
#   'entities': [('ali', 'PERSON'), ('mcd', 'ORG'), ('ss15', 'LOCATION'), ('subang', 'LOCATION')]
# }
```

### Recipe: Batch Process with Progress

```python
import malaysian_manglish_nlp as mnlp
from pathlib import Path

files = Path("data/tweets").glob("*.txt")
results = []

for f in files:
    text = f.read_text(encoding="utf-8").strip()
    score = mnlp.sentiment(text)
    results.append({"file": f.name, "text": text, **score})

# Save results
import json
Path("results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
```

### Recipe: REST API Server

```bash
# Start the API server (requires [api] extra)
$ mnlp serve --port 8000

# Query from another terminal
$ curl http://localhost:8000/sentiment -d '{"text": "best gila la"}'
{"label": "positive", "score": 0.93}
```

---

## Configuration

malaysian-manglish-nlp works out of the box with sensible defaults. For fine-tuning:

```python
import malaysian_manglish_nlp as mnlp

# Set default model variant
mnlp.configure(backend="fast")      # fastest, lower accuracy
mnlp.configure(backend="balanced")  # default
mnlp.configure(backend="accurate")  # highest accuracy, slower

# Enable/disable modules
mnlp.configure(cache=True)          # cache repeated queries
mnlp.configure(batch_size=256)      # batch processing size
```

Or via environment variables:

```bash
export MANGLISH_NLP_BACKEND=balanced
export MANGLISH_NLP_CACHE=true
export MANGLISH_NLP_BATCH_SIZE=256
```

---

## Next Steps

<div class="grid cards" markdown>

-   :material-text-box-multiple:{ .lg .middle } __Browse Modules__

    ---

    Explore all 51 modules grouped by category  -  text processing, analysis, extraction, generation, and more.

    [:octicons-arrow-right-24: Module Overview](modules/index.md)

-   :material-book-open-variant:{ .lg .middle } __API Reference__

    ---

    Full function signatures, parameters, return types, and examples for every public function.

    [:octicons-arrow-right-24: API Reference](api-reference.md)

-   :material-chart-bar:{ .lg .middle } __Benchmarks__

    ---

    Performance numbers on standard hardware. See how malaysian-manglish-nlp compares to alternatives.

    [:octicons-arrow-right-24: Benchmarks](benchmarks.md)

-   :material-hand-heart:{ .lg .middle } __Contribute__

    ---

    Found a bug? Want a new module? Learn how to contribute to malaysian-manglish-nlp.

    [:octicons-arrow-right-24: Contributing Guide](contributing.md)

</div>

!!! tip "Need help?"
    Open an issue on [GitHub](https://github.com/ZafranYusof/malaysia-manglish-nlp/issues) or start a discussion.

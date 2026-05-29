# Integrations

Connect manglish-nlp with external frameworks and deployment targets.

---

## spacy_integration

Use manglish-nlp components as a spaCy pipeline.

```python
import spacy
import manglish_nlp as mnlp

# Load spaCy with manglish-nlp components
nlp = mnlp.spacy_integration.load()

# Or add to existing pipeline
nlp = spacy.blank("ms")
mnlp.spacy_integration.add_pipe(nlp, ["tokenizer", "ner", "sentiment"])
```

### Usage

```python
doc = nlp("Ahmad beli nasi lemak kat Pavilion KL semalam")

# Access tokens
for token in doc:
    print(token.text, token.pos_, token.dep_)

# Access entities
for ent in doc.ents:
    print(ent.text, ent.label_)
# Ahmad PERSON
# Pavilion KL LOCATION

# Access custom attributes
doc._.sentiment
# {'label': 'neutral', 'score': 0.65}
```

### Available Components

| Component | spaCy Pipe Name | Description |
|-----------|----------------|-------------|
| Tokenizer | `mnlp_tokenizer` | Malaysian-aware tokenization |
| NER | `mnlp_ner` | Named entity recognition |
| POS | `mnlp_pos` | Part-of-speech tagging |
| Sentiment | `mnlp_sentiment` | Sentiment as doc attribute |
| Language | `mnlp_language` | Language detection |

```python
# Custom pipeline
nlp = spacy.blank("ms")
nlp.add_pipe("mnlp_tokenizer")
nlp.add_pipe("mnlp_ner")
nlp.add_pipe("mnlp_sentiment")
```

!!! note "Installation"
    Requires the spaCy extra: `pip install manglish-nlp[spacy]`

---

## API (FastAPI)

Deploy manglish-nlp as a REST API with automatic documentation.

### Quick Start

```bash
# Start API server
mnlp serve --port 8000
```

```python
# Or programmatically
from manglish_nlp.api import create_app

app = create_app(modules=["sentiment", "ner", "normalize"])
# Run with: uvicorn app:app --port 8000
```

### Endpoints

Once running, the API exposes:

```
POST /sentiment      - Sentiment analysis
POST /ner            - Named entity recognition
POST /normalize      - Text normalization
POST /clean          - Text cleaning
POST /translate      - Translation
POST /embeddings     - Text embeddings
POST /pipeline       - Custom pipeline execution
GET  /health         - Health check
GET  /docs           - Swagger UI (auto-generated)
```

### Request/Response

```bash
# Sentiment
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Best gila makanan sini!"}'

# Response:
# {"label": "positive", "score": 0.94}
```

```bash
# Batch processing
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Best!", "Teruk la", "Ok je"]}'

# Response:
# [{"label": "positive", "score": 0.92},
#  {"label": "negative", "score": 0.87},
#  {"label": "neutral", "score": 0.78}]
```

### Configuration

```python
from manglish_nlp.api import create_app

app = create_app(
    modules=["sentiment", "ner", "normalize", "translate"],
    cors=True,                    # Enable CORS
    rate_limit="100/minute",      # Rate limiting
    auth="api-key",               # Authentication
    cache=True,                   # Response caching
    batch_max=100                 # Max batch size
)
```

!!! tip "Production Deployment"
    For production, use with gunicorn/uvicorn workers:
    ```bash
    uvicorn manglish_nlp.api:app --workers 4 --port 8000
    ```

---

## CLI

Command-line interface for quick text processing without writing Python.

### Commands

```bash
# Core commands
mnlp sentiment <text>
mnlp normalize <text>
mnlp ner <text>
mnlp clean <text>
mnlp translate <text> --target en
mnlp tokenize <text>
mnlp pos <text>
mnlp keywords <text>
```

### File Processing

```bash
# Process file
mnlp sentiment --input data.txt --output results.json

# Process directory
mnlp batch sentiment ./input/ --output ./output/ --format jsonl

# Stream from stdin
cat tweets.txt | mnlp sentiment --format csv
```

### Pipeline via CLI

```bash
# Chain operations
mnlp pipe "clean | normalize | sentiment" --input data.txt

# Custom pipeline config
mnlp pipe --config pipeline.json --input data.txt
```

### Options

```bash
# Output formats
mnlp sentiment "text" --format json    # Default
mnlp sentiment "text" --format csv
mnlp sentiment "text" --format table

# Verbose mode
mnlp sentiment "text" -v

# Model selection
mnlp sentiment "text" --model fast
mnlp sentiment "text" --model accurate

# Help
mnlp --help
mnlp sentiment --help
```

### Configuration

```bash
# Set defaults
mnlp config set model accurate
mnlp config set output_format json
mnlp config set cache true

# View config
mnlp config show
```

---

## See Also

- [Getting Started](../getting-started.md) — basic CLI usage
- [Tools](tools.md) — pipeline and caching
- [Benchmarks](../benchmarks.md) — API throughput numbers

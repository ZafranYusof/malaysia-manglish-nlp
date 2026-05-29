# REST API Guide

**Serve all manglish-nlp modules over HTTP with FastAPI  -  batch processing, rate limiting, and Docker deployment.**

---

## Why REST API?

Microservice integration, frontend consumption, mobile app backends, and team-wide NLP access without Python dependencies on every machine. Deploy once, call from anywhere.

---

## Installation

```bash
pip install manglish-nlp[api]
```

This installs FastAPI and uvicorn.

---

## Start the server

### Option 1: Python command

```bash
python -m manglish_nlp.rest_api
```

### Option 2: Uvicorn directly

```bash
uvicorn manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000
```

### Option 3: With auto-reload (development)

```bash
uvicorn manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://localhost:8000`.

!!! tip "Interactive docs"
    - Swagger UI: `http://localhost:8000/docs`
    - ReDoc: `http://localhost:8000/redoc`

---

## Endpoints

### System

#### `GET /health`

Health check.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "modules_loaded": 51
}
```

#### `GET /modules`

List available NLP modules.

```bash
curl http://localhost:8000/modules
```

```json
[
  {"name": "sentiment", "description": "Sentiment analysis", "endpoint": "/sentiment"},
  {"name": "normalize", "description": "Expand Manglish shortforms", "endpoint": "/normalize"},
  ...
]
```

---

### NLP Endpoints

All NLP endpoints accept `POST` with JSON body:

```json
{
  "text": "your text here",
  "options": {}
}
```

#### `POST /sentiment`

Sentiment analysis.

```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Best gila makanan kat sini!"}'
```

```json
{
  "result": {"label": "positive", "score": 0.94},
  "processing_time_ms": 0.42
}
```

#### `POST /normalize`

Expand Manglish shortforms.

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"text": "nk tnya brp hrga"}'
```

```json
{
  "result": "nak tanya berapa harga",
  "processing_time_ms": 0.18
}
```

#### `POST /translate`

Translate text.

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya nak makan nasi lemak", "target": "en"}'
```

```json
{
  "result": "I want to eat nasi lemak",
  "processing_time_ms": 0.95
}
```

Target options: `en`, `bm`, `ms`, `formal`

#### `POST /ner`

Named Entity Recognition.

```bash
curl -X POST http://localhost:8000/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "Ahmad kerja kat Petronas KL"}'
```

```json
{
  "result": [["Ahmad", "PERSON"], ["Petronas", "ORG"], ["KL", "LOCATION"]],
  "processing_time_ms": 0.87
}
```

#### `POST /pos`

Part-of-Speech tagging.

```bash
curl -X POST http://localhost:8000/pos \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya suka makan nasi lemak"}'
```

```json
{
  "result": [["Saya", "PRON"], ["suka", "VERB"], ["makan", "VERB"], ...],
  "processing_time_ms": 0.65
}
```

#### `POST /summarize`

Text summarization.

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long article text here..."}'
```

#### `POST /emotion`

Emotion detection.

```bash
curl -X POST http://localhost:8000/emotion \
  -H "Content-Type: application/json" \
  -d '{"text": "Geram betul dengan service dia!"}'
```

```json
{
  "result": {"primary": "anger", "score": 0.88, "secondary": "disgust"},
  "processing_time_ms": 0.38
}
```

#### `POST /keywords`

Keyword extraction.

```bash
curl -X POST http://localhost:8000/keywords \
  -H "Content-Type: application/json" \
  -d '{"text": "Harga minyak sawit meningkat ke paras tertinggi"}'
```

#### `POST /language`

Language detection.

```bash
curl -X POST http://localhost:8000/language \
  -H "Content-Type: application/json" \
  -d '{"text": "Weh jom la makan, I lapar gila"}'
```

```json
{
  "result": {"primary": "manglish", "scores": {"ms": 0.45, "en": 0.55}},
  "processing_time_ms": 0.22
}
```

#### `POST /formalize`

Convert informal to formal BM.

```bash
curl -X POST http://localhost:8000/formalize \
  -H "Content-Type: application/json" \
  -d '{"text": "aku nk g mkn jap"}'
```

#### `POST /dialect`

Dialect detection.

```bash
curl -X POST http://localhost:8000/dialect \
  -H "Content-Type: application/json" \
  -d '{"text": "Ambo nok make nasi kerabu"}'
```

#### `POST /analyze`

Full analysis pipeline (all modules).

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Weh best gila kedai tu!"}'
```

Returns normalized text, sentiment, language, POS, entities, emotion, and keywords in one response.

---

### Batch processing

#### `POST /batch`

Process multiple texts with multiple modules at once.

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Best gila movie tu!",
      "Teruk la service dia",
      "Sedap nasi lemak Mak Cik"
    ],
    "modules": ["sentiment", "normalize", "ner"]
  }'
```

```json
{
  "results": [
    {
      "text": "Best gila movie tu!",
      "sentiment": {"label": "positive", "score": 0.93},
      "normalize": "Best gila movie tu!",
      "ner": []
    },
    ...
  ],
  "processing_time_ms": 2.34,
  "count": 3
}
```

Available batch modules: `sentiment`, `normalize`, `ner`, `pos`, `translate`, `emotion`, `keywords`, `language`, `formalize`, `summarize`

!!! tip "Batch limits"
    Maximum 50 texts per batch request. Text max length: 10,000 characters.

---

## Rate limiting

Built-in rate limiting: **100 requests per minute per IP**.

Exceeded requests return HTTP 429:

```json
{"error": "Rate limit exceeded. Try again later."}
```

### Custom rate limits

Modify in code or via environment variable:

```python
# In rest_api.py
rate_limiter = RateLimiter(max_requests=200, window_seconds=60)  # 200/min
```

---

## Docker deployment

### Using the included Dockerfile

```bash
# Build
docker build -t manglish-nlp-api .

# Run
docker run -p 8000:8000 manglish-nlp-api
```

### Using docker-compose

```bash
docker-compose up -d
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir manglish-nlp[api]

EXPOSE 8000
CMD ["uvicorn", "manglish_nlp.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  api:
    image: manglish-nlp-api
    ports:
      - "8000:8000"
    environment:
      - UVICORN_WORKERS=4
    restart: unless-stopped
```

---

## Python client usage

```python
import requests

API = "http://localhost:8000"

# Sentiment
resp = requests.post(f"{API}/sentiment", json={"text": "Best gila!"})
print(resp.json())
# {'result': {'label': 'positive', 'score': 0.94}, 'processing_time_ms': 0.42}

# Batch
resp = requests.post(f"{API}/batch", json={
    "texts": ["Best!", "Teruk la"],
    "modules": ["sentiment", "normalize"]
})
for r in resp.json()['results']:
    print(r)
```

---

## JavaScript/TypeScript client

```javascript
const API = 'http://localhost:8000';

// Sentiment
const resp = await fetch(`${API}/sentiment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: 'Best gila makanan!' })
});
const data = await resp.json();
console.log(data.result);
// { label: 'positive', score: 0.94 }

// Batch
const batch = await fetch(`${API}/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        texts: ['Best!', 'Teruk la'],
        modules: ['sentiment', 'normalize']
    })
});
```

---

## CLI usage

```bash
# Start server
$ uvicorn manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000

# Or use python
$ python -m manglish_nlp.rest_api

# Docker
$ docker run -p 8000:8000 zafranyusof/manglish-nlp:latest
```

---

## Performance

| Endpoint | Avg Latency | Throughput |
|----------|------------|------------|
| `/sentiment` | < 1ms | 15,000 req/sec |
| `/normalize` | < 0.5ms | 30,000 req/sec |
| `/translate` | < 2ms | 8,000 req/sec |
| `/ner` | < 1ms | 12,000 req/sec |
| `/analyze` | < 5ms | 4,000 req/sec |
| `/batch` (10 texts) | < 10ms | 1,500 req/sec |

!!! note "Production deployment"
    For production, use multiple uvicorn workers behind a reverse proxy (nginx/traefik):
    ```bash
    uvicorn manglish_nlp.rest_api:app --host 0.0.0.0 --port 8000 --workers 4
    ```

---

## See also

- [Pipeline](pipeline.md)  -  understand the /analyze endpoint internals
- [Sentiment Analysis](sentiment.md)  -  details on sentiment output
- [NER](ner.md)  -  entity types and extraction details
- [Translation](translation.md)  -  translation target options
- [API Reference](../api-reference.md)  -  full function signatures
- [Docker Hub](https://hub.docker.com/r/zafranyusof/manglish-nlp)  -  pre-built images

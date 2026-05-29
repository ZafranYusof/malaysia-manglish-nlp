# Integrations

**Connect malaysian-manglish-nlp to spaCy, FastAPI, CLI, and LangChain  -  deploy anywhere.**

---

## Overview

Integration modules let you use malaysian-manglish-nlp inside existing frameworks: as spaCy pipeline components, behind a REST API with auto-generated docs, from the command line, or as LangChain tools for AI agents.

```bash
pip install malaysian-manglish-nlp[all]  # install all integration extras
```

---

## Quick Start

=== "Python"
    ```python
    import malaysian_manglish_nlp as mnlp

    # spaCy pipeline
    nlp = mnlp.spacy_integration.load()
    doc = nlp("Ahmad beli nasi lemak kat Pavilion KL semalam")
    [(ent.text, ent.label_) for ent in doc.ents]
    # [('Ahmad', 'PERSON'), ('Pavilion KL', 'LOCATION')]

    # LangChain tool
    from malaysian_manglish_nlp.langchain_tool import SentimentTool
    tool = SentimentTool()
    tool.run("Best gila makanan sini!")
    # "positive (0.94)"
    ```

=== "CLI"
    ```bash
    # Direct module access
    mnlp sentiment "Best gila makanan sini!"
    # {"label": "positive", "score": 0.94}

    # Start REST API
    mnlp serve --port 8000
    ```

---

## Module Details

### `spacy_integration`

Use malaysian-manglish-nlp modules as native spaCy pipeline components. Access entities, POS tags, and sentiment through the standard spaCy `Doc` API.

**Requires:** `pip install malaysian-manglish-nlp[spacy]`

```python
import spacy
import malaysian_manglish_nlp as mnlp

# Load full pipeline
nlp = mnlp.spacy_integration.load()

doc = nlp("Ahmad beli nasi lemak kat Pavilion KL semalam")

# Tokens + POS
for token in doc:
    print(f"{token.text:12} {token.pos_:6} {token.dep_}")
# Ahmad        PROPN  nsubj
# beli         VERB   ROOT
# nasi         NOUN   obj
# lemak        ADJ    amod
# kat          ADP    case
# Pavilion     PROPN  compound
# KL           PROPN  obl
# semalam      NOUN   obl:tmod

# Entities
for ent in doc.ents:
    print(f"{ent.text:16} {ent.label_}")
# Ahmad            PERSON
# Pavilion KL      LOCATION

# Sentiment (custom attribute)
doc._.sentiment
# {'label': 'neutral', 'score': 0.65}
```

#### Available Components

| Component | Pipe Name | Description |
|-----------|-----------|-------------|
| Tokenizer | `mnlp_tokenizer` | Malaysian-aware tokenisation |
| NER | `mnlp_ner` | Named entity recognition |
| POS | `mnlp_pos` | Part-of-speech tagging |
| Sentiment | `mnlp_sentiment` | Sentiment as `doc._.sentiment` |
| Language | `mnlp_language` | Language detection as `doc._.language` |

#### Custom Pipeline Assembly

```python
nlp = spacy.blank("ms")
nlp.add_pipe("mnlp_tokenizer")
nlp.add_pipe("mnlp_ner")
nlp.add_pipe("mnlp_sentiment")

doc = nlp("Weh best gila movie tu")
doc._.sentiment
# {'label': 'positive', 'score': 0.93}
```

!!! tip "Mixing Components"
    Combine malaysian-manglish-nlp components with standard spaCy components. For example, use `mnlp_tokenizer` + `mnlp_ner` with spaCy's built-in `sentencizer` for sentence boundary detection.

---

### REST API

Deploy malaysian-manglish-nlp as a FastAPI server with automatic Swagger documentation, CORS, rate limiting, and batch support.

#### Starting the Server

=== "CLI"
    ```bash
    # Start with defaults (port 8000, all modules)
    mnlp serve

    # Specific modules + port
    mnlp serve --modules sentiment,ner,normalize --port 9000

    # Production with workers
    uvicorn malaysian_manglish_nlp.api:app --workers 4 --port 8000
    ```

=== "Python"
    ```python
    from malaysian_manglish_nlp.api import create_app

    app = create_app(
        modules=["sentiment", "ner", "normalize", "translate"],
        cors=True,
        rate_limit="100/minute",
        auth="api-key",
        cache=True,
        batch_max=100
    )
    # Run: uvicorn app:app --port 8000
    ```

#### Endpoint Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/sentiment` | Sentiment analysis |
| `POST` | `/ner` | Named entity recognition |
| `POST` | `/normalize` | Text normalisation |
| `POST` | `/clean` | Text cleaning |
| `POST` | `/translate` | Translation |
| `POST` | `/embeddings` | Sentence embeddings |
| `POST` | `/pipeline` | Custom pipeline execution |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI (auto-generated) |

#### Request Examples

=== "Single Text"
    ```bash
    curl -X POST http://localhost:8000/sentiment \
      -H "Content-Type: application/json" \
      -d '{"text": "Best gila makanan sini!"}'

    # Response:
    # {"label": "positive", "score": 0.94}
    ```

=== "Batch"
    ```bash
    curl -X POST http://localhost:8000/sentiment \
      -H "Content-Type: application/json" \
      -d '{"texts": ["Best!", "Teruk la", "Ok je"]}'

    # Response:
    # [{"label": "positive", "score": 0.92},
    #  {"label": "negative", "score": 0.87},
    #  {"label": "neutral", "score": 0.78}]
    ```

=== "NER"
    ```bash
    curl -X POST http://localhost:8000/ner \
      -H "Content-Type: application/json" \
      -d '{"text": "Siti beli iPhone kat Low Yat Plaza"}'

    # Response:
    # [{"text": "Siti", "label": "PERSON"},
    #  {"text": "iPhone", "label": "PRODUCT"},
    #  {"text": "Low Yat Plaza", "label": "LOCATION"}]
    ```

#### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modules` | `list[str]` | all | Modules to expose |
| `cors` | `bool` | `True` | Enable CORS headers |
| `rate_limit` | `str` | `None` | Rate limit (e.g. `"100/minute"`) |
| `auth` | `str` | `None` | Auth mode: `"api-key"`, `"bearer"` |
| `cache` | `bool` | `False` | Enable response caching |
| `batch_max` | `int` | `50` | Maximum batch size |

!!! tip "Production Deployment"
    Use gunicorn with uvicorn workers for production:
    ```bash
    gunicorn malaysian_manglish_nlp.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
    ```

---

### CLI

Command-line interface for every module. Process files, pipe from stdin, chain operations, and manage configuration  -  all without writing Python.

#### Core Commands

```bash
mnlp sentiment "Best gila makanan sini!"
# {"label": "positive", "score": 0.94}

mnlp ner "Siti beli iPhone kat Low Yat Plaza"
# [{"text": "Siti", "label": "PERSON"}, ...]

mnlp normalize "xpe la bro, aku nk g mkn"
# "takpe la bro, aku nak pergi makan"

mnlp clean "Weh @ahmad check ni https://t.co/abc 🔥🔥🔥"
# "Weh check ni"

mnlp translate "Aku nak pergi makan" --target en
# "I want to go eat"

mnlp tokenize "Tak boleh la macam tu"
# ["Tak", "boleh", "la", "macam", "tu"]

mnlp pos "Aku nak pergi makan kat kedai tu"
# [["Aku", "PRON"], ["nak", "AUX"], ...]

mnlp keywords "Kerajaan umum pakej rangsangan RM50 bilion"
# ["pakej rangsangan", "RM50 bilion", "kerajaan"]
```

#### File Processing

```bash
# Process file → JSON output
mnlp sentiment --input data.txt --output results.json

# Process directory
mnlp batch sentiment ./input/ --output ./output/ --format jsonl

# Stream from stdin
cat tweets.txt | mnlp sentiment --format csv
```

#### Pipeline via CLI

```bash
# Chain operations
mnlp pipe "clean | normalize | sentiment" --input data.txt

# From config file
mnlp pipe --config pipeline.json --input data.txt
```

#### Output Formats

```bash
mnlp sentiment "text" --format json    # default
mnlp sentiment "text" --format csv
mnlp sentiment "text" --format table
```

#### Configuration

```bash
mnlp config set model accurate
mnlp config set output_format json
mnlp config set cache true
mnlp config show
```

!!! tip "Verbose Mode"
    Add `-v` to any command for debug output including timing and model info:
    ```bash
    mnlp sentiment "text" -v
    # [model: default | time: 2.3ms]
    # {"label": "positive", "score": 0.94}
    ```

---

### `langchain`

Use malaysian-manglish-nlp modules as LangChain tools for AI agent workflows. Each module wraps as a standard LangChain `BaseTool` with description and schema.

**Requires:** `pip install malaysian-manglish-nlp[langchain]`

```python
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from malaysian_manglish_nlp.langchain_tool import (
    SentimentTool,
    NerTool,
    TranslateTool,
    NormalizeTool
)

llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [
    SentimentTool(),
    NerTool(),
    TranslateTool(),
    NormalizeTool()
]

agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent can now use malaysian-manglish-nlp tools
agent.run("Analyse the sentiment of 'Best gila makanan sini!' and translate it to English")
# > Using tool: sentiment_analysis
# > Input: "Best gila makanan sini!"
# > Result: positive (0.94)
# > Using tool: translate
# > Input: "Best gila makanan sini!" → en
# > Result: "The food here is incredibly good!"
# >
# > The text has a positive sentiment (94% confidence) and translates to
# > "The food here is incredibly good!" in English.
```

#### Available Tools

| Tool Class | Module | Description |
|------------|--------|-------------|
| `SentimentTool` | `sentiment` | Analyse text sentiment |
| `NerTool` | `ner` | Extract named entities |
| `TranslateTool` | `translate` | Translate between languages |
| `NormalizeTool` | `normalize` | Normalise informal text |
| `EmotionTool` | `emotion` | Detect emotions |
| `KeywordsTool` | `keywords` | Extract keywords |
| `SummarizeTool` | `summarize` | Summarise text |
| `QATool` | `qa` | Answer questions from context |

#### Custom Tool Configuration

```python
from malaysian_manglish_nlp.langchain_tool import SentimentTool

# With custom settings
tool = SentimentTool(
    name="malaysian_sentiment",
    description="Analyse sentiment of Malaysian/Manglish text",
    detailed=True  # return all class scores
)
```

!!! example "Building a Moderation Agent"
    ```python
    from malaysian_manglish_nlp.langchain_tool import SentimentTool, HateSpeechTool, ProfanityTool

    moderation_tools = [SentimentTool(), HateSpeechTool(), ProfanityTool()]
    # Agent can triage content using all three signals
    ```

!!! tip "Tool Descriptions"
    Each tool includes a detailed description that LangChain agents use for routing. The descriptions specify that input should be Malaysian or Manglish text, helping the agent choose the right tool for multilingual inputs.

---

## See Also

- [Tools](tools.md)  -  pipeline, caching, and profiling for production deployments
- [REST API + Cache](tools.md#cache)  -  cache API responses for repeated queries
- [Benchmarks](../benchmarks.md)  -  API throughput and latency numbers

---
language:
  - ms
  - en
license: mit
task_categories:
  - text-classification
  - token-classification
tags:
  - manglish
  - malaysian
  - code-switching
  - sentiment-analysis
  - nlp
size_categories:
  - 1K<n<10K
---

# Manglish NLP Dataset

## Dataset Description

A labeled dataset of 1,139 Manglish (Malaysian English) examples for multi-task NLP. Manglish is the colloquial code-switched variety of English spoken in Malaysia, mixing Malay, English, Chinese dialects, and Tamil in everyday conversation.

This dataset captures the linguistic diversity of Malaysian social media, messaging apps, and informal writing — including slang, abbreviations, particles (lah, la, wei, eh), and code-switching patterns.

## Supported Tasks

| Task | Labels | Description |
|------|--------|-------------|
| Sentiment Analysis | positive, negative, neutral | Overall sentiment of the text |
| Emotion Detection | happy, sad, angry, surprised, fearful, disgusted, neutral | Fine-grained emotion |
| Intent Classification | question, statement, request, complaint, greeting, farewell | Communicative intent |
| Topic Classification | food, technology, politics, sports, entertainment, daily_life, education, work | Subject matter |
| Dialect Detection | northern, southern, east_coast, sabah_sarawak, kl_urban | Regional dialect markers |
| Language Detection | malay, english, manglish, mixed | Primary language of the text |

## Languages

- **Malay (ms)** — Standard and colloquial Bahasa Melayu
- **English (en)** — Malaysian English and standard English
- **Manglish** — Code-switched Malay-English with local particles and slang

## Dataset Structure

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The input text |
| `sentiment` | string | Sentiment label |
| `emotion` | string | Emotion label |
| `intent` | string | Intent label |
| `topic` | string | Topic label |
| `dialect` | string | Dialect region |
| `language` | string | Detected language |
| `normalized` | string | Normalized/standard form of the text |
| `shortforms` | list | Identified shortforms and their expansions |

### Label Distributions

- **Sentiment:** ~40% positive, ~30% neutral, ~30% negative
- **Language:** ~50% manglish, ~25% malay, ~20% english, ~5% mixed
- **Intent:** ~35% statement, ~25% question, ~15% complaint, ~10% request, ~10% greeting, ~5% farewell

## Data Collection

This dataset was synthetically generated to cover a wide range of Manglish patterns found in Malaysian social media (Twitter/X, Reddit r/malaysia, WhatsApp groups, Facebook comments). Examples were crafted to represent:

- Common shortforms (tak → x, tidak → tk, macam → mcm)
- Particles and fillers (lah, la, wei, eh, kan, kot)
- Code-switching patterns (intra-sentential and inter-sentential)
- Regional dialect markers
- Internet slang and abbreviations

All examples were manually reviewed for label accuracy.

## Usage

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("ZafranYusof/manglish-nlp-dataset")

# Access splits
train = dataset["train"]

# Example
print(train[0])
# {'text': 'Weh best gila makanan kat sini la', 'sentiment': 'positive', ...}

# Filter by task
positive_examples = train.filter(lambda x: x["sentiment"] == "positive")
```

### Loading from local JSONL

```python
import json

with open("manglish_labeled.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
```

## Citation

```bibtex
@dataset{yusof2025manglish,
  title={Manglish NLP Dataset: Multi-task Labeled Data for Malaysian Code-Switched Text},
  author={Yusof, Zafran},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/ZafranYusof/manglish-nlp-dataset}
}
```

## License

MIT License

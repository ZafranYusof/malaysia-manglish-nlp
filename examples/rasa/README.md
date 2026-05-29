# Rasa + malaysian-manglish-nlp Examples

Build a Rasa chatbot that understands Malaysian Manglish with custom NLU
components and actions powered by malaysian-manglish-nlp.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Project Structure

```
examples/rasa/
├── config.yml           # Rasa pipeline config with custom components
├── domain.yml           # Intents, entities, slots, responses
├── nlu.yml              # NLU training data (Manglish examples)
├── nlu_component.py     # Custom NLU components (Featurizer + NER)
├── actions/
│   └── actions.py       # Custom actions (sentiment, translate, NER)
├── requirements.txt
└── README.md
```

## Setup

### 1. Train the model

```bash
cd examples/rasa
rasa train
```

### 2. Run the actions server

In one terminal:
```bash
rasa run actions
```

### 3. Run the bot

In another terminal:
```bash
rasa shell
```

### Or use Docker Compose (from project root)

```bash
cd examples/
docker-compose up
```

## Custom NLU Components

### ManglishNLPFeaturizer

Preprocesses text with malaysian-manglish-nlp before DIET classification:
- **Normalisation**: Expands shortforms (nk→nak, x→tak, brp→berapa)
- **Language detection**: Tags text as BM/EN/Manglish
- **Sentiment**: Adds sentiment score as feature
- **Intent features**: is_question, is_complaint, is_request

### ManglishNERExtractor

Extracts entities using malaysian-manglish-nlp NER:
- 9 entity types: PERSON, LOCATION, ORGANIZATION, MONEY, DATE, TIME, PHONE, EMAIL, URL
- Handles Malaysian names, RM currency, local date/time formats
- Configurable confidence threshold and entity type filtering

## Custom Actions

| Action | Description |
|--------|-------------|
| `action_analyse_sentiment` | Analyse user message sentiment, set slot |
| `action_translate` | Translate between Manglish/English/formal BM |
| `action_extract_entities` | Extract and display named entities |
| `action_makan_recommendation` | Sentiment-aware food recommendations |
| `action_sentiment_aware_response` | Adapt response tone to user mood |

## Intents

Malaysian-specific intents included:

- `ask_makan` — Food recommendations ("kat mana best makan area SS15")
- `ask_belanja` — Budget/spending queries ("berapa bajet dinner")
- `ask_cuaca` — Weather queries ("hujan tak hari ni")
- `ask_harga` — Price queries ("berapa harga petrol sekarang")
- `complaint` — Complaints with empathetic responses

## Example Conversations

```
You: wei aku lapar gila, ada port makan best tak area Bangsar?
Bot: Kalau area Bangsar, aku suggest cuba makanan. Port best biasanya
     dekat main road atau area pasar malam. Try tanya member local,
     diorang tau port rahsia.

You: sentiment apa bagi "gila kejam parking RM10 sejam"
Bot: Text tu nampak negative (confidence: 87%). Emotion: anger.
     Ada apa-apa yang tak ok ke?

You: translate "aku nak pergi pasar malam" to english
Bot: BM/Manglish → English:
     Original: aku nak pergi pasar malam
     Translated: I want to go to the night market
     Formal BM: Saya hendak pergi ke pasar malam
```

## Customisation

### Add more training data

Edit `nlu.yml` to add more Manglish examples for each intent.
More examples = better accuracy.

### Tune NLU components

In `config.yml`, adjust component parameters:
```yaml
- name: "nlu_component.ManglishNERExtractor"
  min_confidence: 0.5      # Only keep high-confidence entities
  entity_types:             # Only extract specific types
    - PERSON
    - LOCATION
    - MONEY
```

### Add new actions

1. Add action name to `domain.yml` under `actions:`
2. Implement in `actions/actions.py`
3. Add rules/stories in `rules.yml` or `stories.yml`

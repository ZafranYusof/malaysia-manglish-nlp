# Manglish NLP Datasets

Multi-label NLP datasets for Malaysian social media text (Manglish, Malay, English, code-switched).

## Dataset Files

| File | Size | Description |
|------|------|-------------|
| `manglish_labeled.jsonl` | 561 examples | Original hand-labeled v1 dataset |
| `manglish_labeled_v2.jsonl` | 578 examples | Extended v2 with more dialects/topics |
| `manglish_labeled_v3.jsonl` | ~8000+ examples | Synthetic augmented dataset |
| `manglish_auto_labeled.jsonl` | varies | High-confidence auto-labeled scraped data |
| `manglish_review_queue.jsonl` | varies | Low-confidence items needing human review |
| `manglish_full.jsonl` | varies | Merged + deduped + train/test split |
| `manglish_full_train.jsonl` | ~80% of full | Training split |
| `manglish_full_test.jsonl` | ~20% of full | Test split |
| `manglish_merged.jsonl` | ~1139 examples | v1 + v2 merged (original merge script output) |

### Raw Scraped Data

| File | Source | Description |
|------|--------|-------------|
| `raw_scraped/reddit_raw.jsonl` | Reddit | r/malaysia, r/malaysian, etc. |
| `raw_scraped/twitter_raw.jsonl` | Twitter/X | Malaysian hashtags |
| `raw_scraped/lowyat_raw.jsonl` | Lowyat | Malaysian tech forum |
| `raw_scraped/combined_raw.jsonl` | All | Combined + deduped raw data |

## Format

JSONL (one JSON object per line). Each entry:

```json
{
  "text": "Weh nasi lemak kat mamak ni memang gila sedap la bro",
  "sentiment": "positive|negative|neutral",
  "language": "manglish|malay|english|mixed",
  "emotion": "happy|sad|angry|fear|surprise|disgust|love|neutral",
  "intent": "question|statement|request|complaint|greeting|opinion",
  "dialect": "standard|kelantan|terengganu|n9|kedah|sarawak",
  "topic": "food|politics|sports|tech|education|entertainment|religion|daily_life",
  "source_type": "twitter|reddit|whatsapp|news_comment|augmented|scraped",
  "is_code_switch": true|false
}
```

### Augmented entries also include:

```json
{
  "augmentation_methods": ["synonym", "shortform", "emoji"],
  "source_original": "first 80 chars of original text",
  "auto_label_confidence": 0.92,
  "label_source": "auto_high_confidence|needs_review|human"
}
```

## Label Distributions (Original v1+v2: 1139 examples)

### Sentiment
| Label | Count | Percentage |
|-------|-------|------------|
| positive | ~650 | ~57% |
| negative | ~300 | ~26% |
| neutral | ~189 | ~17% |

### Dialect
| Label | Count |
|-------|-------|
| standard | ~380 |
| kelantan | ~156 |
| sarawak | ~156 |
| terengganu | ~148 |
| kedah | ~142 |
| n9 | ~140 |

### Topics
| Label | Count |
|-------|-------|
| daily_life | ~200 |
| food | ~180 |
| tech | ~130 |
| education | ~126 |
| sports | ~126 |
| religion | ~124 |
| entertainment | ~120 |
| politics | ~116 |

### Language
| Label | Count |
|-------|-------|
| manglish | ~730 |
| malay | ~250 |
| mixed | ~140 |
| english | ~2 |

### Emotion
| Label | Count |
|-------|-------|
| happy | ~630 |
| neutral | ~160 |
| angry | ~124 |
| sad | ~94 |
| love | ~46 |
| fear | ~40 |
| surprise | ~16 |
| disgust | ~10 |

### Intent
| Label | Count |
|-------|-------|
| opinion | ~520 |
| statement | ~340 |
| complaint | ~138 |
| question | ~66 |
| request | ~56 |
| greeting | ~6 |

## Augmentation Strategies

The `scripts/augment_data.py` script uses 10 strategies to generate ~8000+ synthetic examples:

| Strategy | Description | Example |
|----------|-------------|---------|
| **Synonym replacement** | Replace words with Manglish synonyms | `sedap` → `best`, `power`, `mantap` |
| **Shortform variation** | Toggle shortforms ↔ full words | `mcm` ↔ `macam`, `nk` ↔ `nak` |
| **Code-switching injection** | Insert English words into Malay text | adds `food`, `service`, `price` etc. |
| **Slang particle injection** | Add Malaysian particles | adds `lah`, `wei`, `kan`, `eh` |
| **Emoji variation** | Add/remove/swap sentiment-matched emojis | 😊🔥 for positive, 😤💀 for negative |
| **Negation flip** | Flip positive ↔ negative | `sedap` → `tak sedap` (label also flips) |
| **Aspect swap** | Change discussed aspect | food → service → price |
| **Length variation** | Shorten or lengthen text | remove fillers or add intensifiers |
| **Spelling variation** | Common Malaysian misspellings | vowel drops, char doubling, truncation |
| **Dialect variation** | Convert to regional dialects | standard → Kelantan/Kedah/N9/Sarawak/Terengganu |

Each original example generates 5-8 augmented variants. Variants use 1-3 chained strategies.

## Data Pipeline

```
1. Original data (v1 + v2, 1139 examples)
       │
       ▼
2. augment_data.py ──────► manglish_labeled_v3.jsonl (~8000+)
       │
       ▼
3. scrape_social_media.py ──► raw_scraped/*.jsonl
       │
       ▼
4. auto_label.py ──────────► manglish_auto_labeled.jsonl (high confidence)
       │                      manglish_review_queue.jsonl (needs review)
       ▼
5. merge_datasets.py ──────► manglish_full.jsonl (all sources merged)
                              manglish_full_train.jsonl (80%)
                              manglish_full_test.jsonl (20%)
```

## Usage

### Load dataset

```python
import json

data = []
with open('datasets/manglish_full.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Filter
positive = [d for d in data if d['sentiment'] == 'positive']
kelantan = [d for d in data if d['dialect'] == 'kelantan']
food = [d for d in data if d['topic'] == 'food']
```

### Train/test split

```python
train = []
test = []
with open('datasets/manglish_full_train.jsonl', 'r', encoding='utf-8') as f:
    train = [json.loads(l) for l in f]
with open('datasets/manglish_full_test.jsonl', 'r', encoding='utf-8') as f:
    test = [json.loads(l) for l in f]
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/augment_data.py` | Generate synthetic augmented data |
| `scripts/scrape_social_media.py` | Scrape Reddit/Twitter/Lowyat |
| `scripts/auto_label.py` | Auto-label scraped data |
| `scripts/merge_datasets.py` | Merge all datasets with dedup + split |

## Sources

- **v1/v2**: Hand-crafted realistic Malaysian social media text
- **Reddit**: r/malaysia, r/malaysian, r/MalaysianTweets, r/AskMalaysia
- **Twitter/X**: #Malaysia, #MalaysianFood, #MakanMalaysia, etc.
- **Lowyat**: Malaysian tech forum threads

## Content Features

- Realistic Malaysian social media text
- Mix of BM + English + code-switched content
- Common slang: la, wei, gila, confirm, lepak, mamak, syok, best
- Shortforms: nk, mcm, yg, sbb, tk, dh, je, kot
- Regional dialects: Kelantan (ambo/mung), Sarawak (kamek/kitak), Kedah (hang/depa), N9 (den), Terengganu (ambo/die)
- Includes hashtag-style text, emoji references, and natural typos

## License

For research and educational use.

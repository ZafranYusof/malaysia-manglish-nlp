# Manglish Labeled Dataset

Multi-label NLP dataset for Malaysian social media text (Manglish, Malay, English, code-switched).

## Format

JSONL (one JSON object per line). Each entry:

```json
{
  "text": "...",
  "sentiment": "positive|negative|neutral",
  "language": "manglish|malay|english|mixed",
  "emotion": "happy|sad|angry|fear|surprise|disgust|love|neutral",
  "intent": "question|statement|request|complaint|greeting|opinion",
  "dialect": "standard|kelantan|terengganu|n9|kedah|sarawak",
  "topic": "food|politics|sports|tech|education|entertainment|religion|daily_life",
  "source_type": "twitter|reddit|whatsapp|news_comment",
  "is_code_switch": true|false
}
```

## Stats

| Label | Distribution |
|-------|-------------|
| **Total** | 561 examples |
| **Sentiments** | positive: 346, negative: 131, neutral: 84 |
| **Dialects** | standard: 190, kelantan: 78, sarawak: 78, terengganu: 74, kedah: 71, n9: 70 |
| **Topics** | daily_life: 101, food: 89, tech: 65, education: 63, sports: 63, religion: 62, entertainment: 60, politics: 58 |
| **Languages** | manglish: 364, malay: 127, mixed: 69, english: 1 |
| **Sources** | twitter: 421, whatsapp: 99, news_comment: 26, reddit: 15 |
| **Emotions** | happy: 316, angry: 62, neutral: 80, sad: 47, love: 23, fear: 20, surprise: 8, disgust: 5 |
| **Intents** | opinion: 259, statement: 169, complaint: 69, question: 33, request: 28, greeting: 3 |

## Content Features

- Realistic Malaysian social media text
- Mix of BM + English + code-switched content
- Common slang: la, wei, gila, confirm, lepak, mamak, syok, best
- Shortforms: nk, mcm, yg, sbb, tk, dh, je, kot
- Regional dialects with authentic vocabulary (e.g., ambo/mung for Kelantan, kamek/kitak for Sarawak, den for N9, hang/depa for Kedah)
- Includes hashtag-style text, emoji references, and natural typos

## Usage

```python
import json

data = []
with open('manglish_labeled.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# Filter by dialect
kelantan = [d for d in data if d['dialect'] == 'kelantan']

# Filter by sentiment
negative = [d for d in data if d['sentiment'] == 'negative']
```

## License

For research and educational use.

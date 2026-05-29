# Emotion Detection

**Detect specific emotions beyond positive/negative — 8 emotion categories with intensity scoring.**

---

## Why emotion detection?

Customer experience analysis, mental health monitoring, social media sentiment deep-dives, and chatbot emotional intelligence. Knowing something is "negative" isn't enough — is the customer *angry* or *sad*? Is the tweet *fearful* or *disgusted*?

manglish-nlp detects 8 emotion categories with intensity, handling Manglish and code-switched text.

---

## Load module

```python
import manglish_nlp as mnlp

result = mnlp.detect_emotion("Geram betul aku dengan dia, dah la lambat pastu buat hal")
print(result)
# {'primary': 'anger', 'score': 0.88, 'secondary': 'disgust'}
```

---

## Basic usage

### Single emotion detection

```python
mnlp.detect_emotion("Happy gila dapat result exam!")
# {'primary': 'joy', 'score': 0.91, 'secondary': 'surprise'}

mnlp.detect_emotion("Sedih betul, kucing aku hilang dah 3 hari")
# {'primary': 'sadness', 'score': 0.89, 'secondary': 'fear'}

mnlp.detect_emotion("Seram gila movie tu, tak boleh tidur malam")
# {'primary': 'fear', 'score': 0.85, 'secondary': 'surprise'}

mnlp.detect_emotion("Eww geli nya, makanan ada ulat")
# {'primary': 'disgust', 'score': 0.92, 'secondary': 'anger'}
```

### All 8 emotions

```python
# Joy
mnlp.detect_emotion("Alhamdulillah dapat kerja baru!")
# {'primary': 'joy', 'score': 0.93, 'secondary': 'anticipation'}

# Sadness
mnlp.detect_emotion("Rindu sangat kat arwah mak")
# {'primary': 'sadness', 'score': 0.91, 'secondary': 'trust'}

# Anger
mnlp.detect_emotion("Bodoh punya driver, potong queue!")
# {'primary': 'anger', 'score': 0.90, 'secondary': 'disgust'}

# Fear
mnlp.detect_emotion("Takut nak balik sorang malam ni")
# {'primary': 'fear', 'score': 0.87, 'secondary': 'sadness'}

# Surprise
mnlp.detect_emotion("Eh tak sangka pulak dia datang!")
# {'primary': 'surprise', 'score': 0.88, 'secondary': 'joy'}

# Disgust
mnlp.detect_emotion("Meluat betul tengok perangai dia")
# {'primary': 'disgust', 'score': 0.89, 'secondary': 'anger'}

# Trust
mnlp.detect_emotion("Aku percaya kau boleh buat, you got this!")
# {'primary': 'trust', 'score': 0.84, 'secondary': 'anticipation'}

# Anticipation
mnlp.detect_emotion("Tak sabar nak tunggu Hari Raya ni!")
# {'primary': 'anticipation', 'score': 0.90, 'secondary': 'joy'}
```

---

## Batch processing

Process multiple texts efficiently:

```python
texts = [
    "Best gila dapat hadiah birthday!",
    "Kecewa betul dengan service dia",
    "Scared gila tengok accident depan mata",
    "Tak sabar nak pergi holiday!",
]

results = mnlp.detect_emotions_batch(texts)
for text, emotion in zip(texts, results):
    print(f"{text[:35]:35s} → {emotion['primary']} ({emotion['score']:.2f})")

# Best gila dapat hadiah birthday!    → joy (0.91)
# Kecewa betul dengan service dia     → sadness (0.82)
# Scared gila tengok accident...      → fear (0.86)
# Tak sabar nak pergi holiday!        → anticipation (0.88)
```

---

## Emotion summary

Get an overview of emotions across a collection of texts:

```python
reviews = [
    "Sedap gila! Confirm repeat!",
    "Service slow nak mampus",
    "Ok je, nothing special",
    "Best! Will come again for sure",
    "Mahal sangat, rasa biasa je",
]

summary = mnlp.emotion_summary(reviews)
print(summary)
# {'dominant': 'joy',
#  'distribution': {'joy': 2, 'anger': 1, 'neutral': 1, 'disgust': 1},
#  'average_intensity': 0.78}
```

---

## Social media text examples

```python
# Twitter/X rage
mnlp.detect_emotion("Geram gila dengan J&T, parcel hilang lagi sekali!!!")
# {'primary': 'anger', 'score': 0.93, 'secondary': 'disgust'}

# WhatsApp sadness
mnlp.detect_emotion("Takpelah, aku biasa dah kena macam ni sorang2")
# {'primary': 'sadness', 'score': 0.84, 'secondary': 'trust'}

# Instagram excitement
mnlp.detect_emotion("OMG!!! Finally sampai juga parcel tu! Cantik gila 😍")
# {'primary': 'joy', 'score': 0.92, 'secondary': 'surprise'}

# Reddit frustration
mnlp.detect_emotion("LRT delay lagi, hari2 macam ni, what's the point of paying")
# {'primary': 'anger', 'score': 0.81, 'secondary': 'sadness'}
```

---

## CLI usage

Emotion detection is primarily a Python API. For CLI:

```bash
# Full analysis (includes emotion)
$ mnlp analyze "Geram betul dengan dia"

# With JSON output
$ mnlp analyze "Happy gila!" --json
```

---

## How it works

1. **Lexicon matching** — emotion word dictionaries for BM and EN (3,000+ terms)
2. **Intensifier handling** — "gila", "sangat", "betul" boost scores
3. **Context analysis** — surrounding words modify emotion classification
4. **Secondary emotion** — mixed emotions detected (e.g., anger + disgust)
5. **Code-switching support** — BM and EN words processed together

---

## Performance

| Metric | Score |
|--------|-------|
| 8-class accuracy | 78.4% |
| Macro F1 | 74.2% |
| Joy F1 | 85.1% |
| Anger F1 | 82.3% |
| Sadness F1 | 76.8% |
| Throughput | 20,000 texts/sec |
| Latency (single) | < 0.5ms |

---

## See also

- [Sentiment Analysis](sentiment.md) — for positive/negative/neutral classification
- [Hate Speech Detection](hate-speech.md) — detect toxic content beyond emotion
- [Sarcasm Detection](../modules/analysis.md#sarcasm) — detect ironic emotional expression
- [Pipeline](pipeline.md) — chain emotion with other modules
- [API Reference](../api-reference.md#emotion) — full function signature

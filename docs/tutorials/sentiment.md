# Sentiment Analysis

**Detect positive, negative, and neutral sentiment in Malaysian text  -  including Manglish, code-switching, and sarcasm.**

---

## Why sentiment analysis?

Social media monitoring, brand reputation tracking, customer feedback analysis, and public opinion mining for Malaysian businesses and researchers. Standard NLP tools fail on Manglish because they don't understand slang like "gila best", "teruk gila", or sarcastic patterns like "Bagus la tu, tunggu 3 jam".

malaysian-manglish-nlp handles all of this natively  -  no preprocessing needed.

---

## Load module

```python
import malaysian_manglish_nlp as mnlp

# Basic sentiment
result = mnlp.sentiment("Weh best gila makanan kat sini!")
print(result)
# {'label': 'positive', 'score': 0.94}
```

That's it. One import, one function call.

---

## Basic usage

### Positive sentiment

```python
mnlp.sentiment("Sedap gila nasi lemak kat kedai tu!")
# {'label': 'positive', 'score': 0.96}

mnlp.sentiment("Best movie tu, memang worth it la")
# {'label': 'positive', 'score': 0.91}

mnlp.sentiment("Alhamdulillah finally dapat kerja baru")
# {'label': 'positive', 'score': 0.89}
```

### Negative sentiment

```python
mnlp.sentiment("Teruk la service dia, tunggu 1 jam")
# {'label': 'negative', 'score': 0.89}

mnlp.sentiment("Mahal gila, tak berbaloi langsung")
# {'label': 'negative', 'score': 0.93}

mnlp.sentiment("Kecewa betul, order wrong again")
# {'label': 'negative', 'score': 0.87}
```

### Neutral sentiment

```python
mnlp.sentiment("Aku nak pergi kedai jap")
# {'label': 'neutral', 'score': 0.72}

mnlp.sentiment("Meeting pukul 3 petang ni")
# {'label': 'neutral', 'score': 0.68}
```

---

## Detailed output

Get scores for all classes instead of just the top prediction:

```python
mnlp.sentiment("Best gila!", detailed=True)
# {'label': 'positive',
#  'scores': {'positive': 0.96, 'neutral': 0.03, 'negative': 0.01}}

mnlp.sentiment("Ok je, nothing special", detailed=True)
# {'label': 'neutral',
#  'scores': {'positive': 0.12, 'neutral': 0.76, 'negative': 0.12}}
```

!!! tip "When to use detailed mode"
    Use `detailed=True` when you need confidence calibration or want to apply custom thresholds. The raw scores help you decide how much to trust the prediction.

---

## Aspect-based sentiment

Extract sentiment for individual aspects within a sentence. Perfect for product reviews where different features have different sentiments.

```python
mnlp.sentiment("Makanan sedap tapi service slow", aspect=True)
# [{'aspect': 'makanan', 'label': 'positive', 'score': 0.92},
#  {'aspect': 'service', 'label': 'negative', 'score': 0.85}]

mnlp.sentiment("Phone cantik gila tapi battery lemah", aspect=True)
# [{'aspect': 'phone', 'label': 'positive', 'score': 0.90},
#  {'aspect': 'battery', 'label': 'negative', 'score': 0.82}]

mnlp.sentiment("Harga murah, rasa sedap, tapi parking susah", aspect=True)
# [{'aspect': 'harga', 'label': 'positive', 'score': 0.88},
#  {'aspect': 'rasa', 'label': 'positive', 'score': 0.91},
#  {'aspect': 'parking', 'label': 'negative', 'score': 0.79}]
```

!!! warning "Aspect extraction limits"
    Aspect-based mode works best with clear noun-adjective patterns. Very complex sentences or implicit aspects may not be captured.

---

## Sarcasm detection

malaysian-manglish-nlp detects sarcastic sentiment automatically  -  no separate flag needed. Sarcastic text with positive words but negative context gets correctly classified.

```python
# Sarcastic  -  "bagus" + negative context
mnlp.sentiment("Bagus la tu, tunggu 3 jam baru sampai")
# {'label': 'negative', 'score': 0.78}  # Detected as sarcasm

# Sarcastic  -  double praise pattern
mnlp.sentiment("Wah pandainya, exam fail pun boleh celebrate")
# {'label': 'negative', 'score': 0.74}  # Detected as sarcasm

# Genuine positive (for comparison)
mnlp.sentiment("Pandai betul dia solve masalah tu")
# {'label': 'positive', 'score': 0.88}  # Genuine
```

For standalone sarcasm detection:

```python
mnlp.detect_sarcasm("Bagus la tu, memang efficient gila, tunggu sampai esok")
# {'is_sarcastic': True, 'score': 0.82, 'cues': ['positive_opener', 'negative_context']}
```

---

## Batch processing

Pass a list for efficient batch inference:

```python
texts = [
    "Best gila movie tu!",
    "Teruk la service dia",
    "Ok je, nothing much",
    "Sedap nasi lemak Mak Cik",
    "Mahal sangat, tak worth it",
]

results = mnlp.sentiment(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]:30s} → {result['label']} ({result['score']:.2f})")

# Best gila movie tu!            → positive (0.93)
# Teruk la service dia           → negative (0.89)
# Ok je, nothing much            → neutral (0.65)
# Sedap nasi lemak Mak Cik       → positive (0.94)
# Mahal sangat, tak worth it     → negative (0.91)
```

!!! tip "Performance"
    Batch mode processes texts sequentially with zero overhead. For true parallel processing, use the [Pipeline](pipeline.md) module or the [REST API](rest-api.md).

---

## Mixed sentiment handling

Text with contrast markers ("tapi", "but", "cuma") gets analyzed for mixed sentiment:

```python
mnlp.sentiment("Makanan best tapi harga mahal", detailed=True)
# {'label': 'mixed',
#  'scores': {'positive': 0.45, 'neutral': 0.10, 'negative': 0.45},
#  'contrast': True}
```

---

## Working with raw social media text

malaysian-manglish-nlp handles noisy text directly. No need to clean first:

```python
# With hashtags
mnlp.sentiment("Best gila!! #nasilemak #sedap #klfood")
# {'label': 'positive', 'score': 0.93}

# With emojis
mnlp.sentiment("Sedap sangat 😍😍😍 confirm repeat")
# {'label': 'positive', 'score': 0.95}

# With elongated words
mnlp.sentiment("Besttttt gilaaaaa!!!")
# {'label': 'positive', 'score': 0.91}

# With mentions
mnlp.sentiment("@restaurant Best food but slow service la")
# {'label': 'mixed', ...}
```

!!! tip "Preprocessing option"
    For maximum accuracy on very noisy text, normalize first:
    ```python
    clean = mnlp.normalize("xpe la best gila mkn dia")
    result = mnlp.sentiment(clean)
    ```

---

## CLI usage

```bash
# Basic sentiment
$ mnlp sentiment "Best gila movie tu!"
positive (0.92)

# With JSON output
$ mnlp sentiment "Teruk la service" --json
{"label": "negative", "score": 0.89}

# Pipe from stdin
$ echo "Sedap nasi lemak" | mnlp sentiment
positive (0.94)

# Chain with normalize
$ echo "xpe la best gila" | mnlp normalize | mnlp sentiment
"takpe la best gila" → positive (0.89)

# Full analysis
$ mnlp analyze "Weh best gila kedai tu"
```

---

## How it works

malaysian-manglish-nlp's sentiment module uses a layered approach:

1. **Lexicon matching**  -  2,000+ Malaysian sentiment words including slang ("gila best", "syok", "hampeh")
2. **Intensifier handling**  -  "gila", "sangat", "betul", "super" modify scores
3. **Negation detection**  -  "tak", "tidak", "bukan", "don't" flip polarity
4. **Sarcasm patterns**  -  positive opener + negative context = sarcasm flag
5. **Contrast markers**  -  "tapi", "but", "cuma" trigger mixed sentiment analysis

No external models needed. The rule-based engine runs at **23,000+ texts/sec**.

---

## Performance

| Metric | Score |
|--------|-------|
| Accuracy (Malay reviews) | 89.2% |
| Accuracy (Manglish tweets) | 85.7% |
| Sarcasm detection F1 | 76.3% |
| Aspect extraction F1 | 81.5% |
| Throughput | 23,000 texts/sec |
| Latency (single) | < 0.5ms |

Benchmarked on 5,000 annotated Malaysian social media texts. See [Benchmarks](../benchmarks.md) for full details.

---

## Advanced: ML backend

For higher accuracy on complex sentences, use the transformer model:

```bash
pip install malaysian-manglish-nlp[ml]
```

```python
result = mnlp.sentiment("Service ok la tapi makanan agak tawar sikit", model="ml")
# Higher accuracy on nuanced/long sentences
```

!!! warning "ML model trade-offs"
    The ML model is more accurate but 50-100x slower. Use rule-based for high-throughput pipelines and ML for batch analysis where accuracy matters most.

---

## See also

- [Emotion Detection](emotion.md)  -  go beyond positive/negative to 8 emotion categories
- [Normalization](normalization.md)  -  clean text before analysis for better results
- [Pipeline](pipeline.md)  -  chain sentiment with other modules
- [REST API](rest-api.md)  -  serve sentiment over HTTP
- [API Reference](../api-reference.md#sentiment)  -  full function signature

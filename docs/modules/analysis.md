# Analysis

**Understand what Malaysian text means  -  sentiment, emotion, language, profanity, and sarcasm.**

---

## Overview

Analysis modules extract meaning, tone, and linguistic characteristics from text. They handle code-switched Manglish natively, so you can feed raw Malaysian social media text directly without preprocessing.

Default models are rule-based + statistical (zero dependencies). Install `[ml]` for transformer-backed models with higher accuracy on complex sentences.

```python
import malaysian_manglish_nlp as mnlp
```

---

## Quick Start

```python
import malaysian_manglish_nlp as mnlp

text = "Sedap gila nasi lemak kat kedai tu, tapi service lambat sikit"

mnlp.sentiment(text)
# {'label': 'positive', 'score': 0.78}

mnlp.sentiment(text, aspect=True)
# [{'aspect': 'nasi lemak', 'label': 'positive', 'score': 0.92},
#  {'aspect': 'service', 'label': 'negative', 'score': 0.81}]

mnlp.emotion(text)
# {'primary': 'joy', 'score': 0.71, 'secondary': 'anticipation'}
```

---

## Module Details

### `sentiment`

Analyse sentiment of Malaysian text with code-switching support. Returns positive, negative, or neutral with confidence score.

```python
import malaysian_manglish_nlp as mnlp

mnlp.sentiment("Sedap gila nasi lemak kat kedai tu!")
# {'label': 'positive', 'score': 0.96}

mnlp.sentiment("Teruk la service dia, tunggu 1 jam")
# {'label': 'negative', 'score': 0.89}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text or list of texts |
| `detailed` | `bool` | `False` | Return scores for all classes |
| `aspect` | `bool` | `False` | Aspect-based sentiment (per-entity) |
| `model` | `str` | `"default"` | `"default"` (rule-based) or `"ml"` (transformer) |

!!! example "Detailed Output"
    ```python
    mnlp.sentiment("Best gila!", detailed=True)
    # {'label': 'positive',
    #  'scores': {'positive': 0.96, 'neutral': 0.03, 'negative': 0.01}}
    ```

!!! example "Aspect-Based Sentiment"
    ```python
    mnlp.sentiment("Makanan sedap tapi service slow", aspect=True)
    # [{'aspect': 'makanan', 'label': 'positive', 'score': 0.92},
    #  {'aspect': 'service', 'label': 'negative', 'score': 0.85}]
    ```

!!! tip "Batch Processing"
    Pass a list for efficient batch inference:
    ```python
    mnlp.sentiment(["Best!", "Teruk la", "Ok je"])
    # [{'label': 'positive', ...}, {'label': 'negative', ...}, {'label': 'neutral', ...}]
    ```

---

### `emotion`

Detects specific emotional states beyond positive/negative. Supports 8 emotion labels with intensity scoring.

**Supported emotions:** `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `trust`, `anticipation`

```python
import malaysian_manglish_nlp as mnlp

mnlp.emotion("Geram betul aku dengan dia, dah la lambat pastu buat hal")
# {'primary': 'anger', 'score': 0.88, 'secondary': 'frustration'}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `multi` | `bool` | `False` | Return multiple emotion labels |
| `intensity` | `bool` | `False` | Include intensity score (1–5) |

!!! example "Multi-Label Emotions"
    ```python
    mnlp.emotion("Takut gila tapi excited jugak", multi=True)
    # [{'label': 'fear', 'score': 0.82},
    #  {'label': 'anticipation', 'score': 0.65}]
    ```

!!! example "Intensity Scoring"
    ```python
    mnlp.emotion("MARAH GILA AKU!!!", intensity=True)
    # {'primary': 'anger', 'score': 0.97, 'intensity': 5}
    ```

---

### `language`

Detect language composition in mixed-language text. Supports per-token detection and regional Malaysian dialect identification.

```python
import malaysian_manglish_nlp as mnlp

mnlp.language("Eh jom la we go makan, I lapar gila already")
# {'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `per_token` | `bool` | `False` | Language label for each token |
| `dialect` | `bool` | `False` | Detect regional Malay dialect |

**Supported languages:** `ms`, `en`, `zh`, `ta`, `manglish`, `mixed`

!!! example "Per-Token Detection"
    ```python
    mnlp.language("Eh jom la we go makan", per_token=True)
    # [('Eh', 'ms'), ('jom', 'ms'), ('la', 'ms'),
    #  ('we', 'en'), ('go', 'en'), ('makan', 'ms')]
    ```

!!! example "Dialect Detection"
    ```python
    mnlp.language("Ambo nok gi make", dialect=True)
    # {'primary': 'ms', 'dialect': 'kelantan', 'confidence': 0.82}
    ```

!!! tip "Supported Dialects"
    Kelantan, Terengganu, Kedah, Negeri Sembilan, Sarawak, and Sabah Malay dialects are detectable.

---

### `profanity`

Detect and filter profanity in Malaysian languages including slang variants, leetspeak, and euphemisms.

```python
import malaysian_manglish_nlp as mnlp

mnlp.profanity("Bodoh la kau ni, sial betul")
# {'has_profanity': True, 'words': ['bodoh', 'sial'], 'severity': 'medium'}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `censor` | `bool` | `False` | Return censored version of text |
| `char` | `str` | `"*"` | Character used for censoring |
| `min_severity` | `str` | `"low"` | Minimum severity to flag: `"low"`, `"medium"`, `"high"` |
| `leetspeak` | `bool` | `False` | Detect leetspeak variants (`b0d0h`, etc.) |
| `context_aware` | `bool` | `False` | Reduce false positives for casual friend-speak |

!!! example "Censoring"
    ```python
    mnlp.profanity("Bodoh la kau ni", censor=True)
    # "B***h la kau ni"

    mnlp.profanity("Bodoh la kau ni", censor=True, char="█")
    # "█████ la kau ni"
    ```

!!! warning "Cultural Context"
    Words like `"sial"` are profane in formal contexts but casual among friends. Enable `context_aware=True` for social media moderation to reduce false positives.

---

### `sarcasm`

Detect sarcasm and irony in Malaysian text. Identifies linguistic cues like exaggerated praise, parenthetical remarks, and tonal contradictions.

```python
import malaysian_manglish_nlp as mnlp

mnlp.sarcasm("Wah bagus la tu, memang pandai")
# {'is_sarcastic': True, 'confidence': 0.78, 'cues': ['wah', 'memang']}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text |
| `explain` | `bool` | `False` | Include explanation of why text is flagged |

!!! example "With Explanation"
    ```python
    mnlp.sarcasm("Memang terbaik service dia (tunggu 2 jam)", explain=True)
    # {'is_sarcastic': True, 'confidence': 0.91,
    #  'cues': ['memang terbaik', '(tunggu 2 jam)'],
    #  'explanation': 'Exaggerated praise contradicted by parenthetical complaint'}
    ```

!!! note "Accuracy"
    Sarcasm detection achieves **~75% accuracy** on Malaysian social media benchmarks. Context markers (parenthetical remarks, excessive praise, emoji mismatch) significantly improve detection.

---

## `analyze_aspect_sentiment` (v3.3.0)

Per-aspect sentiment analysis with domain-specific aspect extraction. Detects sentiment for individual aspects (food, service, price, etc.) within a single text, including conflict detection when different aspects have opposing sentiments.

**Supported domains:** `restaurant`, `product`, `app`, `general`

```python
import malaysian_manglish_nlp as mnlp

mnlp.analyze_aspect_sentiment("makanan sedap tapi service teruk", domain="restaurant")
# {'aspects': [
#   {'aspect': 'food', 'sentiment': 'positive', 'confidence': 0.94},
#   {'aspect': 'service', 'sentiment': 'negative', 'confidence': 0.89}
#  ],
#  'conflict': True,
#  'overall': 'mixed'}

mnlp.analyze_aspect_sentiment("harga mahal tapi quality memang tip top", domain="product")
# {'aspects': [
#   {'aspect': 'price', 'sentiment': 'negative', 'confidence': 0.87},
#   {'aspect': 'quality', 'sentiment': 'positive', 'confidence': 0.93}
#  ],
#  'conflict': True,
#  'overall': 'mixed'}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text or list of texts |
| `domain` | `str` | `"general"` | Domain for aspect extraction: `"restaurant"`, `"product"`, `"app"`, `"general"` |
| `conflict_detect` | `bool` | `True` | Detect conflicting sentiments across aspects |

!!! example "Batch Processing"
    ```python
    mnlp.aspect_sentiment_batch([
        "nasi lemak sedap, air manis",
        "app crash selalu, bugs berlambak"
    ], domain="restaurant")
    ```

!!! tip "Get Aspect Categories"
    ```python
    mnlp.get_aspect_categories()
    # {'restaurant': ['food', 'service', 'price', 'ambience', 'waiting_time'],
    #  'product': ['quality', 'price', 'design', 'durability'],
    #  'app': ['performance', 'ui', 'features', 'stability'],
    #  'general': ['overall']}
    ```

---

## `detect_multi_emotion` (v3.3.0)

Detect multiple emotions simultaneously with confidence scores. Unlike the single-label `emotion` module, this captures complex emotional states like bittersweet feelings.

**Supported co-occurrence patterns:** bittersweet, anxious, nostalgic, conflicted, overwhelmed, relieved, proud, excited, melancholic, grateful

```python
import malaysian_manglish_nlp as mnlp

mnlp.detect_multi_emotion("sedih tapi grateful dapat jumpa family")
# {'emotions': [
#   {'emotion': 'happy', 'confidence': 0.62},
#   {'emotion': 'sad', 'confidence': 0.38}
#  ],
#  'dominant': 'happy',
#  'is_multi': True,
#  'co_occurrence': 'bittersweet'}

mnlp.detect_multi_emotion("takut gila tapi excited jugak nak start kerja baru")
# {'emotions': [
#   {'emotion': 'fear', 'confidence': 0.55},
#   {'emotion': 'anticipation', 'confidence': 0.45}
#  ],
#  'dominant': 'fear',
#  'is_multi': True,
#  'co_occurrence': 'anxious'}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text or list of texts |
| `threshold` | `float` | `0.3` | Minimum confidence to include an emotion |
| `max_emotions` | `int` | `3` | Maximum emotions to return |

!!! example "Batch Processing"
    ```python
    mnlp.detect_multi_emotion_batch([
        "happy tapi rindu kampung",
        "marah tapi faham situasi"
    ])
    ```

!!! tip "Get Co-occurrence Patterns"
    ```python
    mnlp.get_co_occurrence_patterns()
    # {'bittersweet': ['happy', 'sad'],
    #  'anxious': ['fear', 'anticipation'],
    #  'nostalgic': ['happy', 'sad', 'love'],
    #  ...}
    ```

---

## Combining Analysis Modules

```python
text = "Wah pandai la kau, janji Melayu kan"

sentiment = mnlp.sentiment(text)         # neutral (misses sarcasm)
sarcasm   = mnlp.sarcasm(text)           # {'is_sarcastic': True, ...}
emotion   = mnlp.emotion(text)           # {'primary': 'disgust', ...}

# Use sarcasm flag to re-interpret sentiment
if sarcasm['is_sarcastic']:
    sentiment['label'] = 'negative'      # correct interpretation
```

---

## See Also

- [Text Processing](text-processing.md)  -  clean text before analysis for better accuracy
- [Advanced](advanced.md)  -  hate speech, stance detection, code-switching analysis
- [Calibration](tools.md#calibration)  -  calibrate confidence scores for production thresholds
- [Feedback Loop](feedback.md)  -  submit corrections to improve model accuracy

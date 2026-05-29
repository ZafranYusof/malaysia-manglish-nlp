# Analysis

**Understand what Malaysian text means  -  sentiment, emotion, language, profanity, and sarcasm.**

---

## Overview

Analysis modules extract meaning, tone, and linguistic characteristics from text. They handle code-switched Manglish natively, so you can feed raw Malaysian social media text directly without preprocessing.

Default models are rule-based + statistical (zero dependencies). Install `[ml]` for transformer-backed models with higher accuracy on complex sentences.

```python
import manglish_nlp as mnlp
```

---

## Quick Start

```python
import manglish_nlp as mnlp

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
import manglish_nlp as mnlp

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
import manglish_nlp as mnlp

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
import manglish_nlp as mnlp

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
import manglish_nlp as mnlp

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
import manglish_nlp as mnlp

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

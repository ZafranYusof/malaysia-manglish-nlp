# Analysis

Modules for understanding text meaning, emotion, and characteristics.

---

## sentiment

Analyze sentiment of Malaysian text with code-switching support.

```python
import manglish_nlp as mnlp

result = mnlp.sentiment("Sedap gila nasi lemak kat kedai tu!")
print(result)
# {'label': 'positive', 'score': 0.96}

result = mnlp.sentiment("Teruk la service dia, tunggu 1 jam")
print(result)
# {'label': 'negative', 'score': 0.89}
```

### Options

```python
# Detailed output with all class scores
mnlp.sentiment(text, detailed=True)
# {'label': 'positive', 'scores': {'positive': 0.96, 'neutral': 0.03, 'negative': 0.01}}

# Aspect-based sentiment
mnlp.sentiment("Makanan sedap tapi service slow", aspect=True)
# [{'aspect': 'makanan', 'label': 'positive', 'score': 0.92},
#  {'aspect': 'service', 'label': 'negative', 'score': 0.85}]

# Batch processing
results = mnlp.sentiment(["Best!", "Teruk la", "Ok je"])
```

!!! info "Model Backend"
    Default uses rule-based + statistical model (zero deps). Install `[ml]` for transformer-based model with higher accuracy on complex sentences.

---

## emotion

Detect emotions in text — goes beyond positive/negative to specific emotional states.

```python
result = mnlp.emotion("Geram betul aku dengan dia, dah la lambat pastu buat hal")
print(result)
# {'primary': 'anger', 'score': 0.88, 'secondary': 'frustration'}
```

### Supported Emotions

`joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, `trust`, `anticipation`

```python
# Multi-label emotions
mnlp.emotion(text, multi=True)
# [{'label': 'anger', 'score': 0.88}, {'label': 'frustration', 'score': 0.72}]

# Emotion intensity (1-5 scale)
mnlp.emotion(text, intensity=True)
# {'primary': 'anger', 'score': 0.88, 'intensity': 4}
```

---

## language

Detect language composition in mixed-language text.

```python
text = "Eh jom la we go makan, I lapar gila already"
result = mnlp.language(text)
print(result)
# {'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

### Options

```python
# Per-token language detection
mnlp.language(text, per_token=True)
# [('Eh', 'ms'), ('jom', 'ms'), ('la', 'ms'), ('we', 'en'), ('go', 'en'),
#  ('makan', 'ms'), ('I', 'en'), ('lapar', 'ms'), ('gila', 'ms'), ('already', 'en')]

# Detect dialect
mnlp.language("Ambo nok gi make", dialect=True)
# {'primary': 'ms', 'dialect': 'kelantan', 'confidence': 0.82}

# Supported languages
# ms, en, zh, ta, manglish, mixed
```

!!! tip "Dialect Detection"
    The `dialect=True` option can identify regional Malaysian dialects including Kelantan, Terengganu, Kedah, Negeri Sembilan, and Sarawak.

---

## profanity

Detect and filter profanity in Malaysian languages including slang variants.

```python
text = "Bodoh la kau ni, sial betul"
result = mnlp.profanity(text)
print(result)
# {'has_profanity': True, 'words': ['bodoh', 'sial'], 'severity': 'medium'}
```

### Options

```python
# Censor text
mnlp.profanity(text, censor=True)
# "B***h la kau ni, s**l betul"

# Custom censor character
mnlp.profanity(text, censor=True, char="█")
# "█████ la kau ni, ████ betul"

# Severity levels: low, medium, high
mnlp.profanity(text, min_severity="high")

# Include leetspeak variants (b0d0h, etc.)
mnlp.profanity(text, leetspeak=True)
```

!!! warning "Cultural Context"
    Some words are profane in certain contexts but casual in others (e.g., "sial" among friends). Use `context_aware=True` for better accuracy in informal settings.

---

## sarcasm

Detect sarcasm and irony in Malaysian text.

```python
text = "Wah bagus la tu, memang pandai"
result = mnlp.sarcasm(text)
print(result)
# {'is_sarcastic': True, 'confidence': 0.78, 'cues': ['wah', 'memang']}
```

### Options

```python
# With explanation
mnlp.sarcasm(text, explain=True)
# {'is_sarcastic': True, 'confidence': 0.78,
#  'explanation': 'Exaggerated praise pattern with contradicting context'}

# Batch detection
texts = ["Best la kau ni", "Memang terbaik service dia (tunggu 2 jam)"]
mnlp.sarcasm(texts)
```

!!! note "Accuracy"
    Sarcasm detection is inherently challenging. The model achieves ~75% accuracy on Malaysian social media text. Context and tone markers (e.g., parenthetical remarks, excessive praise) improve detection.

---

## See Also

- [Text Processing](text-processing.md) — clean text before analysis
- [Advanced modules](advanced.md) — hate speech, stance detection

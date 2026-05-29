# Hate Speech Detection

**Detect hate speech across 6 target categories with severity levels and leetspeak evasion handling.**

---

## Why hate speech detection?

Content moderation for social media platforms, forum management, brand safety, and research on online toxicity. Malaysian hate speech is particularly challenging because it uses code-switching, local slurs, religious references, and leetspeak to evade basic filters.

---

## Load module

```python
import manglish_nlp as mnlp

result = mnlp.detect_hate_speech("Text to check here")
print(result)
# {'is_hate': False, 'severity': 'none', 'categories': [], 'score': 0.12}
```

---

## Basic usage

### Non-hate speech

```python
mnlp.detect_hate_speech("Sedap gila nasi lemak kat sini!")
# {'is_hate': False, 'severity': 'none', 'categories': [], 'score': 0.05}

mnlp.detect_hate_speech("Aku tak setuju dengan polisi kerajaan")
# {'is_hate': False, 'severity': 'none', 'categories': [], 'score': 0.15}
```

!!! tip "Criticism is not hate speech"
    The module distinguishes between legitimate criticism/opinion and actual hate speech. Political disagreement, product complaints, and negative reviews are not flagged.

### Detected hate speech

```python
result = mnlp.detect_hate_speech("Offensive text here")
# {'is_hate': True, 'severity': 'high', 'categories': ['race'], 'score': 0.92}
```

---

## 6 target categories

| Category | Description |
|----------|-------------|
| `race` | Racist content targeting ethnicity |
| `religion` | Content attacking religious groups |
| `gender` | Sexist or misogynistic content |
| `nationality` | Xenophobic content targeting nationality |
| `disability` | Content mocking disabilities |
| `sexual_orientation` | Homophobic or transphobic content |

### Check severity

```python
mnlp.get_severity("text here")
# 'none' | 'low' | 'medium' | 'high'
```

### Check target groups

```python
mnlp.get_target_groups("text here")
# [] or ['race', 'religion', ...]
```

---

## Severity levels

| Level | Description | Action |
|-------|-------------|--------|
| `none` | Clean text | No action |
| `low` | Mildly offensive, borderline | Flag for review |
| `medium` | Clearly offensive | Moderate |
| `high` | Severe hate speech | Remove immediately |

```python
# Quick binary check
mnlp.is_hate("some text")
# True or False
```

---

## Leetspeak and evasion detection

Malaysians sometimes use leetspeak or spelling tricks to evade filters. manglish-nlp detects common patterns:

```python
# Number substitution
# "b0d0h" instead of "bodoh"
# Racial slurs with @, $, 0 substitutions

# Deliberate misspelling
# Split words, extra characters

# Code-switched slurs
# Mixed BM/EN offensive phrases
```

The module normalizes text internally before detection:

```python
# These evasion attempts are caught:
# - Character substitution (0→o, @→a, $→s, 1→i)
# - Deliberate spacing ("b o d o h")
# - Repeated characters ("booodohh")
# - Abbreviated slurs
```

---

## Batch processing

```python
texts = [
    "Sedap gila makanan ni!",
    "Normal sentence here",
    "Another clean text",
    "Offensive content here",
]

results = mnlp.hate_detect_batch(texts)
for text, result in zip(texts, results):
    status = "🚫 HATE" if result['is_hate'] else "✅ Clean"
    print(f"{status} | {text[:40]}")
```

---

## Integration with content moderation

```python
def moderate_comment(text):
    """Example moderation pipeline."""
    # Check hate speech
    hate = mnlp.detect_hate_speech(text)
    
    if hate['severity'] == 'high':
        return {'action': 'block', 'reason': 'hate_speech'}
    
    if hate['severity'] == 'medium':
        return {'action': 'flag', 'reason': 'review_needed'}
    
    # Check profanity too
    profanity = mnlp.detect_profanity(text)
    if profanity['severity'] == 'high':
        return {'action': 'censor', 'censored': mnlp.censor(text)}
    
    return {'action': 'approve'}

# Usage
moderate_comment("Normal friendly comment")
# {'action': 'approve'}
```

---

## CLI usage

```bash
# Full analysis (includes hate speech check)
$ mnlp analyze "text to check"

# With JSON
$ mnlp analyze "text" --json
```

---

## How it works

1. **Normalization** — leetspeak decoding, repeated char reduction
2. **Lexicon matching** — curated hate speech dictionaries for Malaysian context
3. **Pattern detection** — known hate speech phrases and structures
4. **Context analysis** — distinguish insults from hate speech
5. **Severity classification** — score determines none/low/medium/high
6. **Category assignment** — which target group(s) are attacked

---

## Performance

| Metric | Score |
|--------|-------|
| Binary F1 | 86.3% |
| Severity accuracy | 79.8% |
| Category F1 (macro) | 74.5% |
| Leetspeak detection | 82.1% |
| False positive rate | 3.2% |
| Throughput | 18,000 texts/sec |

!!! warning "False positives"
    3.2% false positive rate means some legitimate text gets flagged. Always provide an appeal mechanism in production moderation systems.

---

## See also

- [Profanity Detection](../modules/analysis.md#profanity) — filter swear words (separate from hate speech)
- [Emotion Detection](emotion.md) — detect anger/disgust that may accompany hate speech
- [Sentiment Analysis](sentiment.md) — general sentiment classification
- [Pipeline](pipeline.md) — chain hate speech with other modules
- [REST API](rest-api.md) — serve moderation over HTTP
- [API Reference](../api-reference.md#hate_speech) — full function signature

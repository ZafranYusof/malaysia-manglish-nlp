# Advanced

Advanced NLP modules for complex linguistic analysis.

---

## code_switching

Detect and analyze code-switching patterns between languages within text.

```python
import manglish_nlp as mnlp

text = "I think kita should go makan first, then baru discuss"
result = mnlp.code_switching(text)
print(result)
# {'switches': 4, 'pattern': 'inter-sentential',
#  'segments': [('I think', 'en'), ('kita', 'ms'), ('should go', 'en'),
#               ('makan', 'ms'), ('first, then', 'en'), ('baru', 'ms'), ('discuss', 'en')]}
```

### Options

```python
# Switch point analysis
mnlp.code_switching(text, points=True)
# [{'position': 2, 'from': 'en', 'to': 'ms', 'trigger': 'pronoun_switch'}]

# Matrix language detection
mnlp.code_switching(text, matrix=True)
# {'matrix_language': 'en', 'embedded_language': 'ms', 'ratio': 0.57}

# Classify switching type
mnlp.code_switching(text, classify=True)
# 'inter-word'  (vs 'intra-word', 'tag-switching')
```

!!! info "Code-Switching Types"
    - **Inter-sentential**: switching between sentences
    - **Intra-sentential**: switching within a sentence
    - **Tag-switching**: inserting tags/particles from another language (e.g., "la", "right?")

---

## intent

Classify user intent from Malaysian text — useful for chatbots and dialog systems.

```python
text = "Nak tanya, kedai tu bukak pukul berapa eh?"
result = mnlp.intent(text)
print(result)
# {'intent': 'question_info', 'confidence': 0.91,
#  'slots': {'entity': 'kedai', 'attribute': 'operating_hours'}}
```

### Supported Intents

| Intent | Example |
|--------|---------|
| `question_info` | "Berapa harga tu?" |
| `request_action` | "Tolong bukak kan pintu" |
| `complaint` | "Service teruk la kat sini" |
| `greeting` | "Assalamualaikum, apa khabar?" |
| `farewell` | "Ok la, jumpa nanti" |
| `confirmation` | "Ok boleh, set" |
| `negation` | "Taknak la, mahal sangat" |
| `opinion` | "Aku rasa best gila movie tu" |

### Options

```python
# Multi-intent detection
mnlp.intent(text, multi=True)
# [{'intent': 'question_info', 'score': 0.91}, {'intent': 'request_action', 'score': 0.12}]

# Custom intent labels
mnlp.intent(text, labels=["order", "cancel", "track", "support"])

# With slot filling
mnlp.intent("Nak order 2 nasi lemak extra sambal", slots=True)
# {'intent': 'order', 'slots': {'item': 'nasi lemak', 'quantity': 2, 'modifier': 'extra sambal'}}
```

---

## topic

Topic modeling and classification for Malaysian text.

```python
text = "Harga minyak naik lagi, memang susah rakyat nak survive"
result = mnlp.topic(text)
print(result)
# {'topic': 'economy', 'subtopic': 'cost_of_living', 'confidence': 0.87}
```

### Options

```python
# Multiple topics
mnlp.topic(text, top_k=3)
# [('economy', 0.87), ('politics', 0.45), ('social', 0.23)]

# Custom topic labels
mnlp.topic(text, labels=["sports", "politics", "entertainment", "tech"])

# Topic modeling on corpus
corpus = ["text1", "text2", "text3", ...]
topics = mnlp.topic(corpus, mode="model", n_topics=10)
# Returns topic clusters with representative words
```

---

## hate_speech

Detect hate speech and offensive content targeting Malaysian communities.

```python
text = "Semua bangsa X memang macam tu, tak boleh dipercayai"
result = mnlp.hate_speech(text)
print(result)
# {'is_hate': True, 'target': 'race', 'severity': 'high', 'confidence': 0.92}
```

### Target Categories

`race`, `religion`, `gender`, `nationality`, `disability`, `sexual_orientation`

### Options

```python
# Detailed classification
mnlp.hate_speech(text, detailed=True)
# {'is_hate': True, 'type': 'dehumanization', 'target': 'race',
#  'severity': 'high', 'spans': [(6, 14, 'target_group')]}

# Distinguish hate vs offensive
mnlp.hate_speech(text, granular=True)
# 'hate' | 'offensive' | 'neither'

# Content moderation mode (returns action recommendation)
mnlp.hate_speech(text, moderate=True)
# {'action': 'remove', 'reason': 'racial_hatred', 'confidence': 0.92}
```

!!! warning "Sensitivity"
    Hate speech detection involves sensitive content. The model is trained on Malaysian social media data and understands local slurs, coded language, and dog whistles specific to the Malaysian context.

---

## stance

Detect stance (support/oppose/neutral) toward a target topic or claim.

```python
text = "Memang patut la naikkan gaji minimum, dah lama tak naik"
result = mnlp.stance(text, target="minimum wage increase")
print(result)
# {'stance': 'support', 'confidence': 0.88}
```

### Options

```python
# Without explicit target (auto-detect)
mnlp.stance(text)
# {'stance': 'support', 'target_detected': 'wage increase', 'confidence': 0.85}

# Multi-target stance
mnlp.stance(text, targets=["wage increase", "government policy", "cost of living"])
# [{'target': 'wage increase', 'stance': 'support', 'score': 0.88}, ...]

# Stance with reasoning
mnlp.stance(text, target="minimum wage increase", explain=True)
# {'stance': 'support', 'confidence': 0.88,
#  'cues': ['patut', 'dah lama tak naik']}
```

---

## coreference

Resolve coreferences (pronouns, mentions) in Malaysian text.

```python
text = "Ahmad jumpa Siti kat mall. Dia cakap dia nak balik awal."
result = mnlp.coreference(text)
print(result)
# {'clusters': [
#   [('Ahmad', 0, 5), ('Dia', 26, 29)],
#   [('Siti', 12, 16), ('dia', 34, 37)]
# ]}
```

### Options

```python
# Resolve and replace
mnlp.coreference(text, resolve=True)
# "Ahmad jumpa Siti kat mall. Ahmad cakap Siti nak balik awal."

# Return mention chains
mnlp.coreference(text, chains=True)
# [{'entity': 'Ahmad', 'mentions': ['Ahmad', 'Dia']},
#  {'entity': 'Siti', 'mentions': ['Siti', 'dia']}]
```

!!! note "Ambiguity"
    Malay pronouns (dia, mereka) are gender-neutral, making coreference resolution more challenging. The model uses contextual cues and world knowledge to resolve ambiguous cases.

---

## discourse

Analyze discourse structure and rhetorical relations in text.

```python
text = "Walaupun hujan lebat, Ahmad tetap pergi kerja sebab deadline esok."
result = mnlp.discourse(text)
print(result)
# {'relations': [
#   {'type': 'concession', 'arg1': 'hujan lebat', 'arg2': 'Ahmad tetap pergi kerja'},
#   {'type': 'cause', 'arg1': 'deadline esok', 'arg2': 'pergi kerja'}
# ]}
```

### Supported Relations

`cause`, `contrast`, `concession`, `elaboration`, `condition`, `temporal`, `purpose`, `result`

### Options

```python
# Full RST tree
mnlp.discourse(text, format="tree")

# Connective detection
mnlp.discourse(text, connectives=True)
# [{'connective': 'walaupun', 'type': 'concession', 'position': 0},
#  {'connective': 'sebab', 'type': 'cause', 'position': 42}]
```

---

## See Also

- [Analysis modules](analysis.md) — basic sentiment and emotion analysis
- [Extraction modules](extraction.md) — NER, POS tagging
- [Generation modules](generation.md) — text generation and summarization

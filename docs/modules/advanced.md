# Advanced NLP

**Complex linguistic analysis  -  code-switching, intent, topic, hate speech, stance, and discourse structure.**

---

## Overview

Advanced modules handle higher-order linguistic phenomena unique to Malaysian multilingual text. These require the `[ml]` extra and are designed for production chatbots, content moderation systems, and research applications.

```bash
pip install malaysian-manglish-nlp[ml]
```

```python
import malaysian_manglish_nlp as mnlp
```

---

## Quick Start

```python
import malaysian_manglish_nlp as mnlp

# Code-switching detection
mnlp.code_switching("I think kita should go makan first, then baru discuss")
# {'switches': 4, 'pattern': 'intra-sentential',
#  'segments': [('I think', 'en'), ('kita', 'ms'), ('should go', 'en'),
#               ('makan', 'ms'), ('first, then', 'en'), ('baru', 'ms'), ('discuss', 'en')]}

# Intent classification
mnlp.intent("Nak tanya, kedai tu bukak pukul berapa eh?")
# {'intent': 'question_info', 'confidence': 0.91,
#  'slots': {'entity': 'kedai', 'attribute': 'operating_hours'}}

# Hate speech moderation
mnlp.hate_speech("Semua kaum X memang sampah masyarakat")
# {'is_hate': True, 'target': 'race', 'severity': 'high', 'confidence': 0.94}
```

---

## Module Details

### `code_switching`

Detect and analyse code-switching patterns between languages. Identifies switch points, matrix language, and switching type.

```python
import malaysian_manglish_nlp as mnlp

text = "I think kita should go makan first, then baru discuss"
mnlp.code_switching(text)
# {'switches': 4, 'pattern': 'intra-sentential',
#  'segments': [('I think', 'en'), ('kita', 'ms'), ('should go', 'en'),
#               ('makan', 'ms'), ('first, then', 'en'), ('baru', 'ms'), ('discuss', 'en')]}
```

#### Switching Types

| Type | Description | Example |
|------|-------------|---------|
| `inter-sentential` | Switch between sentences | "Best movie. Tapi ending hampeh." |
| `intra-sentential` | Switch within a sentence | "I rasa macam nak pergi" |
| `tag-switching` | Insert particles/tags | "Good la, very nice right?" |
| `intra-word` | Morpheme mixing | "download-kan", "upload-lah" |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `points` | `bool` | `False` | Return detailed switch point analysis |
| `matrix` | `bool` | `False` | Identify matrix vs embedded language |
| `classify` | `bool` | `False` | Classify switching type |

!!! example "Switch Point Analysis"
    ```python
    mnlp.code_switching(text, points=True)
    # [{'position': 2, 'from': 'en', 'to': 'ms', 'trigger': 'pronoun_switch'},
    #  {'position': 3, 'from': 'ms', 'to': 'en', 'trigger': 'verb_switch'}]
    ```

!!! example "Matrix Language"
    ```python
    mnlp.code_switching(text, matrix=True)
    # {'matrix_language': 'en', 'embedded_language': 'ms', 'ratio': 0.57}
    ```

---

### `intent`

Classify user intent for chatbots and dialogue systems. Returns intent label, confidence, and extracted slots.

```python
import malaysian_manglish_nlp as mnlp

mnlp.intent("Nak order 2 nasi lemak extra sambal")
# {'intent': 'request_action', 'confidence': 0.89,
#  'slots': {'item': 'nasi lemak', 'quantity': 2, 'modifier': 'extra sambal'}}
```

#### Intent Categories

| Intent | Description | Example |
|--------|-------------|---------|
| `question_info` | Asking for information | "Berapa harga tu?" |
| `request_action` | Requesting an action | "Tolong bukakkan pintu" |
| `complaint` | Expressing dissatisfaction | "Service teruk la kat sini" |
| `greeting` | Opening a conversation | "Assalamualaikum, apa khabar?" |
| `farewell` | Closing a conversation | "Ok la, jumpa nanti" |
| `confirmation` | Agreeing / confirming | "Ok boleh, set" |
| `negation` | Declining / rejecting | "Taknak la, mahal sangat" |
| `opinion` | Expressing a view | "Aku rasa best gila movie tu" |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input utterance |
| `multi` | `bool` | `False` | Detect multiple intents |
| `labels` | `list[str]` | `None` | Custom intent labels (override defaults) |
| `slots` | `bool` | `True` | Extract slot values |

!!! example "Custom Labels for Domain Bots"
    ```python
    mnlp.intent("Nak track parcel aku", labels=["order", "cancel", "track", "support"])
    # {'intent': 'track', 'confidence': 0.93, 'slots': {'item': 'parcel'}}
    ```

---

### `topic`

Topic classification and unsupervised topic modelling for Malaysian text.

```python
import malaysian_manglish_nlp as mnlp

mnlp.topic("Harga minyak naik lagi, memang susah rakyat nak survive")
# {'topic': 'economy', 'subtopic': 'cost_of_living', 'confidence': 0.87}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text or corpus |
| `top_k` | `int` | `1` | Number of topics to return |
| `labels` | `list[str]` | `None` | Custom topic labels |
| `mode` | `str` | `"classify"` | `"classify"` (single text) or `"model"` (corpus clustering) |
| `n_topics` | `int` | `10` | Number of topics for unsupervised modelling |

!!! example "Multi-Topic Classification"
    ```python
    mnlp.topic("Harga minyak naik lagi", top_k=3)
    # [('economy', 0.87), ('politics', 0.45), ('social', 0.23)]
    ```

!!! example "Unsupervised Topic Modelling"
    ```python
    corpus = [doc1, doc2, doc3, ...]  # hundreds of articles
    topics = mnlp.topic(corpus, mode="model", n_topics=10)
    # Returns clusters with representative words per topic
    ```

---

### `hate_speech`

Detect hate speech and offensive content targeting Malaysian communities. Understands local slurs, coded language, and dog whistles specific to the Malaysian context.

```python
import malaysian_manglish_nlp as mnlp

mnlp.hate_speech("Semua bangsa X memang macam tu, tak boleh dipercayai")
# {'is_hate': True, 'target': 'race', 'severity': 'high', 'confidence': 0.92}
```

#### Severity Levels

| Level | Description | Example |
|-------|-------------|---------|
| `low` | Offensive but not dehumanising | Casual slurs among peers |
| `medium` | Stereotyping, generalisation | "Semua orang X memang pemalas" |
| `high` | Dehumanising, inciting hatred | Calls for exclusion or violence |

#### Target Categories

`race`, `religion`, `gender`, `nationality`, `disability`, `sexual_orientation`

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `detailed` | `bool` | `False` | Include type classification and entity spans |
| `granular` | `bool` | `False` | Distinguish `hate` vs `offensive` vs `neither` |
| `moderate` | `bool` | `False` | Return moderation action recommendation |

!!! example "Content Moderation Mode"
    ```python
    mnlp.hate_speech(text, moderate=True)
    # {'action': 'remove', 'reason': 'racial_hatred', 'confidence': 0.92}
    ```

!!! warning "Sensitive Content"
    This module processes hate speech for detection purposes. It does not generate or endorse such content. Use responsibly for moderation systems.

---

### `stance`

Detect stance (support / oppose / neutral) toward a target topic or claim.

```python
import malaysian_manglish_nlp as mnlp

mnlp.stance("Memang patut la naikkan gaji minimum, dah lama tak naik",
            target="minimum wage increase")
# {'stance': 'support', 'confidence': 0.88}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `target` | `str` | `None` | Target topic (auto-detected if omitted) |
| `targets` | `list[str]` | `None` | Check stance against multiple targets |
| `explain` | `bool` | `False` | Return linguistic cues driving the classification |

!!! example "Multi-Target Stance"
    ```python
    mnlp.stance(text, targets=["wage increase", "government policy"])
    # [{'target': 'wage increase', 'stance': 'support', 'score': 0.88},
    #  {'target': 'government policy', 'stance': 'neutral', 'score': 0.52}]
    ```

!!! example "Stance with Explanation"
    ```python
    mnlp.stance(text, target="minimum wage", explain=True)
    # {'stance': 'support', 'confidence': 0.88,
    #  'cues': ['patut', 'dah lama tak naik']}
    ```

---

### `discourse`

Analyse discourse structure and rhetorical relations in text using Rhetorical Structure Theory (RST) adapted for Malay.

```python
import malaysian_manglish_nlp as mnlp

text = "Walaupun hujan lebat, Ahmad tetap pergi kerja sebab deadline esok."
mnlp.discourse(text)
# {'relations': [
#   {'type': 'concession', 'arg1': 'hujan lebat', 'arg2': 'Ahmad tetap pergi kerja'},
#   {'type': 'cause', 'arg1': 'deadline esok', 'arg2': 'pergi kerja'}
# ]}
```

#### Supported Relations

| Relation | Malay Connectives | Example |
|----------|-------------------|---------|
| `cause` | sebab, kerana | "Dia marah sebab lambat" |
| `contrast` | tapi, tetapi | "Mahal tapi berbaloi" |
| `concession` | walaupun, biar pun | "Walaupun penat, dia teruskan" |
| `elaboration` | iaitu, misalnya | "Buah tropika, misalnya durian" |
| `condition` | kalau, jika | "Kalau hujan, bawa payung" |
| `temporal` | lepas, sebelum, sambil | "Lepas makan, dia tidur" |
| `purpose` | supaya, untuk | "Belajar rajin supaya lulus" |
| `result` | maka, jadi | "Hujan lebat, jadi banjir" |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |
| `format` | `str` | `"list"` | Output format: `"list"`, `"tree"` |
| `connectives` | `bool` | `False` | Return detected connectives with positions |

!!! example "Connective Detection"
    ```python
    mnlp.discourse(text, connectives=True)
    # [{'connective': 'walaupun', 'type': 'concession', 'position': 0},
    #  {'connective': 'sebab', 'type': 'cause', 'position': 42}]
    ```

---

## See Also

- [Analysis](analysis.md)  -  sentiment, emotion, and sarcasm detection
- [Extraction](extraction.md)  -  NER, POS, and dependency parsing
- [Intent + Pipeline](tools.md#pipeline)  -  chain intent detection with slot extraction for chatbots
- [Evaluate](tools.md#evaluate)  -  benchmark classification accuracy on your data

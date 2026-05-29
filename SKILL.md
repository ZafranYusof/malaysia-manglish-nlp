# manglish-nlp

Natural Language Processing toolkit for Malaysian Manglish — the informal mix of Bahasa Melayu, English, and local slang used by 30+ million Malaysians daily.

Zero external dependencies. Pure Python 3.8+. Inspired by [Malaya](https://github.com/mesolitica/malaya).

## Installation

```bash
cd ~/.agents/skills/manglish-nlp
pip install -e .
```

Or use directly:
```python
import sys
sys.path.insert(0, '/path/to/manglish-nlp')
import manglish_nlp
```

## Quick Start

```python
import manglish_nlp

# Normalize shortforms
manglish_nlp.normalize("nk tnya brapa sem utk grad")
# → 'nak tanya berapa semester untuk grad'

# Detect language
manglish_nlp.detect_language("aku nak go buy food then balik")
# → {'language': 'manglish', 'bm_ratio': 0.6, 'en_ratio': 0.3, ...}

# Sentiment analysis
manglish_nlp.sentiment("gila best makanan dia")
# → {'sentiment': 'positive', 'score': 1.0, ...}

# Tokenize
manglish_nlp.tokenize("aku nk pergi la weh!")
# → ['aku', 'nk', 'pergi', 'la', 'weh', '!']

# Stem
manglish_nlp.stem("memakan berlarian pelajaran")
# → 'makan lari ajar'

# POS tag
manglish_nlp.pos_tag("aku nak pergi kedai")
# → [('aku', 'PRP'), ('nak', 'MD'), ('pergi', 'VB'), ('kedai', 'NN')]

# Named Entity Recognition
manglish_nlp.ner_tag("Jumpa kat UMPSA KL bayar RM50")
# → [{'text': 'UMPSA', 'type': 'ORGANIZATION'}, {'text': 'KL', 'type': 'LOCATION'}, ...]

# Code-switching segmentation
manglish_nlp.segment("aku nak buy groceries then balik")
# → {'segments': [...], 'switch_count': 2, 'dominant_lang': 'EN'}

# Formalize (informal → formal BM)
manglish_nlp.formalize("aku nk pegi kedai jap, ko nk ikut x?")
# → 'Saya ingin pergi kedai sebentar, anda ingin ikut tidak?'

# Clean noisy text
manglish_nlp.clean("besttttt gilerrrr ahhhh")
# → 'best giler ah'

# Spell correction
manglish_nlp.correct("aku nk pregi mkn")
# → {'corrected': 'aku nk pegi makan', 'changes': [...]}

# Keywords
manglish_nlp.extract_keywords("makanan sedap kat kedai tu harga murah")
# → [{'keyword': 'sedap', 'score': 1.0}, ...]

# Similarity
manglish_nlp.similarity.semantic_similarity("nk mkn nasi", "nak makan nasi")
# → {'score': 1.0, ...}

# Dictionary
manglish_nlp.is_malay("berlari")  # → True
manglish_nlp.is_english("computer")  # → True

# Augmentation
manglish_nlp.augmentation.kelantanese_form("barang")  # → ['bare']
manglish_nlp.augmentation.socialmedia_form("makan")  # → ['mkn', 'makannnn', ...]

# Advanced normalizer
manglish_nlp.normalize_all("besttt rm50 jumpa 28/5/26 pukul 3pm")
# → {'normalized': 'best RM50.00 jumpa 28 Mei 2026 pukul 3:00 PM', ...}
```

## API Reference

---

## Core Modules

### manglish_nlp.normalize(text, preserve_case=False)

Expand shortforms/slang to standard BM/EN form.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text with shortforms |
| preserve_case | bool | Preserve original casing (default: False) |

**Returns:** `str` — normalized text

**Dictionary:** 220+ shortform mappings covering SMS-speak, social media abbreviations, and common typos.

---

### manglish_nlp.detect_language(text)

Detect if text is BM, EN, or Manglish (code-switched).

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `dict` with keys:
- `language` (str): `'bm'`, `'en'`, `'manglish'`, or `'unknown'`
- `bm_ratio` (float): 0–1
- `en_ratio` (float): 0–1
- `manglish_markers` (int): Count of Manglish-specific particles
- `confidence` (float): 0–1
- `word_count` (int)

---

### manglish_nlp.sentiment(text)

Analyze sentiment with Malaysian slang awareness.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `dict` with keys:
- `sentiment` (str): `'positive'`, `'negative'`, or `'neutral'`
- `score` (float): Normalized -1.0 to 1.0
- `raw_score` (float): Unnormalized
- `positive_words` (list): Detected positive words
- `negative_words` (list): Detected negative words
- `context` (str): Scoring explanation

**Features:**
- 48 positive slang words (best, power, padu, mantap, syok...)
- 48 negative slang words (hampeh, teruk, sial, bodoh...)
- Intensifier handling (gila, sangat, memang → multiplier)
- Negation flipping (tak best → negative)

---

### manglish_nlp.tokenize(text)

Tokenize text into words. Handles particles, emoji, punctuation.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `list[str]` — tokens

**Also available:**
- `manglish_nlp.word_tokenize(text)` — same as tokenize
- `manglish_nlp.sentence_tokenize(text)` — split into sentences

---

### manglish_nlp.stem(text) / stem_word(word)

Rule-based Malay stemmer with nasal assimilation restoration.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text (stem) or single word (stem_word) |

**Returns:** `str` — stemmed text/word

**Handles:**
- me-/ber-/ter-/di-/ke-/se-/pe- prefixes
- -kan/-an/-i suffixes
- -lah/-kah/-nya/-pun particles
- Nasal assimilation: meny→s, meng→k, mem→p, men→t
- Known root validation (150+ roots)

---

### manglish_nlp.pos_tag(text)

Part-of-Speech tagging with Manglish-aware word lists.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `list[tuple]` — (word, tag) pairs

**Tags:**
| Tag | Meaning | Examples |
|-----|---------|----------|
| PRP | Pronoun | aku, ko, dia, dorang |
| VB | Verb | pergi, makan, buat |
| NN | Noun | rumah, kedai, orang |
| JJ | Adjective | cantik, best, power |
| MD | Modal | nak, boleh, akan |
| NEG | Negation | tak, x, bukan |
| PTL | Particle | la, weh, kan, je |
| INT | Intensifier | gila, sangat, memang |
| IN | Preposition | di, ke, dari, dengan |
| CC | Conjunction | dan, tapi, sebab |
| DT | Determiner | ini, itu, semua |
| NUM | Number | 1, 2, 100 |
| RB | Adverb | cepat, lambat |
| PUNCT | Punctuation | . , ! ? |
| UNK | Unknown | — |

---

### manglish_nlp.ner_tag(text)

Named Entity Recognition for Malaysian context.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `list[dict]` — entities with `text`, `type`, `start`, `end`

**Entity types:**
| Type | Description | Examples |
|------|-------------|----------|
| PERSON | Person names (via titles) | Encik Ahmad, Dato Seri |
| LOCATION | Malaysian states/cities | KL, Johor, Penang |
| ORGANIZATION | Companies/universities | UMPSA, Petronas, Grab |
| MONEY | Currency amounts | RM50, MYR 1,000 |
| PHONE | Malaysian phone numbers | 012-3456789 |
| EMAIL | Email addresses | user@mail.com |
| URL | Web URLs | https://... |
| DATE | Date expressions | 28/5/2026 |
| TIME | Time expressions | 3.30pm |

---

### manglish_nlp.segment(text)

Identify BM vs EN segments in code-switched text.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |

**Returns:** `dict` with keys:
- `segments` (list): `{'text', 'lang', 'word_count'}` per segment
- `switch_count` (int): Number of language switches
- `total_segments` (int)
- `dominant_lang` (str): `'BM'` or `'EN'`

---

### manglish_nlp.formalize(text)

Convert informal Manglish/BM to formal BM.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Informal text |

**Returns:** `str` — formal BM text (capitalized, punctuated)

---

### manglish_nlp.clean(text) / clean_for_nlp(text)

Clean noisy social media text.

| Function | Description |
|----------|-------------|
| `clean(text)` | Light cleaning: reduce repeats, normalize laughs, fix whitespace |
| `clean_for_nlp(text)` | Aggressive: also removes emoji, URLs, mentions, hashtags |

---

## Extended Modules

### manglish_nlp.correct(text) / correct_word(word)

Spell correction using edit distance + BM dictionary.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Text to correct |
| max_distance | int | Max edit distance (default: 1) |

**Returns (correct):** `dict` with `corrected`, `changes`, `original`

**Returns (correct_word):** `dict` with `original`, `corrected`, `is_valid`, `suggestions`

```python
>>> manglish_nlp.correct("aku nk pregi mkn")
{'corrected': 'aku nk pegi makan', 'changes': [{'original': 'pregi', 'corrected': 'pegi', 'distance': 1}]}

>>> manglish_nlp.correct_word("mkaan")
{'original': 'mkaan', 'corrected': 'makan', 'is_valid': False, 'suggestions': [...]}
```

---

### manglish_nlp.extract_keywords(text, top_n=10, method='frequency')

Extract keywords from text.

| Parameter | Type | Description |
|-----------|------|-------------|
| text | str | Input text |
| top_n | int | Number of keywords (default: 10) |
| method | str | `'frequency'`, `'rake'`, or `'tfidf'` |

**Returns:** `list[dict]` — keywords with `keyword` and `score`

```python
>>> manglish_nlp.extract_keywords("makanan sedap sangat", method='rake')
[{'keyword': 'makanan sedap', 'score': 2.0}, ...]
```

---

### manglish_nlp.similarity

Text similarity module with multiple algorithms.

| Function | Description |
|----------|-------------|
| `similarity.jaccard(t1, t2, use_stem=False)` | Jaccard index (set overlap) |
| `similarity.cosine(t1, t2, use_stem=False)` | Cosine similarity (bag-of-words) |
| `similarity.overlap(t1, t2, use_stem=False)` | Overlap coefficient |
| `similarity.semantic_similarity(t1, t2)` | Combined (normalize+stem+weighted) |
| `similarity.find_most_similar(query, candidates)` | Rank candidates by similarity |

```python
>>> manglish_nlp.similarity.jaccard("aku nak makan nasi", "aku nak makan roti")
0.6

>>> manglish_nlp.similarity.semantic_similarity("nk mkn nasi", "nak makan nasi")
{'score': 1.0, 'jaccard': 1.0, 'cosine': 1.0, 'overlap': 1.0}

>>> manglish_nlp.similarity.find_most_similar("nk mkn", ["nak makan", "nak pergi", "tidur"])
[{'text': 'nak makan', 'score': 0.95, 'index': 0}, ...]
```

---

### manglish_nlp.augmentation

Data augmentation for NLP pipelines.

| Function | Description |
|----------|-------------|
| `augmentation.socialmedia_form(word)` | Generate social media variants (mkn, MAKAN, makan2) |
| `augmentation.kelantanese_form(word)` | Convert to Kelantanese dialect |
| `augmentation.vowel_alternate(word)` | Remove vowels (SMS-style) |
| `augmentation.replace_similar_vowels(word)` | Swap vowels (simulate slang) |
| `augmentation.replace_similar_consonants(word)` | Swap consonants (simulate typo) |
| `augmentation.synonym(word)` | Get BM/Manglish synonyms |
| `augmentation.augment(text, methods, n)` | Generate n text variations |

```python
>>> manglish_nlp.augmentation.socialmedia_form("makan")
['mkn', 'makannnn', 'MAKAN', 'makan2', 'makanan']

>>> manglish_nlp.augmentation.kelantanese_form("barang")
['bare', 'barong']

>>> manglish_nlp.augmentation.synonym("cantik")
['lawa', 'cun', 'jelita']

>>> manglish_nlp.augmentation.augment("makanan sedap", n=3)
['mknn sedap', 'makanan sedapppp', 'makanan best']
```

---

### manglish_nlp.dictionary

Word validation and classification.

| Function | Description |
|----------|-------------|
| `is_malay(word)` | Check if word is BM (dict + morphology) |
| `is_english(word)` | Check if word is English |
| `classify_word(word)` | Classify as 'bm', 'en', 'both', or 'unknown' |
| `get_stopwords(lang)` | Get stop words ('bm', 'en', or 'all') |

```python
>>> manglish_nlp.is_malay("berlari")
True
>>> manglish_nlp.is_english("computer")
True
>>> manglish_nlp.classify_word("hospital")
{'word': 'hospital', 'classification': 'bm', 'is_malay': True, 'is_english': False}
```

---

### manglish_nlp.normalize_all(text) and friends

Advanced text normalization (Malaya-style).

| Function | Description |
|----------|-------------|
| `normalize_elongated(text)` | Reduce repeated chars (besttt→best) |
| `normalize_money(text)` | Standardize money (rm50→RM50.00) |
| `normalize_phone(text)` | Format phone numbers |
| `normalize_date(text)` | Convert dates (28/5/26→28 Mei 2026) |
| `normalize_time(text)` | Convert time (3pm→3:00 PM, 1430→2:30 PM) |
| `normalize_number(text)` | Numbers to BM words |
| `normalize_url(text)` | Simplify URLs to domain |
| `normalize_all(text, options)` | Apply all normalizations |

```python
>>> manglish_nlp.normalize_all("besttt rm50 jumpa 28/5/26 pukul 3pm")
{'normalized': 'best RM50.00 jumpa 28 Mei 2026 pukul 3:00 PM',
 'changes': ['elongated', 'money', 'date', 'time']}
```

---

## CLI Usage

```bash
# Batch processing
python scripts/batch.py --input texts.txt --output results.json --tasks normalize,sentiment,lang

# CSV output
python scripts/batch.py -i data.txt -o out.csv -f csv -t normalize,lang,sentiment

# Pipe from stdin
echo "best giler makanan dia" | python scripts/batch.py -t sentiment --format jsonl
```

## Package Structure

```
manglish-nlp/
├── manglish_nlp/           # Main package (pip installable)
│   ├── __init__.py         # Public API (16 modules exported)
│   ├── normalize.py        # Shortform expansion
│   ├── language.py         # Language detection
│   ├── sentiment.py        # Sentiment analysis
│   ├── clean.py            # Text cleaning
│   ├── formalize.py        # Informal → formal BM
│   ├── tokenizer.py        # Word/sentence/morpheme tokenization
│   ├── stemmer.py          # Malay stemmer
│   ├── segment.py          # Code-switching segmenter
│   ├── pos.py              # POS tagger
│   ├── ner.py              # Named Entity Recognition
│   ├── spelling.py         # Spell correction
│   ├── keywords.py         # Keyword extraction
│   ├── similarity.py       # Text similarity
│   ├── augmentation.py     # Data augmentation
│   ├── dictionary.py       # Word validation
│   ├── normalizer.py       # Advanced normalization
│   ├── utils.py            # Shared utilities
│   ├── resources/
│   │   └── dictionary.json # 220+ shortforms, slang lexicon
│   └── transformers/       # Optional transformer models
│       ├── __init__.py
│       ├── base.py         # Base model classes
│       ├── sentiment.py    # RoBERTa sentiment (BM/Manglish)
│       ├── translation.py  # T5 translation (BM<->EN)
│       ├── summarization.py# T5 summarization
│       ├── classification.py# Emotion/topic classification
│       ├── ner.py          # BERT NER
│       └── pos.py          # BERT POS tagger
├── scripts/                # CLI tools + batch processor
├── tests/
│   └── test_all.py         # 121 unit tests
├── venv-gpu/               # Python 3.12 + CUDA venv (optional)
├── setup.py                # pip install support
├── README.md
└── SKILL.md                # This file
```

## Comparison with Malaya

| Feature | manglish-nlp | Malaya |
|---------|-------------|--------|
| Focus | Informal Manglish/slang | Formal BM |
| Dependencies | Zero (stdlib only) | PyTorch, transformers |
| Install size | ~80KB | 500MB+ |
| Shortform handling | 220+ mappings | Limited |
| Slang sentiment | 96 slang words | Model-based |
| Code-switching | Built-in segmenter | Not focused |
| Particles (la, weh) | First-class support | Treated as noise |
| Augmentation | Rule-based (kelantan, SMS, typo) | Rule + model-based |
| Dictionary | is_malay, is_english, stopwords | is_malay, is_english, DBP lookup |
| Normalizer | Money, phone, date, time, URL | Money, phone, date, time, URL, IC |
| Spell correction | Edit distance + dictionary | Model-based |
| Keywords | Frequency, RAKE, TF-IDF | RAKE, TextRank, Attention, model |
| Similarity | Jaccard, cosine, overlap, semantic | Model-based embeddings |
| Stemmer | Rule-based + known roots | Rule-based + model |
| POS Tagger | Rule-based (15 tags) | Model-based (universal tags) |
| NER | Gazetteer-based (9 types) | Model-based |
| Translation | T5 model (mesolitica) | T5 model-based |
| Summarization | T5 model (optional) | T5 model-based |
| Offline | Always | Always |
| Speed | Instant (rule-based) | Model inference |
| Accuracy | Good for informal text | Better for formal text |

**Use manglish-nlp when:** Processing social media, chat messages, informal Malaysian text, or when you need zero-dependency lightweight NLP.

**Use Malaya when:** Processing formal documents, news articles, or need transformer-level accuracy for translation/summarization.

## Transformer Models (Optional)

For higher accuracy, manglish-nlp includes optional transformer-based models powered by HuggingFace.

**Requirements:** `pip install torch transformers sentencepiece` (or use GPU venv)

**GPU Setup (recommended):**
```bash
# Python 3.12 + CUDA venv (RTX 2070+ recommended)
python3.12 -m venv venv-gpu
venv-gpu/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124
venv-gpu/Scripts/pip install transformers sentencepiece
```

### Sentiment (BM/Manglish)

```python
from manglish_nlp.transformers import sentiment_model

model = sentiment_model()  # default: indonesian-roberta (understands BM+Manglish)
model.predict("gila best la food dia")
# → {'label': 'positive', 'score': 0.973}
model.predict("teruk gila service")
# → {'label': 'negative', 'score': 0.995}
model.predict("weh sedap gila bro the nasi lemak here")
# → {'label': 'positive', 'score': 0.964}

# Batch
model.predict_batch(["best gila", "hampeh la", "ok je"])
```

**Default model:** `w11wo/indonesian-roberta-base-sentiment-classifier`
- Trained on Indonesian/Malay data — understands BM slang + Manglish mixed
- 10/10 correct on Manglish test set (vs distilbert EN-only: 7/10)
- 141K+ downloads, well-maintained

**Available models:**
| Model | Description | Best for |
|-------|-------------|----------|
| w11wo/indonesian-roberta-base-sentiment-classifier | RoBERTa ID/BM (default) | Manglish, BM |
| distilbert/distilbert-base-uncased-finetuned-sst-2-english | DistilBERT EN | English-heavy text |
| cardiffnlp/twitter-roberta-base-sentiment-latest | Twitter RoBERTa | Social media EN |

---

### Translation (BM <-> EN)

```python
from manglish_nlp.transformers import translation_model

model = translation_model()  # default: mesolitica T5-tiny

# BM/Manglish → English
model.translate("saya suka makan nasi goreng", target='en')
# → 'I like to eat nasi goreng'
model.translate("aku nak go buy groceries then balik rumah", target='en')
# → 'I want to go and buy groceries, then go home'
model.translate("cuaca hari ini sangat panas", target='en')
# → 'The weather today is very hot'

# English → BM
model.translate("I want to go home", target='bm')
# → 'Saya mahu pulang'
```

**Available models:**
| Model | Size | Quality |
|-------|------|--------|
| mesolitica/translation-t5-tiny-standard-bahasa-cased | ~60M | Good (default) |
| mesolitica/translation-t5-small-standard-bahasa-cased | ~220M | Better |
| mesolitica/translation-t5-base-standard-bahasa-cased | ~580M | Best |

---

### NER (Named Entity Recognition)

```python
from manglish_nlp.transformers import ner_model

model = ner_model()
model.predict("Ahmad tinggal di Kuala Lumpur")
# → [{'entity': 'PER', 'word': 'Ahmad', 'score': 0.99}, ...]
```

---

### Text Classification (Emotion Detection)

```python
from manglish_nlp.transformers import text_classification_model

model = text_classification_model()  # emotion detection
model.predict("aku gembira sangat hari ni")
# → {'label': 'joy', 'score': 0.89}
```

---

### Summarization

```python
from manglish_nlp.transformers import summarization_model

model = summarization_model()
model.summarize("Kerajaan Malaysia hari ini mengumumkan...")
```

---

### POS Tagging (Transformer)

```python
from manglish_nlp.transformers import pos_model

model = pos_model()
model.predict("saya makan nasi goreng")
# → [{'word': 'saya', 'tag': 'PRON', 'score': 0.99}, ...]
```

---

### Rule-based vs Transformer: When to use which?

| Scenario | Use | Why |
|----------|-----|-----|
| Chat preprocessing | Rule-based | Instant, no GPU needed |
| Batch sentiment analysis | Transformer | Higher accuracy |
| Real-time chatbot | Rule-based | Low latency |
| Research/analytics | Transformer | Better precision |
| Shortform expansion | Rule-based | Transformer can't do this |
| Translation | Transformer | Only option |
| Edge/mobile deployment | Rule-based | Zero dependencies |

---

## Limitations

- Rule-based modules: accuracy depends on dictionary coverage
- Stemmer handles common patterns but may fail on rare/complex words
- NER relies on gazetteers — won't catch unknown entities
- POS tagger uses word lists — ambiguous words default to most common tag
- Sentiment lexicon covers common slang but not exhaustive
- Transformer models need GPU for fast inference (CPU works but slower)
- Translation model struggles with heavy slang ("best gila" kept as-is)
- No dependency parsing, syntax trees, or text generation
- Spell correction limited to edit distance 1-2 (conservative)

## Integration Patterns

### With chatbots (preprocessing)
```python
import manglish_nlp

def preprocess(user_msg):
    cleaned = manglish_nlp.clean(user_msg)
    normalized = manglish_nlp.normalize(cleaned)
    lang = manglish_nlp.detect_language(cleaned)
    entities = manglish_nlp.ner_tag(cleaned)
    return {'text': normalized, 'lang': lang['language'], 'entities': entities}
```

### With sentiment dashboards
```python
import manglish_nlp

def analyze_feedback(texts):
    results = []
    for text in texts:
        s = manglish_nlp.sentiment(text)
        results.append({
            'text': text,
            'sentiment': s['sentiment'],
            'score': s['score'],
            'lang': manglish_nlp.detect_language(text)['language']
        })
    return results
```

### With search engines (query expansion)
```python
import manglish_nlp

def expand_query(query):
    normalized = manglish_nlp.normalize(query)
    tokens = manglish_nlp.tokenize(normalized)
    stems = [manglish_nlp.stem_word(t) for t in tokens]
    return list(set(tokens + stems))
```

### With data augmentation (ML training)
```python
import manglish_nlp

def augment_dataset(texts, n_per_text=5):
    augmented = []
    for text in texts:
        augmented.append(text)  # Original
        variants = manglish_nlp.augmentation.augment(text, n=n_per_text)
        augmented.extend(variants)
    return augmented
```

### With FAQ matching
```python
import manglish_nlp

faq_questions = ["macam mana nak bayar yuran", "bila tarikh tutup pendaftaran", ...]

def find_answer(user_query):
    results = manglish_nlp.similarity.find_most_similar(user_query, faq_questions, top_n=3)
    if results[0]['score'] > 0.6:
        return faq_answers[results[0]['index']]
    return "Maaf, soalan tidak dijumpai."
```

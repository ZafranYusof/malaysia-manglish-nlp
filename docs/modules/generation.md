# Generation

**Translate, summarise, generate, and answer questions in Malaysian text.**

---

## Overview

Generation modules produce or transform text: translation between languages, summarisation of long documents, controlled text generation, and question answering. All require the `[ml]` extra for transformer-backed models.

```bash
pip install manglish-nlp[ml]
```

```python
import manglish_nlp as mnlp
```

!!! warning "ML Dependency Required"
    All generation modules require transformer models. First call downloads the model (~200 MB). Subsequent calls use cached weights.

---

## Quick Start

```python
import manglish_nlp as mnlp

# Translate Manglish → English
mnlp.translate("Weh best gila movie tu bro", target="en")
# "Hey, that movie was really great, bro"

# Summarise
article = "Kerajaan umum pakej RM50B... (long text)"
mnlp.summarize(article, max_length=30)
# "Kerajaan umum pakej rangsangan RM50B merangkumi bantuan tunai, moratorium, dan subsidi upah."

# Question answering
mnlp.qa("Bila UMP ditubuhkan?", context="UMP ditubuhkan pada tahun 2002 di Gambang.")
# {'answer': '2002', 'confidence': 0.95}
```

---

## Module Details

### `translation`

Translate between Bahasa Melayu, English, and Manglish. Supports entity preservation and register-aware output.

```python
import manglish_nlp as mnlp

# BM → English
mnlp.translate("Aku nak pergi makan nasi lemak", target="en")
# "I want to go eat nasi lemak"

# English → BM
mnlp.translate("The weather is really nice today", target="ms")
# "Cuaca hari ini sangat cantik"

# Manglish → formal BM
mnlp.translate("Weh best gila movie tu bro", target="ms_formal")
# "Filem itu sangat bagus"

# Formal → Manglish (natural Malaysian style)
mnlp.translate("Filem itu sangat bagus", target="manglish")
# "Movie tu memang best gila"
```

#### Translation Directions

| From \ To | `en` | `ms` | `ms_formal` | `manglish` |
|-----------|------|------|-------------|------------|
| BM | ✅ | — | ✅ | ✅ |
| English | — | ✅ | ✅ | ✅ |
| Manglish | ✅ | ✅ | ✅ | — |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str \| list[str]` | *required* | Input text or batch |
| `target` | `str` | *required* | Target language code |
| `preserve_entities` | `bool` | `True` | Keep names and places untranslated |
| `informal` | `bool` | `False` | Use informal register in output |
| `alternatives` | `int` | `1` | Number of translation variants to return |

!!! example "Alternative Translations"
    ```python
    mnlp.translate("Cuaca cantik hari ini", target="en", alternatives=3)
    # ['The weather is beautiful today',
    #  'It\'s a lovely day today',
    #  'The weather is nice today']
    ```

!!! tip "Manglish Preservation"
    `target="manglish"` produces text that sounds like natural Malaysian speech — not word-for-word translation. Useful for chatbot responses targeting Malaysian users.

---

### `summarization`

Summarise Malaysian text while preserving key information. Supports extractive (select key sentences) and abstractive (generate new summary) methods.

```python
import manglish_nlp as mnlp

article = """
Kerajaan Malaysia hari ini mengumumkan pakej rangsangan ekonomi bernilai
RM50 bilion untuk membantu rakyat dan perniagaan kecil yang terjejas.
Perdana Menteri berkata pakej ini merangkumi bantuan tunai langsung,
moratorium pinjaman, dan subsidi upah untuk pekerja. Beliau juga
mengumumkan pengurangan cukai untuk PKS selama 6 bulan.
"""

mnlp.summarize(article)
# "Kerajaan umum pakej rangsangan RM50B — bantuan tunai, moratorium,
#  subsidi upah, dan pengurangan cukai PKS 6 bulan."
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input document |
| `max_length` | `int` | `None` | Target word count for summary |
| `ratio` | `float` | `0.3` | Summary length as fraction of original |
| `method` | `str` | `"abstractive"` | `"extractive"` or `"abstractive"` |
| `format` | `str` | `"text"` | Output format: `"text"` or `"bullets"` |
| `lang` | `str` | `None` | Force output language (cross-lingual summary) |

!!! example "Bullet Point Summary"
    ```python
    mnlp.summarize(article, format="bullets")
    # • Pakej rangsangan RM50B diumumkan
    # • Bantuan tunai langsung, moratorium pinjaman
    # • Subsidi upah untuk pekerja
    # • Pengurangan cukai PKS 6 bulan
    ```

!!! example "Cross-Lingual Summary"
    ```python
    mnlp.summarize(bm_article, lang="en")
    # "Government announces RM50B stimulus package — cash aid, loan moratorium,
    #  wage subsidies, and 6-month SME tax cuts."
    ```

!!! tip "Extractive vs Abstractive"
    - **Extractive**: faster, picks exact sentences from source — good for factual accuracy
    - **Abstractive**: slower, generates new sentences — more concise and readable

---

### `text_generation`

Generate Malaysian text with controllable style, format, and creativity level.

```python
import manglish_nlp as mnlp

mnlp.generate("Tulis review restoran nasi lemak", max_length=100)
# "Nasi lemak kat kedai ni memang power. Sambal dia pedas just nice,
#  ikan bilis rangup, dan nasi tu wangi gila. Portion pun besar.
#  Confirm balik lagi next time."
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Generation prompt or seed text |
| `max_length` | `int` | `100` | Maximum output tokens |
| `style` | `str` | `"manglish"` | Output style: `"formal"`, `"manglish"`, `"mixed"` |
| `temperature` | `float` | `0.7` | Creativity (0.1 = deterministic, 1.0 = creative) |
| `format` | `str` | `"text"` | Output format: `"text"`, `"tweet"`, `"review"`, `"caption"` |
| `mode` | `str` | `"generate"` | `"generate"` or `"continue"` (extend existing text) |

!!! example "Temperature Comparison"
    ```python
    # Low temperature — predictable, focused
    mnlp.generate("Nasi lemak is", temperature=0.2, max_length=20)
    # "Nasi lemak is a traditional Malaysian dish made with coconut rice."

    # High temperature — creative, varied
    mnlp.generate("Nasi lemak is", temperature=0.9, max_length=20)
    # "Nasi lemak is basically Malaysia's hug on a plate, no?"
    ```

!!! example "Continuation Mode"
    ```python
    mnlp.generate("Hari ni aku pergi kedai mamak, order", mode="continue", max_length=30)
    # "...teh tarik satu, roti canai dua. Pastu lepak sejam sambil scroll phone."
    ```

---

### `qa`

Question answering over Malaysian text. Supports extractive QA (find answer span in context), open-domain QA (no context), and conversational sessions with pronoun resolution.

```python
import manglish_nlp as mnlp

context = """
Universiti Malaysia Pahang (UMP) ditubuhkan pada tahun 2002.
Kampus utama terletak di Gambang, Pahang. UMP mempunyai lebih
10,000 pelajar dan menawarkan program dalam bidang kejuruteraan,
sains komputer, dan teknologi.
"""

mnlp.qa("Bila UMP ditubuhkan?", context=context)
# {'answer': '2002', 'confidence': 0.95, 'span': (46, 50)}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | *required* | Question in BM, EN, or Manglish |
| `context` | `str` | `None` | Source document (omit for open-domain) |
| `top_k` | `int` | `1` | Number of answer candidates |
| `session` | `bool` | `False` | Enable conversational mode with pronoun resolution |

!!! example "Cross-Lingual QA"
    ```python
    # Question in English, context in BM
    mnlp.qa("When was UMP established?", context=bm_context)
    # {'answer': '2002', 'confidence': 0.93}
    ```

!!! example "Multi-Answer Extraction"
    ```python
    mnlp.qa("Apa program yang ditawarkan UMP?", context=context, top_k=3)
    # [{'answer': 'kejuruteraan', 'score': 0.89},
    #  {'answer': 'sains komputer', 'score': 0.85},
    #  {'answer': 'teknologi', 'score': 0.78}]
    ```

!!! example "Conversational Session"
    ```python
    session = mnlp.qa.session(context=context)
    session.ask("Bila UMP ditubuhkan?")   # {'answer': '2002', ...}
    session.ask("Kat mana?")              # {'answer': 'Gambang, Pahang', ...}
    session.ask("Berapa pelajar?")        # {'answer': 'lebih 10,000', ...}
    ```

!!! tip "Language Handling"
    Questions can be in BM, English, or Manglish regardless of context language. The model handles cross-lingual QA natively — no translation step needed.

---

## See Also

- [Text Processing](text-processing.md) — preprocess text before translation or QA
- [Embeddings](data.md#embeddings) — use sentence embeddings for retrieval-augmented QA
- [Pipeline](tools.md#pipeline) — chain QA with document retrieval
- [Cache](tools.md#cache) — cache generation results for repeated prompts

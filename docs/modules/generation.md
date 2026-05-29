# Generation

Modules for generating, translating, and transforming Malaysian text.

!!! warning "ML Dependency"
    Generation modules require the ML extra: `pip install manglish-nlp[ml]`

---

## text_generation

Generate Malaysian text with controllable style and language mix.

```python
import manglish_nlp as mnlp

result = mnlp.generate("Tulis review restoran nasi lemak", max_length=100)
print(result)
# "Nasi lemak kat kedai ni memang power. Sambal dia pedas just nice,
#  ikan bilis rangup, dan nasi tu wangi gila. Portion pun besar.
#  Confirm balik lagi next time."
```

### Options

```python
# Control language style
mnlp.generate(prompt, style="formal")    # Formal BM
mnlp.generate(prompt, style="manglish")  # Casual Manglish
mnlp.generate(prompt, style="mixed")     # Code-switched

# Control creativity
mnlp.generate(prompt, temperature=0.7)

# Specific format
mnlp.generate(prompt, format="tweet")     # Short, punchy
mnlp.generate(prompt, format="review")    # Structured review
mnlp.generate(prompt, format="caption")   # Social media caption

# Continue from text
mnlp.generate("Hari ni aku pergi...", mode="continue", max_length=50)
```

---

## translation

Translate between Bahasa Melayu, English, and Manglish.

```python
# BM to English
result = mnlp.translate("Aku nak pergi makan", target="en")
print(result)
# "I want to go eat"

# English to BM
result = mnlp.translate("The weather is nice today", target="ms")
print(result)
# "Cuaca hari ini cantik"

# Manglish to formal BM
result = mnlp.translate("Weh best gila movie tu bro", target="ms_formal")
print(result)
# "Filem itu sangat bagus"
```

### Options

```python
# Preserve names and entities
mnlp.translate(text, target="en", preserve_entities=True)

# Informal translation (keep the vibe)
mnlp.translate("The food was amazing", target="ms", informal=True)
# "Makanan dia memang terbaik"

# Batch translation
mnlp.translate(["text1", "text2"], target="en")

# With alternatives
mnlp.translate(text, target="en", alternatives=3)
# ['Translation 1', 'Translation 2', 'Translation 3']
```

!!! tip "Manglish Preservation"
    Use `target="manglish"` to translate formal text into natural Manglish that sounds like how Malaysians actually speak.

---

## summarization

Summarize Malaysian text while preserving key information.

```python
article = """
Kerajaan Malaysia hari ini mengumumkan pakej rangsangan ekonomi bernilai
RM50 bilion untuk membantu rakyat dan perniagaan kecil yang terjejas.
Perdana Menteri berkata pakej ini merangkumi bantuan tunai langsung,
moratorium pinjaman, dan subsidi upah untuk pekerja. Beliau juga
mengumumkan pengurangan cukai untuk PKS selama 6 bulan.
"""

summary = mnlp.summarize(article)
print(summary)
# "Kerajaan umum pakej rangsangan RM50B — bantuan tunai, moratorium,
#  subsidi upah, dan pengurangan cukai PKS 6 bulan."
```

### Options

```python
# Control length
mnlp.summarize(text, max_length=50)   # ~50 words
mnlp.summarize(text, ratio=0.3)       # 30% of original

# Extractive vs abstractive
mnlp.summarize(text, method="extractive")   # Pick key sentences
mnlp.summarize(text, method="abstractive")  # Generate new summary

# Bullet points
mnlp.summarize(text, format="bullets")
# • Pakej rangsangan RM50B diumumkan
# • Bantuan tunai, moratorium, subsidi upah
# • Pengurangan cukai PKS 6 bulan

# Target language for summary
mnlp.summarize(text, lang="en")
# "Government announces RM50B stimulus — cash aid, loan moratorium,
#  wage subsidies, and 6-month SME tax cuts."
```

---

## qa

Question answering over Malaysian text — extractive and generative.

```python
context = """
Universiti Malaysia Pahang (UMP) ditubuhkan pada tahun 2002.
Kampus utama terletak di Gambang, Pahang. UMP mempunyai lebih
10,000 pelajar dan menawarkan program dalam bidang kejuruteraan,
sains komputer, dan teknologi.
"""

answer = mnlp.qa("Bila UMP ditubuhkan?", context=context)
print(answer)
# {'answer': '2002', 'confidence': 0.95, 'span': (46, 50)}
```

### Options

```python
# Without context (open-domain, uses knowledge base)
mnlp.qa("Siapa PM Malaysia pertama?")
# {'answer': 'Tunku Abdul Rahman', 'confidence': 0.92}

# Multiple answers
mnlp.qa("Apa program yang ditawarkan UMP?", context=context, top_k=3)
# [{'answer': 'kejuruteraan', 'score': 0.89},
#  {'answer': 'sains komputer', 'score': 0.85},
#  {'answer': 'teknologi', 'score': 0.78}]

# Yes/No questions
mnlp.qa("UMP ada kat Pahang ke?", context=context)
# {'answer': 'Ya', 'confidence': 0.97, 'evidence': 'Kampus utama terletak di Gambang, Pahang'}

# Conversational QA (maintains context)
session = mnlp.qa.session(context=context)
session.ask("Bila UMP ditubuhkan?")   # 2002
session.ask("Kat mana?")              # Gambang, Pahang (resolves "kat mana" from context)
```

!!! info "Language Handling"
    Questions can be asked in BM, English, or Manglish regardless of the context language. The model handles cross-lingual QA natively.

---

## See Also

- [Text Processing](text-processing.md) — preprocess before generation
- [Data & Embeddings](data.md) — word embeddings for similarity
- [Tools](tools.md) — pipeline and caching for generation workflows

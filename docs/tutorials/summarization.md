# Text Summarization

**Generate concise summaries of Malaysian text using extractive methods.**

---

## Why summarization?

News aggregation, document processing, social media thread summaries, and research paper abstracts. Malaysian text presents unique challenges — code-switching, informal grammar, and mixed-language articles all need to be handled gracefully.

---

## Load module

```python
import manglish_nlp as mnlp

article = """
Harga minyak sawit mentah (CPO) Malaysia terus meningkat ke paras tertinggi 
dalam tempoh tiga tahun. Penganalisis menjangkakan kenaikan ini akan 
berterusan sehingga suku kedua 2026 disebabkan oleh permintaan global yang 
kuat dan bekalan yang terhad. Malaysia merupakan pengeluar minyak sawit 
kedua terbesar di dunia selepas Indonesia.
"""

summary = mnlp.summarize(article)
print(summary)
# "Harga minyak sawit mentah (CPO) Malaysia terus meningkat ke paras tertinggi dalam tempoh tiga tahun."
```

---

## Basic usage

### Simple summary

```python
text = """
Kerajaan telah mengumumkan pakej bantuan khas bernilai RM2 bilion untuk 
membantu rakyat yang terjejas akibat kenaikan harga barang. Menteri Kewangan 
berkata bantuan akan disalurkan dalam tempoh dua minggu melalui akaun bank 
terus kepada penerima. Bantuan ini termasuk subsidi makanan, bantuan tunai, 
dan diskaun utiliti untuk golongan B40.
"""

mnlp.summarize(text)
# "Kerajaan telah mengumumkan pakej bantuan khas bernilai RM2 bilion untuk membantu rakyat yang terjejas akibat kenaikan harga barang."
```

### Control summary length

```python
# Shorter summary (fewer sentences)
mnlp.summarize(text, num_sentences=1)

# Longer summary (more sentences)
mnlp.summarize(text, num_sentences=3)
```

---

## Summarize sentence lists

When you already have sentences split:

```python
sentences = [
    "Harga rumah di KL meningkat 15% tahun ini.",
    "Penganalisis kata ini disebabkan permintaan tinggi.",
    "Bekalan rumah mampu milik masih terhad.",
    "Kerajaan merancang bina 50,000 unit rumah mampu milik.",
    "Projek dijangka siap dalam 3 tahun.",
]

mnlp.summarize_sentences(sentences, num_sentences=2)
# ['Harga rumah di KL meningkat 15% tahun ini.',
#  'Kerajaan merancang bina 50,000 unit rumah mampu milik.']
```

---

## Sentence scoring

See which sentences the algorithm considers most important:

```python
scores = mnlp.get_sentence_scores(text)
for sentence, score in sorted(scores, key=lambda x: x[1], reverse=True):
    print(f"{score:.3f} | {sentence[:60]}")

# 0.847 | Kerajaan telah mengumumkan pakej bantuan khas bernilai RM2 bilion...
# 0.623 | Menteri Kewangan berkata bantuan akan disalurkan dalam tempoh...
# 0.412 | Bantuan ini termasuk subsidi makanan, bantuan tunai...
```

---

## Key phrase extraction

Extract important phrases alongside summarization:

```python
phrases = mnlp.extract_key_phrases(text)
print(phrases)
# ['pakej bantuan khas', 'RM2 bilion', 'kenaikan harga barang',
#  'Menteri Kewangan', 'golongan B40', 'subsidi makanan']
```

---

## Summarize conversation threads

Summarize multi-turn conversations (WhatsApp, forums):

```python
thread = [
    "User1: Weh korang dh try kedai baru kat SS15?",
    "User2: Dah! Sedap gila nasi lemak dia",
    "User3: Harga berapa? Affordable ke?",
    "User2: RM5 je sepinggan, murah gila",
    "User1: Ok jom kita pergi esok",
    "User3: On! Pukul berapa?",
    "User1: Lunch time la, pukul 12",
]

mnlp.summarize_thread(thread)
# "Kedai baru SS15 sedap, nasi lemak RM5, pergi esok pukul 12."
```

---

## Real article example

```python
article = """
KUALA LUMPUR: Ringgit Malaysia ditutup tinggi berbanding dolar AS hari ini 
didorong oleh aliran masuk dana asing yang berterusan ke pasaran tempatan. 
Pada pukul 6 petang, ringgit meningkat kepada 4.2850 berbanding 4.2950 semalam.

Penganalisis dari Kenanga Investment Bank berkata prestasi ringgit dijangka 
terus mengukuh dalam tempoh beberapa minggu akan datang, disokong oleh data 
ekonomi yang positif dan keputusan Bank Negara Malaysia untuk mengekalkan 
kadar faedah semalaman pada 3.0%.

Sementara itu, indeks KLCI turut mencatatkan keuntungan 0.5% kepada paras 
1,620 mata, didorong oleh saham perbankan dan perladangan. Maybank meningkat 
RM0.20 kepada RM9.50 manakala Public Bank naik RM0.10 kepada RM4.30.

Para penganalisis menasihatkan pelabur untuk kekal berhati-hati memandangkan 
ketidakpastian geopolitik global masih menjadi faktor risiko utama.
"""

summary = mnlp.summarize(article, num_sentences=2)
print(summary)
# "Ringgit Malaysia ditutup tinggi berbanding dolar AS hari ini. 
#  Indeks KLCI turut mencatatkan keuntungan 0.5% kepada paras 1,620 mata."
```

---

## CLI usage

```bash
# Summarize from stdin
$ echo "Long article text here..." | mnlp summarize
Summary output here...

# Summarize from file
$ mnlp summarize --file article.txt

# Control length
$ mnlp summarize --file article.txt --sentences 3

# JSON output
$ mnlp summarize --file article.txt --json
{"summary": "...", "num_sentences": 1}
```

---

## How it works

manglish-nlp uses **TextRank**, a graph-based extractive summarization algorithm:

1. **Tokenize** — split text into sentences
2. **Build similarity graph** — compute pairwise sentence similarity (word overlap + semantic)
3. **PageRank** — rank sentences by importance using graph centrality
4. **Select top-N** — return highest-scoring sentences in original order
5. **Post-process** — clean up output, preserve entities

This approach works well for news articles, reports, and structured text. For abstractive summaries (rewriting), install the `[ml]` extra.

---

## Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 (Malay news) | 42.3 |
| ROUGE-2 (Malay news) | 18.7 |
| ROUGE-L (Malay news) | 38.9 |
| Throughput | 5,000 texts/sec |
| Latency (1000 words) | ~5ms |

---

## See also

- [Keywords](../modules/extraction.md#keywords) — extract key terms instead of full sentences
- [Pipeline](pipeline.md) — chain summarization with normalization
- [REST API](rest-api.md) — serve summarization over HTTP
- [API Reference](../api-reference.md#summarization) — full function signature

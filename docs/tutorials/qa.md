# Question Answering

**Extract answers from Malaysian text passages using context-based retrieval.**

---

## Why question answering?

Customer support automation, knowledge base search, document Q&A, and chatbot intelligence. Malaysian businesses need QA systems that understand BM, Manglish, and mixed-language documents.

---

## Load module

```python
import manglish_nlp as mnlp

context = """
Petronas Twin Towers terletak di Kuala Lumpur, Malaysia. Menara berkembar ini 
setinggi 451.9 meter dan mempunyai 88 tingkat. Ia siap dibina pada tahun 1998 
dan merupakan bangunan tertinggi di dunia sehingga tahun 2004. Arkitek yang 
mereka bentuk menara ini ialah Cesar Pelli dari Argentina.
"""

answer = mnlp.qa_answer("Berapa tinggi Petronas Towers?", context)
print(answer)
# {'answer': '451.9 meter', 'confidence': 0.89}
```

---

## Basic usage

### Simple extraction

```python
context = """
Nasi lemak ialah hidangan kebangsaan Malaysia. Ia terdiri daripada nasi yang 
dimasak dengan santan dan daun pandan. Biasanya dihidangkan dengan sambal, 
ikan bilis, kacang tanah, timun, dan telur. Harga sepinggan nasi lemak 
bermula dari RM1.50 di gerai tepi jalan sehingga RM25 di restoran mewah.
"""

mnlp.qa_answer("Apa itu nasi lemak?", context)
# {'answer': 'hidangan kebangsaan Malaysia', 'confidence': 0.87}

mnlp.qa_answer("Berapa harga nasi lemak?", context)
# {'answer': 'RM1.50 sehingga RM25', 'confidence': 0.82}

mnlp.qa_answer("Apa bahan dalam nasi lemak?", context)
# {'answer': 'santan dan daun pandan', 'confidence': 0.79}
```

### Question type classification

```python
mnlp.classify_question_type("Berapa tinggi menara itu?")
# 'quantity'

mnlp.classify_question_type("Siapa arkitek Petronas Towers?")
# 'person'

mnlp.classify_question_type("Bila menara ini siap dibina?")
# 'time'

mnlp.classify_question_type("Di mana Petronas Towers?")
# 'location'
```

---

## Multiple answers

Extract multiple answers from a passage:

```python
context = """
Malaysia mempunyai 14 negeri. Negeri terbesar ialah Sarawak dengan keluasan 
124,450 km persegi. Negeri terkecil ialah Perlis dengan 810 km persegi. 
Negeri paling ramai penduduk ialah Selangor dengan 7 juta orang. Negeri 
paling sedikit penduduk ialah Labuan dengan 100,000 orang.
"""

mnlp.qa_answer_multiple("Negeri mana yang terbesar?", context, topk=3)
# [
#   {'answer': 'Sarawak', 'confidence': 0.91},
#   {'answer': '124,450 km persegi', 'confidence': 0.72},
#   {'answer': 'Negeri terbesar', 'confidence': 0.45},
# ]
```

---

## Find relevant sentences

Locate the most relevant sentence for a question:

```python
sentences = [
    "Petronas Twin Towers terletak di KLCC.",
    "Menara ini siap pada tahun 1998.",
    "Tinggi menara ini 451.9 meter.",
    "Arkitek ialah Cesar Pelli.",
]

mnlp.find_relevant_sentence("Berapa tinggi?", sentences)
# (2, "Tinggi menara ini 451.9 meter.", 0.91)
```

---

## Extract answer span

Find the exact span within a sentence:

```python
sentence = "Menara ini mempunyai tinggi 451.9 meter dan 88 tingkat."
mnlp.extract_answer_span("berapa tinggi", sentence)
# {'span': '451.9 meter', 'start': 27, 'end': 38, 'confidence': 0.89}
```

---

## Malaysian context examples

### Government / policy

```python
context = """
Bantuan Prihatin Rakyat (BPR) ialah program bantuan tunai kerajaan Malaysia. 
Penerima bujang mendapat RM350 setahun manakala isi rumah dengan pendapatan 
bawah RM2,500 mendapat RM1,000 setahun. Permohonan dibuka setiap Januari 
melalui portal LHDN.
"""

mnlp.qa_answer("Berapa bantuan untuk bujang?", context)
# {'answer': 'RM350 setahun', 'confidence': 0.88}

mnlp.qa_answer("Bila permohonan dibuka?", context)
# {'answer': 'setiap Januari', 'confidence': 0.85}
```

### Informal text (Manglish)

```python
context = """
Kedai nasi lemak Mak Cik Zainab kat SS15 Subang tu memang famous gila. 
Buka dari pukul 6 pagi sampai 2 petang. Harga nasi lemak biasa RM3, 
nasi lemak special dengan ayam rendang RM8. Parking agak susah peak hour.
"""

mnlp.qa_answer("What time the shop opens?", context)
# {'answer': 'pukul 6 pagi sampai 2 petang', 'confidence': 0.84}

mnlp.qa_answer("How much for special?", context)
# {'answer': 'RM8', 'confidence': 0.90}
```

---

## CLI usage

QA is primarily a Python API. For CLI usage:

```bash
# Analyze (includes keyword extraction for QA-like features)
$ mnlp analyze "Petronas Tower 451.9 meter tinggi"

# Keywords from context
$ mnlp keywords "Harga minyak sawit meningkat ke paras tertinggi"
```

---

## How it works

1. **Sentence retrieval** — TF-IDF to find the most relevant sentence for the question
2. **Span extraction** — locate the answer within the relevant sentence using pattern matching and POS features
3. **Question classification** — categorize question type (person, location, quantity, time, etc.) to guide extraction
4. **Confidence scoring** — estimate answer reliability based on match quality

For generative QA (free-form answers), install `[ml]`:
```bash
pip install manglish-nlp[ml]
```

---

## Performance

| Metric | Score |
|--------|-------|
| Exact Match (BM news) | 62.3% |
| F1 (BM news) | 71.8% |
| Exact Match (Manglish) | 54.7% |
| Retrieval accuracy | 78.4% |
| Throughput | 8,000 questions/sec |
| Latency (single) | ~2ms |

---

## See also

- [Summarization](summarization.md) — summarize context before QA
- [Keywords](../modules/extraction.md#keywords) — extract key terms for retrieval
- [Embeddings](embeddings.md) — semantic similarity for context matching
- [Pipeline](pipeline.md) — chain QA with normalization
- [API Reference](../api-reference.md#qa) — full function signature

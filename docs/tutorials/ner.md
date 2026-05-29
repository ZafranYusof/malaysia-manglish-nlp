# Named Entity Recognition

**Extract persons, organisations, locations, and Malaysian-specific entities from text.**

---

## Why NER?

Information extraction, knowledge graph construction, content tagging, and search indexing for Malaysian text. Standard NER models trained on news corpora miss Malaysian-specific entities like "Petronas", "UMNO", "nasi lemak", or informal place names like "KL", "Penang".

manglish-nlp recognises entities in Manglish, social media text, and informal writing.

---

## Load module

```python
import manglish_nlp as mnlp

entities = mnlp.ner_tag("Ahmad kerja kat Petronas Tower KL")
print(entities)
# [('Ahmad', 'PERSON'), ('Petronas Tower', 'ORG'), ('KL', 'LOCATION')]
```

---

## Basic usage

### Simple extraction

```python
mnlp.ner_tag("Ali pergi kedai Mamak dekat Bukit Bintang")
# [('Ali', 'PERSON'), ('Mamak', 'ORG'), ('Bukit Bintang', 'LOCATION')]

mnlp.ner_tag("Harga minyak RON95 naik RM0.20 minggu ni")
# [('RON95', 'PRODUCT'), ('RM0.20', 'MONEY')]

mnlp.ner_tag("PM Ismail Sabri melawat Sabah hari Isnin")
# [('Ismail Sabri', 'PERSON'), ('Sabah', 'LOCATION'), ('Isnin', 'DATE')]
```

### Raw tuple output

Each entity is a `(text, label)` tuple:

```python
entities = mnlp.ner_tag("Dr. Siti bekerja di Hospital Kuala Lumpur")
for text, label in entities:
    print(f"{text:25s} → {label}")

# Dr. Siti                  → PERSON
# Hospital Kuala Lumpur     → ORG
```

---

## Entity types

manglish-nlp recognises these entity categories:

| Type | Description | Examples |
|------|-------------|----------|
| `PERSON` | Names of people | Ahmad, Siti Nurhaliza, Dr. Mahathir |
| `ORG` | Organisations | Petronas, UMNO, Maybank, UMP |
| `LOCATION` | Places, addresses | KL, Penang, Bukit Bintang, Sabah |
| `DATE` | Dates, temporal | Isnin, 2026, esok, minggu depan |
| `TIME` | Time expressions | pukul 3, 5:30 petang, tengah malam |
| `MONEY` | Currency amounts | RM50, RM1,200, 50 sen |
| `PRODUCT` | Products, brands | RON95, iPhone, Myvi, nasi lemak |
| `EVENT` | Events | Hari Raya, Merdeka, PRU15 |
| `QUANTITY` | Measurements | 5 kg, 100 km, 3 liter |

### Examples for each type

```python
# PERSON
mnlp.ner_tag("Siti Nurhaliza nyanyi kat KLCC")
# [('Siti Nurhaliza', 'PERSON'), ('KLCC', 'LOCATION')]

# ORG
mnlp.ner_tag("Student UMP menang competition anjuran Google")
# [('UMP', 'ORG'), ('Google', 'ORG')]

# LOCATION
mnlp.ner_tag("Jom lepak mamak dekat SS15 Subang")
# [('SS15', 'LOCATION'), ('Subang', 'LOCATION')]

# DATE
mnlp.ner_tag("Deadline submission 15 Jun 2026")
# [('15 Jun 2026', 'DATE')]

# MONEY
mnlp.ner_tag("Nasi lemak RM1.50, teh tarik RM2")
# [('RM1.50', 'MONEY'), ('RM2', 'MONEY')]

# PRODUCT
mnlp.ner_tag("Myvi baru launch harga RM50k")
# [('Myvi', 'PRODUCT'), ('RM50k', 'MONEY')]

# EVENT
mnlp.ner_tag("Hari Raya tahun ni balik kampung Johor")
# [('Hari Raya', 'EVENT'), ('Johor', 'LOCATION')]

# QUANTITY
mnlp.ner_tag("Beli 2 kg ayam dengan 5 liter minyak")
# [('2 kg', 'QUANTITY'), ('5 liter', 'QUANTITY')]
```

---

## Handling Manglish text

NER works on informal text without preprocessing:

```python
# Informal Manglish
mnlp.ner_tag("weh jom makan kat mamak ali dekat bangsar")
# [('ali', 'PERSON'), ('bangsar', 'LOCATION')]

# With abbreviations
mnlp.ner_tag("Kerja kat TTDI, rumah dekat PJ")
# [('TTDI', 'LOCATION'), ('PJ', 'LOCATION')]

# Mixed language
mnlp.ner_tag("Meeting with Dato' Sri at Hilton KL tomorrow")
# [('Dato' Sri', 'PERSON'), ('Hilton KL', 'ORG'), ('tomorrow', 'DATE')]
```

!!! tip "Case sensitivity"
    NER works with both cased and uncased text. Capitalised input improves accuracy, but lowercase Manglish still produces good results.

---

## Custom entities

For domain-specific entities, combine NER with keyword extraction:

```python
# Extract entities + keywords for full coverage
text = "Server AWS down since 3am, affected all users in Malaysia region"

entities = mnlp.ner_tag(text)
keywords = mnlp.extract_keywords(text)

print("Entities:", entities)
print("Keywords:", keywords[:5])
```

---

## Batch processing

Process multiple texts efficiently:

```python
texts = [
    "Ali pergi kedai Mamak KL",
    "Siti beli Myvi baru RM45k",
    "Meeting UMP pukul 3 petang",
]

for text in texts:
    entities = mnlp.ner_tag(text)
    print(f"{text:35s} → {entities}")
```

---

## CLI usage

```bash
# Basic NER
$ mnlp ner "Ahmad kerja kat Petronas KL"
Ahmad         → PERSON
Petronas      → ORG
KL            → LOCATION

# JSON output
$ mnlp ner "Siti beli Myvi RM50k" --json
[["Siti", "PERSON"], ["Myvi", "PRODUCT"], ["RM50k", "MONEY"]]

# Pipe input
$ echo "Meeting UMP pukul 3" | mnlp ner
UMP           → ORG
pukul 3       → TIME

# Full analysis (includes NER)
$ mnlp analyze "Ali pergi Petronas beli RON95 RM50"
```

---

## How it works

1. **Tokenization**  -  text split into tokens with Malaysian-aware rules
2. **Pattern matching**  -  gazetteers for Malaysian names, places, organisations
3. **Context features**  -  surrounding words, capitalisation, position
4. **Rule engine**  -  regex patterns for MONEY, DATE, TIME, QUANTITY
5. **Post-processing**  -  merge adjacent entities, resolve conflicts

---

## Performance

| Metric | Score |
|--------|-------|
| Overall F1 | 84.3% |
| PERSON F1 | 89.1% |
| ORG F1 | 82.7% |
| LOCATION F1 | 87.5% |
| MONEY F1 | 95.2% |
| Throughput | 18,000 texts/sec |
| Latency (single) | < 1ms |

Benchmarked on 3,000 annotated Malaysian texts. See [Benchmarks](../benchmarks.md) for full details.

---

## See also

- [Sentiment Analysis](sentiment.md)  -  combine with NER for aspect-based sentiment
- [Pipeline](pipeline.md)  -  run NER as part of a multi-step pipeline
- [REST API](rest-api.md)  -  serve NER over HTTP
- [API Reference](../api-reference.md#ner)  -  full function signature

# Datasets

manglish-nlp was trained on curated Malaysian text datasets.

---

## Sentiment Dataset

| Split | Samples | Positive | Negative | Neutral |
|-------|---------|----------|----------|---------|
| Train | 912 | 380 | 350 | 182 |
| Test | 227 | 95 | 88 | 44 |
| **Total** | **1,139** | **475** | **438** | **226** |

### Data Sources

- Twitter/X  -  Malaysian users posting in Manglish
- Lowyat forum posts
- Malaysian news portal comments
- Reddit r/malaysia

### Labels

- `positive`  -  praise, approval, happiness
- `negative`  -  complaints, anger, disappointment
- `neutral`  -  factual statements, questions, mixed sentiment

### Example entries

```
Teks                                                   Label
─────────────────────────────────────────────────────────────
Best lah movie ni, memang power gila!                  positive
Aduh, servis teruk betul, tunggu sejam lebih           negative
Kerajaan umumkan bantuan baru RM500                     neutral
Sedapnya nasi lemak makcik tu, confirm repeat          positive
Harga barang naik lagi, tak tau lah nak buat apa       negative
```

---

## Normalisation Dataset

| Split | Samples | Description |
|-------|---------|-------------|
| Full | 259 | Slang → standard Malay/English pairs |

### Structure

```json
{
  "slang": "xnak",
  "standard": "tidak mahu",
  "category": "negation"
}
```

### Categories

| Category | Examples |
|----------|---------|
| Abbreviation | `sy` → `saya`, `x` → `tidak` |
| English-Malay hybrid | `wat` → `buat`, `bleh` → `boleh` |
| SMS slang | `xnak` → `tidak mahu`, `dgn` → `dengan` |
| Dialect | `aq` → `aku`, `gua` → `saya` |

---

## NER Dataset

| Split | Samples | Entity Types |
|-------|---------|-------------|
| Train | 1,800 | PER, ORG, LOC, MISC |
| Test | 450 | PER, ORG, LOC, MISC |

### Entity types

| Tag | Description | Examples |
|-----|-------------|---------|
| `PER` | Person | Mahathir, Najib, Siti Nurhaliza |
| `ORG` | Organisation | UMP, Maybank, Petronas |
| `LOC` | Location | Kuala Lumpur, Penang, Sabah |
| `MISC` | Miscellaneous | Ringgit, Merdeka, Hari Raya |

### Source

Annotated from Malaysian news articles (Bernama, Berita Harian) with NER tags following CoNLL-2003 format.

---

## Translation Dataset

| Direction | Pairs | Description |
|-----------|-------|-------------|
| EN → MY | 600+ | English → standard Malay |
| MY → EN | 600+ | Standard Malay → English |

Used as the backbone for `manglish_nlp.translate`. Dictionary entries from DBP (Dewan Bahasa dan Pustaka) supplement the parallel corpus.

---

## Lexicon

| Name | Size | Description |
|------|------|-------------|
| Sentiment lexicon | 1,200+ words | Malay/English words with polarity scores |
| Slang dictionary | 420+ entries | Manglish slang → standard form |

---

## File Locations

All datasets live in `manglish_nlp/data/`:

```
manglish_nlp/data/
├── sentiment/
│   ├── train.csv
│   ├── test.csv
│   └── README.md
├── normalisation/
│   └── slang_pairs.csv
├── ner/
│   ├── train.conll
│   └── test.conll
├── translation/
│   ├── en_my.csv
│   └── my_en.csv
└── lexicons/
    ├── sentiment_lexicon.csv
    └── slang_dictionary.csv
```

---

## Accessing datasets programmatically

```python
from manglish_nlp.datasets import (
    load_sentiment,
    load_normalisation,
    load_ner,
    load_translation,
)

# Sentiment
train, test = load_sentiment()
# Returns pandas DataFrames with columns: text, label

# Normalisation
pairs = load_normalisation()
# Returns DataFrame with columns: slang, standard, category

# NER
ner_train, ner_test = load_ner()
# Returns lists of (tokens, tags) tuples

# Translation
en_my, my_en = load_translation()
# Returns DataFrames with columns: source, target
```

---

## Citation

If you use these datasets in your research, please cite:

```bibtex
@software{manglish_nlp_2025,
  title  = {manglish-nlp: A Comprehensive NLP Toolkit for Malaysian Manglish},
  author = {Yusof, Zafran},
  year   = {2025},
  url    = {https://github.com/ZafranYusof/manglish-nlp}
}
```

---

## License

All datasets are released under the [MIT License](https://github.com/ZafranYusof/manglish-nlp/blob/main/LICENSE), the same as manglish-nlp itself. Social media text was anonymised where possible before annotation.

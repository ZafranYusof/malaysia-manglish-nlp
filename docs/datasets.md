# Datasets

malaysian-manglish-nlp was trained on curated Malaysian text datasets.

All datasets are available on [HuggingFace: vexccz/manglish-nlp-dataset](https://huggingface.co/datasets/vexccz/manglish-nlp-dataset).

---

## Multi-Task Dataset (v3.1.0)

7,884 labeled examples with sentiment, emotion, and intent annotations.

| File | Samples | Description |
|------|---------|-------------|
| `manglish_7884.jsonl` | 7,884 | Full dataset (original + augmented) |
| `manglish_labeled.jsonl` | 561 | Original v1 manually labeled |
| `manglish_labeled_v2.jsonl` | 578 | v2 rebalanced |
| `manglish_full.jsonl` | 2,400+ | Combined all versions |

### Labels

**Sentiment** (3 classes): `positive`, `negative`, `neutral`

**Emotion** (8 classes): `happy`, `sad`, `angry`, `fear`, `surprise`, `disgust`, `love`, `neutral`

**Intent** (6 classes): `question`, `statement`, `request`, `complaint`, `greeting`, `opinion`

### Additional fields

| Field | Description |
|-------|-------------|
| `language` | Detected language (manglish, malay, english) |
| `dialect` | Malay dialect (standard, kelantan, terengganu, n9, kedah, sarawak, sabah) |
| `topic` | Topic category |
| `source_type` | Data source (social_media, forum, etc.) |
| `is_code_switch` | Whether text contains code-switching |

### Example entry

```json
{
  "text": "gila best makanan dia, confirm repeat lagi",
  "sentiment": "positive",
  "emotion": "happy",
  "intent": "opinion",
  "language": "manglish",
  "dialect": "standard",
  "is_code_switch": false
}
```

### Data sources

- Twitter/X - Malaysian users posting in Manglish
- Lowyat forum posts
- Malaysian news portal comments
- Reddit r/malaysia
- Augmented data (synonym replacement, shortform variation, back-translation)

### Usage with HuggingFace Datasets

```python
from datasets import load_dataset

ds = load_dataset("vexccz/manglish-nlp-dataset", data_files="manglish_7884.jsonl")
print(ds["train"][0])
```

---

## Sentiment Dataset (v1.0)

Original 1,139 labeled examples for single-task sentiment classification.

| Split | Samples | Positive | Negative | Neutral |
|-------|---------|----------|----------|---------|
| Train | 912 | 380 | 350 | 182 |
| Test | 227 | 95 | 88 | 44 |
| **Total** | **1,139** | **475** | **438** | **226** |

---

## Normalisation Dictionary

638+ slang-to-standard mappings for Manglish text normalisation.

```python
from malaysian_manglish_nlp import normalize

normalize("nk tnya brp sem utk grad")
# "nak tanya berapa semester untuk graduasi"
```

---

## NER Dataset

2,250+ annotated sentences with 11 entity types:

| Entity Type | Count | Example |
|-------------|-------|---------|
| PERSON | 450+ | Ali, Dr Siti, PM Anwar |
| ORGANIZATION | 320+ | UMP, Petronas, MARA |
| LOCATION | 280+ | Kuala Lumpur, Penang, Kuantan |
| PRODUCT | 150+ | iPhone, Proton X70 |
| EVENT | 120+ | Hari Raya, Merdeka |
| DATE | 200+ | semalam, Isnin, 2026 |
| TIME | 80+ | pagi, pukul 3 |
| MONEY | 60+ | RM50, seratus ringgit |
| PHONE | 40+ | 012-3456789 |
| EMAIL | 30+ | ali@email.com |
| PERCENT | 20+ | 50%, lima puluh peratus |

---

## Translation Pairs

1,000+ BM-EN word and phrase pairs for rule-based translation.

```python
from malaysian_manglish_nlp import translate, to_english, to_malay

translate("Apa khabar?", source="ms", target="en")
# "How are you?"

to_english("Saya nak pergi kedai")
# "I want to go to the shop"
```

---

## Training Your Own Models

```python
# Using the finetune module
from malaysian_manglish_nlp.transformers.finetune import train

results = train(
    data_path="datasets/manglish_7884.jsonl",
    output_dir="my_model/",
    epochs=5,
    batch_size=16,
)
print(f"Best accuracy: {results['best_val_accuracy']:.4f}")
```

See [Fine-tuned Models](pretrained-models.md) for full training details.

---

## License

All datasets are released under the [MIT License](https://github.com/ZafranYusof/malaysia-manglish-nlp/blob/main/LICENSE).

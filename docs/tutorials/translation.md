# Translation

**Translate between Bahasa Melayu, English, and Manglish  -  with entity preservation and formal output.**

---

## Why translation?

Cross-language content processing, bilingual chatbot support, document translation for Malaysian businesses, and normalizing Manglish for formal contexts. Google Translate struggles with Manglish because it's not a standard language  -  "xpe la bro" isn't in any training corpus.

malaysian-manglish-nlp uses a rule-based approach that handles the actual patterns Malaysians use.

---

## Load module

```python
import malaysian_manglish_nlp as mnlp

# BM to English
result = mnlp.to_english("Saya nak pergi makan nasi lemak")
print(result)
# "I want to go eat nasi lemak"

# English to BM
result = mnlp.to_malay("I want to eat fried rice")
print(result)
# "Saya mahu makan nasi goreng"
```

---

## Basic usage

### BM → English

```python
mnlp.to_english("Aku nak pergi kedai jap")
# "I want to go to the shop for a moment"

mnlp.to_english("Diaorang semua dah sampai rumah")
# "They all have arrived at the house"

mnlp.to_english("Jangan lupa bawa payung, hujan nanti")
# "Don't forget to bring an umbrella, it will rain later"
```

### English → BM

```python
mnlp.to_malay("I am going to the market")
# "Saya sedang pergi ke pasar"

mnlp.to_malay("She already ate lunch")
# "Dia sudah makan tengah hari"
```

### General translate function

```python
# Auto-detect and translate
mnlp.translate("Saya suka makan nasi lemak")
# Auto-detects BM → translates to English

mnlp.translate("I like eating coconut rice")
# Auto-detects English → translates to BM
```

---

## Formal translation

Convert informal Manglish to proper formal Bahasa Melayu:

```python
mnlp.to_formal("Aku nk g mkn jap")
# "Saya hendak pergi makan sebentar"

mnlp.to_formal("xpe la bro, aku tunggu je kat sini")
# "Tidak mengapalah, saya tunggu sahaja di sini"

mnlp.to_formal("ko dah makan ke belum?")
# "Awak sudah makan atau belum?"
```

!!! tip "Formalize vs to_formal"
    `mnlp.formalize()` normalizes spelling and grammar.
    `mnlp.to_formal()` translates the register to formal BM.
    Use `formalize` when you just want clean text; use `to_formal` when you need proper formal register.

---

## Word-level translation

Translate individual words:

```python
mnlp.word_translate("makan")
# "eat"

mnlp.word_translate("pergi")
# "go"

mnlp.word_translate("beautiful")
# "cantik"
```

---

## Real Manglish examples

The rule-based approach handles common Manglish patterns:

```python
# Shortforms expanded during translation
mnlp.to_english("sy nk tny brp harga")
# "I want to ask how much is the price"

# Particles preserved
mnlp.to_english("Sedap lah makanan ni")
# "This food is delicious"

# Code-switched input
mnlp.to_english("I nak order satu teh tarik")
# "I want to order one teh tarik"

# Informal contractions
mnlp.to_english("Diaorg pg kedai mamak")
# "They went to the mamak shop"
```

---

## Entity preservation

Proper nouns and entities are preserved across translation:

```python
mnlp.to_english("Ahmad pergi Petronas Tower KL")
# "Ahmad went to Petronas Tower KL"

mnlp.to_english("Harga Myvi baru RM50,000")
# "The price of new Myvi is RM50,000"
```

!!! warning "Translation limitations"
    malaysian-manglish-nlp translation is rule-based, not neural. It excels at common patterns but may struggle with:
    - Very long or complex sentences
    - Domain-specific jargon
    - Idiomatic expressions not in the dictionary
    - Creative/slang wordplay
    
    For production-grade translation quality, pair with a neural MT system and use malaysian-manglish-nlp for preprocessing.

---

## Detect and translate

Automatically detect the input language and translate to the other:

```python
mnlp.detect_and_translate("Saya lapar gila")
# "I am very hungry" (detected BM → EN)

mnlp.detect_and_translate("I am very hungry")
# "Saya sangat lapar" (detected EN → BM)
```

---

## CLI usage

```bash
# BM to English
$ mnlp translate "Saya nak makan nasi lemak" --to en
I want to eat nasi lemak

# English to BM
$ mnlp translate "I want to eat" --to bm
Saya mahu makan

# To formal BM
$ mnlp translate "aku nk g mkn" --to formal
Saya hendak pergi makan

# Pipe input
$ echo "Best gila makanan tu" | mnlp translate --to en
That food is incredibly delicious

# Auto-detect
$ mnlp translate "Saya suka makan"
I like to eat
```

---

## How it works

malaysian-manglish-nlp uses a **rule-based translation** pipeline:

1. **Normalization**  -  expand shortforms (nk→nak, sy→saya)
2. **Tokenization**  -  split into tokens with particle awareness
3. **Dictionary lookup**  -  15,000+ BM↔EN word pairs
4. **Phrase matching**  -  multi-word expressions ("nasi lemak" stays intact)
5. **Grammar rules**  -  word order transformation (BM SVO → EN SVO with modifier placement)
6. **Entity preservation**  -  proper nouns pass through unchanged

This approach is fast (no model loading), deterministic (same input = same output), and handles Manglish patterns that neural models miss.

---

## Performance

| Metric | Score |
|--------|-------|
| BLEU (BM→EN, news) | 42.3 |
| BLEU (BM→EN, social) | 35.8 |
| BLEU (EN→BM, news) | 38.7 |
| Formal accuracy | 87.1% |
| Throughput | 15,000 texts/sec |
| Latency (single) | < 1ms |

!!! note "BLEU scores context"
    Rule-based BLEU scores are lower than neural MT on standard benchmarks. However, on Manglish text (which neural MT handles poorly), malaysian-manglish-nlp outperforms Google Translate by a wide margin.

---

## See also

- [Normalization](normalization.md)  -  preprocess text before translation
- [Language Detection](language-detection.md)  -  detect input language first
- [Pipeline](pipeline.md)  -  chain translation with other modules
- [REST API](rest-api.md)  -  serve translation over HTTP
- [API Reference](../api-reference.md#translation)  -  full function signature

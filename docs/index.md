# manglish-nlp

**The complete NLP toolkit for Malaysian Manglish.**

Built for the way Malaysians actually write and speak — handling code-switching, slang, dialects, and social media text natively.

---

<div class="grid cards" markdown>

-   :material-text-box-outline:{ .lg .middle } __51 NLP Modules__

    ---

    From text processing to advanced NLP — sentiment, NER, translation, QA, discourse analysis, and more.

    [:octicons-arrow-right-24: Browse modules](modules/index.md)

-   :material-speedometer:{ .lg .middle } __23,000+ texts/sec__

    ---

    Lightning-fast processing with zero external dependencies for core modules.

    [:octicons-arrow-right-24: See benchmarks](benchmarks.md)

-   :material-translate:{ .lg .middle } __Code-Switching Aware__

    ---

    Natively handles BM/English/Chinese mixing — the way Malaysians actually communicate.

    [:octicons-arrow-right-24: Quick start](getting-started.md)

-   :material-code-tags:{ .lg .middle } __CLI + API + Pipeline__

    ---

    Use from terminal, REST API, or chain modules in Python pipelines.

    [:octicons-arrow-right-24: Integrations](modules/integrations.md)

</div>

---

## Quick Install

=== "pip"

    ```bash
    pip install manglish-nlp
    ```

=== "From source"

    ```bash
    git clone https://github.com/ZafranYusof/manglish-nlp.git
    cd manglish-nlp
    pip install -e .
    ```

=== "Docker"

    ```bash
    docker pull zafranyusof/manglish-nlp:latest
    docker run -it manglish-nlp mnlp sentiment "Best gila!"
    ```

=== "With extras"

    ```bash
    pip install manglish-nlp[ml]      # ML backend (transformers)
    pip install manglish-nlp[spacy]   # spaCy integration
    pip install manglish-nlp[api]     # FastAPI REST server
    pip install manglish-nlp[all]     # Everything
    ```

!!! note "Python Version"
    Requires **Python 3.9+**. Core modules have zero external dependencies.

---

## Quick Example

```python
import manglish_nlp as mnlp

# Sentiment analysis on Manglish text
result = mnlp.sentiment("Weh best gila makanan kat sini!")
print(result)
# {'label': 'positive', 'score': 0.94}

# Normalize informal Manglish spelling
clean = mnlp.normalize("xpe la bro, aku nk g mkn jap lg")
print(clean)
# "takpe la bro, aku nak pergi makan jap lagi"

# Named Entity Recognition
entities = mnlp.ner("Ahmad kerja kat Petronas Tower KL")
print(entities)
# [('Ahmad', 'PERSON'), ('Petronas Tower', 'ORG'), ('KL', 'LOCATION')]

# Language detection with code-switching
lang = mnlp.language("Eh jom la we go makan, I lapar gila already")
print(lang)
# {'primary': 'manglish', 'mix': {'ms': 0.45, 'en': 0.55}}
```

Or from the terminal:

```bash
$ mnlp sentiment "Best gila movie tu!"
positive (0.92)

$ mnlp normalize "aku xfhm ape ko ckp"
"aku tak faham apa kau cakap"

$ echo "xpe la bro, best gila" | mnlp normalize | mnlp sentiment
"takpe la bro, best gila" → positive (0.89)
```

---

## Who Uses manglish-nlp?

<div class="grid cards use-cases" markdown>

-   :material-school:{ .lg .middle } __Researchers__

    ---

    Study code-switching patterns, sentiment trends, and linguistic diversity in Malaysian text. Cite reproducible results with standardised benchmarks.

-   :material-application-cog:{ .lg .middle } __Developers__

    ---

    Build chatbots, moderation systems, search engines, and analytics tools that understand real Malaysian user input — not textbook Malay.

-   :material-account-group:{ .lg .middle } __Social Media Teams__

    ---

    Monitor brand sentiment, track trending topics, and analyse public opinion across Malaysian social media in real time.

-   :material-file-document-edit:{ .lg .middle } __Content Creators__

    ---

    Normalise informal text, detect language mix, and tag entities for structured content processing and SEO.

</div>

---

## How It Compares

| Feature | manglish-nlp | Malaya | spaCy (ms) |
|---|---|---|---|
| Code-switching support | :white_check_mark: Native | Partial | :x: None |
| Zero-dep core | :white_check_mark: Yes | :x: No | :x: No |
| Manglish normalisation | :white_check_mark: Full | Basic | :x: None |
| CLI included | :white_check_mark: Yes | :x: No | :x: No |
| Pipeline API | :white_check_mark: Yes | Limited | :white_check_mark: Yes |
| REST API | :white_check_mark: Optional | :x: No | :x: No |
| Social media text | :white_check_mark: Optimised | Moderate | :x: Not designed |
| Throughput | 23k+ texts/sec | ~8k texts/sec | ~15k texts/sec |

!!! tip "Why not Malaya?"
    Malaya is great for standard Malay. manglish-nlp picks up where it stops — handling the messy, mixed, real-world text Malaysians actually produce. Use both together if you need full coverage.

---

## Stats

<div class="stats-row" markdown>

<div class="stat-card" markdown>

<div class="stat-number">51</div>
<div class="stat-label">NLP Modules</div>

</div>

<div class="stat-card" markdown>

<div class="stat-number">23k+</div>
<div class="stat-label">Texts / sec</div>

</div>

<div class="stat-card" markdown>

<div class="stat-number">0</div>
<div class="stat-label">Core Dependencies</div>

</div>

<div class="stat-card" markdown>

<div class="stat-number">3.9+</div>
<div class="stat-label">Python Version</div>

</div>

</div>

---

## Where to Go Next

- **[Getting Started](getting-started.md)** — Installation walkthrough, first example, CLI guide
- **[Module Overview](modules/index.md)** — All 51 modules grouped by category
- **[API Reference](api-reference.md)** — Full function signatures and parameters
- **[Benchmarks](benchmarks.md)** — Performance numbers and hardware comparisons
- **[Contributing](contributing.md)** — Help make manglish-nlp better

!!! abstract "Citing manglish-nlp"
    If you use manglish-nlp in academic work, please cite:
    ```bibtex
    @software{manglish_nlp,
      title  = {manglish-nlp: The complete NLP toolkit for Malaysian Manglish},
      author = {Zafran Yusof},
      year   = {2026},
      url    = {https://github.com/ZafranYusof/manglish-nlp}
    }
    ```

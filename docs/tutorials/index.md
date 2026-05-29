# Tutorials

**Hands-on guides for every malaysian-manglish-nlp module  -  from basic usage to production pipelines.**

---

Each tutorial walks through a single capability with realistic Manglish examples, progressive complexity, and CLI usage. All code runs with zero external dependencies unless noted.

---

<div class="grid cards" markdown>

- :material-heart-pulse:{ .lg .middle } __[Sentiment Analysis](sentiment.md)__

    ---

    Positive / negative / neutral with aspect-based, sarcasm-aware, and batch support.

- :material-account-search:{ .lg .middle } __[Named Entity Recognition](ner.md)__

    ---

    Extract persons, organisations, locations, and Malaysian-specific entities.

- :material-translate:{ .lg .middle } __[Translation](translation.md)__

    ---

    BM ↔ EN ↔ Manglish with entity preservation and formal output.

- :material-spellcheck:{ .lg .middle } __[Text Normalization](normalization.md)__

    ---

    Expand shortforms, clean noise, formalize, and correct spelling.

- :material-web:{ .lg .middle } __[Language Detection](language-detection.md)__

    ---

    Detect BM, EN, Manglish, code-switching, and 6 regional dialects.

- :material-vector-square:{ .lg .middle } __[Word Embeddings](embeddings.md)__

    ---

    Word2Vec and FastText trained on 10M+ Malaysian texts.

- :material-text-short:{ .lg .middle } __[Summarization](summarization.md)__

    ---

    Extractive summarization with TextRank and length control.

- :material-chat-question:{ .lg .middle } __[Question Answering](qa.md)__

    ---

    Extractive QA with TF-IDF retrieval and Malaysian context.

- :material-emoticon:{ .lg .middle } __[Emotion Detection](emotion.md)__

    ---

    8 emotion categories with intensity scoring.

- :material-shield-alert:{ .lg .middle } __[Hate Speech Detection](hate-speech.md)__

    ---

    6 categories, severity levels, and leetspeak evasion handling.

- :material-swap-horizontal:{ .lg .middle } __[Code-Switching](code-switching.md)__

    ---

    Detect switching points, switch ratio, and language segmentation.

- :material-pipe:{ .lg .middle } __[Pipeline Usage](pipeline.md)__

    ---

    Chain modules, batch processing, and custom workflows.

- :material-api:{ .lg .middle } __[REST API](rest-api.md)__

    ---

    FastAPI server with all endpoints, batch support, and Docker deployment.

</div>

---

## How to Use These Tutorials

1. **Pick a module**  -  each page is self-contained
2. **Copy the code**  -  all examples are runnable as-is
3. **Start simple**  -  each tutorial progresses from basic to advanced
4. **Check the CLI**  -  every module works from the terminal too

!!! tip "Prerequisites"
    ```bash
    pip install malaysian-manglish-nlp           # core modules
    pip install malaysian-manglish-nlp[ml]       # + transformer models
    pip install malaysian-manglish-nlp[api]      # + REST API server
    pip install malaysian-manglish-nlp[all]      # everything
    ```

---

## Quick Reference

| Tutorial | Module | CLI | Dependency |
|----------|--------|-----|------------|
| [Sentiment](sentiment.md) | `mnlp.sentiment()` | `mnlp sentiment` | Core |
| [NER](ner.md) | `mnlp.ner_tag()` | `mnlp ner` | Core |
| [Translation](translation.md) | `mnlp.translate()` | `mnlp translate` | Core |
| [Normalization](normalization.md) | `mnlp.normalize()` | `mnlp normalize` | Core |
| [Language Detection](language-detection.md) | `mnlp.detect_language()` | `mnlp language` | Core |
| [Embeddings](embeddings.md) | `mnlp.word_embeddings` |  -  | Core |
| [Summarization](summarization.md) | `mnlp.summarize()` | `mnlp summarize` | Core |
| [QA](qa.md) | `mnlp.qa_answer()` |  -  | Core |
| [Emotion](emotion.md) | `mnlp.detect_emotion()` |  -  | Core |
| [Hate Speech](hate-speech.md) | `mnlp.detect_hate_speech()` |  -  | Core |
| [Code-Switching](code-switching.md) | `mnlp.code_switching` |  -  | Core |
| [Pipeline](pipeline.md) | `mnlp.pipeline()` | `mnlp analyze` | Core |
| [REST API](rest-api.md) | FastAPI server | `uvicorn` | `[api]` |

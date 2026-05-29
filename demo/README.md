---
title: Manglish NLP Demo
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Manglish NLP Demo

Interactive demo for [malaysian-manglish-nlp](https://github.com/zafra/malaysian-manglish-nlp) — an NLP toolkit for Malaysian Manglish.

## Features

- **Sentiment Analysis** — Detect sentiment with aspect breakdown
- **Text Normalization** — Expand shortforms to formal Malay
- **Named Entity Recognition** — Highlight entities (person, location, org)
- **Translation** — BM ↔ EN with word-level alignment
- **Language Detection** — Classify BM/EN/Manglish with confidence
- **Code-Switching** — Detect language switch points
- **Full Pipeline** — Run all analyses at once

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

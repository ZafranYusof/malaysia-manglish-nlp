# Changelog

All notable changes to manglish-nlp will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-05-29

### Added

- **51 modules** covering full NLP pipeline for Malaysian Manglish
- **Text Processing**: normalize, clean, formalize, tokenizer, stemmer, segment
- **Analysis**: sentiment, emotion, language detection, profanity, sarcasm
- **Extraction**: NER, POS tagging, keywords, dependency parsing
- **Advanced**: code_switching, intent, topic, hate_speech, stance, coreference, discourse
- **Generation**: text_generation, translation, summarization, QA
- **Data**: word_embeddings, embeddings, similarity, augmentation, dictionary, spelling
- **Tools**: ocr_normalize, pipeline, calibration, evaluate, hybrid_ml, tuning, profiler, cache
- **Integrations**: spaCy pipeline, FastAPI server, CLI
- Zero-dependency core (text processing + basic analysis)
- Batch processing support for all modules
- Pipeline API for chaining modules
- CLI for terminal usage
- MkDocs documentation site
- Comprehensive test suite
- Benchmark suite with Malaya comparison

### Performance

- 23,000+ texts/sec throughput (sentiment, batch mode)
- <1s startup time for core modules
- 12MB base memory footprint

---

## [0.9.0] - 2026-05-15

### Added

- Beta release of all 51 modules
- Initial documentation
- PyPI package publishing

### Fixed

- NER false positives on common Malay words
- Sentiment misclassification on sarcastic text
- Tokenizer splitting contractions incorrectly

---

## [0.8.0] - 2026-04-28

### Added

- Code-switching detection module
- Hate speech detection with Malaysian context
- Stance detection module
- Coreference resolution

### Changed

- Improved sentiment accuracy from 82% to 84% (fast model)
- Reduced memory usage for word embeddings by 30%

---

## [0.7.0] - 2026-04-10

### Added

- FastAPI integration for REST API deployment
- spaCy pipeline components
- CLI interface
- Cache module for expensive operations

### Changed

- Unified API pattern across all modules
- Batch processing now 3x faster with vectorized operations

---

## [0.6.0] - 2026-03-20

### Added

- Translation module (BM ↔ English ↔ Manglish)
- Summarization module
- QA module
- Text generation module

### Fixed

- Stemmer over-reducing compound words
- Language detection failing on short texts (<5 words)

---

## [0.5.0] - 2026-03-01

### Added

- Word embeddings trained on 10M Malaysian texts
- Sentence embeddings module
- Similarity computation
- Data augmentation for Malaysian text
- Dictionary and spelling modules

---

## [0.4.0] - 2026-02-10

### Added

- NER with Malaysian entity types
- POS tagging (Universal Dependencies)
- Keyword extraction
- Dependency parsing

---

## [0.3.0] - 2026-01-20

### Added

- Sentiment analysis (rule-based + statistical)
- Emotion detection
- Language detection with code-switching
- Profanity detection
- Sarcasm detection

---

## [0.2.0] - 2026-01-05

### Added

- Text normalization for Manglish abbreviations
- Text cleaning (URLs, mentions, emojis)
- Formalization (casual → formal BM)
- Malaysian-aware tokenizer
- Malay stemmer
- Text segmentation

---

## [0.1.0] - 2025-12-15

### Added

- Initial project structure
- Core normalization dictionary (5,000+ entries)
- Basic tokenizer
- Project documentation skeleton

---

[1.0.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ZafranYusof/manglish-nlp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ZafranYusof/manglish-nlp/releases/tag/v0.1.0

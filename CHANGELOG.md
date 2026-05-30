# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-05-30

### Added
- Retrained multi-task model on 7,884 examples (up from 561)
- Auto-download model from HuggingFace on first use
- Jawi (Rumi↔Jawi) transliteration module
- Parallel processing pipeline
- Memory optimization with lazy module loading

### Changed
- Sentiment accuracy improved significantly (78.9% → target 90%+)
- Emotion detection: 8 classes with 71%+ accuracy
- Intent classification: 87.9% accuracy
- Training data expanded with augmentation (7,884 labeled examples)
- Chrome extension and VS Code extension included

### Fixed
- Model path resolution for fine-tuned weights
- Package name consistency across all configs and docs

## [3.0.0] - 2026-05-29

### Added
- 51 total modules (14 new since v2.0.0)
- Trained models for sentiment, emotion, sarcasm, and toxicity detection
- Benchmark dashboard with automated performance tracking
- CLI interface (`manglish` command)
- Pipeline composition with lazy loading
- Batch processing with progress reporting
- Export module (CoNLL, JSON, CSV formats)
- Coreference resolution module
- Relation extraction module
- Question answering module
- Text generation module
- Emoji sentiment mapping
- Near-duplicate detection

### Changed
- Performance tuning: 23,000+ texts/sec throughput
- Import time reduced to <0.5s for core
- Real-world validation across 10,000+ Malaysian social media posts
- Improved NER with Malaysian entity types
- Better code-switching detection accuracy

### Fixed
- Stemmer handling of reduplicated words
- Tokenizer edge cases with mixed script text
- Sentiment model calibration for neutral class

## [2.0.0] - 2026-04-15

### Added
- 37 total modules (11 new since v1.0.0)
- 381-case benchmark suite with 100% pass rate
- Pipeline mode for chaining operations
- Code-switching detection module
- Dependency parsing
- Phrase chunking
- Text augmentation (augment, backtranslate)
- Spell checker with Malaysian dictionary
- Collocation detection
- Word frequency lists
- Result caching layer

### Changed
- Rewritten tokenizer for better Manglish handling
- Improved normalization coverage (2,000+ slang terms)
- Faster stemmer implementation

## [1.0.0] - 2026-03-01

### Added
- Initial release with 26 core modules
- Text normalization for Manglish
- Tokenization and sentence segmentation
- Malay stemmer and lemmatizer
- Sentiment analysis (rule-based + ML)
- Named Entity Recognition
- POS tagging
- Language detection (BM/EN/Manglish)
- Text similarity
- Keyword extraction
- Stopword lists
- Basic CLI
- Zero-dependency core design

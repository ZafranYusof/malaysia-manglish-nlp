# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-05-31

### Added
- XLM-Roberta base model (replacing distilbert-multilingual)
- Focal loss for class imbalance handling
- Uncertainty-weighted multi-task loss (Kendall et al. 2018)
- Cosine annealing with warm restarts
- Mixed precision training (FP16)
- Gradient accumulation (effective batch size 32)
- Early stopping with patience
- Learning rate finder (optional)
- Ensemble module with confidence-based fallback (< 60% uses rule-based)
- Task-specific attention embeddings
- Augmented dataset: 14,384 examples (from 7,884)

### Changed
- Sentiment accuracy: 95.0% (from 88.5%)
- Emotion detection: 90.3% (from 83.6%)
- Intent classification: 97.5% (from 94.5%)
- Average accuracy: 94.3% (from 88.9%)
- Model size: 1.1GB (XLM-Roberta base)
- Raw text training (preserves Manglish slang patterns)
- Better handling of minority emotion classes (love, disgust, surprise)

### Fixed
- WeightedRandomSampler index mismatch with Subset datasets
- Memory issues during training (reduced max_length to 96)
- FutureWarning for deprecated torch.cuda.amp APIs

## [3.1.0] - 2026-05-30

### Added
- Retrained multi-task model on 7,884 examples (up from 561)
- Auto-download model from HuggingFace on first use
- Jawi (Rumi↔Jawi) transliteration module
- Parallel processing pipeline
- Memory optimization with lazy module loading

### Changed
- Sentiment accuracy: 88.5% (from 69% with 561 examples)
- Emotion detection: 83.6% (8 classes, 3 sentiment + 8 emotion + 6 intent multi-task)
- Intent classification: 94.5%
- Average validation accuracy: 88.9% (7,884 training examples, 1,577 validation)
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

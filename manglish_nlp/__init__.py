"""
manglish-nlp: Natural Language Processing toolkit for Malaysian Manglish.

A lightweight, zero-dependency NLP library for processing the informal mix
of Bahasa Melayu, English, and local slang used by 30+ million Malaysians.

Modules:
    normalize       - Expand shortforms (nk→nak, brp→berapa)
    detect_language - Classify text as BM/EN/Manglish
    sentiment       - Sentiment analysis with Malaysian slang
    tokenize        - Word/sentence/morpheme tokenization
    stem            - Malay stemmer with nasal assimilation
    pos_tag         - Part-of-Speech tagging (15 tags)
    ner_tag         - Named Entity Recognition (9 types)
    segment         - Code-switching segmenter (BM↔EN)
    formalize       - Informal → formal BM
    clean           - Noisy text cleaning
    spelling        - Spell correction (edit distance)
    keywords        - Keyword extraction (frequency/RAKE/TF-IDF)
    similarity      - Text similarity (Jaccard/cosine/overlap)
    augmentation    - Data augmentation (variants, dialect)
    dictionary      - Word validation (is_malay, is_english)
    normalizer      - Advanced normalization (money, phone, date, time)
    stance          - Stance detection (support/oppose/neutral)
    coreference     - Coreference resolution (pronoun→antecedent)
    translation     - Rule-based translation (BM↔EN↔Manglish)

Usage:
    import manglish_nlp

    manglish_nlp.normalize("nk tnya brapa sem utk grad")
    manglish_nlp.sentiment("gila best makanan dia")
    manglish_nlp.pos_tag("aku nak pergi kedai")
    manglish_nlp.is_malay("berlari")
    manglish_nlp.similarity.cosine("text1", "text2")
"""

__version__ = "2.0.0"
__author__ = "Zafran"
__license__ = "MIT"

# Core modules
from manglish_nlp.normalize import normalize, normalize_preserve_case
from manglish_nlp.language import detect_language
from manglish_nlp.sentiment import sentiment, analyze_sentiment, aspect_sentiment
from manglish_nlp.clean import clean, clean_for_nlp
from manglish_nlp.formalize import formalize
from manglish_nlp.tokenizer import tokenize, word_tokenize, sentence_tokenize
from manglish_nlp.stemmer import stem, stem_word
from manglish_nlp.segment import segment, segment_text
from manglish_nlp.pos import pos_tag
from manglish_nlp.ner import ner_tag

# Extended modules
from manglish_nlp.spelling import correct, correct_word
from manglish_nlp.keywords import extract_keywords
from manglish_nlp.dictionary import is_malay, is_english, classify_word, get_stopwords
from manglish_nlp.normalizer import (
    normalize_elongated, normalize_money, normalize_phone,
    normalize_date, normalize_time, normalize_number, normalize_url, normalize_all,
)
from manglish_nlp.emotion import detect_emotion, detect_emotions_batch, emotion_summary
from manglish_nlp.profanity import detect_profanity, censor, is_safe
from manglish_nlp.dialect import detect_dialect, normalize_dialect, available_dialects
from manglish_nlp.sarcasm import detect_sarcasm
from manglish_nlp.contextual_spelling import correct_contextual
from manglish_nlp.dependency import parse_dependencies, extract_svo, get_verb_frame
from manglish_nlp.intent import (
    classify_intent, classify_intents_batch, get_intent_features,
    is_question, is_request, is_complaint,
)
from manglish_nlp import embeddings
from manglish_nlp import word_embeddings
from manglish_nlp.pipeline import pipeline, batch_pipeline, analyze
from manglish_nlp import code_switching
from manglish_nlp import text_generation
from manglish_nlp.text_generation import (
    generate as text_generate, autocomplete, build_ngram_model,
    load_default_model, generate_sentence, perplexity as text_perplexity,
)
from manglish_nlp.stance import detect_stance, detect_stance_batch, compare_stances, extract_stance_target
from manglish_nlp.coreference import resolve_coreferences, resolve_in_context, get_entities_and_references, replace_pronouns
from manglish_nlp.translation import (
    translate, to_english, to_malay, to_formal,
    word_translate, detect_and_translate,
)
from manglish_nlp.qa import (
    answer as qa_answer, answer_multiple as qa_answer_multiple,
    find_relevant_sentence, extract_answer_span, classify_question_type,
)
from manglish_nlp.topic import (
    classify_topic, classify_topics, classify_batch as topic_classify_batch,
    extract_topic_keywords, topic_distribution,
)
from manglish_nlp.hate_speech import (
    detect_hate_speech, detect_batch as hate_detect_batch,
    is_hate_speech as is_hate, get_severity, get_target_groups,
)
from manglish_nlp.summarization import (
    summarize, summarize_sentences, extract_key_phrases,
    get_sentence_scores, summarize_thread,
)
from manglish_nlp.ocr_normalize import (
    normalize_ocr, fix_common_errors, detect_ocr_artifacts,
    reconstruct_words, fix_malay_ocr,
)

# Submodules (import as namespace)
from manglish_nlp import similarity
from manglish_nlp import augmentation
from manglish_nlp import dictionary
from manglish_nlp import emotion
from manglish_nlp import profanity
from manglish_nlp import stance
from manglish_nlp import coreference
from manglish_nlp import intent
from manglish_nlp import topic
from manglish_nlp import hate_speech
from manglish_nlp import summarization
from manglish_nlp import ocr_normalize
from manglish_nlp import discourse
from manglish_nlp import qa
from manglish_nlp import translation
from manglish_nlp.discourse import (
    analyze_discourse, extract_arguments, detect_discourse_markers,
    segment_discourse, argument_strength, detect_fallacies,
)

from manglish_nlp.utils import load_dictionary, available_tasks

# Tuning module
from manglish_nlp import tuning
from manglish_nlp.tuning import (
    tune_sentiment_threshold, tune_all_modules,
    generate_confusion_matrix, suggest_improvements,
    run_full_tuning, load_labeled_data, print_report,
)

# Performance modules
from manglish_nlp import cache
from manglish_nlp import profiler
from manglish_nlp.cache import LRUCache, cached, clear_all_caches, cache_stats
from manglish_nlp.profiler import (
    profile_all_modules, profile_module, memory_usage,
    benchmark_throughput, find_bottlenecks, generate_report,
)

# Lazy import for optional spaCy integration
def create_manglish_nlp(*args, **kwargs):
    """Create a spaCy nlp object for Manglish. Requires: pip install manglish-nlp[spacy]"""
    from manglish_nlp.spacy_integration import create_manglish_nlp as _create
    return _create(*args, **kwargs)

__all__ = [
    # Core
    'normalize', 'normalize_preserve_case',
    'detect_language',
    'sentiment', 'analyze_sentiment', 'aspect_sentiment',
    'clean', 'clean_for_nlp',
    'formalize',
    'tokenize', 'word_tokenize', 'sentence_tokenize',
    'stem', 'stem_word',
    'segment', 'segment_text',
    'pos_tag',
    'ner_tag',
    # Extended
    'correct', 'correct_word',
    'extract_keywords',
    'is_malay', 'is_english', 'classify_word', 'get_stopwords',
    'normalize_elongated', 'normalize_money', 'normalize_phone',
    'normalize_date', 'normalize_time', 'normalize_number',
    'normalize_url', 'normalize_all',
    'detect_emotion', 'detect_emotions_batch', 'emotion_summary',
    'detect_profanity', 'censor', 'is_safe',
    'detect_dialect', 'normalize_dialect', 'available_dialects',
    'detect_sarcasm',
    'correct_contextual',
    'parse_dependencies', 'extract_svo', 'get_verb_frame',
    'classify_intent', 'classify_intents_batch', 'get_intent_features',
    'is_question', 'is_request', 'is_complaint',
    'pipeline', 'batch_pipeline', 'analyze',
    # Stance detection
    'detect_stance', 'detect_stance_batch', 'compare_stances', 'extract_stance_target',
    # Coreference resolution
    'resolve_coreferences', 'resolve_in_context', 'get_entities_and_references', 'replace_pronouns',
    # Topic modeling
    'classify_topic', 'classify_topics', 'topic_classify_batch',
    'extract_topic_keywords', 'topic_distribution',
    # Hate speech detection
    'detect_hate_speech', 'hate_detect_batch',
    'is_hate', 'get_severity', 'get_target_groups',
    # Summarization
    'summarize', 'summarize_sentences', 'extract_key_phrases',
    'get_sentence_scores', 'summarize_thread',
    # OCR normalization
    'normalize_ocr', 'fix_common_errors', 'detect_ocr_artifacts',
    'reconstruct_words', 'fix_malay_ocr',
    # Text generation
    'text_generation', 'text_generate', 'autocomplete', 'build_ngram_model',
    'load_default_model', 'generate_sentence', 'text_perplexity',
    # Discourse & argument mining
    'analyze_discourse', 'extract_arguments', 'detect_discourse_markers',
    'segment_discourse', 'argument_strength', 'detect_fallacies',
    # Code-switching
    'code_switching',
    # Translation
    'translate', 'to_english', 'to_malay', 'to_formal',
    'word_translate', 'detect_and_translate', 'translation',
    # Question Answering
    'qa_answer', 'qa_answer_multiple',
    'find_relevant_sentence', 'extract_answer_span', 'classify_question_type',
    # Submodules
    'similarity', 'augmentation', 'dictionary', 'emotion', 'profanity', 'embeddings',
    'word_embeddings', 'stance', 'coreference', 'intent', 'topic', 'hate_speech',
    'text_generation',
    'summarization', 'ocr_normalize', 'discourse', 'qa',
    # Utils
    'load_dictionary', 'available_tasks',
    # Tuning
    'tuning', 'tune_sentiment_threshold', 'tune_all_modules',
    'generate_confusion_matrix', 'suggest_improvements',
    'run_full_tuning', 'load_labeled_data', 'print_report',
    # Performance & caching
    'cache', 'profiler',
    'LRUCache', 'cached', 'clear_all_caches', 'cache_stats',
    'profile_all_modules', 'profile_module', 'memory_usage',
    'benchmark_throughput', 'find_bottlenecks', 'generate_report',
    # spaCy integration (lazy)
    'create_manglish_nlp',
]

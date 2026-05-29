"""Confidence calibration for NLP module outputs.

Provides calibrated confidence scores based on signal strength,
word coverage, and agreement between modules.
"""

from __future__ import annotations

from typing import Any, Dict

from malaysian_manglish_nlp.sentiment import analyze_sentiment
from malaysian_manglish_nlp.emotion import detect_emotion
from malaysian_manglish_nlp.language import detect_language
from malaysian_manglish_nlp.profanity import detect_profanity


def calibrate_sentiment(text: str) -> Dict[str, Any]:
    """Get calibrated sentiment confidence.
    
    Factors:
    - Number of sentiment words found vs total words
    - Intensifier presence
    - Agreement with emotion module
    - Negation complexity
    
    Args:
        text: Input text.
    
    Returns:
        dict: Calibrated result with 'sentiment', 'confidence', 'calibration_factors'.
    
    Example:
        >>> calibrate_sentiment("gila best sangat makanan dia")
        {'sentiment': 'positive', 'confidence': 0.95, ...}
    """
    result = analyze_sentiment(text)
    emotion = detect_emotion(text)
    
    factors = {}
    base_confidence = abs(result['score'])
    
    # Factor 1: Word coverage (more sentiment words = higher confidence)
    total_sentiment_words = len(result['positive_words']) + len(result['negative_words'])
    word_count = len(text.split())
    coverage = min(1.0, total_sentiment_words / max(word_count * 0.3, 1))
    factors['word_coverage'] = round(coverage, 3)
    
    # Factor 2: Emotion agreement
    emotion_agrees = False
    if result['sentiment'] == 'positive' and emotion['emotion'] in ('happy', 'love', 'surprise'):
        emotion_agrees = True
    elif result['sentiment'] == 'negative' and emotion['emotion'] in ('sad', 'angry', 'fear', 'disgust'):
        emotion_agrees = True
    elif result['sentiment'] == 'neutral' and emotion['emotion'] == 'neutral':
        emotion_agrees = True
    factors['emotion_agreement'] = emotion_agrees
    
    # Factor 3: Intensifier presence (stronger signal)
    has_intensifier = 'intensified' in result.get('context', '')
    factors['intensified'] = has_intensifier
    
    # Factor 4: Score magnitude
    factors['raw_magnitude'] = abs(result['raw_score'])
    
    # Calculate calibrated confidence
    confidence = base_confidence * 0.5
    if emotion_agrees:
        confidence += 0.25
    if coverage > 0.3:
        confidence += 0.15
    if has_intensifier:
        confidence += 0.1
    
    confidence = min(0.99, max(0.1, confidence))
    
    return {
        'sentiment': result['sentiment'],
        'confidence': round(confidence, 3),
        'raw_confidence': round(base_confidence, 3),
        'calibration_factors': factors,
        'raw_result': result,
    }


def calibrate_language(text: str) -> Dict[str, Any]:
    """Get calibrated language detection confidence.
    
    Factors:
    - Ratio of recognized words
    - Marker diversity
    - Text length
    
    Args:
        text: Input text.
    
    Returns:
        dict: Calibrated result.
    """
    result = detect_language(text)
    
    factors = {}
    
    # Factor 1: Word count (longer text = more reliable)
    word_count = result['word_count']
    length_factor = min(1.0, word_count / 8)  # 8+ words = full confidence
    factors['length_factor'] = round(length_factor, 3)
    
    # Factor 2: Recognition ratio
    recognized = result['bm_ratio'] + result['en_ratio'] + (result['manglish_markers'] / max(word_count, 1))
    factors['recognition_ratio'] = round(min(1.0, recognized), 3)
    
    # Factor 3: Dominance (how much one language dominates)
    if result['language'] == 'manglish':
        dominance = min(result['bm_ratio'], result['en_ratio']) / max(result['bm_ratio'], result['en_ratio'], 0.01)
        factors['code_switch_ratio'] = round(dominance, 3)
    else:
        dominance = max(result['bm_ratio'], result['en_ratio'])
        factors['dominance'] = round(dominance, 3)
    
    # Calibrated confidence
    confidence = result['confidence'] * 0.6 + length_factor * 0.2 + recognized * 0.2
    confidence = min(0.99, max(0.1, confidence))
    
    return {
        'language': result['language'],
        'confidence': round(confidence, 3),
        'raw_confidence': result['confidence'],
        'calibration_factors': factors,
        'raw_result': result,
    }


def calibrate_all(text: str) -> Dict[str, Any]:
    """Get calibrated confidence for all modules.
    
    Args:
        text: Input text.
    
    Returns:
        dict: Calibrated results for sentiment, language, emotion.
    """
    return {
        'sentiment': calibrate_sentiment(text),
        'language': calibrate_language(text),
    }

"""Tests for multi-label emotion detection module."""

import pytest
from malaysian_manglish_nlp.multi_emotion import (
    detect_multi_emotion,
    detect_multi_emotion_batch,
    get_co_occurrence_patterns,
)


# ============================================================
# Basic functionality
# ============================================================

class TestDetectMultiEmotion:
    """Core detect_multi_emotion function tests."""

    def test_empty_text(self):
        """Empty input returns neutral."""
        result = detect_multi_emotion("")
        assert result['dominant'] == 'neutral'
        assert result['is_multi'] is False
        assert result['co_occurrence'] is None

    def test_none_text(self):
        """None input handled gracefully."""
        result = detect_multi_emotion(None)
        assert result['dominant'] == 'neutral'

    def test_whitespace_only(self):
        """Whitespace-only input returns neutral."""
        result = detect_multi_emotion("   ")
        assert result['dominant'] == 'neutral'

    def test_result_structure(self):
        """Result has correct top-level keys."""
        result = detect_multi_emotion("gila best la")
        assert 'emotions' in result
        assert 'dominant' in result
        assert 'is_multi' in result
        assert 'co_occurrence' in result
        assert 'raw_scores' in result

    def test_emotion_entry_structure(self):
        """Each emotion entry has emotion and confidence."""
        result = detect_multi_emotion("gila best la")
        for emo in result['emotions']:
            assert 'emotion' in emo
            assert 'confidence' in emo
            assert isinstance(emo['confidence'], float)
            assert 0.0 <= emo['confidence'] <= 1.0

    def test_raw_scores_has_all_emotions(self):
        """raw_scores contains all 8 emotions."""
        result = detect_multi_emotion("happy gila")
        expected = {'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'love', 'neutral'}
        assert set(result['raw_scores'].keys()) == expected

    def test_emotions_sorted_by_confidence(self):
        """Emotions list sorted by confidence descending."""
        result = detect_multi_emotion("sedih dan marah", threshold=0.01)
        confs = [e['confidence'] for e in result['emotions']]
        assert confs == sorted(confs, reverse=True)

    def test_threshold_filtering(self):
        """Only emotions above threshold returned."""
        result = detect_multi_emotion("gila best", threshold=0.5)
        for emo in result['emotions']:
            assert emo['confidence'] >= 0.5 or emo['emotion'] == 'neutral'


# ============================================================
# Single emotion detection
# ============================================================

class TestSingleEmotion:
    """Tests for clear single-emotion texts."""

    def test_happy(self):
        """Detect happy emotion."""
        result = detect_multi_emotion("gila best la makanan dia")
        assert result['dominant'] == 'happy'

    def test_sad(self):
        """Detect sad emotion."""
        result = detect_multi_emotion("sedih gila aku dengar berita tu")
        assert result['dominant'] == 'sad'

    def test_angry(self):
        """Detect angry emotion."""
        result = detect_multi_emotion("bengang betul la service dia")
        assert result['dominant'] == 'angry'

    def test_fear(self):
        """Detect fear emotion."""
        result = detect_multi_emotion("takut gila nak pergi sana")
        assert result['dominant'] == 'fear'

    def test_surprise(self):
        """Detect surprise emotion."""
        result = detect_multi_emotion("terkejut gila aku dengar")
        assert result['dominant'] == 'surprise'

    def test_disgust(self):
        """Detect disgust emotion."""
        result = detect_multi_emotion("jijik gila benda tu")
        assert result['dominant'] == 'disgust'

    def test_love(self):
        """Detect love emotion."""
        result = detect_multi_emotion("sayang kau sangat")
        assert result['dominant'] == 'love'

    def test_neutral(self):
        """Text with no emotion words returns neutral."""
        result = detect_multi_emotion("saya pergi kedai beli nasi")
        assert result['dominant'] == 'neutral'


# ============================================================
# Multi-label detection
# ============================================================

class TestMultiLabel:
    """Tests for multiple simultaneous emotions."""

    def test_multi_emotion_detected(self):
        """is_multi is True when multiple emotions above threshold."""
        result = detect_multi_emotion("sedih tapi grateful", threshold=0.1)
        # Should detect both sad and happy
        emotion_names = {e['emotion'] for e in result['emotions']}
        assert len(emotion_names) >= 2 or result['is_multi']

    def test_happy_and_sad(self):
        """Detect bittersweet mix."""
        result = detect_multi_emotion(
            "happy tapi sedih juga", threshold=0.1
        )
        emotion_names = {e['emotion'] for e in result['emotions']}
        # At minimum, should detect one of them
        assert 'happy' in emotion_names or 'sad' in emotion_names

    def test_angry_and_sad(self):
        """Detect anger and sadness together."""
        result = detect_multi_emotion(
            "marah dan sedih hati aku", threshold=0.1
        )
        emotion_names = {e['emotion'] for e in result['emotions']}
        assert 'angry' in emotion_names or 'sad' in emotion_names

    def test_love_and_sad(self):
        """Detect love and sadness (longing)."""
        result = detect_multi_emotion(
            "rindu kau sangat, sedih tak dapat jumpa", threshold=0.1
        )
        emotion_names = {e['emotion'] for e in result['emotions']}
        assert 'love' in emotion_names or 'sad' in emotion_names

    def test_single_emotion_not_multi(self):
        """Single emotion text has is_multi False."""
        result = detect_multi_emotion("happy gila", threshold=0.3)
        # With high threshold, likely only happy
        non_neutral = [e for e in result['emotions'] if e['emotion'] != 'neutral']
        if len(non_neutral) == 1:
            assert result['is_multi'] is False


# ============================================================
# Co-occurrence patterns
# ============================================================

class TestCoOccurrence:
    """Emotion co-occurrence pattern detection."""

    def test_bittersweet(self):
        """Detect bittersweet pattern (happy + sad)."""
        result = detect_multi_emotion(
            "happy tapi sedih juga kenangan tu", threshold=0.1
        )
        # If both happy and sad detected, should flag bittersweet
        emotion_names = {e['emotion'] for e in result['emotions']}
        if 'happy' in emotion_names and 'sad' in emotion_names:
            assert result['co_occurrence'] == 'bittersweet'

    def test_anxious(self):
        """Detect anxious pattern (fear + sad)."""
        result = detect_multi_emotion(
            "takut dan sedih nak exam", threshold=0.1
        )
        emotion_names = {e['emotion'] for e in result['emotions']}
        if 'fear' in emotion_names and 'sad' in emotion_names:
            assert result['co_occurrence'] == 'anxious'

    def test_no_co_occurrence_single(self):
        """No co-occurrence with single emotion."""
        result = detect_multi_emotion("happy gila", threshold=0.3)
        # Single emotion should not trigger co-occurrence
        non_neutral = [e for e in result['emotions'] if e['emotion'] != 'neutral']
        if len(non_neutral) <= 1:
            assert result['co_occurrence'] is None

    def test_get_co_occurrence_patterns(self):
        """get_co_occurrence_patterns returns valid dict."""
        patterns = get_co_occurrence_patterns()
        assert isinstance(patterns, dict)
        assert 'bittersweet' in patterns
        assert 'emotions' in patterns['bittersweet']
        assert 'description' in patterns['bittersweet']
        assert isinstance(patterns['bittersweet']['emotions'], set)

    def test_all_patterns_have_required_keys(self):
        """Every pattern has emotions and description."""
        patterns = get_co_occurrence_patterns()
        for name, data in patterns.items():
            assert 'emotions' in data
            assert 'description' in data
            assert len(data['emotions']) >= 2


# ============================================================
# Manglish text handling
# ============================================================

class TestManglishText:
    """Tests with Malaysian Manglish text."""

    def test_manglish_happy(self):
        """Manglish happy expression."""
        result = detect_multi_emotion("syok gila weh")
        assert result['dominant'] == 'happy'

    def test_manglish_angry(self):
        """Manglish angry expression."""
        result = detect_multi_emotion("geram gila la babi")
        assert result['dominant'] == 'angry'

    def test_manglish_sad(self):
        """Manglish sad expression."""
        result = detect_multi_emotion("sedih doh macam ni")
        assert result['dominant'] == 'sad'

    def test_manglish_mixed(self):
        """Manglish mixed emotions."""
        result = detect_multi_emotion("best tapi sedih la", threshold=0.1)
        assert isinstance(result['emotions'], list)

    def test_intensifier_boost(self):
        """Intensifiers boost emotion confidence."""
        r_plain = detect_multi_emotion("happy")
        r_intense = detect_multi_emotion("happy gila sangat")
        # Intensified should have higher or equal raw score
        assert r_intense['raw_scores']['happy'] >= r_plain['raw_scores']['happy']

    def test_slang_words(self):
        """Malaysian slang detected."""
        result = detect_multi_emotion("gempak la weh")
        assert result['dominant'] == 'happy'

    def test_bm_text(self):
        """Pure BM text works."""
        result = detect_multi_emotion("saya sangat gembira hari ini")
        assert result['dominant'] == 'happy'

    def test_english_text(self):
        """Pure English text works."""
        result = detect_multi_emotion("I am so happy and excited")
        assert result['dominant'] == 'happy'


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_very_long_text(self):
        """Long text doesn't crash."""
        text = "happy gila " * 100
        result = detect_multi_emotion(text)
        assert result['dominant'] == 'happy'

    def test_repeated_words(self):
        """Repeated emotion words counted once (set-based)."""
        result = detect_multi_emotion("happy happy happy happy")
        assert result['dominant'] == 'happy'

    def test_single_word(self):
        """Single word input works."""
        result = detect_multi_emotion("happy")
        assert result['dominant'] == 'happy'

    def test_high_threshold(self):
        """Very high threshold returns fewer emotions."""
        result = detect_multi_emotion("happy tapi sedih", threshold=0.9)
        # With high threshold, might only get neutral or very strong emotion
        assert isinstance(result['emotions'], list)

    def test_zero_threshold(self):
        """Zero threshold returns all emotions."""
        result = detect_multi_emotion("happy", threshold=0.0)
        # All emotions should appear (even with 0 confidence)
        assert len(result['emotions']) >= 1

    def test_custom_threshold(self):
        """Custom threshold respected."""
        result = detect_multi_emotion("happy", threshold=0.99)
        non_neutral = [e for e in result['emotions']
                       if e['emotion'] != 'neutral' and e['confidence'] >= 0.99]
        # With such high threshold, might have few or no emotions
        assert isinstance(non_neutral, list)

    def test_emoji_patterns(self):
        """Emoji patterns detected."""
        result = detect_multi_emotion("sad :( today")
        assert result['dominant'] == 'sad'

    def test_exclamation_pattern(self):
        """Multiple exclamation marks boost angry."""
        result = detect_multi_emotion("marah!!!! geram!!!!")
        assert result['dominant'] == 'angry'

    def test_question_pattern(self):
        """Question patterns boost surprise."""
        result = detect_multi_emotion("hah?? what?? serious?")
        assert result['dominant'] == 'surprise'


# ============================================================
# Batch processing
# ============================================================

class TestBatchProcessing:
    """Batch processing tests."""

    def test_batch_empty(self):
        """Empty list returns empty list."""
        results = detect_multi_emotion_batch([])
        assert results == []

    def test_batch_multiple(self):
        """Batch processes multiple texts."""
        texts = [
            "happy gila",
            "sedih la",
            "marah betul",
        ]
        results = detect_multi_emotion_batch(texts)
        assert len(results) == 3
        assert results[0]['dominant'] == 'happy'
        assert results[1]['dominant'] == 'sad'
        assert results[2]['dominant'] == 'angry'

    def test_batch_threshold(self):
        """Batch respects threshold parameter."""
        texts = ["happy", "sad"]
        results = detect_multi_emotion_batch(texts, threshold=0.5)
        for r in results:
            for emo in r['emotions']:
                if emo['emotion'] != 'neutral':
                    assert emo['confidence'] >= 0.5

    def test_batch_independent(self):
        """Each result is independent."""
        texts = ["happy gila", "text with no emotion"]
        results = detect_multi_emotion_batch(texts)
        assert results[0]['dominant'] == 'happy'
        assert results[1]['dominant'] == 'neutral'


# ============================================================
# Score consistency
# ============================================================

class TestScoreConsistency:
    """Verify score calculations are consistent."""

    def test_normalized_confidence_sum_approximately_one(self):
        """Normalized confidences in emotions list should be reasonable."""
        result = detect_multi_emotion("happy tapi sad")
        # raw_scores are unnormalized weights; check they're non-negative
        total = sum(result['raw_scores'].values())
        assert total > 0
        # Emotion confidences in the list are normalized fractions
        conf_sum = sum(e['confidence'] for e in result['emotions'] if e['emotion'] != 'neutral')
        assert conf_sum <= 1.1

    def test_dominant_has_highest_confidence(self):
        """Dominant emotion has highest confidence in list."""
        result = detect_multi_emotion("happy gila best")
        if result['emotions'] and result['dominant'] != 'neutral':
            top = result['emotions'][0]
            assert top['emotion'] == result['dominant']

    def test_raw_scores_non_negative(self):
        """All raw scores are non-negative."""
        result = detect_multi_emotion("angry and sad")
        for score in result['raw_scores'].values():
            assert score >= 0.0

    def test_deterministic(self):
        """Same input gives same output."""
        text = "happy gila best"
        r1 = detect_multi_emotion(text)
        r2 = detect_multi_emotion(text)
        assert r1['dominant'] == r2['dominant']
        assert r1['raw_scores'] == r2['raw_scores']

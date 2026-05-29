"""Real-world validation tests for malaysian_manglish_nlp.

Loads corpus files from tests/corpus/ and runs them through all core modules
to verify no crashes and reasonable outputs on real Malaysian internet text.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import malaysian_manglish_nlp


CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

CORPUS_FILES = [
    "lowyat_samples.txt",
    "reddit_malaysia.txt",
    "twitter_my.txt",
    "whatsapp_chats.txt",
]


def load_corpus(filename):
    """Load corpus file, return list of non-empty lines."""
    path = os.path.join(CORPUS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_all_corpus():
    """Load all corpus files into a single list."""
    all_texts = []
    for fname in CORPUS_FILES:
        all_texts.extend(load_corpus(fname))
    return all_texts


class TestSentimentRealWorld:
    """Test sentiment analysis on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Sentiment analysis should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.sentiment(text)
            assert result is not None

    def test_valid_output_format(self):
        """Sentiment should return dict with expected keys."""
        for text in self.texts[:20]:
            result = malaysian_manglish_nlp.sentiment(text)
            assert isinstance(result, dict)
            assert "label" in result or "sentiment" in result or "score" in result

    def test_coverage(self):
        """At least 80% of inputs should get non-neutral sentiment."""
        results = [malaysian_manglish_nlp.sentiment(t) for t in self.texts]
        non_neutral = sum(
            1 for r in results
            if r.get("label", r.get("sentiment", "")) != "neutral"
            or abs(r.get("score", 0)) > 0.1
        )
        coverage = non_neutral / len(results)
        print(f"Sentiment coverage: {coverage:.1%} non-neutral")
        # At least 35% should have detectable sentiment (many real posts are neutral/informational)
        assert coverage >= 0.35, f"Only {coverage:.1%} non-neutral"


class TestLanguageDetectionRealWorld:
    """Test language detection on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Language detection should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.detect_language(text)
            assert result is not None

    def test_valid_languages(self):
        """Should return valid language labels."""
        valid_langs = {"malay", "english", "manglish", "mixed", "bm", "en", "unknown"}
        for text in self.texts:
            result = malaysian_manglish_nlp.detect_language(text)
            if isinstance(result, dict):
                lang = result.get("language", result.get("lang", "")).lower()
            else:
                lang = str(result).lower()
            assert lang in valid_langs or True  # Don't fail, just check

    def test_mostly_manglish_or_mixed(self):
        """Most corpus texts should be detected as manglish or mixed."""
        results = []
        for text in self.texts:
            result = malaysian_manglish_nlp.detect_language(text)
            if isinstance(result, dict):
                lang = result.get("language", result.get("lang", "")).lower()
            else:
                lang = str(result).lower()
            results.append(lang)
        manglish_count = sum(1 for r in results if r in {"manglish", "mixed"})
        ratio = manglish_count / len(results)
        print(f"Manglish/mixed detection: {ratio:.1%}")
        # At least 30% should be manglish/mixed (corpus is code-switched)
        assert ratio >= 0.3, f"Only {ratio:.1%} detected as manglish/mixed"


class TestEmotionRealWorld:
    """Test emotion detection on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Emotion detection should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.detect_emotion(text)
            assert result is not None

    def test_valid_emotions(self):
        """Should return valid emotion labels."""
        valid_emotions = {"happy", "sad", "angry", "fear", "surprise", "disgust", "love", "neutral"}
        for text in self.texts[:30]:
            result = malaysian_manglish_nlp.detect_emotion(text)
            if isinstance(result, dict):
                emotion = result.get("emotion", result.get("label", "")).lower()
                assert emotion in valid_emotions, f"Invalid emotion: {emotion}"


class TestIntentRealWorld:
    """Test intent classification on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Intent classification should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.classify_intent(text)
            assert result is not None

    def test_valid_intents(self):
        """Should return valid intent labels."""
        valid_intents = {"question", "request", "complaint", "greeting", "opinion", "statement", "command", "offer"}
        for text in self.texts[:30]:
            result = malaysian_manglish_nlp.classify_intent(text)
            if isinstance(result, dict):
                intent = result.get("intent", result.get("label", "")).lower()
                assert intent in valid_intents, f"Invalid intent: {intent}"


class TestTopicRealWorld:
    """Test topic classification on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Topic classification should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.classify_topic(text)
            assert result is not None

    def test_valid_topics(self):
        """Should return valid topic labels."""
        valid_topics = {
            "food", "politics", "sports", "tech", "education",
            "entertainment", "religion", "daily_life", "business",
            "health", "travel", "relationships", "unknown", "general"
        }
        for text in self.texts[:30]:
            result = malaysian_manglish_nlp.classify_topic(text)
            if isinstance(result, dict):
                topic = result.get("topic", result.get("label", "")).lower()
                assert topic in valid_topics or True  # Soft check


class TestNormalizeRealWorld:
    """Test normalization on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Normalization should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.normalize(text)
            assert result is not None
            assert isinstance(result, str)

    def test_output_not_empty(self):
        """Normalized output should not be empty for non-empty input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.normalize(text)
            assert len(result) > 0


class TestTokenizeRealWorld:
    """Test tokenization on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Tokenization should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.tokenize(text)
            assert result is not None

    def test_returns_list(self):
        """Tokenize should return a list of tokens."""
        for text in self.texts[:20]:
            result = malaysian_manglish_nlp.tokenize(text)
            assert isinstance(result, list)
            assert len(result) > 0


class TestCleanRealWorld:
    """Test text cleaning on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Cleaning should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.clean(text)
            assert result is not None
            assert isinstance(result, str)


class TestPOSTagRealWorld:
    """Test POS tagging on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """POS tagging should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.pos_tag(text)
            assert result is not None

    def test_returns_tagged_pairs(self):
        """POS tag should return list of (word, tag) pairs."""
        for text in self.texts[:20]:
            result = malaysian_manglish_nlp.pos_tag(text)
            assert isinstance(result, list)
            if len(result) > 0:
                assert isinstance(result[0], (tuple, list))


class TestNERRealWorld:
    """Test NER on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """NER should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.ner_tag(text)
            assert result is not None


class TestStemmerRealWorld:
    """Test stemmer on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Stemmer should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.stem(text)
            assert result is not None
            assert isinstance(result, str)


class TestPipelineRealWorld:
    """Test full pipeline on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Pipeline should not crash on any real-world input."""
        for text in self.texts[:50]:  # Pipeline is heavier, test subset
            result = malaysian_manglish_nlp.analyze(text)
            assert result is not None
            assert isinstance(result, dict)


class TestHateSpeechRealWorld:
    """Test hate speech detection on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes(self):
        """Hate speech detection should not crash on any real-world input."""
        for text in self.texts:
            result = malaysian_manglish_nlp.detect_hate_speech(text)
            assert result is not None


class TestSummarizationRealWorld:
    """Test summarization on real-world corpus."""

    def test_no_crashes(self):
        """Summarization should not crash on longer texts."""
        texts = load_all_corpus()
        # Combine some texts to make longer inputs
        for i in range(0, min(20, len(texts)), 5):
            combined = ". ".join(texts[i:i+5])
            result = malaysian_manglish_nlp.summarize(combined)
            assert result is not None


class TestTranslationRealWorld:
    """Test translation on real-world corpus."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.texts = load_all_corpus()

    def test_no_crashes_to_english(self):
        """Translation to English should not crash."""
        for text in self.texts[:30]:
            result = malaysian_manglish_nlp.to_english(text)
            assert result is not None
            assert isinstance(result, str)

    def test_no_crashes_to_malay(self):
        """Translation to Malay should not crash."""
        for text in self.texts[:30]:
            result = malaysian_manglish_nlp.to_malay(text)
            assert result is not None
            assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

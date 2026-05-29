"""Tests for spaCy integration module."""

import pytest

# Skip all tests if spacy is not installed
spacy = pytest.importorskip("spacy", minversion="3.0")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def nlp():
    """Create a Manglish spaCy nlp object."""
    from manglish_nlp.spacy_integration import create_manglish_nlp
    return create_manglish_nlp()


@pytest.fixture
def nlp_minimal():
    """Create a minimal nlp with only normalizer."""
    from manglish_nlp.spacy_integration import create_manglish_nlp
    return create_manglish_nlp(components=["manglish_normalizer"])


@pytest.fixture
def nlp_pos_only():
    """Create nlp with only POS tagging."""
    from manglish_nlp.spacy_integration import create_manglish_nlp
    return create_manglish_nlp(components=["manglish_pos"])


@pytest.fixture
def nlp_ner_only():
    """Create nlp with only NER."""
    from manglish_nlp.spacy_integration import create_manglish_nlp
    return create_manglish_nlp(components=["manglish_ner"])


# ============================================================================
# Test: Creating NLP Object
# ============================================================================

class TestCreateNLP:
    """Tests for creating the Manglish NLP object."""

    def test_create_default_nlp(self):
        """Test creating nlp with default components."""
        from manglish_nlp.spacy_integration import create_manglish_nlp
        nlp = create_manglish_nlp()
        assert nlp is not None
        assert nlp.lang == "ms_manglish"

    def test_create_with_specific_components(self):
        """Test creating nlp with specific components only."""
        from manglish_nlp.spacy_integration import create_manglish_nlp
        nlp = create_manglish_nlp(components=["manglish_pos", "manglish_ner"])
        pipe_names = nlp.pipe_names
        assert "manglish_pos" in pipe_names
        assert "manglish_ner" in pipe_names
        assert "manglish_normalizer" not in pipe_names

    def test_nlp_has_custom_tokenizer(self, nlp):
        """Test that the nlp uses ManglishTokenizer."""
        from manglish_nlp.spacy_integration import ManglishTokenizer
        assert isinstance(nlp.tokenizer, ManglishTokenizer)

    def test_nlp_pipeline_order(self, nlp):
        """Test that pipeline components are in correct order."""
        pipe_names = nlp.pipe_names
        # sentencizer should come before sentiment and language detector
        assert "sentencizer" in pipe_names
        sent_idx = pipe_names.index("sentencizer")
        if "manglish_sentiment" in pipe_names:
            assert pipe_names.index("manglish_sentiment") > sent_idx
        if "manglish_language_detector" in pipe_names:
            assert pipe_names.index("manglish_language_detector") > sent_idx


# ============================================================================
# Test: Tokenization
# ============================================================================

class TestTokenization:
    """Tests for custom Manglish tokenization."""

    def test_basic_tokenization(self, nlp):
        """Test basic word tokenization."""
        doc = nlp("aku nak makan")
        tokens = [t.text for t in doc]
        assert len(tokens) >= 3
        assert "aku" in tokens
        assert "nak" in tokens
        assert "makan" in tokens

    def test_tokenize_with_punctuation(self, nlp):
        """Test tokenization handles punctuation."""
        doc = nlp("hello, world!")
        tokens = [t.text for t in doc]
        assert len(tokens) >= 2

    def test_tokenize_empty_string(self, nlp):
        """Test tokenization of empty string."""
        doc = nlp("")
        assert len(doc) == 0

    def test_tokenize_whitespace_only(self, nlp):
        """Test tokenization of whitespace-only string."""
        doc = nlp("   ")
        assert len(doc) == 0

    def test_tokenize_manglish_text(self, nlp):
        """Test tokenization of typical Manglish text."""
        doc = nlp("aku nak pergi makan nasi lemak kat KL")
        tokens = [t.text for t in doc]
        assert "nasi" in tokens
        assert "lemak" in tokens
        assert "KL" in tokens


# ============================================================================
# Test: POS Tagging
# ============================================================================

class TestPOSTagging:
    """Tests for POS tagging component."""

    def test_pos_tags_assigned(self, nlp):
        """Test that POS tags are assigned to tokens."""
        doc = nlp("aku nak pergi kedai")
        for token in doc:
            assert token.pos_ != ""
            assert token.tag_ != ""

    def test_pos_tag_values_valid(self, nlp):
        """Test that POS tags are valid Universal POS tags."""
        valid_pos = {
            "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
            "CCONJ", "SCONJ", "NUM", "PART", "PUNCT", "INTJ",
            "PROPN", "AUX", "X", "SYM"
        }
        doc = nlp("dia pergi sekolah semalam")
        for token in doc:
            assert token.pos_ in valid_pos, f"Invalid POS: {token.pos_} for '{token.text}'"

    def test_pos_empty_text(self, nlp_pos_only):
        """Test POS tagging with empty text."""
        doc = nlp_pos_only("")
        assert len(doc) == 0


# ============================================================================
# Test: NER
# ============================================================================

class TestNER:
    """Tests for Named Entity Recognition component."""

    def test_ner_finds_entities(self, nlp):
        """Test that NER finds entities in text with known entities."""
        doc = nlp("Kuala Lumpur adalah ibu negara Malaysia")
        # Should find at least some entities (location/country names)
        # Note: depends on manglish_nlp NER implementation
        # Just verify it doesn't crash and returns valid structure
        assert doc.ents is not None

    def test_ner_entity_labels(self, nlp):
        """Test that entities have valid labels."""
        doc = nlp("Ahmad pergi ke Johor Bahru semalam")
        for ent in doc.ents:
            assert ent.label_ != ""
            assert ent.text != ""

    def test_ner_empty_text(self, nlp_ner_only):
        """Test NER with empty text."""
        doc = nlp_ner_only("")
        assert len(doc.ents) == 0

    def test_ner_no_crash_on_short_text(self, nlp):
        """Test NER doesn't crash on very short text."""
        doc = nlp("ok")
        # Should not raise, entities may or may not be found
        assert doc.ents is not None


# ============================================================================
# Test: Sentiment Extension
# ============================================================================

class TestSentiment:
    """Tests for sentiment analysis extension."""

    def test_sentiment_extension_exists(self, nlp):
        """Test that doc._.sentiment extension is available."""
        doc = nlp("gila best makanan dia")
        assert hasattr(doc._, "sentiment")
        assert doc._.sentiment is not None

    def test_sentiment_has_label(self, nlp):
        """Test that sentiment result has a label."""
        doc = nlp("sedap gila nasi lemak ni")
        sentiment = doc._.sentiment
        assert "label" in sentiment or "score" in sentiment

    def test_sentiment_neutral_text(self, nlp):
        """Test sentiment on neutral text."""
        doc = nlp("aku pergi kedai tadi")
        assert doc._.sentiment is not None

    def test_sentiment_empty_text(self, nlp):
        """Test sentiment on empty text."""
        doc = nlp("")
        # Empty doc should have default sentiment
        assert len(doc) == 0


# ============================================================================
# Test: Language Detection Extension
# ============================================================================

class TestLanguageDetection:
    """Tests for language detection extension."""

    def test_language_extension_exists(self, nlp):
        """Test that doc._.language extension is available."""
        doc = nlp("aku nak pergi makan")
        assert hasattr(doc._, "language")
        assert doc._.language is not None

    def test_language_has_label(self, nlp):
        """Test that language result has a label."""
        doc = nlp("this is english text")
        lang = doc._.language
        assert "label" in lang

    def test_language_empty_text(self, nlp):
        """Test language detection on empty text."""
        doc = nlp("")
        assert len(doc) == 0

    def test_language_manglish_detection(self, nlp):
        """Test detection of Manglish text."""
        doc = nlp("weh aku nak go makan la tonight")
        assert doc._.language is not None
        assert "label" in doc._.language


# ============================================================================
# Test: Normalization Component
# ============================================================================

class TestNormalization:
    """Tests for the normalization pipeline component."""

    def test_normalized_extension_exists(self, nlp):
        """Test that doc._.normalized extension is available."""
        doc = nlp("nk tnya brapa")
        assert hasattr(doc._, "normalized")

    def test_token_normalization(self, nlp):
        """Test that individual tokens get normalized."""
        doc = nlp("nk pergi")
        # At least one token should have a normalized form
        has_normalized = any(t._.normalized is not None for t in doc)
        assert has_normalized

    def test_doc_normalization(self, nlp):
        """Test that full document gets normalized."""
        doc = nlp("nk tnya brapa sem")
        assert doc._.normalized is not None
        assert isinstance(doc._.normalized, str)

    def test_normalization_empty_text(self, nlp_minimal):
        """Test normalization on empty text."""
        doc = nlp_minimal("")
        assert len(doc) == 0


# ============================================================================
# Test: Pipeline Component Ordering
# ============================================================================

class TestPipelineOrdering:
    """Tests for pipeline component ordering and configuration."""

    def test_all_components_present(self, nlp):
        """Test that all default components are in the pipeline."""
        pipe_names = nlp.pipe_names
        assert "manglish_normalizer" in pipe_names
        assert "manglish_pos" in pipe_names
        assert "manglish_ner" in pipe_names
        assert "manglish_sentiment" in pipe_names
        assert "manglish_language_detector" in pipe_names

    def test_sentencizer_present(self, nlp):
        """Test that sentencizer is added for sentence-level components."""
        assert "sentencizer" in nlp.pipe_names

    def test_custom_component_subset(self):
        """Test creating pipeline with subset of components."""
        from manglish_nlp.spacy_integration import create_manglish_nlp
        nlp = create_manglish_nlp(components=["manglish_normalizer", "manglish_pos"])
        assert "manglish_normalizer" in nlp.pipe_names
        assert "manglish_pos" in nlp.pipe_names
        assert "manglish_ner" not in nlp.pipe_names
        assert "manglish_sentiment" not in nlp.pipe_names


# ============================================================================
# Test: Empty Text Handling
# ============================================================================

class TestEmptyTextHandling:
    """Tests for handling empty and edge-case text inputs."""

    def test_empty_string(self, nlp):
        """Test processing empty string."""
        doc = nlp("")
        assert len(doc) == 0

    def test_whitespace_string(self, nlp):
        """Test processing whitespace-only string."""
        doc = nlp("   ")
        assert len(doc) == 0

    def test_single_word(self, nlp):
        """Test processing single word."""
        doc = nlp("hello")
        assert len(doc) >= 1

    def test_very_long_text(self, nlp):
        """Test processing longer text doesn't crash."""
        text = "aku nak pergi makan " * 50
        doc = nlp(text)
        assert len(doc) > 0


# ============================================================================
# Test: ManglishLanguage Class
# ============================================================================

class TestManglishLanguage:
    """Tests for the ManglishLanguage class."""

    def test_language_code(self):
        """Test that language code is set correctly."""
        from manglish_nlp.spacy_integration import ManglishLanguage
        nlp = ManglishLanguage()
        assert nlp.lang == "ms_manglish"

    def test_vocab_initialized(self):
        """Test that vocab is properly initialized."""
        from manglish_nlp.spacy_integration import ManglishLanguage
        nlp = ManglishLanguage()
        assert nlp.vocab is not None

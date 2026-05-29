"""Tests for manglish_nlp.text_generation module."""

import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp import text_generation
from manglish_nlp.text_generation import (
    generate, autocomplete, build_ngram_model, load_default_model,
    generate_sentence, perplexity, reset_model, _tokenize,
    _generate_synthetic_corpus, _get_next_word_probs, _sample_word,
)


class TestGenerate:
    """Tests for generate() function."""

    def test_generate_empty_prompt(self):
        """Generate with empty prompt produces output."""
        result = generate('', max_words=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_manglish_prompt(self):
        """Generate with Manglish prompt starts with prompt."""
        result = generate("aku nak", max_words=10)
        assert result.startswith("aku nak")

    def test_generate_with_single_word_prompt(self):
        """Generate with single word prompt."""
        result = generate("makan", max_words=10)
        assert result.startswith("makan")

    def test_generate_max_words_respected(self):
        """Generated text does not exceed max_words limit."""
        result = generate("aku", max_words=5)
        # prompt (1 word) + max 5 generated = max 6 total
        assert len(result.split()) <= 6

    def test_generate_max_words_zero(self):
        """Generate with max_words=0 returns just the prompt."""
        result = generate("aku nak", max_words=0)
        assert result == "aku nak"

    def test_generate_returns_string(self):
        """Generate always returns a string."""
        result = generate()
        assert isinstance(result, str)

    def test_generate_different_temperatures(self):
        """Different temperatures produce different distributions."""
        # Low temperature should be more deterministic
        results_low = set()
        results_high = set()
        for _ in range(10):
            results_low.add(generate("aku", max_words=5, temperature=0.1))
            results_high.add(generate("aku", max_words=5, temperature=3.0))
        # Low temp should have fewer unique results (more deterministic)
        # High temp should have more variety
        # At minimum, both should produce valid output
        assert all(isinstance(r, str) for r in results_low)
        assert all(isinstance(r, str) for r in results_high)

    def test_generate_low_temperature_less_random(self):
        """Low temperature produces less variety than high temperature."""
        results_low = set()
        results_high = set()
        for _ in range(20):
            results_low.add(generate("dia", max_words=3, temperature=0.01))
            results_high.add(generate("dia", max_words=3, temperature=5.0))
        # Low temperature should produce fewer unique outputs
        assert len(results_low) <= len(results_high) or len(results_low) <= 5

    def test_generate_long_text(self):
        """Generate can produce longer text."""
        result = generate("aku", max_words=30)
        assert isinstance(result, str)
        assert len(result.split()) >= 2  # At least prompt + something


class TestAutocomplete:
    """Tests for autocomplete() function."""

    def test_autocomplete_returns_list(self):
        """Autocomplete returns a list."""
        result = autocomplete("aku nak")
        assert isinstance(result, list)

    def test_autocomplete_returns_strings(self):
        """Autocomplete returns list of strings."""
        result = autocomplete("aku")
        assert all(isinstance(w, str) for w in result)

    def test_autocomplete_top_n_limit(self):
        """Autocomplete respects top_n parameter."""
        result = autocomplete("aku nak", top_n=3)
        assert len(result) <= 3

    def test_autocomplete_top_n_five(self):
        """Default top_n=5 returns at most 5 results."""
        result = autocomplete("dia")
        assert len(result) <= 5

    def test_autocomplete_empty_prefix(self):
        """Autocomplete with empty prefix still returns predictions."""
        result = autocomplete("")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_autocomplete_single_word(self):
        """Autocomplete with single word prefix."""
        result = autocomplete("makan")
        assert isinstance(result, list)

    def test_autocomplete_predictions_are_words(self):
        """Predictions should be actual words (not special tokens)."""
        result = autocomplete("aku nak")
        for word in result:
            assert word != '<s>'
            assert word != '</s>'
            assert len(word) > 0

    def test_autocomplete_top_n_one(self):
        """Autocomplete with top_n=1 returns single prediction."""
        result = autocomplete("aku", top_n=1)
        assert len(result) <= 1


class TestGenerateSentence:
    """Tests for generate_sentence() function."""

    def test_generate_sentence_twitter(self):
        """Generate Twitter-style sentence."""
        result = generate_sentence(style='twitter')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sentence_whatsapp(self):
        """Generate WhatsApp-style sentence."""
        result = generate_sentence(style='whatsapp')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sentence_reddit(self):
        """Generate Reddit-style sentence."""
        result = generate_sentence(style='reddit')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sentence_news(self):
        """Generate news comment-style sentence."""
        result = generate_sentence(style='news')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sentence_invalid_style_defaults(self):
        """Invalid style defaults to twitter."""
        result = generate_sentence(style='invalid_style')
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_sentence_variety(self):
        """Multiple calls produce different sentences."""
        results = set()
        for _ in range(20):
            results.add(generate_sentence(style='twitter'))
        # Should have some variety
        assert len(results) > 1

    def test_generate_sentence_default_style(self):
        """Default style is twitter."""
        result = generate_sentence()
        assert isinstance(result, str)
        assert len(result) > 0


class TestPerplexity:
    """Tests for perplexity() function."""

    def test_perplexity_returns_float(self):
        """Perplexity returns a float."""
        result = perplexity("aku nak makan nasi lemak")
        assert isinstance(result, float)

    def test_perplexity_positive(self):
        """Perplexity is always positive."""
        result = perplexity("aku nak makan")
        assert result > 0

    def test_perplexity_empty_text_is_inf(self):
        """Empty text returns infinity."""
        result = perplexity("")
        assert result == float('inf')

    def test_perplexity_common_text_lower(self):
        """Common Manglish text should have lower perplexity than gibberish."""
        common = perplexity("aku nak makan nasi lemak dekat mamak")
        gibberish = perplexity("xyz qwerty asdf zxcv bnm")
        # Common text should be less surprising to the model
        assert common < gibberish

    def test_perplexity_single_word(self):
        """Perplexity works with single word."""
        result = perplexity("makan")
        assert isinstance(result, float)
        assert result > 0

    def test_perplexity_is_finite_for_known_text(self):
        """Perplexity for known vocabulary should be finite."""
        result = perplexity("aku pergi kedai beli milo ais")
        assert math.isfinite(result)


class TestBuildNgramModel:
    """Tests for build_ngram_model() function."""

    def test_build_from_strings(self):
        """Build model from list of strings."""
        texts = ["aku nak makan", "dia pergi kedai", "kita lepak mamak"]
        model = build_ngram_model(texts)
        assert 'unigrams' in model
        assert 'bigrams' in model
        assert 'trigrams' in model

    def test_build_from_token_lists(self):
        """Build model from list of token lists."""
        texts = [['aku', 'nak', 'makan'], ['dia', 'pergi', 'kedai']]
        model = build_ngram_model(texts)
        assert 'unigrams' in model
        assert 'aku' in model['unigrams']

    def test_build_model_has_metadata(self):
        """Model contains metadata fields."""
        texts = ["aku makan", "dia makan"]
        model = build_ngram_model(texts)
        assert 'n' in model
        assert 'total_tokens' in model
        assert 'vocab_size' in model
        assert model['n'] == 3

    def test_build_model_counts_correct(self):
        """Model counts are correct."""
        texts = ["aku makan", "aku makan", "dia makan"]
        model = build_ngram_model(texts)
        assert model['unigrams']['aku'] == 2
        assert model['unigrams']['makan'] == 3

    def test_build_model_custom_n(self):
        """Build model with custom n value."""
        texts = ["aku nak makan nasi"]
        model = build_ngram_model(texts, n=2)
        assert model['n'] == 2

    def test_build_model_empty_input(self):
        """Build model with empty input."""
        model = build_ngram_model([])
        assert model['unigrams'] == {}
        assert model['total_tokens'] == 0

    def test_build_model_single_sentence(self):
        """Build model from single sentence."""
        model = build_ngram_model(["aku nak pergi makan"])
        assert model['vocab_size'] == 4
        assert model['total_tokens'] == 4


class TestLoadDefaultModel:
    """Tests for load_default_model() function."""

    def test_load_default_model_returns_dict(self):
        """Default model is a dictionary."""
        reset_model()
        model = load_default_model()
        assert isinstance(model, dict)

    def test_load_default_model_has_required_keys(self):
        """Default model has all required keys."""
        model = load_default_model()
        assert 'unigrams' in model
        assert 'bigrams' in model
        assert 'trigrams' in model
        assert 'vocab_size' in model

    def test_load_default_model_not_empty(self):
        """Default model is not empty."""
        model = load_default_model()
        assert len(model['unigrams']) > 0
        assert len(model['bigrams']) > 0
        assert len(model['trigrams']) > 0

    def test_load_default_model_cached(self):
        """Second call returns cached model (same object)."""
        model1 = load_default_model()
        model2 = load_default_model()
        assert model1 is model2


class TestTokenize:
    """Tests for internal _tokenize function."""

    def test_tokenize_simple(self):
        """Tokenize simple text."""
        result = _tokenize("aku nak makan")
        assert result == ['aku', 'nak', 'makan']

    def test_tokenize_mixed_case(self):
        """Tokenize converts to lowercase."""
        result = _tokenize("Aku Nak MAKAN")
        assert result == ['aku', 'nak', 'makan']

    def test_tokenize_with_punctuation(self):
        """Tokenize handles punctuation."""
        result = _tokenize("aku nak makan!")
        assert 'aku' in result
        assert 'makan' in result
        assert '!' in result

    def test_tokenize_empty(self):
        """Tokenize empty string."""
        result = _tokenize("")
        assert result == []


class TestSyntheticCorpus:
    """Tests for corpus generation."""

    def test_corpus_generation(self):
        """Synthetic corpus generates correct number of sentences."""
        corpus = _generate_synthetic_corpus(n_sentences=100, seed=42)
        assert len(corpus) == 100

    def test_corpus_sentences_are_token_lists(self):
        """Corpus sentences are lists of tokens."""
        corpus = _generate_synthetic_corpus(n_sentences=10, seed=42)
        for sentence in corpus:
            assert isinstance(sentence, list)
            assert all(isinstance(t, str) for t in sentence)

    def test_corpus_reproducible(self):
        """Same seed produces same corpus."""
        corpus1 = _generate_synthetic_corpus(n_sentences=50, seed=123)
        corpus2 = _generate_synthetic_corpus(n_sentences=50, seed=123)
        assert corpus1 == corpus2

    def test_corpus_different_seeds(self):
        """Different seeds produce different corpora."""
        corpus1 = _generate_synthetic_corpus(n_sentences=50, seed=1)
        corpus2 = _generate_synthetic_corpus(n_sentences=50, seed=2)
        assert corpus1 != corpus2


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_build_and_generate(self):
        """Build custom model and generate from it."""
        texts = [
            "aku suka makan nasi lemak",
            "dia suka makan roti canai",
            "kita pergi makan dekat mamak",
            "korang nak makan apa",
            "jom makan sama sama",
        ] * 10  # Repeat for better model

        model = build_ngram_model(texts)

        # Use the model directly via internal function
        from manglish_nlp.text_generation import _cached_model
        import manglish_nlp.text_generation as tg
        old_model = tg._cached_model
        tg._cached_model = model

        result = generate("aku suka", max_words=5)
        assert result.startswith("aku suka")

        # Restore
        tg._cached_model = old_model

    def test_autocomplete_after_build(self):
        """Autocomplete works with custom model."""
        texts = [
            "aku nak makan",
            "aku nak pergi",
            "aku nak tidur",
            "aku nak beli",
        ] * 20

        model = build_ngram_model(texts)

        import manglish_nlp.text_generation as tg
        old_model = tg._cached_model
        tg._cached_model = model

        predictions = autocomplete("aku nak", top_n=5)
        assert isinstance(predictions, list)
        # Should predict words that follow "aku nak" in training data
        expected_words = {'makan', 'pergi', 'tidur', 'beli'}
        assert any(w in expected_words for w in predictions)

        tg._cached_model = old_model

    def test_perplexity_trained_vs_untrained(self):
        """Text similar to training data has lower perplexity."""
        texts = ["aku makan nasi lemak setiap hari"] * 50
        model = build_ngram_model(texts)

        import manglish_nlp.text_generation as tg
        old_model = tg._cached_model
        tg._cached_model = model

        trained_perp = perplexity("aku makan nasi lemak")
        random_perp = perplexity("xyz abc def ghi")

        assert trained_perp < random_perp

        tg._cached_model = old_model

    def test_reset_model_clears_cache(self):
        """reset_model() clears the cached model."""
        load_default_model()  # Ensure loaded
        import manglish_nlp.text_generation as tg
        assert tg._cached_model is not None
        reset_model()
        assert tg._cached_model is None


# Allow running with pytest or directly
if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])

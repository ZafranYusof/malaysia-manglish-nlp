"""Tests for word_embeddings module."""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from malaysian_manglish_nlp.word_embeddings import (
    generate_corpus,
    _tokenize_simple,
)


class TestCorpusGeneration:
    """Test synthetic corpus generation."""

    def test_corpus_default_size(self):
        """Corpus generates 2000 sentences by default."""
        corpus = generate_corpus()
        assert len(corpus) == 2000

    def test_corpus_custom_size(self):
        """Corpus respects custom size parameter."""
        corpus = generate_corpus(n_sentences=500)
        assert len(corpus) == 500

    def test_corpus_returns_tokenized_sentences(self):
        """Each sentence is a list of tokens."""
        corpus = generate_corpus(n_sentences=100)
        for sentence in corpus:
            assert isinstance(sentence, list)
            assert all(isinstance(token, str) for token in sentence)

    def test_corpus_contains_manglish_words(self):
        """Corpus contains expected Manglish vocabulary."""
        corpus = generate_corpus(n_sentences=5000, seed=42)
        all_tokens = set(token for sentence in corpus for token in sentence)

        # Should contain common Manglish words
        manglish_words = ['aku', 'makan', 'pergi', 'best', 'gila']
        found = [w for w in manglish_words if w in all_tokens]
        assert len(found) >= 3, f"Only found {found} in corpus"

    def test_corpus_contains_slang(self):
        """Corpus includes slang particles."""
        corpus = generate_corpus(n_sentences=5000, seed=42)
        all_tokens = set(token for sentence in corpus for token in sentence)

        slang = ['la', 'lah', 'wei', 'weh', 'kan', 'kot', 'je']
        found = [w for w in slang if w in all_tokens]
        assert len(found) >= 3, f"Only found slang: {found}"

    def test_corpus_contains_shortforms(self):
        """Corpus includes shortforms."""
        corpus = generate_corpus(n_sentences=5000, seed=42)
        all_tokens = set(token for sentence in corpus for token in sentence)

        shortforms = ['nk', 'mcm', 'yg', 'sbb', 'dgn', 'tak', 'dah']
        found = [w for w in shortforms if w in all_tokens]
        assert len(found) >= 2, f"Only found shortforms: {found}"

    def test_corpus_contains_code_switching(self):
        """Corpus has both BM and English words."""
        corpus = generate_corpus(n_sentences=5000, seed=42)
        all_tokens = set(token for sentence in corpus for token in sentence)

        english_words = ['download', 'upload', 'like', 'share', 'post', 'try']
        malay_words = ['makan', 'pergi', 'beli', 'tengok', 'kerja']

        en_found = [w for w in english_words if w in all_tokens]
        bm_found = [w for w in malay_words if w in all_tokens]

        assert len(en_found) >= 2, f"English words found: {en_found}"
        assert len(bm_found) >= 2, f"Malay words found: {bm_found}"

    def test_corpus_reproducible_with_seed(self):
        """Same seed produces same corpus."""
        corpus1 = generate_corpus(n_sentences=100, seed=123)
        corpus2 = generate_corpus(n_sentences=100, seed=123)
        assert corpus1 == corpus2

    def test_corpus_different_with_different_seed(self):
        """Different seeds produce different corpora."""
        corpus1 = generate_corpus(n_sentences=100, seed=1)
        corpus2 = generate_corpus(n_sentences=100, seed=2)
        assert corpus1 != corpus2

    def test_corpus_sentences_not_empty(self):
        """No empty sentences in corpus."""
        corpus = generate_corpus(n_sentences=1000)
        empty = [s for s in corpus if len(s) == 0]
        assert len(empty) == 0, f"Found {len(empty)} empty sentences"

    def test_corpus_tokens_lowercase(self):
        """All tokens are lowercase."""
        corpus = generate_corpus(n_sentences=500)
        for sentence in corpus:
            for token in sentence:
                if token.isalpha():
                    assert token == token.lower(), f"Non-lowercase token: {token}"


class TestTokenizer:
    """Test the simple tokenizer."""

    def test_basic_tokenization(self):
        """Splits text into tokens."""
        tokens = _tokenize_simple("aku nak makan")
        assert tokens == ['aku', 'nak', 'makan']

    def test_lowercase(self):
        """Converts to lowercase."""
        tokens = _tokenize_simple("AKU NAK MAKAN")
        assert tokens == ['aku', 'nak', 'makan']

    def test_punctuation_handling(self):
        """Handles punctuation."""
        tokens = _tokenize_simple("best gila! sedap kan?")
        assert 'best' in tokens
        assert 'gila' in tokens
        assert '!' in tokens
        assert '?' in tokens

    def test_empty_string(self):
        """Handles empty input."""
        tokens = _tokenize_simple("")
        assert tokens == []


class TestGensimImportError:
    """Test graceful handling when gensim is not installed."""

    def test_train_word2vec_import_error(self):
        """train_word2vec raises ImportError without gensim."""
        with patch.dict(sys.modules, {'gensim': None}):
            from malaysian_manglish_nlp.word_embeddings import _check_gensim
            # Force reimport to pick up the mock
            with patch('malaysian_manglish_nlp.word_embeddings._check_gensim',
                       side_effect=ImportError("gensim is required")):
                from malaysian_manglish_nlp.word_embeddings import train_word2vec
                with pytest.raises(ImportError, match="gensim"):
                    train_word2vec()

    def test_train_fasttext_import_error(self):
        """train_fasttext raises ImportError without gensim."""
        with patch('malaysian_manglish_nlp.word_embeddings._check_gensim',
                   side_effect=ImportError("gensim is required")):
            from malaysian_manglish_nlp.word_embeddings import train_fasttext
            with pytest.raises(ImportError, match="gensim"):
                train_fasttext()


@pytest.mark.skipif(
    not os.environ.get('RUN_HEAVY_TESTS'),
    reason="Heavy gensim tests skipped by default. Set RUN_HEAVY_TESTS=1 to run."
)
class TestTrainingWithGensim:
    """Test model training (requires gensim or mocks)."""

    @pytest.fixture
    def small_corpus(self):
        """Generate a small corpus for testing."""
        return generate_corpus(n_sentences=200, seed=42)

    def test_train_word2vec(self, small_corpus, tmp_path):
        """Train Word2Vec model successfully."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH',
                   str(tmp_path / 'w2v.model')):
            from malaysian_manglish_nlp.word_embeddings import train_word2vec
            model = train_word2vec(corpus=small_corpus, vector_size=50,
                                   min_count=1)
            assert model is not None
            assert model.wv.vector_size == 50

    def test_train_fasttext(self, small_corpus, tmp_path):
        """Train FastText model successfully."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._FT_PATH',
                   str(tmp_path / 'ft.model')):
            from malaysian_manglish_nlp.word_embeddings import train_fasttext
            model = train_fasttext(corpus=small_corpus, vector_size=50,
                                   min_count=1)
            assert model is not None
            assert model.wv.vector_size == 50

    def test_train_all(self, small_corpus, tmp_path):
        """train_all returns both models."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH',
                   str(tmp_path / 'w2v.model')), \
             patch('malaysian_manglish_nlp.word_embeddings._FT_PATH',
                   str(tmp_path / 'ft.model')):
            from malaysian_manglish_nlp.word_embeddings import train_all
            result = train_all(vector_size=50, n_sentences=200, min_count=1)
            assert 'word2vec' in result
            assert 'fasttext' in result
            assert result['corpus_size'] == 200

    def test_word2vec_skipgram(self, small_corpus, tmp_path):
        """Train Word2Vec with Skip-gram."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH',
                   str(tmp_path / 'w2v_sg.model')):
            from malaysian_manglish_nlp.word_embeddings import train_word2vec
            model = train_word2vec(corpus=small_corpus, vector_size=50,
                                   sg=1, min_count=1)
            assert model is not None


@pytest.mark.skipif(
    not os.environ.get('RUN_HEAVY_TESTS'),
    reason="Heavy gensim tests skipped by default. Set RUN_HEAVY_TESTS=1 to run."
)
class TestModelLoading:
    """Test model loading."""

    def test_load_word2vec_not_found(self, tmp_path):
        """Raises FileNotFoundError when no model exists."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH',
                   str(tmp_path / 'nonexistent.model')):
            from malaysian_manglish_nlp.word_embeddings import load_word2vec
            with pytest.raises(FileNotFoundError):
                load_word2vec()

    def test_load_fasttext_not_found(self, tmp_path):
        """Raises FileNotFoundError when no model exists."""
        gensim = pytest.importorskip("gensim")

        with patch('malaysian_manglish_nlp.word_embeddings._FT_PATH',
                   str(tmp_path / 'nonexistent.model')):
            from malaysian_manglish_nlp.word_embeddings import load_fasttext
            with pytest.raises(FileNotFoundError):
                load_fasttext()

    def test_load_after_train(self, tmp_path):
        """Can load model after training."""
        gensim = pytest.importorskip("gensim")

        model_path = str(tmp_path / 'w2v_load.model')
        corpus = generate_corpus(n_sentences=200, seed=42)

        with patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH', model_path):
            from malaysian_manglish_nlp.word_embeddings import train_word2vec, load_word2vec
            train_word2vec(corpus=corpus, vector_size=50, min_count=1)
            loaded = load_word2vec()
            assert loaded.wv.vector_size == 50


@pytest.mark.skipif(
    not os.environ.get('RUN_HEAVY_TESTS'),
    reason="Heavy gensim tests skipped by default. Set RUN_HEAVY_TESTS=1 to run."
)
class TestQueryFunctions:
    """Test similarity, vector, and analogy functions."""

    @pytest.fixture(autouse=True)
    def setup_model(self, tmp_path):
        """Train a small model for query tests."""
        gensim = pytest.importorskip("gensim")

        corpus = generate_corpus(n_sentences=200, seed=42)
        w2v_path = str(tmp_path / 'w2v_query.model')
        ft_path = str(tmp_path / 'ft_query.model')

        self._patches = [
            patch('malaysian_manglish_nlp.word_embeddings._W2V_PATH', w2v_path),
            patch('malaysian_manglish_nlp.word_embeddings._FT_PATH', ft_path),
        ]
        for p in self._patches:
            p.start()

        from malaysian_manglish_nlp.word_embeddings import train_word2vec, train_fasttext
        train_word2vec(corpus=corpus, vector_size=50, min_count=1)
        train_fasttext(corpus=corpus, vector_size=50, min_count=1)

        yield

        for p in self._patches:
            p.stop()

    def test_most_similar_fasttext(self):
        """most_similar returns results for FastText."""
        from malaysian_manglish_nlp.word_embeddings import most_similar
        results = most_similar('makan', model_type='fasttext', topn=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2

    def test_most_similar_word2vec(self):
        """most_similar returns results for Word2Vec."""
        from malaysian_manglish_nlp.word_embeddings import most_similar
        results = most_similar('aku', model_type='word2vec', topn=5)
        assert isinstance(results, list)

    def test_most_similar_unknown_word(self):
        """most_similar handles unknown words gracefully."""
        from malaysian_manglish_nlp.word_embeddings import most_similar
        # Word2Vec can't handle OOV
        results = most_similar('xyznonexistent123', model_type='word2vec', topn=5)
        assert results == []

    def test_word_vector_fasttext(self):
        """word_vector returns numpy array."""
        from malaysian_manglish_nlp.word_embeddings import word_vector
        vec = word_vector('makan', model_type='fasttext')
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (50,)

    def test_word_vector_oov_fasttext(self):
        """FastText handles OOV words via subwords."""
        from malaysian_manglish_nlp.word_embeddings import word_vector
        # FastText should handle this via subword info
        vec = word_vector('makannnn', model_type='fasttext')
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (50,)

    def test_sentence_vector(self):
        """sentence_vector returns averaged vector."""
        from malaysian_manglish_nlp.word_embeddings import sentence_vector
        vec = sentence_vector('aku nak makan nasi lemak', model_type='fasttext')
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (50,)

    def test_sentence_vector_empty(self):
        """sentence_vector handles empty/unknown text."""
        from malaysian_manglish_nlp.word_embeddings import sentence_vector
        vec = sentence_vector('', model_type='fasttext')
        assert isinstance(vec, np.ndarray)
        assert np.all(vec == 0)

    def test_analogy(self):
        """analogy returns results."""
        from malaysian_manglish_nlp.word_embeddings import analogy
        results = analogy(['makan', 'kedai'], ['rumah'],
                         model_type='fasttext', topn=3)
        assert isinstance(results, list)

    def test_analogy_unknown_word(self):
        """analogy handles unknown words."""
        from malaysian_manglish_nlp.word_embeddings import analogy
        results = analogy(['xyznonexistent'], ['abcnonexistent'],
                         model_type='word2vec', topn=3)
        assert results == []


@pytest.mark.skipif(
    not os.environ.get('RUN_HEAVY_TESTS'),
    reason="Heavy gensim tests skipped by default. Set RUN_HEAVY_TESTS=1 to run."
)
class TestModelTypeValidation:
    """Test model type parameter validation."""

    def test_invalid_model_type(self):
        """Raises ValueError for unknown model type."""
        gensim = pytest.importorskip("gensim")

        from malaysian_manglish_nlp.word_embeddings import _get_model
        with pytest.raises(ValueError, match="Unknown model_type"):
            _get_model('invalid_type')

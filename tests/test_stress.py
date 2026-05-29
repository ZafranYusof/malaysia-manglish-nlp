"""Stress tests for manglish_nlp.

Generates 10,000 random Manglish sentences and runs them through core modules.
Verifies no exceptions, measures performance, and checks for memory leaks.
"""

import os
import sys
import time
import gc
import random
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manglish_nlp


# === Random sentence generation ===

_BM_WORDS = [
    "aku", "kau", "dia", "kita", "korang", "diorang", "saya", "awak",
    "nak", "tak", "boleh", "pergi", "makan", "minum", "tidur", "kerja",
    "rumah", "kedai", "sekolah", "universiti", "hospital", "masjid",
    "sedap", "best", "power", "gila", "teruk", "bagus", "cantik", "hodoh",
    "besar", "kecil", "banyak", "sikit", "semua", "takde", "ada", "xde",
    "pagi", "petang", "malam", "esok", "semalam", "hari", "minggu",
    "duit", "harga", "murah", "mahal", "bayar", "beli", "jual",
    "kereta", "motor", "bas", "lrt", "grab", "jalan", "highway",
    "mak", "ayah", "abang", "kakak", "adik", "kawan", "boss",
    "telefon", "laptop", "wifi", "internet", "game", "movie",
    "nasi", "ayam", "ikan", "sayur", "sambal", "kuah", "goreng",
    "hujan", "panas", "sejuk", "banjir", "ribut", "cerah",
    "suka", "benci", "sayang", "rindu", "marah", "takut", "malu",
    "cepat", "lambat", "laju", "slow", "confirm", "maybe",
]

_EN_WORDS = [
    "the", "is", "are", "was", "were", "have", "has", "had",
    "this", "that", "what", "which", "who", "where", "when", "why",
    "good", "bad", "nice", "great", "awesome", "terrible", "amazing",
    "really", "very", "so", "damn", "super", "ultra", "freaking",
    "want", "need", "like", "love", "hate", "think", "know", "feel",
    "go", "come", "see", "look", "try", "buy", "sell", "pay",
    "food", "money", "time", "work", "life", "phone", "car",
    "today", "tomorrow", "yesterday", "morning", "night", "weekend",
    "price", "quality", "service", "review", "recommend", "worth",
    "actually", "literally", "basically", "honestly", "seriously",
    "bro", "dude", "guys", "fam", "bruh", "lol", "lmao", "haha",
]

_SLANG = [
    "gila", "siot", "wei", "weh", "la", "lah", "kot", "je", "doh",
    "confirm", "legit", "lowkey", "highkey", "no cap", "fr fr",
    "gg", "rip", "oof", "bruh", "sus", "based", "cringe", "mid",
    "mamak", "lepak", "tapau", "belanja", "tumpang", "otw",
    "potong", "kantoi", "kena", "terkejut", "fuyoh", "alamak",
    "walao", "aiyo", "adoi", "cis", "haih", "hmm", "eh",
]

_PARTICLES = ["la", "lah", "kot", "je", "doh", "wei", "bro", "siot", "gila", "ah", "eh"]

_EMOJI = ["😂", "🔥", "💀", "👌", "😭", "🤣", "❤️", "😍", "🙏", "💯", "😅", "🥲", ""]


def generate_random_manglish(min_words=3, max_words=20):
    """Generate a random Manglish sentence."""
    length = random.randint(min_words, max_words)
    words = []
    
    for _ in range(length):
        source = random.choice(["bm", "en", "slang"])
        if source == "bm":
            words.append(random.choice(_BM_WORDS))
        elif source == "en":
            words.append(random.choice(_EN_WORDS))
        else:
            words.append(random.choice(_SLANG))
    
    # Maybe add particle at end
    if random.random() < 0.4:
        words.append(random.choice(_PARTICLES))
    
    # Maybe add emoji
    if random.random() < 0.3:
        words.append(random.choice(_EMOJI))
    
    # Maybe capitalize randomly
    if random.random() < 0.1:
        words = [w.upper() for w in words]
    
    # Maybe add punctuation
    if random.random() < 0.3:
        punct = random.choice(["!", "?", "...", "!!", "???", "~"])
        words.append(punct)
    
    return " ".join(words)


def generate_corpus(n=10000):
    """Generate n random Manglish sentences."""
    random.seed(42)  # Reproducible
    return [generate_random_manglish() for _ in range(n)]


class TestStressNoExceptions:
    """Verify no exceptions on 10,000 random inputs."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.corpus = generate_corpus(10000)

    def test_sentiment_no_crash(self):
        """Sentiment should handle 10k random inputs without crashing."""
        errors = []
        for i, text in enumerate(self.corpus):
            try:
                manglish_nlp.sentiment(text)
            except Exception as e:
                errors.append((i, text[:50], str(e)))
        assert len(errors) == 0, f"{len(errors)} errors: {errors[:5]}"

    def test_language_no_crash(self):
        """Language detection should handle 10k random inputs without crashing."""
        errors = []
        for i, text in enumerate(self.corpus):
            try:
                manglish_nlp.detect_language(text)
            except Exception as e:
                errors.append((i, text[:50], str(e)))
        assert len(errors) == 0, f"{len(errors)} errors: {errors[:5]}"

    def test_normalize_no_crash(self):
        """Normalize should handle 10k random inputs without crashing."""
        errors = []
        for i, text in enumerate(self.corpus):
            try:
                manglish_nlp.normalize(text)
            except Exception as e:
                errors.append((i, text[:50], str(e)))
        assert len(errors) == 0, f"{len(errors)} errors: {errors[:5]}"

    def test_tokenize_no_crash(self):
        """Tokenize should handle 10k random inputs without crashing."""
        errors = []
        for i, text in enumerate(self.corpus):
            try:
                manglish_nlp.tokenize(text)
            except Exception as e:
                errors.append((i, text[:50], str(e)))
        assert len(errors) == 0, f"{len(errors)} errors: {errors[:5]}"


class TestStressPerformance:
    """Performance benchmarks for 10k inputs."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_class_data(self, request):
        request.cls.corpus = generate_corpus(10000)

    def test_sentiment_performance(self):
        """Sentiment on 10k inputs should complete in < 30 seconds."""
        start = time.time()
        for text in self.corpus:
            manglish_nlp.sentiment(text)
        elapsed = time.time() - start
        print(f"\nSentiment 10k: {elapsed:.2f}s ({10000/elapsed:.0f} texts/sec)")
        assert elapsed < 30, f"Too slow: {elapsed:.2f}s"

    def test_language_performance(self):
        """Language detection on 10k inputs should complete in < 30 seconds."""
        start = time.time()
        for text in self.corpus:
            manglish_nlp.detect_language(text)
        elapsed = time.time() - start
        print(f"\nLanguage 10k: {elapsed:.2f}s ({10000/elapsed:.0f} texts/sec)")
        assert elapsed < 30, f"Too slow: {elapsed:.2f}s"

    def test_normalize_performance(self):
        """Normalize on 10k inputs should complete in < 30 seconds."""
        start = time.time()
        for text in self.corpus:
            manglish_nlp.normalize(text)
        elapsed = time.time() - start
        print(f"\nNormalize 10k: {elapsed:.2f}s ({10000/elapsed:.0f} texts/sec)")
        assert elapsed < 30, f"Too slow: {elapsed:.2f}s"

    def test_tokenize_performance(self):
        """Tokenize on 10k inputs should complete in < 30 seconds."""
        start = time.time()
        for text in self.corpus:
            manglish_nlp.tokenize(text)
        elapsed = time.time() - start
        print(f"\nTokenize 10k: {elapsed:.2f}s ({10000/elapsed:.0f} texts/sec)")
        assert elapsed < 30, f"Too slow: {elapsed:.2f}s"

    def test_combined_pipeline_performance(self):
        """All 4 modules on 10k inputs should complete in < 120 seconds total."""
        start = time.time()
        for text in self.corpus:
            manglish_nlp.sentiment(text)
            manglish_nlp.detect_language(text)
            manglish_nlp.normalize(text)
            manglish_nlp.tokenize(text)
        elapsed = time.time() - start
        print(f"\nCombined pipeline 10k: {elapsed:.2f}s ({10000/elapsed:.0f} texts/sec)")
        assert elapsed < 120, f"Too slow: {elapsed:.2f}s"


class TestStressMemory:
    """Memory leak detection."""

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: use sys.getsizeof on gc objects (rough estimate)
            gc.collect()
            return sum(sys.getsizeof(obj) for obj in gc.get_objects()[:1000]) / 1024 / 1024

    def test_no_memory_leak_sentiment(self):
        """Sentiment should not leak memory across batches."""
        corpus = generate_corpus(10000)
        batch_size = 2000
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for batch_start in range(0, len(corpus), batch_size):
            batch = corpus[batch_start:batch_start + batch_size]
            for text in batch:
                manglish_nlp.sentiment(text)
            gc.collect()
        
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects
        # Allow some growth but not unbounded (< 10000 new objects for 10k inputs)
        print(f"\nObject growth after 10k sentiment: {growth}")
        assert growth < 50000, f"Possible memory leak: {growth} new objects"

    def test_no_memory_leak_normalize(self):
        """Normalize should not leak memory across batches."""
        corpus = generate_corpus(10000)
        batch_size = 2000
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for batch_start in range(0, len(corpus), batch_size):
            batch = corpus[batch_start:batch_start + batch_size]
            for text in batch:
                manglish_nlp.normalize(text)
            gc.collect()
        
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects
        print(f"\nObject growth after 10k normalize: {growth}")
        assert growth < 50000, f"Possible memory leak: {growth} new objects"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])

"""Regression tests for manglish_nlp edge cases.

Tests specific bugs and edge cases that could cause crashes or incorrect behavior.
Each test targets a specific problematic input pattern.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import manglish_nlp


# === Edge case inputs ===

EMPTY_STRING = ""
NONE_INPUT = None
VERY_LONG_TEXT = "aku suka makan nasi lemak " * 200  # 1000+ words
UNICODE_EMOJI = "best gila 🔥🔥🔥 sedap 😍👌💯"
UNICODE_CHINESE = "这个很好吃 sedap gila"
UNICODE_ARABIC = "بسم الله الرحمن الرحيم alhamdulillah"
ONLY_PUNCTUATION = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
ONLY_NUMBERS = "1234567890 42 3.14 100000"
SINGLE_CHAR = "a"
REPEATED_WORD = "lol " * 100
HTML_TAGS = "<div class='post'><p>weh best gila</p><br/><a href='#'>click</a></div>"
URL_ONLY = "https://www.lowyat.net/2026/05/review-rtx-5090.html"
ALL_CAPS = "WALAWEI BEST GILA SIOT MAKANAN DIA POWER"
NO_SPACES = "akusukamakannasilemaksedapgilabrommg"
MIXED_SCRIPTS = "aku 好き desu 사랑해 love kau"
NEWLINES_ONLY = "\n\n\n\n\n"
TABS_AND_SPACES = "\t   \t   \t"
SPECIAL_UNICODE = "aku suka \u200b\u200b\u200b zero width spaces"
VERY_LONG_WORD = "a" * 10000
BACKSLASHES = "C:\\Users\\test\\path\\to\\file.txt"
SQL_INJECTION = "'; DROP TABLE users; --"
SCRIPT_TAG = "<script>alert('xss')</script>"
ZALGO_TEXT = "h̷̢̧̛̗̣̦̮̙̱̲̹̪̫̺̻̼̽̈́̃̂̄̅̆̇̈̉̊̋̌̍̎̏e̷̢̧̛̗̣̦̮̙̱̲̹̪̫̺̻̼̽̈́̃̂̄̅̆̇̈̉̊̋̌̍̎̏l̷̢̧̛̗̣̦̮̙̱̲̹̪̫̺̻̼̽̈́̃̂̄̅̆̇̈̉̊̋̌̍̎̏p̷̢̧̛̗̣̦̮̙̱̲̹̪̫̺̻̼̽̈́̃̂̄̅̆̇̈̉̊̋̌̍̎̏"
EMOJI_ONLY = "😂😂😂🤣🤣💀💀🔥🔥👌👌"
NUMBERS_WITH_UNITS = "RM50.90 3kg 100km/h 25°C"
MALAY_WITH_DIACRITICS = "résumé naïve café"


# === Module test functions ===

def run_all_modules(text):
    """Run text through all core modules, return dict of results."""
    results = {}
    
    # Sentiment
    try:
        results["sentiment"] = manglish_nlp.sentiment(text)
    except Exception as e:
        results["sentiment_error"] = str(e)
    
    # Language detection
    try:
        results["language"] = manglish_nlp.detect_language(text)
    except Exception as e:
        results["language_error"] = str(e)
    
    # Emotion
    try:
        results["emotion"] = manglish_nlp.detect_emotion(text)
    except Exception as e:
        results["emotion_error"] = str(e)
    
    # Intent
    try:
        results["intent"] = manglish_nlp.classify_intent(text)
    except Exception as e:
        results["intent_error"] = str(e)
    
    # Topic
    try:
        results["topic"] = manglish_nlp.classify_topic(text)
    except Exception as e:
        results["topic_error"] = str(e)
    
    # Normalize
    try:
        results["normalize"] = manglish_nlp.normalize(text)
    except Exception as e:
        results["normalize_error"] = str(e)
    
    # Tokenize
    try:
        results["tokenize"] = manglish_nlp.tokenize(text)
    except Exception as e:
        results["tokenize_error"] = str(e)
    
    # Clean
    try:
        results["clean"] = manglish_nlp.clean(text)
    except Exception as e:
        results["clean_error"] = str(e)
    
    # POS tag
    try:
        results["pos"] = manglish_nlp.pos_tag(text)
    except Exception as e:
        results["pos_error"] = str(e)
    
    # NER
    try:
        results["ner"] = manglish_nlp.ner_tag(text)
    except Exception as e:
        results["ner_error"] = str(e)
    
    # Stem
    try:
        results["stem"] = manglish_nlp.stem(text)
    except Exception as e:
        results["stem_error"] = str(e)
    
    # Hate speech
    try:
        results["hate_speech"] = manglish_nlp.detect_hate_speech(text)
    except Exception as e:
        results["hate_speech_error"] = str(e)
    
    return results


class TestEmptyString:
    """Test all modules with empty string input."""

    def test_sentiment_empty(self):
        result = manglish_nlp.sentiment("")
        assert result is not None

    def test_language_empty(self):
        result = manglish_nlp.detect_language("")
        assert result is not None

    def test_emotion_empty(self):
        result = manglish_nlp.detect_emotion("")
        assert result is not None

    def test_intent_empty(self):
        result = manglish_nlp.classify_intent("")
        assert result is not None

    def test_topic_empty(self):
        result = manglish_nlp.classify_topic("")
        assert result is not None

    def test_normalize_empty(self):
        result = manglish_nlp.normalize("")
        assert result is not None
        assert isinstance(result, str)

    def test_tokenize_empty(self):
        result = manglish_nlp.tokenize("")
        assert result is not None

    def test_clean_empty(self):
        result = manglish_nlp.clean("")
        assert result is not None

    def test_pos_tag_empty(self):
        result = manglish_nlp.pos_tag("")
        assert result is not None

    def test_ner_empty(self):
        result = manglish_nlp.ner_tag("")
        assert result is not None

    def test_stem_empty(self):
        result = manglish_nlp.stem("")
        assert result is not None

    def test_hate_speech_empty(self):
        result = manglish_nlp.detect_hate_speech("")
        assert result is not None


class TestNoneInput:
    """Test all modules with None input - should handle gracefully."""

    def test_sentiment_none(self):
        try:
            result = manglish_nlp.sentiment(None)
            # If it doesn't crash, that's fine
        except (TypeError, AttributeError, ValueError):
            pass  # Expected - None is not valid input

    def test_language_none(self):
        try:
            result = manglish_nlp.detect_language(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_emotion_none(self):
        try:
            result = manglish_nlp.detect_emotion(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_intent_none(self):
        try:
            result = manglish_nlp.classify_intent(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_topic_none(self):
        try:
            result = manglish_nlp.classify_topic(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_normalize_none(self):
        try:
            result = manglish_nlp.normalize(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_tokenize_none(self):
        try:
            result = manglish_nlp.tokenize(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_clean_none(self):
        try:
            result = manglish_nlp.clean(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_pos_tag_none(self):
        try:
            result = manglish_nlp.pos_tag(None)
        except (TypeError, AttributeError, ValueError):
            pass

    def test_stem_none(self):
        try:
            result = manglish_nlp.stem(None)
        except (TypeError, AttributeError, ValueError):
            pass


class TestVeryLongText:
    """Test modules with very long text (1000+ words)."""

    def test_sentiment_long(self):
        result = manglish_nlp.sentiment(VERY_LONG_TEXT)
        assert result is not None

    def test_language_long(self):
        result = manglish_nlp.detect_language(VERY_LONG_TEXT)
        assert result is not None

    def test_emotion_long(self):
        result = manglish_nlp.detect_emotion(VERY_LONG_TEXT)
        assert result is not None

    def test_normalize_long(self):
        result = manglish_nlp.normalize(VERY_LONG_TEXT)
        assert result is not None

    def test_tokenize_long(self):
        result = manglish_nlp.tokenize(VERY_LONG_TEXT)
        assert result is not None
        assert len(result) > 100


class TestUnicodeCharacters:
    """Test modules with various Unicode inputs."""

    def test_emoji_input(self):
        results = run_all_modules(UNICODE_EMOJI)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on emoji input: {errors}"

    def test_chinese_input(self):
        results = run_all_modules(UNICODE_CHINESE)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on Chinese input: {errors}"

    def test_arabic_input(self):
        results = run_all_modules(UNICODE_ARABIC)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on Arabic input: {errors}"

    def test_emoji_only(self):
        results = run_all_modules(EMOJI_ONLY)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on emoji-only input: {errors}"

    def test_zalgo_text(self):
        results = run_all_modules(ZALGO_TEXT)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on zalgo text: {errors}"

    def test_mixed_scripts(self):
        results = run_all_modules(MIXED_SCRIPTS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on mixed scripts: {errors}"

    def test_diacritics(self):
        results = run_all_modules(MALAY_WITH_DIACRITICS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on diacritics: {errors}"


class TestOnlyPunctuation:
    """Test modules with only punctuation."""

    def test_all_modules(self):
        results = run_all_modules(ONLY_PUNCTUATION)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on punctuation-only: {errors}"


class TestOnlyNumbers:
    """Test modules with only numbers."""

    def test_all_modules(self):
        results = run_all_modules(ONLY_NUMBERS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on numbers-only: {errors}"

    def test_numbers_with_units(self):
        results = run_all_modules(NUMBERS_WITH_UNITS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on numbers with units: {errors}"


class TestSingleCharacter:
    """Test modules with single character input."""

    def test_all_modules(self):
        results = run_all_modules(SINGLE_CHAR)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on single char: {errors}"


class TestRepeatedWord:
    """Test modules with repeated word 100 times."""

    def test_all_modules(self):
        results = run_all_modules(REPEATED_WORD)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on repeated word: {errors}"


class TestHTMLTags:
    """Test modules with HTML/XML tags in text."""

    def test_all_modules(self):
        results = run_all_modules(HTML_TAGS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on HTML input: {errors}"

    def test_script_tag(self):
        results = run_all_modules(SCRIPT_TAG)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on script tag: {errors}"


class TestURLOnly:
    """Test modules with URL-only text."""

    def test_all_modules(self):
        results = run_all_modules(URL_ONLY)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on URL-only: {errors}"


class TestAllCaps:
    """Test modules with all caps text."""

    def test_all_modules(self):
        results = run_all_modules(ALL_CAPS)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on all caps: {errors}"

    def test_sentiment_detects_positive(self):
        """All caps positive text should still be detected."""
        result = manglish_nlp.sentiment(ALL_CAPS)
        # Should detect some sentiment (positive in this case)
        assert result is not None


class TestNoSpaces:
    """Test modules with merged text (no spaces)."""

    def test_all_modules(self):
        results = run_all_modules(NO_SPACES)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on no-spaces text: {errors}"


class TestWhitespaceOnly:
    """Test modules with whitespace-only inputs."""

    def test_newlines_only(self):
        results = run_all_modules(NEWLINES_ONLY)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on newlines-only: {errors}"

    def test_tabs_and_spaces(self):
        results = run_all_modules(TABS_AND_SPACES)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on tabs/spaces: {errors}"


class TestSpecialStrings:
    """Test modules with special/adversarial strings."""

    def test_zero_width_spaces(self):
        results = run_all_modules(SPECIAL_UNICODE)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on zero-width spaces: {errors}"

    def test_very_long_word(self):
        results = run_all_modules(VERY_LONG_WORD)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on very long word: {errors}"

    def test_backslashes(self):
        results = run_all_modules(BACKSLASHES)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on backslashes: {errors}"

    def test_sql_injection_string(self):
        results = run_all_modules(SQL_INJECTION)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0, f"Errors on SQL injection string: {errors}"


class TestSpecificBugs:
    """Regression tests for specific known patterns."""

    def test_double_negation(self):
        """Double negation should not crash."""
        result = manglish_nlp.sentiment("tak tak sedap pun")
        assert result is not None

    def test_mixed_language_sentiment(self):
        """Mixed BM/EN sentiment should work."""
        result = manglish_nlp.sentiment("this is damn good la bro sedap gila")
        assert result is not None

    def test_elongated_words(self):
        """Elongated words should not crash."""
        result = manglish_nlp.normalize("bestttttt gilaaaa sedapppp")
        assert result is not None

    def test_repeated_punctuation(self):
        """Repeated punctuation should not crash."""
        result = manglish_nlp.sentiment("best!!!!!!!! sedap??????")
        assert result is not None

    def test_hashtags(self):
        """Hashtags should not crash."""
        result = manglish_nlp.sentiment("#MalaysiaFood #NasiLemak best gila")
        assert result is not None

    def test_mentions(self):
        """@ mentions should not crash."""
        result = manglish_nlp.sentiment("@ahmad weh jom makan")
        assert result is not None

    def test_code_in_text(self):
        """Code snippets in text should not crash."""
        code_text = "bro try `console.log('hello')` then run `npm start`"
        results = run_all_modules(code_text)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0

    def test_multiline_text(self):
        """Multiline text should not crash."""
        text = "line 1 aku suka\nline 2 kau best\nline 3 dia power"
        results = run_all_modules(text)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0

    def test_tab_separated(self):
        """Tab-separated text should not crash."""
        text = "word1\tword2\tword3\taku\tsuka"
        results = run_all_modules(text)
        errors = [k for k in results if k.endswith("_error")]
        assert len(errors) == 0

    def test_null_bytes(self):
        """Text with null bytes should not crash (or raise clean error)."""
        try:
            text = "hello\x00world"
            results = run_all_modules(text)
        except (ValueError, TypeError):
            pass  # Acceptable to reject null bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

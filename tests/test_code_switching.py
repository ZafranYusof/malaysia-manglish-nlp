"""
Tests for code_switching module.
"""

import pytest
from manglish_nlp.code_switching import (
    detect_switches,
    switch_points,
    switch_ratio,
    dominant_language,
    segment_by_language,
    switch_matrix,
    classify_switch_type,
)


# === detect_switches tests ===

class TestDetectSwitches:
    def test_pure_english(self):
        result = detect_switches("I want to go home")
        en_tokens = [r for r in result if r["language"] == "en"]
        assert len(en_tokens) >= 4

    def test_pure_malay(self):
        result = detect_switches("Aku nak pergi makan")
        ms_tokens = [r for r in result if r["language"] == "ms"]
        assert len(ms_tokens) >= 3

    def test_mixed_sentence(self):
        result = detect_switches("Aku nak go makan lunch")
        languages = set(r["language"] for r in result)
        assert "en" in languages
        assert "ms" in languages

    def test_position_tracking(self):
        text = "Hello dunia"
        result = detect_switches(text)
        assert result[0]["position"] == (0, 5)
        assert result[0]["token"] == "Hello"

    def test_particles_detected_as_malay(self):
        result = detect_switches("That's nice la")
        particle = [r for r in result if r["token"].lower() == "la"]
        assert len(particle) == 1
        assert particle[0]["language"] == "ms"

    def test_borrowed_words_are_mixed(self):
        result = detect_switches("Aku guna computer")
        comp = [r for r in result if r["token"].lower() == "computer"]
        assert comp[0]["language"] == "mixed"

    def test_empty_string(self):
        result = detect_switches("")
        assert result == []

    def test_single_word_english(self):
        result = detect_switches("hello")
        assert result[0]["language"] == "en"

    def test_single_word_malay(self):
        result = detect_switches("makan")
        assert result[0]["language"] == "ms"

    def test_malay_slang_detected(self):
        result = detect_switches("camtu je lah")
        ms_tokens = [r for r in result if r["language"] == "ms"]
        assert len(ms_tokens) >= 2


# === switch_points tests ===

class TestSwitchPoints:
    def test_no_switch_english(self):
        points = switch_points("I want to go home now")
        assert points == []

    def test_no_switch_malay(self):
        points = switch_points("Aku nak pergi makan nasi")
        assert points == []

    def test_single_switch(self):
        points = switch_points("Aku nak go home")
        assert len(points) >= 1

    def test_multiple_switches(self):
        points = switch_points("Aku want makan lunch then balik")
        assert len(points) >= 2

    def test_empty_text(self):
        points = switch_points("")
        assert points == []


# === switch_ratio tests ===

class TestSwitchRatio:
    def test_monolingual_zero(self):
        ratio = switch_ratio("I want to go home")
        assert ratio == 0.0

    def test_fully_alternating(self):
        # Every other word switches
        ratio = switch_ratio("Aku want makan now pergi home")
        assert ratio > 0.5

    def test_empty_text(self):
        ratio = switch_ratio("")
        assert ratio == 0.0

    def test_single_word(self):
        ratio = switch_ratio("hello")
        assert ratio == 0.0

    def test_ratio_between_zero_and_one(self):
        ratio = switch_ratio("Aku nak pergi shopping dengan kawan")
        assert 0.0 <= ratio <= 1.0


# === dominant_language tests ===

class TestDominantLanguage:
    def test_english_dominant(self):
        result = dominant_language("I want to go to the store and buy something")
        assert result == "en"

    def test_malay_dominant(self):
        result = dominant_language("Aku nak pergi kedai beli barang untuk makan")
        assert result == "ms"

    def test_mixed_balanced(self):
        result = dominant_language("Aku want pergi buy makan sell")
        assert result == "mixed"

    def test_empty_text(self):
        result = dominant_language("")
        assert result == "mixed"

    def test_pure_english(self):
        result = dominant_language("The quick brown fox jumps over the lazy dog")
        assert result == "en"

    def test_pure_malay(self):
        result = dominant_language("Dia pergi ke kedai untuk membeli makanan")
        assert result == "ms"


# === segment_by_language tests ===

class TestSegmentByLanguage:
    def test_single_language(self):
        segments = segment_by_language("I want to go home")
        assert len(segments) >= 1
        assert segments[0]["language"] == "en"

    def test_two_segments(self):
        segments = segment_by_language("Aku nak go home")
        assert len(segments) >= 2

    def test_segment_text_content(self):
        text = "Hello dunia"
        segments = segment_by_language(text)
        # Should have at least the text portions
        all_text = "".join(s["text"] for s in segments)
        assert "Hello" in all_text or "dunia" in all_text

    def test_segment_positions(self):
        segments = segment_by_language("Aku nak go home")
        for seg in segments:
            assert seg["start"] >= 0
            assert seg["end"] > seg["start"]
            assert "text" in seg
            assert "language" in seg

    def test_empty_text(self):
        segments = segment_by_language("")
        assert segments == []


# === switch_matrix tests ===

class TestSwitchMatrix:
    def test_no_switches(self):
        matrix = switch_matrix("I want to go home")
        assert matrix["en->ms"] == 0
        assert matrix["ms->en"] == 0

    def test_en_to_ms_switch(self):
        matrix = switch_matrix("I want nak makan")
        assert matrix["en->ms"] >= 1

    def test_ms_to_en_switch(self):
        matrix = switch_matrix("Aku nak go home")
        assert matrix["ms->en"] >= 1

    def test_bidirectional_switches(self):
        matrix = switch_matrix("Aku want makan lunch then balik rumah")
        total = matrix["en->ms"] + matrix["ms->en"]
        assert total >= 2

    def test_empty_text(self):
        matrix = switch_matrix("")
        assert matrix == {"en->ms": 0, "ms->en": 0}


# === classify_switch_type tests ===

class TestClassifySwitchType:
    def test_no_switching(self):
        result = classify_switch_type("I want to go home now please")
        assert result == "none"

    def test_no_switching_malay(self):
        result = classify_switch_type("Aku nak pergi makan nasi goreng")
        assert result == "none"

    def test_tag_switching(self):
        result = classify_switch_type("That was really nice la")
        assert result == "tag-switching"

    def test_tag_switching_kan(self):
        result = classify_switch_type("You should come to the party kan")
        assert result == "tag-switching"

    def test_intra_sentential(self):
        result = classify_switch_type("Aku nak go makan lunch with kawan")
        assert result == "intra-sentential"

    def test_inter_sentential(self):
        result = classify_switch_type("I went there yesterday. Lepas tu aku balik rumah.")
        assert result == "inter-sentential"

    def test_empty_text(self):
        result = classify_switch_type("")
        assert result == "none"

    def test_dialect_text(self):
        # Dialect with heavy Malay
        result = classify_switch_type("Ambe nak gi make nasi kerabu")
        # Should be none or detected as ms-dominant
        assert result in ("none", "intra-sentential", "tag-switching")


# === Integration / Edge case tests ===

class TestEdgeCases:
    def test_numbers_only(self):
        result = detect_switches("123 456")
        assert isinstance(result, list)

    def test_mixed_with_punctuation(self):
        result = detect_switches("Aku nak go, tapi busy la!")
        assert len(result) > 0

    def test_repeated_switches(self):
        ratio = switch_ratio("Aku go dia come kita leave mereka stay")
        assert ratio > 0.3

    def test_all_particles(self):
        result = dominant_language("la kan kot je lah")
        assert result == "ms"

    def test_long_english_with_one_malay(self):
        text = "I really want to go to the store and buy some groceries tapi malas"
        result = classify_switch_type(text)
        assert result in ("intra-sentential", "tag-switching")

    def test_malay_affixed_words(self):
        result = detect_switches("Dia sedang membaca buku")
        membaca = [r for r in result if r["token"] == "membaca"]
        assert membaca[0]["language"] == "ms"

    def test_english_suffixed_words(self):
        result = detect_switches("She is running quickly")
        running = [r for r in result if r["token"] == "running"]
        assert running[0]["language"] == "en"

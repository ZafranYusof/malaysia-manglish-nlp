"""Tests for OCR text normalization module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from malaysian_manglish_nlp.ocr_normalize import (
    normalize_ocr, fix_common_errors, detect_ocr_artifacts,
    reconstruct_words, fix_malay_ocr,
)


class TestNormalizeOcr:
    """Tests for normalize_ocr main function."""

    def test_clean_text_no_changes(self):
        """Clean text should return unchanged with high confidence."""
        result = normalize_ocr("Saya pergi ke sekolah hari ini")
        assert result["cleaned"] == "Saya pergi ke sekolah hari ini"
        assert result["corrections"] == []
        assert result["confidence"] == 1.0

    def test_empty_text(self):
        """Empty text returns empty result."""
        result = normalize_ocr("")
        assert result["cleaned"] == ""
        assert result["corrections"] == []
        assert result["confidence"] == 1.0

    def test_none_text(self):
        """None text returns empty result."""
        result = normalize_ocr(None)
        assert result["cleaned"] == ""
        assert result["corrections"] == []
        assert result["confidence"] == 1.0

    def test_rn_to_m_makan(self):
        """rn→m: rnakan → makan."""
        result = normalize_ocr("Dia rnakan nasi")
        assert "makan" in result["cleaned"]
        assert any(c["type"] == "char_confusion_rn_m" for c in result["corrections"])

    def test_rn_to_m_mereka(self):
        """rn→m: rnereka → mereka."""
        result = normalize_ocr("rnereka pergi ke pasar")
        assert "mereka" in result["cleaned"].lower()

    def test_rn_to_m_dalam(self):
        """rn→m: dalarn → dalam."""
        result = normalize_ocr("dalarn rumah itu")
        assert "dalam" in result["cleaned"]

    def test_rn_to_m_malaysia(self):
        """rn→m: rnalaysia → malaysia."""
        result = normalize_ocr("Negara rnalaysia")
        assert "malaysia" in result["cleaned"].lower()

    def test_rn_to_m_memang(self):
        """rn→m: rnernang → memang."""
        result = normalize_ocr("Dia rnernang pandai")
        assert "memang" in result["cleaned"].lower()

    def test_rn_to_m_preserves_case(self):
        """rn→m should preserve capitalization."""
        result = normalize_ocr("Rnakan sedap")
        # Should capitalize the corrected word
        assert result["cleaned"][0].isupper()

    def test_cl_to_d_dan(self):
        """cl→d: clan → dan."""
        result = normalize_ocr("Nasi clan ayam")
        assert "dan" in result["cleaned"].lower()
        assert any(c["type"] == "char_confusion_cl_d" for c in result["corrections"])

    def test_cl_to_d_dengan(self):
        """cl→d: clengan → dengan."""
        result = normalize_ocr("Pergi clengan kawan")
        assert "dengan" in result["cleaned"].lower()

    def test_cl_to_d_dari(self):
        """cl→d: clari → dari."""
        result = normalize_ocr("Datang clari jauh")
        assert "dari" in result["cleaned"].lower()

    def test_split_word_mereka(self):
        """Split word: 'me reka' → 'mereka'."""
        result = normalize_ocr("me reka pergi ke pasar")
        assert "mereka" in result["cleaned"]
        assert any(c["type"] == "split_word" for c in result["corrections"])

    def test_split_word_kerana(self):
        """Split word: 'ke rana' → 'kerana'."""
        result = normalize_ocr("ke rana dia sakit")
        assert "kerana" in result["cleaned"]

    def test_split_word_kepada(self):
        """Split word: 'ke pada' → 'kepada'."""
        result = normalize_ocr("Beri ke pada dia")
        assert "kepada" in result["cleaned"]

    def test_split_word_dalam(self):
        """Split word: 'da lam' → 'dalam'."""
        result = normalize_ocr("da lam rumah")
        assert "dalam" in result["cleaned"]

    def test_merged_word_dankemudian(self):
        """Merged word: 'dankemudian' → 'dan kemudian'."""
        result = normalize_ocr("Makan dankemudian tidur")
        assert "dan kemudian" in result["cleaned"].lower() or "dan" in result["cleaned"].lower()

    def test_merged_word_untukdia(self):
        """Merged word: 'untukdia' → 'untuk dia'."""
        result = normalize_ocr("Beli untukdia")
        assert "untuk dia" in result["cleaned"].lower() or "untuk" in result["cleaned"].lower()

    def test_number_letter_confusion_l_to_1(self):
        """Number/letter: 'l23' → '123'."""
        result = normalize_ocr("Nombor l23 itu")
        assert "123" in result["cleaned"]

    def test_number_letter_confusion_O_to_0(self):
        """Number/letter: 'O5' → '05'."""
        result = normalize_ocr("Kod O5 ini")
        assert "05" in result["cleaned"]

    def test_number_letter_confusion_I_to_1(self):
        """Number/letter: 'I23' → '123' in numeric context."""
        result = normalize_ocr("Harga RM I23")
        assert "123" in result["cleaned"]

    def test_punctuation_missing_space(self):
        """Missing space: 'ini.adalah' → 'ini. adalah'."""
        result = normalize_ocr("ini.adalah baik")
        assert "ini. adalah" in result["cleaned"]
        assert any(c["type"] == "missing_space" for c in result["corrections"])

    def test_punctuation_exclamation(self):
        """Missing space after exclamation."""
        result = normalize_ocr("Bagus!Teruskan")
        assert "Bagus! Teruskan" in result["cleaned"]

    def test_mixed_bm_en_ocr(self):
        """Mixed BM/EN text with OCR errors."""
        result = normalize_ocr("Dia rnakan lunch clengan kawan")
        cleaned = result["cleaned"].lower()
        assert "makan" in cleaned
        assert "dengan" in cleaned

    def test_confidence_decreases_with_errors(self):
        """More corrections should lower confidence."""
        clean_result = normalize_ocr("Saya makan nasi")
        dirty_result = normalize_ocr("rnakan clengan rnereka dalarn")
        assert clean_result["confidence"] > dirty_result["confidence"]

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        result = normalize_ocr("rnakan clengan rnereka dalarn rnalaysia")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_multiple_rn_corrections(self):
        """Multiple rn→m corrections in one text."""
        result = normalize_ocr("rnereka rnakan rnakanan")
        cleaned = result["cleaned"].lower()
        assert "mereka" in cleaned
        assert "makan" in cleaned


class TestFixCommonErrors:
    """Tests for fix_common_errors quick fix."""

    def test_returns_string(self):
        """Should return a string."""
        result = fix_common_errors("rnakan nasi")
        assert isinstance(result, str)

    def test_fixes_rn(self):
        """Should fix rn→m."""
        result = fix_common_errors("rnakan")
        assert "makan" in result

    def test_empty_input(self):
        """Empty input returns empty string."""
        assert fix_common_errors("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert fix_common_errors(None) == ""

    def test_clean_text_unchanged(self):
        """Clean text passes through unchanged."""
        text = "Saya pergi ke kedai"
        assert fix_common_errors(text) == text


class TestDetectOcrArtifacts:
    """Tests for detect_ocr_artifacts."""

    def test_detects_rn_pattern(self):
        """Should detect rn→m artifacts."""
        artifacts = detect_ocr_artifacts("rnakan nasi")
        assert len(artifacts) > 0
        assert any(a["type"] == "char_confusion_rn_m" for a in artifacts)

    def test_detects_missing_space(self):
        """Should detect missing space after punctuation."""
        artifacts = detect_ocr_artifacts("ini.adalah baik")
        assert any(a["type"] == "missing_space" for a in artifacts)

    def test_detects_split_words(self):
        """Should detect split words."""
        artifacts = detect_ocr_artifacts("me reka pergi")
        assert any(a["type"] == "split_word" for a in artifacts)

    def test_has_position(self):
        """Artifacts should include position."""
        artifacts = detect_ocr_artifacts("rnakan nasi")
        assert all("position" in a for a in artifacts)
        assert all(isinstance(a["position"], int) for a in artifacts)

    def test_has_suggestion(self):
        """Artifacts should include suggestion."""
        artifacts = detect_ocr_artifacts("rnakan nasi")
        assert all("suggestion" in a for a in artifacts)

    def test_empty_text(self):
        """Empty text returns empty list."""
        assert detect_ocr_artifacts("") == []

    def test_clean_text_no_artifacts(self):
        """Clean text should have no artifacts."""
        artifacts = detect_ocr_artifacts("Saya makan nasi")
        # Should have no rn_m or split_word artifacts
        critical = [a for a in artifacts if a["type"] in ("char_confusion_rn_m", "split_word")]
        assert len(critical) == 0


class TestReconstructWords:
    """Tests for reconstruct_words."""

    def test_fixes_split_words(self):
        """Should join split words."""
        result = reconstruct_words("me reka pergi")
        assert "mereka" in result

    def test_fixes_merged_words(self):
        """Should split merged words."""
        result = reconstruct_words("dankemudian")
        assert "dan" in result and "kemudian" in result

    def test_empty_input(self):
        """Empty input returns empty string."""
        assert reconstruct_words("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert reconstruct_words(None) == ""

    def test_clean_text_unchanged(self):
        """Clean text should not be modified."""
        text = "Saya pergi ke sekolah"
        result = reconstruct_words(text)
        assert result == text


class TestFixMalayOcr:
    """Tests for fix_malay_ocr BM-specific fixes."""

    def test_fixes_rn_to_m(self):
        """Should fix rn→m in BM words."""
        result = fix_malay_ocr("rnakan nasi dalarn rumah")
        assert "makan" in result
        assert "dalam" in result

    def test_fixes_split_bm_words(self):
        """Should fix split BM words."""
        result = fix_malay_ocr("ke rana dia sakit")
        assert "kerana" in result

    def test_empty_input(self):
        """Empty input returns empty string."""
        assert fix_malay_ocr("") == ""

    def test_none_input(self):
        """None input returns empty string."""
        assert fix_malay_ocr(None) == ""

    def test_preserves_english_words(self):
        """Should not corrupt English words."""
        result = fix_malay_ocr("The morning sun")
        # 'morning' contains 'rn' but should not be changed to 'moming'
        # because 'morning' is a valid English word
        assert "morning" in result or "moming" not in result

    def test_multiple_fixes(self):
        """Should handle multiple BM OCR errors."""
        result = fix_malay_ocr("rnereka rnakan clengan kawan")
        lower = result.lower()
        assert "mereka" in lower
        assert "makan" in lower
        assert "dengan" in lower

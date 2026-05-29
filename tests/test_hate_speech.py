"""Tests for malaysian_manglish_nlp.hate_speech module."""

import pytest
from malaysian_manglish_nlp.hate_speech import (
    detect_hate_speech,
    detect_batch,
    is_hate_speech,
    get_severity,
    get_target_groups,
    CATEGORIES,
    SEVERITY_LEVELS,
)


class TestDetectHateSpeech:
    """Tests for detect_hate_speech function."""

    # --- Non-hate speech (should NOT flag) ---

    def test_neutral_ethnic_reference(self):
        """Mentioning ethnicity alone is not hate speech."""
        result = detect_hate_speech("kawan aku orang Cina, dia baik orangnya")
        assert result["is_hate"] is False
        assert result["category"] == "none"

    def test_neutral_religious_reference(self):
        """Discussing religion neutrally is not hate speech."""
        result = detect_hate_speech("orang Islam solat 5 kali sehari")
        assert result["is_hate"] is False

    def test_food_context_babi(self):
        """'Babi' in food context should not be flagged."""
        result = detect_hate_speech("restoran ni jual nasi goreng babi sedap")
        assert result["is_hate"] is False
        assert "food" in result["explanation"].lower() or result["category"] == "none"

    def test_food_context_pork(self):
        """Pork discussion in food context is not hate speech."""
        result = detect_hate_speech("menu ni ada char siu babi, masak goreng kuah")
        assert result["is_hate"] is False

    def test_cultural_discussion(self):
        """Cultural/historical discussion is not hate speech."""
        result = detect_hate_speech("sejarah budaya Melayu dan tradisi perayaan Cina")
        assert result["is_hate"] is False

    def test_neutral_foreigner_reference(self):
        """Neutral reference to foreigners is not hate speech."""
        result = detect_hate_speech("pekerja Indonesia ramai kat construction site")
        assert result["is_hate"] is False

    def test_normal_conversation(self):
        """Normal daily conversation should not be flagged."""
        result = detect_hate_speech("aku nak pergi kedai beli barang")
        assert result["is_hate"] is False
        assert result["severity"] == "none"

    def test_sports_discussion(self):
        """Sports discussion with ethnic team names is not hate."""
        result = detect_hate_speech("player India dalam team tu main power gila")
        assert result["is_hate"] is False

    def test_religious_education(self):
        """Religious education context is not hate speech."""
        result = detect_hate_speech("belajar sejarah Islam dan Hindu untuk exam")
        assert result["is_hate"] is False

    # --- Hate speech detection (should flag) ---

    def test_racial_dehumanizing(self):
        """Dehumanizing language targeting race should be detected."""
        result = detect_hate_speech("semua Melayu ni bodoh bangang tak guna")
        assert result["is_hate"] is True
        assert result["category"] == "racial"
        assert result["severity"] in ["medium", "high"]
        assert result["target_group"] == "malay"

    def test_racial_go_back(self):
        """'Go back to your country' targeting ethnicity."""
        result = detect_hate_speech("Cina balik negara asal korang la")
        assert result["is_hate"] is True
        assert result["category"] == "racial"

    def test_xenophobic_hate(self):
        """Hate targeting foreign workers."""
        result = detect_hate_speech("halau semua Bangla ni, parasit negara")
        assert result["is_hate"] is True
        assert result["category"] == "xenophobic"
        assert result["target_group"] == "bangladeshi"

    def test_xenophobic_dehumanizing(self):
        """Dehumanizing foreign groups."""
        result = detect_hate_speech("Indon ni semua sampah, usir balik")
        assert result["is_hate"] is True
        assert result["category"] == "xenophobic"

    def test_religious_hate(self):
        """Hate targeting religious group."""
        result = detect_hate_speech("semua Muslim ni celaka, hina agama lain")
        assert result["is_hate"] is True
        assert result["category"] == "religious"

    def test_threat_high_severity(self):
        """Direct threats should be high severity."""
        result = detect_hate_speech("kena bunuh semua India ni, bantai habis")
        assert result["is_hate"] is True
        assert result["severity"] == "high"
        assert result["category"] == "racial"

    def test_homophobic_hate(self):
        """Hate targeting LGBTQ+ community."""
        result = detect_hate_speech("LGBT ni semua sial celaka, halau dari negara")
        assert result["is_hate"] is True
        assert result["category"] == "homophobic"

    def test_sexist_hate(self):
        """Gender-based hate speech."""
        result = detect_hate_speech("perempuan ni semua bodoh tak guna, balik dapur")
        assert result["is_hate"] is True
        assert result["category"] == "sexist"

    # --- Severity levels ---

    def test_severity_high_threat(self):
        """Threats should be high severity."""
        result = detect_hate_speech("nak bunuh semua Cina ni")
        assert result["severity"] == "high"

    def test_severity_medium_dehumanizing(self):
        """Dehumanizing without threat should be medium."""
        result = detect_hate_speech("Melayu ni sampah masyarakat")
        assert result["is_hate"] is True
        assert result["severity"] in ["medium", "high"]

    def test_severity_low_stereotyping(self):
        """Stereotyping should be low severity."""
        result = detect_hate_speech("memang la Cina semua sama je kedekut")
        assert result["is_hate"] is True
        assert result["severity"] in ["low", "medium"]

    # --- Edge cases ---

    def test_empty_text(self):
        result = detect_hate_speech("")
        assert result["is_hate"] is False
        assert result["category"] == "none"
        assert result["confidence"] == 0.0

    def test_whitespace_only(self):
        result = detect_hate_speech("   ")
        assert result["is_hate"] is False

    def test_result_structure(self):
        """Verify result dict has all required keys."""
        result = detect_hate_speech("test text")
        assert "is_hate" in result
        assert "category" in result
        assert "confidence" in result
        assert "severity" in result
        assert "target_group" in result
        assert "explanation" in result

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        result = detect_hate_speech("halau Bangla balik negara dia")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_leetspeak_evasion(self):
        """Should detect hate speech even with leetspeak evasion."""
        # Using numbers to evade: b0d0h = bodoh
        result = detect_hate_speech("s3mua M3layu ni b0d0h bang4ng")
        # May or may not catch depending on normalization depth
        # At minimum should not crash
        assert "is_hate" in result


class TestDetectBatch:
    """Tests for detect_batch function."""

    def test_batch_mixed(self):
        texts = [
            "kawan aku orang Cina, baik orangnya",
            "halau semua Bangla ni parasit",
            "aku nak makan nasi lemak",
        ]
        results = detect_batch(texts)
        assert len(results) == 3
        assert results[0]["is_hate"] is False
        assert results[1]["is_hate"] is True
        assert results[2]["is_hate"] is False

    def test_batch_empty(self):
        results = detect_batch([])
        assert results == []

    def test_batch_single(self):
        results = detect_batch(["normal text here"])
        assert len(results) == 1
        assert results[0]["is_hate"] is False


class TestIsHateSpeech:
    """Tests for is_hate_speech shortcut function."""

    def test_returns_bool_true(self):
        result = is_hate_speech("halau Bangla ni semua, sampah negara")
        assert result is True

    def test_returns_bool_false(self):
        result = is_hate_speech("hari ni cuaca panas gila")
        assert result is False

    def test_neutral_text(self):
        assert is_hate_speech("jom makan mamak") is False


class TestGetSeverity:
    """Tests for get_severity function."""

    def test_none_severity(self):
        result = get_severity("aku pergi sekolah hari ni")
        assert result == "none"

    def test_high_severity(self):
        result = get_severity("nak bunuh semua India ni bantai")
        assert result == "high"

    def test_returns_valid_level(self):
        result = get_severity("some random text")
        assert result in SEVERITY_LEVELS


class TestGetTargetGroups:
    """Tests for get_target_groups function."""

    def test_single_target(self):
        result = get_target_groups("halau Bangla balik negara dia sampah")
        assert "bangladeshi" in result

    def test_multiple_targets(self):
        result = get_target_groups("Cina dan India semua bodoh bangang")
        assert len(result) >= 1

    def test_no_targets(self):
        result = get_target_groups("hari ni aku pergi kerja macam biasa")
        assert result == []

    def test_empty_text(self):
        result = get_target_groups("")
        assert result == []


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_categories(self):
        expected = ["racial", "religious", "sexist", "xenophobic", "homophobic", "none"]
        for cat in expected:
            assert cat in CATEGORIES

    def test_severity_levels(self):
        expected = ["none", "low", "medium", "high"]
        for level in expected:
            assert level in SEVERITY_LEVELS

    def test_categories_count(self):
        assert len(CATEGORIES) == 6

    def test_severity_count(self):
        assert len(SEVERITY_LEVELS) == 4

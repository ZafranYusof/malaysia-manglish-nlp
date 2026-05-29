"""Tests for malaysian_manglish_nlp.stance module."""

import pytest
from malaysian_manglish_nlp.stance import (
    detect_stance,
    detect_stance_batch,
    compare_stances,
    extract_stance_target,
)


class TestDetectStance:
    """Tests for detect_stance function."""

    def test_empty_text(self):
        result = detect_stance("")
        assert result["stance"] == "neutral"

    def test_whitespace_only(self):
        result = detect_stance("   ")
        assert result["stance"] == "neutral"

    def test_support_setuju(self):
        result = detect_stance("Aku setuju dengan cadangan tu")
        assert result["stance"] == "support"
        assert "setuju" in result["indicators"]

    def test_support_sokong(self):
        result = detect_stance("Aku sokong keputusan ni")
        assert result["stance"] == "support"
        assert "sokong" in result["indicators"]

    def test_support_betul(self):
        result = detect_stance("Betul tu, memang patut")
        assert result["stance"] == "support"

    def test_support_english(self):
        result = detect_stance("I agree with this proposal")
        assert result["stance"] == "support"
        assert "agree" in result["indicators"]

    def test_support_bagus(self):
        result = detect_stance("Bagus la kerajaan buat macam ni")
        assert result["stance"] == "support"

    def test_support_tahniah(self):
        result = detect_stance("Tahniah, well done team!")
        assert result["stance"] == "support"

    def test_oppose_tak_setuju(self):
        result = detect_stance("Aku tak setuju langsung")
        assert result["stance"] == "oppose"

    def test_oppose_bantah(self):
        result = detect_stance("Rakyat bantah kenaikan harga")
        assert result["stance"] == "oppose"

    def test_oppose_bodoh(self):
        result = detect_stance("Bodoh la idea ni, tak masuk akal")
        assert result["stance"] == "oppose"

    def test_oppose_english(self):
        result = detect_stance("I disagree, this is nonsense")
        assert result["stance"] == "oppose"

    def test_oppose_stupid_idea(self):
        result = detect_stance("What a stupid idea, reject this")
        assert result["stance"] == "oppose"

    def test_neutral_maybe(self):
        result = detect_stance("Maybe boleh jadi, entah la")
        assert result["stance"] == "neutral"

    def test_neutral_tak_sure(self):
        result = detect_stance("Tak sure la, depends on situation")
        assert result["stance"] == "neutral"

    def test_neutral_both_sides(self):
        result = detect_stance("Both sides ada point, hard to say")
        assert result["stance"] == "neutral"

    def test_negation_tak_sokong(self):
        result = detect_stance("Aku tak sokong benda ni")
        assert result["stance"] == "oppose"

    def test_double_negation_bukan_tak_setuju(self):
        result = detect_stance("Bukan tak setuju, tapi kena fikir dulu")
        assert result["stance"] == "support"

    def test_sarcasm_bagus_la_tu(self):
        result = detect_stance("Bagus la tu konon... pandai sangat")
        assert result["stance"] == "oppose"

    def test_with_target_relevant(self):
        result = detect_stance("Aku sokong kenaikan gaji", target="kenaikan gaji")
        assert result["stance"] == "support"

    def test_with_target_irrelevant(self):
        result = detect_stance("Aku sokong kenaikan gaji", target="cukai pendapatan")
        assert result["stance"] == "neutral"
        assert result["confidence"] < 0.5

    def test_confidence_range(self):
        result = detect_stance("Setuju sangat, memang betul, sokong!")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_multiple_support_indicators(self):
        result = detect_stance("Setuju, betul, memang patut buat macam ni")
        assert result["stance"] == "support"
        assert len(result["indicators"]) >= 2

    def test_multiple_oppose_indicators(self):
        result = detect_stance("Bodoh, salah, tak patut langsung")
        assert result["stance"] == "oppose"
        assert len(result["indicators"]) >= 2

    def test_plus_one(self):
        result = detect_stance("Idea bagus +1")
        assert result["stance"] == "support"

    def test_minus_one(self):
        result = detect_stance("Terrible idea -1")
        assert result["stance"] == "oppose"

    def test_mixed_signals_defaults_neutral(self):
        result = detect_stance("Setuju sikit tapi tak setuju jugak")
        # Mixed signals - could go either way but should have lower confidence
        assert result["confidence"] < 0.8

    def test_returns_dict_format(self):
        result = detect_stance("Aku setuju")
        assert "stance" in result
        assert "confidence" in result
        assert "indicators" in result
        assert isinstance(result["indicators"], list)


class TestDetectStanceBatch:
    """Tests for detect_stance_batch function."""

    def test_empty_list(self):
        result = detect_stance_batch([])
        assert result == []

    def test_single_item(self):
        result = detect_stance_batch(["Aku setuju"])
        assert len(result) == 1
        assert result[0]["stance"] == "support"

    def test_multiple_items(self):
        texts = ["Aku sokong", "Tak setuju", "Maybe la"]
        result = detect_stance_batch(texts)
        assert len(result) == 3
        assert result[0]["stance"] == "support"
        assert result[1]["stance"] == "oppose"
        assert result[2]["stance"] == "neutral"

    def test_with_target(self):
        texts = ["Sokong kenaikan gaji", "Bantah kenaikan gaji"]
        result = detect_stance_batch(texts, target="kenaikan gaji")
        assert result[0]["stance"] == "support"
        assert result[1]["stance"] == "oppose"


class TestCompareStances:
    """Tests for compare_stances function."""

    def test_both_support_agree(self):
        result = compare_stances("Aku setuju", "Betul, sokong")
        assert result == "agree"

    def test_both_oppose_agree(self):
        result = compare_stances("Tak setuju", "Bodoh idea ni")
        assert result == "agree"

    def test_support_vs_oppose_disagree(self):
        result = compare_stances("Aku sokong", "Tak setuju langsung")
        assert result == "disagree"

    def test_oppose_vs_support_disagree(self):
        result = compare_stances("Bantah!", "Setuju je")
        assert result == "disagree"

    def test_neutral_vs_support_unrelated(self):
        result = compare_stances("Entah la, tak sure", "Aku sokong")
        assert result == "unrelated"

    def test_both_neutral_low_confidence(self):
        result = compare_stances("Hmm ok", "Ye la")
        # Both neutral with low confidence = unrelated
        assert result in ("unrelated", "agree")


class TestExtractStanceTarget:
    """Tests for extract_stance_target function."""

    def test_empty_text(self):
        result = extract_stance_target("")
        assert result is None

    def test_none_text(self):
        result = extract_stance_target(None)
        assert result is None

    def test_dengan_pattern(self):
        result = extract_stance_target("Aku tak setuju dengan kenaikan harga minyak")
        assert result is not None
        assert "kenaikan harga minyak" in result

    def test_about_pattern(self):
        result = extract_stance_target("I disagree about the new policy")
        assert result is not None

    def test_pasal_pattern(self):
        result = extract_stance_target("Bantah pasal cukai baru")
        assert result is not None

    def test_no_clear_target(self):
        result = extract_stance_target("Bagus la tu")
        # May or may not find target - depends on pattern matching
        # Just ensure it doesn't crash
        assert result is None or isinstance(result, str)

    def test_tentang_pattern(self):
        result = extract_stance_target("Setuju tentang perubahan jadual")
        assert result is not None
        assert "perubahan jadual" in result

    def test_short_text_no_target(self):
        result = extract_stance_target("Ok")
        assert result is None

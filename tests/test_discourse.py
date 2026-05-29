"""Tests for manglish_nlp.discourse module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp.discourse import (
    analyze_discourse,
    extract_arguments,
    detect_discourse_markers,
    segment_discourse,
    argument_strength,
    detect_fallacies,
)


# ============================================================
# Tests for analyze_discourse
# ============================================================

def test_analyze_discourse_simple_argument():
    """Simple claim + evidence structure."""
    text = "Aku rasa harga minyak naik sebab demand tinggi"
    result = analyze_discourse(text)
    assert "structure" in result
    assert "coherence_score" in result
    assert len(result["structure"]) >= 1
    assert result["coherence_score"] >= 0.0


def test_analyze_discourse_complex():
    """Complex argument with multiple parts."""
    text = ("I think public transport should be free. "
            "Because 60% of people cannot afford cars. "
            "But some say it will cost too much. "
            "So we need to find a balance.")
    result = analyze_discourse(text)
    assert len(result["structure"]) >= 3
    roles = [s["role"] for s in result["structure"]]
    assert "claim" in roles
    assert result["coherence_score"] > 0.3


def test_analyze_discourse_empty():
    """Empty text returns empty structure."""
    result = analyze_discourse("")
    assert result == {"structure": [], "coherence_score": 0.0}


def test_analyze_discourse_whitespace():
    """Whitespace-only text returns empty."""
    result = analyze_discourse("   ")
    assert result == {"structure": [], "coherence_score": 0.0}


def test_analyze_discourse_single_sentence():
    """Single sentence gets classified."""
    text = "Aku rasa kerajaan patut buat lebih banyak"
    result = analyze_discourse(text)
    assert len(result["structure"]) >= 1
    assert result["structure"][0]["role"] in ["claim", "evidence", "rebuttal", "conclusion", "background"]


def test_analyze_discourse_confidence_range():
    """All confidence values should be 0-1."""
    text = "I think this is wrong. Because the data shows 80% disagree. So we should change it."
    result = analyze_discourse(text)
    for segment in result["structure"]:
        assert 0.0 <= segment["confidence"] <= 1.0


def test_analyze_discourse_coherence_range():
    """Coherence score should be 0-1."""
    text = "Aku rasa ni salah. Sebab data tunjuk 80% tak setuju. Jadi kena tukar."
    result = analyze_discourse(text)
    assert 0.0 <= result["coherence_score"] <= 1.0


# ============================================================
# Tests for extract_arguments
# ============================================================

def test_extract_arguments_simple():
    """Simple argument extraction."""
    text = "I think we should ban plastic bags. Because 90% end up in the ocean."
    result = extract_arguments(text)
    assert len(result) >= 1
    assert "claim" in result[0]
    assert "evidence" in result[0]
    assert "stance" in result[0]


def test_extract_arguments_with_stance_for():
    """Detect positive stance."""
    text = "Aku setuju kita patut sokong local brands. Sebab diorang bagi kerja kat orang kita."
    result = extract_arguments(text)
    assert len(result) >= 1
    assert result[0]["stance"] == "for"


def test_extract_arguments_with_stance_against():
    """Detect negative stance."""
    text = "Tak patut buat macam tu. Salah dari segi undang-undang."
    result = extract_arguments(text)
    assert len(result) >= 1
    assert result[0]["stance"] == "against"


def test_extract_arguments_empty():
    """Empty text returns empty list."""
    result = extract_arguments("")
    assert result == []


def test_extract_arguments_multiple():
    """Multiple arguments in one text."""
    text = ("I think education should be free. Because everyone deserves access. "
            "I also believe healthcare must be universal. Research shows it saves money.")
    result = extract_arguments(text)
    assert len(result) >= 2


def test_extract_arguments_with_evidence():
    """Arguments should capture evidence."""
    text = "Aku rasa public transport kena improve. Contoh, 70% pekerja guna kereta sebab bas tak reliable."
    result = extract_arguments(text)
    assert len(result) >= 1
    assert len(result[0]["evidence"]) >= 1


# ============================================================
# Tests for detect_discourse_markers
# ============================================================

def test_detect_markers_causal_bm():
    """Detect BM causal markers."""
    text = "Dia tak datang sebab hujan lebat"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "causal" for m in result)


def test_detect_markers_causal_en():
    """Detect English causal markers."""
    text = "I stayed home because it was raining"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "causal" for m in result)


def test_detect_markers_contrast():
    """Detect contrast markers."""
    text = "Dia pandai tapi malas"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "contrast" for m in result)


def test_detect_markers_addition():
    """Detect addition markers."""
    text = "Dia rajin dan pandai"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "addition" for m in result)


def test_detect_markers_temporal():
    """Detect temporal markers."""
    text = "Lepas tu kita pergi makan"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "temporal" for m in result)


def test_detect_markers_conclusion():
    """Detect conclusion markers."""
    text = "Jadi kesimpulannya kita kena buat lebih baik"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert any(m["type"] == "conclusion" for m in result)


def test_detect_markers_manglish_mixed():
    """Detect markers in code-switched text."""
    text = "First dia cakap ok tapi lepas tu dia tukar fikiran so aku pun confused"
    result = detect_discourse_markers(text)
    assert len(result) >= 2
    types = [m["type"] for m in result]
    assert "temporal" in types or "contrast" in types or "conclusion" in types


def test_detect_markers_empty():
    """Empty text returns empty list."""
    result = detect_discourse_markers("")
    assert result == []


def test_detect_markers_position():
    """Markers should have correct position info."""
    text = "tapi aku tak setuju"
    result = detect_discourse_markers(text)
    assert len(result) >= 1
    assert result[0]["position"] == 0


def test_detect_markers_no_overlap():
    """Markers should not overlap."""
    text = "sebab tu la dia marah"
    result = detect_discourse_markers(text)
    positions = [(m["position"], m["position"] + len(m["marker"])) for m in result]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert positions[i][1] <= positions[j][0] or positions[j][1] <= positions[i][0]


# ============================================================
# Tests for segment_discourse
# ============================================================

def test_segment_discourse_basic():
    """Basic segmentation."""
    text = "Aku rasa harga naik. Sebab supply kurang. Jadi kena cari alternatif."
    result = segment_discourse(text)
    assert len(result) >= 2
    for seg in result:
        assert "segment" in seg
        assert "role" in seg
        assert "confidence" in seg


def test_segment_discourse_empty():
    """Empty text returns empty list."""
    result = segment_discourse("")
    assert result == []


def test_segment_roles_valid():
    """All roles should be valid."""
    text = "I think this is important. Because data shows 50% improvement. But others disagree. So we need more research."
    result = segment_discourse(text)
    valid_roles = {"claim", "evidence", "rebuttal", "conclusion", "background"}
    for seg in result:
        assert seg["role"] in valid_roles


# ============================================================
# Tests for argument_strength
# ============================================================

def test_argument_strength_strong():
    """Well-supported argument should score high."""
    text = ("I believe we should invest in renewable energy. "
            "Research shows 75% reduction in emissions. "
            "Data from 50 countries confirms this. "
            "Therefore, the evidence is clear.")
    score = argument_strength(text)
    assert score >= 0.5


def test_argument_strength_weak():
    """Unsupported claim should score low."""
    text = "Aku rasa betul la tu"
    score = argument_strength(text)
    assert score < 0.5


def test_argument_strength_empty():
    """Empty text returns 0."""
    score = argument_strength("")
    assert score == 0.0


def test_argument_strength_range():
    """Score should always be 0-1."""
    texts = [
        "ok",
        "I think this because that",
        "According to research, 90% of experts agree. The data shows clear improvement. Therefore we should proceed.",
    ]
    for text in texts:
        score = argument_strength(text)
        assert 0.0 <= score <= 1.0


def test_argument_strength_with_numbers():
    """Arguments with data should score higher."""
    text_no_data = "I think public transport is better"
    text_with_data = "I think public transport is better. Statistics show 60% less emissions and 30% cost savings."
    score_no_data = argument_strength(text_no_data)
    score_with_data = argument_strength(text_with_data)
    assert score_with_data > score_no_data


# ============================================================
# Tests for detect_fallacies
# ============================================================

def test_detect_fallacies_ad_hominem():
    """Detect ad hominem attacks."""
    text = "Kau bodoh la, mana tau pasal ekonomi"
    result = detect_fallacies(text)
    assert len(result) >= 1
    assert any(f["type"] == "ad_hominem" for f in result)


def test_detect_fallacies_appeal_to_authority():
    """Detect appeal to authority."""
    text = "Expert said this is correct so it must be true"
    result = detect_fallacies(text)
    assert len(result) >= 1
    assert any(f["type"] == "appeal_to_authority" for f in result)


def test_detect_fallacies_strawman():
    """Detect strawman arguments."""
    text = "So you're saying we should just ignore the problem completely?"
    result = detect_fallacies(text)
    assert len(result) >= 1
    assert any(f["type"] == "strawman" for f in result)


def test_detect_fallacies_false_dichotomy():
    """Detect false dichotomy."""
    text = "Either you support this policy or you hate the country"
    result = detect_fallacies(text)
    assert len(result) >= 1
    assert any(f["type"] == "false_dichotomy" for f in result)


def test_detect_fallacies_empty():
    """Empty text returns no fallacies."""
    result = detect_fallacies("")
    assert result == []


def test_detect_fallacies_clean_argument():
    """Clean argument should have no/few fallacies."""
    text = "I think we should invest more in education. Research shows countries with higher education spending have better GDP growth."
    result = detect_fallacies(text)
    # Clean arguments might still trigger some patterns, but should be minimal
    assert len(result) <= 1


def test_detect_fallacies_confidence():
    """Fallacy confidence should be in range."""
    text = "Kau bodoh la tak tau apa-apa"
    result = detect_fallacies(text)
    for f in result:
        assert 0.0 <= f["confidence"] <= 1.0
        assert "description" in f
        assert "evidence" in f


# ============================================================
# Tests for political/complex text
# ============================================================

def test_political_discussion():
    """Political discussion with mixed arguments."""
    text = ("Aku rasa kerajaan patut naikkan gaji minimum. "
            "Sebab harga barang dah naik 30% tahun ni. "
            "Tapi ada orang cakap nanti company rugi. "
            "Jadi kena cari jalan tengah la.")
    
    # Discourse analysis
    discourse = analyze_discourse(text)
    assert len(discourse["structure"]) >= 3
    roles = [s["role"] for s in discourse["structure"]]
    assert "claim" in roles
    
    # Argument extraction
    args = extract_arguments(text)
    assert len(args) >= 1
    
    # Markers
    markers = detect_discourse_markers(text)
    assert len(markers) >= 2


def test_code_switched_argument():
    """Code-switched BM/EN argument."""
    text = ("I think Malaysia should invest more in tech. "
            "Sebab kita dah ketinggalan compared to Singapore. "
            "Plus our graduates tak cukup skilled. "
            "So government kena allocate more budget for STEM.")
    
    discourse = analyze_discourse(text)
    assert len(discourse["structure"]) >= 3
    assert discourse["coherence_score"] > 0.3
    
    markers = detect_discourse_markers(text)
    marker_types = set(m["type"] for m in markers)
    assert len(marker_types) >= 2


if __name__ == "__main__":
    # Run all tests
    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in test_functions:
        try:
            test_fn()
            passed += 1
            print(f"  PASS: {test_fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test_fn.__name__} - {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {test_fn.__name__} - {type(e).__name__}: {e}")
    
    print(f"\nResults: {passed} passed, {failed} failed, {passed + failed} total")

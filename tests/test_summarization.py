"""Tests for malaysian_manglish_nlp.summarization module."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from malaysian_manglish_nlp.summarization import (
    summarize, summarize_sentences, extract_key_phrases,
    get_sentence_scores, summarize_thread,
)


class TestSummarize:
    """Tests for summarize() function."""

    def test_empty_text(self):
        assert summarize("") == ""

    def test_none_text(self):
        assert summarize(None) == ""

    def test_whitespace_only(self):
        assert summarize("   ") == ""

    def test_single_sentence(self):
        text = "Aku nak pergi kedai beli makanan."
        result = summarize(text, num_sentences=3)
        assert result == text.strip()

    def test_short_text_fewer_than_requested(self):
        text = "Hari ni panas gila. Nak makan ais krim."
        result = summarize(text, num_sentences=5)
        # Should return all sentences since fewer than requested
        assert "panas gila" in result
        assert "ais krim" in result

    def test_long_text_extracts_key_sentences(self):
        text = (
            "Malaysia ada banyak tempat menarik untuk dilawati. "
            "Pulau Langkawi terkenal dengan pantai yang cantik. "
            "Kuala Lumpur pula ada KLCC dan Menara KL. "
            "Makanan Malaysia memang sedap dan murah. "
            "Nasi lemak adalah makanan kebangsaan Malaysia. "
            "Ramai pelancong datang setiap tahun untuk menikmati budaya Malaysia."
        )
        result = summarize(text, num_sentences=3)
        sentences = result.split('. ')
        # Should have roughly 3 sentences
        assert len(result) > 0
        assert len(result) < len(text)

    def test_code_switched_text(self):
        text = (
            "Today I went to pasar malam dekat rumah. "
            "The food there memang sedap gila bro. "
            "I bought nasi goreng and teh tarik for dinner. "
            "Harga pun murah, only RM5 for everything. "
            "Next week nak pergi lagi with my friends."
        )
        result = summarize(text, num_sentences=2)
        assert len(result) > 0
        # Original text preserved (not normalized)
        assert "nak" in result or "pasar" in result or "nasi" in result or "murah" in result

    def test_method_textrank(self):
        text = "Sentence one here. Sentence two here. Sentence three here."
        result = summarize(text, num_sentences=2, method='textrank')
        assert len(result) > 0

    def test_invalid_method_raises(self):
        try:
            summarize("Some text here.", method='invalid')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported method" in str(e)

    def test_num_sentences_one(self):
        text = (
            "Kerajaan umum cuti peristiwa. "
            "Semua sekolah dan pejabat tutup. "
            "Rakyat gembira dapat cuti tambahan."
        )
        result = summarize(text, num_sentences=1)
        # Should be a single sentence
        assert len(result) > 0


class TestSummarizeSentences:
    """Tests for summarize_sentences() function."""

    def test_empty_returns_empty_list(self):
        assert summarize_sentences("") == []

    def test_none_returns_empty_list(self):
        assert summarize_sentences(None) == []

    def test_returns_list(self):
        text = "First sentence. Second sentence. Third sentence."
        result = summarize_sentences(text, num_sentences=2)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_preserves_original_order(self):
        text = (
            "Introduction to the topic. "
            "Some filler content here. "
            "Important conclusion at the end."
        )
        result = summarize_sentences(text, num_sentences=2)
        assert isinstance(result, list)
        # Results should be in document order
        for i in range(len(result) - 1):
            assert text.index(result[i]) <= text.index(result[i + 1])

    def test_fewer_sentences_returns_all(self):
        text = "Only one sentence here."
        result = summarize_sentences(text, num_sentences=5)
        assert len(result) == 1
        assert result[0] == "Only one sentence here."

    def test_manglish_shortforms(self):
        text = (
            "Aku nk pergi kedai sbb lapar gila. "
            "Tp kedai tu dah tutup pukul 10. "
            "Jadi aku blk rumah masak maggi je. "
            "Nasib baik ada telur dlm peti ais. "
            "Lepas makan aku tido terus sbb penat."
        )
        result = summarize_sentences(text, num_sentences=2)
        assert len(result) == 2
        # Should contain original shortforms, not normalized
        for sent in result:
            assert sent in text


class TestExtractKeyPhrases:
    """Tests for extract_key_phrases() function."""

    def test_empty_text(self):
        assert extract_key_phrases("") == []

    def test_none_text(self):
        assert extract_key_phrases(None) == []

    def test_returns_list(self):
        text = "Malaysia ada banyak tempat menarik. Tempat menarik di Malaysia sangat cantik."
        result = extract_key_phrases(text)
        assert isinstance(result, list)

    def test_top_n_limit(self):
        text = (
            "Nasi lemak sedap. Roti canai pun sedap. "
            "Teh tarik best. Milo ais pun best. "
            "Makanan Malaysia memang terbaik."
        )
        result = extract_key_phrases(text, top_n=3)
        assert len(result) <= 3

    def test_extracts_meaningful_phrases(self):
        text = (
            "Artificial intelligence is transforming technology. "
            "Machine learning and deep learning are subfields of artificial intelligence. "
            "Natural language processing uses machine learning for text analysis."
        )
        result = extract_key_phrases(text, top_n=5)
        assert len(result) > 0
        # Should find relevant terms
        found_relevant = any(
            'machine' in p or 'learning' in p or 'artificial' in p or 'intelligence' in p
            for p in result
        )
        assert found_relevant

    def test_manglish_phrases(self):
        text = (
            "Projek FYP aku pasal sentiment analysis. "
            "Guna model HuggingFace untuk sentiment analysis. "
            "Dataset dari berita Malaysia untuk sentiment analysis."
        )
        result = extract_key_phrases(text, top_n=5)
        assert len(result) > 0

    def test_single_word_text(self):
        text = "Hello"
        result = extract_key_phrases(text, top_n=5)
        # Might be empty if word is too short or is stopword
        assert isinstance(result, list)


class TestGetSentenceScores:
    """Tests for get_sentence_scores() function."""

    def test_empty_text(self):
        assert get_sentence_scores("") == []

    def test_none_text(self):
        assert get_sentence_scores(None) == []

    def test_returns_correct_format(self):
        text = "First sentence. Second sentence. Third sentence."
        result = get_sentence_scores(text)
        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert "sentence" in item
            assert "score" in item
            assert "position" in item
            assert isinstance(item["score"], float)
            assert isinstance(item["position"], int)

    def test_scores_between_0_and_1(self):
        text = (
            "Malaysia is a beautiful country. "
            "The food is amazing and diverse. "
            "People are friendly and welcoming."
        )
        result = get_sentence_scores(text)
        for item in result:
            assert 0.0 <= item["score"] <= 1.0

    def test_position_is_sequential(self):
        text = "One. Two. Three. Four."
        result = get_sentence_scores(text)
        for i, item in enumerate(result):
            assert item["position"] == i

    def test_first_sentence_gets_position_boost(self):
        # First sentence should get a boost even if content is similar
        text = (
            "Penting sangat benda ni kena buat. "
            "Benda lain pun ada jugak. "
            "Tapi yang penting kena siap dulu."
        )
        result = get_sentence_scores(text)
        # First sentence should have a relatively high score due to position boost
        assert result[0]["score"] > 0.5

    def test_single_sentence(self):
        text = "Just one sentence here."
        result = get_sentence_scores(text)
        assert len(result) == 1
        assert result[0]["score"] == 1.0
        assert result[0]["position"] == 0


class TestSummarizeThread:
    """Tests for summarize_thread() function."""

    def test_empty_list(self):
        assert summarize_thread([]) == ""

    def test_none_messages(self):
        # Should handle gracefully
        assert summarize_thread(None) == ""

    def test_single_message(self):
        result = summarize_thread(["Hello semua"])
        assert "Hello semua" in result

    def test_few_messages_returns_all(self):
        messages = ["Msg 1", "Msg 2", "Msg 3"]
        result = summarize_thread(messages, num_points=5)
        assert "Msg 1" in result
        assert "Msg 2" in result
        assert "Msg 3" in result

    def test_bullet_point_format(self):
        messages = ["First point", "Second point", "Third point"]
        result = summarize_thread(messages, num_points=5)
        assert "•" in result

    def test_many_messages_condensed(self):
        messages = [
            "Wei korang nak pergi makan kat mana?",
            "Aku suggest nasi kandar",
            "Nasi kandar best la memang",
            "Jom la pergi mamak je",
            "Mamak dekat rumah aku ada",
            "Ok set mamak",
            "Pukul berapa nak pergi?",
            "Aku free lepas 7",
            "Ok 7.30 la kita jumpa",
            "Sape drive?",
            "Aku boleh drive",
            "Ok aku tunggu kat rumah",
        ]
        result = summarize_thread(messages, num_points=3)
        assert len(result) > 0
        # Should be condensed
        bullet_count = result.count("•")
        assert bullet_count <= 5

    def test_filters_empty_messages(self):
        messages = ["Hello", "", "  ", "World"]
        result = summarize_thread(messages, num_points=5)
        assert "Hello" in result
        assert "World" in result

    def test_code_switched_thread(self):
        messages = [
            "Bro have you done the assignment?",
            "Which one? The programming one ke?",
            "Ya la the Java assignment",
            "Belum lagi bro, deadline bila?",
            "Friday this week",
            "Ok ok I'll start tonight",
        ]
        result = summarize_thread(messages, num_points=3)
        assert len(result) > 0

    def test_num_points_respected(self):
        messages = [f"Message number {i} about topic {i}" for i in range(20)]
        result = summarize_thread(messages, num_points=3)
        bullet_count = result.count("•")
        assert bullet_count <= 5  # Should be around num_points


class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_single_sentence(self):
        text = "word " * 200
        result = summarize(text.strip(), num_sentences=3)
        assert len(result) > 0

    def test_unicode_text(self):
        text = "Harga naik 10% 📈. Rakyat susah nak beli barang. Kerajaan kena buat sesuatu."
        result = summarize(text, num_sentences=2)
        assert len(result) > 0

    def test_repeated_sentences(self):
        text = "Same thing. Same thing. Same thing. Different thing here."
        result = summarize(text, num_sentences=2)
        assert len(result) > 0

    def test_newline_separated(self):
        text = "First paragraph here.\nSecond paragraph here.\nThird paragraph here."
        result = summarize_sentences(text, num_sentences=2)
        assert len(result) == 2

    def test_mixed_punctuation(self):
        text = "Is this working? Yes it is! Great news. Let's continue."
        result = get_sentence_scores(text)
        assert len(result) >= 3


# Run tests if executed directly
if __name__ == '__main__':
    import traceback

    test_classes = [
        TestSummarize,
        TestSummarizeSentences,
        TestExtractKeyPhrases,
        TestGetSentenceScores,
        TestSummarizeThread,
        TestEdgeCases,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"{cls.__name__}.{method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"\nFailures:")
        for err in errors:
            print(f"  ✗ {err}")
    else:
        print("All tests passed! ✓")

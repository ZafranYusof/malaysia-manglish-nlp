"""Tests for malaysian_manglish_nlp.translation module."""

import pytest
from malaysian_manglish_nlp.translation import (
    translate, to_english, to_malay, to_formal,
    word_translate, detect_and_translate,
)


class TestTranslate:
    """Test the main translate() function."""

    def test_bm_to_en_basic(self):
        result = translate("saya nak pergi kedai", target='en')
        assert result['target_lang'] == 'en'
        assert 'I' in result['translated']
        assert 'want' in result['translated']
        assert result['confidence'] > 0

    def test_bm_to_en_with_source(self):
        result = translate("dia makan nasi", source='bm', target='en')
        assert result['source_lang'] == 'bm'
        assert 'eat' in result['translated'] or 'rice' in result['translated']

    def test_en_to_bm_basic(self):
        result = translate("I want to go home", source='en', target='bm')
        assert result['target_lang'] == 'bm'
        assert result['translated'] != ''

    def test_manglish_to_en(self):
        result = translate("aku nk makan la", source='manglish', target='en')
        assert result['source_lang'] == 'manglish'
        assert 'eat' in result['translated']
        # Particle 'la' should be removed
        assert 'la' not in result['translated'].split()

    def test_manglish_to_formal(self):
        result = translate("aku nk pegi kedai jap", source='manglish', target='formal')
        assert 'Saya' in result['translated']
        assert 'ingin' in result['translated']
        assert result['target_lang'] == 'formal_bm'

    def test_auto_detect_bm(self):
        result = translate("saya suka makan nasi goreng", target='en')
        assert result['source_lang'] in ('bm', 'manglish')
        assert 'like' in result['translated']

    def test_empty_text(self):
        result = translate("", target='en')
        assert result['translated'] == ''
        assert result['confidence'] == 0.0

    def test_whitespace_only(self):
        result = translate("   ", target='en')
        assert result['translated'] == ''

    def test_confidence_range(self):
        result = translate("saya pergi sekolah", target='en')
        assert 0.0 <= result['confidence'] <= 1.0


class TestToEnglish:
    """Test to_english() function."""

    def test_basic_sentence(self):
        result = to_english("saya suka makan")
        assert 'like' in result
        assert 'eat' in result

    def test_pronouns(self):
        result = to_english("dia pergi sekolah")
        assert 'go' in result or 'school' in result

    def test_with_particles_removed(self):
        result = to_english("best la makanan ini")
        assert 'la' not in result.split()
        assert 'this' in result

    def test_phrase_translation(self):
        result = to_english("terima kasih")
        assert 'thank' in result.lower()

    def test_apa_khabar(self):
        result = to_english("apa khabar")
        assert 'how are you' in result.lower()

    def test_unknown_words_kept(self):
        result = to_english("saya suka pizza")
        assert 'pizza' in result

    def test_multiple_sentences(self):
        result = to_english("saya lapar. nak makan.")
        assert result != ''


class TestToMalay:
    """Test to_malay() function."""

    def test_basic_sentence(self):
        result = to_malay("I like food")
        assert result != ''
        assert 'suka' in result or 'makanan' in result

    def test_articles_removed(self):
        # 'the' and 'a' should map to empty and be removed
        result = to_malay("the cat is big")
        assert 'the' not in result.split()

    def test_pronouns(self):
        result = to_malay("she is beautiful")
        assert 'dia' in result or 'cantik' in result

    def test_unknown_words_kept(self):
        result = to_malay("I like sushi")
        assert 'sushi' in result


class TestToFormal:
    """Test to_formal() function."""

    def test_pronouns_formalized(self):
        result = to_formal("aku nak pergi")
        assert 'Saya' in result
        assert 'ingin' in result

    def test_shortforms_expanded(self):
        result = to_formal("nk pegi skrg")
        assert 'ingin' in result or 'pergi' in result
        assert 'sekarang' in result.lower()

    def test_particles_removed(self):
        result = to_formal("best la makanan tu kan")
        assert 'la' not in result.split()
        assert 'kan' not in result.split()

    def test_ends_with_punctuation(self):
        result = to_formal("aku nak makan")
        assert result[-1] in '.!?'

    def test_capitalized(self):
        result = to_formal("aku suka")
        assert result[0].isupper()

    def test_ko_to_anda(self):
        result = to_formal("ko nak pergi mana")
        # 'ko' normalizes to 'awak' then formalizes to 'anda', or stays as 'awak'
        assert 'anda' in result.lower() or 'awak' in result.lower()

    def test_complex_manglish(self):
        result = to_formal("aku dh penat sgt la wei")
        assert 'Saya' in result
        assert 'sangat' in result.lower()
        assert 'wei' not in result.lower()


class TestWordTranslate:
    """Test word_translate() function."""

    def test_bm_to_en(self):
        assert word_translate("rumah", target='en') == 'house'

    def test_en_to_bm(self):
        assert word_translate("house", target='bm') == 'rumah'

    def test_unknown_word(self):
        assert word_translate("xyzabc", target='en') is None

    def test_case_insensitive(self):
        assert word_translate("Rumah", target='en') == 'house'

    def test_adjective(self):
        assert word_translate("cantik", target='en') == 'beautiful'

    def test_verb(self):
        assert word_translate("makan", target='en') == 'eat'

    def test_en_verb_to_bm(self):
        result = word_translate("eat", target='bm')
        assert result == 'makan'


class TestDetectAndTranslate:
    """Test detect_and_translate() function."""

    def test_bm_auto_to_en(self):
        result = detect_and_translate("saya suka makan")
        assert result['original'] == 'saya suka makan'
        assert result['source_lang'] in ('bm', 'manglish')
        assert result['target_lang'] == 'en'
        assert 'like' in result['translated']

    def test_en_auto_to_bm(self):
        result = detect_and_translate("I like to eat rice")
        assert result['source_lang'] == 'en'
        assert result['target_lang'] == 'bm'
        assert result['translated'] != ''

    def test_empty_text(self):
        result = detect_and_translate("")
        assert result['translated'] == ''
        assert result['source_lang'] == 'unknown'

    def test_has_confidence(self):
        result = detect_and_translate("saya pergi sekolah")
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0

    def test_manglish_detected(self):
        result = detect_and_translate("aku nk lepak jom la")
        assert result['source_lang'] in ('manglish', 'bm')
        assert result['target_lang'] == 'en'


class TestParticlesHandling:
    """Test that particles are properly handled."""

    def test_la_removed(self):
        result = to_english("bagus la")
        assert 'la' not in result.split()

    def test_kan_removed(self):
        result = to_english("best kan")
        assert 'kan' not in result.split()

    def test_weh_removed(self):
        result = to_english("jom weh")
        assert 'weh' not in result.split()

    def test_multiple_particles(self):
        result = to_english("ok la kan weh")
        for p in ['la', 'kan', 'weh']:
            assert p not in result.split()

    def test_kot_translated(self):
        # 'kot' should become 'maybe'
        result = to_english("dia datang kot")
        assert 'maybe' in result.lower() or 'kot' not in result.split()


class TestPhrases:
    """Test phrase-level translations."""

    def test_terima_kasih(self):
        result = to_english("terima kasih banyak")
        assert 'thank you' in result.lower()

    def test_selamat_pagi(self):
        result = to_english("selamat pagi")
        assert 'good morning' in result.lower()

    def test_tak_boleh(self):
        result = to_english("saya tak boleh pergi")
        assert 'cannot' in result.lower()

    def test_minta_maaf(self):
        result = to_english("minta maaf")
        assert 'sorry' in result.lower()

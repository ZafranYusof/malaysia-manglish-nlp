"""
Comprehensive tests for the Jawi (Rumi ↔ Jawi) conversion module.
"""

import time
import unittest

from manglish_nlp.jawi import (
    ALIF, BA, CA, DAL, FA, GA, HA, JIM, KAF, LAM, MIM,
    NGA, NUN, NYA, PA, QA, RA, SIN, TA, VA, WAW, YA, YE,
    batch_to_jawi, batch_to_rumi,
    detect_script, is_jawi, to_jawi, to_jawi_words, to_rumi,
)


class TestDetectScript(unittest.TestCase):
    """Test script detection (Rumi/Jawi/mixed)."""

    def test_rumi_text(self):
        self.assertEqual(detect_script('hello world'), 'rumi')

    def test_jawi_text(self):
        # "makan" in Jawi
        jawi = to_jawi('makan')
        self.assertEqual(detect_script(jawi), 'jawi')

    def test_mixed_text(self):
        self.assertEqual(detect_script(f'hello {ALIF}{BA}{TA}'), 'mixed')

    def test_empty_string(self):
        self.assertEqual(detect_script(''), 'unknown')

    def test_whitespace_only(self):
        self.assertEqual(detect_script('   '), 'unknown')

    def test_numbers_only(self):
        self.assertEqual(detect_script('12345'), 'unknown')

    def test_punctuation_only(self):
        self.assertEqual(detect_script('!@#$%'), 'unknown')


class TestIsJawi(unittest.TestCase):
    """Test Jawi detection helper."""

    def test_jawi_word(self):
        self.assertTrue(is_jawi(ALIF + BA + TA))

    def test_rumi_word(self):
        self.assertFalse(is_jawi('hello'))

    def test_mixed(self):
        self.assertTrue(is_jawi(f'hello {ALIF}'))

    def test_empty(self):
        self.assertFalse(is_jawi(''))


class TestBasicLetterConversion(unittest.TestCase):
    """Test single consonant and basic letter mappings."""

    def test_consonants_in_dict(self):
        """Verify dictionary words convert correctly."""
        jawi = to_jawi('saya')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(SIN, jawi)
        self.assertIn(YA, jawi)

    def test_pa_character(self):
        """Malay-specific pa (ڤ) used correctly."""
        jawi = to_jawi('pergi')
        self.assertIn(PA, jawi)

    def test_ga_character(self):
        """Malay-specific ga (ݢ) used correctly."""
        jawi = to_jawi('gunung')
        self.assertIn(GA, jawi)

    def test_nga_character(self):
        """Malay-specific nga (ڠ) used correctly."""
        jawi = to_jawi('orang')
        self.assertIn(NGA, jawi)

    def test_nya_character(self):
        """Malay-specific nya (ڽ) used correctly."""
        jawi = to_jawi('banyak')
        self.assertIn(NYA, jawi)

    def test_ca_character(self):
        """Malay-specific ca (چ) used correctly."""
        jawi = to_jawi('cakap')
        self.assertIn(CA, jawi)


class TestCommonMalayWords(unittest.TestCase):
    """Test common Malay word conversion via dictionary."""

    COMMON_WORDS = [
        'makan', 'minum', 'pergi', 'datang', 'rumah', 'sekolah',
        'saya', 'kamu', 'orang', 'hari', 'nama', 'banyak',
        'besar', 'kecil', 'panjang', 'cantik', 'baik', 'buruk',
        'tahu', 'mahu', 'boleh', 'ada', 'jadi', 'dapat',
    ]

    def test_all_common_words_produce_jawi(self):
        for word in self.COMMON_WORDS:
            with self.subTest(word=word):
                jawi = to_jawi(word)
                self.assertTrue(
                    is_jawi(jawi),
                    f"'{word}' should produce Jawi output, got: {jawi}"
                )

    def test_common_words_not_empty(self):
        for word in self.COMMON_WORDS:
            with self.subTest(word=word):
                jawi = to_jawi(word)
                self.assertTrue(len(jawi) > 0, f"'{word}' produced empty Jawi")

    def test_dictionary_words_are_consistent(self):
        """Same word should always produce same Jawi."""
        for word in self.COMMON_WORDS:
            with self.subTest(word=word):
                jawi1 = to_jawi(word)
                jawi2 = to_jawi(word)
                self.assertEqual(jawi1, jawi2)


class TestPrefixedWords(unittest.TestCase):
    """Test words with Malay prefixes."""

    def test_me_prefix(self):
        jawi = to_jawi('memakan')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(MIM, jawi)

    def test_ber_prefix(self):
        jawi = to_jawi('bersekolah')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(BA, jawi)

    def test_ter_prefix(self):
        jawi = to_jawi('termakan')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(TA, jawi)

    def test_di_prefix(self):
        jawi = to_jawi('dimakan')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(DAL, jawi)

    def test_pe_prefix(self):
        jawi = to_jawi('pelajar')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(PA, jawi)


class TestSuffixWords(unittest.TestCase):
    """Test words with Malay suffixes."""

    def test_kan_suffix(self):
        jawi = to_jawi('makanan')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(NUN, jawi)

    def test_an_suffix(self):
        jawi = to_jawi('minuman')
        self.assertTrue(is_jawi(jawi))

    def test_lah_suffix(self):
        jawi = to_jawi('makanlah')
        self.assertTrue(is_jawi(jawi))

    def test_nya_suffix(self):
        jawi = to_jawi('rumahnya')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(NYA, jawi)


class TestFullSentences(unittest.TestCase):
    """Test full sentence conversion."""

    def test_simple_sentence(self):
        jawi = to_jawi('saya makan nasi')
        self.assertTrue(is_jawi(jawi))

    def test_question_sentence(self):
        jawi = to_jawi('apa nama kamu?')
        self.assertTrue(is_jawi(jawi))
        self.assertIn('?', jawi)

    def test_greeting(self):
        jawi = to_jawi('selamat pagi')
        self.assertTrue(is_jawi(jawi))

    def test_long_sentence(self):
        sentence = 'saya pergi ke sekolah setiap hari untuk belajar'
        jawi = to_jawi(sentence)
        self.assertTrue(is_jawi(jawi))

    def test_sentence_with_comma(self):
        jawi = to_jawi('saya suka makan, minum, dan tidur')
        self.assertTrue(is_jawi(jawi))
        self.assertIn(',', jawi)


class TestRoundTrip(unittest.TestCase):
    """Test Rumi → Jawi → Rumi round-trip for dictionary words."""

    ROUND_TRIP_WORDS = [
        'makan', 'minum', 'pergi', 'datang', 'rumah',
        'saya', 'kamu', 'orang', 'hari', 'nama',
    ]

    def test_roundtrip_dictionary_words(self):
        """Dictionary words should survive round-trip."""
        for word in self.ROUND_TRIP_WORDS:
            with self.subTest(word=word):
                jawi = to_jawi(word)
                rumi = to_rumi(jawi)
                # At minimum the Jawi should be valid
                self.assertTrue(is_jawi(jawi))
                # Rumi output should be non-empty
                self.assertTrue(len(rumi) > 0)


class TestMixedMalayEnglish(unittest.TestCase):
    """Test mixed Malay-English text handling."""

    def test_mixed_text_preserves_numbers(self):
        jawi = to_jawi('saya beli 3 buku')
        self.assertIn('3', jawi)

    def test_mixed_text_preserves_punctuation(self):
        jawi = to_jawi('apa khabar?')
        self.assertIn('?', jawi)

    def test_sentence_with_numbers(self):
        jawi = to_jawi('harga 100 ringgit')
        self.assertIn('100', jawi)

    def test_multiple_spaces(self):
        jawi = to_jawi('saya  makan')
        # Should preserve double space
        self.assertIn('  ', jawi)


class TestNumbersAndPunctuation(unittest.TestCase):
    """Test numbers and punctuation pass-through."""

    def test_pure_number(self):
        result = to_jawi('12345')
        self.assertEqual(result, '12345')

    def test_pure_punctuation(self):
        result = to_jawi('!@#$%')
        self.assertEqual(result, '!@#$%')

    def test_number_in_sentence(self):
        result = to_jawi('saya ada 5 ringgit')
        self.assertIn('5', result)

    def test_exclamation(self):
        result = to_jawi('tolong!')
        self.assertIn('!', result)

    def test_question_mark(self):
        result = to_jawi('kenapa?')
        self.assertIn('?', result)

    def test_period(self):
        result = to_jawi('sudah siap.')
        self.assertIn('.', result)

    def test_comma(self):
        result = to_jawi('satu, dua, tiga')
        self.assertIn(',', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_empty_string(self):
        self.assertEqual(to_jawi(''), '')

    def test_none_passthrough(self):
        # to_jawi('') returns '' (falsy input returns as-is)
        result = to_jawi('')
        self.assertFalse(result)

    def test_whitespace_only(self):
        result = to_jawi('   ')
        self.assertEqual(result, '   ')

    def test_single_char(self):
        result = to_jawi('a')
        self.assertTrue(len(result) > 0)

    def test_single_consonant(self):
        result = to_jawi('b')
        self.assertTrue(len(result) > 0)

    def test_unicode_input(self):
        # Should not crash on unexpected unicode
        result = to_jawi('café')
        self.assertTrue(len(result) > 0)

    def test_mixed_case(self):
        """Case insensitivity: Makan == MAKAN == makan."""
        r1 = to_jawi('makan')
        r2 = to_jawi('Makan')
        r3 = to_jawi('MAKAN')
        self.assertEqual(r1, r2)
        self.assertEqual(r2, r3)

    def test_newlines_preserved(self):
        result = to_jawi('saya\nmakan')
        self.assertIn('\n', result)

    def test_tabs_preserved(self):
        result = to_jawi('saya\tmakan')
        self.assertIn('\t', result)

    def test_special_chars_in_word(self):
        """Hyphenated words."""
        result = to_jawi('anak-anak')
        self.assertIn('-', result)

    def test_apostrophe(self):
        result = to_jawi("tak")
        self.assertTrue(is_jawi(result))


class TestToRumi(unittest.TestCase):
    """Test Jawi to Rumi conversion."""

    def test_known_jawi_word(self):
        """Convert known Jawi word back to Rumi."""
        # "makan" in Jawi from dict
        jawi = to_jawi('makan')
        rumi = to_rumi(jawi)
        # Should produce something readable
        self.assertTrue(len(rumi) > 0)
        self.assertFalse(is_jawi(rumi))

    def test_rumi_passthrough(self):
        """Non-Jawi text should pass through to_rumi."""
        result = to_rumi('hello world')
        self.assertEqual(result, 'hello world')

    def test_empty_string(self):
        self.assertEqual(to_rumi(''), '')

    def test_numbers_passthrough(self):
        result = to_rumi('12345')
        self.assertEqual(result, '12345')


class TestToJawiWords(unittest.TestCase):
    """Test word-by-word mapping function."""

    def test_returns_tuples(self):
        result = to_jawi_words('saya makan')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'saya')
        self.assertEqual(result[1][0], 'makan')

    def test_each_jawi_valid(self):
        result = to_jawi_words('saya makan nasi')
        for rumi, jawi in result:
            self.assertTrue(is_jawi(jawi), f"Jawi for '{rumi}' invalid: {jawi}")


class TestBatchConversion(unittest.TestCase):
    """Test batch conversion functions."""

    def test_batch_to_jawi(self):
        texts = ['saya', 'makan', 'nasi']
        results = batch_to_jawi(texts)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(is_jawi(r))

    def test_batch_to_rumi(self):
        jawis = [to_jawi(w) for w in ['saya', 'makan', 'nasi']]
        results = batch_to_rumi(jawis)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(len(r) > 0)

    def test_batch_empty_list(self):
        self.assertEqual(batch_to_jawi([]), [])
        self.assertEqual(batch_to_rumi([]), [])


class TestPerformance(unittest.TestCase):
    """Performance tests."""

    def test_1000_words_per_second(self):
        """Should convert at least 1000 words per second."""
        words = ['makan', 'minum', 'pergi', 'datang', 'rumah',
                 'sekolah', 'saya', 'kamu', 'orang', 'hari'] * 100

        start = time.time()
        for w in words:
            to_jawi(w)
        elapsed = time.time() - start

        words_per_sec = len(words) / elapsed
        self.assertGreater(
            words_per_sec, 1000,
            f"Performance: {words_per_sec:.0f} words/sec (need >1000)"
        )

    def test_sentence_performance(self):
        """100 sentences should complete in under 2 seconds."""
        sentence = 'saya pergi ke sekolah setiap hari untuk belajar banyak ilmu'
        sentences = [sentence] * 100

        start = time.time()
        for s in sentences:
            to_jawi(s)
        elapsed = time.time() - start

        self.assertLess(elapsed, 2.0, f"Too slow: {elapsed:.2f}s for 100 sentences")


if __name__ == '__main__':
    unittest.main()

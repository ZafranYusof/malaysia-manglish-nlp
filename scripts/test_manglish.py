#!/usr/bin/env python3
"""
Unit tests for malaysian-manglish-nlp scripts.
Run: python -m pytest scripts/test_manglish.py -v
Or:  python scripts/test_manglish.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalize import normalize, normalize_preserve_case
from detect_lang import detect_language
from sentiment import analyze_sentiment
from clean import clean_text, clean_for_nlp
from formalize import formalize


class TestNormalize:
    def test_basic_shortforms(self):
        assert 'nak' in normalize('nk pergi')
        assert 'macam' in normalize('mcm tu la')
        assert 'berapa' in normalize('brapa harga')
    
    def test_multiple_shortforms(self):
        result = normalize('nk tnya brapa sem utk grad')
        assert 'nak' in result
        assert 'berapa' in result
        assert 'semester' in result
        assert 'untuk' in result
    
    def test_no_change_normal_words(self):
        assert normalize('hello world') == 'hello world'
    
    def test_negation_shortforms(self):
        assert 'tidak' in normalize('x boleh')
        assert 'tiada' in normalize('xde masa')
    
    def test_preserve_case(self):
        result = normalize_preserve_case('Nk pergi Kedai')
        assert 'Nak' in result
    
    def test_punctuation_preserved(self):
        result = normalize('nk pergi?')
        assert '?' in result
    
    def test_number_suffix(self):
        # "nk2" should normalize to "nak"
        result = normalize('nk2 pergi')
        assert 'nak' in result


class TestDetectLang:
    def test_pure_bm(self):
        result = detect_language('saya nak pergi makan nasi')
        assert result['language'] == 'bm'
    
    def test_pure_en(self):
        result = detect_language('I would like to have some food please')
        assert result['language'] == 'en'
    
    def test_manglish_mixed(self):
        result = detect_language('aku nak pergi makan then balik rumah')
        assert result['language'] == 'manglish'
    
    def test_manglish_particles(self):
        result = detect_language('ok la boleh je kot')
        assert result['language'] in ['manglish', 'bm']
    
    def test_empty_text(self):
        result = detect_language('')
        assert result['language'] == 'unknown'
    
    def test_returns_ratios(self):
        result = detect_language('saya nak the food')
        assert 'bm_ratio' in result
        assert 'en_ratio' in result
        assert 0 <= result['bm_ratio'] <= 1
        assert 0 <= result['en_ratio'] <= 1


class TestSentiment:
    def test_positive_basic(self):
        result = analyze_sentiment('best la makanan dia')
        assert result['sentiment'] == 'positive'
    
    def test_positive_intensified(self):
        result = analyze_sentiment('gila best')
        assert result['sentiment'] == 'positive'
        assert result['raw_score'] > 1.0  # intensified
    
    def test_negative_basic(self):
        result = analyze_sentiment('hampeh la service dia')
        assert result['sentiment'] == 'negative'
    
    def test_negated_positive(self):
        result = analyze_sentiment('tak best langsung')
        assert result['sentiment'] == 'negative'
    
    def test_neutral(self):
        result = analyze_sentiment('aku pergi kedai tadi')
        assert result['sentiment'] == 'neutral'
    
    def test_score_range(self):
        result = analyze_sentiment('gila best power mantap')
        assert -1.0 <= result['score'] <= 1.0
    
    def test_multiple_negative(self):
        result = analyze_sentiment('bodoh sial hampeh')
        assert result['sentiment'] == 'negative'


class TestClean:
    def test_repeated_chars(self):
        assert clean_text('besttttttt') == 'best'
        assert clean_text('gilerrrr') == 'giler'
    
    def test_repeated_punctuation(self):
        assert '???' not in clean_text('hello???')
        assert '?' in clean_text('hello???')
    
    def test_normalize_whitespace(self):
        assert '  ' not in clean_text('hello   world')
    
    def test_laugh_normalize(self):
        assert clean_text('hahahahaha') == 'haha'
        assert clean_text('wkwkwkwk') == 'wkwk'
    
    def test_nlp_mode_strips_urls(self):
        result = clean_for_nlp('check https://google.com ni')
        assert 'https' not in result
    
    def test_nlp_mode_strips_mentions(self):
        result = clean_for_nlp('hey @user check this')
        assert '@user' not in result
    
    def test_double_letter_fix(self):
        assert clean_text('mmakan') == 'makan'


class TestFormalize:
    def test_pronouns(self):
        result = formalize('aku nak pergi')
        assert 'Saya' in result
        assert 'ingin' in result
    
    def test_ko_to_anda(self):
        result = formalize('ko nak ikut')
        assert 'anda' in result.lower()
    
    def test_negation(self):
        result = formalize('x boleh')
        assert 'tidak' in result.lower()
    
    def test_particles_removed(self):
        result = formalize('ok la boleh')
        assert ' la ' not in result.lower()
    
    def test_ends_with_period(self):
        result = formalize('aku nak pergi')
        assert result.endswith('.')
    
    def test_capitalized(self):
        result = formalize('aku nak makan')
        assert result[0].isupper()
    
    def test_connectors(self):
        result = formalize('sbb dia dgn aku')
        assert 'kerana' in result.lower()
        assert 'dengan' in result.lower()


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestNormalize, TestDetectLang, TestSentiment, TestClean, TestFormalize
    ]
    
    total = 0
    passed = 0
    failed = 0
    
    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        print(f"\n{'='*50}")
        print(f"  {cls.__name__} ({len(methods)} tests)")
        print(f"{'='*50}")
        
        for method_name in methods:
            total += 1
            try:
                getattr(instance, method_name)()
                print(f"  PASS {method_name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL {method_name} - {e}")
                failed += 1
            except Exception as e:
                print(f"  FAIL {method_name} - ERROR: {e}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*50}")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

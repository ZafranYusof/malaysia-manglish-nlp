#!/usr/bin/env python3
"""Comprehensive test suite for manglish-nlp package v2.0 (pytest compatible)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malaysian_manglish_nlp import (
    normalize, normalize_preserve_case,
    detect_language,
    sentiment, analyze_sentiment,
    clean, clean_for_nlp,
    formalize,
    tokenize, word_tokenize, sentence_tokenize,
    stem, stem_word,
    segment, segment_text,
    pos_tag,
    ner_tag,
    correct, correct_word,
    extract_keywords,
    is_malay, is_english, classify_word, get_stopwords,
    normalize_elongated, normalize_money, normalize_phone,
    normalize_date, normalize_time, normalize_all,
    available_tasks, load_dictionary,
    similarity, augmentation,
)
from malaysian_manglish_nlp.tokenizer import morpheme_tokenize
from malaysian_manglish_nlp.pos import pos_tag_detailed
from malaysian_manglish_nlp.ner import extract_entities
from malaysian_manglish_nlp.stemmer import get_root
from malaysian_manglish_nlp.augmentation import (
    socialmedia_form, kelantanese_form, vowel_alternate,
    replace_similar_vowels, replace_similar_consonants, synonym, augment,
)


# ============================================================
# NORMALIZE
# ============================================================
class TestNormalize:
    def test_basic_shortforms(self):
        assert normalize("nk tnya brapa sem utk grad") == "nak tanya berapa semester untuk grad"

    def test_mixed(self):
        assert normalize("aku nk pegi mkn") == "aku nak pergi makan"

    def test_preserve_case(self):
        assert normalize_preserve_case("Nk Pergi Kedai") == "Nak Pergi Kedai"

    def test_empty(self):
        assert normalize("") == ""

    def test_no_shortforms(self):
        assert normalize("hello world") == "hello world"

    def test_dgn_utk(self):
        assert normalize("dgn kawan utk makan") == "dengan kawan untuk makan"


# ============================================================
# LANGUAGE DETECTION
# ============================================================
class TestLanguageDetection:
    def test_pure_bm(self):
        r = detect_language("aku nak pergi makan")
        assert r['language'] == 'bm'

    def test_pure_en(self):
        r = detect_language("I want to eat some food")
        assert r['language'] == 'en'

    def test_manglish(self):
        r = detect_language("aku nak go buy some food then balik")
        assert r['language'] == 'manglish'

    def test_empty(self):
        r = detect_language("")
        assert r['language'] == 'unknown'

    def test_has_confidence(self):
        assert 'confidence' in detect_language("test")

    def test_has_word_count(self):
        assert 'word_count' in detect_language("test")


# ============================================================
# SENTIMENT
# ============================================================
class TestSentiment:
    def test_positive(self):
        r = sentiment("gila best makanan dia")
        assert r['sentiment'] == 'positive'

    def test_positive_score(self):
        r = sentiment("gila best makanan dia")
        assert r['score'] > 0

    def test_negative(self):
        r = sentiment("hampeh la service teruk")
        assert r['sentiment'] == 'negative'

    def test_negative_score(self):
        r = sentiment("hampeh la service teruk")
        assert r['score'] < 0

    def test_neutral(self):
        r = sentiment("aku pergi kedai")
        assert r['sentiment'] == 'neutral'

    def test_negated_positive(self):
        r = sentiment("tak best langsung")
        assert r['sentiment'] == 'negative'

    def test_has_raw_score(self):
        assert 'raw_score' in sentiment("test")

    def test_has_positive_words(self):
        assert 'positive_words' in sentiment("test")

    def test_has_negative_words(self):
        assert 'negative_words' in sentiment("test")


# ============================================================
# CLEAN
# ============================================================
class TestClean:
    def test_repeated_chars(self):
        assert clean("besttttt gilerrrr") == "best giler"

    def test_repeated_punct(self):
        assert clean("hello!!!") == "hello!"

    def test_laugh_normalize(self):
        assert clean("hahahahaha") == "haha"

    def test_wkwk_normalize(self):
        assert clean("wkwkwkwk") == "wkwk"

    def test_whitespace(self):
        assert clean("hello    world") == "hello world"

    def test_nlp_removes_urls(self):
        assert "check" in clean_for_nlp("check https://google.com bro")

    def test_nlp_removes_urls2(self):
        assert "https" not in clean_for_nlp("check https://google.com")

    def test_nlp_removes_mentions(self):
        assert "@user" not in clean_for_nlp("hey @user check this")


# ============================================================
# FORMALIZE
# ============================================================
class TestFormalize:
    def test_aku_to_saya(self):
        r = formalize("aku nk pegi kedai jap")
        assert "Saya" in r

    def test_nk_to_ingin(self):
        r = formalize("aku nk pegi kedai jap")
        assert "ingin" in r

    def test_jap_to_sebentar(self):
        r = formalize("aku nk pegi kedai jap")
        assert "sebentar" in r

    def test_ko_to_anda(self):
        r = formalize("ko nk ikut x?")
        assert "Anda" in r

    def test_x_to_tidak(self):
        r = formalize("ko nk ikut x?")
        assert "tidak" in r

    def test_ends_with_punct(self):
        assert formalize("aku nak pergi")[-1] in '.!?'

    def test_capitalized(self):
        assert formalize("aku nak pergi")[0].isupper()


# ============================================================
# TOKENIZER
# ============================================================
class TestTokenizer:
    def test_basic(self):
        assert tokenize("aku nk pergi") == ['aku', 'nk', 'pergi']

    def test_with_punct(self):
        assert tokenize("hello!") == ['hello', '!']

    def test_particles(self):
        assert tokenize("pergi la weh") == ['pergi', 'la', 'weh']

    def test_sentence_split(self):
        sents = sentence_tokenize("Aku nak pergi. Ko nak ikut?")
        assert len(sents) == 2

    def test_morpheme_prefix(self):
        m = morpheme_tokenize("berlarian")
        assert m['prefix'] == 'ber'

    def test_morpheme_has_root(self):
        m = morpheme_tokenize("berlarian")
        assert len(m['root']) > 0


# ============================================================
# STEMMER
# ============================================================
class TestStemmer:
    def test_memakan(self):
        assert stem_word("memakan") == "makan"

    def test_menulis(self):
        assert stem_word("menulis") == "tulis"

    def test_berlari(self):
        assert stem_word("berlari") == "lari"

    def test_pelajaran(self):
        assert stem_word("pelajaran") == "ajar"

    def test_menyapu(self):
        assert stem_word("menyapu") == "sapu"

    def test_mengambil(self):
        assert stem_word("mengambil") == "ambil"

    def test_membaca(self):
        assert stem_word("membaca") == "baca"

    def test_berlarian(self):
        assert stem_word("berlarian") == "lari"

    def test_terbang(self):
        assert stem_word("terbang") == "terbang"

    def test_sekolahan(self):
        assert stem_word("sekolahan") == "sekolah"

    def test_mempersoalkan(self):
        assert stem_word("mempersoalkan") == "soal"

    def test_mendapat(self):
        assert stem_word("mendapat") == "dapat"

    def test_mencari(self):
        assert stem_word("mencari") == "cari"

    def test_memasak(self):
        assert stem_word("memasak") == "masak"

    def test_short_word(self):
        assert stem_word("di") == "di"

    def test_stop_word(self):
        assert stem_word("mereka") == "mereka"

    def test_get_root(self):
        r = get_root("berlarian")
        assert r['root'] == 'lari'

    def test_get_root_prefix(self):
        r = get_root("berlarian")
        assert r['prefix'] == 'ber'

    def test_get_root_suffix(self):
        r = get_root("berlarian")
        assert r['suffix'] == 'an'


# ============================================================
# SEGMENT
# ============================================================
class TestSegment:
    def test_has_segments(self):
        r = segment("aku nak buy some groceries then balik")
        assert len(r['segments']) > 1

    def test_has_switch_count(self):
        r = segment("aku nak buy some groceries then balik")
        assert r['switch_count'] >= 1

    def test_first_segment_bm(self):
        r = segment("aku nak buy some groceries then balik")
        assert r['segments'][0]['lang'] == 'BM'

    def test_pure_bm(self):
        r = segment("aku nak pergi makan")
        assert r['switch_count'] == 0

    def test_pure_en(self):
        r = segment("I want to buy some food")
        assert r['switch_count'] == 0


# ============================================================
# POS TAGGER
# ============================================================
class TestPosTagger:
    def test_pronoun(self):
        r = pos_tag("aku nak pergi kedai")
        assert r[0] == ('aku', 'PRP')

    def test_modal(self):
        r = pos_tag("aku nak pergi kedai")
        assert r[1] == ('nak', 'MD')

    def test_verb(self):
        r = pos_tag("aku nak pergi kedai")
        assert r[2] == ('pergi', 'VB')

    def test_noun(self):
        r = pos_tag("aku nak pergi kedai")
        assert r[3] == ('kedai', 'NN')

    def test_negation(self):
        r = pos_tag("tak nak")
        assert r[0][1] == 'NEG'

    def test_detailed_has_confidence(self):
        r = pos_tag_detailed("aku makan")
        assert 'confidence' in r[0]

    def test_detailed_has_tag_name(self):
        r = pos_tag_detailed("aku makan")
        assert 'tag_name' in r[0]


# ============================================================
# NER
# ============================================================
class TestNer:
    def test_finds_org(self):
        r = ner_tag("Jumpa kat UMPSA KL bayar RM50")
        types = [e['type'] for e in r]
        assert 'ORGANIZATION' in types

    def test_finds_loc(self):
        r = ner_tag("Jumpa kat UMPSA KL bayar RM50")
        types = [e['type'] for e in r]
        assert 'LOCATION' in types

    def test_finds_money(self):
        r = ner_tag("Jumpa kat UMPSA KL bayar RM50")
        types = [e['type'] for e in r]
        assert 'MONEY' in types

    def test_finds_phone(self):
        r = ner_tag("Call 0123456789 or email test@mail.com")
        types = [e['type'] for e in r]
        assert 'PHONE' in types

    def test_finds_email(self):
        r = ner_tag("Call 0123456789 or email test@mail.com")
        types = [e['type'] for e in r]
        assert 'EMAIL' in types

    def test_finds_person(self):
        r = ner_tag("Encik Ahmad dari Johor")
        types = [e['type'] for e in r]
        assert 'PERSON' in types

    def test_extract_grouped(self):
        r = extract_entities("UMPSA kat KL, bayar RM100")
        assert 'ORGANIZATION' in r or 'LOCATION' in r


# ============================================================
# SPELLING
# ============================================================
class TestSpelling:
    def test_corrects_pregi(self):
        r = correct("aku nk pregi mkn")
        assert 'pegi' in r['corrected'] or 'pergi' in r['corrected']

    def test_has_changes(self):
        r = correct("aku nk pregi mkn")
        assert len(r['changes']) > 0

    def test_correct_word_makan(self):
        r = correct_word("mkaan")
        assert r['corrected'] == 'makan' or any(s['word'] == 'makan' for s in r['suggestions'])

    def test_valid_word_unchanged(self):
        r = correct_word("makan")
        assert r['is_valid'] == True


# ============================================================
# KEYWORDS
# ============================================================
class TestKeywords:
    def test_returns_list(self):
        kw = extract_keywords("makanan sedap kat kedai tu harga murah tapi sedap gila")
        assert len(kw) > 0

    def test_has_score(self):
        kw = extract_keywords("makanan sedap kat kedai tu harga murah tapi sedap gila")
        assert 'score' in kw[0]

    def test_has_keyword(self):
        kw = extract_keywords("makanan sedap kat kedai tu harga murah tapi sedap gila")
        assert 'keyword' in kw[0]

    def test_rake_returns_results(self):
        kw = extract_keywords("makanan sedap sangat kat kedai baru tu", method='rake')
        assert len(kw) > 0


# ============================================================
# SIMILARITY
# ============================================================
class TestSimilarity:
    def test_jaccard(self):
        j = similarity.jaccard("aku nak makan nasi", "aku nak makan roti")
        assert j == 0.6

    def test_cosine_identical(self):
        c = similarity.cosine("aku nak makan", "aku nak makan")
        assert c == 1.0

    def test_overlap_subset(self):
        o = similarity.overlap("nak makan", "aku nak makan nasi goreng")
        assert o == 1.0

    def test_semantic_high_score(self):
        sem = similarity.semantic_similarity("nk mkn nasi", "nak makan nasi")
        assert sem['score'] >= 0.8

    def test_find_similar_ranked(self):
        results = similarity.find_most_similar("nk mkn", ["nak makan nasi", "nak pergi", "tidur"])
        assert results[0]['score'] >= results[1]['score']


# ============================================================
# AUGMENTATION
# ============================================================
class TestAugmentation:
    def test_socialmedia_has_variants(self):
        forms = socialmedia_form("makan")
        assert len(forms) >= 3

    def test_socialmedia_has_mkn(self):
        forms = socialmedia_form("makan")
        assert 'mkn' in forms

    def test_kelantan_barang(self):
        kel = kelantanese_form("barang")
        assert 'bare' in kel

    def test_kelantan_makan(self):
        kel = kelantanese_form("makan")
        assert 'make' in kel

    def test_synonym_cantik(self):
        syns = synonym("cantik")
        assert 'lawa' in syns or 'cun' in syns

    def test_augment_returns_variants(self):
        aug = augment("makanan sedap", n=3)
        assert len(aug) >= 1

    def test_augment_different_from_original(self):
        aug = augment("makanan sedap", n=3)
        assert all(v != "makanan sedap" for v in aug)


# ============================================================
# DICTIONARY
# ============================================================
class TestDictionary:
    def test_makan_is_malay(self):
        assert is_malay("makan") == True

    def test_berlari_is_malay(self):
        assert is_malay("berlari") == True

    def test_computer_is_english(self):
        assert is_english("computer") == True

    def test_makan_not_english(self):
        assert is_english("makan") == False

    def test_hospital_is_bm(self):
        r = classify_word("hospital")
        assert r['is_malay'] == True

    def test_stopwords_has_yang(self):
        stops = get_stopwords('bm')
        assert 'yang' in stops

    def test_stopwords_count(self):
        stops = get_stopwords('bm')
        assert len(stops) > 30

    def test_all_stopwords_more_than_bm(self):
        stops_bm = get_stopwords('bm')
        stops_all = get_stopwords('all')
        assert len(stops_all) > len(stops_bm)


# ============================================================
# NORMALIZER (Advanced)
# ============================================================
class TestNormalizer:
    def test_elongated(self):
        assert normalize_elongated("bestttttt gilaaaa") == "best gila"

    def test_money_rm50(self):
        assert normalize_money("harga rm50") == "harga RM50.00"

    def test_date(self):
        assert normalize_date("jumpa 28/5/2026") == "jumpa 28 Mei 2026"

    def test_time_3pm(self):
        assert normalize_time("pukul 3pm") == "pukul 3:00 PM"

    def test_time_1430(self):
        assert normalize_time("meeting 1430") == "meeting 2:30 PM"

    def test_all_normalized_money(self):
        r = normalize_all("besttt rm50 jumpa 28/5/26 pukul 3pm")
        assert "RM50.00" in r['normalized']

    def test_all_has_date(self):
        r = normalize_all("besttt rm50 jumpa 28/5/26 pukul 3pm")
        assert "Mei" in r['normalized']

    def test_all_has_changes(self):
        r = normalize_all("besttt rm50 jumpa 28/5/26 pukul 3pm")
        assert len(r['changes']) > 0


# ============================================================
# UTILS
# ============================================================
class TestUtils:
    def test_has_10_tasks(self):
        tasks = available_tasks()
        assert len(tasks) == 10

    def test_normalize_in_tasks(self):
        tasks = available_tasks()
        assert 'normalize' in tasks

    def test_dict_has_shortforms(self):
        d = load_dictionary()
        assert 'shortforms' in d

    def test_dict_has_slang_positive(self):
        d = load_dictionary()
        assert 'slang_positive' in d

    def test_dict_200_shortforms(self):
        d = load_dictionary()
        assert len(d['shortforms']) >= 200

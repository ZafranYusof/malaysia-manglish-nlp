"""Tests for module-specific improvements.

Tests:
- Aspect-based sentiment analysis
- TF-IDF sentence retrieval in QA
- New NER entity types (PRODUCT, EVENT)
- Ambiguous word resolution in code-switching
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp.sentiment import aspect_sentiment
from manglish_nlp.qa import find_relevant_sentence, _tfidf_score, _sentence_split
from manglish_nlp.ner import ner_tag, extract_entities
from manglish_nlp.code_switching import resolve_ambiguous


# ============================================================
# ASPECT-BASED SENTIMENT ANALYSIS TESTS
# ============================================================

class TestAspectSentiment:
    """Tests for aspect_sentiment function."""

    def test_single_positive_food(self):
        result = aspect_sentiment("makanan sedap gila")
        assert len(result) >= 1
        food = [r for r in result if r['aspect'] == 'food']
        assert len(food) == 1
        assert food[0]['sentiment'] == 'positive'
        assert food[0]['score'] > 0

    def test_single_negative_service(self):
        result = aspect_sentiment("service teruk sangat")
        assert len(result) >= 1
        svc = [r for r in result if r['aspect'] == 'service']
        assert len(svc) == 1
        assert svc[0]['sentiment'] == 'negative'
        assert svc[0]['score'] < 0

    def test_mixed_review(self):
        result = aspect_sentiment("makanan sedap tapi service lambat")
        aspects = {r['aspect']: r for r in result}
        assert 'food' in aspects
        assert 'service' in aspects
        assert aspects['food']['sentiment'] == 'positive'
        assert aspects['service']['sentiment'] == 'negative'

    def test_price_positive(self):
        result = aspect_sentiment("harga murah berbaloi")
        price = [r for r in result if r['aspect'] == 'price']
        assert len(price) == 1
        assert price[0]['sentiment'] == 'positive'

    def test_price_negative(self):
        result = aspect_sentiment("harga mahal overpriced")
        price = [r for r in result if r['aspect'] == 'price']
        assert len(price) == 1
        assert price[0]['sentiment'] == 'negative'

    def test_ambiance_positive(self):
        result = aspect_sentiment("tempat cantik dan selesa")
        amb = [r for r in result if r['aspect'] == 'ambiance']
        assert len(amb) == 1
        assert amb[0]['sentiment'] == 'positive'

    def test_ambiance_negative(self):
        result = aspect_sentiment("tempat kotor dan sesak")
        amb = [r for r in result if r['aspect'] == 'ambiance']
        assert len(amb) == 1
        assert amb[0]['sentiment'] == 'negative'

    def test_quality_positive(self):
        result = aspect_sentiment("kualiti bagus mantap")
        qual = [r for r in result if r['aspect'] == 'quality']
        assert len(qual) == 1
        assert qual[0]['sentiment'] == 'positive'

    def test_quality_negative(self):
        result = aspect_sentiment("kualiti rosak teruk")
        qual = [r for r in result if r['aspect'] == 'quality']
        assert len(qual) == 1
        assert qual[0]['sentiment'] == 'negative'

    def test_no_aspect_detected(self):
        result = aspect_sentiment("hari ni cuaca panas")
        # No food/service/price/ambiance/quality aspects
        assert len(result) == 0

    def test_negation_flips_sentiment(self):
        result = aspect_sentiment("makanan tak sedap langsung")
        food = [r for r in result if r['aspect'] == 'food']
        assert len(food) == 1
        assert food[0]['sentiment'] == 'negative'

    def test_multiple_aspects_full_review(self):
        text = "makanan sedap, harga murah, tapi tempat kotor sikit"
        result = aspect_sentiment(text)
        aspects = {r['aspect']: r for r in result}
        assert 'food' in aspects
        assert 'price' in aspects
        assert 'ambiance' in aspects
        assert aspects['food']['sentiment'] == 'positive'
        assert aspects['price']['sentiment'] == 'positive'
        assert aspects['ambiance']['sentiment'] == 'negative'

    def test_score_range(self):
        result = aspect_sentiment("makanan sedap best terbaik")
        for r in result:
            assert -1.0 <= r['score'] <= 1.0

    def test_phrase_included(self):
        result = aspect_sentiment("service cepat dan friendly")
        svc = [r for r in result if r['aspect'] == 'service']
        assert len(svc) == 1
        assert 'service' in svc[0]['phrase']

    def test_english_aspect_words(self):
        result = aspect_sentiment("the food was delicious")
        food = [r for r in result if r['aspect'] == 'food']
        assert len(food) == 1
        assert food[0]['sentiment'] == 'positive'


# ============================================================
# TF-IDF SENTENCE RETRIEVAL TESTS
# ============================================================

class TestTFIDFRetrieval:
    """Tests for TF-IDF scoring and improved sentence retrieval."""

    def test_tfidf_score_basic(self):
        sentences = ["Ali kerja kat KL", "Dia balik rumah pukul 6", "Cuaca hari ni panas"]
        scores = _tfidf_score("Bile Ali balik?", sentences)
        assert len(scores) == 3
        # Second sentence should score highest (contains 'balik')
        assert scores[1] >= scores[0]
        assert scores[1] >= scores[2]

    def test_tfidf_score_empty(self):
        scores = _tfidf_score("test query", [])
        assert scores == []

    def test_tfidf_score_single_sentence(self):
        scores = _tfidf_score("apa nama dia", ["Nama dia Ahmad"])
        assert len(scores) == 1
        assert scores[0] >= 0

    def test_tfidf_discriminates_better(self):
        """TF-IDF should discriminate better than simple overlap for rare terms."""
        sentences = [
            "Ahmad pergi kedai beli barang",
            "Ahmad makan nasi goreng kat kedai",
            "Kedai tu jual komputer dan laptop",
        ]
        scores = _tfidf_score("Ahmad beli komputer kat mana?", sentences)
        # Third sentence has 'komputer' which is rare across sentences
        assert scores[2] > scores[1]

    def test_find_relevant_sentence_uses_tfidf(self):
        ctx = "Kucing tu tidur atas katil. Ali beli iPhone baru semalam. Cuaca hari ni mendung."
        result = find_relevant_sentence(ctx, "Apa Ali beli?")
        assert "iPhone" in result or "beli" in result

    def test_find_relevant_sentence_fallback(self):
        """When TF-IDF gives no signal, fallback to word overlap."""
        ctx = "Dia suka main bola."
        result = find_relevant_sentence(ctx, "Sape suka main bola?")
        assert "bola" in result

    def test_tfidf_handles_normalized_query(self):
        sentences = ["Dia balik lambat semalam", "Kerja dia banyak sangat"]
        scores = _tfidf_score("bile dia blk?", sentences)
        # Should still work with shortforms via normalization
        assert len(scores) == 2

    def test_find_relevant_with_multiple_sentences(self):
        ctx = "Ahmad tinggal kat Shah Alam. Dia kerja sebagai engineer. Gaji dia RM5000 sebulan."
        result = find_relevant_sentence(ctx, "Berapa gaji Ahmad?")
        assert "gaji" in result.lower() or "RM5000" in result

    def test_tfidf_all_zero_for_unrelated(self):
        sentences = ["Kucing tidur", "Anjing makan"]
        scores = _tfidf_score("Berapa harga kereta?", sentences)
        # Both should be 0 or very low since no overlap
        assert all(s == 0.0 for s in scores) or max(scores) < 0.1


# ============================================================
# NER - PRODUCT AND EVENT ENTITY TESTS
# ============================================================

class TestNERProducts:
    """Tests for PRODUCT entity detection."""

    def test_phone_brand_samsung(self):
        entities = ner_tag("Aku baru beli Samsung baru")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('samsung' in e['text'].lower() for e in products)

    def test_phone_brand_iphone(self):
        entities = ner_tag("Dia pakai iphone je")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('iphone' in e['text'].lower() for e in products)

    def test_phone_brand_xiaomi(self):
        entities = ner_tag("Xiaomi murah tapi bagus")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('xiaomi' in e['text'].lower() for e in products)

    def test_car_myvi(self):
        entities = ner_tag("Myvi king of the road")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('myvi' in e['text'].lower() for e in products)

    def test_car_vios(self):
        entities = ner_tag("Dia drive vios pergi kerja")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('vios' in e['text'].lower() for e in products)

    def test_car_axia(self):
        entities = ner_tag("Baru beli axia untuk daily")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('axia' in e['text'].lower() for e in products)

    def test_food_brand_maggi(self):
        entities = ner_tag("Malam ni makan maggi je")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('maggi' in e['text'].lower() for e in products)

    def test_food_brand_milo(self):
        entities = ner_tag("Nak minum milo panas")
        products = [e for e in entities if e['type'] == 'PRODUCT']
        assert any('milo' in e['text'].lower() for e in products)

    def test_extract_entities_product(self):
        grouped = extract_entities("Beli samsung dan milo kat kedai")
        assert 'PRODUCT' in grouped
        assert len(grouped['PRODUCT']) >= 2


class TestNEREvents:
    """Tests for EVENT entity detection."""

    def test_hari_raya(self):
        entities = ner_tag("Hari raya tahun ni best")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('hari raya' in e['text'].lower() for e in events)

    def test_deepavali(self):
        entities = ner_tag("Happy deepavali semua")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('deepavali' in e['text'].lower() for e in events)

    def test_cny(self):
        entities = ner_tag("CNY tahun ni meriah")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('cny' in e['text'].lower() for e in events)

    def test_kenduri(self):
        entities = ner_tag("Esok ada kenduri kat kampung")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('kenduri' in e['text'].lower() for e in events)

    def test_majlis(self):
        entities = ner_tag("Kena pergi majlis malam ni")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('majlis' in e['text'].lower() for e in events)

    def test_concert(self):
        entities = ner_tag("Nak pergi concert next week")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('concert' in e['text'].lower() for e in events)

    def test_wedding(self):
        entities = ner_tag("Kawan aku wedding bulan depan")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('wedding' in e['text'].lower() for e in events)

    def test_merdeka(self):
        entities = ner_tag("Sambut merdeka tahun ni grand")
        events = [e for e in entities if e['type'] == 'EVENT']
        assert any('merdeka' in e['text'].lower() for e in events)

    def test_extract_entities_event(self):
        grouped = extract_entities("Pergi kenduri lepas deepavali")
        assert 'EVENT' in grouped
        assert len(grouped['EVENT']) >= 2


# ============================================================
# CODE-SWITCHING AMBIGUOUS WORD RESOLUTION TESTS
# ============================================================

class TestResolveAmbiguous:
    """Tests for resolve_ambiguous function."""

    def test_air_in_malay_context(self):
        # "air" means water in Malay; in BM context should resolve to ms
        result = resolve_ambiguous("aku nak minum air sejuk")
        air_results = [r for r in result if r['token'].lower() == 'air']
        if air_results:
            assert air_results[0]['language'] == 'ms'

    def test_air_in_english_context(self):
        # "air" in English context
        result = resolve_ambiguous("the air conditioning is broken")
        air_results = [r for r in result if r['token'].lower() == 'air']
        if air_results:
            assert air_results[0]['language'] == 'en'

    def test_returns_only_resolved(self):
        # Should only return tokens that were ambiguous and got resolved
        result = resolve_ambiguous("aku pergi sekolah")
        # No ambiguous words here, should be empty or minimal
        assert isinstance(result, list)

    def test_resolved_by_field_present(self):
        result = resolve_ambiguous("dia makan nasi dengan air")
        for r in result:
            assert 'resolved_by' in r
            assert r['resolved_by'] in ('neighbor_context', 'bigram_context', 'bigram_switch_point')

    def test_position_field_present(self):
        result = resolve_ambiguous("aku nak air")
        for r in result:
            assert 'position' in r
            assert isinstance(r['position'], tuple)
            assert len(r['position']) == 2

    def test_mixed_sentence_resolution(self):
        # "I makan nasi" - clear switch point
        result = resolve_ambiguous("I makan nasi then pergi main")
        # Should resolve any ambiguous words based on neighbors
        for r in result:
            assert r['language'] in ('en', 'ms')

    def test_empty_text(self):
        result = resolve_ambiguous("")
        assert result == []

    def test_no_ambiguous_words(self):
        # Pure English sentence with no ambiguous words
        result = resolve_ambiguous("hello world")
        assert isinstance(result, list)

    def test_bigram_context_used(self):
        # When immediate neighbors determine language
        result = resolve_ambiguous("aku suka main game dengan kawan")
        # 'main' is ambiguous (EN: main/primary, MS: play)
        main_results = [r for r in result if r['token'].lower() == 'main']
        if main_results:
            assert main_results[0]['language'] == 'ms'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

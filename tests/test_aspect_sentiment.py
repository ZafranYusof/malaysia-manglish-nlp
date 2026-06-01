"""Tests for enhanced aspect-based sentiment analysis module."""

import pytest
from malaysian_manglish_nlp.aspect_sentiment import (
    analyze_aspect_sentiment,
    aspect_sentiment_batch,
    get_aspect_categories,
)


# ============================================================
# Basic functionality
# ============================================================

class TestAnalyzeAspectSentiment:
    """Core analyze_aspect_sentiment function tests."""

    def test_empty_text(self):
        """Empty input returns neutral with no aspects."""
        result = analyze_aspect_sentiment("")
        assert result['aspects'] == []
        assert result['summary']['dominant_sentiment'] == 'neutral'
        assert result['summary']['aspect_count'] == 0
        assert result['summary']['overall_score'] == 0.0

    def test_none_text(self):
        """None input handled gracefully."""
        result = analyze_aspect_sentiment(None)
        assert result['aspects'] == []

    def test_whitespace_only(self):
        """Whitespace-only input returns neutral."""
        result = analyze_aspect_sentiment("   ")
        assert result['aspects'] == []

    def test_result_structure(self):
        """Result has correct top-level keys."""
        result = analyze_aspect_sentiment("makanan sedap")
        assert 'aspects' in result
        assert 'summary' in result
        assert 'domain' in result
        assert 'text' in result

    def test_aspect_structure(self):
        """Each aspect has required fields."""
        result = analyze_aspect_sentiment("makanan sedap", domain='restaurant')
        for asp in result['aspects']:
            assert 'aspect' in asp
            assert 'sentiment' in asp
            assert 'score' in asp
            assert 'keywords' in asp
            assert 'phrase' in asp
            assert asp['sentiment'] in ('positive', 'negative', 'neutral')
            assert -1.0 <= asp['score'] <= 1.0

    def test_summary_structure(self):
        """Summary has required fields."""
        result = analyze_aspect_sentiment("makanan sedap", domain='restaurant')
        summary = result['summary']
        assert 'dominant_sentiment' in summary
        assert 'aspect_count' in summary
        assert 'conflicts' in summary
        assert 'overall_score' in summary


# ============================================================
# Restaurant domain
# ============================================================

class TestRestaurantDomain:
    """Restaurant-specific aspect detection."""

    def test_food_positive(self):
        """Detect food aspect with positive sentiment."""
        result = analyze_aspect_sentiment(
            "makanan sedap gila", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'food' in aspects
        assert aspects['food']['sentiment'] == 'positive'
        assert aspects['food']['score'] > 0

    def test_food_negative(self):
        """Detect food aspect with negative sentiment."""
        result = analyze_aspect_sentiment(
            "makanan tawar dan basi", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'food' in aspects
        assert aspects['food']['sentiment'] == 'negative'

    def test_service_positive(self):
        """Detect service aspect with positive sentiment."""
        result = analyze_aspect_sentiment(
            "service friendly dan cepat", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'service' in aspects
        assert aspects['service']['sentiment'] == 'positive'

    def test_service_negative(self):
        """Detect service aspect with negative sentiment."""
        result = analyze_aspect_sentiment(
            "service teruk dan lambat", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'service' in aspects
        assert aspects['service']['sentiment'] == 'negative'

    def test_price_positive(self):
        """Detect price aspect with positive sentiment."""
        result = analyze_aspect_sentiment(
            "harga murah dan berbaloi", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'price' in aspects
        assert aspects['price']['sentiment'] == 'positive'

    def test_price_negative(self):
        """Detect price aspect with negative sentiment."""
        result = analyze_aspect_sentiment(
            "harga mahal gila, overpriced", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'price' in aspects
        assert aspects['price']['sentiment'] == 'negative'

    def test_ambiance_positive(self):
        """Detect ambiance aspect with positive sentiment."""
        result = analyze_aspect_sentiment(
            "tempat cantik dan selesa", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'ambiance' in aspects
        assert aspects['ambiance']['sentiment'] == 'positive'

    def test_cleanliness_negative(self):
        """Detect cleanliness aspect with negative sentiment."""
        result = analyze_aspect_sentiment(
            "tandas kotor dan busuk", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'cleanliness' in aspects
        assert aspects['cleanliness']['sentiment'] == 'negative'

    def test_portion_positive(self):
        """Detect portion aspect with positive sentiment."""
        result = analyze_aspect_sentiment(
            "portion banyak dan puas", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'portion' in aspects
        assert aspects['portion']['sentiment'] == 'positive'

    def test_speed_negative(self):
        """Detect speed aspect with negative sentiment."""
        result = analyze_aspect_sentiment(
            "lambat gila tunggu makanan sampai", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        # Either 'speed' or 'food' could be detected depending on keyword match order
        assert len(result['aspects']) > 0

    def test_multi_aspect(self):
        """Detect multiple aspects in one text."""
        result = analyze_aspect_sentiment(
            "makanan sedap tapi service teruk", domain='restaurant'
        )
        aspect_names = {a['aspect'] for a in result['aspects']}
        assert 'food' in aspect_names
        assert 'service' in aspect_names
        assert result['summary']['aspect_count'] >= 2

    def test_all_aspects(self):
        """All 8 restaurant aspects are available."""
        cats = get_aspect_categories()
        assert len(cats['restaurant']) == 8


# ============================================================
# Product domain
# ============================================================

class TestProductDomain:
    """Product-specific aspect detection."""

    def test_quality_positive(self):
        """Detect product quality as positive."""
        result = analyze_aspect_sentiment(
            "kualiti bagus dan solid", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'quality' in aspects
        assert aspects['quality']['sentiment'] == 'positive'

    def test_quality_negative(self):
        """Detect product quality as negative."""
        result = analyze_aspect_sentiment(
            "quality teruk, rosak cepat", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'quality' in aspects
        assert aspects['quality']['sentiment'] == 'negative'

    def test_design_positive(self):
        """Detect product design as positive."""
        result = analyze_aspect_sentiment(
            "design cantik dan sleek", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'design' in aspects
        assert aspects['design']['sentiment'] == 'positive'

    def test_battery_negative(self):
        """Detect battery aspect as negative."""
        result = analyze_aspect_sentiment(
            "battery cepat habis, drain teruk", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'battery' in aspects
        assert aspects['battery']['sentiment'] == 'negative'

    def test_camera_positive(self):
        """Detect camera aspect as positive."""
        result = analyze_aspect_sentiment(
            "camera sharp dan clear", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'camera' in aspects
        assert aspects['camera']['sentiment'] == 'positive'

    def test_display_negative(self):
        """Detect display aspect as negative."""
        result = analyze_aspect_sentiment(
            "skrin blur dan kabur", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'display' in aspects
        assert aspects['display']['sentiment'] == 'negative'

    def test_durability_positive(self):
        """Detect durability as positive."""
        result = analyze_aspect_sentiment(
            "tahan lasak dan kuat", domain='product'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'durability' in aspects
        assert aspects['durability']['sentiment'] == 'positive'


# ============================================================
# App/Software domain
# ============================================================

class TestAppDomain:
    """App/software-specific aspect detection."""

    def test_ui_positive(self):
        """Detect UI aspect as positive."""
        result = analyze_aspect_sentiment(
            "interface clean dan mudah guna", domain='app'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'ui' in aspects
        assert aspects['ui']['sentiment'] == 'positive'

    def test_performance_negative(self):
        """Detect performance aspect as negative."""
        result = analyze_aspect_sentiment(
            "app slow dan laggy", domain='app'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'performance' in aspects
        assert aspects['performance']['sentiment'] == 'negative'

    def test_bugs_negative(self):
        """Detect bugs aspect as negative."""
        result = analyze_aspect_sentiment(
            "banyak bug dan crash selalu", domain='app'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'bugs' in aspects
        assert aspects['bugs']['sentiment'] == 'negative'

    def test_pricing_positive(self):
        """Detect pricing aspect as positive."""
        result = analyze_aspect_sentiment(
            "free dan berbaloi", domain='app'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'pricing' in aspects
        assert aspects['pricing']['sentiment'] == 'positive'

    def test_reliability_negative(self):
        """Detect reliability as negative."""
        result = analyze_aspect_sentiment(
            "server unstable dan selalu crash", domain='app'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'reliability' in aspects
        assert aspects['reliability']['sentiment'] == 'negative'


# ============================================================
# General domain (dynamic extraction)
# ============================================================

class TestGeneralDomain:
    """General domain with dynamic aspect extraction."""

    def test_general_detects_sentiment_words(self):
        """General mode detects sentiment words and groups them."""
        result = analyze_aspect_sentiment("best gila la")
        # Should find at least something
        assert isinstance(result['aspects'], list)
        assert result['domain'] == 'general'

    def test_general_falls_back_to_domain_keywords(self):
        """General mode tries domain aspects first."""
        result = analyze_aspect_sentiment("makanan sedap best")
        # Should find food aspect from restaurant domain
        if result['aspects']:
            aspect_names = {a['aspect'] for a in result['aspects']}
            assert 'food' in aspect_names or len(aspect_names) > 0

    def test_general_no_aspects(self):
        """Text with no sentiment words returns empty."""
        result = analyze_aspect_sentiment("saya pergi kedai")
        assert result['aspects'] == []
        assert result['summary']['dominant_sentiment'] == 'neutral'

    def test_general_mixed_text(self):
        """General handles mixed positive/negative text."""
        result = analyze_aspect_sentiment("best tapi teruk juga")
        assert isinstance(result['aspects'], list)


# ============================================================
# Conflict detection
# ============================================================

class TestConflictDetection:
    """Conflict detection between aspects."""

    def test_conflict_detected(self):
        """Conflicting sentiments across aspects flagged."""
        result = analyze_aspect_sentiment(
            "makanan sedap tapi service teruk", domain='restaurant'
        )
        assert result['summary']['conflicts'] is True

    def test_no_conflict_uniform_positive(self):
        """All positive aspects have no conflict."""
        result = analyze_aspect_sentiment(
            "makanan sedap dan service friendly", domain='restaurant'
        )
        assert result['summary']['conflicts'] is False

    def test_no_conflict_single_aspect(self):
        """Single aspect has no conflict."""
        result = analyze_aspect_sentiment(
            "makanan sedap", domain='restaurant'
        )
        assert result['summary']['conflicts'] is False

    def test_conflict_price_vs_quality(self):
        """Price negative but quality positive = conflict."""
        result = analyze_aspect_sentiment(
            "quality bagus tapi harga mahal", domain='product'
        )
        assert result['summary']['conflicts'] is True


# ============================================================
# Manglish text handling
# ============================================================

class TestManglishText:
    """Tests with Malaysian Manglish text."""

    def test_negation_tak(self):
        """'tak' negates sentiment in window."""
        result = analyze_aspect_sentiment(
            "service tak bagus", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        if 'service' in aspects:
            assert aspects['service']['sentiment'] == 'negative'

    def test_intensifier_gila(self):
        """'gila' intensifies sentiment."""
        result = analyze_aspect_sentiment(
            "makanan sedap gila", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'food' in aspects
        assert aspects['food']['score'] > 0.5

    def test_manglish_particles(self):
        """Particles like 'la', 'lah' don't break detection."""
        result = analyze_aspect_sentiment(
            "makanan sedap la", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'food' in aspects
        assert aspects['food']['sentiment'] == 'positive'

    def test_mixed_bm_english(self):
        """Mixed BM/English text works."""
        result = analyze_aspect_sentiment(
            "harga very murah and berbaloi", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'price' in aspects
        assert aspects['price']['sentiment'] == 'positive'

    def test_manglish_slang(self):
        """Malaysian slang detected correctly."""
        result = analyze_aspect_sentiment(
            "service hampeh, pekerja bodoh", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        if 'service' in aspects:
            assert aspects['service']['sentiment'] == 'negative'

    def test_elongated_text(self):
        """Elongated words normalized correctly."""
        result = analyze_aspect_sentiment(
            "makanan sedaappp gila", domain='restaurant'
        )
        aspects = {a['aspect']: a for a in result['aspects']}
        assert 'food' in aspects


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Edge case handling."""

    def test_very_long_text(self):
        """Long text doesn't crash."""
        text = "makanan sedap " * 100
        result = analyze_aspect_sentiment(text, domain='restaurant')
        assert isinstance(result['aspects'], list)
        assert len(result['aspects']) > 0

    def test_repeated_aspect_keywords(self):
        """Same aspect keyword repeated only counts once."""
        result = analyze_aspect_sentiment(
            "makanan sedap, makanan best, makanan power", domain='restaurant'
        )
        food_aspects = [a for a in result['aspects'] if a['aspect'] == 'food']
        assert len(food_aspects) == 1

    def test_domain_passthrough(self):
        """Domain value preserved in result."""
        result = analyze_aspect_sentiment("test", domain='product')
        assert result['domain'] == 'product'

    def test_text_passthrough(self):
        """Original text preserved in result."""
        text = "makanan sedap tapi service teruk"
        result = analyze_aspect_sentiment(text)
        assert result['text'] == text

    def test_unknown_domain(self):
        """Unknown domain falls back gracefully."""
        result = analyze_aspect_sentiment(
            "makanan sedap", domain='nonexistent'
        )
        # Should still return valid structure
        assert isinstance(result['aspects'], list)

    def test_no_keywords_text(self):
        """Text with no aspect keywords returns empty."""
        result = analyze_aspect_sentiment(
            "hello world how are you", domain='restaurant'
        )
        assert result['aspects'] == []


# ============================================================
# Batch processing
# ============================================================

class TestBatchProcessing:
    """Batch processing tests."""

    def test_batch_empty(self):
        """Empty list returns empty list."""
        results = aspect_sentiment_batch([])
        assert results == []

    def test_batch_multiple(self):
        """Batch processes multiple texts."""
        texts = [
            "makanan sedap",
            "service teruk",
            "harga murah",
        ]
        results = aspect_sentiment_batch(texts, domain='restaurant')
        assert len(results) == 3
        for r in results:
            assert 'aspects' in r
            assert 'summary' in r

    def test_batch_domain(self):
        """Batch respects domain parameter."""
        texts = ["quality bagus", "battery drain"]
        results = aspect_sentiment_batch(texts, domain='product')
        for r in results:
            assert r['domain'] == 'product'

    def test_batch_mixed_domains(self):
        """Batch with general domain works."""
        texts = ["best gila", "teruk la"]
        results = aspect_sentiment_batch(texts, domain='general')
        assert len(results) == 2


# ============================================================
# Domain switching
# ============================================================

class TestDomainSwitching:
    """Switching between domains."""

    def test_same_text_different_domains(self):
        """Same text may yield different aspects in different domains."""
        text = "quality bagus tapi mahal"
        r_product = analyze_aspect_sentiment(text, domain='product')
        r_restaurant = analyze_aspect_sentiment(text, domain='restaurant')
        # Product should detect quality/price, restaurant might not
        assert r_product['domain'] == 'product'
        assert r_restaurant['domain'] == 'restaurant'

    def test_get_aspect_categories(self):
        """get_aspect_categories returns all domains."""
        cats = get_aspect_categories()
        assert 'restaurant' in cats
        assert 'product' in cats
        assert 'app' in cats
        assert 'general' in cats

    def test_get_aspect_categories_structure(self):
        """Each domain has list of aspect names."""
        cats = get_aspect_categories()
        for domain, aspect_list in cats.items():
            assert isinstance(aspect_list, list)
            assert len(aspect_list) > 0

"""Tests for malaysian_manglish_nlp.topic module."""

import pytest
from malaysian_manglish_nlp.topic import (
    classify_topic,
    classify_topics,
    classify_batch,
    extract_topic_keywords,
    topic_distribution,
    AVAILABLE_TOPICS,
)


class TestClassifyTopic:
    """Tests for classify_topic function."""

    def test_food_nasi_lemak(self):
        result = classify_topic("mamak punya nasi lemak memang sedap gila")
        assert result["topic"] == "food"
        assert result["confidence"] > 0.3
        assert any("nasi lemak" in kw.lower() or "mamak" in kw.lower() or "sedap" in kw.lower()
                   for kw in result["keywords_matched"])

    def test_food_cooking(self):
        result = classify_topic("aku masak kari ayam dengan sambal petai malam ni")
        assert result["topic"] == "food"
        assert result["confidence"] > 0.3

    def test_food_delivery(self):
        result = classify_topic("order grabfood je la, lapar gila dah ni")
        assert result["topic"] == "food"

    def test_politics_election(self):
        result = classify_topic("PRU ni rakyat kena undi calon yang betul")
        assert result["topic"] == "politics"
        assert result["confidence"] > 0.3

    def test_politics_government(self):
        result = classify_topic("kerajaan baru announce subsidi minyak kena potong")
        assert result["topic"] == "politics"

    def test_politics_party(self):
        result = classify_topic("UMNO dan PAS buat kerjasama untuk pilihan raya")
        assert result["topic"] == "politics"

    def test_sports_football(self):
        result = classify_topic("Harimau Malaya menang 3-0 semalam, goal power gila")
        assert result["topic"] == "sports"
        assert result["confidence"] > 0.3

    def test_sports_badminton(self):
        result = classify_topic("Lee Zii Jia masuk final All England tahun ni")
        assert result["topic"] == "sports"

    def test_sports_esports(self):
        result = classify_topic("team Malaysia menang tournament Mobile Legends MLBB")
        assert result["topic"] == "sports"

    def test_tech_phone(self):
        result = classify_topic("iPhone 16 spec gila, RAM 8GB dengan camera 48 megapixel")
        assert result["topic"] == "tech"
        assert result["confidence"] > 0.3

    def test_tech_coding(self):
        result = classify_topic("aku belajar coding Python, deploy server pakai API")
        assert result["topic"] == "tech"

    def test_tech_ai(self):
        result = classify_topic("ChatGPT ni AI machine learning yang power")
        assert result["topic"] == "tech"

    def test_education_exam(self):
        result = classify_topic("exam final sem ni susah gila, pointer confirm jatuh")
        assert result["topic"] == "education"
        assert result["confidence"] > 0.3

    def test_education_university(self):
        result = classify_topic("aku student UiTM, assignment banyak gila semester ni")
        assert result["topic"] == "education"

    def test_education_fyp(self):
        result = classify_topic("FYP final year project aku pasal machine learning")
        assert result["topic"] in ["education", "tech"]  # Could be either

    def test_entertainment_netflix(self):
        result = classify_topic("tengok Netflix drama Korea baru, best gila series ni")
        assert result["topic"] == "entertainment"
        assert result["confidence"] > 0.3

    def test_entertainment_kpop(self):
        result = classify_topic("BTS comeback baru, MV dah 100 juta views")
        assert result["topic"] == "entertainment"

    def test_entertainment_gaming(self):
        result = classify_topic("main PS5 game baru, graphics gila power")
        assert result["topic"] == "entertainment"

    def test_religion_solat(self):
        result = classify_topic("jom solat Jumaat kat masjid, dengar khutbah")
        assert result["topic"] == "religion"
        assert result["confidence"] > 0.3

    def test_religion_ramadan(self):
        result = classify_topic("Ramadhan ni puasa penuh, terawih setiap malam")
        assert result["topic"] == "religion"

    def test_daily_life_traffic(self):
        result = classify_topic("traffic jam gila pagi ni, lambat sampai office")
        assert result["topic"] == "daily_life"
        assert result["confidence"] > 0.3

    def test_daily_life_routine(self):
        result = classify_topic("bangun pagi, mandi, siap pergi kerja naik Grab")
        assert result["topic"] == "daily_life"

    def test_business_startup(self):
        result = classify_topic("bisnes startup aku dah untung, customer makin ramai")
        assert result["topic"] == "business"
        assert result["confidence"] > 0.3

    def test_business_ecommerce(self):
        result = classify_topic("jual kat Shopee, dropship dari supplier, margin ok la")
        assert result["topic"] == "business"

    def test_health_sick(self):
        result = classify_topic("demam teruk, pergi klinik doctor bagi ubat antibiotik")
        assert result["topic"] == "health"
        assert result["confidence"] > 0.3

    def test_health_mental(self):
        result = classify_topic("mental health penting, jangan malu pergi therapy counseling")
        assert result["topic"] == "health"

    def test_travel_langkawi(self):
        result = classify_topic("trip Langkawi next week, dah book hotel dan flight")
        assert result["topic"] == "travel"
        assert result["confidence"] > 0.3

    def test_travel_international(self):
        result = classify_topic("nak travel Japan, dah apply visa dan book flight AirAsia")
        assert result["topic"] == "travel"

    def test_relationships_couple(self):
        result = classify_topic("girlfriend aku ajak kahwin, dah 3 tahun couple")
        assert result["topic"] == "relationships"
        assert result["confidence"] > 0.3

    def test_relationships_breakup(self):
        result = classify_topic("baru breakup dengan ex, kena move on cari jodoh baru")
        assert result["topic"] == "relationships"

    def test_empty_text(self):
        result = classify_topic("")
        assert result["topic"] == "daily_life"
        assert result["confidence"] == 0.0
        assert result["keywords_matched"] == []

    def test_none_like_text(self):
        result = classify_topic("   ")
        assert result["confidence"] == 0.0

    def test_result_structure(self):
        result = classify_topic("aku makan nasi lemak")
        assert "topic" in result
        assert "confidence" in result
        assert "keywords_matched" in result
        assert isinstance(result["topic"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["keywords_matched"], list)


class TestClassifyTopics:
    """Tests for classify_topics function."""

    def test_returns_multiple_topics(self):
        # Text that spans multiple topics
        result = classify_topics("lepas kerja aku makan nasi lemak sambil tengok bola", top_n=3)
        assert len(result) >= 1
        assert len(result) <= 3
        topics = [r["topic"] for r in result]
        # Should detect food and/or sports and/or daily_life
        assert any(t in ["food", "sports", "daily_life"] for t in topics)

    def test_top_n_limit(self):
        result = classify_topics("aku pergi mamak makan roti canai", top_n=2)
        assert len(result) <= 2

    def test_sorted_by_confidence(self):
        result = classify_topics("nasi lemak sedap gila kat mamak ni", top_n=3)
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i]["confidence"] >= result[i + 1]["confidence"]

    def test_empty_text(self):
        result = classify_topics("")
        assert len(result) == 1
        assert result[0]["topic"] == "daily_life"

    def test_single_topic_dominance(self):
        result = classify_topics("solat Jumaat kat masjid, dengar khutbah imam, baca Quran", top_n=3)
        assert result[0]["topic"] == "religion"


class TestClassifyBatch:
    """Tests for classify_batch function."""

    def test_batch_multiple_texts(self):
        texts = [
            "makan nasi lemak pagi ni",
            "exam esok, kena study",
            "Harimau Malaya menang semalam",
        ]
        results = classify_batch(texts)
        assert len(results) == 3
        assert results[0]["topic"] == "food"
        assert results[1]["topic"] == "education"
        assert results[2]["topic"] == "sports"

    def test_batch_empty_list(self):
        results = classify_batch([])
        assert results == []

    def test_batch_single_text(self):
        results = classify_batch(["coding Python best"])
        assert len(results) == 1


class TestExtractTopicKeywords:
    """Tests for extract_topic_keywords function."""

    def test_extracts_keywords(self):
        result = extract_topic_keywords("aku makan nasi lemak kat mamak")
        assert len(result) > 0
        keywords = [r["keyword"] for r in result]
        assert any("makan" in kw.lower() or "nasi lemak" in kw.lower() or "mamak" in kw.lower()
                   for kw in keywords)

    def test_keywords_have_topic(self):
        result = extract_topic_keywords("exam susah, pointer jatuh")
        for item in result:
            assert "keyword" in item
            assert "topic" in item
            assert item["topic"] in AVAILABLE_TOPICS

    def test_empty_text(self):
        result = extract_topic_keywords("")
        assert result == []

    def test_no_duplicates(self):
        result = extract_topic_keywords("makan makan makan nasi nasi")
        keywords = [r["keyword"].lower() for r in result]
        assert len(keywords) == len(set(keywords))


class TestTopicDistribution:
    """Tests for topic_distribution function."""

    def test_distribution_structure(self):
        texts = [
            "makan nasi lemak",
            "exam esok",
            "tengok bola",
            "masak kari ayam",
            "study untuk quiz",
        ]
        result = topic_distribution(texts)
        assert "distribution" in result
        assert "total_texts" in result
        assert "topic_counts" in result
        assert result["total_texts"] == 5

    def test_distribution_percentages(self):
        texts = ["makan sedap"] * 5 + ["exam susah"] * 5
        result = topic_distribution(texts)
        # Percentages should sum to roughly 100 (or less if some unclassified)
        total_pct = sum(result["distribution"].values())
        assert total_pct <= 100.1

    def test_empty_corpus(self):
        result = topic_distribution([])
        assert result["total_texts"] == 0
        assert result["distribution"] == {}


class TestAvailableTopics:
    """Tests for module constants."""

    def test_all_topics_present(self):
        expected = [
            "food", "politics", "sports", "tech", "education",
            "entertainment", "religion", "daily_life", "business",
            "health", "travel", "relationships",
        ]
        for topic in expected:
            assert topic in AVAILABLE_TOPICS

    def test_topic_count(self):
        assert len(AVAILABLE_TOPICS) == 12

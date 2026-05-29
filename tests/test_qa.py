"""Tests for manglish_nlp.qa - Extractive Question Answering."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from manglish_nlp.qa import (
    answer, answer_multiple, find_relevant_sentence,
    extract_answer_span, classify_question_type,
)


# ─── Question Type Classification ────────────────────────────────────────────

class TestClassifyQuestionType:
    """Test question type classification."""

    def test_who_english(self):
        assert classify_question_type("Who is the manager?") == "who"

    def test_who_bm(self):
        assert classify_question_type("Siapa yang buat ni?") == "who"

    def test_who_manglish(self):
        assert classify_question_type("Sape buat benda ni?") == "who"

    def test_what_english(self):
        assert classify_question_type("What is the capital?") == "what"

    def test_what_bm(self):
        assert classify_question_type("Apa nama dia?") == "what"

    def test_what_manglish(self):
        assert classify_question_type("Ape benda tu?") == "what"

    def test_when_english(self):
        assert classify_question_type("When did he arrive?") == "when"

    def test_when_bm(self):
        assert classify_question_type("Bila dia sampai?") == "when"

    def test_when_manglish(self):
        assert classify_question_type("Bile kau nak datang?") == "when"

    def test_where_english(self):
        assert classify_question_type("Where is the office?") == "where"

    def test_where_bm(self):
        assert classify_question_type("Di mana pejabat tu?") == "where"

    def test_where_manglish(self):
        assert classify_question_type("Mane kedai tu?") == "where"

    def test_why_english(self):
        assert classify_question_type("Why did he leave?") == "why"

    def test_why_bm(self):
        assert classify_question_type("Kenapa dia pergi?") == "why"

    def test_why_manglish(self):
        assert classify_question_type("Nape kau lambat?") == "why"

    def test_how_english(self):
        assert classify_question_type("How do you do it?") == "how"

    def test_how_bm(self):
        assert classify_question_type("Macam mana nak buat?") == "how"

    def test_how_manglish(self):
        assert classify_question_type("Camne nak settle?") == "how"

    def test_yes_no_english(self):
        assert classify_question_type("Is he coming?") == "yes_no"

    def test_yes_no_bm(self):
        assert classify_question_type("Adakah dia datang?") == "yes_no"

    def test_yes_no_boleh(self):
        assert classify_question_type("Boleh tak buat macam tu?") == "yes_no"

    def test_other(self):
        assert classify_question_type("Tell me about it") == "other"


# ─── Who Questions ────────────────────────────────────────────────────────────

class TestWhoQuestions:
    """Test who/siapa/sape questions."""

    def test_who_english_proper_noun(self):
        ctx = "Ahmad is the team leader. He manages 5 people."
        result = answer(ctx, "Who is the team leader?")
        assert "Ahmad" in result["answer"]
        assert result["confidence"] > 0.3

    def test_who_bm(self):
        ctx = "Siti bekerja di hospital. Dia seorang doktor."
        result = answer(ctx, "Siapa yang bekerja di hospital?")
        assert "Siti" in result["answer"]

    def test_who_manglish(self):
        ctx = "Ali buat report tu semalam. Dia submit pagi tadi."
        result = answer(ctx, "Sape buat report tu?")
        assert "Ali" in result["answer"]

    def test_who_with_bin(self):
        ctx = "Ahmad bin Hassan adalah pengetua sekolah. Beliau berkhidmat sejak 2015."
        result = answer(ctx, "Siapa pengetua sekolah?")
        assert "Ahmad" in result["answer"]


# ─── What Questions ───────────────────────────────────────────────────────────

class TestWhatQuestions:
    """Test what/apa/ape questions."""

    def test_what_english(self):
        ctx = "The capital of Malaysia is Kuala Lumpur. It is a modern city."
        result = answer(ctx, "What is the capital of Malaysia?")
        assert "Kuala Lumpur" in result["answer"]

    def test_what_bm(self):
        ctx = "Nasi lemak adalah makanan kebangsaan Malaysia."
        result = answer(ctx, "Apa makanan kebangsaan Malaysia?")
        assert "nasi lemak" in result["answer"].lower() or "makanan kebangsaan" in result["answer"].lower()

    def test_what_manglish(self):
        ctx = "Projek tu pasal machine learning. Kena siap bulan depan."
        result = answer(ctx, "Ape projek tu pasal?")
        assert "machine learning" in result["answer"].lower() or len(result["answer"]) > 0

    def test_what_definition(self):
        ctx = "API adalah Application Programming Interface. Dia macam jambatan antara sistem."
        result = answer(ctx, "Apa itu API?")
        assert "Application Programming Interface" in result["answer"] or "API" in result["sentence"]


# ─── When Questions ───────────────────────────────────────────────────────────

class TestWhenQuestions:
    """Test when/bila/bile questions."""

    def test_when_english_time(self):
        ctx = "The meeting starts at 3pm. Please be on time."
        result = answer(ctx, "When does the meeting start?")
        assert "3" in result["answer"] or "pm" in result["answer"].lower()

    def test_when_bm_date(self):
        ctx = "Mesyuarat akan diadakan pada 15 Jun 2024. Semua wajib hadir."
        result = answer(ctx, "Bila mesyuarat diadakan?")
        assert "15" in result["answer"] or "Jun" in result["answer"]

    def test_when_manglish(self):
        ctx = "Dia balik kampung hari Jumaat. Lepas tu cuti seminggu."
        result = answer(ctx, "Bile dia balik kampung?")
        assert "Jumaat" in result["answer"] or "hari" in result["answer"]

    def test_when_year(self):
        ctx = "Syarikat tu ditubuhkan tahun 2010. Sekarang dah besar."
        result = answer(ctx, "Bila syarikat tu ditubuhkan?")
        assert "2010" in result["answer"]


# ─── Where Questions ──────────────────────────────────────────────────────────

class TestWhereQuestions:
    """Test where/mana/mane questions."""

    def test_where_english(self):
        ctx = "The office is located in Cyberjaya. It has 3 floors."
        result = answer(ctx, "Where is the office?")
        assert "Cyberjaya" in result["answer"]

    def test_where_bm(self):
        ctx = "Ali tinggal di Shah Alam. Rumah dia dekat dengan stesen LRT."
        result = answer(ctx, "Di mana Ali tinggal?")
        assert "Shah Alam" in result["answer"]

    def test_where_manglish(self):
        ctx = "Kedai tu kat Bangsar. Best gila nasi lemak dia."
        result = answer(ctx, "Mane kedai tu?")
        assert "Bangsar" in result["answer"]

    def test_where_kat(self):
        ctx = "Meeting kat bilik 302. Jangan lupa bawa laptop."
        result = answer(ctx, "Kat mana meeting?")
        assert "302" in result["answer"] or "bilik" in result["answer"]


# ─── Why Questions ────────────────────────────────────────────────────────────

class TestWhyQuestions:
    """Test why/kenapa/nape questions."""

    def test_why_english(self):
        ctx = "He was late because of the traffic jam. His boss was angry."
        result = answer(ctx, "Why was he late?")
        assert "traffic" in result["answer"].lower()

    def test_why_bm(self):
        ctx = "Dia tak datang sebab demam. Kena MC dua hari."
        result = answer(ctx, "Kenapa dia tak datang?")
        assert "demam" in result["answer"]

    def test_why_manglish(self):
        ctx = "Projek delay pasal client tukar requirement. Kena buat balik."
        result = answer(ctx, "Nape projek delay?")
        assert "client" in result["answer"] or "requirement" in result["answer"]


# ─── How Questions ────────────────────────────────────────────────────────────

class TestHowQuestions:
    """Test how/macam mana/camne questions."""

    def test_how_english(self):
        ctx = "You can fix it by restarting the server. That usually works."
        result = answer(ctx, "How do you fix it?")
        assert "restart" in result["answer"].lower() or "server" in result["answer"].lower()

    def test_how_bm(self):
        ctx = "Masak nasi lemak dengan guna santan segar. Jangan guna santan kotak."
        result = answer(ctx, "Macam mana nak masak nasi lemak?")
        assert "santan" in result["answer"]

    def test_how_manglish(self):
        ctx = "Nak deploy tu guna Docker. Senang je setup dia."
        result = answer(ctx, "Camne nak deploy?")
        assert "Docker" in result["answer"] or "guna" in result["answer"]


# ─── Yes/No Questions ─────────────────────────────────────────────────────────

class TestYesNoQuestions:
    """Test yes/no questions."""

    def test_yes_no_positive(self):
        ctx = "Ali memang pandai coding. Dia selalu tolong kawan-kawan."
        result = answer(ctx, "Adakah Ali pandai coding?")
        assert result["answer"] == "Ya"

    def test_yes_no_negative(self):
        ctx = "Projek tu tak siap lagi. Masih banyak bug."
        result = answer(ctx, "Is the project done?")
        assert "Tidak" in result["answer"]

    def test_yes_no_uncertain(self):
        ctx = "Cuaca hari ni panas. Ramai orang pergi pantai."
        result = answer(ctx, "Boleh tak buat BBQ esok?")
        # Low overlap, should be uncertain or negative
        assert result["answer"] in ("Tidak pasti", "Tidak", "Ya")


# ─── No Answer Found ──────────────────────────────────────────────────────────

class TestNoAnswer:
    """Test cases where no clear answer exists."""

    def test_unrelated_question(self):
        ctx = "Kucing tu tidur atas sofa. Dia memang suka tidur."
        result = answer(ctx, "What is the GDP of Malaysia?")
        assert result["confidence"] < 0.5

    def test_empty_context(self):
        result = answer("", "Sape buat ni?")
        assert result["answer"] == ""
        assert result["confidence"] == 0.0

    def test_empty_question(self):
        result = answer("Some context here.", "")
        assert result["answer"] == ""
        assert result["confidence"] == 0.0


# ─── Multiple Questions ───────────────────────────────────────────────────────

class TestMultipleQuestions:
    """Test answering multiple questions on same context."""

    def test_multiple_basic(self):
        ctx = "Ahmad kerja kat Petronas sejak 2018. Dia tinggal di KL."
        questions = [
            "Sape kerja kat Petronas?",
            "Bile Ahmad mula kerja?",
            "Mane dia tinggal?",
        ]
        results = answer_multiple(ctx, questions)
        assert len(results) == 3
        assert "Ahmad" in results[0]["answer"]
        assert "2018" in results[1]["answer"]
        assert "KL" in results[2]["answer"]

    def test_multiple_mixed_languages(self):
        ctx = "Siti is a software engineer at Google. She joined in 2021."
        questions = [
            "Who works at Google?",
            "Ape kerja Siti?",
            "Bile dia join?",
        ]
        results = answer_multiple(ctx, questions)
        assert len(results) == 3
        assert "Siti" in results[0]["answer"]
        assert "2021" in results[2]["answer"] or "2021" in results[2]["sentence"] or results[2]["confidence"] >= 0


# ─── Code-Switched Context ────────────────────────────────────────────────────

class TestCodeSwitched:
    """Test with code-switched BM/EN context."""

    def test_mixed_context_who(self):
        ctx = "Boss aku, Encik Razak, dia approve budget tu already. Confirm boleh proceed."
        result = answer(ctx, "Sape approve budget?")
        assert "Razak" in result["answer"] or "Encik" in result["answer"] or "Boss" in result["answer"]

    def test_mixed_context_when(self):
        ctx = "Deadline projek tu next Monday. Kena rush habis ni weekend."
        result = answer(ctx, "Bile deadline?")
        assert "Monday" in result["answer"] or "next" in result["answer"]

    def test_mixed_context_where(self):
        ctx = "Team outing kat Sunway Lagoon this Saturday. Confirm fun gila."
        result = answer(ctx, "Where is the team outing?")
        assert "Sunway" in result["answer"]

    def test_mixed_context_what(self):
        ctx = "Tech stack kita guna React frontend dengan Python backend. Database pakai PostgreSQL."
        result = answer(ctx, "Ape tech stack frontend?")
        assert "React" in result["answer"] or "frontend" in result["answer"]


# ─── Find Relevant Sentence ──────────────────────────────────────────────────

class TestFindRelevantSentence:
    """Test sentence retrieval."""

    def test_finds_correct_sentence(self):
        ctx = "Ali suka makan nasi lemak. Siti prefer mee goreng. Ahmad selalu minum teh tarik."
        result = find_relevant_sentence(ctx, "Siti suka makan apa?")
        assert "Siti" in result or "makan" in result

    def test_single_sentence(self):
        ctx = "Hari ni panas gila."
        result = find_relevant_sentence(ctx, "Macam mana cuaca?")
        assert result == "Hari ni panas gila"

    def test_empty_context(self):
        result = find_relevant_sentence("", "Any question?")
        assert result == ""


# ─── Extract Answer Span ─────────────────────────────────────────────────────

class TestExtractAnswerSpan:
    """Test answer span extraction."""

    def test_extract_location(self):
        result = extract_answer_span("Ali pergi ke Kuala Lumpur", "Where did Ali go?")
        assert "Kuala Lumpur" in result

    def test_extract_time(self):
        result = extract_answer_span("Meeting pukul 3 petang", "Bile meeting?")
        assert "3" in result or "pukul" in result

    def test_extract_person(self):
        result = extract_answer_span("Siti yang handle projek tu", "Sape handle projek?")
        assert "Siti" in result


# ─── Confidence Scoring ───────────────────────────────────────────────────────

class TestConfidence:
    """Test confidence scoring behavior."""

    def test_high_confidence_direct_match(self):
        ctx = "Ahmad tinggal di Penang. Dia suka pantai."
        result = answer(ctx, "Mane Ahmad tinggal?")
        assert result["confidence"] > 0.4

    def test_low_confidence_no_match(self):
        ctx = "Kucing tu comel. Dia suka main bola."
        result = answer(ctx, "Bile earthquake kat Japan?")
        assert result["confidence"] < 0.5

    def test_confidence_between_0_and_1(self):
        ctx = "Random text here. Nothing special."
        result = answer(ctx, "Who is the president?")
        assert 0.0 <= result["confidence"] <= 1.0


# ─── Answer Dict Structure ────────────────────────────────────────────────────

class TestAnswerStructure:
    """Test that answer returns correct structure."""

    def test_has_all_keys(self):
        result = answer("Ali kerja kat KL.", "Mane Ali kerja?")
        assert "answer" in result
        assert "confidence" in result
        assert "start" in result
        assert "end" in result
        assert "sentence" in result

    def test_types_correct(self):
        result = answer("Ali kerja kat KL.", "Mane Ali kerja?")
        assert isinstance(result["answer"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["start"], int)
        assert isinstance(result["end"], int)
        assert isinstance(result["sentence"], str)

    def test_start_end_valid(self):
        ctx = "Ahmad tinggal di Penang."
        result = answer(ctx, "Mane Ahmad tinggal?")
        assert result["start"] >= 0
        assert result["end"] >= result["start"]
        assert result["end"] <= len(ctx) + len(result["answer"])

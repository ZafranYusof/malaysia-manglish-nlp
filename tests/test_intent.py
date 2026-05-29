"""Tests for intent classification module."""

import pytest
from malaysian_manglish_nlp.intent import (
    classify_intent,
    classify_intents_batch,
    get_intent_features,
    is_question,
    is_request,
    is_complaint,
)


class TestClassifyIntentQuestion:
    """Test question intent detection."""

    def test_question_mark(self):
        result = classify_intent("berapa harga?")
        assert result['intent'] == 'question'

    def test_question_word_apa(self):
        result = classify_intent("apa kau buat?")
        assert result['intent'] == 'question'

    def test_question_word_bila(self):
        result = classify_intent("bila exam?")
        assert result['intent'] == 'question'

    def test_question_word_mana(self):
        result = classify_intent("kat mana kedai tu?")
        assert result['intent'] == 'question'

    def test_question_english_where(self):
        result = classify_intent("where got?")
        assert result['intent'] == 'question'

    def test_question_particle_ke(self):
        result = classify_intent("kau pergi ke")
        assert result['intent'] == 'question'

    def test_question_particle_tak(self):
        result = classify_intent("dia datang tak")
        assert result['intent'] == 'question'

    def test_question_how_much(self):
        result = classify_intent("how much this one?")
        assert result['intent'] == 'question'
        assert result['sub_type'] == 'quantity'

    def test_question_kenapa(self):
        result = classify_intent("kenapa lambat sangat?")
        assert result['intent'] == 'question'
        assert result['sub_type'] == 'reason'

    def test_question_macam_mana(self):
        result = classify_intent("macam mana nak buat?")
        assert result['intent'] == 'question'


class TestClassifyIntentRequest:
    """Test request intent detection."""

    def test_tolong(self):
        result = classify_intent("tolong belikan aku air")
        assert result['intent'] == 'request'

    def test_please(self):
        result = classify_intent("please help me with this")
        assert result['intent'] == 'request'

    def test_boleh_tak(self):
        result = classify_intent("boleh tak hantar aku balik")
        assert result['intent'] == 'request'

    def test_can_you(self):
        result = classify_intent("can you help me?")
        assert result['intent'] == 'request'

    def test_shortform_tlg(self):
        result = classify_intent("tlg bagi sikit")
        assert result['intent'] == 'request'

    def test_minta(self):
        result = classify_intent("minta maaf, boleh tak explain lagi")
        assert result['intent'] == 'request'


class TestClassifyIntentComplaint:
    """Test complaint intent detection."""

    def test_service_teruk(self):
        result = classify_intent("service teruk gila kat sini")
        assert result['intent'] == 'complaint'

    def test_lambat_gila(self):
        result = classify_intent("lambat gila delivery ni")
        assert result['intent'] == 'complaint'

    def test_fed_up(self):
        result = classify_intent("fed up already with this nonsense")
        assert result['intent'] == 'complaint'

    def test_tak_puas_hati(self):
        result = classify_intent("tak puas hati dengan layanan dia")
        assert result['intent'] == 'complaint'

    def test_bengang(self):
        result = classify_intent("bengang betul aku dengan system ni")
        assert result['intent'] == 'complaint'


class TestClassifyIntentGreeting:
    """Test greeting intent detection."""

    def test_assalamualaikum(self):
        result = classify_intent("assalamualaikum")
        assert result['intent'] == 'greeting'
        assert result['sub_type'] == 'islamic'

    def test_hi_guys(self):
        result = classify_intent("hi guys")
        assert result['intent'] == 'greeting'

    def test_morning_semua(self):
        result = classify_intent("morning semua")
        assert result['intent'] == 'greeting'
        assert result['sub_type'] == 'morning'

    def test_hello(self):
        result = classify_intent("hello")
        assert result['intent'] == 'greeting'

    def test_weh_standalone(self):
        result = classify_intent("weh")
        assert result['intent'] == 'greeting'


class TestClassifyIntentOpinion:
    """Test opinion intent detection."""

    def test_aku_rasa(self):
        result = classify_intent("aku rasa ok la movie tu")
        assert result['intent'] == 'opinion'

    def test_best_gila(self):
        result = classify_intent("best gila movie tu")
        assert result['intent'] == 'opinion'

    def test_tak_berbaloi(self):
        result = classify_intent("tak berbaloi beli phone tu")
        assert result['intent'] == 'opinion'

    def test_i_think(self):
        result = classify_intent("i think dia patut resign")
        assert result['intent'] == 'opinion'

    def test_bagi_aku(self):
        result = classify_intent("bagi aku overrated la tempat tu")
        assert result['intent'] == 'opinion'


class TestClassifyIntentStatement:
    """Test statement intent detection."""

    def test_factual_statement(self):
        result = classify_intent("dia kerja kat KL")
        assert result['intent'] == 'statement'

    def test_neutral_info(self):
        result = classify_intent("exam next week")
        assert result['intent'] == 'statement'

    def test_harga_naik(self):
        result = classify_intent("harga naik bulan ni")
        assert result['intent'] == 'statement'

    def test_empty_string(self):
        result = classify_intent("")
        assert result['intent'] == 'statement'
        assert result['confidence'] == 0.0


class TestClassifyIntentCommand:
    """Test command intent detection."""

    def test_pergi_sana(self):
        result = classify_intent("pergi sana")
        assert result['intent'] == 'command'

    def test_tutup_pintu(self):
        result = classify_intent("tutup pintu")
        assert result['intent'] == 'command'

    def test_stop_it(self):
        result = classify_intent("stop it")
        assert result['intent'] == 'command'

    def test_jangan(self):
        result = classify_intent("jangan kacau aku")
        assert result['intent'] == 'command'
        assert result['sub_type'] == 'prohibition'

    def test_diam(self):
        result = classify_intent("diam!")
        assert result['intent'] == 'command'


class TestClassifyIntentOffer:
    """Test offer intent detection."""

    def test_nak_aku_belikan(self):
        result = classify_intent("nak aku belikan?")
        assert result['intent'] == 'offer'

    def test_i_can_help(self):
        result = classify_intent("I can help you with that")
        assert result['intent'] == 'offer'

    def test_jom_aku_teman(self):
        result = classify_intent("jom aku teman kau")
        assert result['intent'] == 'offer'

    def test_let_me(self):
        result = classify_intent("let me handle this")
        assert result['intent'] == 'offer'

    def test_boleh_aku(self):
        result = classify_intent("boleh aku tolong?")
        assert result['intent'] == 'offer'


class TestClassifyIntentsBatch:
    """Test batch classification."""

    def test_batch_multiple(self):
        texts = [
            "berapa harga?",
            "tolong belikan",
            "service teruk",
            "hi guys",
        ]
        results = classify_intents_batch(texts)
        assert len(results) == 4
        assert results[0]['intent'] == 'question'
        assert results[1]['intent'] == 'request'
        assert results[2]['intent'] == 'complaint'
        assert results[3]['intent'] == 'greeting'

    def test_batch_empty(self):
        results = classify_intents_batch([])
        assert results == []


class TestGetIntentFeatures:
    """Test feature extraction."""

    def test_question_features(self):
        features = get_intent_features("berapa harga ni?")
        assert features['question_marks'] == 1
        assert features['question_words'] >= 1
        assert features['has_question_mark'] is True

    def test_command_features(self):
        features = get_intent_features("tutup pintu")
        assert features['command_verbs_at_start'] is True
        assert features['is_short'] is True

    def test_word_count(self):
        features = get_intent_features("aku nak pergi kedai")
        assert features['word_count'] == 4


class TestShortcutFunctions:
    """Test convenience boolean functions."""

    def test_is_question_true(self):
        assert is_question("apa kau buat?") is True

    def test_is_question_false(self):
        assert is_question("dia kerja kat KL") is False

    def test_is_request_true(self):
        assert is_request("tolong belikan aku air") is True

    def test_is_request_false(self):
        assert is_request("hi guys") is False

    def test_is_complaint_true(self):
        assert is_complaint("service teruk gila") is True

    def test_is_complaint_false(self):
        assert is_complaint("morning semua") is False


class TestConfidence:
    """Test confidence scoring."""

    def test_high_confidence_question(self):
        result = classify_intent("apa ni?")
        assert result['confidence'] >= 0.5

    def test_high_confidence_greeting(self):
        result = classify_intent("assalamualaikum")
        assert result['confidence'] >= 0.5

    def test_confidence_range(self):
        result = classify_intent("something random here")
        assert 0.0 <= result['confidence'] <= 1.0

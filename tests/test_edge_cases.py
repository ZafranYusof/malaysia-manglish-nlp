"""
Edge case test suite for manglish-nlp modules.
Tests specifically targeting known weak spots in sentiment, language detection,
dialect detection, and other modules.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import malaysian_manglish_nlp


class TestDoubleNegation:
    """Double/triple negation should resolve correctly."""

    def test_tak_tak_nak_is_positive(self):
        result = malaysian_manglish_nlp.sentiment("tak tak nak pergi")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_bukan_tak_suka(self):
        result = malaysian_manglish_nlp.sentiment("Bukan tak suka tapi tak boleh")
        assert result["sentiment"] in ["negative", "neutral"]

    def test_tak_pernah_tak_datang(self):
        result = malaysian_manglish_nlp.sentiment("Tak pernah tak datang class pun")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_takde_sapa_tak_kena(self):
        result = malaysian_manglish_nlp.sentiment("Takde sapa tak kena marah")
        assert result["sentiment"] == "negative"

    def test_tak_mustahil(self):
        result = malaysian_manglish_nlp.sentiment("Tak mustahil kalau usaha")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_tak_boleh_tak_pergi(self):
        result = malaysian_manglish_nlp.sentiment("Tak boleh tak pergi sebab boss suruh")
        assert result["sentiment"] == "negative"


class TestSarcasm:
    """Sarcastic positive words should be detected as negative."""

    def test_bagus_la_tu(self):
        result = malaysian_manglish_nlp.sentiment("Bagus la tu, memang pandai")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_tahniah_lambat(self):
        result = malaysian_manglish_nlp.sentiment("Tahniah la bro, baru 3 jam lambat")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_best_gila_kerja_12_jam(self):
        result = malaysian_manglish_nlp.sentiment("Best gila hidup macam ni kan, kerja 12 jam sehari")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_wow_amazing_tunggu_2_jam(self):
        result = malaysian_manglish_nlp.sentiment("Wow amazing la service dia, tunggu 2 jam baru dapat makanan")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_efficient_gila_4_jam(self):
        result = malaysian_manglish_nlp.sentiment("Wah efficient gila JPJ ni, 4 jam baru siap")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_rajin_betul_tidur(self):
        result = malaysian_manglish_nlp.sentiment("Rajin betul la kau ni, tidur je kerja")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_cantik_parking_dua_lot(self):
        result = malaysian_manglish_nlp.sentiment("Cantik la parking kau ni, ambik dua lot terus")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)

    def test_murah_rm500k(self):
        result = malaysian_manglish_nlp.sentiment("Murah la harga rumah sekarang, RM500k je untuk apartment")
        assert result["sentiment"] == "negative" or result.get("sarcasm", False)


class TestElongatedText:
    """Elongated characters should not break analysis."""

    def test_terukkkk(self):
        result = malaysian_manglish_nlp.sentiment("Terukkkkkk gila babi service dia")
        assert result["sentiment"] == "negative"

    def test_sedappppp(self):
        result = malaysian_manglish_nlp.sentiment("Sedappppp gilaaaa nasi goreng dia")
        assert result["sentiment"] == "positive"

    def test_boringggg(self):
        result = malaysian_manglish_nlp.sentiment("Boringggg la lecture ni tak habis habis")
        assert result["sentiment"] == "negative"

    def test_bestttttt(self):
        result = malaysian_manglish_nlp.sentiment("Bestttttt gila concert semalam")
        assert result["sentiment"] == "positive"

    def test_lamaaaa(self):
        result = malaysian_manglish_nlp.sentiment("Lamaaaa gila queue kat bank ni")
        assert result["sentiment"] == "negative"

    def test_penattttt(self):
        result = malaysian_manglish_nlp.sentiment("Penattttt gila kerja hari ni non stop")
        assert result["sentiment"] == "negative"


class TestMixedSentiment:
    """Sentences with both positive and negative aspects."""

    def test_sedap_tapi_service_teruk(self):
        result = malaysian_manglish_nlp.sentiment("Makanan sedap tapi service teruk gila")
        assert result["sentiment"] in ["mixed", "negative"]

    def test_cantik_tapi_mahal(self):
        result = malaysian_manglish_nlp.sentiment("View cantik tapi harga hotel mahal sangat")
        assert result["sentiment"] in ["mixed", "negative"]

    def test_phone_cantik_battery_bad(self):
        result = malaysian_manglish_nlp.sentiment("Phone cantik design dia tapi battery cepat habis")
        assert result["sentiment"] in ["mixed", "negative"]

    def test_gaji_ok_workload_gila(self):
        result = malaysian_manglish_nlp.sentiment("Gaji ok la tapi workload gila babi")
        assert result["sentiment"] in ["mixed", "negative"]

    def test_movie_best_ending_bad(self):
        result = malaysian_manglish_nlp.sentiment("Movie best tapi ending dia disappointing gila")
        assert result["sentiment"] in ["mixed", "negative"]


class TestAmbiguous:
    """Ambiguous short phrases that could go either way."""

    def test_ok_la(self):
        result = malaysian_manglish_nlp.sentiment("Ok la")
        assert result["sentiment"] in ["neutral", "positive"]

    def test_boleh_la(self):
        result = malaysian_manglish_nlp.sentiment("Boleh la")
        assert result["sentiment"] in ["neutral", "positive"]

    def test_biasa_je(self):
        result = malaysian_manglish_nlp.sentiment("Biasa je")
        assert result["sentiment"] == "neutral"

    def test_hmm_entah(self):
        result = malaysian_manglish_nlp.sentiment("Hmm entah la")
        assert result["sentiment"] == "neutral"

    def test_tengok_la(self):
        result = malaysian_manglish_nlp.sentiment("Tengok la macam mana")
        assert result["sentiment"] == "neutral"

    def test_depends(self):
        result = malaysian_manglish_nlp.sentiment("Depends la on situation")
        assert result["sentiment"] == "neutral"


class TestDialectEdgeCases:
    """Dialect-specific text should be handled correctly."""

    def test_kelantan_positive(self):
        result = malaysian_manglish_nlp.sentiment("Sedak gilo nasi daghe mok cik buat")
        assert result["sentiment"] == "positive"

    def test_kelantan_negative(self):
        result = malaysian_manglish_nlp.sentiment("Tok suko la politik skang ni")
        assert result["sentiment"] == "negative"

    def test_kelantan_neutral(self):
        result = malaysian_manglish_nlp.sentiment("Mung nok gi mano ni mlm ni")
        assert result["sentiment"] == "neutral"

    def test_sarawak_positive(self):
        result = malaysian_manglish_nlp.sentiment("Sedap gilak laksa Sarawak ya")
        assert result["sentiment"] == "positive"

    def test_sarawak_negative(self):
        result = malaysian_manglish_nlp.sentiment("Sik best la movie ya kamek tengok semalam")
        assert result["sentiment"] == "negative"

    def test_sabah_positive(self):
        result = malaysian_manglish_nlp.sentiment("Siok bah naik Kinabalu pagi tadi")
        assert result["sentiment"] == "positive"

    def test_sabah_negative(self):
        result = malaysian_manglish_nlp.sentiment("Susa bah mau cari kerja sini")
        assert result["sentiment"] == "negative"

    def test_negeri_sembilan_positive(self):
        result = malaysian_manglish_nlp.sentiment("Sodap bona masakan mak den ni")
        assert result["sentiment"] == "positive"

    def test_negeri_sembilan_negative(self):
        result = malaysian_manglish_nlp.sentiment("Den tak suko la cito macam tu")
        assert result["sentiment"] == "negative"

    def test_kedah_negative(self):
        result = malaysian_manglish_nlp.sentiment("Awat hang lambat sangat ni")
        assert result["sentiment"] == "negative"

    def test_terengganu_positive(self):
        result = malaysian_manglish_nlp.sentiment("Dok rok cettong la makang ari ni sedak gile")
        assert result["sentiment"] == "positive"

    def test_dialect_detection_kelantan(self):
        result = malaysian_manglish_nlp.detect_dialect("Ambo tok leh nok gi kijo ari ni")
        assert result["dialect"] == "kelantan"

    def test_dialect_detection_sarawak(self):
        result = malaysian_manglish_nlp.detect_dialect("Kamek sik tauk la apa jadi")
        assert result["dialect"] == "sarawak"

    def test_dialect_detection_sabah(self):
        result = malaysian_manglish_nlp.detect_dialect("Bah ko mau pi mana ni")
        assert result["dialect"] == "sabah"


class TestVeryShortText:
    """Single word or very short text should still produce valid results."""

    def test_single_word_positive(self):
        result = malaysian_manglish_nlp.sentiment("Best")
        assert result["sentiment"] == "positive"

    def test_single_word_negative(self):
        result = malaysian_manglish_nlp.sentiment("Teruk")
        assert result["sentiment"] == "negative"

    def test_single_word_neutral(self):
        result = malaysian_manglish_nlp.sentiment("Gila")
        assert "sentiment" in result

    def test_two_words_positive(self):
        result = malaysian_manglish_nlp.sentiment("Cantik gila")
        assert result["sentiment"] == "positive"

    def test_single_word_sad(self):
        result = malaysian_manglish_nlp.sentiment("Sedih")
        assert result["sentiment"] == "negative"

    def test_single_interjection(self):
        result = malaysian_manglish_nlp.sentiment("Haih")
        assert result["sentiment"] in ["negative", "neutral"]

    def test_single_letter(self):
        result = malaysian_manglish_nlp.sentiment("k")
        assert result["sentiment"] == "neutral"

    def test_abbreviation(self):
        result = malaysian_manglish_nlp.sentiment("otw")
        assert result["sentiment"] == "neutral"


class TestAllCaps:
    """ALL CAPS text should be handled (often indicates strong emotion)."""

    def test_caps_positive(self):
        result = malaysian_manglish_nlp.sentiment("LETS GOOOOO MENANG FINALLLLL")
        assert result["sentiment"] == "positive"

    def test_caps_negative(self):
        result = malaysian_manglish_nlp.sentiment("TOLONG LA JANGAN MACAM NI")
        assert result["sentiment"] == "negative"

    def test_caps_surprise(self):
        result = malaysian_manglish_nlp.sentiment("GILA BEST WEYYYY")
        assert result["sentiment"] == "positive"

    def test_caps_anger(self):
        result = malaysian_manglish_nlp.sentiment("BODOH LA SEMUA NI")
        assert result["sentiment"] == "negative"


class TestNumbersMixedWithText:
    """Numbers in text should not break analysis."""

    def test_rm_amount_negative(self):
        result = malaysian_manglish_nlp.sentiment("Baru spend RM2000 untuk repair kereta fml")
        assert result["sentiment"] == "negative"

    def test_score_positive(self):
        result = malaysian_manglish_nlp.sentiment("Score 95/100 untuk test tadi alhamdulillah")
        assert result["sentiment"] == "positive"

    def test_time_reference(self):
        result = malaysian_manglish_nlp.sentiment("Tidur 3 jam je semalam")
        assert result["sentiment"] == "negative"

    def test_percentage(self):
        result = malaysian_manglish_nlp.sentiment("Gaji naik 500 je tapi workload naik 200%")
        assert result["sentiment"] == "negative"

    def test_followers_positive(self):
        result = malaysian_manglish_nlp.sentiment("Followers dah 10k thank you semua")
        assert result["sentiment"] == "positive"


class TestPassiveAggressive:
    """Passive aggressive statements that look neutral but are negative."""

    def test_takpe_buat_sorang(self):
        result = malaysian_manglish_nlp.sentiment("Takpe la aku buat sorang je macam biasa")
        assert result["sentiment"] == "negative"

    def test_ok_tak_kisah(self):
        result = malaysian_manglish_nlp.sentiment("Aku ok je tak kisah pun korang tak ajak")
        assert result["sentiment"] == "negative"

    def test_ye_la_aku_salah(self):
        result = malaysian_manglish_nlp.sentiment("Ye la aku yang salah macam biasa kan")
        assert result["sentiment"] == "negative"

    def test_memang_tak_penting(self):
        result = malaysian_manglish_nlp.sentiment("Aku memang tak penting pun dalam group ni")
        assert result["sentiment"] == "negative"


class TestCodeSwitchBoundaries:
    """Complex code-switching should be segmented correctly."""

    def test_english_start_malay_end(self):
        result = malaysian_manglish_nlp.detect_language("The audacity of dia cakap macam tu")
        assert result["language"] in ["mixed", "manglish"]

    def test_malay_start_english_end(self):
        result = malaysian_manglish_nlp.detect_language("Aku literally cannot even right now")
        assert result["language"] in ["mixed", "manglish"]

    def test_gen_z_slang_mixed(self):
        result = malaysian_manglish_nlp.detect_language("Its giving main character energy la dia ni")
        assert result["language"] in ["mixed", "manglish"]

    def test_internet_slang(self):
        result = malaysian_manglish_nlp.detect_language("Ngl this is actually sedap gila")
        assert result["language"] in ["mixed", "manglish"]

    def test_pure_malay(self):
        result = malaysian_manglish_nlp.detect_language("Saya tidak faham apa yang berlaku")
        assert result["language"] in ["malay", "manglish", "bm"]

    def test_pure_english(self):
        result = malaysian_manglish_nlp.detect_language("I dont understand what is happening")
        assert result["language"] in ["english", "mixed", "en"]


class TestImplicitSentiment:
    """Sentiment implied by context rather than explicit words."""

    def test_dah_la_tu_negative(self):
        result = malaysian_manglish_nlp.sentiment("Dah la tu")
        assert result["sentiment"] in ["negative", "neutral"]

    def test_suka_hati_kau(self):
        result = malaysian_manglish_nlp.sentiment("Suka hati kau la")
        assert result["sentiment"] == "negative"

    def test_lantak_kau(self):
        result = malaysian_manglish_nlp.sentiment("Lantak kau la nak buat apa")
        assert result["sentiment"] == "negative"

    def test_tau_takpe(self):
        result = malaysian_manglish_nlp.sentiment("Tau takpe")
        assert result["sentiment"] in ["negative", "neutral"]

    def test_kenapa_la_aku(self):
        result = malaysian_manglish_nlp.sentiment("Kenapa la aku macam ni selalu")
        assert result["sentiment"] == "negative"

    def test_bila_nak_habis(self):
        result = malaysian_manglish_nlp.sentiment("Bila la nak habis sem ni penat")
        assert result["sentiment"] == "negative"


class TestUnderstatement:
    """Understatements that are actually positive."""

    def test_tak_teruk(self):
        result = malaysian_manglish_nlp.sentiment("Tak teruk la untuk first attempt")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_boleh_tahan(self):
        result = malaysian_manglish_nlp.sentiment("Boleh tahan la makanan dia")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_not_bad(self):
        result = malaysian_manglish_nlp.sentiment("Not bad la presentation kau tadi")
        assert result["sentiment"] in ["positive", "neutral"]

    def test_lumayan(self):
        result = malaysian_manglish_nlp.sentiment("Lumayan la gaji untuk fresh grad")
        assert result["sentiment"] in ["positive", "neutral"]


class TestHashtagsAndSocialMedia:
    """Text with hashtags should still be analyzed correctly."""

    def test_positive_with_hashtag(self):
        result = malaysian_manglish_nlp.sentiment("Hari ni productive gila #grindmode")
        assert result["sentiment"] == "positive"

    def test_negative_with_hashtag(self):
        result = malaysian_manglish_nlp.sentiment("Kenapa la aku macam ni #sadlife #overthinking")
        assert result["sentiment"] == "negative"

    def test_neutral_with_hashtag(self):
        result = malaysian_manglish_nlp.sentiment("OOTD hari ni simple je #minimalist")
        assert result["sentiment"] in ["neutral", "positive"]


class TestRobustness:
    """General robustness tests - should not crash on edge inputs."""

    def test_empty_string(self):
        try:
            result = malaysian_manglish_nlp.sentiment("")
            assert "sentiment" in result or result is not None
        except (ValueError, KeyError):
            pass

    def test_only_numbers(self):
        result = malaysian_manglish_nlp.sentiment("12345")
        assert "sentiment" in result

    def test_only_punctuation(self):
        result = malaysian_manglish_nlp.sentiment("!!??...")
        assert "sentiment" in result

    def test_very_long_text(self):
        long_text = "sedap " * 100
        result = malaysian_manglish_nlp.sentiment(long_text)
        assert result["sentiment"] == "positive"

    def test_unicode_emoji_description(self):
        result = malaysian_manglish_nlp.sentiment("haha gelak tak boleh stop")
        assert result["sentiment"] == "positive"

    def test_repeated_characters(self):
        result = malaysian_manglish_nlp.sentiment("aaaaaaaaa")
        assert "sentiment" in result

    def test_mixed_scripts(self):
        result = malaysian_manglish_nlp.sentiment("Aku suka world")
        assert "sentiment" in result

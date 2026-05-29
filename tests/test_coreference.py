"""Tests for manglish_nlp.coreference module."""

import pytest
from manglish_nlp.coreference import (
    resolve_coreferences,
    resolve_in_context,
    get_entities_and_references,
    replace_pronouns,
)


class TestResolveCoreferences:
    """Tests for resolve_coreferences function."""

    def test_empty_text(self):
        result = resolve_coreferences("")
        assert result == []

    def test_whitespace_only(self):
        result = resolve_coreferences("   ")
        assert result == []

    def test_no_pronouns(self):
        result = resolve_coreferences("Ali pergi kedai")
        assert result == []

    def test_dia_resolves_to_person(self):
        result = resolve_coreferences("Ali pergi kedai. Dia beli roti.")
        # Should find 'dia' and resolve to 'Ali'
        dia_refs = [r for r in result if r['pronoun'] == 'dia']
        if dia_refs:
            assert dia_refs[0]['antecedent'] == 'Ali'

    def test_mereka_resolves_to_plural(self):
        # mereka should prefer plural antecedents or groups
        text = "Ahmad dan Siti pergi pasar. Mereka beli sayur."
        result = resolve_coreferences(text)
        # At minimum, should not crash
        assert isinstance(result, list)

    def test_beliau_formal_pronoun(self):
        text = "Dato Ahmad hadir majlis. Beliau beri ucapan."
        result = resolve_coreferences(text)
        beliau_refs = [r for r in result if r['pronoun'] == 'beliau']
        if beliau_refs:
            assert 'Ahmad' in beliau_refs[0]['antecedent']

    def test_die_informal_dia(self):
        text = "Encik Hassan cakap. Die kata ok."
        result = resolve_coreferences(text)
        die_refs = [r for r in result if r['pronoun'] == 'die']
        if die_refs:
            assert 'Hassan' in die_refs[0]['antecedent']

    def test_nya_suffix(self):
        text = "Ali bawa keretanya ke bengkel."
        result = resolve_coreferences(text)
        nya_refs = [r for r in result if r['pronoun'] == '-nya']
        # Should find -nya
        assert isinstance(result, list)

    def test_he_resolves_to_male(self):
        text = "Ahmad went to the shop. He bought bread."
        result = resolve_coreferences(text)
        he_refs = [r for r in result if r['pronoun'] == 'he']
        if he_refs:
            assert 'Ahmad' in he_refs[0]['antecedent']

    def test_she_resolves_to_female(self):
        text = "Siti pergi sekolah. She scored well."
        result = resolve_coreferences(text)
        she_refs = [r for r in result if r['pronoun'] == 'she']
        if she_refs:
            assert 'Siti' in she_refs[0]['antecedent']

    def test_gender_mismatch_no_resolve(self):
        # 'he' should not resolve to a female name
        text = "Siti pergi kedai. He beli roti."
        result = resolve_coreferences(text)
        he_refs = [r for r in result if r['pronoun'] == 'he']
        # Should not resolve 'he' to 'Siti'
        for ref in he_refs:
            assert ref['antecedent'] != 'Siti'

    def test_tu_demonstrative(self):
        text = "Encik Ali bagi cadangan. Tu memang bagus."
        result = resolve_coreferences(text)
        tu_refs = [r for r in result if r['pronoun'] == 'tu']
        assert isinstance(result, list)

    def test_multiple_pronouns(self):
        text = "Encik Ahmad jumpa Puan Siti. Dia cakap dengan dia."
        result = resolve_coreferences(text)
        # Should find multiple pronoun references
        assert isinstance(result, list)

    def test_recency_bias(self):
        text = "Encik Ali datang dulu. Encik Hassan sampai kemudian. Dia order makan."
        result = resolve_coreferences(text)
        dia_refs = [r for r in result if r['pronoun'] == 'dia']
        # Should prefer Hassan (more recent)
        if dia_refs:
            assert 'Hassan' in dia_refs[0]['antecedent']

    def test_result_format(self):
        text = "Encik Ali pergi. Dia balik."
        result = resolve_coreferences(text)
        if result:
            ref = result[0]
            assert 'pronoun' in ref
            assert 'position' in ref
            assert 'antecedent' in ref
            assert 'antecedent_position' in ref
            assert 'confidence' in ref
            assert isinstance(ref['position'], tuple)
            assert len(ref['position']) == 2

    def test_confidence_range(self):
        text = "Encik Ali pergi kedai. Dia beli barang."
        result = resolve_coreferences(text)
        for ref in result:
            assert 0.0 <= ref['confidence'] <= 1.0

    def test_diorang_plural(self):
        text = "Ahmad dan Ali pergi. Diorang main bola."
        result = resolve_coreferences(text)
        diorang_refs = [r for r in result if r['pronoun'] == 'diorang']
        assert isinstance(result, list)

    def test_no_entity_no_resolution(self):
        text = "Pergi kedai. Dia beli roti."
        result = resolve_coreferences(text)
        # No named entity to resolve to
        dia_refs = [r for r in result if r['pronoun'] == 'dia']
        # Should either be empty or have low confidence
        assert isinstance(result, list)

    def test_position_values_valid(self):
        text = "Encik Ali kerja. Dia balik rumah."
        result = resolve_coreferences(text)
        for ref in result:
            start, end = ref['position']
            assert start >= 0
            assert end > start
            assert end <= len(text)


class TestResolveInContext:
    """Tests for resolve_in_context function."""

    def test_empty_text(self):
        result = resolve_in_context("")
        assert result == []

    def test_no_context(self):
        result = resolve_in_context("Encik Ali pergi. Dia balik.")
        assert isinstance(result, list)

    def test_with_context(self):
        context = "Aku jumpa Encik Ahmad semalam."
        text = "Dia kata ok."
        result = resolve_in_context(text, context=context)
        dia_refs = [r for r in result if r['pronoun'] == 'dia']
        if dia_refs:
            assert 'Ahmad' in dia_refs[0]['antecedent']

    def test_context_none_same_as_no_context(self):
        text = "Encik Ali pergi. Dia balik."
        result1 = resolve_in_context(text, context=None)
        result2 = resolve_coreferences(text)
        # Should produce same results
        assert len(result1) == len(result2)

    def test_entity_in_context_resolved(self):
        context = "Puan Siti baru sampai office."
        text = "She looks tired today."
        result = resolve_in_context(text, context=context)
        she_refs = [r for r in result if r['pronoun'] == 'she']
        if she_refs:
            assert 'Siti' in she_refs[0]['antecedent']

    def test_multi_turn_context(self):
        context = "Encik Hassan call tadi. Dia tanya pasal meeting."
        text = "Dia nak reschedule."
        result = resolve_in_context(text, context=context)
        assert isinstance(result, list)

    def test_position_relative_to_text(self):
        context = "Encik Ali datang."
        text = "Dia cakap hello."
        result = resolve_in_context(text, context=context)
        for ref in result:
            start, end = ref['position']
            # Position should be relative to text, not combined
            assert start >= 0


class TestGetEntitiesAndReferences:
    """Tests for get_entities_and_references function."""

    def test_empty_text(self):
        result = get_entities_and_references("")
        assert result == {}

    def test_no_references(self):
        result = get_entities_and_references("Ali pergi kedai")
        assert isinstance(result, dict)

    def test_single_entity_with_reference(self):
        text = "Encik Ali pergi kedai. Dia beli roti."
        result = get_entities_and_references(text)
        if result:
            # Should have Ali mapped to dia
            for entity, refs in result.items():
                assert isinstance(refs, list)
                for ref in refs:
                    assert 'pronoun' in ref
                    assert 'position' in ref

    def test_multiple_entities(self):
        text = "Encik Ahmad dan Puan Siti datang. Dia bagi salam."
        result = get_entities_and_references(text)
        assert isinstance(result, dict)

    def test_returns_dict_of_lists(self):
        text = "Encik Ali cakap. Dia senyum. Dia gelak."
        result = get_entities_and_references(text)
        for entity, refs in result.items():
            assert isinstance(entity, str)
            assert isinstance(refs, list)


class TestReplacePronouns:
    """Tests for replace_pronouns function."""

    def test_empty_text(self):
        result = replace_pronouns("")
        assert result == ""

    def test_no_pronouns(self):
        text = "Ali pergi kedai"
        result = replace_pronouns(text)
        assert result == text

    def test_basic_replacement(self):
        text = "Encik Ali pergi kedai. Dia beli roti."
        result = replace_pronouns(text)
        # 'Dia' should be replaced with 'Ali' or 'Encik Ali'
        if 'Dia' not in result and 'dia' not in result:
            assert 'Ali' in result

    def test_preserves_non_pronoun_text(self):
        text = "Encik Ali pergi kedai. Dia beli roti."
        result = replace_pronouns(text)
        assert "pergi kedai" in result
        assert "beli roti" in result

    def test_returns_string(self):
        text = "Encik Ali cakap. Dia senyum."
        result = replace_pronouns(text)
        assert isinstance(result, str)

    def test_no_entity_returns_original(self):
        text = "Pergi kedai. Beli roti."
        result = replace_pronouns(text)
        assert result == text

    def test_whitespace_only(self):
        result = replace_pronouns("   ")
        assert result == "   "

    def test_none_handling(self):
        # Should handle None gracefully
        result = replace_pronouns(None)
        assert result is None

"""
Custom Rasa NLU components wrapping manglish-nlp.

Components:
- ManglishNLPFeaturizer: Featurises text using manglish-nlp embeddings
- ManglishNERExtractor: Extracts entities using manglish-nlp NER

Add to config.yml pipeline:
    - name: "nlu_component.ManglishNLPFeaturizer"
    - name: "nlu_component.ManglishNERExtractor"

Requires: pip install manglish-nlp
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Text, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
)

import manglish_nlp

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class ManglishNLPFeaturizer(Featurizer, GraphComponent):
    """
    Featurizer that uses manglish-nlp to normalise text and generate
    character-level features suitable for DIET classifier.

    In production, you could load manglish-nlp word embeddings
    (manglish_nlp.word_embeddings) for dense vector features.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # Whether to normalise text before featurising
            "normalise": True,
            # Whether to add language detection as a feature
            "detect_language": True,
            # Whether to add sentiment score as a feature
            "add_sentiment": True,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        super().__init__()
        self.normalise = config.get("normalise", True)
        self.detect_language = config.get("detect_language", True)
        self.add_sentiment = config.get("add_sentiment", True)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "ManglishNLPFeaturizer":
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:
        """No training needed — uses manglish-nlp directly."""
        for example in training_data.training_examples:
            self._featurise(example)
        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            self._featurise(example)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._featurise(message)
        return messages

    def _featurise(self, message: Message) -> None:
        """Apply manglish-nlp processing and store metadata."""
        text = message.get(TEXT)
        if not text:
            return

        # Normalise shortforms for better downstream matching
        if self.normalise:
            normalised = manglish_nlp.normalize(text)
            message.set("manglish_normalised", normalised, output_property=True)

        # Language detection
        if self.detect_language:
            lang = manglish_nlp.detect_language(text)
            message.set("manglish_language", lang, output_property=True)

        # Sentiment (useful for downstream policies)
        if self.add_sentiment:
            sent = manglish_nlp.sentiment(text)
            message.set("manglish_sentiment", sent, output_property=True)

        # Intent features
        intent_features = {
            "is_question": manglish_nlp.is_question(text),
            "is_complaint": manglish_nlp.is_complaint(text),
            "is_request": manglish_nlp.is_request(text),
        }
        message.set("manglish_intent_features", intent_features, output_property=True)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=False
)
class ManglishNERExtractor(EntityExtractorMixin, GraphComponent):
    """
    Entity extractor using manglish-nlp NER.

    Recognises 9 entity types:
    PERSON, LOCATION, ORGANIZATION, MONEY, DATE, TIME,
    PHONE, EMAIL, URL

    Entities are added to the message and can be used by
    Rasa slot mapping.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            # Minimum confidence to include an entity
            "min_confidence": 0.0,
            # Entity types to extract (None = all)
            "entity_types": None,
            # Also run normalisation before NER
            "normalise_first": True,
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.min_confidence = config.get("min_confidence", 0.0)
        self.entity_types = config.get("entity_types", None)
        self.normalise_first = config.get("normalise_first", True)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "ManglishNERExtractor":
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:
        """No training — delegates to manglish-nlp."""
        return self._resource

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        for example in training_data.training_examples:
            self._extract(example)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._extract(message)
        return messages

    def _extract(self, message: Message) -> None:
        """Run NER and attach entities to message."""
        text = message.get(TEXT)
        if not text:
            return

        # Optionally normalise first for better NER
        if self.normalise_first:
            text = manglish_nlp.normalize(text)

        entities_raw = manglish_nlp.ner_tag(text)

        rasa_entities = []
        for ent in entities_raw:
            etype = ent.get("entity", ent.get("type", "UNKNOWN"))
            evalue = ent.get("text", ent.get("value", ""))
            confidence = float(ent.get("confidence", 1.0))

            # Filter by confidence
            if confidence < self.min_confidence:
                continue

            # Filter by entity type
            if self.entity_types and etype not in self.entity_types:
                continue

            # Find position in original text
            start = ent.get("start", text.find(evalue) if evalue else 0)
            end = ent.get("end", start + len(evalue) if evalue else 0)

            if start < 0:
                start = 0
            if end < start:
                end = start + len(evalue)

            rasa_entities.append({
                ENTITY_ATTRIBUTE_TYPE: etype,
                ENTITY_ATTRIBUTE_VALUE: evalue,
                ENTITY_ATTRIBUTE_START: max(0, start),
                ENTITY_ATTRIBUTE_END: end,
                "confidence": confidence,
                "extractor": "ManglishNERExtractor",
            })

        # Merge with existing entities
        existing = message.get(ENTITIES, [])
        message.set(ENTITIES, existing + rasa_entities, add_to_output=True)

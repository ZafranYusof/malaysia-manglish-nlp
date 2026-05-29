"""
Custom Rasa actions using malaysian-manglish-nlp.

These actions leverage malaysian-manglish-nlp for:
- Sentiment-aware responses
- NER extraction
- Translation fallback
- Normalisation for better intent matching

Run Rasa actions server:
    rasa run actions
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.executor import CollectingDispatcher

import malaysian_manglish_nlp

logger = logging.getLogger(__name__)


class ActionAnalyseSentiment(Action):
    """Analyse sentiment of user's last message and set slot."""

    def name(self) -> str:
        return "action_analyse_sentiment"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        text = tracker.latest_message.get("text", "")

        # Run sentiment analysis
        result = malaysian_manglish_nlp.sentiment(text)
        label = result.get("label", "neutral").lower()
        score = result.get("score", 0.5)

        # Also detect emotion for richer context
        emotion = malaysian_manglish_nlp.detect_emotion(text)

        # Build response based on sentiment
        if label == "positive":
            response = (
                f"Text tu nampak positive (confidence: {score:.0%}). "
                f"Emotion: {emotion.get('emotion', 'neutral')} 😊"
            )
        elif label == "negative":
            response = (
                f"Text tu nampak negative (confidence: {score:.0%}). "
                f"Emotion: {emotion.get('emotion', 'neutral')}. "
                "Ada apa-apa yang tak ok ke?"
            )
        else:
            response = (
                f"Text tu neutral (confidence: {score:.0%}). "
                f"Emotion: {emotion.get('emotion', 'neutral')}."
            )

        dispatcher.utter_message(text=response)

        return [
            SlotSet("detected_sentiment", label),
            SlotSet("sentiment_result", json.dumps(result)),
        ]


class ActionTranslate(Action):
    """Translate user's message using malaysian-manglish-nlp."""

    def name(self) -> str:
        return "action_translate"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        text = tracker.latest_message.get("text", "")

        # Detect language first
        lang = malaysian_manglish_nlp.detect_language(text)

        # Translate based on detected language
        if lang.get("language", "") == "en":
            translated = malaysian_manglish_nlp.to_malay(text)
            direction = "English → BM"
        else:
            translated = malaysian_manglish_nlp.to_english(text)
            direction = "BM/Manglish → English"

        # Also get formal version
        formal = malaysian_manglish_nlp.to_formal(text)

        response = (
            f"**{direction}:**\n"
            f"Original: {text}\n"
            f"Translated: {translated}\n"
            f"Formal BM: {formal}"
        )

        dispatcher.utter_message(text=response)

        return [SlotSet("translation_result", translated)]


class ActionExtractEntities(Action):
    """Extract entities from user message using malaysian-manglish-nlp NER."""

    def name(self) -> str:
        return "action_extract_entities"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        text = tracker.latest_message.get("text", "")

        entities = malaysian_manglish_nlp.ner_tag(text)

        if not entities:
            dispatcher.utter_message(text="Aku tak jumpa named entities dalam text tu.")
            return []

        # Format entities for display
        parts = []
        slots = []
        for ent in entities:
            etype = ent.get("entity", ent.get("type", "UNKNOWN"))
            evalue = ent.get("text", ent.get("value", ""))
            parts.append(f"- {etype}: {evalue}")

            # Map to Rasa slots where applicable
            if etype == "LOCATION":
                slots.append(SlotSet("location", evalue))
            elif etype == "PERSON":
                slots.append(SlotSet("person", evalue))

        response = "Entities yang aku jumpa:\n" + "\n".join(parts)
        dispatcher.utter_message(text=response)

        return slots


class ActionMakanRecommendation(Action):
    """Give food recommendations with sentiment-aware tone."""

    def name(self) -> str:
        return "action_makan_recommendation"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        location = tracker.get_slot("location") or "area korang"
        food = tracker.get_slot("food") or "makanan"

        # Normalise user's message for better matching
        text = tracker.latest_message.get("text", "")
        normalised = malaysian_manglish_nlp.normalize(text)
        sentiment = malaysian_manglish_nlp.sentiment(text)

        # Mood-aware response
        if sentiment.get("label") == "positive":
            opener = "Wah semangat nak makan tu! "
        elif sentiment.get("label") == "negative":
            opener = "Comfort food memang best time camni. "
        else:
            opener = ""

        response = (
            f"{opener}Kalau area {location}, aku suggest cuba {food}. "
            f"Port best biasanya dekat main road atau area pasar malam. "
            f"Try tanya member local, diorang tau port rahsia."
        )

        dispatcher.utter_message(text=response)
        return []


class ActionSentimentAwareResponse(Action):
    """Generate response that adapts tone based on detected sentiment."""

    def name(self) -> str:
        return "action_sentiment_aware_response"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        text = tracker.latest_message.get("text", "")

        # Full analysis
        sentiment = malaysian_manglish_nlp.sentiment(text)
        emotion = malaysian_manglish_nlp.detect_emotion(text)
        normalised = malaysian_manglish_nlp.normalize(text)

        label = sentiment.get("label", "neutral")
        emo = emotion.get("emotion", "neutral")

        # Adapt tone
        if label == "negative":
            if emo in ("anger", "disgust"):
                tone = "Faham wei, memang frustrating. "
            elif emo == "sadness":
                tone = "Alamak, kesian pulak aku dengar. "
            else:
                tone = "Hmm, tak best bunyi tu. "
        elif label == "positive":
            if emo == "joy":
                tone = "Best gila tu wei! Tahniah! "
            elif emo == "surprise":
                tone = "Wah serious ke?! Best tu! "
            else:
                tone = "Good lah macam tu! "
        else:
            tone = "Ok noted. "

        response = (
            f"{tone}"
            f"Aku faham kau cakap: '{normalised}'. "
            f"Ada apa-apa lagi aku boleh tolong?"
        )

        dispatcher.utter_message(text=response)

        return [
            SlotSet("detected_sentiment", label.lower()),
            SlotSet("sentiment_result", json.dumps(sentiment)),
        ]

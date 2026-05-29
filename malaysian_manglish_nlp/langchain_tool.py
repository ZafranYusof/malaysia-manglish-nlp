"""
manglish-nlp LangChain tool integration.

Provides LangChain-compatible tools for Malaysian Manglish NLP processing.

Usage:
    from malaysian_manglish_nlp.langchain_tool import ManglishNLPTool, ManglishSentimentTool

    # Use in a LangChain agent
    tools = [ManglishNLPTool(), ManglishSentimentTool()]

Requires: pip install manglish-nlp[langchain]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "LangChain is required for this module. "
        "Install with: pip install manglish-nlp[langchain]"
    )

import malaysian_manglish_nlp


# --- Input Schemas ---

class TextInput(BaseModel):
    """Input schema for text processing tools."""
    text: str = Field(description="The Manglish/Malaysian text to process")


class TranslateInput(BaseModel):
    """Input schema for translation tool."""
    text: str = Field(description="The Manglish/Malaysian text to translate")
    target: str = Field(
        default="en",
        description="Target language: 'en' (English), 'bm' (Bahasa Melayu), 'formal' (formal BM)",
    )


class NERInput(BaseModel):
    """Input schema for NER tool."""
    text: str = Field(description="The text to extract named entities from")


# --- Tools ---

class ManglishNLPTool(BaseTool):
    """LangChain tool that runs full Manglish NLP analysis."""

    name: str = "malaysian_manglish_nlp_analyze"
    description: str = (
        "Analyze Malaysian Manglish text with full NLP pipeline. "
        "Returns sentiment, entities, POS tags, language detection, emotion, "
        "and keywords. Use this when you need comprehensive analysis of "
        "Malaysian/Manglish text (mix of Bahasa Melayu and English)."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Run full analysis on text."""
        return {
            "normalized": malaysian_manglish_nlp.normalize(text),
            "sentiment": malaysian_manglish_nlp.sentiment(text),
            "language": malaysian_manglish_nlp.detect_language(text),
            "pos_tags": malaysian_manglish_nlp.pos_tag(text),
            "entities": malaysian_manglish_nlp.ner_tag(text),
            "emotion": malaysian_manglish_nlp.detect_emotion(text),
            "keywords": malaysian_manglish_nlp.extract_keywords(text),
        }


class ManglishSentimentTool(BaseTool):
    """LangChain tool for Manglish sentiment analysis."""

    name: str = "manglish_sentiment"
    description: str = (
        "Analyze the sentiment of Malaysian Manglish text. "
        "Returns sentiment label (positive/negative/neutral) and confidence score. "
        "Understands Malaysian slang, shortforms, and code-switching between "
        "Bahasa Melayu and English."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        """Analyze sentiment."""
        return malaysian_manglish_nlp.sentiment(text)


class ManglishTranslateTool(BaseTool):
    """LangChain tool for Manglish translation."""

    name: str = "manglish_translate"
    description: str = (
        "Translate Malaysian Manglish text to English, Bahasa Melayu, or formal BM. "
        "Handles code-switching, slang, and shortforms common in Malaysian informal text. "
        "Set target to 'en' for English, 'bm' for Bahasa Melayu, or 'formal' for formal BM."
    )
    args_schema: Type[BaseModel] = TranslateInput

    def _run(
        self,
        text: str,
        target: str = "en",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Translate text."""
        target = target.lower()
        if target == "en":
            return malaysian_manglish_nlp.to_english(text)
        elif target in ("bm", "ms", "malay"):
            return malaysian_manglish_nlp.to_malay(text)
        elif target == "formal":
            return malaysian_manglish_nlp.to_formal(text)
        else:
            return malaysian_manglish_nlp.translate(text)


class ManglishNERTool(BaseTool):
    """LangChain tool for Manglish Named Entity Recognition."""

    name: str = "manglish_ner"
    description: str = (
        "Extract named entities from Malaysian Manglish text. "
        "Recognizes 9 entity types: PERSON, LOCATION, ORGANIZATION, MONEY, "
        "DATE, TIME, PHONE, EMAIL, URL. Handles Malaysian names, locations, "
        "currency (RM), and local formats."
    )
    args_schema: Type[BaseModel] = NERInput

    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, str]]:
        """Extract named entities."""
        return malaysian_manglish_nlp.ner_tag(text)


# Convenience: list all available tools
def get_tools() -> List[BaseTool]:
    """Get all available manglish-nlp LangChain tools."""
    return [
        ManglishNLPTool(),
        ManglishSentimentTool(),
        ManglishTranslateTool(),
        ManglishNERTool(),
    ]

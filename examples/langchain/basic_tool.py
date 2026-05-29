"""
Basic LangChain tools for manglish-nlp.

Registers sentiment, NER, translation, and normalization modules as
LangChain tools, then demonstrates an agent that uses them together.

Run:
    pip install langchain langchain-openai manglish-nlp
    export OPENAI_API_KEY="sk-..."
    python basic_tool.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

import manglish_nlp

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(description="Malaysian Manglish text to process")

class TranslateInput(BaseModel):
    text: str = Field(description="Manglish text to translate")
    target: str = Field(
        default="en",
        description="Target language: 'en', 'bm', or 'formal'",
    )

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class SentimentTool(BaseTool):
    """Analyse sentiment of Manglish text."""

    name: str = "manglish_sentiment"
    description: str = (
        "Return sentiment (positive / negative / neutral) and confidence "
        "for Malaysian Manglish text. Understands slang, shortforms, and "
        "code-switching between Bahasa Melayu and English."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        result = manglish_nlp.sentiment(text)
        return {"text": text, **result}


class NERTool(BaseTool):
    """Extract named entities from Manglish text."""

    name: str = "manglish_ner"
    description: str = (
        "Extract named entities (PERSON, LOCATION, ORGANIZATION, MONEY, "
        "DATE, TIME, PHONE, EMAIL, URL) from Malaysian text."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, str]]:
        return manglish_nlp.ner_tag(text)


class TranslateTool(BaseTool):
    """Translate between Manglish, English, and formal BM."""

    name: str = "manglish_translate"
    description: str = (
        "Translate Manglish to English ('en'), Bahasa Melayu ('bm'), "
        "or formal BM ('formal')."
    )
    args_schema: Type[BaseModel] = TranslateInput

    def _run(
        self,
        text: str,
        target: str = "en",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, str]:
        t = target.lower()
        if t == "en":
            translated = manglish_nlp.to_english(text)
        elif t in ("bm", "ms"):
            translated = manglish_nlp.to_malay(text)
        elif t == "formal":
            translated = manglish_nlp.to_formal(text)
        else:
            translated = manglish_nlp.translate(text)
        return {"original": text, "translated": translated, "target": target}


class NormalizeTool(BaseTool):
    """Normalise Manglish shortforms and slang."""

    name: str = "manglish_normalize"
    description: str = (
        "Expand Manglish shortforms (nk→nak, brp→berapa, x→tak) and "
        "normalise slang. Returns cleaned, expanded text."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, str]:
        normalized = manglish_nlp.normalize(text)
        cleaned = manglish_nlp.clean(text)
        return {"original": text, "normalized": normalized, "cleaned": cleaned}


# ---------------------------------------------------------------------------
# Agent demo
# ---------------------------------------------------------------------------

def build_agent():
    """Build a LangChain agent with all manglish-nlp tools."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools = [SentimentTool(), NERTool(), TranslateTool(), NormalizeTool()]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Malaysian Manglish NLP assistant. "
            "Use the provided tools to analyse, translate, and normalise "
            "Manglish text. Always show your reasoning before the final answer."
        )),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    """Run example queries through the agent."""
    agent = build_agent()

    queries = [
        "Apa sentiment bagi 'gila best makanan kat kedai Pak Ali tu, murah gila RM5 je nasi goreng?'",
        "Translate 'aku nk pegi shopping mall ngan member ptg ni' to English",
        "Extract entities from 'Jumpa Dr. Ahmad kat Hospital Serdang esok pukul 3 petang, bajet RM200'",
        "Normalise and analyse: 'xpe la slow2 je kitaorg gerak nti sampai la tu'",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)
        result = agent.invoke({"input": q})
        print(f"Answer: {result['output']}")


if __name__ == "__main__":
    main()

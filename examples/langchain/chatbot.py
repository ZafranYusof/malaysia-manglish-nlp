"""
LangChain conversational agent with malaysian-manglish-nlp tools.

A chatbot that understands Manglish queries, uses NLP tools to process
them, and responds naturally in the user's preferred style.

Features:
- Conversation memory (buffer window)
- Sentiment-aware responses
- NER extraction for structured data
- Translation fallback for mixed-language queries

Run:
    pip install langchain langchain-openai malaysian-manglish-nlp
    export OPENAI_API_KEY="sk-..."
    python chatbot.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

import malaysian_manglish_nlp

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = Field(description="Manglish/Malaysian text")


class SentimentTool(BaseTool):
    name: str = "sentiment"
    description: str = "Get sentiment of Manglish text. Returns label and score."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return malaysian_manglish_nlp.sentiment(text)


class EmotionTool(BaseTool):
    name: str = "emotion"
    description: str = (
        "Detect emotion in Manglish text (anger, joy, sadness, fear, surprise, disgust). "
        "Use when you need to understand the user's emotional state."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        return malaysian_manglish_nlp.detect_emotion(text)


class NERTool(BaseTool):
    name: str = "ner"
    description: str = "Extract named entities from Malaysian text."
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> List[Dict[str, str]]:
        return malaysian_manglish_nlp.ner_tag(text)


class TranslateTool(BaseTool):
    name: str = "translate"
    description: str = (
        "Translate Manglish text. target: 'en' (English), 'bm' (Malay), 'formal' (formal BM). "
        "Use when user asks to translate or you need to understand mixed-language text."
    )
    args_schema: Type[BaseModel] = TextInput  # simplified for chat

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, str]:
        return {
            "to_english": malaysian_manglish_nlp.to_english(text),
            "to_formal": malaysian_manglish_nlp.to_formal(text),
        }


class NormalizeTool(BaseTool):
    name: str = "normalize"
    description: str = (
        "Expand Manglish shortforms and slang into standard form. "
        "Use when you need to understand abbreviated text."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, str]:
        return {"normalized": malaysian_manglish_nlp.normalize(text)}


class IntentTool(BaseTool):
    name: str = "intent"
    description: str = (
        "Classify user intent from Manglish text. Returns intent type "
        "(question, request, complaint, greeting, etc.) and confidence."
    )
    args_schema: Type[BaseModel] = TextInput

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> Dict[str, Any]:
        intent = malaysian_manglish_nlp.classify_intent(text)
        return {
            "intent": intent,
            "is_question": malaysian_manglish_nlp.is_question(text),
            "is_complaint": malaysian_manglish_nlp.is_complaint(text),
        }


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Kau adalah ManglishBot — chatbot yang faham bahasa Manglish (campuran Bahasa \
Melayu, English, dan slang Malaysia).

Rules:
1. Reply dalam bahasa yang sama dengan user (kalau user cakap Manglish, reply Manglish)
2. Guna tools untuk analyse text sebelum reply
3. Kalau user tanya sentiment, guna sentiment tool dulu
4. Kalau ada nama/tempat/harga, extract dengan NER
5. Kalau text terlalu abbreviated, normalize dulu
6. Sentiment-aware: kalau user sedih/marah, reply empati. Kalau happy, match energy.
7. Keep replies concise dan natural, jangan robotic.

Contoh interaction:
- User: "gila kejam parking RM10 sejam"
- Bot: "Parking RM10 sejam memang mahal gila. Kat mana tu? Kalau KL memang biasa kena camtu 😅"
"""


def create_chatbot():
    """Create a conversational agent with memory."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory

    tools = [
        SentimentTool(),
        EmotionTool(),
        NERTool(),
        TranslateTool(),
        NormalizeTool(),
        IntentTool(),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        return_intermediate_steps=True,
    )

    # Wrap with memory
    message_history = ChatMessageHistory()
    return RunnableWithMessageHistory(
        executor,
        lambda session_id: message_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def chat_loop():
    """Interactive chat loop."""
    print("=" * 50)
    print("  ManglishBot 🇲🇾  (type 'quit' to exit)")
    print("=" * 50)

    bot = create_chatbot()
    session_id = "default"

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "k bye"):
            print("Bot: Ok bye! Jumpa lagi 👋")
            break

        result = bot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Bot: {result['output']}")


if __name__ == "__main__":
    chat_loop()

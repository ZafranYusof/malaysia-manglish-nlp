"""
RAG pipeline using manglish-nlp for preprocessing.

Flow:
1. User asks question in Manglish
2. manglish-nlp normalises + translates the query
3. Normalised query goes to vector store retrieval
4. Sentiment of user query is detected
5. LLM generates response using retrieved context + sentiment-aware tone

Run:
    pip install langchain langchain-openai chromadb manglish-nlp
    export OPENAI_API_KEY="sk-..."
    python rag_pipeline.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import manglish_nlp

# ---------------------------------------------------------------------------
# Document store (in-memory for demo; swap with ChromaDB / FAISS in prod)
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {
        "content": "Harga petrol RON95 naik 10 sen kepada RM2.15 seliter berkuat kuasa tengah malam ini.",
        "source": "berita_harian/2024-01-15",
    },
    {
        "content": "Kerajaan umum subsidi RM600 untuk golongan B40 bermula Mac 2024.",
        "source": "thestar/economy/2024-01-10",
    },
    {
        "content": "Grab Malaysia kenakan caj RM3 untuk penghantaran makanan bawah RM15.",
        "source": "tech/2024-02-01",
    },
    {
        "content": "Jambatan Pulau Pinang ketiga dijangka siap 2028, kos RM8 bilion.",
        "source": "berita_harian/infrastruktur/2024-01-20",
    },
    {
        "content": "Nasi lemak Pak Ali di SS15 Subang menang anugerah makanan terbaik 2024.",
        "source": "makan/lifestyle/2024-02-05",
    },
    {
        "content": "LRT laluan Shah Alam akan mula beroperasi Disember 2024 dengan 25 stesen.",
        "source": "transport/2024-01-25",
    },
]


class SimpleVectorStore:
    """Minimal in-memory vector store for demo. Replace with ChromaDB/FAISS."""

    def __init__(self, docs: List[Dict[str, str]]):
        self.docs = docs

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Naive keyword overlap search. Use real embeddings in production."""
        query_words = set(query.lower().split())

        scored = []
        for doc in self.docs:
            doc_words = set(doc["content"].lower().split())
            overlap = len(query_words & doc_words)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]


# ---------------------------------------------------------------------------
# Preprocessing with manglish-nlp
# ---------------------------------------------------------------------------

def preprocess_query(raw_query: str) -> Dict[str, Any]:
    """Normalise and enrich the user query before retrieval."""
    normalised = manglish_nlp.normalize(raw_query)
    translated = manglish_nlp.to_english(raw_query)
    sentiment = manglish_nlp.sentiment(raw_query)
    entities = manglish_nlp.ner_tag(raw_query)
    keywords = manglish_nlp.extract_keywords(raw_query)
    language = manglish_nlp.detect_language(raw_query)

    # Build search query: combine normalised + translated for better retrieval
    search_query = f"{normalised} {translated}"

    return {
        "raw": raw_query,
        "normalised": normalised,
        "translated": translated,
        "sentiment": sentiment,
        "entities": entities,
        "keywords": keywords,
        "language": language,
        "search_query": search_query,
    }


def build_retrieval_context(docs: List[Dict[str, str]]) -> str:
    """Format retrieved docs into context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] {doc['content']} (source: {doc['source']})")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment-aware response generation
# ---------------------------------------------------------------------------

RESPONSE_PROMPT = ChatPromptTemplate.from_template("""\
You are a helpful Malaysian news/information assistant.

## Context (retrieved documents):
{context}

## User query (original): {raw_query}
## User query (normalised): {normalised_query}
## Detected sentiment: {sentiment}
## Detected entities: {entities}

## Instructions:
- Answer the user's question based on the context above
- If the user seems frustrated/angry (negative sentiment), be empathetic
- If the user seems happy/excited (positive sentiment), match their energy
- Reply in the same language style as the user (Manglish if they use Manglish)
- Cite sources when possible
- If context doesn't contain the answer, say so honestly

## Response:
""")


def generate_response(query_data: Dict[str, Any], context: str) -> str:
    """Generate sentiment-aware response using LLM."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = RESPONSE_PROMPT | llm | parser

    return chain.invoke({
        "context": context,
        "raw_query": query_data["raw"],
        "normalised_query": query_data["normalised"],
        "sentiment": json.dumps(query_data["sentiment"]),
        "entities": json.dumps(query_data["entities"]),
    })


# ---------------------------------------------------------------------------
# Full RAG pipeline
# ---------------------------------------------------------------------------

def rag_query(query: str, store: SimpleVectorStore) -> Dict[str, Any]:
    """End-to-end RAG with manglish-nlp preprocessing."""

    # Step 1: Preprocess
    query_data = preprocess_query(query)

    # Step 2: Retrieve (using normalised search query)
    retrieved = store.similarity_search(query_data["search_query"], k=3)
    context = build_retrieval_context(retrieved)

    # Step 3: Generate
    response = generate_response(query_data, context)

    return {
        "query": query,
        "preprocessing": {
            "normalised": query_data["normalised"],
            "translated": query_data["translated"],
            "sentiment": query_data["sentiment"],
            "entities": query_data["entities"],
        },
        "retrieved_docs": retrieved,
        "response": response,
    }


# ---------------------------------------------------------------------------
# LangChain chain version (for production use)
# ---------------------------------------------------------------------------

def build_rag_chain(store: SimpleVectorStore):
    """Build a LangChain chain version of the RAG pipeline."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    def retrieve_and_respond(inputs: dict) -> str:
        query = inputs["query"]
        query_data = preprocess_query(query)
        retrieved = store.similarity_search(query_data["search_query"], k=3)
        context = build_retrieval_context(retrieved)

        return generate_response(query_data, context)

    chain = {"query": RunnablePassthrough()} | retrieve_and_respond
    return chain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    store = SimpleVectorStore(SAMPLE_DOCS)

    queries = [
        "brapa harga petrol skarang?",
        "ada subsidi apa untuk rakyat?",
        "bila LRT baru siap wei?",
        "asal grab mahal sgt sekarang ni?!",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)

        result = rag_query(q, store)

        print(f"Normalised: {result['preprocessing']['normalised']}")
        print(f"Translated: {result['preprocessing']['translated']}")
        print(f"Sentiment:  {result['preprocessing']['sentiment']}")
        print(f"Entities:   {result['preprocessing']['entities']}")
        print(f"Retrieved:  {len(result['retrieved_docs'])} docs")
        print(f"\nResponse:   {result['response']}")


if __name__ == "__main__":
    main()

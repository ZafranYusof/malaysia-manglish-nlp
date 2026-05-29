# LangChain + manglish-nlp Examples

Integrate Malaysian Manglish NLP capabilities into LangChain agents, chatbots,
and RAG pipelines.

## Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # or use any LangChain-supported LLM
```

## Examples

### 1. Basic Tools (`basic_tool.py`)

Register manglish-nlp modules as LangChain tools:

| Tool | Function |
|------|----------|
| `SentimentTool` | Sentiment analysis with Malaysian slang support |
| `NERTool` | Named entity extraction (PERSON, LOCATION, ORG, MONEY, etc.) |
| `TranslateTool` | Translate between Manglish, English, and formal BM |
| `NormalizeTool` | Expand shortforms (nk→nak, x→tak, brp→berapa) |

```bash
python basic_tool.py
```

### 2. Chatbot (`chatbot.py`)

Conversational agent with memory that:
- Understands and replies in Manglish
- Detects sentiment and emotion for empathetic responses
- Extracts entities for structured data capture
- Classifies intent (question, complaint, greeting)

```bash
python chatbot.py
```

**Example conversation:**
```
You: wei kedai mamak depan rumah aku naik harga gila2
Bot: Kedai mamak naik harga memang sakit hati wei. Kat area mana tu?
    Maybe boleh cari alternatif lain nearby.
```

### 3. RAG Pipeline (`rag_pipeline.py`)

Retrieval-Augmented Generation with manglish-nlp preprocessing:

1. **Normalise** Manglish query (expand slang/shortforms)
2. **Translate** to English for broader document matching
3. **Retrieve** relevant documents from vector store
4. **Detect sentiment** for tone-aware response generation
5. **Generate** contextual answer citing sources

```bash
python rag_pipeline.py
```

**Note:** The demo uses a naive keyword-overlap store. For production,
swap `SimpleVectorStore` with ChromaDB, FAISS, or Pinecone using
manglish-nlp embeddings (`manglish_nlp.embeddings`).

## Customisation

### Using your own LLM

Replace `ChatOpenAI` with any LangChain-compatible model:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-haiku-20240307")
```

### Adding more tools

```python
from manglish_nlp.langchain_tool import get_tools
all_tools = get_tools()  # returns all built-in tools
```

### Using with local models

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
```

## Architecture

```
User (Manglish)
    │
    ▼
┌─────────────────────┐
│  manglish-nlp tools  │  sentiment, NER, translate, normalize
└─────────┬───────────┘
          │
    ▼              ▼
LangChain Agent    Vector Store
    │                   │
    └──────┬────────────┘
           ▼
      LLM (GPT-4o / Claude / local)
           │
           ▼
     Response (Manglish)
```

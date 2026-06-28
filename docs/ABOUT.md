# About — Ford Vehicle Safety Intelligence Agent

## What Is This?

The **Ford Vehicle Safety Intelligence Agent** is a portfolio-grade conversational AI application that lets anyone ask plain-English questions about Ford vehicle safety records — powered entirely by public NHTSA data and open-source frameworks.

Ask it things like:
- *"What are the most common issues with the 2022 Ford Bronco?"*
- *"Are there any open recalls on the 2023 Ford Explorer?"*
- *"Compare Bronco and Mustang complaints for 2023"*
- *"Which Ford models are covered?"*

It responds with structured, cited answers — complaint summaries from its local knowledge base, live component breakdowns from NHTSA, and active recall details including remedies.

---

## How It Works

### Step 1 — Ingestion (one-time setup)

`ingest.py` downloads Ford complaint summaries from the NHTSA Complaints API for each supported model and year. Each complaint text is embedded using **Google's Gemini Embedding model** (`gemini-embedding-001`) and stored in a local **ChromaDB** vector database. The ingestion is resumable — re-running skips model/year combinations already in the database.

### Step 2 — Query Execution

When you send a message via the chat UI or REST API:

1. **FastAPI** receives your message and passes it to the **LangGraph ReAct agent**
2. The agent is backed by **Groq's LLaMA-3.3-70B** model, which is fast and supports tool-calling
3. The LLM decides which tools to call:
   - **`search_vehicle_complaints`** — runs a semantic vector search over the local ChromaDB knowledge base to find relevant complaint summaries
   - **`get_complaint_stats`** — calls the live NHTSA API to get a real-time component breakdown by complaint count
   - **`get_recall_summary`** — calls the live NHTSA Recalls API to fetch active safety campaigns
4. The LLM synthesises the tool results into a formatted markdown response
5. The response is returned to your browser and rendered by the chat UI

### Step 3 — Fallback Handling

The Groq free tier has a 100,000 token/day limit. When that limit is reached, the system automatically switches to a **deterministic rule-based fallback agent** (`_FallbackAgent`) that:
- Detects query intent via regex (recalls, complaints, stats, comparison, greetings, etc.)
- Validates inputs (non-Ford brands, years outside 2020–2024)
- Calls the same NHTSA and ChromaDB tools directly — no LLM required
- Returns a meaningful response for every possible query type

This means the application is always functional, even with zero LLM quota remaining.

---

## Frameworks and Technologies

| Layer | Technology | Why |
|---|---|---|
| **Web framework** | FastAPI (Python) | Fast async API, automatic OpenAPI docs |
| **Agent orchestration** | LangGraph `create_react_agent` | ReAct loop with tool-calling and MemorySaver checkpointer for multi-turn memory |
| **Chat LLM** | Groq — LLaMA-3.3-70B | Fast inference, free tier, native tool-calling support |
| **Embeddings** | Google Gemini `gemini-embedding-001` | High-quality text embeddings, 1000 req/day free tier |
| **Vector store** | ChromaDB (PersistentClient) | Local disk-backed vector DB, no infra required |
| **LLM framework** | LangChain (tools, tool decorator) | Standard tool interface for LangGraph |
| **HTTP client** | requests | NHTSA live API calls |
| **UI** | Vanilla HTML/CSS + marked.js | Zero-dependency chat interface, markdown rendering |
| **Package manager** | uv | Fast Python package resolution |
| **Environment** | python-dotenv | `.env`-based API key management |

---

## Data Source

All safety data comes from the **[NHTSA Vehicle Safety Complaints & Recalls API](https://api.nhtsa.gov)** — a public-domain US government dataset. No API key is required to query it. Data is used under public domain terms.

---

## Logic Applied

### RAG (Retrieval-Augmented Generation)
The agent doesn't hallucinate complaint details. It first retrieves real complaint records from the vector store (or live NHTSA API) and feeds them as context to the LLM, which then synthesises and formats the answer.

### ReAct (Reason + Act) Loop
The LangGraph ReAct pattern allows the LLM to reason about what information it needs, call the appropriate tools, observe the results, and repeat — until it has enough information to produce a final answer.

### Quota Isolation
Two separate Google API keys are used:
- `GOOGLE_API_KEY` — used by `ingest.py` for batch embedding (1000 req/day quota)
- `GOOGLE_API_KEY_QUERY_EMBEDDINGS` — used by the live chat for query embedding (separate 1000 req/day quota)

This prevents ingestion from consuming the query quota and vice versa.

### Resumable Ingestion
`ingest.py` checks whether each model/year combination is already present in ChromaDB before making API calls. This allows the process to be safely interrupted and resumed without wasting embedding quota on already-processed data.

---

## Running Locally

### Prerequisites

- Python 3.11 or higher
- A [Groq API key](https://console.groq.com) (free)
- A [Google AI API key](https://aistudio.google.com) (free, for Gemini embeddings)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ford-genai-replica.git
cd ford-genai-replica

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
# — or, if you have uv —
uv sync

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys:
#   GOOGLE_API_KEY=AIza...
#   GOOGLE_API_KEY_QUERY_EMBEDDINGS=AIza...
#   GROQ_API_KEY=gsk_...

# 5. Run the ingestion pipeline (one-time)
python ingest.py
# Or with uv:
uv run python ingest.py

# 6. Start the server
uvicorn main:app --reload
# Or with uv:
uv run uvicorn main:app --reload

# 7. Open your browser
# http://localhost:8000
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Google AI API key for ingest-time embedding |
| `GOOGLE_API_KEY_QUERY_EMBEDDINGS` | Yes | Google AI API key for live query embedding (separate quota pool) |
| `GROQ_API_KEY` | Yes | Primary Groq API key |
| `GROQ_API_KEY_2` … `GROQ_API_KEY_5` | Optional | Backup Groq keys for rate-limit rotation |

---

## API Reference

### `POST /chat`

Send a message to the agent.

**Request body:**
```json
{
  "message": "What are the top issues on a 2022 Ford Bronco?",
  "thread_id": "optional-uuid-for-multi-turn"
}
```

**Response:**
```json
{
  "response": "## Top Issues — Ford BRONCO 2022\n...",
  "thread_id": "abc-123",
  "fallback": false,
  "notice": null
}
```

- `fallback: true` — the LLM was unavailable; response came from the rule-based agent
- `notice` — non-null when the knowledge base had no local match and live data was used instead

### `GET /health`

```json
{ "status": "ok" }
```

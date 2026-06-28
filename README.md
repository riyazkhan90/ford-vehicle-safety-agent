# Ford Vehicle Safety Intelligence Agent

A portfolio-grade **RAG + tool-calling agent** over NHTSA public vehicle safety data, built to mirror the architecture of enterprise GenAI Q&A platforms.

---

## Architecture

```
User Query  (chat UI / REST)
      │
      ▼
FastAPI  (/chat)
      │
      ▼
LangGraph ReAct Agent  ─── Groq  llama-3.3-70b-versatile
      │
      ├── search_vehicle_complaints ──► ChromaDB  (local vector store)
      │                                 └─ ford_complaints collection
      │                                 └─ Gemini gemini-embedding-001
      │
      ├── get_recall_summary ──────────► NHTSA Recalls API  (live)
      │
      └── get_complaint_stats ─────────► NHTSA Complaints API  (live)
```

**RAG layer** — NHTSA complaint summaries are embedded with `gemini-embedding-001` and stored in a local ChromaDB collection. At query time the agent performs semantic search with optional metadata filters (model, year).

**Tool calling** — `create_react_agent` (LangGraph) decides which tools to call based on the user's question. Tools can be chained: e.g. fetch live stats first, then do a semantic search for matching complaint summaries.

**Memory** — `MemorySaver` checkpointer enables multi-turn conversations scoped by `thread_id`.

**Resilient fallback** — When the Groq LLM is rate-limited, a deterministic rule-based `_FallbackAgent` handles every query type (recalls, complaints, comparisons, meta, greetings) without the LLM, so users always receive a meaningful response.

---

## Supported Scope

| Dimension | Value |
|---|---|
| Make | Ford |
| Models | Bronco, Explorer, Mustang, Escape, Edge |
| Year range | 2020 – 2024 |
| Data source | NHTSA public API (no auth required) |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) — free tier, no credit card
- [Google AI API key](https://aistudio.google.com) — free tier (Gemini embeddings)

### 2. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ford-vehicle-safety-agent.git
cd ford-vehicle-safety-agent

python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GOOGLE_API_KEY=AIza...                    # ingest-time embedding quota
GOOGLE_API_KEY_QUERY_EMBEDDINGS=AIza...   # live query embedding quota (separate pool)
GROQ_API_KEY=gsk_...
```

> **Tip:** Use two separate Google AI API keys — one for ingest and one for live queries — so they don't share the 1,000 requests/day free-tier quota.

### 4. Run the ingestion pipeline (one-time)

```bash
python ingest.py
# or: uv run python ingest.py
```

This downloads Ford complaint records from NHTSA, embeds them with Gemini, and stores them in `./chroma_db/`. The process is **resumable** — re-running skips already-ingested model/year combinations.

### 5. Start the server

```bash
uvicorn main:app --reload
# or: uv run uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Example Queries

| Query | What happens |
|---|---|
| `What are the top issues on a 2022 Bronco?` | KB semantic search + live NHTSA stats |
| `Any open recalls on the 2023 Explorer?` | Live NHTSA recalls API |
| `Recalls for Mustang` | Auto-selects most recent year with data |
| `Compare Bronco and Explorer 2022` | Side-by-side component breakdown |
| `Which model years are supported?` | Reads ChromaDB metadata + lists coverage |
| `Toyota Camry issues` | Graceful redirect (Ford-only scope) |
| `Bronco 2020 recalls` | Explains Bronco wasn't produced in 2020 |

---

## API Reference

### `POST /chat`

```json
// Request
{ "message": "What are the top issues on a 2022 Ford Bronco?", "thread_id": "optional-uuid" }

// Response
{
  "response": "## Ford BRONCO 2022 — Top Complaint Areas\n...",
  "thread_id": "abc-123",
  "fallback": false,
  "notice": null
}
```

- `fallback: true` — LLM was rate-limited; rule-based agent handled the query
- `notice` — non-null when local KB had no match and live NHTSA data was used instead

### `GET /health`

```json
{ "status": "ok" }
```

---

## Production Comparison

| Component | This repo | Production |
|---|---|---|
| Vector store | Local ChromaDB | pgvector on Cloud SQL / Pinecone |
| Embedding | Gemini API (free tier) | Batched async pipeline, dedicated quota |
| LLM | Groq free tier (llama-3.3-70b) | Groq / Anthropic production tier |
| State / checkpointing | In-memory MemorySaver | Redis or Postgres-backed checkpointer |
| Serving | Single Uvicorn process | Cloud Run / GKE with horizontal scaling |
| Data freshness | Manual re-ingest | Pub/Sub triggered on NHTSA feed updates |

The LangGraph agent graph, tool interface, and RAG retrieval pattern are identical at both scales.

---

## Project Structure

```
├── agent/
│   ├── graph.py        # LangGraph agent, Groq key rotation, fallback agent
│   └── tools.py        # Three LangChain tools + ChromaDB + NHTSA clients
├── static/
│   └── index.html      # Chat UI (dark theme, marked.js markdown)
├── docs/
│   ├── ABOUT.md            # How it works + local setup guide
│   ├── PRD.md              # Product requirements document
│   ├── TECHNICAL_DESIGN.md # Architecture + error handling matrix
│   └── BUILD_PROMPT.md     # Prompt to recreate this project from scratch
├── ingest.py           # One-time embedding ingest pipeline
├── main.py             # FastAPI entry point
├── requirements.txt    # pip dependencies
├── pyproject.toml      # uv project config
└── .env.example        # Environment variable template
```

---

## Data Source

[NHTSA Vehicle Safety Complaints & Recalls API](https://api.nhtsa.gov) — US public domain, no API key required.

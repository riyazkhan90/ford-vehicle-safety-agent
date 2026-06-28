# Technical Design Document — Ford Vehicle Safety Intelligence Agent

## 1. System Overview

This system is a **Retrieval-Augmented Generation (RAG) + tool-calling agent** that answers natural-language questions about Ford vehicle safety using NHTSA public data. It combines a local vector store of pre-embedded complaint summaries with live NHTSA API lookups, orchestrated by a LangGraph ReAct agent backed by Groq's LLaMA-3.3-70B model.

---

## 2. Architecture

```
User Query (HTTP POST /chat)
        │
        ▼
  FastAPI  (/chat endpoint)
        │
        ▼
  LangGraph ReAct Agent  ─── ChatGroq (llama-3.3-70b-versatile)
        │
        ├── search_vehicle_complaints ──► ChromaDB (local)
        │                                 └─ ford_complaints collection
        │                                 └─ Gemini embeddings (query key)
        │
        ├── get_recall_summary ──────────► NHTSA Recalls API (live)
        │
        └── get_complaint_stats ─────────► NHTSA Complaints API (live)
```

When the Groq LLM is rate-limited (daily token quota exhausted), the system falls back to a deterministic rule-based `_FallbackAgent` that handles every query class without requiring the LLM.

---

## 3. Component Breakdown

### 3.1 Ingest Pipeline (`ingest.py`)

| Property | Value |
|---|---|
| Data source | NHTSA Complaints API (`/complaintsByVehicle`) |
| Models covered | Bronco, Explorer, Mustang, Escape, Edge |
| Year range | 2020–2024 |
| Embedding model | `models/gemini-embedding-001` (Google Generative AI) |
| Vector store | ChromaDB PersistentClient (`./chroma_db`) |
| Collection | `ford_complaints` |
| Max docs per model/year | 150 |
| Batch size | 100 per embed call |
| Resumable | Yes — `already_ingested()` skips completed model/year combos |

**Quota management:**
- Free-tier Gemini embedding limit: 1,000 requests/day
- Dual API keys: `GOOGLE_API_KEY` (ingest) and `GOOGLE_API_KEY_QUERY_EMBEDDINGS` (live chat) — separate quota pools
- Exponential backoff on 429 errors (up to 6 retries)

### 3.2 Agent Layer (`agent/graph.py`)

**`_GroqAgentWithKeyRotation`**
- Wraps `create_react_agent` from LangGraph
- Holds a list of Groq API keys (`GROQ_API_KEY`, `GROQ_API_KEY_2` … `GROQ_API_KEY_5`)
- On 429 rate limit: rotates to the next key and retries
- On exhaustion of all keys: raises to trigger the fallback path in `main.py`

**`_FallbackAgent`**
- Pure-Python rule-based agent — no LLM dependency
- Intent detection via regex patterns covering: greetings, thanks, recalls, complaints, stats, comparisons, meta/coverage, help, and general vehicle queries
- Input validation: rejects non-Ford brands, out-of-range years (< 2020 or > 2024)
- Smart partial-query handling: model-only → tries 2023 first; year-only → defaults to Explorer
- Calls the same tool functions (`get_recall_summary_raw`, `get_complaint_stats_raw`, `search_vehicle_complaints_raw`) as the LLM agent

**`_trim_hook`**
- LangGraph pre-model hook
- Trims oldest messages when context exceeds `_MAX_TOKENS = 5,500` tokens (rough estimate: 1 token ≈ 4 characters)
- Prevents context-window overflow on long conversations

### 3.3 Tool Layer (`agent/tools.py`)

| Tool | Source | Description |
|---|---|---|
| `search_vehicle_complaints` | ChromaDB (local) | Semantic similarity search over ingested complaint summaries; filters by model/year metadata |
| `get_recall_summary` | NHTSA Recalls API | Fetches active recall campaigns for a model/year; returns campaign number, component, summary, consequence, remedy |
| `get_complaint_stats` | NHTSA Complaints API | Fetches all complaints for a model/year; returns top-5 components by count |

**Validation applied to all three tools:**
- `_validate_model_year()` — rejects unsupported models or years outside 2020–2024 before making any API call
- HTTP 400 from NHTSA → user-friendly message (e.g. Bronco 2020 does not exist)
- HTTP timeout → actionable timeout message
- ChromaDB query failure (embedding quota) → graceful degradation to live NHTSA stats

### 3.4 API Layer (`main.py`)

- **Framework:** FastAPI
- **Endpoint:** `POST /chat` — accepts `{ message, thread_id? }`, returns `{ response, thread_id, fallback, notice? }`
- **Memory:** `MemorySaver` checkpointer keyed by `thread_id` for multi-turn conversation
- **Static UI:** served from `./static/` at `/`
- **CORS:** configured for `localhost:8000` and `localhost:8001`
- **Rate-limit catch:** if `agent.invoke()` raises a 429, `main.py` invokes `_FallbackAgent` directly

---

## 4. Data Flow — Query Execution

```
1. POST /chat  { message: "What are the top issues on a 2022 Bronco?" }
2. FastAPI calls agent.invoke({ messages: [...] }, config={ thread_id })
3. LangGraph routes to ChatGroq (llama-3.3-70b-versatile)
4. LLM decides to call:
     a. get_complaint_stats("bronco", 2022)  →  live NHTSA API
     b. search_vehicle_complaints("Bronco 2022 issues", "bronco", 2022)
           → ChromaDB semantic search → Gemini embeds query → top-5 docs
5. LLM synthesises results into formatted markdown
6. FastAPI extracts last AI message, wraps in ChatResponse
7. Static UI renders markdown with marked.js
```

---

## 5. Key Design Decisions

| Decision | Rationale |
|---|---|
| Dual Gemini API keys | Ingest and live-query on separate quota pools to avoid interference |
| Groq instead of Gemini for chat | Groq's inference speed; LLaMA-3.3-70B supports tool-calling |
| Absolute ChromaDB path | Avoids `./chroma_db` resolving to wrong directory when uvicorn CWD differs from project root |
| `_FallbackAgent` handles all intents | Users always get a meaningful response even when LLM quota is exhausted |
| No `--reset` on ingest | Resumable ingestion avoids wasting embedding quota on already-ingested data |
| `MAX_PER_COMBO = 150` | Stays within daily free-tier limit across 25 model/year combos (5 models × 5 years) |

---

## 6. Error Handling Matrix

| Failure | Detected By | Response |
|---|---|---|
| Groq 429 (key N) | `_is_rate_limit_error()` | Rotate to key N+1 |
| Groq 429 (all keys) | key rotation exhausted | `main.py` invokes `_FallbackAgent` |
| Gemini embed quota (query) | Exception in `collection.query()` | Graceful fallback to live NHTSA stats |
| ChromaDB path not found | Exception in `_get_collection()` | `_get_collection()` returns `None`; tool degrades to live data |
| NHTSA 400 (bad model/year) | `resp.status_code == 400` | User-friendly explanation |
| NHTSA timeout | `requests.Timeout` | Informative retry message |
| Non-Ford brand in query | Regex in `_FallbackAgent._validate_query()` | Scope-redirect message |
| Year out of 2020–2024 range | `_validate_model_year()` | Clear range error |

---

## 7. File Structure

```
ford_genai_replica/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph agent, key rotation, fallback agent
│   └── tools.py          # LangChain tools, ChromaDB, NHTSA API clients
├── static/
│   ├── index.html        # Chat UI (marked.js markdown rendering)
│   └── Ford_Banner.png
├── tests/                # Test directory
├── chroma_db/            # Persisted ChromaDB vector store (git-ignored)
├── ingest.py             # One-time embedding ingest pipeline
├── main.py               # FastAPI application entry point
├── requirements.txt      # pip-installable dependencies
├── pyproject.toml        # uv project configuration
├── .env.example          # Environment variable template
└── docs/
    ├── TECHNICAL_DESIGN.md   (this file)
    ├── PRD.md
    ├── ABOUT.md
    └── BUILD_PROMPT.md
```

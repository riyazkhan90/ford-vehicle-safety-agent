# Product Requirements Document — Ford Vehicle Safety Intelligence Agent

## 1. Product Overview

**Product Name:** Ford Vehicle Safety Intelligence Agent
**Type:** Portfolio / demonstration project
**Domain:** Automotive safety data, Generative AI, RAG systems

The Ford Vehicle Safety Intelligence Agent is a conversational AI assistant that enables users to query, analyse, and understand Ford vehicle safety data from the US National Highway Traffic Safety Administration (NHTSA). It combines a local semantic knowledge base with live API lookups to answer questions about complaints, recalls, and component failure patterns.

---

## 2. Problem Statement

Ford vehicle owners, safety researchers, and automotive analysts must navigate raw NHTSA datasets to understand vehicle safety trends. Existing NHTSA tools (safercar.gov, NHTSA API) are data-first, not insight-first — users must know which endpoints to query and how to interpret raw complaint records.

There is no conversational interface that allows a user to ask "What are the top braking issues on a 2022 Bronco?" and get an immediate, synthesised, cited answer.

---

## 3. Goals and Success Metrics

| Goal | Metric |
|---|---|
| Answer vehicle safety questions in natural language | All queries return a response with cited NHTSA data |
| No dead-end responses | Zero queries result in an uncaught error or empty response |
| Accurate recall and complaint data | All data sourced directly from NHTSA public API |
| Fast response time | P90 response time < 10 seconds including LLM call |
| Graceful degradation | When LLM is unavailable, rule-based agent handles all query types |

---

## 4. User Stories

| ID | As a... | I want to... | So that... |
|---|---|---|---|
| US-01 | Ford vehicle owner | Ask about common problems with my specific model and year | I can identify known issues before they escalate |
| US-02 | Car buyer | Check if a vehicle I'm considering has open recalls | I can make an informed purchase decision |
| US-03 | Safety researcher | Compare complaint counts across two Ford models | I can identify relative safety risk |
| US-04 | General user | Ask what vehicles and years are covered | I understand the scope before querying |
| US-05 | General user | Ask a vague question and still get useful guidance | I'm never left with a generic error message |

---

## 5. Functional Requirements

### 5.1 Query Types Supported

| Query Type | Example | Source |
|---|---|---|
| Model + year complaint search | "What are the issues with a 2022 Explorer?" | ChromaDB (local KB) + NHTSA live |
| Recall lookup | "Any open recalls on the 2023 Bronco?" | NHTSA Recalls API |
| Complaint statistics | "Top failure components for 2021 Mustang" | NHTSA Complaints API |
| Model-only query | "Common problems for Escape" | KB search + live stats (2023 default) |
| Year-only query | "What failed most in 2022?" | NHTSA live stats (Explorer as reference) |
| Model comparison | "Compare Bronco vs Explorer 2022" | NHTSA live stats for both |
| Coverage / meta | "Which model years are supported?" | ChromaDB metadata |
| Greeting / help | "Hello", "What can you do?" | Built-in response |
| Polite acknowledgement | "Thanks" | Built-in response |

### 5.2 Input Validation

- **Unsupported Ford model** → inform user of supported models
- **Non-Ford brand** → redirect to Ford-only scope
- **Year outside 2020–2024** → explain the supported range
- **NHTSA 400 (model/year doesn't exist)** → explain (e.g. Bronco 2020 wasn't produced)
- **Empty query** → return capability summary

### 5.3 Data Sources

| Source | Type | Requirement |
|---|---|---|
| ChromaDB local collection | Vector store | Pre-populated via `ingest.py` |
| NHTSA Complaints API | REST (live) | Public, no auth |
| NHTSA Recalls API | REST (live) | Public, no auth |

### 5.4 API

**Endpoint:** `POST /chat`

Request:
```json
{ "message": "string", "thread_id": "string (optional)" }
```

Response:
```json
{
  "response": "string (markdown)",
  "thread_id": "string",
  "fallback": "boolean",
  "notice": "string | null"
}
```

**Endpoint:** `GET /health`
```json
{ "status": "ok" }
```

---

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Availability | Server must start and serve requests even if Groq API is unavailable |
| Resilience | All queries must return a non-error response regardless of upstream API status |
| Quota management | Gemini embedding quota split across two API keys (ingest vs. query) |
| Security | API keys stored in `.env` only; never committed to version control |
| Portability | Runs locally on macOS / Linux with Python 3.11+ |
| Observability | All rate limit events and fallback activations logged at WARNING level |

---

## 7. Supported Scope

| Dimension | Scope |
|---|---|
| Make | Ford only |
| Models | Bronco, Explorer, Mustang, Escape, Edge |
| Years | 2020 – 2024 |
| Data | NHTSA complaints and recalls (public domain) |
| Language | English only |

---

## 8. Out of Scope

- Other vehicle makes (Toyota, GM, etc.)
- Vehicle pricing, availability, or purchasing information
- Service centre locations or booking
- Real-time crash or accident data
- Ford model years before 2020 or after 2024
- Predictive / prescriptive recommendations ("you should replace your brakes")

---

## 9. Constraints

- **Free-tier Gemini API:** 1,000 embedding requests/day — ingest is resumable and capped at 150 documents per model/year combo
- **Free-tier Groq API:** 100,000 tokens/day shared across all API keys from the same organisation — rule-based fallback ensures continuity
- **Local storage:** ChromaDB persists to disk; production would require a managed vector store (e.g. pgvector)

---

## 10. Future Enhancements

| Enhancement | Priority |
|---|---|
| Ingest remaining models (Mustang, Escape, Edge 2020-2024) | High |
| Expand Explorer coverage to all years | High |
| Add Ford F-150 and other popular models | Medium |
| Production vector store (pgvector / Pinecone) | Medium |
| Streaming responses via SSE | Medium |
| Citation links to NHTSA complaint pages | Low |
| Multi-language support | Low |

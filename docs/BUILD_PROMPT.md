# Build Prompt — Ford Vehicle Safety Intelligence Agent

Use the following prompt with Claude (or any capable LLM with code generation) to recreate this project from scratch.

---

## The Prompt

```
Build a portfolio-grade RAG + tool-calling agent over public Ford vehicle safety data.

## Stack
- Python 3.11+
- FastAPI for the REST API
- LangGraph (create_react_agent) for the ReAct agent loop
- LangChain tool decorator for defining tools
- Groq (llama-3.3-70b-versatile) as the chat/reasoning LLM
- Google Gemini (gemini-embedding-001) for generating embeddings
- ChromaDB (PersistentClient) as the local vector store
- python-dotenv for environment management
- requests for live NHTSA API calls
- A static HTML/CSS/JS chat UI served by FastAPI

## Data Source
NHTSA public API:
- Complaints: https://api.nhtsa.gov/complaints/complaintsByVehicle?make=ford&model={model}&modelYear={year}
- Recalls: https://api.nhtsa.gov/recalls/recallsByVehicle?make=ford&model={model}&modelYear={year}

## Coverage
- Models: bronco, explorer, mustang, escape, edge
- Years: 2020–2024
- Max 150 complaints per model/year combo

## File structure to create:
  ingest.py          — one-time embedding pipeline (resumable, no --reset)
  main.py            — FastAPI server with /chat and /health endpoints
  agent/__init__.py  — empty
  agent/graph.py     — LangGraph agent, Groq key rotation, fallback agent
  agent/tools.py     — three LangChain tools + ChromaDB + NHTSA clients
  static/index.html  — chat UI with markdown rendering (marked.js)
  .env.example       — environment variable template
  requirements.txt   — pip dependencies

## Ingest pipeline (ingest.py)

- Use Google Generative AI SDK (google.genai) directly (not langchain-google-genai)
- Implement a custom GeminiEmbeddingFunction(chromadb.EmbeddingFunction)
- Keep track of which model/year combos are already ingested via collection.get() checks
- Cap at MAX_PER_COMBO = 150 documents per model/year
- Use batch_size = 100 for embedding calls
- Retry on 429 (quota) with exponential backoff: parse suggested retry delay from error string
- Use GOOGLE_API_KEY env var for ingest (separate from query key)
- Never use --reset (resumable ingestion only)
- Store metadata fields: model, year, odi_number, component (use "components" field from NHTSA, fall back to "component")

## Three LangChain tools (agent/tools.py)

1. search_vehicle_complaints(query: str, model: str = None, year: int = None)
   - Embeds query via Gemini (GOOGLE_API_KEY_QUERY_EMBEDDINGS, falling back to GOOGLE_API_KEY)
   - Runs ChromaDB collection.query() with optional metadata filters
   - Returns top-5 complaint summaries with ODI number and component
   - If no results: fallback to get_complaint_stats live data
   - If collection unavailable: fallback to live stats with explanation
   - Wrap exceptions from collection.query() gracefully

2. get_recall_summary(model: str, year: int)
   - Calls NHTSA recalls API
   - Handle HTTP 400 with a user-friendly explanation (Bronco 2020 doesn't exist etc.)
   - Handle timeout separately
   - Validate model (must be in supported set) and year (2020-2024) upfront

3. get_complaint_stats(model: str, year: int)
   - Calls NHTSA complaints API
   - Handle HTTP 400 explicitly before calling raise_for_status()
   - Returns top-5 components by complaint count
   - Validate model and year upfront

For both get_recall_summary and get_complaint_stats:
- Add _validate_model_year(model, year) that returns an error string for unsupported inputs
- Use requests.Timeout exception for timeout handling

Export each tool as both:
  search_vehicle_complaints = tool(_search_vehicle_complaints)
  search_vehicle_complaints_raw = _search_vehicle_complaints

## Agent (agent/graph.py)

### Main agent
- Use ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
- Read up to 5 Groq keys from env: GROQ_API_KEY, GROQ_API_KEY_2 ... GROQ_API_KEY_5
  (also support comma-separated GROQ_API_KEYS)
- Implement _GroqAgentWithKeyRotation that tries each key on 429 and raises on the last
- Use MemorySaver checkpointer for multi-turn memory keyed by thread_id
- Add a pre_model_hook (_trim_hook) that trims oldest messages when context > 5500 tokens
- Use absolute path for ChromaDB: os.path.dirname(os.path.dirname(__file__)) / "chroma_db"

### System prompt
Tell the LLM to:
- Always call get_complaint_stats first, then search_vehicle_complaints
- Call get_recall_summary when user asks about recalls
- Answer "which models/years are supported?" directly without calling tools
- Use markdown formatting with ### headings per model and bullet lists per component
- Flag safety-critical issues with ⚠

### Fallback agent (_FallbackAgent)
When Groq is unavailable (all keys rate-limited), use this rule-based agent.
It must handle ALL of these query types without an LLM:
  - Greetings ("hi", "hello") → helpful welcome message
  - Thanks ("thanks", "ok") → polite acknowledgement
  - Meta queries ("which models/years supported?") → read ChromaDB metadata, list indexed combos
  - Recall queries → call get_recall_summary_raw; if model-only, try years 2023→2022→2024→2021→2020
  - Complaint/stats queries → call search_vehicle_complaints_raw + get_complaint_stats_raw
  - Comparison queries ("compare Bronco vs Explorer") → side-by-side complaint stats for both models
  - Model-only queries → KB search across all years + live stats for most recent year with data
  - Year-only queries → default to Explorer for that year as reference
  - Non-Ford brand mentioned → explain the Ford-only scope
  - Year outside 2020-2024 → clear out-of-range message
  - Help queries ("what can you do?") → capability summary
  - Empty input → capability summary

## FastAPI (main.py)

- Mount static files from ./static at /static
- GET / → redirect to /static/index.html
- POST /chat:
  - Accept { message: str, thread_id: str? }
  - Return { response: str, thread_id: str, fallback: bool, notice: str? }
  - If agent raises with "429" in error text → invoke _FallbackAgent as fallback
  - Set notice field when KB had no match and live data was used
- GET /health → { status: "ok" }
- CORS: allow localhost:8000 and localhost:8001

## Static UI (static/index.html)

- Dark theme matching Ford brand colours (dark navy background, amber accents)
- Chat bubble layout with user right, assistant left
- Use marked.js (CDN) for markdown rendering in assistant messages
- Show a loading spinner while waiting for response
- Thread ID persisted in sessionStorage for multi-turn memory
- Display a notice banner when fallback=true or notice is non-null

## Environment variables (.env.example)

GOOGLE_API_KEY=AIza...              # for ingest.py
GOOGLE_API_KEY_QUERY_EMBEDDINGS=AIza...  # for live query embeddings
GROQ_API_KEY=gsk_...
# Optional backup keys:
# GROQ_API_KEY_2=gsk_...
# GROQ_API_KEY_3=gsk_...

## Important constraints

1. NEVER use --reset in ingest.py; always resume from existing collection state
2. Use absolute ChromaDB path derived from __file__, not "./chroma_db"
3. Wrap collection.query() in try/except — quota exhaustion must degrade gracefully
4. Add _validate_model_year() validation to all three tools before making any API call
5. The fallback agent must return a meaningful non-error response for every possible input
6. Use SUPPORTED_MODELS = frozenset({"bronco","explorer","mustang","escape","edge"})
   and SUPPORTED_YEARS = frozenset(range(2020, 2025)) as shared constants
```

---

## Incremental Build Order

If building step-by-step, implement in this order to allow testing at each stage:

1. **`.env` and `requirements.txt`** — set up keys and install dependencies
2. **`ingest.py`** — verify ChromaDB is populated before building the agent
3. **`agent/tools.py`** — test each tool independently with direct Python calls
4. **`agent/graph.py`** — verify the ReAct agent calls tools correctly
5. **`main.py`** — wire up FastAPI; test with `curl`
6. **`static/index.html`** — add the UI last once the API is stable

---

## Testing Checklist

After building, verify these queries all return meaningful responses:

```
"hi"                                           → greeting
"Which model years are supported?"             → coverage list from ChromaDB
"What are the top issues with the 2022 Bronco?" → KB results + live stats
"Any open recalls on the 2023 Explorer?"       → NHTSA recall campaigns
"Recalls for Mustang"                          → auto-selects 2023
"Recalls for Bronco 2020"                      → explains 2020 doesn't exist
"Compare Bronco and Explorer 2022"             → side-by-side stats
"Problems in 2022"                             → defaults to Explorer, asks for model
"Toyota Camry issues"                          → Ford-only redirect
"Bronco 2019 problems"                         → out-of-range year error
"thanks"                                       → polite acknowledgement
""  (empty)                                    → capability summary
```

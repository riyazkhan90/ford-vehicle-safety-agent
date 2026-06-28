"""
LangChain tools for the Ford vehicle intelligence agent.
"""

import os
from typing import Optional
from collections import Counter

try:
    import requests
except Exception:  # pragma: no cover - depends on environment
    requests = None

try:
    import chromadb
except Exception:  # pragma: no cover - depends on environment
    chromadb = None

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover - depends on environment
    google_genai = None

try:
    from langchain.tools import tool
except Exception:  # pragma: no cover - depends on environment
    def tool(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - depends on environment
    def load_dotenv():
        return False

load_dotenv()

NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
NHTSA_RECALLS_URL = "https://api.nhtsa.gov/recalls/recallsByVehicle"
MAKE = "ford"
COLLECTION_NAME = "ford_complaints"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"

SUPPORTED_MODELS = frozenset({"bronco", "explorer", "mustang", "escape", "edge"})
SUPPORTED_YEARS = frozenset(range(2020, 2025))

_chroma_client = None
_collection = None


def _validate_model_year(model: Optional[str], year: Optional[int]) -> Optional[str]:
    """Return a user-facing error string when inputs are out of range, else None."""
    errors = []
    if model and model.lower() not in SUPPORTED_MODELS:
        pretty = ", ".join(m.title() for m in sorted(SUPPORTED_MODELS))
        errors.append(f"'{model}' is not a supported Ford model. Supported models: {pretty}.")
    if year is not None and year not in SUPPORTED_YEARS:
        errors.append(
            f"{year} is outside the supported year range (2020–2024). "
            "Please try a model year between 2020 and 2024."
        )
    return " ".join(errors) if errors else None


if chromadb is not None:
    class _GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self, api_key: str, model_name: str):
            self._client = google_genai.Client(api_key=api_key)
            self._model = model_name

        def __call__(self, input: list) -> list:
            import time
            from google.genai.errors import ClientError
            for attempt in range(5):
                try:
                    response = self._client.models.embed_content(
                        model=self._model,
                        contents=input,
                    )
                    return [e.values for e in response.embeddings]
                except ClientError as e:
                    if "429" in str(e) and attempt < 4:
                        time.sleep(15 * (attempt + 1))
                    else:
                        raise
else:
    class _GeminiEmbeddingFunction:  # pragma: no cover - fallback path
        def __init__(self, api_key: str, model_name: str):
            self._api_key = api_key
            self._model = model_name

        def __call__(self, input: list) -> list:
            raise RuntimeError("chromadb is not installed")


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        # Use the query-only key so live chat doesn't compete with ingest quota
        api_key = os.getenv("GOOGLE_API_KEY_QUERY_EMBEDDINGS") or os.getenv("GOOGLE_API_KEY")
        if chromadb is None or google_genai is None or not api_key:
            return None
        try:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
            _chroma_client = chromadb.PersistentClient(path=db_path)
            ef = _GeminiEmbeddingFunction(api_key=api_key, model_name=GEMINI_EMBEDDING_MODEL)
            _collection = _chroma_client.get_collection(
                COLLECTION_NAME, embedding_function=ef
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Failed to open ChromaDB collection")
            return None
    return _collection


def _search_vehicle_complaints(query: str, model: str = None, year: int = None) -> str:
    """
    Semantic search over the Ford complaints vector store.
    Returns the top 5 most relevant complaint summaries with component and ODI number.
    Optionally filter by vehicle model (e.g. 'explorer') and/or year (e.g. 2022).
    """
    if not query or not query.strip():
        query = f"vehicle safety complaints {model or 'Ford'} {year or ''}".strip()

    err = _validate_model_year(model, year)
    if err:
        return err

    collection = _get_collection()

    if collection is None:
        if model and year:
            return (
                "The vector store is unavailable, so I am using live NHTSA complaint statistics instead.\n\n"
                + _get_complaint_stats(model=model, year=year)
            )
        if model:
            return (
                "The vector store is unavailable right now. "
                "I can still provide live NHTSA complaint statistics if you include a model and year, e.g. '2023 Explorer'."
            )
        if year:
            return (
                "The vector store is unavailable right now. "
                "I can still provide live NHTSA complaint statistics if you include a Ford model, e.g. 'Explorer'."
            )
        return (
            "The vector store is unavailable right now. "
            "Please provide a specific Ford model and year so I can return live NHTSA complaint or recall information."
        )

    where: Optional[dict] = None
    if model and year:
        where = {"$and": [{"model": model.lower()}, {"year": year}]}
    elif model:
        where = {"model": model.lower()}
    elif year:
        where = {"year": year}

    try:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("collection.query failed: %s", e)
        if model and year:
            return (
                "The knowledge-base search is temporarily unavailable (embedding quota may be exhausted). "
                "Falling back to live NHTSA data:\n\n"
                + _get_complaint_stats(model=model, year=year)
            )
        return (
            "The knowledge-base search is temporarily unavailable. "
            "Provide a specific Ford model and year for live NHTSA data."
        )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        notice = (
            "Information is not available in the local knowledge base at the moment. "
            "No matching complaints were found in the vector store for this query. "
            "I am providing relevant live Ford vehicle safety information instead.\n\n"
        )

        if model and year:
            live_stats = _get_complaint_stats(model=model, year=year)
            return notice + live_stats

        return (
            notice +
            "If you provide a specific Ford model and year, I can summarize live NHTSA complaint trends or recall data."
        )

    lines = [f"Top {len(docs)} complaints matching '{query}' from the knowledge base:\n"]
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        lines.append(
            f"{i}. [{meta.get('model', '?').upper()} {meta.get('year', '?')}] "
            f"ODI #{meta.get('odi_number', 'N/A')} | Component: {meta.get('component', 'N/A')}\n"
            f"   {doc[:300]}{'...' if len(doc) > 300 else ''}"
        )
    return "\n".join(lines)

search_vehicle_complaints = tool(_search_vehicle_complaints)
search_vehicle_complaints_raw = _search_vehicle_complaints


def _get_recall_summary(model: str, year: int) -> str:
    """
    Fetches active NHTSA recall campaigns for a Ford vehicle by model and year.
    Returns formatted recall details including campaign number, component, summary, consequence, and remedy.
    """
    if requests is None:
        return "Live recall lookup is unavailable because the requests dependency is not installed."

    err = _validate_model_year(model, year)
    if err:
        return err

    try:
        resp = requests.get(
            NHTSA_RECALLS_URL,
            params={"make": MAKE, "model": model.lower(), "modelYear": year},
            timeout=15,
        )
        if resp.status_code == 400:
            return (
                f"No recall data found for Ford {model.upper()} {year}. "
                "NHTSA does not recognise this model/year combination — "
                f"the {model.title()} may not have been produced in {year} "
                "(e.g. the Bronco was reintroduced starting with the 2021 model year). "
                "Try a year in the 2021–2024 range."
            )
        resp.raise_for_status()
        recalls = resp.json().get("results", [])
    except requests.Timeout:
        return f"The NHTSA recall API timed out for Ford {model.upper()} {year}. Please try again."
    except Exception as e:
        return f"Error fetching recalls for Ford {model.upper()} {year}: {e}"

    if not recalls:
        return f"No open recalls found for Ford {model.upper()} {year}."

    lines = [f"NHTSA Recalls for Ford {model.upper()} {year} ({len(recalls)} campaigns):\n"]
    for r in recalls:
        lines.append(
            f"Campaign: {r.get('NHTSACampaignNumber', 'N/A')}\n"
            f"  Component:    {r.get('Component', 'N/A')}\n"
            f"  Summary:      {r.get('Summary', 'N/A')}\n"
            f"  Consequence:  {r.get('Consequence', 'N/A')}\n"
            f"  Remedy:       {r.get('Remedy', 'N/A')}\n"
        )
    return "\n".join(lines)

get_recall_summary = tool(_get_recall_summary)
get_recall_summary_raw = _get_recall_summary


def _get_complaint_stats(model: str, year: int) -> str:
    """
    Fetches all NHTSA complaints for a Ford vehicle and returns the top 5 components by complaint count.
    Useful for understanding which systems are most problematic on a given model/year.
    """
    if requests is None:
        return "Live complaint lookup is unavailable because the requests dependency is not installed."

    err = _validate_model_year(model, year)
    if err:
        return err

    try:
        resp = requests.get(
            NHTSA_COMPLAINTS_URL,
            params={"make": MAKE, "model": model.lower(), "modelYear": year},
            timeout=15,
        )
        if resp.status_code == 400:
            return (
                f"No complaint data found for Ford {model.upper()} {year}. "
                "NHTSA does not recognise this model/year combination — "
                f"the {model.title()} may not have been produced in {year} "
                "(e.g. the Bronco was reintroduced starting with the 2021 model year). "
                "Try a year in the 2021–2024 range."
            )
        data = resp.json()
        if resp.status_code != 200:
            if data and data.get("count", 0) == 0:
                return f"No complaints on record for Ford {model.upper()} {year}."
            resp.raise_for_status()
        complaints = data.get("results", [])
    except requests.Timeout:
        return f"The NHTSA complaints API timed out for Ford {model.upper()} {year}. Please try again."
    except Exception as e:
        return f"Error fetching complaints for Ford {model.upper()} {year}: {e}"

    if not complaints:
        return f"No complaints on record for Ford {model.upper()} {year}."

    counter = Counter(
        comp.strip()
        for c in complaints
        for comp in (c.get("components") or c.get("component") or "UNKNOWN").split(",")
    )
    top5 = counter.most_common(5)

    lines = [
        f"Complaint breakdown for Ford {model.upper()} {year} "
        f"({len(complaints)} total complaints, live NHTSA data):\n"
        f"Top {len(top5)} components by complaint count:"
    ]
    for component, count in top5:
        lines.append(f"  - {component}: {count} complaints")

    return "\n".join(lines)

get_complaint_stats = tool(_get_complaint_stats)
get_complaint_stats_raw = _get_complaint_stats

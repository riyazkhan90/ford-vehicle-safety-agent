"""
Ingests Ford vehicle complaints from NHTSA API into a local Chroma collection.
Embeds each complaint's summary text with gemini-embedding-001.

Resumable: re-running skips model/year combos already present in the collection.
Pass --reset to wipe and re-ingest everything from scratch.
"""

import os
import re
import sys
import time
import requests
import chromadb
from google import genai
from google.genai.errors import ClientError
from dotenv import load_dotenv

load_dotenv()

NHTSA_COMPLAINTS_URL = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
MAKE = "ford"
MODELS = ["bronco", "explorer", "mustang", "escape", "edge"]
YEARS = list(range(2020, 2025))
COLLECTION_NAME = "ford_complaints"
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"


class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name

    def __call__(self, input: list) -> list:
        for attempt in range(6):
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=input,
                )
                return [e.values for e in response.embeddings]
            except ClientError as e:
                if "429" in str(e) and attempt < 5:
                    # Parse the API-suggested retry delay; fall back to 30s * attempt
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(e))
                    suggested = float(m.group(1)) if m else 0
                    wait = max(suggested + 10, 30 * (attempt + 1))
                    print(f"  [rate limit] waiting {int(wait)}s before retry {attempt + 1}/5...")
                    time.sleep(wait)
                else:
                    raise


def fetch_complaints(make: str, model: str, year: int) -> list[dict]:
    try:
        resp = requests.get(
            NHTSA_COMPLAINTS_URL,
            params={"make": make, "model": model, "modelYear": year},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  [warn] fetch failed for {make}/{model}/{year}: {e}")
        return []


def already_ingested(collection, model: str, year: int) -> bool:
    """Return True if this model/year combo already has documents in the collection."""
    result = collection.get(where={"$and": [{"model": model}, {"year": year}]}, limit=1)
    return len(result["ids"]) > 0


def main():
    reset = "--reset" in sys.argv

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment / .env")

    chroma = chromadb.PersistentClient(path="./chroma_db")
    ef = GeminiEmbeddingFunction(api_key=api_key, model_name=GEMINI_EMBEDDING_MODEL)

    if reset:
        try:
            chroma.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
        collection = chroma.create_collection(COLLECTION_NAME, embedding_function=ef)
    else:
        collection = chroma.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)
        existing = collection.count()
        if existing:
            print(f"Resuming — collection already has {existing} documents. Run with --reset to start fresh.")

    total_ingested = 0

    for model in MODELS:
        for year in YEARS:
            if already_ingested(collection, model, year):
                print(f"  {model} {year}: already ingested, skipping")
                continue

            complaints = fetch_complaints(MAKE, model, year)

            if not complaints:
                print(f"  {model} {year}: no complaints found, skipping")
                continue

            MAX_PER_COMBO = 150
            docs, ids, metas = [], [], []
            for c in complaints:
                if len(docs) >= MAX_PER_COMBO:
                    break
                summary = (c.get("summary") or "").strip()
                if not summary:
                    continue
                odi = str(c.get("odiNumber", ""))
                doc_id = f"{model}_{year}_{odi}" if odi else f"{model}_{year}_{len(docs)}"
                docs.append(summary)
                ids.append(doc_id)
                metas.append(
                    {
                        "model": model,
                        "year": year,
                        "odi_number": odi,
                        "component": c.get("components") or c.get("component") or "UNKNOWN",
                    }
                )

            if not docs:
                print(f"  {model} {year}: {len(complaints)} complaints but no usable summaries")
                continue

            batch_size = 100  # API max per call
            for i in range(0, len(docs), batch_size):
                collection.add(
                    documents=docs[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                    metadatas=metas[i : i + batch_size],
                )
                if i + batch_size < len(docs):
                    time.sleep(30)  # conservative gap between batches

            total_ingested += len(docs)
            print(f"  {model} {year}: ingested {len(docs)} complaints")

    print(f"\nDone. Collection '{COLLECTION_NAME}' total: {collection.count()} documents ({total_ingested} added this run)")


if __name__ == "__main__":
    main()

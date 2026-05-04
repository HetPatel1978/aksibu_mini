"""
index/vector_store.py
──────────────────────
ChromaDB-based vector store for semantic search across all normalized records.

ChromaDB runs locally (no server needed) and auto-embeds text using
sentence-transformers under the hood.

Provides:
  index_records()    — embed and store all normalized records
  vector_search()    — semantic similarity search with optional filters
  get_by_id()        — retrieve a specific record by ID

The vector store handles "what does X look like" questions.
The graph handles "how is X connected to Y" questions.
Together they cover the full retrieval surface.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

os.makedirs("data/chroma", exist_ok=True)

# ── CHROMA CLIENT ─────────────────────────────────────────────
# PersistentClient saves embeddings to disk — survives restarts.
_chroma = chromadb.PersistentClient(path="data/chroma")

# Use sentence-transformers for free local embeddings
# (no OpenAI API calls needed for embedding)
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _collection():
    return _chroma.get_or_create_collection(
        name="engineering_context",
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_records(normalized_records: list[dict]):
    """
    Embed and store all normalized records in ChromaDB.
    Each record's title + body is embedded as a single chunk.
    Metadata is stored for filtering (source type, owner, timestamp).
    """
    col = _collection()

    # Check what's already indexed to avoid re-embedding
    existing = set(col.get()["ids"])

    ids, docs, metas = [], [], []

    for r in normalized_records:
        if r["id"] in existing:
            continue  # already indexed

        text = f"{r['title']}\n\n{r['body']}"
        ids.append(r["id"])
        docs.append(text)
        metas.append({
            "type":      r.get("type") or "",
            "source":    r.get("source") or "",
            "owner":     r.get("owner") or "",
            "timestamp": r.get("timestamp") or "",
            "services":  ",".join(r.get("services_mentioned") or []),
            "priority":  r.get("priority") or "",
            "status":    r.get("status") or "",
        })

    if not ids:
        print("  ℹ️  Vector store: all records already indexed")
        return

    # ChromaDB handles batching internally
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"  ✅ Vector store: indexed {len(ids)} new records")


def vector_search(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
    type_filter: str | None = None,
) -> list[dict]:
    """
    Semantic similarity search.

    Args:
        query:         Natural language query
        n_results:     Number of results to return
        source_filter: Optional — filter by source ('jira', 'github', 'confluence', 'logs')
        type_filter:   Optional — filter by type ('ticket', 'commit', 'doc', 'log')

    Returns list of dicts with id, text, metadata, distance.
    """
    col = _collection()

    where = {}
    if source_filter:
        where["source"] = source_filter
    if type_filter:
        where["type"] = type_filter

    kwargs = {
        "query_texts": [query],
        "n_results": min(n_results, col.count() or 1),
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    return [
        {
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],  # convert distance to similarity
        }
        for i in range(len(results["ids"][0]))
    ]


def get_stats() -> dict:
    """Return basic stats about the vector store."""
    col = _collection()
    return {"record_count": col.count()}


def get_by_id(record_id: str) -> dict | None:
    """Retrieve a single record from the vector store by ID."""
    col = _collection()
    result = col.get(ids=[record_id], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {
        "id":       result["ids"][0],
        "text":     result["documents"][0],
        "metadata": result["metadatas"][0],
    }

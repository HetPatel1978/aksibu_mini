"""
index/doc_store.py
───────────────────
Manages the `doc_chunks` ChromaDB collection and a serialized BM25 index.
Fully parallel to vector_store.py — does not touch `aksibu_records`.
"""

import os
import pickle
from functools import lru_cache

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from ingestion.doc_chunker import ChunkDict

_DOC_COLLECTION = "doc_chunks"
_BM25_PATH = "data/doc_bm25.pkl"


# ── ChromaDB client & collection ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _ef():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


@lru_cache(maxsize=1)
def _chroma():
    os.makedirs("data/doc_chroma", exist_ok=True)
    return chromadb.PersistentClient(path="data/doc_chroma")


def _collection():
    return _chroma().get_or_create_collection(
        name=_DOC_COLLECTION,
        embedding_function=_ef(),
        metadata={"hnsw:space": "cosine"},
    )


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_documents(
    chunks: list[ChunkDict],
    rebuild_bm25: bool = True,
) -> dict:
    col = _collection()

    existing = set(col.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if new_chunks:
        col.upsert(
            ids=[c["chunk_id"] for c in new_chunks],
            documents=[c["text"] for c in new_chunks],
            metadatas=[
                {
                    "doc_id":       c["doc_id"],
                    "filename":     c["filename"],
                    "chunk_index":  c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                }
                for c in new_chunks
            ],
        )

    if rebuild_bm25:
        _rebuild_bm25()

    return {
        "indexed": len(new_chunks),
        "skipped": len(chunks) - len(new_chunks),
        "total_chunks_in_store": col.count(),
    }


def ingest_folder(
    folder_path: str,
    batch_size: int = 200,
    progress_callback=None,
) -> dict:
    """
    Walk folder_path recursively, chunk every .pdf/.txt/.md file, and upsert
    to the doc_chunks collection in batches.  BM25 is rebuilt exactly once at
    the end so memory and time stay flat regardless of corpus size.

    progress_callback(files_done, files_total, chunks_indexed) is called after
    each file if provided — use it to drive a Streamlit progress bar.

    Returns:
        {"files_found": int, "files_processed": int, "files_skipped": int,
         "chunks_indexed": int, "chunks_skipped": int,
         "total_chunks_in_store": int}
    """
    from ingestion.doc_chunker import iter_folder, chunk_file_from_path

    stats = {
        "files_found": 0, "files_processed": 0, "files_skipped": 0,
        "chunks_indexed": 0, "chunks_skipped": 0,
    }

    batch: list[ChunkDict] = []
    files_done = 0
    files_total = 0

    for file_path, total in iter_folder(folder_path):
        if file_path == "":          # sentinel with total count
            files_total = total
            stats["files_found"] = total
            if total == 0:
                return {**stats, "total_chunks_in_store": _collection().count()}
            continue

        try:
            chunks = chunk_file_from_path(file_path)
        except Exception as ex:
            print(f"  ⚠️  Skipping {file_path}: {ex}")
            stats["files_skipped"] += 1
            files_done += 1
            continue

        batch.extend(chunks)
        files_done += 1
        stats["files_processed"] += 1

        if len(batch) >= batch_size:
            result = ingest_documents(batch, rebuild_bm25=False)
            stats["chunks_indexed"] += result["indexed"]
            stats["chunks_skipped"] += result["skipped"]
            batch = []

        if progress_callback:
            progress_callback(files_done, files_total, stats["chunks_indexed"])

    # Flush remaining batch
    if batch:
        result = ingest_documents(batch, rebuild_bm25=False)
        stats["chunks_indexed"] += result["indexed"]
        stats["chunks_skipped"] += result["skipped"]

    # Single BM25 rebuild over the full corpus
    _rebuild_bm25()

    return {**stats, "total_chunks_in_store": _collection().count()}


# ── BM25 ─────────────────────────────────────────────────────────────────────

def _rebuild_bm25() -> None:
    col = _collection()
    all_data = col.get(include=["documents", "metadatas"])

    if not all_data["ids"]:
        return

    corpus_tokens = [doc.lower().split() for doc in all_data["documents"]]
    bm25 = BM25Okapi(corpus_tokens)

    os.makedirs("data", exist_ok=True)
    with open(_BM25_PATH, "wb") as f:
        pickle.dump(
            {
                "bm25":      bm25,
                "ids":       all_data["ids"],
                "metadatas": all_data["metadatas"],
            },
            f,
        )


def _load_bm25() -> tuple | None:
    if not os.path.exists(_BM25_PATH):
        return None
    with open(_BM25_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["ids"], data["metadatas"]


def bm25_search(query: str, top_k: int = 50) -> list[dict]:
    loaded = _load_bm25()
    if loaded is None:
        return []
    bm25, ids, metadatas = loaded

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"chunk_id": ids[i], "score": float(scores[i]), "metadata": metadatas[i]}
        for i in top_indices
    ]


# ── Vector search ─────────────────────────────────────────────────────────────

def vector_search_docs(query: str, top_k: int = 50) -> list[dict]:
    col = _collection()
    count = col.count()
    if count == 0:
        return []

    n = min(top_k, count)
    raw = col.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for i, chunk_id in enumerate(raw["ids"][0]):
        dist = raw["distances"][0][i]
        results.append({
            "chunk_id": chunk_id,
            "score":    float(1.0 - dist),
            "text":     raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
        })
    return results


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    bm25_res    = bm25_search(query, top_k=50)
    vector_res  = vector_search_docs(query, top_k=50)

    fused_ids = _rrf_fuse(bm25_res, vector_res)[:top_k]

    # Build a lookup from the combined result sets for metadata/text
    lookup: dict[str, dict] = {}
    for r in bm25_res:
        lookup[r["chunk_id"]] = r
    for r in vector_res:
        if r["chunk_id"] not in lookup:
            lookup[r["chunk_id"]] = r
        else:
            # Prefer vector result (has text)
            lookup[r["chunk_id"]]["text"] = r.get("text", "")

    return [lookup[cid] for cid in fused_ids if cid in lookup]


# ── Fetch by IDs ──────────────────────────────────────────────────────────────

def get_chunks_by_ids(chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    col = _collection()
    raw = col.get(
        ids=chunk_ids,
        include=["documents", "metadatas"],
    )
    result = []
    for i, cid in enumerate(raw["ids"]):
        result.append({
            "chunk_id": cid,
            "text":     raw["documents"][i],
            "metadata": raw["metadatas"][i],
        })
    return result


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_store_stats() -> dict:
    col = _collection()
    count = col.count()
    if count == 0:
        return {"chunk_count": 0, "doc_count": 0, "filenames": []}

    all_meta = col.get(include=["metadatas"])["metadatas"]
    filenames = sorted({m["filename"] for m in all_meta if m.get("filename")})
    doc_ids   = {m["doc_id"] for m in all_meta if m.get("doc_id")}

    return {
        "chunk_count": count,
        "doc_count":   len(doc_ids),
        "filenames":   filenames,
    }

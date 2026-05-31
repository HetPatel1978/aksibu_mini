"""
retrieval/cited_retriever.py
─────────────────────────────
Hybrid retrieval + cross-encoder reranking + citation-enforced generation.
Fully parallel to agent.py — does not use graph tools or aksibu_records.
"""

import functools
import os
import re
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

from openai import OpenAI
from index.doc_store import hybrid_search, get_chunks_by_ids

_api_key = os.getenv("OPENROUTER_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "\n\nOPENROUTER_API_KEY not found.\n"
        "Make sure aksibu-mini/.env contains:\n"
        "  OPENROUTER_API_KEY=sk-or-v1-...\n"
    )

_client = OpenAI(api_key=_api_key, base_url="https://openrouter.ai/api/v1")
_FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "anthropic/claude-sonnet-4-5")

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Cross-encoder (lazy-loaded once per process) ──────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(_CROSS_ENCODER_MODEL)


# ── Reranker ──────────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []
    ce = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
    return [c for _, c in ranked[:top_n]]


# ── Citation prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise document analyst. Your only job is to answer \
questions strictly from the provided source passages.

Rules:
1. Every factual claim MUST be followed immediately by an inline citation [N] where N \
is the source number.
2. You may combine information from multiple sources: "X [1] and Y [2]."
3. You MUST NOT state anything that cannot be attributed to the provided sources.
4. If the answer is not in the sources, say exactly: \
"The provided documents do not contain enough information to answer this question."
5. Do not invent, extrapolate, or paraphrase beyond what the sources literally state."""


def _build_user_message(query: str, chunks: list[dict]) -> str:
    source_block = ""
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        filename    = meta.get("filename", "unknown")
        chunk_index = meta.get("chunk_index", 0)
        total       = meta.get("total_chunks", 1)
        source_block += (
            f"[{i}] (from: {filename}, chunk {chunk_index + 1} of {total})\n"
            f"{chunk['text']}\n\n"
        )

    return (
        f"Question: {query}\n\n"
        f"Sources:\n{source_block.rstrip()}\n\n"
        f"Answer with inline citations. Begin your answer now:"
    )


# ── Citation parser ───────────────────────────────────────────────────────────

def _parse_citations(raw_answer: str, chunks: list[dict]) -> dict:
    valid_refs = set(range(1, len(chunks) + 1))
    used_refs  = {int(m) for m in re.findall(r'\[(\d+)\]', raw_answer)}
    valid_used = used_refs & valid_refs
    invalid    = used_refs - valid_refs

    # Strip sentences that cite a ref that wasn't supplied
    cleaned = raw_answer
    for ref in invalid:
        cleaned = re.sub(rf'[^.!?\n]*\[{ref}\][^.!?\n]*[.!?\n]?', '', cleaned)
    cleaned = cleaned.strip()

    citations = []
    for i, chunk in enumerate(chunks):
        ref_num = i + 1
        if ref_num in valid_used:
            meta = chunk.get("metadata", {})
            citations.append({
                "ref":         ref_num,
                "chunk_id":    chunk["chunk_id"],
                "filename":    meta.get("filename", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks":meta.get("total_chunks", 1),
                "text":        chunk["text"],
            })

    return {"answer": cleaned, "citations": citations}


# ── LLM call ─────────────────────────────────────────────────────────────────

def _llm_cited_answer(query: str, chunks: list[dict]) -> dict:
    user_msg = _build_user_message(query, chunks)
    try:
        resp = _client.chat.completions.create(
            model=_FRONTIER_MODEL,
            max_tokens=800,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as ex:
        raw = f"LLM call failed: {ex}"

    return _parse_citations(raw, chunks)


# ── Main entry point ──────────────────────────────────────────────────────────

def retrieve_with_citations(query: str, top_k: int = 5) -> dict:
    """
    Full pipeline: hybrid search → cross-encoder rerank → LLM with citations.

    Returns:
        {
          "answer": str,
          "citations": list[dict],
          "retrieval_stats": {"bm25_hits": int, "vector_hits": int,
                              "after_rrf": int, "after_rerank": int}
        }
    """
    from index.doc_store import bm25_search, vector_search_docs

    bm25_raw   = bm25_search(query, top_k=50)
    vector_raw = vector_search_docs(query, top_k=50)

    # hybrid_search runs RRF internally; we call it directly for the fused list
    rrf_candidates = hybrid_search(query, top_k=20)

    # Fetch full text for any candidates that came from BM25 (may lack "text")
    ids_needing_text = [
        c["chunk_id"] for c in rrf_candidates if not c.get("text")
    ]
    if ids_needing_text:
        fetched = {r["chunk_id"]: r for r in get_chunks_by_ids(ids_needing_text)}
        for c in rrf_candidates:
            if not c.get("text") and c["chunk_id"] in fetched:
                c["text"]     = fetched[c["chunk_id"]]["text"]
                c["metadata"] = fetched[c["chunk_id"]]["metadata"]

    # Drop any candidate still missing text (shouldn't happen in practice)
    rrf_candidates = [c for c in rrf_candidates if c.get("text")]

    top_chunks = rerank(query, rrf_candidates, top_n=top_k)

    answer_data = _llm_cited_answer(query, top_chunks)

    return {
        **answer_data,
        "retrieval_stats": {
            "bm25_hits":    len(bm25_raw),
            "vector_hits":  len(vector_raw),
            "after_rrf":    len(rrf_candidates),
            "after_rerank": len(top_chunks),
        },
    }

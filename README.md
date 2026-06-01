# Aksibu Mini — Engineering Context Engine

> Ask complex operational questions across your entire engineering stack and get grounded, cited answers.

A production-ready mini-implementation of a context aggregation system. Ingests fragmented engineering data (Jira, GitHub, Confluence, Logs), builds a GraphRAG knowledge index, and also supports large-scale document search with inline citations — so every answer is traceable to its exact source.

**Live demo:** [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hetpatel1978-aksibu-mini-uiapp-jj4vup.streamlit.app/)

---

## What It Does

**Tab 1 — Engineering Context Engine**
> *"Why is checkout broken and who should I page?"*
> *"What changed in payment-gateway recently?"*
> *"Give me a root cause analysis of the current incident."*

Aggregates Jira tickets, GitHub commits, Confluence docs, and logs into a unified GraphRAG knowledge base. An agentic retrieval loop picks the right tools, traverses the graph, and synthesises an answer — all at near-zero cost.

**Tab 2 — Document Search (new)**
> *"What does our architecture doc say about caching?"*
> Upload any folder of PDFs, markdown, or text files — even millions of them — and ask questions. Every sentence in the answer cites the exact source chunk it came from.

**Tab 3 — Health Dashboard**
Live index stats and an interactive 3D knowledge graph you can rotate, zoom, and filter by node/edge type.

---

## Architecture

### Pipeline 1 — GraphRAG (Engineering Context)

```
[Data Sources]
 Jira · GitHub · Confluence · Logs
          ↓
[Normalization]          unified schema · entity/owner resolution
          ↓
[Tiered Entity Extraction]
  Tier 1 — regex rules       free,  ~70% of records
  Tier 2 — cheap LLM         $0.0002/record
  Tier 3 — frontier LLM      $0.003/record  (complex docs only)
  + extraction cache          never reprocess same text
  + incremental manifest      skip unchanged records
  + semantic deduplication    cosine > 0.95 removed
          ↓
[Knowledge Graph]   networkx DiGraph
  Nodes: service · engineer · ticket · commit · log · doc
  Edges: owns · affects · fixes · depends_on · caused_by · modified
          ↓
[Vector Store]      ChromaDB (all-MiniLM-L6-v2, local, free)
          ↓
[Agentic Retrieval]
  Simple queries  → fast path (vector search only)
  Complex queries → tool-use loop (up to 7 calls)
    Tools: semantic_search · get_service_context · find_owner
           get_recent_changes · get_active_incidents · get_dependencies
          ↓
[Streamlit UI]
```

### Pipeline 2 — Cited Document Search (large-scale RAG)

```
[Upload files  OR  point at a local folder]
  Recursive walk · .pdf · .txt · .md
          ↓
[Chunker]   400-token sliding window · 50-token overlap
          ↓
[Index]
  ChromaDB collection  doc_chunks   (HNSW, cosine, O(log n) at 1M docs)
  BM25 index           doc_bm25.pkl (rebuilt once per ingest)
          ↓
[Hybrid Retrieval]
  BM25 top-50  +  vector top-50
          → RRF fusion (k=60) → top-20 candidates
          → cross-encoder rerank (ms-marco-MiniLM-L-6-v2) → top-5
          ↓
[Citation-Enforced Generation]
  System prompt forbids uncited claims
  Post-processing validates every [N] ref against supplied chunks
  Strips sentences with out-of-range citations
          ↓
[Answer + expandable source cards]
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-...
```

Get your key from **[openrouter.ai/keys](https://openrouter.ai/keys)** — free tier is enough to run the full demo.

### 3. Build the knowledge index

```bash
python main.py
```

Runs once (or when data changes). Incremental — safe to re-run, skips unchanged records.

### 4. Launch the UI

```bash
streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501). All three tabs are live immediately.

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | **Required.** OpenRouter key for all LLM calls | — |
| `CHEAP_MODEL` | Tier 2 extraction model | `google/gemma-4-27b-it:free` |
| `FRONTIER_MODEL` | Tier 3 extraction + document search generation | `openai/gpt-oss-120b:free` |
| `AGENT_MODEL` | Agentic query answering | `openai/gpt-oss-120b:free` |
| `MAX_AGENT_TOOL_CALLS` | Per-query tool call cap | `7` |
| `MAX_EXTRACTION_TOKENS` | Token cap per record during extraction | `800` |

Free models on OpenRouter share per-model daily quotas. If one hits its limit, swap the model ID in `.env` — no rebuild needed.

---

## File Guide

### Existing pipeline

| File | What it does |
|---|---|
| `mock_data_generator.py` | Creates realistic fake Jira / GitHub / Confluence / Log data |
| `ingestion/normalizer.py` | Converts all 4 sources into one unified schema |
| `ingestion/rule_extractor.py` | Free regex-based entity extraction (Tier 1) |
| `ingestion/tiered_extractor.py` | Cost-optimised pipeline: cache → rules → cheap LLM → frontier LLM |
| `graph/graph_builder.py` | Builds networkx knowledge graph from extracted entities |
| `graph/graph_query.py` | Graph traversal: ownership, recent changes, dependencies, incidents |
| `index/vector_store.py` | ChromaDB semantic search over engineering records |
| `retrieval/agent.py` | Agentic retrieval loop with 6 tools + cost controls |
| `ui/app.py` | Main Streamlit page — chat interface |
| `ui/pages/health.py` | Health dashboard + interactive 3D knowledge graph |
| `main.py` | One-command index builder |

### Document Search pipeline (new)

| File | What it does |
|---|---|
| `ingestion/doc_chunker.py` | Sliding-window chunker; `chunk_file_from_path()` and `iter_folder()` for large corpora |
| `index/doc_store.py` | `doc_chunks` ChromaDB collection + BM25 index; `ingest_folder()` batched ingestion |
| `retrieval/cited_retriever.py` | Hybrid search → cross-encoder rerank → citation-enforced LLM generation |
| `ui/pages/doc_search.py` | Upload files or point at a folder; cited Q&A with expandable source cards |

---

## Cost Profile

### GraphRAG pipeline

| Scenario | Cost |
|---|---|
| First full index build (18 mock records) | ~$0.01 |
| Re-index with no changes | $0.00 (all cached) |
| Per agent query | ~$0.01–0.05 |
| Naive approach (frontier LLM every record) | ~$3,500+ at real scale |

### Document Search pipeline

| Scenario | Cost |
|---|---|
| Chunking + BM25 index | $0.00 (local, in-process) |
| Embedding 1M chunks with all-MiniLM-L6-v2 | $0.00 (local model) |
| Cross-encoder reranking | $0.00 (local model) |
| Per search query (LLM generation only) | ~$0.01–0.03 |

---

## Example Questions

**Engineering Context tab:**
- *"Why is the checkout service returning 503 errors?"*
- *"Who owns payment-gateway and who should I page?"*
- *"What changed in payment-gateway recently?"*
- *"Give me a root cause analysis of the current incident"*
- *"What are the dependencies of checkout-api?"*
- *"Which open P1 incidents exist right now?"*

**Document Search tab:**
- *"What does the architecture doc say about database failover?"*
- *"Which research paper covers 3D foot tracking?"*
- *"Summarise the caching strategy across all uploaded docs"*

---

<img width="1495" height="1216" alt="Engineering Context Engine" src="https://github.com/user-attachments/assets/8d766421-6a0a-49f5-a813-288ac1bdf739" />

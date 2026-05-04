# Aksibu Mini — Engineering Context Engine

A working mini-implementation of the context aggregation system Aksibu is building.

Ingests fragmented engineering data (Jira, GitHub, Confluence, Logs), builds a
GraphRAG knowledge index, and answers questions like:

> *"Why is checkout broken and who should I page?"*

---

## Architecture

```
[Data Sources]                    [Cost-Optimized Extraction]
Jira | GitHub | Confluence | Logs
         ↓
[Normalization Layer]             ← unified schema, entity/owner resolution
         ↓
[Tiered Entity Extraction]        ← rules (free) → haiku ($0.0002) → sonnet ($0.003)
   + Incremental indexing         ← skip unchanged records
   + Deduplication                ← skip near-identical records
   + Extraction cache             ← never reprocess same text
         ↓
[Knowledge Graph]  (networkx)     ← nodes + edges: services, engineers, tickets, commits
[Vector Store]     (ChromaDB)     ← semantic search across all records
         ↓
[Agentic Retrieval]               ← LLM decides what to look up, traverses graph
   + Tool call budget cap         ← no runaway API cost
   + Simple query fast path       ← cheap questions skip the agent
         ↓
[Streamlit UI]
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
# Edit .env and set ANTHROPIC_API_KEY=your_key_here
```

**Get your key from:** https://console.anthropic.com/

### 3. Build the index

```bash
python main.py
```

This runs once (or when data changes). Incremental — safe to re-run.

### 4. Launch the UI

```bash
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser.

---

## Where to Set Your API Key

**Only one place:** `.env` file in the project root.

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

The following scripts use the API key (via `python-dotenv`):
- `ingestion/tiered_extractor.py` — LLM entity extraction (Tier 2 + 3)
- `retrieval/agent.py` — agentic query answering

All other scripts (normalization, graph building, vector store) are free — no API calls.

---

## File Guide

| File | What it does |
|---|---|
| `mock_data_generator.py` | Creates realistic fake Jira/GitHub/Confluence/Log data |
| `ingestion/normalizer.py` | Converts all 4 sources into one unified schema |
| `ingestion/rule_extractor.py` | Free regex-based entity extraction (Tier 1) |
| `ingestion/tiered_extractor.py` | Full cost-optimized pipeline: cache → rules → cheap LLM → frontier LLM |
| `graph/graph_builder.py` | Builds networkx knowledge graph from extracted entities |
| `graph/graph_query.py` | Graph traversal: ownership, recent changes, dependencies, incidents |
| `index/vector_store.py` | ChromaDB semantic search across all records |
| `retrieval/agent.py` | Agentic retrieval loop with tool use + cost controls |
| `ui/app.py` | Streamlit web UI |
| `main.py` | One-command index builder |

---

## Cost Profile

| Scenario | Cost |
|---|---|
| First full index build (mock data, 14 records) | ~$0.01 |
| Re-index with no changes | $0.00 (all cached) |
| Per agent query | ~$0.01–0.05 |
| Naive approach (frontier LLM every record) | ~$0.04 for mock data, $3,500+ at real scale |

---

## Example Questions to Try

- *"Why is the checkout service returning 503 errors?"*
- *"Who owns payment-gateway and who should I page?"*
- *"What changed in payment-gateway recently?"*
- *"Give me a root cause analysis of the current incident"*
- *"What are the dependencies of checkout-api?"*
- *"Which open incidents exist right now?"*

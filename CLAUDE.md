# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Aksibu Mini** is a Python-based Engineering Context Engine that aggregates fragmented engineering data (Jira, GitHub, Confluence, Logs) into a unified knowledge base and answers complex operational questions via an agentic retrieval system.

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY=sk-or-v1-...

# Build the knowledge index (Steps 1–5: mock data → normalize → extract → graph → vectors)
python main.py

# Launch the web UI (http://localhost:8501)
streamlit run ui/app.py
```

No linting or test suite is configured.

## Architecture

The system has two distinct phases:

### Index Build Phase (`main.py` runs these sequentially)

1. **Mock Data Generation** (`mock_data_generator.py`) — Creates realistic fake data in `data/raw/{confluence,jira,github,logs}.json`

2. **Normalization** (`ingestion/normalizer.py`) — Converts all 4 source formats to a unified schema; resolves service and owner aliases; outputs `data/normalized/all.json`

3. **Tiered Entity Extraction** (`ingestion/tiered_extractor.py`) — The core cost-optimization mechanism:
   - **Tier 1 (free):** `ingestion/rule_extractor.py` — regex-based extraction, handles ~70% of records
   - **Tier 2 (cheap):** Claude Haiku via OpenRouter — processes text unresolved by rules
   - **Tier 3 (frontier):** Claude Sonnet via OpenRouter — triggered for complex records (Confluence docs, body > 800 chars, long bodies with few entities found)
   - Backed by SQLite extraction cache (`data/extraction_cache.db`) and incremental manifest (`data/index_manifest.db`) — re-runs are safe and nearly free

4. **Knowledge Graph** (`graph/graph_builder.py`) — Builds a `networkx.DiGraph` from entities and relationships. Node types: `service`, `engineer`, `ticket`, `commit`, `log`, `doc`. Edge types: `owns`, `affects`, `fixes`, `depends_on`, `caused_by`, `modified`, `mentions`, `owned_by`. Saved to `data/graph.json`.

5. **Vector Store** (`index/vector_store.py`) — Embeds records with `all-MiniLM-L6-v2` (local, free) into ChromaDB (persistent at `data/chroma/`).

### Query Phase (`retrieval/agent.py`)

- **Simple queries** (< 8 words, starts with what/who/how) → fast path: vector search only
- **Complex queries** → agentic loop using Claude Sonnet with up to 7 tool calls (configurable via `MAX_AGENT_TOOL_CALLS`)
- Available tools: `semantic_search`, `get_service_context`, `find_owner`, `get_recent_changes`, `get_active_incidents`, `get_dependencies`
- Deduplicates repeated identical tool calls; stops with a budget notice if the call cap is hit

## Key Conventions

**Node ID format:** Normalized IDs use double-underscore prefix — `service__checkout_api`, `engineer__rahul_sharma`. Record nodes preserve original source IDs (CONF-001, JR-441, etc.).

**Tier escalation heuristics** (in `tiered_extractor.py`): body > 800 chars → frontier; < 2 entities + body > 300 chars → frontier; source is Confluence → frontier; 0 relationships + > 3 entities → frontier.

**Deduplication threshold:** Cosine similarity > 0.95 (sentence-transformers embeddings) removes near-identical records before indexing.

**API access:** Both `ingestion/tiered_extractor.py` and `retrieval/agent.py` consume `OPENROUTER_API_KEY` via `python-dotenv`. OpenRouter exposes an OpenAI-compatible endpoint.

**All external dependencies are local except OpenRouter** — ChromaDB, NetworkX, and sentence-transformers run entirely in-process with no external servers.

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Required. OpenRouter API key for LLM calls |
| `CHEAP_MODEL` | Tier 2 model (default: `anthropic/claude-haiku-4-5`) |
| `FRONTIER_MODEL` | Tier 3 model (default: `anthropic/claude-sonnet-4-5`) |
| `AGENT_MODEL` | Query answering model (default: `anthropic/claude-sonnet-4-5`) |
| `MAX_AGENT_TOOL_CALLS` | Per-query tool call cap (default: `7`) |
| `MAX_EXTRACTION_TOKENS` | Token cap per record during extraction (default: `800`) |

## Cost Profile

- First full index build (14 mock records): ~$0.01
- Re-index with no changes: $0.00 (all cached)
- Per agent query: ~$0.01–$0.05

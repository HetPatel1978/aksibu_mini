"""
main.py
────────
One-time index builder. Run this before launching the UI.

What it does:
  Step 1 — Generates mock data (Jira, GitHub, Confluence, Logs) → data/raw/
  Step 2 — Normalizes all sources into unified schema → data/normalized/
  Step 3 — Runs tiered entity extraction + builds knowledge graph → data/graph.json
  Step 4 — Embeds records into ChromaDB vector store → data/chroma/

After this runs successfully, start the UI:
  streamlit run ui/app.py

Re-running main.py is safe — incremental indexing skips unchanged records.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ── PRE-FLIGHT CHECK ──────────────────────────────────────────
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key or api_key == "your_openrouter_api_key_here":
    print("❌ OPENROUTER_API_KEY not set.")
    print("   1. Copy .env.example to .env")
    print("   2. Set your API key from https://openrouter.ai/keys")
    sys.exit(1)

print("=" * 55)
print("  Aksibu Mini — Engineering Context Engine")
print("  Index Builder")
print("=" * 55)

# ── STEP 1: MOCK DATA ─────────────────────────────────────────
print("\n📁 Step 1: Generating mock data...")
import mock_data_generator  # noqa — runs on import

# ── STEP 2: NORMALIZE ─────────────────────────────────────────
print("\n🔧 Step 2: Normalizing all sources...")
from ingestion.normalizer import normalize_all
records = normalize_all()

# ── STEP 3: GRAPH ─────────────────────────────────────────────
print("\n🕸️  Step 3: Building knowledge graph (tiered extraction)...")
from graph.graph_builder import build_graph, save_graph
G = build_graph(records)
save_graph(G)

# ── STEP 4: VECTOR STORE ──────────────────────────────────────
print("\n📦 Step 4: Indexing into ChromaDB vector store...")
from index.vector_store import index_records
# Strip raw field before indexing
clean_records = [{k: v for k, v in r.items() if k != "raw"} for r in records]
index_records(clean_records)

# ── DONE ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ Index built successfully!")
print("=" * 55)
print("\nTo launch the UI:")
print("  streamlit run ui/app.py")
print("\nOr query directly in Python:")
print("  from retrieval.agent import run_agent")
print("  result = run_agent('Why is checkout broken?')")
print("  print(result['answer'])")

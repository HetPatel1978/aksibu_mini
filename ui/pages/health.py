"""
ui/pages/health.py
───────────────────
Health dashboard — shows index build status and key counts.
Streamlit discovers this file automatically and adds it to sidebar nav.

Run the main app with: streamlit run ui/app.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(page_title="Health", page_icon="🟢", layout="wide")

GRAPH_PATH = "data/graph.json"


@st.cache_data(ttl=60)
def _load_health_stats() -> dict:
    stats = {}

    # ── Records in vector store ────────────────────────────────
    try:
        from index.vector_store import get_stats
        stats["record_count"] = get_stats()["record_count"]
    except Exception as e:
        stats["record_count"] = None
        stats["record_count_err"] = str(e)

    # ── Graph node count (read JSON directly, no NetworkX load) ─
    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats["node_count"] = len(data.get("nodes", []))
        stats["edge_count"] = len(data.get("links", []))
    except FileNotFoundError:
        stats["node_count"] = None
        stats["edge_count"] = None
    except Exception as e:
        stats["node_count"] = None
        stats["edge_count"] = None
        stats["graph_err"] = str(e)

    # ── Last build time ────────────────────────────────────────
    try:
        from ingestion.tiered_extractor import get_last_build_time
        stats["last_build"] = get_last_build_time()
    except Exception as e:
        stats["last_build"] = None
        stats["last_build_err"] = str(e)

    return stats


# ── PAGE ──────────────────────────────────────────────────────
st.title("🟢 Health")

stats = _load_health_stats()

not_built = stats["record_count"] is None and stats["node_count"] is None
if not_built:
    st.error("Index has not been built yet. Run `python main.py` first.")
    st.stop()

col1, col2, col3 = st.columns(3)

with col1:
    val = stats["record_count"]
    st.metric("Records Indexed", val if val is not None else "—")
    if "record_count_err" in stats:
        st.caption(f"Error: {stats['record_count_err']}")

with col2:
    val = stats["node_count"]
    label = f"{val:,}" if val is not None else "—"
    st.metric("Graph Nodes", label)
    if stats.get("edge_count") is not None:
        st.caption(f"{stats['edge_count']:,} edges")

with col3:
    build_time = stats["last_build"]
    st.metric("Last Build", build_time if build_time else "—")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

"""
ui/pages/health.py
───────────────────
Health dashboard — index stats + interactive 3D GraphRAG visualization.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Health", page_icon="🟢", layout="wide")

GRAPH_PATH = "data/graph.json"

NODE_COLORS = {
    "service":  "#4C9BE8",
    "engineer": "#52C788",
    "ticket":   "#F4A442",
    "commit":   "#A78BFA",
    "log":      "#F87171",
    "doc":      "#FBBF24",
}
NODE_SIZES = {
    "service":  12,
    "engineer": 10,
    "ticket":   8,
    "commit":   7,
    "log":      7,
    "doc":      8,
}
EDGE_COLORS = {
    "owns":      "#94a3b8", "owned_by":  "#94a3b8",
    "mentions":  "#64748b", "affects":   "#f59e0b",
    "fixes":     "#34d399", "depends_on":"#60a5fa",
    "caused_by": "#f87171", "modified":  "#c084fc",
}


@st.cache_data(ttl=60)
def _load_stats() -> dict:
    stats = {}
    try:
        from index.vector_store import get_stats
        stats["record_count"] = get_stats()["record_count"]
    except Exception:
        stats["record_count"] = None

    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        stats["node_count"] = len(data.get("nodes", []))
        stats["edge_count"] = len(data.get("links", []))
    except Exception:
        stats["node_count"] = None
        stats["edge_count"] = None

    try:
        from ingestion.tiered_extractor import get_last_build_time
        stats["last_build"] = get_last_build_time()
    except Exception:
        stats["last_build"] = None

    return stats


@st.cache_data(ttl=60)
def _load_graph_data() -> dict | None:
    try:
        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _render_graph_3d(graph_data: dict):
    nodes = graph_data.get("nodes", [])
    if not nodes:
        st.info("Graph has no nodes yet — run `python main.py` to build the index.")
        return

    G = nx.node_link_graph(graph_data, directed=True, multigraph=False)
    pos = nx.spring_layout(G, seed=42, k=1.8, dim=3)

    # ── Edge traces per relation ──────────────────────────────────
    relation_edges: dict[str, list] = {}
    for u, v, edata in G.edges(data=True):
        rel = edata.get("relation", "related")
        relation_edges.setdefault(rel, []).append((u, v))

    edge_traces = []
    for rel, pairs in relation_edges.items():
        ex, ey, ez = [], [], []
        for u, v in pairs:
            x0, y0, z0 = pos[u]
            x1, y1, z1 = pos[v]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
            ez += [z0, z1, None]

        color = EDGE_COLORS.get(rel, "#475569")
        edge_traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            line=dict(width=2, color=color),
            hoverinfo="none",
            name=rel,
            legendgroup=f"edge_{rel}",
            showlegend=True,
            opacity=0.6,
        ))

    # ── Node traces grouped by type ───────────────────────────────
    by_type: dict[str, list] = {}
    for nid in G.nodes():
        t = G.nodes[nid].get("type", "unknown")
        by_type.setdefault(t, []).append(nid)

    node_traces = []
    for ntype, nids in sorted(by_type.items()):
        x = [pos[n][0] for n in nids]
        y = [pos[n][1] for n in nids]
        z = [pos[n][2] for n in nids]

        labels, hovers = [], []
        for n in nids:
            nd = G.nodes[n]
            label = nd.get("name") or nd.get("title") or n
            label = label[:30] + "…" if len(label) > 30 else label
            labels.append(label)

            hover_lines = [
                f"<b>{nd.get('name') or nd.get('title') or n}</b>",
                f"Type: {ntype}",
                f"Connections: {G.degree(n)}",
            ]
            for key in ("source", "status", "owner"):
                if nd.get(key):
                    hover_lines.append(f"{key.capitalize()}: {nd[key]}")
            hovers.append("<br>".join(hover_lines))

        node_traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=8, color="#e2e8f0"),
            marker=dict(
                size=NODE_SIZES.get(ntype, 8),
                color=NODE_COLORS.get(ntype, "#aaa"),
                line=dict(width=1, color="#0f172a"),
                opacity=0.92,
            ),
            hovertext=hovers,
            hoverinfo="text",
            name=ntype,
            legendgroup=ntype,
        ))

    axis = dict(
        showbackground=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title="",
    )

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            height=650,
            showlegend=True,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=0),
            paper_bgcolor="#0e1117",
            legend=dict(
                bgcolor="#1e293b",
                bordercolor="#334155",
                borderwidth=1,
                font=dict(size=11, color="#e2e8f0"),
                title=dict(text="Node / Edge type", font=dict(size=11, color="#94a3b8")),
                itemsizing="constant",
            ),
            scene=dict(
                xaxis=axis, yaxis=axis, zaxis=axis,
                bgcolor="#0e1117",
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            ),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Node type colour key ──────────────────────────────────────
    cols = st.columns(len(NODE_COLORS))
    for col, (ntype, color) in zip(cols, NODE_COLORS.items()):
        count = len(by_type.get(ntype, []))
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:11px;height:11px;border-radius:50%;'
            f'background:{color};flex-shrink:0"></div>'
            f'<span style="font-size:12px;color:#cbd5e1">'
            f'{ntype} <b style="color:white">({count})</b></span></div>',
            unsafe_allow_html=True,
        )


# ── PAGE ──────────────────────────────────────────────────────────
st.title("🟢 Health")

stats = _load_stats()

if stats["record_count"] is None and stats["node_count"] is None:
    st.error("Index has not been built yet. Run `python main.py` first.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    val = stats["record_count"]
    st.metric("Records Indexed", val if val is not None else "—")
with col2:
    val = stats["node_count"]
    st.metric("Graph Nodes", f"{val:,}" if val is not None else "—",
              delta=f"{stats['edge_count']:,} edges" if stats.get("edge_count") is not None else None)
with col3:
    st.metric("Last Build", stats["last_build"] or "—")

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.divider()

st.subheader("Knowledge Graph — 3D")
st.caption("Rotate · Zoom · Hover nodes for details · Toggle types in legend")

graph_data = _load_graph_data()
if graph_data is None:
    st.error(f"Could not load graph from `{GRAPH_PATH}`.")
else:
    _render_graph_3d(graph_data)

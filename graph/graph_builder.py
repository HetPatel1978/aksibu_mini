"""
graph/graph_builder.py
───────────────────────
Builds and persists the knowledge graph using networkx (DiGraph).

Nodes = engineering entities (services, engineers, tickets, commits, logs, docs)
Edges = relationships (owns, depends_on, caused_by, affects, modified, fixes, mentions)

The graph is the core of what makes this system smarter than plain RAG.
Instead of just finding "similar text", we can traverse relationships:
  checkout-api → depends_on → payment-gateway → owned_by → Rahul Sharma

Graph is saved as JSON (node-link format) and reloaded for queries.

For production: swap networkx for Neo4j using the same interface.
"""

import json
import os
import networkx as nx

from ingestion.tiered_extractor import extract_all_tiered


def build_graph(normalized_records: list[dict]) -> nx.DiGraph:
    """
    Build a directed knowledge graph from normalized records.

    For each record:
      1. Add the record itself as a node
      2. Run tiered extraction to get entities + relationships
      3. Add extracted entity nodes + relationship edges
      4. Add direct edges: record → owner, record → services_mentioned
    """
    G = nx.DiGraph()

    # Run full tiered extraction across all records
    extraction_results = extract_all_tiered(normalized_records)

    print(f"\n🕸️  Building knowledge graph...")

    for record, extraction, tier in extraction_results:

        # ── ADD RECORD NODE ───────────────────────────────────
        G.add_node(
            record["id"],
            type=record["type"],
            title=record["title"],
            body=record["body"][:500],
            source=record["source"],
            timestamp=record.get("timestamp") or "",
            owner=record.get("owner") or "",
            priority=record.get("priority") or "",
            status=record.get("status") or "",
        )

        # ── ADD EXTRACTED ENTITY NODES ────────────────────────
        for entity in extraction.entities:
            if not G.has_node(entity.id):
                G.add_node(
                    entity.id,
                    type=entity.type,
                    name=entity.name,
                    confidence=entity.confidence,
                    extraction_source=entity.source,
                )

        # ── ADD EXTRACTED RELATIONSHIP EDGES ──────────────────
        for rel in extraction.relationships:
            if G.has_node(rel.from_id) and G.has_node(rel.to_id):
                G.add_edge(
                    rel.from_id,
                    rel.to_id,
                    relation=rel.relation,
                    confidence=rel.confidence,
                    extraction_source=rel.source,
                )

        # ── ADD OWNERSHIP EDGE ────────────────────────────────
        if record.get("owner"):
            owner_id = "engineer__" + record["owner"].lower().replace(" ", "_")
            if not G.has_node(owner_id):
                G.add_node(owner_id, type="engineer", name=record["owner"])
            G.add_edge(record["id"], owner_id, relation="owned_by", confidence=0.99)

        # ── ADD SERVICE MENTION EDGES ─────────────────────────
        for svc in record.get("services_mentioned", []):
            if not svc:
                continue
            svc_id = "service__" + svc.replace("-", "_")
            if not G.has_node(svc_id):
                G.add_node(svc_id, type="service", name=svc)
            G.add_edge(record["id"], svc_id, relation="mentions", confidence=0.95)

    node_types = {}
    for nid in G.nodes():
        t = G.nodes[nid].get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1

    print(f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    print(f"   Node types: {node_types}")

    return G


def save_graph(G: nx.DiGraph, path: str = "data/graph.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nx.node_link_data(G), f)
    print(f"\n✅ Graph saved → {path}")


def load_graph(path: str = "data/graph.json") -> nx.DiGraph:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Graph not found at {path}. Run `python main.py` first to build the index."
        )
    with open(path, "r", encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f), directed=True, multigraph=False)

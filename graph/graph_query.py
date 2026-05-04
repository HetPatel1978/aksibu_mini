"""
graph/graph_query.py
─────────────────────
All graph traversal logic. Given a node, we walk the graph
to collect related context that vector search alone would miss.

Key functions:
  get_connected_context()  — BFS N hops from a node, returns all connected data
  find_service_owner()     — Walk graph to find who owns a service
  find_recent_changes()    — Find commits that touched a service
  find_active_incidents()  — Find open P1/P2 tickets for a service
  service_dependencies()   — Find what a service depends on
"""

import networkx as nx
from graph.graph_builder import load_graph


def get_connected_context(G: nx.DiGraph, node_id: str, hops: int = 2) -> list[dict]:
    """
    BFS from node_id up to `hops` hops in any direction.
    Returns all connected nodes with their data + edge relationships.

    This is the GraphRAG core — one call assembles full context
    across all source types that are related to a given entity.
    """
    if node_id not in G:
        # Try fuzzy match — node_id might be a service name not exact graph ID
        candidates = [n for n in G.nodes() if node_id.replace("-", "_") in n]
        if not candidates:
            return []
        node_id = candidates[0]

    # ego_graph: all nodes within `radius` hops (treats graph as undirected for traversal)
    subgraph = nx.ego_graph(G, node_id, radius=hops, undirected=True)

    context = []
    for nid in subgraph.nodes():
        data = G.nodes[nid]

        # Collect outgoing edges for this node
        outgoing = []
        for neighbor in G.successors(nid):
            edge_data = G[nid][neighbor]
            outgoing.append({
                "to": neighbor,
                "relation": edge_data.get("relation", "unknown"),
                "confidence": edge_data.get("confidence", 1.0),
            })

        # Collect incoming edges
        incoming = []
        for neighbor in G.predecessors(nid):
            edge_data = G[neighbor][nid]
            incoming.append({
                "from": neighbor,
                "relation": edge_data.get("relation", "unknown"),
            })

        context.append({
            "id": nid,
            "type": data.get("type", "unknown"),
            "title": data.get("title") or data.get("name") or nid,
            "body": data.get("body", ""),
            "owner": data.get("owner", ""),
            "timestamp": data.get("timestamp", ""),
            "source": data.get("source", ""),
            "priority": data.get("priority", ""),
            "status": data.get("status", ""),
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
        })

    return context


def find_service_node(G: nx.DiGraph, service_name: str) -> str | None:
    """Find the graph node ID for a given service name."""
    # Try exact service__ prefixed ID first
    candidates = [
        f"service__{service_name.replace('-', '_')}",
        f"service__{service_name}",
    ]
    for c in candidates:
        if G.has_node(c):
            return c

    # Fuzzy fallback — partial match on node ID or name attribute
    name_lower = service_name.lower().replace("-", "_")
    for nid in G.nodes():
        node = G.nodes[nid]
        if node.get("type") == "service":
            if name_lower in nid.lower() or name_lower in node.get("name", "").lower():
                return nid

    return None


def find_service_owner(G: nx.DiGraph, service_name: str) -> dict:
    """
    Walk the graph to find who owns a service and escalation path.
    Returns primary owner + escalation contact if available.
    """
    result = {"primary": None, "escalation": None, "source": None}

    # First: look for records that mention this service and have an owner
    svc_node = find_service_node(G, service_name)

    owners_found = []
    for nid in G.nodes():
        node = G.nodes[nid]
        owner = node.get("owner")
        if not owner:
            continue

        # Check if this record is connected to the service
        svc_mentioned = service_name.lower().replace("-", "_") in nid.lower()
        if not svc_mentioned and svc_node:
            # Check edges
            svc_mentioned = G.has_edge(nid, svc_node) or G.has_edge(svc_node, nid)

        if svc_mentioned:
            owners_found.append({
                "owner": owner,
                "source": node.get("source", ""),
                "type": node.get("type", ""),
            })

    if owners_found:
        # Prefer confluence doc ownership (most authoritative)
        conf_owners = [o for o in owners_found if o["source"] == "confluence"]
        primary = conf_owners[0] if conf_owners else owners_found[0]
        result["primary"] = primary["owner"]
        result["source"] = primary["source"]

        # Look for on-call/escalation in the doc body
        for nid in G.nodes():
            node = G.nodes[nid]
            if node.get("source") == "confluence" and node.get("owner") == primary["owner"]:
                body = node.get("body", "")
                for line in body.split("\n"):
                    if "on-call:" in line.lower() or "escalation" in line.lower():
                        parts = line.split(":", 1)
                        if len(parts) > 1:
                            result["escalation"] = parts[1].strip()

    return result


def find_recent_changes(G: nx.DiGraph, service_name: str, limit: int = 5) -> list[dict]:
    """Find commits that touched a given service, sorted by recency."""
    changes = []
    name_lower = service_name.lower().replace("-", "_")

    for nid in G.nodes():
        node = G.nodes[nid]
        if node.get("type") != "commit":
            continue

        # Check if commit is connected to this service
        connected = False
        for neighbor in list(G.successors(nid)) + list(G.predecessors(nid)):
            if name_lower in neighbor.lower():
                connected = True
                break
        if not connected:
            # Also check body text
            if name_lower in node.get("body", "").lower() or name_lower in node.get("title", "").lower():
                connected = True

        if connected:
            changes.append({
                "id": nid,
                "title": node.get("title", ""),
                "timestamp": node.get("timestamp", ""),
                "owner": node.get("owner", ""),
                "body": node.get("body", ""),
            })

    changes.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    return changes[:limit]


def find_active_incidents(G: nx.DiGraph, service_name: str) -> list[dict]:
    """Find open/active tickets related to a service."""
    incidents = []
    name_lower = service_name.lower().replace("-", "_")

    for nid in G.nodes():
        node = G.nodes[nid]
        if node.get("type") != "ticket":
            continue
        if node.get("status") in ("resolved", "closed", "done"):
            continue

        body = (node.get("body") or "").lower()
        title = (node.get("title") or "").lower()
        if name_lower in body or name_lower in title or name_lower in nid.lower():
            incidents.append({
                "id": nid,
                "title": node.get("title", ""),
                "priority": node.get("priority", ""),
                "status": node.get("status", ""),
                "owner": node.get("owner", ""),
                "timestamp": node.get("timestamp", ""),
            })

    incidents.sort(key=lambda x: x.get("priority") or "P9")
    return incidents


def service_dependencies(G: nx.DiGraph, service_name: str) -> list[str]:
    """Find what services a given service depends on."""
    svc_node = find_service_node(G, service_name)
    if not svc_node:
        return []

    deps = []
    for nid in G.nodes():
        node = G.nodes[nid]
        if node.get("type") == "doc":
            body = (node.get("body") or "").lower()
            if service_name.lower().replace("-", "_") in nid.lower():
                # Parse Dependencies section from doc
                in_deps = False
                for line in body.split("\n"):
                    if "dependencies" in line.lower():
                        in_deps = True
                    elif in_deps and line.strip().startswith("-"):
                        dep = line.strip().lstrip("- ").split("(")[0].strip()
                        if dep:
                            deps.append(dep)
                    elif in_deps and line.startswith("#"):
                        in_deps = False

    return list(set(deps))

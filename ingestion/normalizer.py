"""
ingestion/normalizer.py
───────────────────────
Converts raw records from all 4 sources (Confluence, Jira, GitHub, Logs)
into a single unified schema before they touch the graph or vector DB.

Also handles:
  - Entity resolution (same service, different names across systems)
  - Owner resolution (email vs full name vs abbreviation)

This is the "lingua franca" layer — everything downstream speaks one language.
"""

import json
import os
import re


# ── CANONICAL ALIAS MAPS ──────────────────────────────────────
# Same entity, different names across systems → one canonical form

SERVICE_ALIASES = {
    "checkout-api": "checkout-api",
    "checkout service": "checkout-api",
    "checkout_api": "checkout-api",
    "checkout": "checkout-api",
    "payment-gateway": "payment-gateway",
    "payment gateway": "payment-gateway",
    "payment_gateway": "payment-gateway",
    "payments": "payment-gateway",
    "pgw": "payment-gateway",
    "inventory-service": "inventory-service",
    "inventory service": "inventory-service",
    "inventory_service": "inventory-service",
    "inventory": "inventory-service",
    "fraud-detection": "fraud-detection",
    "fraud detection": "fraud-detection",
    "fraud_detection": "fraud-detection",
    "redis-cache": "redis-cache",
    "redis cache": "redis-cache",
    "redis": "redis-cache",
    "ml-feature-store": "ml-feature-store",
    "ml feature store": "ml-feature-store",
    "feature-store": "ml-feature-store",
    "postgres-primary": "postgres-primary",
    "postgres": "postgres-primary",
}

OWNER_ALIASES = {
    "rahul@company.com": "Rahul Sharma",
    "rahul sharma": "Rahul Sharma",
    "rahul s.": "Rahul Sharma",
    "rahul": "Rahul Sharma",
    "priya@company.com": "Priya Mehta",
    "priya mehta": "Priya Mehta",
    "priya": "Priya Mehta",
    "james@company.com": "James Liu",
    "james liu": "James Liu",
    "james": "James Liu",
}


def resolve_service(name: str) -> str | None:
    """Map any service alias to its canonical name."""
    return SERVICE_ALIASES.get(name.lower().strip())


def resolve_owner(raw: str | None) -> str | None:
    """Map any owner representation to canonical full name."""
    if not raw:
        return None
    return OWNER_ALIASES.get(raw.lower().strip(), raw)


def extract_services_from_text(text: str) -> list[str]:
    """Find all canonical service names mentioned in a text block."""
    found = []
    text_lower = text.lower()
    for alias, canonical in SERVICE_ALIASES.items():
        if alias in text_lower and canonical not in found:
            found.append(canonical)
    return found


def extract_owner_from_body(text: str) -> str | None:
    """Pull owner from Confluence-style 'Owner: ...' lines."""
    for line in text.split("\n"):
        if "owner:" in line.lower():
            raw = line.split(":", 1)[1].strip()
            if "(" in raw and ")" in raw:
                email = raw[raw.find("(") + 1 : raw.find(")")]
                return email
            return raw
    return None


# ── PER-SOURCE NORMALIZERS ────────────────────────────────────

def normalize_confluence(doc: dict) -> dict:
    owner_raw = extract_owner_from_body(doc["body"])
    return {
        "id": doc["id"],
        "type": "doc",
        "title": doc["title"],
        "body": doc["body"].strip(),
        "owner": resolve_owner(owner_raw),
        "services_mentioned": extract_services_from_text(doc["body"]),
        "timestamp": doc["last_updated"],
        "source": "confluence",
        "priority": None,
        "status": None,
        "raw": doc,
    }


def normalize_jira(ticket: dict) -> dict:
    return {
        "id": ticket["id"],
        "type": "ticket",
        "title": ticket["summary"],
        "body": ticket["description"].strip(),
        "owner": resolve_owner(ticket.get("assignee")),
        "services_mentioned": [
            resolve_service(s) or s
            for s in ticket.get("services_affected", [])
        ],
        "timestamp": ticket["created"],
        "source": "jira",
        "priority": ticket.get("priority"),
        "status": ticket.get("status"),
        "raw": ticket,
    }


def normalize_github(commit: dict) -> dict:
    files_str = ", ".join(commit["files_changed"])
    return {
        "id": commit["sha"],
        "type": "commit",
        "title": commit["message"],
        "body": (
            f"Author: {commit['author']}\n"
            f"Repo: {commit['repo']}\n"
            f"Files changed: {files_str}\n"
            f"PR: {commit.get('pr', 'N/A')}"
        ),
        "owner": resolve_owner(commit["author"]),
        "services_mentioned": [resolve_service(commit["repo"]) or commit["repo"]],
        "timestamp": commit["timestamp"],
        "source": "github",
        "priority": None,
        "status": None,
        "raw": commit,
    }


def normalize_log(log: dict) -> dict:
    return {
        "id": log["id"],
        "type": "log",
        "title": f"[{log['level']}] {log['service']}: {log['message'][:80]}",
        "body": log["message"],
        "owner": None,
        "services_mentioned": [resolve_service(log["service"]) or log["service"]],
        "timestamp": log["timestamp"],
        "source": "logs",
        "priority": "P1" if log["level"] == "ERROR" else "P2",
        "status": "active",
        "raw": log,
    }


# ── MAIN ENTRY POINT ──────────────────────────────────────────

def normalize_all() -> list[dict]:
    os.makedirs("data/normalized", exist_ok=True)
    normalized = []

    loaders = [
        ("data/raw/confluence.json", normalize_confluence),
        ("data/raw/jira.json", normalize_jira),
        ("data/raw/github.json", normalize_github),
        ("data/raw/logs.json", normalize_log),
    ]

    for path, fn in loaders:
        if not os.path.exists(path):
            print(f"  ⚠️  {path} not found — skipping")
            continue
        with open(path) as f:
            records = json.load(f)
        for r in records:
            normalized.append(fn(r))
        print(f"  ✅ {path}: {len(records)} records normalized")

    # Strip raw field before saving (keeps file size down)
    saveable = [{k: v for k, v in r.items() if k != "raw"} for r in normalized]
    with open("data/normalized/all.json", "w") as f:
        json.dump(saveable, f, indent=2)

    print(f"\n  📦 Total: {len(normalized)} records → data/normalized/all.json")
    return normalized


if __name__ == "__main__":
    normalize_all()

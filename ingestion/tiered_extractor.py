"""
ingestion/tiered_extractor.py
──────────────────────────────
Tiered extraction: cache → rules → cheap LLM → frontier LLM
Uses OpenRouter API (OpenAI-compatible).
"""

import json, hashlib, sqlite3, os, re
from contextlib import closing
from pathlib import Path
from typing import Optional

# ── CRITICAL: Load .env BEFORE OpenAI client init ─────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=str(_env_path))

from openai import OpenAI
from ingestion.rule_extractor import (
    ExtractionResult, Entity, Relationship, extract_with_rules
)

_api_key = os.getenv("OPENROUTER_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "\n\nOPENROUTER_API_KEY not found in environment.\n"
        "Make sure you have a .env file in the aksibu-mini/ folder:\n"
        "  OPENROUTER_API_KEY=sk-or-v1-...\n"
        "Get your key from: https://openrouter.ai/keys\n"
    )

client = OpenAI(
    api_key=_api_key,
    base_url="https://openrouter.ai/api/v1",
)

CHEAP_MODEL    = os.getenv("CHEAP_MODEL",    "anthropic/claude-haiku-4-5")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "anthropic/claude-sonnet-4-5")


# ── ROBUST JSON PARSER ────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {"entities": [], "relationships": []}


# ── EXTRACTION CACHE ──────────────────────────────────────────
def _cache_db() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/extraction_cache.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            text_hash   TEXT PRIMARY KEY,
            result_json TEXT NOT NULL,
            tier_used   TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def cache_get(text: str) -> Optional[dict]:
    with closing(_cache_db()) as conn:
        row = conn.execute(
            "SELECT result_json FROM cache WHERE text_hash = ?",
            (_text_hash(text),)
        ).fetchone()
    return json.loads(row[0]) if row else None

def cache_set(text: str, result: dict, tier: str):
    with closing(_cache_db()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, datetime('now'))",
            (_text_hash(text), json.dumps(result), tier)
        )
        conn.commit()


# ── INCREMENTAL INDEX MANIFEST ────────────────────────────────
def _manifest_db() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/index_manifest.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manifest (
            record_id    TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            source       TEXT,
            tier_used    TEXT,
            indexed_at   TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn

def _content_hash(record: dict) -> str:
    return hashlib.md5(f"{record.get('title','')}{record.get('body','')}".encode()).hexdigest()

def filter_changed_records(records: list) -> tuple:
    to_process, stats = [], {"total": len(records), "new": 0, "changed": 0, "unchanged": 0}
    with closing(_manifest_db()) as conn:
        for r in records:
            h = _content_hash(r)
            row = conn.execute(
                "SELECT content_hash FROM manifest WHERE record_id = ?", (r["id"],)
            ).fetchone()
            if row is None:
                stats["new"] += 1; to_process.append(r)
            elif row[0] != h:
                stats["changed"] += 1; to_process.append(r)
            else:
                stats["unchanged"] += 1
    return to_process, stats

def mark_indexed(record: dict, tier: str):
    with closing(_manifest_db()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO manifest VALUES (?, ?, ?, ?, datetime('now'))",
            (record["id"], _content_hash(record), record.get("source"), tier)
        )
        conn.commit()

def get_last_build_time() -> str | None:
    """Return the most recent indexed_at timestamp from the manifest, or None if unbuilt."""
    if not os.path.exists("data/index_manifest.db"):
        return None
    with closing(_manifest_db()) as conn:
        row = conn.execute("SELECT MAX(indexed_at) FROM manifest").fetchone()
    return row[0] if row else None


# ── DEDUPLICATION ─────────────────────────────────────────────
def deduplicate_records(records: list) -> list:
    if len(records) < 2:
        return records
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{r['title']} {r['body'][:200]}" for r in records]
        embeddings = model.encode(texts, show_progress_bar=False)
        threshold = 0.95
        kept, kept_embs = [], []
        for i, emb in enumerate(embeddings):
            dup = any(
                float(np.dot(emb, ke) / (np.linalg.norm(emb) * np.linalg.norm(ke) + 1e-9)) >= threshold
                for ke in kept_embs
            )
            if not dup:
                kept.append(records[i])
                kept_embs.append(emb)
        skipped = len(records) - len(kept)
        if skipped:
            print(f"  🗜  Dedup: {len(records)} → {len(kept)} ({skipped} removed)")
        return kept
    except ImportError:
        return records


# ── COMPLEXITY HEURISTICS ─────────────────────────────────────
def _needs_frontier(rule_result: ExtractionResult, record: dict) -> bool:
    body_len = len(record.get("body", ""))
    if body_len > 800: return True
    if len(rule_result.entities) < 2 and body_len > 300: return True
    if record.get("source") == "confluence": return True
    if len(rule_result.relationships) == 0 and len(rule_result.entities) > 3: return True
    return False


# ── LLM EXTRACTION ────────────────────────────────────────────
CHEAP_PROMPT = """Extract engineering entities and relationships from this text.
Return ONLY valid JSON, no markdown:
{{"entities": [{{"id": "snake_case_id", "type": "service|engineer|ticket|commit", "name": "display name"}}],
  "relationships": [{{"from_id": "id", "relation": "owns|affects|fixes|depends_on|caused_by|modified", "to_id": "id"}}]}}

Text: {text}"""

FRONTIER_PROMPT = """You are an expert engineering knowledge graph extractor.
Extract ALL entities and relationships. Include implicit ones.
Return ONLY valid JSON, no markdown fences, no explanation:
{{"entities": [{{"id": "snake_case", "type": "service|engineer|ticket|commit|config|log", "name": "..."}}],
  "relationships": [{{"from_id": "id", "relation": "owns|affects|fixes|depends_on|caused_by|modified|references", "to_id": "id"}}]}}

Source: {source}
Title: {title}
Body: {body}"""

def _llm_call(prompt: str, model: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_json(resp.choices[0].message.content)
    except Exception as ex:
        print(f"    ⚠️  LLM call failed ({model}): {ex}")
        return {"entities": [], "relationships": []}

def _merge(base: ExtractionResult, d: dict, source: str) -> ExtractionResult:
    existing_ids = {e.id for e in base.entities}
    for e in d.get("entities", []):
        if e.get("id") not in existing_ids:
            base.entities.append(Entity(
                id=e["id"], type=e.get("type", "unknown"),
                name=e.get("name", e["id"]), confidence=0.85, source=source
            ))
    for r in d.get("relationships", []):
        base.relationships.append(Relationship(
            from_id=r.get("from_id", r.get("from", "")),
            relation=r.get("relation", ""),
            to_id=r.get("to_id", r.get("to", "")),
            confidence=0.80, source=source
        ))
    return base

def _result_to_dict(r: ExtractionResult) -> dict:
    return {
        "entities":      [vars(e) for e in r.entities],
        "relationships": [vars(rel) for rel in r.relationships],
    }

def _dict_to_result(d: dict) -> ExtractionResult:
    r = ExtractionResult()
    for e in d.get("entities", []): r.entities.append(Entity(**e))
    for rel in d.get("relationships", []): r.relationships.append(Relationship(**rel))
    return r


# ── MAIN TIERED EXTRACTION ────────────────────────────────────
def extract_tiered(record: dict, verbose: bool = True) -> tuple:
    text = f"{record.get('title', '')} {record.get('body', '')}"
    prefix = f"  [{record['id']}]"

    cached = cache_get(text)
    if cached:
        if verbose: print(f"{prefix} 💾 cache hit")
        return _dict_to_result(cached), "cache"

    result = extract_with_rules(record)
    tier = "rule"

    if result.unresolved_text:
        if verbose: print(f"{prefix} 🤏 cheap LLM ({CHEAP_MODEL})")
        try:
            out = _llm_call(CHEAP_PROMPT.format(text=result.unresolved_text[:600]), CHEAP_MODEL)
            result = _merge(result, out, "llm_cheap")
            tier = "llm_cheap"
        except Exception as ex:
            print(f"{prefix} ⚠️  cheap LLM failed: {ex}")

    if _needs_frontier(result, record):
        if verbose: print(f"{prefix} 🧠 frontier LLM ({FRONTIER_MODEL})")
        try:
            out = _llm_call(FRONTIER_PROMPT.format(
                source=record.get("source", ""),
                title=record.get("title", "")[:200],
                body=record.get("body", "")[:800],
            ), FRONTIER_MODEL)
            result = _merge(result, out, "llm_frontier")
            tier = "llm_frontier"
        except Exception as ex:
            print(f"{prefix} ⚠️  frontier LLM failed: {ex}")

    cache_set(text, _result_to_dict(result), tier)
    if verbose: print(f"{prefix} ✅ tier={tier} entities={len(result.entities)}")
    return result, tier


def extract_all_tiered(records: list) -> list:
    print("\n📋 Incremental change detection...")
    to_process, stats = filter_changed_records(records)
    print(f"   New: {stats['new']}  Changed: {stats['changed']}  Unchanged: {stats['unchanged']} (skipped)")

    print("\n🗜  Deduplicating records...")
    to_process = deduplicate_records(to_process)

    print(f"\n⚙️  Extracting from {len(to_process)} records...\n")
    results = []
    tier_counts = {"cache": 0, "rule": 0, "llm_cheap": 0, "llm_frontier": 0}
    COSTS = {"cache": 0.0, "rule": 0.0, "llm_cheap": 0.0002, "llm_frontier": 0.003}

    for record in to_process:
        extraction, tier = extract_tiered(record)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        results.append((record, extraction, tier))
        mark_indexed(record, tier)

    total = sum(tier_counts.get(t, 0) * c for t, c in COSTS.items())
    naive = len(to_process) * COSTS["llm_frontier"]
    print(f"\n💰 Cost: ${total:.4f} (naive would be ${naive:.4f}, saved {100*(naive-total)/max(naive,0.001):.0f}%)\n")
    return results
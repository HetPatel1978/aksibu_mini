"""
ingestion/rule_extractor.py
────────────────────────────
Layer 1 of the tiered extraction pipeline — completely FREE.

Uses regex patterns to extract ~70% of all engineering entities
without any LLM calls:
  - Jira ticket IDs (JR-441)
  - Commit SHAs (c7d3e91)
  - PR numbers (PR-221)
  - Engineer emails (rahul@company.com)
  - Service names (checkout-api, payment-gateway)
  - File paths (src/config.py)
  - HTTP error codes (503, 500)
  - Timeout values (300ms)

Also runs lightweight relationship extraction from trigger phrases
like "caused by", "depends on", "owner:", "fixes JR-xxx".

Returns structured ExtractionResult + flags any text it couldn't
resolve (passed to LLM tiers for further processing).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── DATA STRUCTURES ───────────────────────────────────────────

@dataclass
class Entity:
    id: str
    type: str
    name: str
    confidence: float   # 0.0–1.0
    source: str         # "rule" | "llm_cheap" | "llm_frontier"


@dataclass
class Relationship:
    from_id: str
    relation: str
    to_id: str
    confidence: float
    source: str


@dataclass
class ExtractionResult:
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    unresolved_text: Optional[str] = None  # text rules couldn't handle → sent to LLM


# ── REGEX PATTERNS ────────────────────────────────────────────

ENTITY_PATTERNS = {
    "jira_ticket": re.compile(r'\b([A-Z]{2,6}-\d+)\b'),
    "commit_sha":  re.compile(r'\b([a-f0-9]{7,40})\b'),
    "pr_number":   re.compile(r'\b(?:PR|pr)[- #]*(\d+)\b'),
    "email":       re.compile(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'),
    "service":     re.compile(
        r'\b([a-z][a-z0-9]*(?:-[a-z0-9]+)*'
        r'(?:-service|-api|-gateway|-worker|-cache|-store|-db))\b'
    ),
    "file_path":   re.compile(r'\b([\w./]+\.(?:py|js|ts|go|java|yaml|yml|json|cfg))\b'),
    "error_code":  re.compile(r'\b(5\d{2}|4\d{2})\b'),
    "timeout_ms":  re.compile(r'(\d+)\s*ms\b'),
}

RELATION_PATTERNS = [
    (re.compile(r'caused\s+by\s+([\w-]+)', re.I),              "caused_by",  "right"),
    (re.compile(r'([\w-]+)\s+depends[_\s]on\s+([\w-]+)', re.I),"depends_on", "both"),
    (re.compile(r'owner[:\s]+([\w@.]+)', re.I),                 "owned_by",   "right"),
    (re.compile(r'(?:fix|fixed|fixes)\s+([A-Z]{2,6}-\d+)', re.I),"fixes",    "right"),
    (re.compile(r'modified\s+([\w./]+)', re.I),                 "modified",   "right"),
    (re.compile(r'timeout.*?(?:in|from|of)\s+([\w-]+)', re.I), "affects",    "right"),
    (re.compile(r'([\w-]+)\s+unresponsive', re.I),              "affects",    "left"),
]

# Capitalized multi-word phrases likely to be names/proper nouns
UNRESOLVED_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


def _normalize_id(entity_type: str, value: str) -> str:
    """Convert raw match value to a clean snake_case ID."""
    value = value.lower().strip()
    value = re.sub(r'[^a-z0-9_]', '_', value)
    return f"{entity_type}__{value}"


def extract_with_rules(record: dict) -> ExtractionResult:
    """
    Free, fast entity + relationship extraction via regex.
    Flags unresolved text for LLM tiers.
    """
    text = f"{record.get('title', '')} {record.get('body', '')}"
    result = ExtractionResult()
    matched_spans: set[tuple] = set()

    # ── ENTITY EXTRACTION ─────────────────────────────────────
    for etype, pattern in ENTITY_PATTERNS.items():
        for match in pattern.finditer(text):
            span = match.span()
            if span in matched_spans:
                continue
            matched_spans.add(span)

            value = match.group(1)
            entity_id = _normalize_id(etype, value)

            result.entities.append(Entity(
                id=entity_id,
                type=etype,
                name=value,
                confidence=0.95,
                source="rule"
            ))

    # ── RELATIONSHIP EXTRACTION ───────────────────────────────
    existing_ids = {e.id for e in result.entities}

    for pattern, relation, direction in RELATION_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()

            if direction == "both" and len(groups) == 2:
                result.relationships.append(Relationship(
                    from_id=_normalize_id("service", groups[0]),
                    relation=relation,
                    to_id=_normalize_id("service", groups[1]),
                    confidence=0.80,
                    source="rule"
                ))
            elif direction == "right" and len(groups) >= 1:
                result.relationships.append(Relationship(
                    from_id=record["id"],
                    relation=relation,
                    to_id=_normalize_id("unknown", groups[-1]),
                    confidence=0.75,
                    source="rule"
                ))
            elif direction == "left" and len(groups) >= 1:
                result.relationships.append(Relationship(
                    from_id=_normalize_id("service", groups[0]),
                    relation=relation,
                    to_id=record["id"],
                    confidence=0.75,
                    source="rule"
                ))

    # ── UNRESOLVED TEXT ───────────────────────────────────────
    # Find capitalized phrases that weren't caught by any pattern
    unresolved = []
    for match in UNRESOLVED_PATTERN.finditer(text):
        if match.span() not in matched_spans:
            phrase = match.group()
            # Skip common English phrases unlikely to be entities
            if phrase not in {"The", "This", "That", "These", "Those",
                               "What", "When", "Where", "Which", "How"}:
                unresolved.append(phrase)

    if unresolved:
        result.unresolved_text = ", ".join(sorted(set(unresolved)))

    return result

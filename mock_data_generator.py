"""
mock_data_generator.py
─────────────────────
Generates realistic fake engineering data simulating:
  - Confluence docs (architecture, runbooks, ownership)
  - Jira tickets (incidents, bugs, tasks)
  - GitHub commits (code changes with metadata)
  - Log lines (errors, warnings from services)

Run this ONCE before building the index.
Output: data/raw/*.json
"""

import json
import os

os.makedirs("data/raw", exist_ok=True)

# ── CONFLUENCE DOCS ───────────────────────────────────────────
confluence_docs = [
    {
        "id": "CONF-001",
        "title": "Checkout Service — Architecture Overview",
        "body": """
## Checkout Service
Owner: Rahul Sharma (rahul@company.com)
Team: Payments

### What it does
Handles end-to-end checkout flow including cart validation,
payment processing, and order confirmation.

### Dependencies
- payment-gateway (critical path)
- inventory-service (stock validation)
- fraud-detection (pre-auth checks)

### Known Issues
- Timeout sensitivity: if payment-gateway response exceeds 500ms,
  checkout-api will fail with a 503.
- High traffic periods require manual scaling of payment-gateway pods.

### SLAs
- P99 latency: < 800ms
- Uptime: 99.9%
        """,
        "source": "confluence",
        "last_updated": "2026-01-15"
    },
    {
        "id": "CONF-002",
        "title": "Payment Gateway — Runbook",
        "body": """
## Payment Gateway Service
Owner: Rahul Sharma (rahul@company.com)
On-call: Priya Mehta (priya@company.com)

### What it does
Interfaces with Stripe and internal fraud systems to authorize
and capture payments.

### Common Failure Modes
1. Stripe API rate limiting — back off and retry with exponential delay
2. Timeout spikes — usually caused by upstream DB connection pool exhaustion
3. Config drift — timeout values must match checkout-api expectations

### Escalation Path
Page Rahul first. If unresponsive after 10min, escalate to Priya.

### Recent Changes
Timeout config changed in PR-221 (April 12, 2026) — reduced from 800ms to 300ms.
        """,
        "source": "confluence",
        "last_updated": "2025-11-20"
    },
    {
        "id": "CONF-003",
        "title": "Inventory Service — Overview",
        "body": """
## Inventory Service
Owner: James Liu (james@company.com)

### What it does
Tracks real-time stock levels and reserves items during checkout.

### Dependencies
- postgres-primary (read/write)
- redis-cache (stock level cache, TTL 60s)

### Notes
Cache invalidation bugs have caused stale stock reads in the past.
Fixed in commit b9f2a44.
        """,
        "source": "confluence",
        "last_updated": "2025-09-10"
    },
    {
        "id": "CONF-004",
        "title": "Fraud Detection Service — Overview",
        "body": """
## Fraud Detection Service
Owner: Priya Mehta (priya@company.com)

### What it does
Runs ML-based fraud scoring on every transaction before auth.
Adds ~50-80ms to checkout latency.

### Dependencies
- checkout-api (caller)
- ml-feature-store (feature retrieval)
- postgres-fraud (fraud history DB)

### Known Bottlenecks
Feature store latency spikes under high load.
        """,
        "source": "confluence",
        "last_updated": "2026-02-01"
    }
]

# ── JIRA TICKETS ──────────────────────────────────────────────
jira_tickets = [
    {
        "id": "JR-441",
        "summary": "Checkout service returning 503 errors intermittently",
        "description": """
Users reporting checkout failures since ~2am UTC April 14.
Error logs show checkout-api timing out waiting on payment-gateway.
Frequency: ~15% of requests failing.
Suspected cause: payment-gateway slowdown after last night's deploy.
Related to JR-438 — timeout config change.
        """,
        "assignee": "rahul@company.com",
        "reporter": "priya@company.com",
        "status": "open",
        "priority": "P1",
        "services_affected": ["checkout-api", "payment-gateway"],
        "created": "2026-04-14T03:45:00Z",
        "source": "jira"
    },
    {
        "id": "JR-438",
        "summary": "payment-gateway timeout config needs review",
        "description": """
After the April 12 deploy (commit c7d3e91), timeout values in
payment-gateway/config.py were changed from 800ms to 300ms.
This may be too aggressive for peak traffic.
Checkout-api expects up to 500ms. Need alignment between teams.
        """,
        "assignee": "rahul@company.com",
        "reporter": "james@company.com",
        "status": "open",
        "priority": "P2",
        "services_affected": ["payment-gateway", "checkout-api"],
        "created": "2026-04-12T14:00:00Z",
        "source": "jira"
    },
    {
        "id": "JR-420",
        "summary": "Inventory service cache stale reads during high traffic",
        "description": """
During the April 10 flash sale, inventory-service was returning
stale stock levels due to redis cache not invalidating correctly.
Items showing in-stock were actually sold out.
Fixed in commit b9f2a44 by James Liu.
        """,
        "assignee": "james@company.com",
        "reporter": "rahul@company.com",
        "status": "resolved",
        "priority": "P2",
        "services_affected": ["inventory-service", "redis-cache"],
        "created": "2026-04-10T10:00:00Z",
        "source": "jira"
    },
    {
        "id": "JR-415",
        "summary": "fraud-detection latency spike affecting checkout P99",
        "description": """
Fraud detection service showing P99 latency of 250ms vs normal 80ms.
Likely caused by ml-feature-store degradation.
Checkout P99 has increased by ~170ms as a result.
Priya investigating.
        """,
        "assignee": "priya@company.com",
        "reporter": "james@company.com",
        "status": "resolved",
        "priority": "P3",
        "services_affected": ["fraud-detection", "checkout-api", "ml-feature-store"],
        "created": "2026-04-08T09:00:00Z",
        "source": "jira"
    }
]

# ── GITHUB COMMITS ────────────────────────────────────────────
github_commits = [
    {
        "sha": "c7d3e91",
        "message": "chore: reduce payment-gateway timeout from 800ms to 300ms for performance",
        "author": "rahul@company.com",
        "repo": "payment-gateway",
        "files_changed": ["src/config.py", "src/stripe_client.py"],
        "timestamp": "2026-04-12T11:30:00Z",
        "pr": "PR-221",
        "source": "github"
    },
    {
        "sha": "b9f2a44",
        "message": "fix: correct redis cache invalidation on stock update in inventory-service",
        "author": "james@company.com",
        "repo": "inventory-service",
        "files_changed": ["src/cache_manager.py"],
        "timestamp": "2026-04-10T15:00:00Z",
        "pr": "PR-218",
        "source": "github"
    },
    {
        "sha": "f1a8c02",
        "message": "feat: add retry logic for stripe rate limiting in payment-gateway",
        "author": "priya@company.com",
        "repo": "payment-gateway",
        "files_changed": ["src/stripe_client.py", "tests/test_stripe.py"],
        "timestamp": "2026-04-08T09:00:00Z",
        "pr": "PR-215",
        "source": "github"
    },
    {
        "sha": "a1b2c3d",
        "message": "fix: increase checkout-api upstream timeout to 600ms",
        "author": "rahul@company.com",
        "repo": "checkout-api",
        "files_changed": ["src/gateway_client.py", "config/timeouts.yaml"],
        "timestamp": "2026-04-14T08:00:00Z",
        "pr": "PR-224",
        "source": "github"
    }
]

# ── LOG LINES ─────────────────────────────────────────────────
log_lines = [
    {
        "id": "LOG-001",
        "level": "ERROR",
        "service": "checkout-api",
        "message": "Connection timeout waiting for payment-gateway response (elapsed: 312ms, limit: 300ms)",
        "timestamp": "2026-04-14T02:03:11Z",
        "source": "logs"
    },
    {
        "id": "LOG-002",
        "level": "ERROR",
        "service": "checkout-api",
        "message": "Upstream payment-gateway returned 503 Service Unavailable",
        "timestamp": "2026-04-14T02:03:45Z",
        "source": "logs"
    },
    {
        "id": "LOG-003",
        "level": "WARN",
        "service": "payment-gateway",
        "message": "DB connection pool nearing exhaustion (87/100 connections used)",
        "timestamp": "2026-04-14T01:58:22Z",
        "source": "logs"
    },
    {
        "id": "LOG-004",
        "level": "ERROR",
        "service": "payment-gateway",
        "message": "Stripe API call exceeded internal timeout threshold of 300ms",
        "timestamp": "2026-04-14T02:01:05Z",
        "source": "logs"
    },
    {
        "id": "LOG-005",
        "level": "ERROR",
        "service": "checkout-api",
        "message": "Order confirmation failed — downstream payment-gateway unresponsive",
        "timestamp": "2026-04-14T02:05:00Z",
        "source": "logs"
    },
    {
        "id": "LOG-006",
        "level": "WARN",
        "service": "fraud-detection",
        "message": "ml-feature-store response latency elevated: 210ms (threshold: 150ms)",
        "timestamp": "2026-04-14T01:55:00Z",
        "source": "logs"
    }
]

# ── SAVE ALL ──────────────────────────────────────────────────
with open("data/raw/confluence.json", "w") as f:
    json.dump(confluence_docs, f, indent=2)

with open("data/raw/jira.json", "w") as f:
    json.dump(jira_tickets, f, indent=2)

with open("data/raw/github.json", "w") as f:
    json.dump(github_commits, f, indent=2)

with open("data/raw/logs.json", "w") as f:
    json.dump(log_lines, f, indent=2)

print("✅ Mock data generated in data/raw/")
print(f"   confluence.json  — {len(confluence_docs)} docs")
print(f"   jira.json        — {len(jira_tickets)} tickets")
print(f"   github.json      — {len(github_commits)} commits")
print(f"   logs.json        — {len(log_lines)} log lines")

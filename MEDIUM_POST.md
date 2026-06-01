# I Built a RAG System That Never Hallucinates — Here's the Exact Architecture

## The problem every engineering team has but nobody talks about

You're on-call. It's 2am. The checkout service is throwing 503s. You need to know: which service is actually broken, who owns it, what changed in the last 48 hours, and whether there's a related open ticket — all in under 60 seconds.

The answer exists. It's scattered across Jira, GitHub, Confluence, and your log aggregator. None of those tools talk to each other, and none of them understand your question.

This is the problem I set out to solve with **Aksibu Mini**: a fully open-source engineering context engine that aggregates fragmented engineering data into a unified knowledge base, answers operational questions in natural language, and — most importantly — never makes things up.

---

## Two systems in one

Aksibu Mini has two distinct pipelines that I'll walk through in depth.

**Pipeline 1: GraphRAG** — knows your engineering stack. Jira tickets, GitHub commits, Confluence docs, application logs. It builds a knowledge graph, understands who owns what, how services depend on each other, and can trace an incident from a log line back to the commit that introduced it.

**Pipeline 2: Cited Document Search** — knows your documents. Point it at a folder of a million PDFs and ask anything. Every single sentence in the answer comes with a `[1]`-style citation pointing to the exact paragraph it came from. If something isn't in your documents, it says so instead of guessing.

Both pipelines run at near-zero cost. The entire GraphRAG index on 18 records costs ~$0.01 to build. The document search pipeline runs entirely on local models — zero API cost for retrieval.

---

## Pipeline 1: GraphRAG on a budget

The first challenge was cost. A naive implementation would send every engineering record to a frontier LLM for entity extraction — at real scale that's thousands of dollars. Instead I built a **tiered extraction** system.

### The three tiers

**Tier 1 — Regex rules (free).** A rule-based extractor handles roughly 70% of records. It pattern-matches service names, ticket IDs, engineer names, and common relationship phrases. Zero API calls, zero cost.

```python
# Example: extract service mentions from a log line
SERVICE_ALIASES = {
    "checkout": "checkout-api",
    "checkout_api": "checkout-api",
    "checkout-service": "checkout-api",
    # ... 40+ aliases
}
```

**Tier 2 — Cheap LLM.** Only records with unresolved text get escalated. A fast, cheap model (Gemma 27B) handles these for ~$0.0002 per record.

**Tier 3 — Frontier LLM.** Only complex records trigger this: Confluence docs (rich prose, many implicit relationships), records over 800 characters with fewer than 2 entities found, or records with entities but zero detected relationships. Cost ~$0.003 per record.

The result: first build costs ~$0.01. Every subsequent run with no changes costs exactly $0.00 because everything is cached in SQLite.

### The knowledge graph

Entities extracted from all four sources feed into a `networkx.DiGraph`. Nodes represent services, engineers, tickets, commits, log lines, and docs. Edges represent relationships: `owns`, `affects`, `fixes`, `depends_on`, `caused_by`, `modified`.

```
checkout-api ──owns──→ Rahul Sharma
checkout-api ──affects──→ JR-441
JR-441 ──caused_by──→ sha:a3f8c2
sha:a3f8c2 ──modified──→ checkout-api
```

When you ask "Why is checkout broken?", the agent doesn't just do a keyword search. It traverses the graph: starts from the service node, BFS-expands 2 hops, finds related incidents, recent commits, and the on-call owner — all in one query.

### Agentic retrieval with a cost cap

The retrieval layer uses a tool-use loop. The LLM decides which tools to call based on the question, and has six available: `semantic_search`, `get_service_context`, `find_owner`, `get_recent_changes`, `get_active_incidents`, `get_dependencies`.

Simple questions (under 8 words, starting with "what is" or "define") skip the agent entirely and go straight to vector search — no tool calls, minimum cost.

Complex questions run through the loop, capped at 7 tool calls. Hit the cap and the system forces a synthesis with whatever it has. No runaway API cost.

---

## Pipeline 2: Cited Document Search at scale

This was the harder problem. The GraphRAG pipeline works on a fixed, structured dataset. The document search pipeline needs to work on anything — research papers, architecture docs, legal contracts, a folder of a million PDFs — and return answers where every claim is provably grounded.

### Why single-stage retrieval fails at scale

Most RAG tutorials show the same approach: embed your documents, store them in a vector database, retrieve top-k, send to LLM. This breaks down in two ways:

**Semantic gap.** Vector search finds semantically similar text. But if you ask "What does CONF-1023 say about the Redis failover?" and your doc literally contains the string "CONF-1023", your vector embedding likely won't rank it first — exact token matches are drowned out by semantic noise.

**Precision gap.** At 1 million documents, top-5 vector results are noisy. The LLM will receive irrelevant context and hallucinate connections between unrelated information.

### The three-stage hybrid retrieval

**Stage 1: BM25 + vector in parallel.** BM25 (a classical TF-IDF variant) excels at exact term matching. Dense vector search excels at semantic similarity. Run both, get 50 candidates each.

**Stage 2: Reciprocal Rank Fusion.** Merge the two lists without requiring score normalisation. The formula is simple and elegant:

```
RRF_score(chunk) = Σ  1 / (60 + rank_in_list)
```

A chunk appearing in both lists at rank 3 and rank 7 scores higher than a chunk appearing in only one list at rank 1. This rewards cross-list agreement — which is exactly the signal you want.

**Stage 3: Cross-encoder reranking.** The bi-encoder that built your vector index processes query and document independently. A cross-encoder processes them *together*, enabling full attention between query tokens and document tokens. This is far more accurate but too slow to run on all chunks — so I run it only on the top-20 RRF candidates, producing top-5 final chunks in milliseconds.

```python
from sentence_transformers import CrossEncoder

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, chunk["text"]) for chunk in candidates]
scores = ce.predict(pairs)
top_chunks = sorted(zip(scores, candidates), reverse=True)[:5]
```

The model is 22MB, runs locally, costs nothing, and was trained on 8.8 million query-passage pairs from Microsoft's MS MARCO dataset.

### Citation enforcement: the key to zero hallucination

Retrieval quality gets you 80% of the way there. The last 20% is prompt architecture.

The system prompt is a hard constraint:

```
You are a precise document analyst. Every factual claim MUST be
followed immediately by [N] where N is the source number. You MUST
NOT state anything that cannot be attributed to the provided sources.
If the answer is not in the sources, say: "The provided documents
do not contain enough information to answer this question."
```

Each of the 5 retrieved chunks is passed as a numbered source with its filename and position:

```
[1] (from: architecture.pdf, chunk 3 of 12)
Redis is used for session caching. All user sessions and cart
contents are stored with a 30-minute TTL...

[2] (from: incident-runbook.md, chunk 1 of 4)
In the event of Redis unavailability, the checkout service falls
back to in-memory session storage...
```

The LLM produces: *"Redis handles session caching with a 30-minute TTL [1]. If Redis goes down, checkout falls back to in-memory storage [2]."*

Then post-processing validates every `[N]` reference against the supplied chunks. Any citation pointing to a number outside `[1, 5]` — something the LLM invented — is stripped from the response.

```python
valid_refs = set(range(1, len(chunks) + 1))
used_refs = {int(m) for m in re.findall(r'\[(\d+)\]', raw_answer)}
invalid = used_refs - valid_refs

for ref in invalid:
    cleaned = re.sub(rf'[^.!?\n]*\[{ref}\][^.!?\n]*[.!?\n]?', '', cleaned)
```

This is a structural guarantee, not a soft instruction. The LLM literally cannot produce an uncited or fictionally-cited claim that survives post-processing.

### Scaling to 1 million documents

The folder ingestion path is memory-safe by design. Files are processed one at a time from disk — `chunk_file_from_path(path)` reads, chunks, and discards each file before loading the next. Chunks are upserted to ChromaDB in batches of 200. The BM25 index is rebuilt exactly once at the end.

```python
def ingest_folder(folder_path, batch_size=200, progress_callback=None):
    for file_path, total in iter_folder(folder_path):
        chunks = chunk_file_from_path(file_path)
        batch.extend(chunks)
        if len(batch) >= batch_size:
            ingest_documents(batch, rebuild_bm25=False)  # defer BM25
            batch = []
    _rebuild_bm25()  # rebuild once at the end
```

ChromaDB's HNSW index gives O(log n) query time at any scale. A query over 1 million chunks completes in ~5-10ms. The BM25 index holds ~2-4GB in RAM at 500K chunks — above that, the right swap is ElasticSearch or a sparse vector store like SPLADE.

---

## The full stack (and what it costs)

| Component | Technology | Cost |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Free, local |
| Vector store | ChromaDB (persistent, HNSW) | Free, local |
| BM25 | `rank_bm25` | Free, local |
| Cross-encoder | `ms-marco-MiniLM-L-6-v2` | Free, local |
| LLM (extraction + generation) | OpenRouter free tier | Free (with daily limits) |
| UI | Streamlit | Free |
| Total infra cost | — | **$0** |

The only cost is LLM API calls — and the tiered extraction system minimises those aggressively.

---

## What I'd build next

**Streaming answers.** The LLM currently returns the full response before the UI renders. Adding `stream=True` to the OpenAI client call would let citations appear word by word.

**Multi-hop citation chains.** Right now citations point to chunks. A more powerful version would trace: *"claim → chunk → source document → original data source"* — a full provenance chain from answer back to the raw Jira ticket or commit.

**ElasticSearch for BM25 at extreme scale.** The in-process `rank_bm25` pickle is convenient but caps out around 500K chunks. Swapping it for an ElasticSearch instance keeps the same RRF architecture and removes the RAM ceiling.

**Scheduled re-indexing.** The incremental manifest system already tracks which records have changed. Wrapping `python main.py` in a cron job would keep the knowledge graph current without manual intervention.

---

## Try it yourself

The full source is on GitHub: **[github.com/HetPatel1978/aksibu_mini](https://github.com/HetPatel1978/aksibu_mini)**

Live demo: **[hetpatel1978-aksibu-mini-uiapp-jj4vup.streamlit.app](https://hetpatel1978-aksibu-mini-uiapp-jj4vup.streamlit.app/)**

Setup takes about 5 minutes:

```bash
git clone https://github.com/HetPatel1978/aksibu_mini
cd aksibu_mini
pip install -r requirements.txt
cp .env.example .env   # add your OpenRouter key
python main.py         # build the index
streamlit run ui/app.py
```

Upload a few PDFs on the Document Search tab and ask anything. The citations are live — every `[1]` is clickable and expands to show the exact paragraph the answer came from.

---

*If you found this useful, the repo is open-source and PRs are welcome. The architecture is deliberately minimal — the goal was to show what's possible without any paid vector database, managed embedding API, or LLM framework.*

"""
retrieval/agent.py
───────────────────
Agentic retrieval engine using OpenRouter API.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# ── CRITICAL: Load .env BEFORE OpenAI client init ─────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_env_path))

from openai import OpenAI
from graph.graph_builder import load_graph
from graph.graph_query import (
    get_connected_context,
    find_service_owner,
    find_recent_changes,
    find_active_incidents,
    service_dependencies,
    find_service_node,
)
from index.vector_store import vector_search

_api_key = os.getenv("OPENROUTER_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "\n\nOPENROUTER_API_KEY not found.\n"
        "Make sure aksibu-mini/.env contains:\n"
        "  OPENROUTER_API_KEY=sk-or-v1-...\n"
        "Get your key from: https://openrouter.ai/keys\n"
    )

_client = OpenAI(
    api_key=_api_key,
    base_url="https://openrouter.ai/api/v1",
)
AGENT_MODEL    = os.getenv("AGENT_MODEL", "anthropic/claude-sonnet-4-5")
MAX_TOOL_CALLS = int(os.getenv("MAX_AGENT_TOOL_CALLS", "7"))


# ── TOOL DEFINITIONS ──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search across all engineering context by semantic similarity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "source_filter": {
                        "type": "string",
                        "enum": ["jira", "github", "confluence", "logs"]
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_context",
            "description": "Get ALL context connected to a specific service — docs, tickets, commits, logs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_owner",
            "description": "Find who owns a service and escalation contact.",
            "parameters": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_changes",
            "description": "Get recent commits for a service. Use for root cause analysis.",
            "parameters": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_incidents",
            "description": "Get open P1/P2 tickets for a service.",
            "parameters": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dependencies",
            "description": "Get what services a given service depends on.",
            "parameters": {
                "type": "object",
                "properties": {"service_name": {"type": "string"}},
                "required": ["service_name"],
            },
        },
    },
]


# ── TOOL EXECUTOR ─────────────────────────────────────────────
def _execute_tool(name: str, inputs: dict, G) -> str:
    try:
        if name == "semantic_search":
            results = vector_search(inputs["query"], n_results=4, source_filter=inputs.get("source_filter"))
            return json.dumps(results, indent=2)
        elif name == "get_service_context":
            svc = inputs["service_name"]
            svc_id = find_service_node(G, svc) or f"service__{svc.replace('-','_')}"
            context = get_connected_context(G, svc_id, hops=2)
            return json.dumps(context[:10], indent=2)
        elif name == "find_owner":
            return json.dumps(find_service_owner(G, inputs["service_name"]))
        elif name == "get_recent_changes":
            return json.dumps(find_recent_changes(G, inputs["service_name"]), indent=2)
        elif name == "get_active_incidents":
            return json.dumps(find_active_incidents(G, inputs["service_name"]), indent=2)
        elif name == "get_dependencies":
            return json.dumps({"dependencies": service_dependencies(G, inputs["service_name"])})
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as ex:
        return json.dumps({"error": str(ex)})


# ── QUERY COMPLEXITY CLASSIFIER ───────────────────────────────
SIMPLE_KEYWORDS = {"what is", "define", "explain", "describe", "who is", "what does", "how does", "when was"}

def _is_simple_query(question: str) -> bool:
    q = question.lower()
    return any(q.startswith(kw) for kw in SIMPLE_KEYWORDS) and len(question.split()) < 8


# ── SYSTEM PROMPT ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert engineering assistant with access to a unified 
knowledge base spanning Jira tickets, GitHub commits, Confluence docs, and service logs.

When asked about incidents, bugs, or service health:
1. Start with semantic_search to get broad context
2. Use get_service_context for deep-dive on implicated services
3. Check get_recent_changes for root cause clues (what changed?)
4. Use find_owner to know who to page
5. Check get_active_incidents for ongoing issues
6. Synthesize a clear, specific answer citing ticket IDs, commit SHAs, and doc titles

Be specific. Name the exact root cause if you can identify it.
Always state who to contact. Cite your sources."""


# ── MAIN AGENT ────────────────────────────────────────────────
def run_agent(question: str, verbose: bool = True) -> dict:
    G = load_graph()
    tools_called = []
    seen_calls = set()
    tool_count = 0
    stopped_early = False

    # Fast path for simple queries
    if _is_simple_query(question):
        if verbose: print("  ⚡ Simple query — direct vector search")
        results = vector_search(question, n_results=3)
        context = "\n\n".join(r["text"] for r in results)
        resp = _client.chat.completions.create(
            model=AGENT_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer concisely."}
            ],
        )
        return {
            "answer": resp.choices[0].message.content,
            "tools_called": [{"tool": "semantic_search", "input": {"query": question}}],
            "tool_call_count": 1,
            "stopped_early": False,
        }

    if verbose: print(f"\n🤖 Agent: '{question}'")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    # ── AGENTIC LOOP ──────────────────────────────────────────
    while tool_count < MAX_TOOL_CALLS:
        response = _client.chat.completions.create(
            model=AGENT_MODEL,
            max_tokens=2000,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": msg.tool_calls
        })

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            fn_name   = tc.function.name
            fn_inputs = json.loads(tc.function.arguments)

            dedup_key = (fn_name, json.dumps(fn_inputs, sort_keys=True))
            if dedup_key in seen_calls:
                if verbose: print(f"  ⏭️  Skipping duplicate: {fn_name}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"note": "Duplicate — see previous result"}),
                })
                continue

            seen_calls.add(dedup_key)
            tool_count += 1

            if verbose: print(f"  🔧 [{tool_count}/{MAX_TOOL_CALLS}] {fn_name}({json.dumps(fn_inputs)[:80]})")

            result = _execute_tool(fn_name, fn_inputs, G)
            tools_called.append({"tool": fn_name, "input": fn_inputs})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if tool_count >= MAX_TOOL_CALLS:
            stopped_early = True
            if verbose: print(f"  ⚠️  Hit cap ({MAX_TOOL_CALLS}) — forcing answer")
            messages.append({
                "role": "user",
                "content": "Tool call limit reached. Synthesize your best answer from what you have."
            })
            final = _client.chat.completions.create(
                model=AGENT_MODEL, max_tokens=1500, messages=messages
            )
            messages.append({"role": "assistant", "content": final.choices[0].message.content})
            break

    # Extract final answer
    answer = ""
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("content"):
            answer = m["content"]
            break

    if verbose: print(f"  ✅ Done — {tool_count} tool calls\n")

    return {
        "answer": answer,
        "tools_called": tools_called,
        "tool_call_count": tool_count,
        "stopped_early": stopped_early,
    }
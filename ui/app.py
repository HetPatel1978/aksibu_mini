"""
ui/app.py
──────────
Streamlit web interface for the Engineering Context Engine.

Features:
  - Natural language question input
  - Example questions sidebar
  - Shows the final answer with source citations
  - Shows which tools the agent called (for transparency)
  - Cost/tool call counter

Run with: streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from retrieval.agent import run_agent

st.set_page_config(
    page_title="Engineering Context Engine",
    page_icon="🔍",
    layout="wide",
)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Aksibu-Mini-blue?style=for-the-badge")
    st.markdown("### 💡 Example Questions")

    examples = [
        "Why is the checkout service returning 503 errors?",
        "Who owns payment-gateway and who should I page?",
        "What changed in payment-gateway recently?",
        "What are the dependencies of checkout-api?",
        "Give me a root cause analysis of the current incident",
        "Which services does fraud-detection depend on?",
        "What open incidents exist right now?",
    ]

    for q in examples:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["question"] = q

    st.divider()
    st.markdown("**Stack**")
    st.markdown("""
- 🧠 GraphRAG (networkx)
- 🔍 ChromaDB vector search  
- 🤖 Claude agentic retrieval
- 💰 Tiered extraction (rules → haiku → sonnet)
    """)

# ── MAIN ──────────────────────────────────────────────────────
st.title("🔍 Engineering Context Engine")
st.caption("Ask anything about your services, incidents, ownership, or recent changes.")

col1, col2 = st.columns([4, 1])

with col1:
    question = st.text_input(
        "Ask a question:",
        value=st.session_state.get("question", ""),
        placeholder="Why is checkout broken and who should I page?",
        label_visibility="collapsed",
    )

with col2:
    run = st.button("🔍 Investigate", type="primary", use_container_width=True)

if run and question:
    st.session_state["question"] = question

    with st.spinner("🤖 Agent investigating across Jira, GitHub, Confluence, and logs..."):
        try:
            result = run_agent(question, verbose=False)
        except FileNotFoundError as e:
            st.error(f"⚠️ Index not built yet. Run `python main.py` first.\n\n{e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            st.stop()

    # ── ANSWER ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Answer")
    st.markdown(result["answer"])

    # ── AGENT TRACE ───────────────────────────────────────────
    with st.expander(f"🔧 Agent trace — {result['tool_call_count']} tool calls"):
        if result["stopped_early"]:
            st.warning(f"⚠️ Hit tool call limit ({result['tool_call_count']}). Answer may be incomplete.")

        for i, call in enumerate(result["tools_called"], 1):
            st.markdown(f"**Step {i}:** `{call['tool']}` with `{call['input']}`")

elif not question and run:
    st.warning("Please enter a question.")

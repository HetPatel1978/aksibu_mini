"""
ui/pages/doc_search.py
───────────────────────
Document Search — upload individual files or point at a local folder
(e.g. a database of 1M PDFs), ask questions, get fully cited answers.
Parallel pipeline: own chunker, own ChromaDB collection, own BM25 index.
Does NOT touch the existing engineering context index.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

st.set_page_config(
    page_title="Document Search",
    page_icon="📄",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="background:#1e293b;border-radius:8px;padding:10px 14px;'
        'margin-bottom:16px;">'
        '<span style="font-size:15px;font-weight:700;color:#e2e8f0">📄 Doc Search</span><br>'
        '<span style="font-size:11px;color:#94a3b8">Hybrid RAG · Cited answers</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    ingest_mode = st.radio(
        "Ingest method",
        ["Upload files", "Folder path"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Option A: file uploader ───────────────────────────────────────────────
    if ingest_mode == "Upload files":
        st.markdown("### Upload Documents")
        st.caption("Supported: `.txt` `.md` `.pdf`")

        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["txt", "md", "pdf"],
            label_visibility="collapsed",
        )

        ingest_clicked = st.button(
            "Ingest Documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )

        if ingest_clicked and uploaded_files:
            progress = st.progress(0, text="Reading files...")
            try:
                from ingestion.doc_chunker import chunk_files

                files_data = [(f.name, f.read()) for f in uploaded_files]
                progress.progress(20, text="Chunking documents...")

                chunks = chunk_files(files_data)
                progress.progress(55, text=f"Indexing {len(chunks)} chunks...")

                from index.doc_store import ingest_documents
                stats = ingest_documents(chunks)

                progress.progress(100, text="Done!")
                st.success(
                    f"Indexed **{stats['indexed']}** new chunks "
                    f"({stats['skipped']} already present).  \n"
                    f"Total in store: **{stats['total_chunks_in_store']}**"
                )
            except Exception as ex:
                progress.empty()
                st.error(f"Ingestion failed: {ex}")

    # ── Option B: folder path ─────────────────────────────────────────────────
    else:
        st.markdown("### Folder Ingestion")
        st.caption(
            "Enter a local folder path. All `.txt` `.md` `.pdf` files are "
            "indexed recursively. Safe for large corpora — processes in batches, "
            "BM25 rebuilt once at the end."
        )

        folder_path = st.text_input(
            "Folder path",
            placeholder="/data/my_pdfs  or  C:\\docs\\reports",
            label_visibility="collapsed",
        )

        folder_clicked = st.button(
            "Ingest Folder",
            type="primary",
            use_container_width=True,
            disabled=not folder_path.strip() if folder_path else True,
        )

        if folder_clicked and folder_path and folder_path.strip():
            path = folder_path.strip()
            if not os.path.isdir(path):
                st.error(f"Directory not found: `{path}`")
            else:
                progress_bar = st.progress(0, text="Scanning folder…")
                status_text  = st.empty()

                def _cb(files_done: int, files_total: int, chunks_indexed: int):
                    pct = int(files_done / max(files_total, 1) * 95)
                    progress_bar.progress(
                        pct,
                        text=f"File {files_done}/{files_total} · "
                             f"{chunks_indexed:,} chunks indexed so far…",
                    )

                try:
                    from index.doc_store import ingest_folder
                    stats = ingest_folder(path, batch_size=200, progress_callback=_cb)

                    progress_bar.progress(100, text="Done!")
                    status_text.success(
                        f"Found **{stats['files_found']}** files · "
                        f"Processed **{stats['files_processed']}** · "
                        f"Skipped **{stats['files_skipped']}**  \n"
                        f"New chunks: **{stats['chunks_indexed']:,}** · "
                        f"Already present: **{stats['chunks_skipped']:,}**  \n"
                        f"Total in store: **{stats['total_chunks_in_store']:,}**"
                    )
                except Exception as ex:
                    progress_bar.empty()
                    st.error(f"Folder ingestion failed: {ex}")

    st.divider()

    # Store stats
    try:
        from index.doc_store import get_store_stats
        store = get_store_stats()

        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", f"{store['chunk_count']:,}")
        col_b.metric("Docs",   store["doc_count"])

        if store["filenames"]:
            with st.expander(f"Indexed files ({len(store['filenames'])})"):
                for fn in store["filenames"][:50]:
                    st.markdown(f"- `{fn}`")
                if len(store["filenames"]) > 50:
                    st.caption(f"… and {len(store['filenames']) - 50} more")
        else:
            st.info("No documents indexed yet.")
    except Exception:
        st.info("No documents indexed yet.")

    st.divider()
    st.caption(
        "Pipeline: BM25 + Vector → RRF fusion → Cross-encoder rerank → "
        "Citation-enforced LLM"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("📄 Document Search")
st.caption(
    "Index documents via the sidebar (upload files or point at a folder), "
    "then ask any question. Every claim is backed by a cited source chunk."
)

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Question",
        placeholder="What does the document say about caching strategies?",
        label_visibility="collapsed",
    )
with col2:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

if search_clicked:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        from index.doc_store import get_store_stats
        s = get_store_stats()
        if s["chunk_count"] == 0:
            st.warning(
                "No documents indexed yet. "
                "Upload files or enter a folder path in the sidebar first."
            )
            st.stop()
    except Exception as ex:
        st.error(f"Could not reach document store: {ex}")
        st.stop()

    with st.spinner("Searching and generating cited answer… (first query loads the reranker ~3s)"):
        try:
            from retrieval.cited_retriever import retrieve_with_citations
            result = retrieve_with_citations(query.strip(), top_k=5)
        except Exception as ex:
            st.error(f"Retrieval error: {ex}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Answer")
    st.markdown(result["answer"])

    # ── Retrieval stats ───────────────────────────────────────────────────────
    rs = result.get("retrieval_stats", {})
    with st.expander("Retrieval pipeline stats"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BM25 candidates",   rs.get("bm25_hits",    "—"))
        c2.metric("Vector candidates", rs.get("vector_hits",  "—"))
        c3.metric("After RRF fusion",  rs.get("after_rrf",    "—"))
        c4.metric("After rerank",      rs.get("after_rerank", "—"))

    # ── Citations ─────────────────────────────────────────────────────────────
    citations = result.get("citations", [])
    if citations:
        st.markdown("### Sources")
        for cite in citations:
            header = (
                f"[{cite['ref']}] {cite['filename']} — "
                f"chunk {cite['chunk_index'] + 1} of {cite['total_chunks']}"
            )
            with st.expander(header):
                st.markdown(
                    f'<span style="font-size:12px;color:#94a3b8">'
                    f'File: <code>{cite["filename"]}</code> &nbsp;|&nbsp; '
                    f'Chunk {cite["chunk_index"] + 1} / {cite["total_chunks"]}'
                    f"</span>",
                    unsafe_allow_html=True,
                )
                st.divider()
                st.markdown(cite["text"])
    else:
        st.info("No cited sources were produced for this answer.")

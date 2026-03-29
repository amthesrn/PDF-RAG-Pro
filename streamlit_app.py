"""
streamlit_app.py
────────────────
Web UI for the PDF RAG Chatbot.
Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make src importable from app/
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config.settings import settings
from src.generation.llm_chain import LLMChainError, RAGChain
from src.ingestion.chunker import DocumentChunker
from src.ingestion.image_processor import ImageProcessor
from src.ingestion.pdf_parser import PDFParser, PDFParserError
from src.models.schemas import ChatMessage, RAGRequest, RetrievalMode
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import HybridRetriever
from src.utils.logger import get_logger
from src.vectorstore.chroma_store import VectorStore
from src.utils.pdf_registry import PDFRegistry

logger = get_logger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle   { color: #888; font-size: 0.95rem; margin-top: 0; }
    .source-box { background: #1e1e2e; border-radius: 8px; padding: 12px;
                  border-left: 4px solid #7c3aed; font-size: 0.85rem; margin-top: 8px; }
    .metric-row { display: flex; gap: 16px; }
    .status-ok  { color: #22c55e; font-weight: 600; }
    .status-err { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def load_pipeline():
    store     = VectorStore()
    retriever = HybridRetriever(store)
    reranker  = Reranker()
    chain     = RAGChain(retriever, reranker)
    return store, retriever, chain


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.caption(f"LLM: `{settings.llm_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model}`")

    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "semantic", "keyword"],
        index=0,
        help="Hybrid (recommended) combines semantic + keyword search",
    )
    top_k = st.slider("Sources to retrieve", min_value=1, max_value=8, value=4)
    llm_provider = st.radio(
        "LLM Provider",
        options=["groq", "gemini"],
        format_func=lambda x: "Groq (Llama 3)" if x == "groq" else "Gemini 3 (Flash)",
        help="Switch dynamically between LLM providers to compare answer quality"
    )

    st.divider()
    st.markdown("### 📤 Upload & Index PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    skip_images   = st.checkbox("Skip image descriptions (faster)", value=False)

    registry = PDFRegistry()

    if uploaded_file and st.button("🚀 Index PDF", type="primary"):
        file_bytes = uploaded_file.getbuffer()
        file_hash  = PDFRegistry.compute_hash(bytes(file_bytes))

        if registry.is_indexed(file_hash):
            # ── Previously indexed — skip re-indexing ─────────────────
            entry = registry.get_entry(file_hash)
            store, retriever, chain = load_pipeline()
            retriever.build_bm25_index()  # rebuild BM25 from ChromaDB
            st.session_state.ingested = True
            st.success(
                f"✅ **Already indexed!** `{entry['filename']}` — "
                f"{entry['total_pages']} pages, {entry['total_chunks']} chunks. "
                f"Skipped re-indexing."
            )
        else:
            # ── New PDF — full indexing ────────────────────────────────
            tmp_path = Path("data/pdfs") / uploaded_file.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner("Parsing & indexing… this may take a few minutes."):
                try:
                    parser    = PDFParser(tmp_path, image_output_dir="data/images")
                    raw_pages = parser.parse()

                    img_descs = []
                    if not skip_images:
                        processor = ImageProcessor()
                        for page in raw_pages:
                            if page.image_paths:
                                img_descs.extend(
                                    processor.describe_batch(page.image_paths, page.page_number)
                                )

                    chunker = DocumentChunker()
                    chunks  = chunker.chunk_pages(raw_pages, source_pdf=uploaded_file.name, image_descriptions=img_descs)

                    store, retriever, chain = load_pipeline()

                    from src.embeddings.embedder import Embedder
                    embedder   = Embedder()
                    embeddings = embedder.embed_documents([c.text for c in chunks])
                    store.add_chunks(chunks, embeddings)

                    retriever.build_bm25_index(chunks)

                    # Register this PDF so future uploads are skipped
                    registry.register(
                        file_hash=file_hash,
                        filename=uploaded_file.name,
                        total_pages=len(raw_pages),
                        total_chunks=len(chunks),
                    )

                    st.session_state.ingested = True
                    st.success(f"✅ Indexed {len(raw_pages)} pages → {len(chunks)} chunks")

                except PDFParserError as exc:
                    st.error(f"PDF parsing error: {exc}")
                except Exception as exc:
                    logger.error("Ingestion error: %s", exc, exc_info=True)
                    st.error(f"Unexpected error: {exc}")

    st.divider()

    # ── Index status ──────────────────────────────────────────────────
    try:
        store, _, _ = load_pipeline()
        count = store.count()
        if count > 0:
            st.markdown(f"**Index status:** <span class='status-ok'>● {count} chunks</span>",
                        unsafe_allow_html=True)
            st.session_state.ingested = True
        else:
            st.markdown("**Index status:** <span class='status-err'>● Empty — upload a PDF</span>",
                        unsafe_allow_html=True)
    except Exception:
        st.markdown("**Index status:** <span class='status-err'>● Error</span>",
                    unsafe_allow_html=True)

    # ── Tracked PDFs ──────────────────────────────────────────────────
    tracked = registry.get_all()
    if tracked:
        st.divider()
        st.markdown("### 📑 Tracked PDFs")
        for entry in tracked:
            st.markdown(
                f"📄 **{entry['filename']}**  \n"
                f"&nbsp;&nbsp;&nbsp; {entry['total_pages']} pages · {entry['total_chunks']} chunks  \n"
                f"&nbsp;&nbsp;&nbsp; _Indexed: {entry['indexed_at'][:10]}_"
            )

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📚 PDF RAG Chatbot</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask anything about your document — '
    'answers come ONLY from the indexed PDF, no hallucination.</p>',
    unsafe_allow_html=True,
)
st.divider()

if not st.session_state.ingested:
    st.info("👈 Upload and index a PDF using the sidebar to get started.")
    st.stop()

# ── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source pages"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>Page {src["page"]} · {src["type"].upper()}</b><br>'
                        f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the document…"):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build history for context
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in st.session_state.messages[-12:]
        if m["role"] in ("user", "assistant")
    ]

    mode_map = {
        "hybrid":   RetrievalMode.HYBRID,
        "semantic": RetrievalMode.SEMANTIC,
        "keyword":  RetrievalMode.KEYWORD,
    }

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                _, _, chain = load_pipeline()
                request  = RAGRequest(
                    question=prompt,
                    chat_history=history,
                    retrieval_mode=mode_map[retrieval_mode],
                    top_k=top_k,
                    llm_provider=llm_provider,
                )
                response = chain.answer(request)
                answer   = response.answer

                sources  = [
                    {
                        "page": c.page_number,
                        "type": c.content_type.value,
                        "text": c.text,
                    }
                    for c in response.source_chunks
                ]

            except LLMChainError as exc:
                answer  = f"⚠️ LLM Error: {exc}"
                sources = []
            except Exception as exc:
                logger.error("Chat error: %s", exc, exc_info=True)
                answer  = f"⚠️ Unexpected error: {exc}"
                sources = []

        st.markdown(answer)

        if sources:
            pages = sorted({s["page"] for s in sources})
            st.caption(f"📄 Sources: pages {pages} | ⏱ {response.processing_time_ms:.0f}ms")
            with st.expander("📄 Source pages"):
                for src in sources:
                    st.markdown(
                        f'<div class="source-box">'
                        f'<b>Page {src["page"]} · {src["type"].upper()}</b><br>'
                        f'{src["text"][:400]}{"…" if len(src["text"]) > 400 else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # Persist to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })

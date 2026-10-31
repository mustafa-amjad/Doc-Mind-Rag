"""DocMind — Streamlit UI. Run with:  streamlit run app.py"""
import html  # escapes text so file names / PDF text can never break our HTML
import re  # used to clean whitespace in snippets
from pathlib import Path  # builds file paths that work on Windows and Mac

import streamlit as st

from docmind import config
from docmind.embedder import Embedder
from docmind.llm import LLM
from docmind.pipeline import RAGPipeline
from docmind.reranker import Reranker

ROOT = Path(__file__).parent  # folder where app.py lives

st.set_page_config(  # must be the FIRST Streamlit call
    page_title="DocMind · Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [  # one-click starter questions shown after indexing
    "Summarize this document in 5 bullet points",
    "What are the key takeaways?",
    "List any important dates, numbers or names",
]


# ====================== helpers ======================
def load_css() -> None:
    """Read assets/style.css and inject it into the page."""
    css = (ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def esc(value) -> str:
    """Safe-for-HTML version of any value."""
    return html.escape(str(value))


def shorten(text: str, limit: int = 220) -> str:
    """Collapse whitespace and cut long text at a word boundary, so evidence stays short."""
    text = re.sub(r"\s+", " ", text).strip()  # many spaces/newlines -> one space
    if len(text) <= limit:  # already short, keep as is
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"  # cut, drop the half-word, add dots


@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_models():
    """Heavy models load ONCE and are shared by every browser session."""
    return Embedder(config.EMBED_MODEL), Reranker(config.RERANK_MODEL), LLM(config.GROQ_API_KEY, config.LLM_MODEL)


def source_chips(sources: list[dict]) -> str:
    """Small pills like 'report.pdf · p.3' (one per unique file+page)."""
    seen, chips = set(), []
    for src in sources:
        key = (src["source"], src["page"])
        if key in seen:  # skip duplicates
            continue
        seen.add(key)
        chips.append(f'<span class="chip cyan">📄 {esc(src["source"])} · p.{esc(src["page"])}</span>')
    return f'<div class="chips">{"".join(chips)}</div>'


def evidence_html(sources: list[dict]) -> str:
    """Compact evidence list. Each passage is cut short, full text hides in a <details> toggle."""
    rows = []
    for number, src in enumerate(sources, start=1):
        score = f'<span class="chip muted">match {src["rerank_score"]:.2f}</span>' if "rerank_score" in src else ""
        rows.append(
            f'<div class="src">'
            f'<div class="src-head"><span class="src-num">{number}</span>'
            f'{esc(src["source"])} · page {esc(src["page"])} {score}</div>'
            f'<div class="src-snippet">{esc(shorten(src["text"]))}</div>'
            f'<details><summary>Read full passage</summary><p>{esc(src["text"])}</p></details>'
            f"</div>"
        )
    return "".join(rows)


def render_answer_extras(sources: list[dict], meta: str) -> None:
    """Chips + speed badge + collapsed evidence box. Used for new AND old messages."""
    st.markdown(
        source_chips(sources) + f'<div class="chips"><span class="chip muted">⚡ {esc(meta)}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander(f"View evidence · {len(sources)} passages", expanded=False):  # collapsed by default
        st.markdown(evidence_html(sources), unsafe_allow_html=True)


def chat_to_markdown(messages: list[dict]) -> str:
    """Turn the chat into a .md file the user can download."""
    lines = ["# DocMind chat export", ""]
    for message in messages:
        if message["role"] == "user":
            lines += [f"## Q: {message['content']}", ""]
        else:
            lines += [message["content"], ""]
            pages = sorted({f"{s['source']} p.{s['page']}" for s in message.get("sources", [])})
            if pages:
                lines += [f"_Sources: {', '.join(pages)}_", ""]
    return "\n".join(lines)


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-badge">Document Intelligence</div>
          <h1>Doc<span>Mind</span></h1>
          <p>Ask questions across your PDFs and get answers backed by the exact pages they came from.</p>
          <div class="chips">
            <span class="chip">🔎 Hybrid search</span>
            <span class="chip">🧠 Cross-encoder re-ranking</span>
            <span class="chip">⚡ FAISS HNSW</span>
            <span class="chip">🦙 Llama 3.3 70B</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def steps(ready: bool) -> None:
    """3-step guide. The current step is highlighted."""
    state = ["done", "active", ""] if not ready else ["done", "done", "active"]  # step 1 shows done once page loads
    st.markdown(
        f"""
        <div class="step-grid">
          <div class="step-card {'active' if not ready else 'done'}"><div class="step-num">1</div>
            <div class="step-title">Upload</div><div class="step-text">Drop one or more PDFs into the sidebar.</div></div>
          <div class="step-card {'' if not ready else 'done'}"><div class="step-num">2</div>
            <div class="step-title">Build index</div><div class="step-text">We read, chunk and embed your documents once.</div></div>
          <div class="step-card {'active' if ready else ''}"><div class="step-num">3</div>
            <div class="step-title">Ask</div><div class="step-text">Chat normally. Every answer shows its sources.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_cards(stats: dict) -> None:
    st.markdown(
        f"""
        <div class="stat-grid">
          <div class="stat-card"><div class="stat-label">Files</div><div class="stat-value">{stats['files']}</div></div>
          <div class="stat-card"><div class="stat-label">Pages</div><div class="stat-value">{stats['pages']}</div></div>
          <div class="stat-card"><div class="stat-label">Chunks</div><div class="stat-value">{stats['chunks']}</div></div>
          <div class="stat-card"><div class="stat-label">Index</div><div class="stat-value accent">{esc(stats['index']).upper()}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ====================== start ======================
load_css()
hero()

if not config.GROQ_API_KEY:  # no key -> stop early with a clear message
    st.error("**GROQ_API_KEY is missing.** Copy `.env.example` to `.env`, add your key, then restart the app.")
    st.stop()

embedder, reranker, llm = load_models()

if "pipeline" not in st.session_state:  # each browser tab gets its OWN pipeline (own documents)
    st.session_state.pipeline = RAGPipeline(embedder, reranker, llm)
if "messages" not in st.session_state:  # chat history
    st.session_state.messages = []
if "stats" not in st.session_state:  # index stats for the cards
    st.session_state.stats = None
pipeline: RAGPipeline = st.session_state.pipeline

# ====================== sidebar ======================
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">📄</div>'
        '<div><div class="brand-name">DocMind</div><div class="brand-tag">Chat with your documents</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-title"><span class="num">1</span>Documents</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if st.button("⚙️ Build index", type="primary", use_container_width=True, disabled=not uploads):
        with st.spinner("Reading, chunking and embedding..."):
            try:
                st.session_state.stats = pipeline.ingest([(f.name, f.getvalue()) for f in uploads])
                st.session_state.messages = []  # new documents -> fresh chat
                st.toast("Index ready. Ask your first question!", icon="✅")
            except ValueError as error:  # friendly message instead of a crash
                st.error(str(error))

    st.markdown('<div class="side-title"><span class="num">2</span>Retrieval</div>', unsafe_allow_html=True)
    use_hybrid = st.toggle("Hybrid search", value=True, help="Combine keyword (BM25) and meaning (vector) search.")
    use_rerank = st.toggle("Re-ranking", value=True, help="A cross-encoder re-orders results for higher accuracy.")
    top_k = st.slider("Passages sent to the AI", 1, 8, config.FINAL_K, help="More passages = more context, slower answers.")
    mode = ("Hybrid" if use_hybrid else "Vector") + (" + Re-rank" if use_rerank else "")
    st.markdown(f'<div class="mode-note"><b>Active:</b> {mode} · top {top_k}</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title"><span class="num">3</span>Session</div>', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Export chat",
        data=chat_to_markdown(st.session_state.messages),
        file_name="docmind_chat.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    if st.button("🗑️ Clear chat", use_container_width=True, disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

# ====================== main area ======================
typed = st.chat_input("Ask anything about your documents...", disabled=not pipeline.is_ready)
question = st.session_state.pop("pending_question", None) or typed  # clicked suggestion wins over typed text

if not pipeline.is_ready:  # nothing indexed yet -> show the guide
    st.markdown('<div class="section-title">Get started</div>', unsafe_allow_html=True)
    steps(ready=False)
else:
    st.markdown('<div class="section-title">Your knowledge base</div>', unsafe_allow_html=True)
    stat_cards(st.session_state.stats)
    st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

for message in st.session_state.messages:  # redraw history on every rerun
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "📄"):
        st.markdown(message["content"])
        if message.get("sources"):
            render_answer_extras(message["sources"], message["meta"])

if pipeline.is_ready and not st.session_state.messages and not question:  # empty chat -> offer starters
    st.caption("Try one of these:")
    columns = st.columns(len(SUGGESTIONS))
    for column, suggestion in zip(columns, SUGGESTIONS):
        if column.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion  # saved, picked up on the next run
            st.rerun()

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching your documents..."):
            result = pipeline.ask(question, use_hybrid, use_rerank, top_k)
        st.markdown(result["answer"])
        meta = f"{result['total_ms'] / 1000:.1f}s total · retrieval {result['retrieve_ms']:.0f} ms · {mode}"
        render_answer_extras(result["sources"], meta)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"], "meta": meta}
    )
    st.rerun()  # redraw so the sidebar buttons (export/clear) enable immediately

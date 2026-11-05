"""DocMind: Streamlit UI with login, saved chats, pinned chats and chat memory.
Run with:  streamlit run app.py"""
import html  # escapes text so file names / PDF text can never break our HTML
import re  # used to clean whitespace in snippets
import shutil  # deletes a chat's saved index folder
from pathlib import Path  # builds file paths that work on Windows and Mac

import streamlit as st

from docmind import auth, config, cookies, db
from docmind.embedder import Embedder
from docmind.llm import LLM
from docmind.pipeline import RAGPipeline
from docmind.reranker import Reranker

ROOT = Path(__file__).parent  # folder where app.py lives
DEFAULT_TITLE = "New chat"

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


# ====================== small helpers ======================
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


def index_folder(user_id: int, chat_id: int) -> Path:
    """Where this chat's chunks + vectors are saved."""
    return config.INDEX_DIR / str(user_id) / str(chat_id)


# ====================== rendering helpers ======================
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


def render_answer_extras(sources: list[dict], meta: str, searched: str = "", question: str = "") -> None:
    """Chips + speed badge + collapsed evidence box. Used for new AND old messages."""
    badges = [f'<span class="chip muted">⚡ {esc(meta)}</span>']
    if searched and searched.strip().lower() != question.strip().lower():  # show when memory rewrote the question
        badges.append(f'<span class="chip">🔁 Understood as: {esc(shorten(searched, 90))}</span>')
    st.markdown(source_chips(sources) + f'<div class="chips">{"".join(badges)}</div>', unsafe_allow_html=True)
    with st.expander(f"View evidence · {len(sources)} passages", expanded=False):  # collapsed by default
        st.markdown(evidence_html(sources), unsafe_allow_html=True)


def chat_to_markdown(title: str, messages: list[dict]) -> str:
    """Turn the chat into a .md file the user can download."""
    lines = [f"# {title}", ""]
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
            <span class="chip">💾 Saved chats</span>
            <span class="chip">🦙 Built with Llama</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def steps(ready: bool) -> None:
    """3-step guide. The current step is highlighted."""
    st.markdown(
        f"""
        <div class="step-grid">
          <div class="step-card {'active' if not ready else 'done'}"><div class="step-num">1</div>
            <div class="step-title">Upload</div><div class="step-text">Open <b>Documents</b> in the sidebar and drop in your PDFs.</div></div>
          <div class="step-card"><div class="step-num">2</div>
            <div class="step-title">Build index</div><div class="step-text">We read, chunk and embed them once. Saved with this chat.</div></div>
          <div class="step-card {'active' if ready else ''}"><div class="step-num">3</div>
            <div class="step-title">Ask</div><div class="step-text">Chat normally. Follow-up questions work too.</div></div>
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
    chips = "".join(f'<span class="chip">📄 {esc(name)}</span>' for name in stats.get("names", []))
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)


# ====================== session state: which user, which chat ======================
def fresh_pipeline() -> RAGPipeline:
    return RAGPipeline(st.session_state.embedder, st.session_state.reranker, st.session_state.llm)


def new_chat() -> None:
    """Blank chat. Nothing is saved to the database until the user uploads or asks something."""
    st.session_state.chat_id = None
    st.session_state.messages = []
    st.session_state.pipeline = fresh_pipeline()
    st.session_state.stats = None
    st.session_state.confirm_delete = None


def open_chat(chat_id: int) -> None:
    """Switch to a saved chat: load its messages from SQLite and its index from disk."""
    user_id = st.session_state.user["id"]
    if db.get_chat(chat_id, user_id) is None:  # not found or not yours
        new_chat()
        return
    st.session_state.chat_id = chat_id
    st.session_state.messages = db.get_messages(chat_id, user_id)
    pipeline = fresh_pipeline()
    folder = index_folder(user_id, chat_id)
    st.session_state.stats = pipeline.load(folder) if (folder / "chunks.json").exists() else None
    st.session_state.pipeline = pipeline
    st.session_state.confirm_delete = None


def start_login(user: dict) -> None:
    """Log in for this tab AND remember the user in a browser cookie (survives refresh)."""
    st.session_state.user = user
    cookies.queue_set(auth.create_session(user["id"]))


def logout() -> None:
    auth.end_session(cookies.read_token())  # kill the saved login so the old cookie stops working
    cookies.queue_clear()  # and remove the cookie from the browser
    for key in ("user", "chat_id", "messages", "pipeline", "stats", "confirm_delete", "pending_question"):
        st.session_state.pop(key, None)


# ====================== login screen ======================
def render_auth() -> None:
    hero()
    _, middle, _ = st.columns([1, 1.6, 1])
    with middle:
        login_tab, signup_tab = st.tabs(["🔐 Log in", "✨ Create account"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if submitted:
                user = auth.login(username, password)
                if user:
                    start_login(user)
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
        with signup_tab:
            with st.form("signup_form"):
                new_user = st.text_input("Choose a username", key="signup_user")
                new_pass = st.text_input("Choose a password (8+ characters)", type="password", key="signup_pass")
                confirm = st.text_input("Repeat password", type="password", key="signup_confirm")
                created = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if created:
                if new_pass != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        start_login(auth.signup(new_user, new_pass))
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        st.caption("Your chats and indexed document text are saved on this server, under your account only.")


# ====================== sidebar: chat list ======================
def chat_row(chat: dict, active: bool, user_id: int) -> None:
    """One line in Pinned / Recent: open button, pin button, delete button."""
    chat_id = chat["id"]
    if st.session_state.get("confirm_delete") == chat_id:  # second click: ask before deleting
        st.caption(f"Delete “{shorten(chat['title'], 24)}”?")
        yes, no = st.columns(2)
        if yes.button("Yes, delete", key=f"yes_{chat_id}", use_container_width=True):
            db.delete_chat(chat_id, user_id)
            shutil.rmtree(index_folder(user_id, chat_id), ignore_errors=True)  # remove saved index too
            if st.session_state.chat_id == chat_id:
                new_chat()
            st.session_state.confirm_delete = None
            st.rerun()
        if no.button("Cancel", key=f"no_{chat_id}", use_container_width=True):
            st.session_state.confirm_delete = None
            st.rerun()
        return
    open_col, pin_col, del_col = st.columns([0.66, 0.17, 0.17], gap="small")
    icon = "📍" if chat["pinned"] else "💬"
    if open_col.button(f"{icon} {shorten(chat['title'], 26)}", key=f"open_{chat_id}", use_container_width=True,
                       type="primary" if active else "secondary"):
        open_chat(chat_id)
        st.rerun()
    if pin_col.button("📌" if not chat["pinned"] else "✖️", key=f"pin_{chat_id}",
                      help="Pin to top" if not chat["pinned"] else "Unpin", use_container_width=True):
        db.set_pinned(chat_id, user_id, not chat["pinned"])
        st.rerun()
    if del_col.button("🗑️", key=f"del_{chat_id}", help="Delete chat", use_container_width=True):
        st.session_state.confirm_delete = chat_id
        st.rerun()


# ====================== START ======================
load_css()
db.init_db()  # creates data/docmind.db and tables on first run
cookies.apply_pending()  # write or clear the login cookie if a login/logout just happened

if st.session_state.get("user") is None:  # new browser tab: maybe the saved cookie still works
    st.session_state.user = auth.user_from_token(cookies.read_token())

if st.session_state.get("user") is None:  # still not logged in -> show login and stop
    render_auth()
    st.stop()

if not config.GROQ_API_KEY:  # no key -> stop early with a clear message
    st.error("**GROQ_API_KEY is missing.** Copy `.env.example` to `.env`, add your key, then restart the app.")
    st.stop()

if "embedder" not in st.session_state:  # models are cached globally, we just keep handy references
    st.session_state.embedder, st.session_state.reranker, st.session_state.llm = load_models()
if "pipeline" not in st.session_state:  # first run after login -> blank chat
    new_chat()

user = st.session_state.user
user_id = user["id"]
pipeline: RAGPipeline = st.session_state.pipeline
chat = db.get_chat(st.session_state.chat_id, user_id) if st.session_state.chat_id else None
chat_title = chat["title"] if chat else DEFAULT_TITLE

# ====================== sidebar ======================
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo">📄</div>'
        '<div><div class="brand-name">DocMind</div><div class="brand-tag">Chat with your documents</div></div></div>',
        unsafe_allow_html=True,
    )
    chip_col, out_col = st.columns([0.68, 0.32], gap="small")
    chip_col.markdown(
        f'<div class="user-chip"><div class="user-avatar">{esc(user["username"][0].upper())}</div>'
        f'<div><div class="user-name">{esc(user["username"])}</div><div class="user-sub">Signed in</div></div></div>',
        unsafe_allow_html=True,
    )
    if out_col.button("Log out", use_container_width=True):
        logout()
        st.rerun()

    st.write("")
    if st.button("＋ New chat", type="primary", use_container_width=True):
        new_chat()
        st.rerun()

    chats = db.list_chats(user_id)
    pinned = [c for c in chats if c["pinned"]]
    recent = [c for c in chats if not c["pinned"]]
    if pinned:
        st.markdown('<div class="side-title">📌 Pinned</div>', unsafe_allow_html=True)
        for item in pinned:
            chat_row(item, item["id"] == st.session_state.chat_id, user_id)
    st.markdown('<div class="side-title">🕘 Recent chats</div>', unsafe_allow_html=True)
    if not recent:
        st.caption("No chats yet. Upload a PDF to start one.")
    for item in recent:
        chat_row(item, item["id"] == st.session_state.chat_id, user_id)

    st.markdown('<div class="side-title">This chat</div>', unsafe_allow_html=True)
    with st.expander("📄 Documents", expanded=not pipeline.is_ready):
        uploads = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True,
                                   label_visibility="collapsed", key=f"upload_{st.session_state.chat_id}")
        add_mode = True
        if pipeline.is_ready:  # chat already has documents -> let the user choose
            add_mode = st.radio("When building", ["Add to this chat", "Replace documents"], horizontal=False) \
                == "Add to this chat"
        label = "⚙️ Build index" if not pipeline.is_ready else "⚙️ Update index"
        if st.button(label, type="primary", use_container_width=True, disabled=not uploads):
            with st.spinner("Reading, chunking and embedding..."):
                try:
                    files = [(f.name, f.getvalue()) for f in uploads]
                    st.session_state.stats = pipeline.ingest(files, append=add_mode)
                    if st.session_state.chat_id is None:  # first save: create the chat now
                        title = Path(files[0][0]).stem[:40] or DEFAULT_TITLE
                        st.session_state.chat_id = db.create_chat(user_id, title)
                    pipeline.save(index_folder(user_id, st.session_state.chat_id))  # keep the index for later
                    st.toast("Index saved. Ask your first question!", icon="✅")
                    st.rerun()
                except ValueError as error:  # friendly message instead of a crash
                    st.error(str(error))

    with st.expander("⚙️ Retrieval settings"):
        use_hybrid = st.toggle("Hybrid search", value=True, help="Combine keyword (BM25) and meaning (vector) search.")
        use_rerank = st.toggle("Re-ranking", value=True, help="A cross-encoder re-orders results for higher accuracy.")
        use_memory = st.toggle("Chat memory", value=True, help="Understand follow-up questions like 'and for gift cards?'")
        top_k = st.slider("Passages sent to the AI", 1, 8, config.FINAL_K)
    mode = ("Hybrid" if use_hybrid else "Vector") + (" + Re-rank" if use_rerank else "")
    st.markdown(f'<div class="mode-note"><b>Active:</b> {mode} · top {top_k}</div>', unsafe_allow_html=True)

    st.write("")
    st.download_button(
        "⬇️ Export this chat",
        data=chat_to_markdown(chat_title, st.session_state.messages),
        file_name="docmind_chat.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    st.markdown('<div class="side-foot">Built with Llama · Inference by Groq</div>', unsafe_allow_html=True)

# ====================== main area ======================
typed = st.chat_input("Ask anything about your documents...", disabled=not pipeline.is_ready)
question = st.session_state.pop("pending_question", None) or typed  # clicked suggestion wins over typed text

head_col, rename_col = st.columns([0.8, 0.2])
head_col.markdown(f'<div class="chat-header"><div class="chat-title">{esc(chat_title)}</div></div>',
                  unsafe_allow_html=True)
if chat:  # rename only makes sense for a saved chat
    with rename_col.popover("✏️ Rename", use_container_width=True):
        new_title = st.text_input("Chat name", value=chat_title, max_chars=60, key=f"rename_{chat['id']}")
        if st.button("Save name", key=f"rename_save_{chat['id']}", type="primary"):
            db.rename_chat(chat["id"], user_id, new_title.strip() or chat_title)
            st.rerun()

if not pipeline.is_ready:  # nothing indexed yet -> show the guide
    st.markdown('<div class="section-title">Get started</div>', unsafe_allow_html=True)
    steps(ready=False)
else:
    st.markdown('<div class="section-title">Knowledge base</div>', unsafe_allow_html=True)
    stat_cards(st.session_state.stats)
    st.markdown('<div class="section-title">Conversation</div>', unsafe_allow_html=True)

previous_user_question = ""
for message in st.session_state.messages:  # redraw history on every rerun
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "📄"):
        st.markdown(message["content"])
        if message["role"] == "user":
            previous_user_question = message["content"]
        elif message.get("sources"):
            render_answer_extras(message["sources"], message["meta"], message.get("searched", ""),
                                 previous_user_question)

if pipeline.is_ready and not st.session_state.messages and not question:  # empty chat -> offer starters
    st.caption("Try one of these:")
    columns = st.columns(len(SUGGESTIONS))
    for column, suggestion in zip(columns, SUGGESTIONS):
        if column.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion  # saved, picked up on the next run
            st.rerun()

if question and st.session_state.chat_id:
    chat_id = st.session_state.chat_id
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] if use_memory else []
    if not st.session_state.messages:  # first question names the chat
        db.rename_chat(chat_id, user_id, shorten(question, 40))
    db.add_message(chat_id, user_id, "user", question)  # save the question right away
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching your documents..."):
            result = pipeline.ask(question, use_hybrid, use_rerank, top_k, history)
        st.markdown(result["answer"])
    meta = f"{result['total_ms'] / 1000:.1f}s total · retrieval {result['retrieve_ms']:.0f} ms · {mode}"
    searched = result["search_query"]
    db.add_message(chat_id, user_id, "assistant", result["answer"], result["sources"], meta, searched)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"],
         "meta": meta, "searched": searched}
    )
    st.rerun()  # redraw so the sidebar (recent order, export button) updates immediately

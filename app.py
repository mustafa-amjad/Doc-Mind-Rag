"""DocMind: Streamlit UI with login, saved chats, pinned chats and chat memory.
Run with:  streamlit run app.py"""
import re  # used to clean whitespace in snippets
import shutil  # deletes a chat's saved index folder
from pathlib import Path  # builds file paths that work on Windows and Mac

import streamlit as st

from docmind import auth, config, cookies, db, ocr
from docmind.embedder import Embedder
from docmind.llm import LLM, friendly_error
from docmind.loader import SUPPORTED_TYPES, fetch_url
from docmind.pipeline import RAGPipeline
from docmind.reranker import Reranker

DEFAULT_TITLE = "New chat"

st.set_page_config(  # must be the FIRST Streamlit call
    page_title="DocMind",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [  # one-click starter questions shown after indexing
    "Summarize the main points",
    "What are the key findings?",
    "List important dates, figures and names",
]
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/description:"
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")  # characters Streamlit markdown would interpret


# ====================== small helpers ======================
def md_escape(value) -> str:
    """Make any text (file names, PDF passages) safe to show inside Streamlit markdown."""
    return _MD_SPECIAL.sub(r"\\\1", str(value))


def shorten(text: str, limit: int = 220) -> str:
    """Collapse whitespace and cut long text at a word boundary, so evidence stays short."""
    text = re.sub(r"\s+", " ", text).strip()  # many spaces/newlines -> one space
    if len(text) <= limit:  # already short, keep as is
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"  # cut, drop the half-word, add dots


@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    """Heavy models load ONCE and are shared by every browser session."""
    return Embedder(config.EMBED_MODEL), Reranker(config.RERANK_MODEL)


def index_folder(user_id: int, chat_id: int) -> Path:
    """Where this chat's chunks + vectors are saved."""
    return config.INDEX_DIR / str(user_id) / str(chat_id)


# ====================== rendering helpers ======================
def unique_pages(sources: list[dict]) -> list[str]:
    """'report.pdf, p. 3' once per file+page, in the order they were cited."""
    seen, labels = set(), []
    for src in sources:
        key = (src["source"], src["page"])
        if key not in seen:
            seen.add(key)
            labels.append(f"{src['source']}, p. {src['page']}")
    return labels


def render_answer_extras(sources: list[dict], meta: str, searched: str = "", question: str = "") -> None:
    """Citations + timing + collapsed evidence. Used for new AND old messages."""
    cited = " · ".join(md_escape(label) for label in unique_pages(sources))
    details = md_escape(meta)
    if searched and searched.strip().lower() != question.strip().lower():  # memory rewrote the question
        details += f"  \nSearched for: “{md_escape(shorten(searched, 120))}”"
    st.caption(f":material/menu_book: {cited}  \n{details}")
    with st.expander(f"Evidence ({len(sources)} passages)", icon=":material/format_quote:"):
        for number, src in enumerate(sources, start=1):
            with st.container(border=True):
                with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                    st.markdown(f"**{number}.** {md_escape(src['source'])}, page {md_escape(src['page'])}")
                    if "rerank_score" in src:
                        st.badge(f"score {src['rerank_score']:.2f}", color="gray",
                                 help="Cross-encoder re-rank score. Higher means more relevant.")
                    if src.get("ocr"):
                        st.badge("OCR", icon=":material/document_scanner:", color="orange",
                                 help="Read from a scanned page; may contain recognition errors.")
                st.caption(md_escape(shorten(src["text"], 1200)))


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


def page_header() -> None:
    """Title block for the sign-in screens."""
    st.space("large")
    st.title("DocMind", anchor=False)
    st.caption("Ask questions about your documents and get answers with page-level citations.")


def getting_started() -> None:
    """Empty state: what to do before the first question."""
    with st.container(border=True):
        st.markdown("**Add documents to this chat**")
        st.caption("Open **Documents** in the sidebar, upload files or paste a link, then build the index.")
        steps = [
            (":material/upload_file:", "Upload", "PDF, Word, text, Markdown, HTML or a public web link."),
            (":material/database:", "Index", "Documents are chunked and embedded once, then saved with the chat."),
            (":material/forum:", "Ask", "Answers cite the file and page. Follow-up questions keep context."),
        ]
        for column, (icon, title, text) in zip(st.columns(3), steps):
            column.markdown(f"{icon} **{title}**")
            column.caption(text)


def knowledge_base(stats: dict) -> None:
    """Indexed document summary for the current chat."""
    cols = st.columns(4)
    cols[0].metric("Files", stats["files"], border=True)
    cols[1].metric("Pages", stats["pages"], border=True)
    cols[2].metric("Chunks", stats["chunks"], border=True)
    cols[3].metric("Index", str(stats["index"]).upper(), border=True)
    names = stats.get("names", [])
    if names:
        st.caption(":material/folder_open: " + " · ".join(md_escape(name) for name in names))


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
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle:
        page_header()
        login_tab, signup_tab, forgot_tab = st.tabs(["Sign in", "Create account", "Reset password"])
        with login_tab:
            with st.form("login_form", border=False):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            if submitted:
                try:
                    user = auth.login(username, password)
                except ValueError as error:  # locked out after too many wrong attempts
                    st.error(str(error))
                    user = None
                else:
                    if user:
                        start_login(user)
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
        with signup_tab:
            with st.form("signup_form", border=False):
                new_user = st.text_input("Username", key="signup_user")
                new_pass = st.text_input("Password (at least 8 characters)", type="password", key="signup_pass")
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                created = st.form_submit_button("Create account", type="primary", width="stretch")
            if created:
                if new_pass != confirm:
                    st.error("Passwords do not match.")
                else:
                    try:
                        created_user = auth.signup(new_user, new_pass)
                        st.session_state.new_recovery_code = created_user.pop("recovery_code")  # shown once, next screen
                        start_login(created_user)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        with forgot_tab:
            with st.form("forgot_form", border=False):
                st.caption("Use the recovery code you saved when you created your account.")
                forgot_user = st.text_input("Username", key="forgot_user")
                forgot_code = st.text_input("Recovery code", key="forgot_code", placeholder="XXXXX-XXXXX-XXXXX-XXXXX")
                forgot_pass = st.text_input("New password (at least 8 characters)", type="password", key="forgot_pass")
                reset_clicked = st.form_submit_button("Reset password", type="primary", width="stretch")
            if reset_clicked:
                try:
                    reset_user = auth.reset_password(forgot_user, forgot_code, forgot_pass)
                    st.session_state.new_recovery_code = reset_user.pop("recovery_code")  # the old code is dead
                    start_login(reset_user)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
        st.caption(":material/lock: Chats and indexed document text are stored on this server under your account.")


def render_recovery_code_screen(code: str) -> None:
    """Shown once after signup / reset / new code. Nothing else works until the user confirms."""
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle:
        page_header()
        st.subheader("Save your recovery code", anchor=False)
        st.warning("This code is the only way to reset a forgotten password, and it is shown once. "
                   "Store it in a password manager.", icon=":material/key:")
        st.code(code, language=None)
        st.download_button("Download as text file", data=f"DocMind recovery code: {code}\n",
                           file_name="docmind_recovery_code.txt", icon=":material/download:", width="stretch")
        if st.button("I have saved it, continue", type="primary", width="stretch"):
            st.session_state.pop("new_recovery_code", None)
            st.rerun()


# ====================== sidebar: chat list ======================
def chat_row(chat: dict, active: bool, user_id: int) -> None:
    """One line in Pinned / Recent: open button, pin button, delete button."""
    chat_id = chat["id"]
    if st.session_state.get("confirm_delete") == chat_id:  # second click: ask before deleting
        st.caption(f"Delete “{md_escape(shorten(chat['title'], 28))}”?")
        yes, no = st.columns(2)
        if yes.button("Delete", key=f"yes_{chat_id}", type="primary", width="stretch"):
            db.delete_chat(chat_id, user_id)
            shutil.rmtree(index_folder(user_id, chat_id), ignore_errors=True)  # remove saved index too
            if st.session_state.chat_id == chat_id:
                new_chat()
            st.session_state.confirm_delete = None
            st.rerun()
        if no.button("Cancel", key=f"no_{chat_id}", width="stretch"):
            st.session_state.confirm_delete = None
            st.rerun()
        return
    with st.container(horizontal=True, vertical_alignment="center", gap="xxsmall", wrap=False):
        if st.button(shorten(chat["title"], 28), key=f"open_{chat_id}", width="stretch",
                     type="secondary" if active else "tertiary"):
            open_chat(chat_id)
            st.rerun()
        with st.popover("", icon=":material/more_vert:", type="tertiary", key=f"menu_{chat_id}"):
            if st.button("Unpin" if chat["pinned"] else "Pin to top", key=f"pin_{chat_id}", type="tertiary",
                         icon=":material/keep_off:" if chat["pinned"] else ":material/keep:"):
                db.set_pinned(chat_id, user_id, not chat["pinned"])
                st.rerun()
            if st.button("Delete", key=f"del_{chat_id}", type="tertiary", icon=":material/delete:"):
                st.session_state.confirm_delete = chat_id
                st.rerun()


# ====================== START ======================
db.init_db()  # creates data/docmind.db and tables on first run
cookies.apply_pending()  # write or clear the login cookie if a login/logout just happened

if st.session_state.get("user") is None:  # new browser tab: maybe the saved cookie still works
    st.session_state.user = auth.user_from_token(cookies.read_token())

if st.session_state.get("user") is None:  # still not logged in -> show login and stop
    render_auth()
    st.stop()

if st.session_state.get("new_recovery_code"):  # must see the code before using the app
    render_recovery_code_screen(st.session_state.new_recovery_code)
    st.stop()

api_key = config.GROQ_API_KEY or st.session_state.get("visitor_groq_key", "")
if not api_key:  # no server key (e.g. public demo) -> let the visitor use their own
    st.info("This demo runs on Groq. Add your own free API key to continue.", icon=":material/key:")
    with st.form("groq_key"):
        entered = st.text_input(
            "Groq API key", type="password",
            help="Get one free at console.groq.com. It stays in this browser session and is never saved.",
        )
        if st.form_submit_button("Continue", type="primary") and entered.strip():
            st.session_state.visitor_groq_key = entered.strip()
            st.rerun()
    st.stop()

if "embedder" not in st.session_state:  # models are cached globally, we just keep handy references
    st.session_state.embedder, st.session_state.reranker = load_models()
    st.session_state.llm = LLM(api_key, config.LLM_MODEL)  # per session: the key may be the visitor's
if "pipeline" not in st.session_state:  # first run after login -> blank chat
    new_chat()

user = st.session_state.user
user_id = user["id"]
pipeline: RAGPipeline = st.session_state.pipeline
chat = db.get_chat(st.session_state.chat_id, user_id) if st.session_state.chat_id else None
chat_title = chat["title"] if chat else DEFAULT_TITLE

# ====================== sidebar ======================
with st.sidebar:
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("#### :material/description: DocMind")
        st.space("stretch")
        if st.button("", icon=":material/logout:", type="tertiary", help="Sign out"):
            logout()
            st.rerun()
    st.caption(f"Signed in as **{md_escape(user['username'])}**")

    if st.button("New chat", icon=":material/add:", type="primary", width="stretch"):
        new_chat()
        st.rerun()

    chats = db.list_chats(user_id)
    pinned = [c for c in chats if c["pinned"]]
    recent = [c for c in chats if not c["pinned"]]
    if pinned:
        st.caption("**Pinned**")
        for item in pinned:
            chat_row(item, item["id"] == st.session_state.chat_id, user_id)
    st.caption("**Recent**")
    if not recent:
        st.caption("No chats yet. Upload a document to start one.")
    for item in recent:
        chat_row(item, item["id"] == st.session_state.chat_id, user_id)

    st.space("small")
    with st.expander("Documents", icon=":material/folder:", expanded=not pipeline.is_ready):
        uploads = st.file_uploader("Files", type=SUPPORTED_TYPES, accept_multiple_files=True,
                                   label_visibility="collapsed", key=f"upload_{st.session_state.chat_id}",
                                   help="PDF, DOCX, TXT, MD or HTML")
        link = st.text_input("Or add a web page / PDF link", placeholder="https://example.com/page",
                             key=f"link_{st.session_state.chat_id}").strip()
        ocr_ready = ocr.is_available()
        ocr_on = st.toggle("OCR for scanned pages", value=ocr_ready, disabled=not ocr_ready,
                           help=f"Reads PDF pages that contain only images. Slower; up to "
                                f"{config.MAX_OCR_PAGES} pages per file.")
        if not ocr_ready:
            st.caption("OCR is not installed. Run `pip install -r requirements-ocr.txt` and restart.")
        add_mode = True
        if pipeline.is_ready:  # chat already has documents -> let the user choose
            add_mode = st.segmented_control("When building", ["Add to chat", "Replace all"], default="Add to chat",
                                            key=f"mode_{st.session_state.chat_id}") != "Replace all"
        label = "Build index" if not pipeline.is_ready else "Update index"
        if st.button(label, icon=":material/database:", type="primary", width="stretch",
                     disabled=not (uploads or link)):
            with st.spinner("Reading and indexing documents..." + (" OCR may take a while." if ocr_on else "")):
                try:
                    files = [(f.name, f.getvalue()) for f in uploads]
                    if link:  # download the link (public sites only)
                        files.append(fetch_url(link))
                    st.session_state.stats = pipeline.ingest(files, append=add_mode, ocr=ocr_on)
                    if st.session_state.chat_id is None:  # first save: create the chat now
                        title = Path(files[0][0]).stem[:40] or DEFAULT_TITLE
                        st.session_state.chat_id = db.create_chat(user_id, title)
                    pipeline.save(index_folder(user_id, st.session_state.chat_id))  # keep the index for later
                    ocr_pages = st.session_state.stats.get("ocr_pages", 0)
                    st.toast("Index saved." + (f" {ocr_pages} scanned pages read with OCR." if ocr_pages else ""),
                             icon=":material/check_circle:")
                    st.rerun()
                except ValueError as error:  # friendly message instead of a crash
                    st.error(str(error), icon=":material/error:")

    with st.expander("Retrieval settings", icon=":material/tune:"):
        use_hybrid = st.toggle("Hybrid search", value=True, help="Combine keyword (BM25) and semantic (vector) search.")
        use_rerank = st.toggle("Re-ranking", value=True, help="Re-order candidates with a cross-encoder for accuracy.")
        use_memory = st.toggle("Chat memory", value=True, help="Resolve follow-up questions using earlier turns.")
        top_k = st.slider("Passages sent to the model", 1, 8, config.FINAL_K)
    mode = ("Hybrid" if use_hybrid else "Vector") + (" + re-rank" if use_rerank else "")

    st.download_button(
        "Export chat",
        data=chat_to_markdown(chat_title, st.session_state.messages),
        file_name="docmind_chat.md",
        mime="text/markdown",
        icon=":material/download:",
        width="stretch",
        disabled=not st.session_state.messages,
    )
    with st.expander("Account", icon=":material/manage_accounts:"):
        with st.form("change_password_form", clear_on_submit=True, border=False):
            st.caption("Change password")
            old_pass = st.text_input("Current password", type="password")
            new_pass_1 = st.text_input("New password (at least 8 characters)", type="password")
            change_clicked = st.form_submit_button("Change password", width="stretch")
        if change_clicked:
            try:
                auth.change_password(user_id, old_pass, new_pass_1)
                start_login(user)  # old logins were removed, so issue a fresh one for this browser
                st.toast("Password changed. Other devices were signed out.", icon=":material/check_circle:")
            except ValueError as error:
                st.error(str(error))
        with st.form("new_code_form", clear_on_submit=True, border=False):
            st.caption("New recovery code (the old one stops working)")
            code_pass = st.text_input("Your password", type="password", key="code_pass")
            code_clicked = st.form_submit_button("Generate new code", width="stretch")
        if code_clicked:
            try:
                st.session_state.new_recovery_code = auth.regenerate_recovery_code(user_id, code_pass)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    st.caption(f"Retrieval: {mode}, top {top_k}  \nModel: {md_escape(config.LLM_MODEL)} via Groq")

# ====================== main area ======================
typed = st.chat_input("Ask a question about your documents", disabled=not pipeline.is_ready)
question = st.session_state.pop("pending_question", None) or typed  # clicked suggestion wins over typed text

with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(chat_title, anchor=False)
    if chat:  # rename only makes sense for a saved chat
        with st.popover("", icon=":material/edit:", type="tertiary", help="Rename chat"):
            new_title = st.text_input("Chat name", value=chat_title, max_chars=60, key=f"rename_{chat['id']}")
            if st.button("Save", key=f"rename_save_{chat['id']}", type="primary"):
                db.rename_chat(chat["id"], user_id, new_title.strip() or chat_title)
                st.rerun()

if not pipeline.is_ready:  # nothing indexed yet -> show the guide
    getting_started()
else:
    knowledge_base(st.session_state.stats)

previous_user_question = ""
for message in st.session_state.messages:  # redraw history on every rerun
    is_user = message["role"] == "user"
    with st.chat_message(message["role"], avatar=USER_AVATAR if is_user else ASSISTANT_AVATAR):
        st.markdown(message["content"])
        if is_user:
            previous_user_question = message["content"]
        elif message.get("sources"):
            render_answer_extras(message["sources"], message["meta"], message.get("searched", ""),
                                 previous_user_question)

if pipeline.is_ready and not st.session_state.messages and not question:  # empty chat -> offer starters
    st.caption("Suggested questions")
    with st.container(horizontal=True, gap="small"):
        for suggestion in SUGGESTIONS:
            if st.button(suggestion, icon=":material/arrow_outward:", icon_position="right"):
                st.session_state.pending_question = suggestion  # saved, picked up on the next run
                st.rerun()

if question and st.session_state.chat_id:
    chat_id = st.session_state.chat_id
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] if use_memory else []
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching your documents..."):
            try:
                result = pipeline.ask(question, use_hybrid, use_rerank, top_k, history)
            except Exception as error:  # API key, rate limit, network... show it, keep the chat usable
                st.error(friendly_error(error), icon=":material/error:")
                st.stop()  # nothing is saved, so the user can simply ask again
        st.markdown(result["answer"])
    if not st.session_state.messages:  # first question names the chat
        db.rename_chat(chat_id, user_id, shorten(question, 40))
    db.add_message(chat_id, user_id, "user", question)
    st.session_state.messages.append({"role": "user", "content": question})
    meta = f"{result['total_ms'] / 1000:.1f} s total · retrieval {result['retrieve_ms']:.0f} ms · {mode}"
    searched = result["search_query"]
    db.add_message(chat_id, user_id, "assistant", result["answer"], result["sources"], meta, searched)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"],
         "meta": meta, "searched": searched}
    )
    st.rerun()  # redraw so the sidebar (recent order, export button) updates immediately

"""DocMind: Streamlit UI with login, saved chats, pinned chats and chat memory.
Run with:  streamlit run app.py"""
import html  # escapes the username in the account row
import re  # used to clean whitespace in snippets
import shutil  # deletes a chat's saved index folder
from pathlib import Path  # builds file paths that work on Windows and Mac

import streamlit as st

from docmind import auth, config, cookies, db, evidence, ocr
from docmind.embedder import Embedder
from docmind.llm import LLM, NOT_FOUND, friendly_error
from docmind.loader import SUPPORTED_TYPES, fetch_url
from docmind.files import how_read
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


def cited_numbers(text: str, sources: list[dict]) -> list[int]:
    """[n] markers the answer actually uses, in order, that point at a real source."""
    found = []
    for match in re.finditer(r"\[(\d+)\]", text):
        number = int(match.group(1))
        if 1 <= number <= len(sources) and number not in found:
            found.append(number)
    return found


def answer_markdown(text: str) -> str:
    """Show [1] citation markers as small emerald badges, like the web app."""
    text = re.sub("[\u00a0\u2007\u202f]", " ", text).replace("\u2011", "-")  # older saved answers: odd spaces/hyphens
    return re.sub(r"\s*\[(\d+)\]", r" :green-badge[\1]", text)


def render_answer(message: dict, index: int, question: str = "") -> None:
    """One saved answer: status badge, text with citation badges, source buttons for the viewer, details."""
    sources, content = message.get("sources", []), message["content"]
    numbers = cited_numbers(content, sources)
    if content.strip() == NOT_FOUND:
        st.badge("Not in your documents", color="gray")
    elif numbers:
        files = len({sources[n - 1]["source"] for n in numbers})
        st.badge(f"Cited from {files} source{'s' if files > 1 else ''}", icon=":material/check:", color="green")
    st.markdown(answer_markdown(content))
    if numbers:
        with st.container(horizontal=True, gap="small"):
            for n in numbers:
                src = sources[n - 1]
                active = st.session_state.get("viewer") == (index, n)
                if st.button(f"{n} · {src['source']} · p.{src['page']}", key=f"cite_{index}_{n}",
                             type="primary" if active else "secondary", icon=":material/description:"):
                    st.session_state.viewer = (index, n)
                    st.rerun()
    if sources:
        render_answer_extras(sources, message.get("meta", ""), message.get("searched", ""), question)


@st.cache_data(show_spinner=False, max_entries=64)
def highlighted_page(path: str, modified: float, page: int, sentences: tuple) -> tuple:
    """A PDF page as an image with the cited sentences highlighted. Returns (image, found_any)."""
    import pypdfium2
    from PIL import Image, ImageDraw

    data = Path(path).read_bytes()
    pdf = pypdfium2.PdfDocument(data)
    try:
        image = pdf[page - 1].render(scale=2).to_pil().convert("RGBA")
    finally:
        pdf.close()
    boxes = evidence.pdf_highlights(data, page, list(sentences))["boxes"] if sentences else []
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for box in boxes:  # PDF points -> pixels (rendered at 2x)
        draw.rounded_rectangle([(box["x0"] - 2) * 2, (box["top"] - 2) * 2, (box["x1"] + 2) * 2, (box["bottom"] + 2) * 2],
                               radius=4, fill=(247, 213, 74, 110))
    return Image.alpha_composite(image, overlay).convert("RGB"), bool(boxes)


def readable_passage(text: str, sentences: list[str]) -> str:
    """Passage as markdown with the cited sentences highlighted; index tags shown as plain labels."""
    def tidy(value: str) -> str:
        value = value.replace("[Image description]", "AI description of the picture:")
        return value.replace("[Image]", "Picture (text read from it):").replace("[Table]", "Table columns:")
    marked = md_escape(tidy(text)).replace("\n", "  \n")
    for sentence in sentences:
        escaped = md_escape(tidy(sentence))
        if escaped and escaped in marked:
            marked = marked.replace(escaped, f":orange-background[{escaped}]", 1)
    return marked


def render_viewer(messages: list[dict], user_id: int, chat_id: int | None) -> None:
    """Right-hand source viewer: the page behind the selected citation, with the sentences highlighted."""
    choice = st.session_state.get("viewer")
    if not choice or choice[0] >= len(messages) or messages[choice[0]]["role"] != "assistant":
        choice = None
        for index in range(len(messages) - 1, -1, -1):  # default: newest answer that cites something
            message = messages[index]
            if message["role"] == "assistant":
                numbers = cited_numbers(message["content"], message.get("sources", []))
                if numbers:
                    choice = (index, numbers[0])
                    break
    with st.container(border=True, key="viewer"):
        if not choice:
            st.markdown("#### Source")
            st.caption("Pick a numbered source under an answer to see the page it came from.")
            return
        message = messages[choice[0]]
        source = message["sources"][choice[1] - 1]
        with st.container(key="viewer_head"):
            st.markdown(f":green-badge[{choice[1]}] **{md_escape(source['source'])}**")
            notes = [f"Page {source['page']}"] + (["read with OCR"] if source.get("ocr") else []) \
                + (["includes a picture"] if source.get("image") else [])
            st.caption(" · ".join(notes))
        sentences = evidence.pick_sentences(source["text"], message["content"])
        stored = index_folder(user_id, chat_id) / "files" / Path(source["source"]).name if chat_id else None
        shown_page = False
        if stored and stored.exists() and stored.suffix.lower() == ".pdf":
            try:
                image, found = highlighted_page(str(stored), stored.stat().st_mtime, int(source["page"]), tuple(sentences))
                st.image(image, width="stretch")
                shown_page = True
                if not found and sentences:  # scanned page: show the OCR text with the sentence marked
                    st.caption("Text read from this page" + (" with OCR" if source.get("ocr") else ""))
                    st.markdown(readable_passage(source["text"], sentences))
            except Exception:
                shown_page = False
        if not shown_page:
            st.markdown(readable_passage(source["text"], sentences))


def render_answer_extras(sources: list[dict], meta: str, searched: str = "", question: str = "") -> None:
    """Timing, the rewritten search and every passage used (collapsed)."""
    details = md_escape(meta)
    if searched and searched.strip().lower() != question.strip().lower():  # memory rewrote the question
        details += f"  \nSearched for: “{md_escape(shorten(searched, 120))}”"
    st.caption(details)
    with st.expander(f"Passages used ({len(sources)})", icon=":material/format_quote:"):
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
                    if src.get("image"):
                        st.badge("Picture", icon=":material/image:", color="violet",
                                 help="This page includes text read from a picture (OCR) or an AI description "
                                      "of a chart or diagram. Check the original for exact values.")
                st.caption(md_escape(shorten(src["text"], 1200)))


def model_label() -> str:
    """'gpt-oss-120b' for 'openai/gpt-oss-120b'."""
    llm = st.session_state.get("llm")
    return (llm.model if llm else config.LLM_MODEL).split("/")[-1]


@st.cache_data(ttl=86400, show_spinner=False)
def context_window(model: str, _client) -> int | None:
    """The model's context window in tokens, as Groq reports it (None if Groq doesn't say)."""
    try:
        return int(getattr(_client.models.retrieve(model), "context_window", 0) or 0) or None
    except Exception:
        return None


def file_icon(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return ":material/picture_as_pdf:"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return ":material/image:"
    if suffix in {".html", ".htm"}:
        return ":material/language:"
    return ":material/description:"


BOX_FILES = ":small[:gray[:material/folder_open: In this chat]]"
NEW_LINE = "  \n"  # markdown line break: label on top, value under it, in one block


def render_overview(pipeline: RAGPipeline, chat_id: int | None, mode: str, top_k: int, memory: bool) -> None:
    """Three boxes under the header: what this chat contains, how full the context window is, which model answers."""
    llm = st.session_state.llm
    with st.container(horizontal=True, gap="small", key="overview"):
        with st.container(border=True, key="box_files"):
            if pipeline.is_ready:
                stats = st.session_state.stats or pipeline.stats()
                files = stats["files"]
                st.markdown(BOX_FILES + NEW_LINE + f"**{files} file{'s' if files != 1 else ''}** · "
                            f"{stats['pages']} pages · {stats['chunks']} passages")
                with st.container(horizontal=True, gap="xxsmall", key="file_chips"):
                    for name in pipeline.file_names[:4]:
                        st.badge(shorten(name, 26), icon=file_icon(name), color="gray")
                    if len(pipeline.file_names) > 4:
                        st.badge(f"+{len(pipeline.file_names) - 4} more", color="gray")
            else:
                st.markdown(BOX_FILES + NEW_LINE + "**No files yet**")
                st.caption("Add files from the sidebar.")
        with st.container(border=True, key="box_context"):
            window = context_window(llm.model, llm.client)
            used = (st.session_state.get("context_usage") or {}).get(chat_id)
            tokens = used["prompt_tokens"] if used else 0
            st.markdown(":small[:gray[:material/memory: Context window]]" + NEW_LINE
                        + f"**{tokens:,}**" + (f" of {window:,} tokens" if window else " tokens"))
            if window:
                st.progress(min(tokens / window, 1.0))
            st.caption(f"Last question: rules, {used['passages']} passages and recent chat" if used
                       else "Fills as you ask: the rules, the passages found and recent chat")
        with st.container(border=True, key="box_model"):
            st.markdown(":small[:gray[:material/neurology: Model]]" + NEW_LINE + f"**{md_escape(model_label())}** · Groq")
            st.caption(f"{mode} · {top_k} passages per answer · memory {'on' if memory else 'off'}")


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


def page_header(title: str = "Sign in to DocMind", note: str = "") -> None:
    """Title block for the sign-in screens."""
    st.space("large")
    st.title(title, anchor=False)
    if note:
        st.caption(note)


def getting_started() -> None:
    """Empty state: three numbered steps in one panel, divided by lines."""
    steps = [("1", "Add files", "Use Add files in the sidebar: PDFs, Word files, images or a link."),
             ("2", "Ask a question", "Type in the box below. Follow-up questions keep their context."),
             ("3", "Check the source", "Each answer opens the page it came from, with the sentence highlighted.")]
    with st.container(horizontal=True, gap=None, key="steps"):
        for number, title, text in steps:
            with st.container(key=f"step_{number}"):
                st.html(f'<span class="step-num">{number}</span>')
                st.markdown(f"**{title}**")
                st.caption(text)


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
    st.session_state.ingest_warnings = []
    st.session_state.viewer = None
    st.session_state.want_tab = "Documents"  # a new chat starts by adding files


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
    st.session_state.ingest_warnings = []
    st.session_state.viewer = None
    st.session_state.want_tab = "Chat" if st.session_state.stats else "Documents"


def start_login(user: dict) -> None:
    """Log in for this tab AND remember the user in a browser cookie (survives refresh)."""
    st.session_state.user = user
    cookies.queue_set(auth.create_session(user["id"]))


def logout() -> None:
    auth.end_session(cookies.read_token())  # kill the saved login so the old cookie stops working
    cookies.queue_clear()  # and remove the cookie from the browser
    for key in ("user", "chat_id", "messages", "pipeline", "stats", "confirm_delete", "pending_question", "viewer", "prefs"):
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
        st.caption("Your chats and files are private to your account.")


def render_recovery_code_screen(code: str) -> None:
    """Shown once after signup / reset / new code. Nothing else works until the user confirms."""
    _, middle, _ = st.columns([1, 1.3, 1])
    with middle:
        page_header("Save your recovery code",
                    "You need this code to reset a forgotten password. It is shown only once, "
                    "so keep it in a password manager.")
        st.code(code, language=None)
        st.download_button("Download as text file", data=f"DocMind recovery code: {code}\n",
                           file_name="docmind_recovery_code.txt", icon=":material/download:", width="stretch")
        if st.button("I've saved it", type="primary", width="stretch"):
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
    with st.container(horizontal=True, vertical_alignment="center", gap="xxsmall", wrap=False, key=f"row_{chat_id}" + ("_on" if active else "")):
        if st.button(shorten(chat["title"], 28), key=f"open_{chat_id}", width="stretch",
                     type="secondary" if active else "tertiary"):
            open_chat(chat_id)
            st.rerun()
        with st.popover("", icon=":material/more_horiz:", type="tertiary", key=f"menu_{chat_id}"):
            if st.button("Unpin" if chat["pinned"] else "Pin to top", key=f"pin_{chat_id}", type="tertiary",
                         icon=":material/keep_off:" if chat["pinned"] else ":material/keep:"):
                db.set_pinned(chat_id, user_id, not chat["pinned"])
                st.rerun()
            if st.button("Delete", key=f"del_{chat_id}", type="tertiary", icon=":material/delete:"):
                st.session_state.confirm_delete = chat_id
                st.rerun()


def file_rows(chunks: list[dict]) -> list[dict]:
    """One row per indexed file: pages, passages and how it was read."""
    rows = {}
    for chunk in chunks:
        row = rows.setdefault(chunk["source"], {"File": chunk["source"], "pages": set(), "Passages": 0})
        row["pages"].add(chunk["page"])
        row["Passages"] += 1
    return [{"File": r["File"], "Pages": len(r["pages"]), "Passages": r["Passages"],
             "Read as": how_read(chunks, r["File"])} for r in rows.values()]


def add_files_panel(pipeline: RAGPipeline, user_id: int) -> None:
    """Upload, link and reading options, then index into this chat (shown in the sidebar)."""
    with st.container(key="upload"):
        uploads = st.file_uploader("Files", type=SUPPORTED_TYPES, accept_multiple_files=True,
                                   label_visibility="collapsed", key=f"upload_{st.session_state.chat_id}",
                                   help="PDF, DOCX, TXT, MD, HTML or an image (PNG, JPG, WEBP, BMP, GIF, TIFF)")
        link = st.text_input("Link", placeholder="Paste a web page or PDF link",
                             key=f"link_{st.session_state.chat_id}").strip()
        ocr_ready = ocr.is_available()
        ocr_on = st.toggle("Read scans and images", value=ocr_ready, disabled=not ocr_ready,
                           help=f"Reads scanned PDF pages, image files, and the words inside charts and screenshots "
                                f"in PDF and Word files. Slower; up to {config.MAX_OCR_PAGES} scanned pages and "
                                f"{config.MAX_IMAGES_PER_FILE} pictures per file.")
        if not ocr_ready:
            st.caption("OCR is not installed. Run `pip install -r requirements.txt` and restart.")
        describe_on = st.toggle("Describe charts with AI", value=False,
                                help=f"Sends each picture to the Groq vision model ({config.VISION_MODEL}) and "
                                     f"indexes its description, so questions about charts and diagrams can be "
                                     f"answered. Up to {config.MAX_VISION_IMAGES} pictures per file; uses API quota.")
        add_mode = True
        if pipeline.is_ready:  # chat already has documents -> let the user choose
            add_mode = st.segmented_control("New files", ["Add to these", "Replace these"], default="Add to these",
                                            key=f"mode_{st.session_state.chat_id}") != "Replace these"
        if st.button("Add to chat", type="primary", width="stretch", disabled=not (uploads or link)):
            slow = " Scans and images take a little longer." if (ocr_on or describe_on) else ""
            with st.spinner("Reading your files..." + slow):
                try:
                    files = [(f.name, f.getvalue()) for f in uploads]
                    if link:  # download the link (public sites only)
                        files.append(fetch_url(link))
                    st.session_state.stats = pipeline.ingest(files, append=add_mode, ocr=ocr_on,
                                                             describe_images=describe_on)
                    if st.session_state.chat_id is None:  # first save: create the chat now
                        title = Path(files[0][0]).stem[:40] or DEFAULT_TITLE
                        st.session_state.chat_id = db.create_chat(user_id, title)
                    saved = index_folder(user_id, st.session_state.chat_id)
                    pipeline.save(saved)  # keep the index for later
                    (saved / "files").mkdir(parents=True, exist_ok=True)
                    for name, data in files:  # originals, so the source viewer can show the real page
                        (saved / "files" / Path(name).name).write_bytes(data)
                    ocr_pages = st.session_state.stats.get("ocr_pages", 0)
                    image_pages = st.session_state.stats.get("image_pages", 0)
                    message = "Files ready."
                    if ocr_pages:
                        message += f" {ocr_pages} scanned page{'s' if ocr_pages != 1 else ''} read."
                    if image_pages:
                        message += f" Text found in pictures on {image_pages} page{'s' if image_pages != 1 else ''}."
                    st.session_state.flash = message
                    st.session_state.ingest_warnings = st.session_state.stats.get("warnings", [])
                    st.rerun()
                except ValueError as error:  # friendly message instead of a crash
                    st.error(str(error), icon=":material/error:")
        for warning in st.session_state.get("ingest_warnings", []):  # e.g. vision model unavailable
            st.warning(warning, icon=":material/warning:")


def render_documents(pipeline: RAGPipeline, user_id: int) -> None:
    """Documents tab: what this chat searches. Files are added from the sidebar."""
    with st.container():
        stats = st.session_state.stats or {"files": 0, "pages": 0, "chunks": 0}
        scanned = len({(c["source"], c["page"]) for c in pipeline.chunks if c.get("ocr")}) if pipeline.is_ready else 0
        with st.container(horizontal=True, gap="small", key="stats"):
            st.metric("Files", stats["files"], border=True)
            st.metric("Pages", stats["pages"], border=True)
            st.metric("Passages", stats["chunks"], border=True)
            st.metric("Scanned pages", scanned, border=True)
        if pipeline.is_ready:
            st.dataframe(
                file_rows(pipeline.chunks), hide_index=True, width="stretch",
                column_config={
                    "File": st.column_config.TextColumn("File", width="medium"),
                    "Pages": st.column_config.NumberColumn("Pages", width="small"),
                    "Passages": st.column_config.NumberColumn("Passages", width="small",
                                                              help="Searchable pieces the file was split into"),
                    "Read as": st.column_config.ListColumn("Read as", help="Text, Table, Picture or OCR (scanned)"),
                },
            )
        else:
            with st.container(border=True, key="no_files"):
                st.markdown("**No files in this chat yet**")
                st.caption("Add files from the sidebar. DocMind reads PDFs, Word files, text, web pages and images, "
                           "including scanned pages.")


def delete_all_chats(user_id: int) -> int:
    """Delete every chat, its messages, its index and its stored files. Returns how many."""
    chats = db.list_chats(user_id, limit=100_000)
    for item in chats:
        db.delete_chat(item["id"], user_id)
        shutil.rmtree(index_folder(user_id, item["id"]), ignore_errors=True)
    return len(chats)


DEFAULT_PREFS = {"hybrid": True, "rerank": True, "memory": True, "top_k": config.FINAL_K}


@st.dialog("Settings", width="medium")
def settings_dialog() -> None:
    """Account, security and privacy live here, away from the chat."""
    user = st.session_state.user
    user_id = user["id"]
    prefs = st.session_state.prefs
    general, account, security, privacy = st.tabs(["General", "Account", "Security", "Privacy"])

    with general:
        st.markdown("**Search**")
        prefs["hybrid"] = st.toggle("Hybrid search", value=prefs["hybrid"], key="pref_hybrid",
                                    help="Combines keyword (BM25) and meaning (vector) search.")
        prefs["rerank"] = st.toggle("Re-rank results", value=prefs["rerank"], key="pref_rerank",
                                    help="A cross-encoder puts the most relevant passage first.")
        prefs["memory"] = st.toggle("Chat memory", value=prefs["memory"], key="pref_memory",
                                    help="Follow-up questions use the earlier ones for context.")
        prefs["top_k"] = st.slider("Passages per answer", 1, 8, prefs["top_k"], key="pref_top_k")
        st.caption(f"Answers are written by {md_escape(st.session_state.llm.model)} on Groq.")

    with account:
        st.text_input("Username", value=user["username"], disabled=True)
        st.caption("Usernames can't be changed.")
        if st.button("Sign out", icon=":material/logout:"):
            logout()
            st.rerun()

    with security:
        st.markdown("**Change password**")
        with st.form("change_password_form", clear_on_submit=True, border=False):
            old_pass = st.text_input("Current password", type="password")
            new_pass = st.text_input("New password", type="password", help="At least 8 characters.")
            change_clicked = st.form_submit_button("Update password")
        if change_clicked:
            try:
                auth.change_password(user_id, old_pass, new_pass)
                start_login(user)  # old sign-ins were removed, so issue a fresh one for this browser
                st.session_state.flash = "Password updated. Other devices were signed out."
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.divider()
        st.markdown("**Recovery code**")
        st.caption("Lets you reset a forgotten password. A new code replaces the old one.")
        with st.form("new_code_form", clear_on_submit=True, border=False):
            code_pass = st.text_input("Password", type="password", key="code_pass")
            code_clicked = st.form_submit_button("Create new code")
        if code_clicked:
            try:
                st.session_state.new_recovery_code = auth.regenerate_recovery_code(user_id, code_pass)
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.divider()
        st.markdown("**Sessions**")
        st.caption("Sign out everywhere except this browser.")
        if st.button("Sign out of other devices"):
            db.delete_user_sessions(user_id)
            start_login(user)
            st.session_state.flash = "Signed out of all other devices."
            st.rerun()

    with privacy:
        st.markdown("**What DocMind stores**")
        st.caption("Your username, a salted hash of your password, your chats, and the files you upload "
                   "with their search index. Only your account can open them. To write an answer, your "
                   "question and the matching passages are sent to Groq.")
        st.divider()
        st.markdown("**Delete all chats**")
        st.caption("Removes every chat, its messages and its uploaded files. This can't be undone.")
        sure = st.checkbox("I understand", key="confirm_wipe")
        if st.button("Delete all chats", type="primary", disabled=not sure):
            count = delete_all_chats(user_id)
            new_chat()
            st.session_state.flash = f"Deleted {count} chat{'s' if count != 1 else ''}."
            st.rerun()


# ====================== START ======================
db.init_db()  # creates data/docmind.db and tables on first run
saved_token = cookies.sync()  # saves/clears the sign-in token in the browser, returns the saved one

if st.session_state.get("user") is None and saved_token:  # new tab or refresh: the saved sign-in still works?
    st.session_state.user = auth.user_from_token(saved_token)

if st.session_state.get("user") is None:  # still not signed in
    if saved_token is None and not st.session_state.get("_waited_for_browser"):
        st.session_state._waited_for_browser = True  # the browser answers in a moment; avoid flashing the form
        st.caption("Loading...")
        st.stop()
    render_auth()
    st.stop()

if st.session_state.get("new_recovery_code"):  # must see the code before using the app
    render_recovery_code_screen(st.session_state.new_recovery_code)
    st.stop()

def secret_groq_key() -> str:
    """GROQ_API_KEY from Streamlit secrets, read on every run so a newly saved secret works without a restart."""
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "")
    except Exception:  # no secrets file (local runs use .env)
        return ""


api_key = config.GROQ_API_KEY or secret_groq_key() or st.session_state.get("visitor_groq_key", "")
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
if "prefs" not in st.session_state:
    st.session_state.prefs = dict(DEFAULT_PREFS)
prefs = st.session_state.prefs
use_hybrid, use_rerank, use_memory, top_k = prefs["hybrid"], prefs["rerank"], prefs["memory"], prefs["top_k"]
mode = ("Hybrid" if use_hybrid else "Vector") + (" + re-rank" if use_rerank else "")
flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash, icon=":material/check_circle:")

CSS = (Path(__file__).parent / "assets" / "docmind.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)  # structure: header bar, panels, lines, motion

with st.sidebar:
    st.html('<div class="brand"><span class="mark">D</span><span>DocMind</span></div>')
    if st.button("New chat", icon=":material/edit_square:", width="stretch", key="new_chat"):
        new_chat()
        st.rerun()

    with st.expander("Add files", icon=":material/upload_file:", expanded=not pipeline.is_ready,
                     key=f"add_files_{st.session_state.chat_id}"):
        add_files_panel(pipeline, user_id)

    chats = db.list_chats(user_id)
    pinned = [c for c in chats if c["pinned"]]
    recent = [c for c in chats if not c["pinned"]]
    with st.container(key="chat_list"):
        if pinned:
            st.html('<p class="side-label">Pinned</p>')
            for item in pinned:
                chat_row(item, item["id"] == st.session_state.chat_id, user_id)
        st.html('<p class="side-label">Recents</p>')
        if not recent and not pinned:
            st.caption("Your chats will appear here.")
        for item in recent:
            chat_row(item, item["id"] == st.session_state.chat_id, user_id)

    with st.container(horizontal=True, vertical_alignment="center", gap="small", key="account"):
        name = user["username"]
        st.html(f'<div class="who"><span class="avatar">{html.escape(name[:1].upper())}</span>'
                f'<span class="who-text"><b>{html.escape(name)}</b><small>{html.escape(model_label())}</small></span></div>')
        if st.button("", icon=":material/settings:", type="tertiary", key="open_settings", help="Settings"):
            settings_dialog()

# ====================== main area ======================
typed = st.chat_input("Ask a question about your documents", disabled=not pipeline.is_ready)
question = st.session_state.pop("pending_question", None) or typed  # clicked suggestion wins over typed text
if question:
    st.session_state.main_tab = "Chat"  # answers appear in the Chat tab, so open it

with st.container(horizontal=True, vertical_alignment="center", key="topbar"):
    st.markdown(f"#### {md_escape(chat_title)}")
    if pipeline.is_ready:
        files = st.session_state.stats["files"]
        st.badge(f"{files} file{'s' if files != 1 else ''} ready", icon=":material/check_circle:", color="green")
    else:
        st.badge("No files yet", color="gray")
    st.space("stretch")
    if chat:  # rename only makes sense for a saved chat
        with st.popover("", icon=":material/edit:", type="tertiary", help="Rename chat"):
            new_title = st.text_input("Chat name", value=chat_title, max_chars=60, key=f"rename_{chat['id']}")
            if st.button("Save", key=f"rename_save_{chat['id']}", type="primary"):
                db.rename_chat(chat["id"], user_id, new_title.strip() or chat_title)
                st.rerun()
    if st.session_state.messages:
        st.download_button("Export", data=chat_to_markdown(chat_title, st.session_state.messages),
                           file_name="docmind_chat.md", mime="text/markdown", icon=":material/download:",
                           type="tertiary")

render_overview(pipeline, st.session_state.chat_id, mode, top_k, use_memory)

if "want_tab" in st.session_state:  # set by new_chat / open_chat; applied before the tabs are drawn
    st.session_state.main_tab = st.session_state.pop("want_tab")
if "main_tab" not in st.session_state:
    st.session_state.main_tab = "Chat" if pipeline.is_ready else "Documents"
chat_tab, docs_tab = st.tabs(["Chat", "Documents"], key="main_tab", on_change="rerun")

with docs_tab:
    if docs_tab.open:
        render_documents(pipeline, user_id)

messages = st.session_state.messages
chat_area = viewer_area = None
if chat_tab.open:
    with chat_tab:
        if not pipeline.is_ready:  # nothing indexed yet -> show the guide
            getting_started()
        show_viewer = pipeline.is_ready and any(m["role"] == "assistant" for m in messages)
        if show_viewer:  # the conversation on the left, the cited page on the right
            chat_area, viewer_area = st.columns([3, 2], gap="large")
        else:
            chat_area = st.container()

        with chat_area:
            if pipeline.is_ready and len(pipeline.file_names) > 1:  # metadata filter: search only some files
                st.multiselect("Search in", pipeline.file_names, key=f"scope_{st.session_state.chat_id}",
                               placeholder="All documents in this chat",
                               help="Pick files to search only those. Leave empty to search every file in this chat.")
            previous_user_question = ""
            for index, message in enumerate(messages):  # redraw history on every rerun
                is_user = message["role"] == "user"
                with st.chat_message(message["role"], avatar=USER_AVATAR if is_user else ASSISTANT_AVATAR):
                    if is_user:
                        st.markdown(f"**{md_escape(message['content'])}**")
                        previous_user_question = message["content"]
                    else:
                        render_answer(message, index, previous_user_question)

            if pipeline.is_ready and not messages and not question:  # empty chat -> offer starters
                with st.container(key="starters"):
                    st.caption("Try asking")
                    with st.container(horizontal=True, gap="small"):
                        for suggestion in SUGGESTIONS:
                            if st.button(suggestion, icon=":material/arrow_outward:", icon_position="right"):
                                st.session_state.pending_question = suggestion  # picked up on the next run
                                st.rerun()

        if viewer_area is not None:
            with viewer_area:
                render_viewer(messages, user_id, st.session_state.chat_id)

if question and st.session_state.chat_id and chat_area is not None:
    chat_id = st.session_state.chat_id
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] if use_memory else []
    scope = [name for name in st.session_state.get(f"scope_{chat_id}") or [] if name in pipeline.file_names]
    scope_kwargs = {"sources": scope} if scope else {}  # only passed when the user narrowed the search
    with chat_area:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(f"**{md_escape(question)}**")
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Searching your files..."):
                try:
                    result = pipeline.ask(question, use_hybrid, use_rerank, top_k, history, **scope_kwargs)
                except Exception as error:  # API key, rate limit, network... show it, keep the chat usable
                    st.error(friendly_error(error), icon=":material/error:")
                    st.stop()  # nothing is saved, so the user can simply ask again
    if not st.session_state.messages:  # first question names the chat
        db.rename_chat(chat_id, user_id, shorten(question, 40))
    db.add_message(chat_id, user_id, "user", question)
    st.session_state.messages.append({"role": "user", "content": question})
    meta = f"{result['total_ms'] / 1000:.1f} s total · retrieval {result['retrieve_ms']:.0f} ms · {mode}"
    if scope:
        meta += f" · searched {len(scope)} of {len(pipeline.file_names)} files"
    searched = result["search_query"]
    db.add_message(chat_id, user_id, "assistant", result["answer"], result["sources"], meta, searched)
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"],
         "meta": meta, "searched": searched}
    )
    usage = getattr(st.session_state.llm, "last_usage", None)  # tokens the model saw for this answer
    if usage:
        st.session_state.setdefault("context_usage", {})[chat_id] = {**usage, "passages": len(result["sources"])}
    first = cited_numbers(result["answer"], result["sources"])
    st.session_state.viewer = (len(st.session_state.messages) - 1, first[0]) if first else None  # show the new source
    st.rerun()  # redraw so the sidebar (recent order, export button) updates immediately

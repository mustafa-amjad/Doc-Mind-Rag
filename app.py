"""DocMind: Streamlit UI with login, saved chats, pinned chats and chat memory.
Layout: chats on the left, the conversation in the middle, the chat's documents on the right.
Run with:  streamlit run app.py"""
import base64  # embeds the logo's lens and spark as images
import html  # escapes the username in the account row
import re  # used to clean whitespace in snippets
import shutil  # deletes a chat's saved index folder
from pathlib import Path  # builds file paths that work on Windows and Mac

import streamlit as st

from docmind import auth, config, cookies, db, evidence, ocr, voice
from docmind.embedder import Embedder
from docmind.llm import LLM, NOT_FOUND, friendly_error
from docmind.loader import SUPPORTED_TYPES, fetch_url
from docmind.files import how_read
from docmind.pipeline import RAGPipeline
from docmind.reranker import Reranker

DEFAULT_TITLE = "New chat"
PLAN = "Free plan"  # every DocMind account is free; there are no paid plans

st.set_page_config(  # must be the FIRST Streamlit call
    page_title="DocMind",
    page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),  # the DocMind mark
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
SPOKEN = ":small[:gray[:material/mic: Spoken · transcribed by Whisper]]"
# The DocMind wordmark: "Doc" speaks from a chat bubble and its "o" is a magnifying glass that finds the line;
# the "i" in Mind carries an idea spark. Built from the app's own font, so it stays sharp at any size.
# st.html drops inline <svg>, so the lens and the spark are small SVG images (data URIs).
def _svg_image(css_class: str, view_box: str, drawing: str) -> str:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">{drawing}</svg>'
    return f'<img class="{css_class}" alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


_LENS_O = _svg_image("dm-lens", "0 0 26 26",
                     '<circle cx="10" cy="10" r="8.2" fill="none" stroke="#FAFAF7" stroke-width="3.8"/>'
                     '<rect x="5.2" y="8.4" width="9.6" height="3.2" rx="1.6" fill="#E8692B"/>'
                     '<path d="M16.4 16.4 22.5 22.5" stroke="#E8692B" stroke-width="4.2" stroke-linecap="round"/>')
_SPARK_I = '<span class="dm-i">ı' + _svg_image(
    "dm-spark", "0 0 10 10", '<path d="M5 0C5.6 3.4 6.6 4.4 10 5 6.6 5.6 5.6 6.6 5 10 4.4 6.6 3.4 5.6 0 5 3.4 4.4 4.4 '
                             '3.4 5 0z" fill="#E8692B"/>') + '</span>'
WORDMARK = (f'<span class="dm-word" role="img" aria-label="DocMind"><span class="dm-bubble">D{_LENS_O}c</span>'
            f'<span class="dm-mind">M{_SPARK_I}nd</span></span>')

# Reads answers aloud with the browser's own speech voice (no API call). Each request has an id, so an
# answer is spoken once, not again on every rerun; "Listen" sends a new id to hear it again.
SPEAKER = st.components.v2.component(
    "docmind_speaker",
    html="<span hidden></span>",
    js="""
export default function (component) {
  const { data } = component
  const synth = window.speechSynthesis
  if (!synth || !data || !data.id || !data.text) return
  let last = ""
  try { last = sessionStorage.getItem("docmind_spoken") || "" } catch (error) {}
  if (last === data.id) return
  try { sessionStorage.setItem("docmind_spoken", data.id) } catch (error) {}
  synth.cancel()
  const say = () => {
    const line = new SpeechSynthesisUtterance(data.text)
    const voices = synth.getVoices()
    const english = voices.filter(v => /^en/i.test(v.lang))
    line.voice = english.find(v => /natural|online|google|samantha|aria|jenny/i.test(v.name)) || english[0] || null
    line.lang = line.voice ? line.voice.lang : "en-US"
    synth.speak(line)
  }
  if (synth.getVoices().length) say(); else synth.addEventListener("voiceschanged", say, { once: true })
}
""",
)


def queue_speech(index: int, text: str, again: bool = False) -> None:
    """Read this answer aloud on the next draw of the speaker."""
    count = st.session_state.get("speech_count", 0) + (1 if again else 0)
    st.session_state.speech_count = count
    st.session_state.speak = {"id": f"{st.session_state.chat_id}-{index}-{count}", "text": voice.speech_text(text)}
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


# ====================== answers ======================
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
    text = re.sub("[   ]", " ", text).replace("‑", "-")  # older saved answers: odd spaces/hyphens
    return re.sub(r"\s*\[(\d+)\]", r" :green-badge[\1]", text)


def render_answer(message: dict, index: int, question: str = "") -> None:
    """One saved answer: status badge, text with citation badges, source buttons (open the page), details."""
    sources, content = message.get("sources", []), message["content"]
    numbers = cited_numbers(content, sources)
    if content.strip() == NOT_FOUND:
        st.badge("Not in your documents", color="gray")
    elif numbers:
        files = len({sources[n - 1]["source"] for n in numbers})
        st.badge(f"Cited from {files} source{'s' if files > 1 else ''}", icon=":material/check:", color="green")
    st.markdown(answer_markdown(content))
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        for n in numbers:
            src = sources[n - 1]
            if st.button(f"{n} · {src['source']} · p.{src['page']}", key=f"cite_{index}_{n}",
                         icon=":material/description:", help="Open the page with the cited sentence highlighted"):
                source_dialog(index, n)
        if st.button("Listen", key=f"listen_{index}", type="tertiary", icon=":material/volume_up:",
                     help="Read this answer aloud"):
            queue_speech(index, content, again=True)
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


def source_page(message: dict, number: int, user_id: int, chat_id: int | None) -> None:
    """The page behind one citation, with the cited sentences highlighted."""
    source = message["sources"][number - 1]
    notes = [f"Page {source['page']}"] + (["read with OCR"] if source.get("ocr") else []) \
        + (["includes a picture"] if source.get("image") else [])
    st.markdown(f":green-badge[{number}] **{md_escape(source['source'])}**")
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


@st.dialog("Source", width="large")
def source_dialog(message_index: int, number: int) -> None:
    """Opens from a source button under an answer: the real page, the cited sentence highlighted."""
    messages = st.session_state.messages
    if message_index < len(messages) and messages[message_index]["role"] == "assistant":
        source_page(messages[message_index], number, st.session_state.user["id"], st.session_state.chat_id)


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


# ====================== models (picker under the message box) ======================
@st.cache_data(ttl=3600, show_spinner=False)
def answer_models(_client) -> list[dict]:
    """Groq models that can write DocMind's answers, with their context windows (live from Groq).
    Speech, transcription, safety-filter and tiny-window models are left out."""
    try:
        listed = _client.models.list().data
    except Exception:
        return []
    models = []
    for model in listed:
        info = model.model_dump()
        window = int(info.get("context_window") or 0)
        if (info.get("active", True) and info.get("output_modalities", ["text"]) == ["text"]
                and "text" in (info.get("input_modalities") or ["text"]) and window >= 32_000
                and "guard" not in model.id and "whisper" not in model.id):
            models.append({"id": model.id, "context_window": window})
    return sorted(models, key=lambda m: (m["id"] != config.LLM_MODEL, m["id"]))  # default first, then A-Z


def short_model(model_id: str) -> str:
    """'gpt-oss-120b' for 'openai/gpt-oss-120b'."""
    return model_id.split("/")[-1]


# ====================== documents (right panel) ======================
def file_icon(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        return ":material/picture_as_pdf:"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        return ":material/image:"
    if suffix in {".html", ".htm"}:
        return ":material/language:"
    return ":material/description:"


def file_rows(chunks: list[dict]) -> list[dict]:
    """One row per indexed file: pages, passages and how it was read."""
    rows = {}
    for chunk in chunks:
        row = rows.setdefault(chunk["source"], {"name": chunk["source"], "pages": set(), "passages": 0})
        row["pages"].add(chunk["page"])
        row["passages"] += 1
    return [{"name": r["name"], "pages": len(r["pages"]), "passages": r["passages"],
             "read_as": how_read(chunks, r["name"])} for r in rows.values()]


TAG_COLORS = {"Text": "gray", "Table": "blue", "Picture": "violet", "OCR": "orange"}


def last_cited_files(messages: list[dict]) -> set[str]:
    """Files the newest answer cites (their cards get an emerald outline)."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            return {sources[n - 1]["source"] for n in cited_numbers(message["content"], sources)}
    return set()


def render_documents_panel(pipeline: RAGPipeline, chat_id: int | None, messages: list[dict]) -> None:
    """Right panel: this chat's documents as a grid of cards. Untick a card to leave it out of answers."""
    with st.container(key="docs_panel"):
        st.html('<p class="panel-title">Documents</p>')
        if not pipeline.is_ready:
            st.caption("No documents in this chat yet. Upload files or add a link below "
                       "(or use the + in the message box).")
        else:
            rows = file_rows(pipeline.chunks)
            stats = st.session_state.stats or pipeline.stats()
            st.caption(f"{len(rows)} file{'s' if len(rows) != 1 else ''} · {stats['pages']} pages · "
                       f"{stats['chunks']} passages")
            cited = last_cited_files(messages)
            with st.container(horizontal=True, gap="small", key="doc_grid"):
                for number, row in enumerate(rows):
                    with st.container(border=True, key=f"dcard_{number}" + ("_cited" if row["name"] in cited else "")):
                        st.markdown(f"{file_icon(row['name'])} **{md_escape(shorten(row['name'], 40))}**")
                        st.caption(f"{row['pages']} page{'s' if row['pages'] != 1 else ''} · "
                                   f"{row['passages']} passage{'s' if row['passages'] != 1 else ''}")
                        with st.container(horizontal=True, gap="xxsmall"):
                            for tag in row["read_as"]:
                                st.badge(tag, color=TAG_COLORS.get(tag, "gray"))
                        st.checkbox("Use in answers", value=True, key=f"use_{chat_id}_{row['name']}",
                                    help="Untick to leave this file out of the search (a filter applied inside "
                                         "both searches).")
        with st.form(f"add_form_{chat_id}", border=False, clear_on_submit=True):
            uploads = st.file_uploader("Upload files", type=SUPPORTED_TYPES, accept_multiple_files=True,
                                       label_visibility="collapsed",
                                       help="PDF, Word, text, Markdown, HTML or an image (PNG, JPG, WEBP, BMP, GIF, TIFF)")
            link = st.text_input("Add a link", placeholder="Or paste a web page or PDF link",
                                 label_visibility="collapsed")
            if st.form_submit_button("Add to chat", icon=":material/upload_file:", width="stretch", type="primary"):
                if uploads:
                    st.session_state.pending_uploads = [(f.name, f.getvalue()) for f in uploads]
                if link.strip():
                    st.session_state.pending_link = link.strip()
                if not uploads and not link.strip():
                    st.toast("Choose a file or paste a link first.", icon=":material/upload_file:")
        for warning in st.session_state.get("ingest_warnings", []):  # e.g. vision model unavailable
            st.warning(warning, icon=":material/warning:")


def search_scope(pipeline: RAGPipeline, chat_id: int | None) -> list[str]:
    """Files ticked in the documents panel; [] means every file (nothing unticked)."""
    if not pipeline.is_ready:
        return []
    ticked = [name for name in pipeline.file_names if st.session_state.get(f"use_{chat_id}_{name}", True)]
    return [] if len(ticked) == len(pipeline.file_names) else ticked


def add_files_to_chat(files: list[tuple[str, bytes]]) -> None:
    """Read files into this chat (with the reading options from the sidebar), save them, report back."""
    pipeline: RAGPipeline = st.session_state.pipeline
    user_id = st.session_state.user["id"]
    use_ocr = st.session_state.get("opt_ocr", ocr.is_available()) and ocr.is_available()
    describe = st.session_state.get("opt_describe", False)
    st.session_state.stats = pipeline.ingest(files, append=True, ocr=use_ocr, describe_images=describe)
    if st.session_state.chat_id is None:  # first save: create the chat now
        st.session_state.chat_id = db.create_chat(user_id, Path(files[0][0]).stem[:40] or DEFAULT_TITLE)
    saved = index_folder(user_id, st.session_state.chat_id)
    pipeline.save(saved)  # keep the index for later
    (saved / "files").mkdir(parents=True, exist_ok=True)
    for name, data in files:  # originals, so the source pop-up can show the real page
        (saved / "files" / Path(name).name).write_bytes(data)
    stats = st.session_state.stats
    message = f"Added {len(files)} file{'s' if len(files) != 1 else ''}."
    if stats.get("ocr_pages"):
        message += f" {stats['ocr_pages']} scanned page{'s' if stats['ocr_pages'] != 1 else ''} read."
    if stats.get("image_pages"):
        message += f" Text found in pictures on {stats['image_pages']} page{'s' if stats['image_pages'] != 1 else ''}."
    st.session_state.flash = message
    st.session_state.ingest_warnings = stats.get("warnings", [])


# ====================== conversation (center) ======================
def welcome() -> None:
    """Empty chat: what DocMind does, in three steps."""
    st.html('<div class="hello"><h2>Ask your documents</h2>'
            '<p>Answers come only from your files, and every answer shows the page it came from.</p></div>')
    steps = [("1", "Add files", "Upload PDFs, Word files, text or images in the Documents panel on the right, "
                                "or paste a link. The + in the message box works too."),
             ("2", "Ask a question", "Follow-up questions keep their context."),
             ("3", "Check the source", "Click a source under an answer to open its page, sentence highlighted.")]
    with st.container(horizontal=True, gap=None, key="steps"):
        for number, title, text in steps:
            with st.container(key=f"step_{number}"):
                st.html(f'<span class="step-num">{number}</span>')
                st.markdown(f"**{title}**")
                st.caption(text)


def render_composer(chat_id: int | None, has_files: bool):
    """Message box (attach files with the +, speak with the mic) and, under it, the model picker,
    the read-aloud switch and token use."""
    with st.container(key="composer"):
        value = st.chat_input("Ask about your documents, or tap the mic" if has_files else "Attach files with +, then ask",
                              accept_file="multiple", file_type=SUPPORTED_TYPES, accept_audio=True, key="ask_box")
        SPEAKER(data=st.session_state.get("speak"), key="speaker", height=1)
        llm = st.session_state.llm
        models = answer_models(llm.client) or [{"id": llm.model, "context_window": None}]
        ids = [m["id"] for m in models]
        if llm.model not in ids:
            ids.insert(0, llm.model)
        with st.container(horizontal=True, vertical_alignment="center", gap="small", key="composer_meta"):
            chosen = st.selectbox("Model", ids, index=ids.index(llm.model), format_func=short_model,
                                  key="model_choice", label_visibility="collapsed", width=210,
                                  help="Groq model that writes the answers. The list comes live from Groq.")
            llm.model = chosen
            st.toggle("Read aloud", key="read_aloud", help="Read every answer aloud. Answers to spoken questions "
                                                           "are always read aloud.")
            window = next((m["context_window"] for m in models if m["id"] == chosen), None)
            used = (st.session_state.get("context_usage") or {}).get(chat_id)
            tokens = used["prompt_tokens"] if used else 0
            st.space("stretch")
            text = f"{tokens:,} / {window:,} tokens" if window else f"{tokens:,} tokens"
            st.markdown(f":small[:gray[:material/memory: {text}]]", width="content",
                        help="Tokens the last answer sent to the model (rules, passages found and recent chat), "
                             "out of the model's context window.")
            if window:
                with st.container(width=90, key="token_bar"):
                    st.progress(min(tokens / window, 1.0))
    return value


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


def start_login(user: dict) -> None:
    """Log in for this tab AND remember the user in the browser (survives refresh)."""
    st.session_state.user = user
    cookies.queue_set(auth.create_session(user["id"]))


def logout() -> None:
    auth.end_session(cookies.read_token())  # kill the saved login so the old token stops working
    cookies.queue_clear()  # and remove it from the browser
    for key in ("user", "chat_id", "messages", "pipeline", "stats", "confirm_delete", "pending_question", "prefs"):
        st.session_state.pop(key, None)


# ====================== login screen ======================
def page_header(title: str = "Sign in to DocMind", note: str = "") -> None:
    """Title block for the sign-in screens: the DocMind logo, then the title."""
    st.space("large")
    st.html(f'<div class="brand brand-lg">{WORDMARK}</div>')
    st.title(title, anchor=False)
    if note:
        st.caption(note)


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
@st.dialog("Rename chat")
def rename_dialog(chat_id: int, title: str) -> None:
    new_title = st.text_input("Chat name", value=title, max_chars=60)
    if st.button("Save", type="primary"):
        db.rename_chat(chat_id, st.session_state.user["id"], new_title.strip() or title)
        st.rerun()


def chat_row(chat: dict, active: bool, user_id: int) -> None:
    """One line in Pinned / Recents: open the chat, or use its "..." menu (pin, rename, export, delete)."""
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
            if st.button("Unpin" if chat["pinned"] else "Pin", key=f"pin_{chat_id}", type="tertiary", width="stretch",
                         icon=":material/keep_off:" if chat["pinned"] else ":material/keep:"):
                db.set_pinned(chat_id, user_id, not chat["pinned"])
                st.rerun()
            if st.button("Rename", key=f"ren_{chat_id}", type="tertiary", icon=":material/edit:", width="stretch"):
                rename_dialog(chat_id, chat["title"])
            if active and st.session_state.messages:
                st.download_button("Export", data=chat_to_markdown(chat["title"], st.session_state.messages),
                                   file_name="docmind_chat.md", mime="text/markdown", type="tertiary",
                                   icon=":material/download:", key=f"exp_{chat_id}", width="stretch")
            if st.button("Delete", key=f"del_{chat_id}", type="tertiary", icon=":material/delete:", width="stretch"):
                st.session_state.confirm_delete = chat_id
                st.rerun()


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
        st.caption(f"Answers are written by {md_escape(st.session_state.llm.model)} on Groq. "
                   "Change the model under the message box.")

    with account:
        st.text_input("Username", value=user["username"], disabled=True)
        st.caption(f"{PLAN}. Usernames can't be changed.")
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
CSS = (Path(__file__).parent / "assets" / "docmind.css").read_text(encoding="utf-8")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)  # every screen, sign-in included: orange buttons, panels, motion

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
if "prefs" not in st.session_state:
    st.session_state.prefs = dict(DEFAULT_PREFS)
prefs = st.session_state.prefs
use_hybrid, use_rerank, use_memory, top_k = prefs["hybrid"], prefs["rerank"], prefs["memory"], prefs["top_k"]
mode = ("Hybrid" if use_hybrid else "Vector") + (" + re-rank" if use_rerank else "")
flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash, icon=":material/check_circle:")

# ====================== left sidebar: logo, reading options, chats, account ======================
with st.sidebar:
    st.html(f'<div class="brand">{WORDMARK}</div>')
    if st.button("New chat", icon=":material/edit_square:", width="stretch", key="new_chat", type="primary"):
        new_chat()
        st.rerun()

    with st.container(key="reading_options"):
        st.html('<p class="side-label">When reading files</p>')
        ocr_ready = ocr.is_available()
        st.toggle("Read scans and images", value=ocr_ready, disabled=not ocr_ready, key="opt_ocr",
                  help=f"Reads scanned PDF pages, image files, and the words inside charts and screenshots "
                       f"in PDF and Word files. Slower; up to {config.MAX_OCR_PAGES} scanned pages and "
                       f"{config.MAX_IMAGES_PER_FILE} pictures per file.")
        st.toggle("Describe charts with AI", value=False, key="opt_describe",
                  help=f"Sends each picture to the Groq vision model ({config.VISION_MODEL}) and indexes its "
                       f"description, so questions about charts and diagrams can be answered. Up to "
                       f"{config.MAX_VISION_IMAGES} pictures per file; uses API quota.")

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
                f'<span class="who-text"><b>{html.escape(name)}</b><small>{PLAN}</small></span></div>')
        if st.button("", icon=":material/settings:", type="tertiary", key="open_settings", help="Settings"):
            settings_dialog()

# ====================== center: conversation  |  right: documents ======================
messages = st.session_state.messages
center, right = st.container(key="workspace").columns([2.6, 1.15], gap="large")  # stacks on narrow screens (CSS)
with right:
    render_documents_panel(pipeline, st.session_state.chat_id, messages)

with center:
    chat_area = st.container(key="conversation")
    with chat_area:
        if not messages and not pipeline.is_ready:
            welcome()
        previous_user_question = ""
        for index, message in enumerate(messages):  # redraw history on every rerun
            is_user = message["role"] == "user"
            with st.chat_message(message["role"], avatar=USER_AVATAR if is_user else ASSISTANT_AVATAR):
                if is_user:
                    st.markdown(f"**{md_escape(message['content'])}**")
                    if (st.session_state.chat_id, index) in st.session_state.get("voice_msgs", set()):
                        st.markdown(SPOKEN)
                    previous_user_question = message["content"]
                else:
                    render_answer(message, index, previous_user_question)
        if pipeline.is_ready and not messages:  # empty chat with files -> offer starters
            with st.container(key="starters"):
                st.caption("Try asking")
                with st.container(horizontal=True, gap="small"):
                    for suggestion in SUGGESTIONS:
                        if st.button(suggestion, icon=":material/arrow_outward:", icon_position="right"):
                            st.session_state.pending_question = suggestion  # picked up on the next run
                            st.rerun()
    submitted = render_composer(st.session_state.chat_id, pipeline.is_ready)

# ----- a spoken question: Whisper turns it into text, shown as the user's message -----
spoken = getattr(submitted, "audio", None) if submitted else None
transcript = ""
if spoken is not None:
    with chat_area:
        with st.spinner("Transcribing..."):
            try:
                transcript = voice.transcribe(st.session_state.llm.client, spoken.getvalue(), spoken.name or "question.wav")
            except Exception as error:  # API problem: say so, keep the chat usable
                st.error(friendly_error(error), icon=":material/error:")
                st.stop()
    if not transcript:
        st.toast("I didn't hear a question. Check that the right microphone is on, then speak a little closer.",
                 icon=":material/mic_off:")

# ----- new files: from the + in the message box, or the upload and link boxes in the Documents panel -----
attached = list(submitted.files) if submitted and getattr(submitted, "files", None) else []
uploaded = st.session_state.pop("pending_uploads", None) or []
link = st.session_state.pop("pending_link", None)
if attached or uploaded or link:
    with chat_area:
        with st.spinner("Reading your files..." + (" Scans and images take a little longer."
                                                   if st.session_state.get("opt_ocr") else "")):
            try:
                files = [(f.name, f.getvalue()) for f in attached] + uploaded
                if link:
                    files.append(fetch_url(link))  # public sites only
                add_files_to_chat(files)
            except ValueError as error:  # friendly message instead of a crash
                st.error(str(error), icon=":material/error:")
                st.stop()
    typed = (submitted.text or "").strip() if submitted else ""
    if typed or transcript:
        st.session_state.pending_question = typed or transcript  # ask it once the files are in
        st.session_state.pending_voice = bool(transcript and not typed)
    st.rerun()

typed = (getattr(submitted, "text", submitted) or "").strip() if submitted else ""
question = st.session_state.pop("pending_question", None) or typed or transcript  # suggestion, typed, then spoken
asked_by_voice = st.session_state.pop("pending_voice", False) or bool(transcript and question == transcript)
if question and not pipeline.is_ready:
    st.toast("Attach a file first: use the + in the message box.", icon=":material/attach_file:")
    question = None

if question and st.session_state.chat_id:
    chat_id = st.session_state.chat_id
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] if use_memory else []
    scope = search_scope(pipeline, chat_id)
    scope_kwargs = {"sources": scope} if scope else {}  # only passed when files were unticked
    with chat_area:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(f"**{md_escape(question)}**")
            if asked_by_voice:
                st.markdown(SPOKEN)
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
    if asked_by_voice:  # remembered for this session, so the bubble keeps its "Spoken" tag
        st.session_state.setdefault("voice_msgs", set()).add((chat_id, len(st.session_state.messages)))
    st.session_state.messages.append({"role": "user", "content": question})
    meta = (f"{result['total_ms'] / 1000:.1f} s total · retrieval {result['retrieve_ms']:.0f} ms · {mode} · "
            f"{short_model(st.session_state.llm.model)}")
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
        st.session_state.setdefault("context_usage", {})[chat_id] = usage
    if asked_by_voice or st.session_state.get("read_aloud"):  # talk back: spoken question -> spoken answer
        queue_speech(len(st.session_state.messages) - 1, result["answer"])
    st.rerun()  # redraw so the sidebar (recent order) and the documents panel update

"""DocMind web app: a JSON API over the same RAG pipeline, plus the React front end (web/dist).

Run:  uvicorn server:app --port 8000      (build the front end first: cd web && npm run build)
A workspace is one saved 'chat' row: its documents, index and Q&A history.
"""
import json
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from docmind import auth, config, db, evidence
from docmind.llm import LLM, friendly_error
from docmind.loader import SUPPORTED_TYPES, fetch_url
from docmind.pipeline import RAGPipeline

app = FastAPI(title="DocMind", docs_url="/api/docs", openapi_url="/api/openapi.json")
db.init_db()
WEB = Path(__file__).parent / "web" / "dist"
SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")

# ---------------- shared state ----------------
_lock = threading.Lock()
_models = None
_llm = None
_pipelines: dict[int, RAGPipeline] = {}
_workspace_locks: dict[int, threading.Lock] = {}


def models():
    global _models
    with _lock:
        if _models is None:  # heavy: loaded once, on first use
            from docmind.embedder import Embedder
            from docmind.reranker import Reranker
            _models = (Embedder(config.EMBED_MODEL), Reranker(config.RERANK_MODEL))
    return _models


def llm() -> LLM:
    global _llm
    if not config.GROQ_API_KEY:
        raise HTTPException(503, "The server has no GROQ_API_KEY set.")
    with _lock:
        if _llm is None:
            _llm = LLM(config.GROQ_API_KEY, config.LLM_MODEL)
    return _llm


def workspace_lock(workspace_id: int) -> threading.Lock:
    with _lock:
        return _workspace_locks.setdefault(workspace_id, threading.Lock())


def folder(user_id: int, workspace_id: int) -> Path:
    return config.INDEX_DIR / str(user_id) / str(workspace_id)


def pipeline_for(user_id: int, workspace_id: int) -> RAGPipeline:
    if workspace_id not in _pipelines:
        embedder, reranker = models()
        pipeline = RAGPipeline(embedder, reranker, _llm)
        saved = folder(user_id, workspace_id)
        if (saved / "chunks.json").exists():
            pipeline.load(saved)
        _pipelines[workspace_id] = pipeline
    return _pipelines[workspace_id]


def read_manifest(user_id: int, workspace_id: int) -> list[dict]:
    path = folder(user_id, workspace_id) / "files.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_manifest(user_id: int, workspace_id: int, files: list[dict]) -> None:
    path = folder(user_id, workspace_id) / "files.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(files, indent=1), encoding="utf-8")


# ---------------- auth ----------------
def current_user(request: Request) -> dict:
    user = auth.user_from_token(request.cookies.get(config.COOKIE_NAME))
    if not user:
        raise HTTPException(401, "Please sign in.")
    return user


def own_workspace(workspace_id: int, user: dict = Depends(current_user)) -> dict:
    chat = db.get_chat(workspace_id, user["id"])
    if not chat:
        raise HTTPException(404, "Workspace not found.")
    return {"user": user, "chat": chat}


def set_session_cookie(request: Request, response: Response, user_id: int) -> None:
    response.set_cookie(config.COOKIE_NAME, auth.create_session(user_id), max_age=config.SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=request.url.scheme == "https", path="/")


class Credentials(BaseModel):
    username: str = Field(max_length=60)
    password: str = Field(max_length=200)


@app.post("/api/auth/signup")
def signup(body: Credentials, request: Request, response: Response):
    try:
        user = auth.signup(body.username, body.password)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    set_session_cookie(request, response, user["id"])
    return {"user": {"id": user["id"], "username": user["username"]}, "recovery_code": user["recovery_code"]}


@app.post("/api/auth/login")
def login(body: Credentials, request: Request, response: Response):
    try:
        user = auth.login(body.username, body.password)
    except ValueError as error:  # locked after too many tries
        raise HTTPException(429, str(error)) from None
    if not user:
        raise HTTPException(401, "Incorrect username or password.")
    set_session_cookie(request, response, user["id"])
    return {"user": user}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    auth.end_session(request.cookies.get(config.COOKIE_NAME))
    response.delete_cookie(config.COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/me")
def me(user: dict = Depends(current_user)):
    return {"user": user, "model": config.LLM_MODEL, "types": SUPPORTED_TYPES}


# ---------------- workspaces ----------------
def summary(user_id: int, chat: dict) -> dict:
    files = read_manifest(user_id, chat["id"])
    return {"id": chat["id"], "title": chat["title"], "updated_at": chat["updated_at"],
            "documents": len(files), "pages": sum(f["pages"] for f in files),
            "file_names": [f["name"] for f in files]}


@app.get("/api/workspaces")
def list_workspaces(user: dict = Depends(current_user)):
    return {"workspaces": [summary(user["id"], c) for c in db.list_chats(user["id"], limit=100)]}


class Title(BaseModel):
    title: str = Field(min_length=1, max_length=60)


@app.post("/api/workspaces")
def create_workspace(body: Title, user: dict = Depends(current_user)):
    workspace_id = db.create_chat(user["id"], body.title.strip())
    return summary(user["id"], db.get_chat(workspace_id, user["id"]))


@app.patch("/api/workspaces/{workspace_id}")
def rename_workspace(body: Title, ctx: dict = Depends(own_workspace)):
    db.rename_chat(ctx["chat"]["id"], ctx["user"]["id"], body.title.strip())
    return {"ok": True}


@app.delete("/api/workspaces/{workspace_id}")
def delete_workspace(ctx: dict = Depends(own_workspace)):
    workspace_id, user_id = ctx["chat"]["id"], ctx["user"]["id"]
    db.delete_chat(workspace_id, user_id)
    shutil.rmtree(folder(user_id, workspace_id), ignore_errors=True)
    _pipelines.pop(workspace_id, None)
    return {"ok": True}


@app.get("/api/workspaces/{workspace_id}")
def get_workspace(ctx: dict = Depends(own_workspace)):
    user_id, chat = ctx["user"]["id"], ctx["chat"]
    return {**summary(user_id, chat), "files": read_manifest(user_id, chat["id"]),
            "messages": db.get_messages(chat["id"], user_id)}


@app.get("/api/recent")
def recent_answers(user: dict = Depends(current_user)):
    """Latest answers across workspaces, paired with the question that asked them (for Home)."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT m.chat_id, c.title, m.content, m.sources, "
            "(SELECT q.content FROM messages q WHERE q.chat_id = m.chat_id AND q.id < m.id AND q.role = 'user' "
            " ORDER BY q.id DESC LIMIT 1) AS question "
            "FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE c.user_id = ? AND m.role = 'assistant' ORDER BY m.id DESC LIMIT 6", (user["id"],)).fetchall()
    out = []
    for row in rows:
        sources = json.loads(row["sources"])
        out.append({"workspace_id": row["chat_id"], "workspace": row["title"], "question": row["question"],
                    "answer": row["content"], "source": sources[0] if sources else None})
    return {"recent": out}


# ---------------- documents ----------------
def how_read(chunks: list[dict], name: str) -> list[str]:
    mine = [c for c in chunks if c["source"] == name]
    tags = []
    if any("[Table]" in c["text"] for c in mine):
        tags.append("Table")
    if any(c.get("image") for c in mine):
        tags.append("Picture")
    if any(c.get("ocr") for c in mine):
        tags.append("OCR")
    # "Text" = the file has its own typed text (not only words read from scans or pictures)
    typed = any(not c.get("ocr") and len(c["text"].split("[Image]")[0].strip()) > 40 for c in mine)
    if typed or not tags:
        tags.insert(0, "Text")
    return tags


@app.post("/api/workspaces/{workspace_id}/documents")
def add_document(ctx: dict = Depends(own_workspace), file: UploadFile | None = File(None),
                 url: str = Form(""), ocr: bool = Form(True), describe: bool = Form(False)):
    user_id, workspace_id = ctx["user"]["id"], ctx["chat"]["id"]
    try:
        if file is not None:
            name, data = file.filename or "document", file.file.read()
        elif url.strip():
            name, data = fetch_url(url.strip())
        else:
            raise ValueError("Choose a file or paste a link.")
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    name = SAFE_NAME.sub("_", Path(name).name).strip() or "document"
    files = read_manifest(user_id, workspace_id)
    if any(f["name"] == name for f in files):
        raise HTTPException(409, f"{name} is already in this workspace.")
    if describe:
        llm()  # needs the key; fails early with a clear message
    with workspace_lock(workspace_id):
        pipeline = pipeline_for(user_id, workspace_id)
        pipeline.llm = _llm
        pages_before = pipeline.page_count
        try:
            stats = pipeline.ingest([(name, data)], append=True, ocr=ocr, describe_images=describe)
        except ValueError as error:
            raise HTTPException(400, str(error)) from None
        saved = folder(user_id, workspace_id)
        pipeline.save(saved)
        (saved / "files").mkdir(parents=True, exist_ok=True)
        (saved / "files" / name).write_bytes(data)
        entry = {"name": name, "size": len(data), "pages": pipeline.page_count - pages_before,
                 "read_as": how_read(pipeline.chunks, name), "added_at": time.time()}
        write_manifest(user_id, workspace_id, files + [entry])
    return {"file": entry, "warnings": stats.get("warnings", [])}


def stored_file(user_id: int, workspace_id: int, name: str) -> Path:
    path = folder(user_id, workspace_id) / "files" / Path(name).name
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/api/workspaces/{workspace_id}/files/{name}")
def download_file(name: str, ctx: dict = Depends(own_workspace)):
    return FileResponse(stored_file(ctx["user"]["id"], ctx["chat"]["id"], name))


@app.get("/api/workspaces/{workspace_id}/pages/{name}/{page}.png")
def page_image(name: str, page: int, ctx: dict = Depends(own_workspace)):
    """A PDF page rendered as an image (cached), for the source viewer."""
    path = stored_file(ctx["user"]["id"], ctx["chat"]["id"], name)
    cache = path.parent.parent / "renders" / f"{path.name}.{page}.png"
    if not cache.exists():
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(path))
        try:
            if not 1 <= page <= len(pdf):
                raise HTTPException(404, "Page not found.")
            image = pdf[page - 1].render(scale=2).to_pil()
        finally:
            pdf.close()
        cache.parent.mkdir(parents=True, exist_ok=True)
        image.save(cache, "PNG", optimize=True)
    return FileResponse(cache, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


class EvidenceRequest(BaseModel):
    source: str
    page: int
    passage: str
    answer: str


@app.post("/api/workspaces/{workspace_id}/evidence")
def evidence_view(body: EvidenceRequest, ctx: dict = Depends(own_workspace)):
    """What the viewer shows for one cited passage: a highlighted PDF page, or the passage text."""
    sentences = evidence.pick_sentences(body.passage, body.answer)
    path = stored_file(ctx["user"]["id"], ctx["chat"]["id"], body.source)
    if path.suffix.lower() == ".pdf":
        try:
            boxes = evidence.pdf_highlights(path.read_bytes(), body.page, sentences)
            return {"kind": "pdf", "sentences": sentences, "text": body.passage, **boxes,
                    "image": f"/api/workspaces/{ctx['chat']['id']}/pages/{path.name}/{body.page}.png"}
        except Exception:  # odd PDF: fall back to the passage text
            pass
    return {"kind": "text", "sentences": sentences, "text": body.passage}


# ---------------- asking ----------------
class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    hybrid: bool = True
    rerank: bool = True
    memory: bool = True
    top_k: int = Field(default=config.FINAL_K, ge=1, le=8)


@app.post("/api/workspaces/{workspace_id}/ask")
def ask(body: Question, ctx: dict = Depends(own_workspace)):
    user_id, workspace_id = ctx["user"]["id"], ctx["chat"]["id"]
    model = llm()
    question = body.question.strip()
    with workspace_lock(workspace_id):
        pipeline = pipeline_for(user_id, workspace_id)
        if not pipeline.is_ready:
            raise HTTPException(400, "Add a document to this workspace first.")
        pipeline.llm = model
        history = [{"role": m["role"], "content": m["content"]}
                   for m in db.get_messages(workspace_id, user_id)] if body.memory else []
        try:
            result = pipeline.ask(question, body.hybrid, body.rerank, body.top_k, history)
        except Exception as error:
            raise HTTPException(502, friendly_error(error)) from None
    sources = [{"source": s["source"], "page": s["page"], "text": s["text"],
                "ocr": bool(s.get("ocr")), "image": bool(s.get("image"))} for s in result["sources"]]
    meta = f"retrieval {result['retrieve_ms']:.0f} ms · total {result['total_ms']:.0f} ms · {model.model}"
    searched = result["search_query"] if result["search_query"] != question else ""
    db.add_message(workspace_id, user_id, "user", question)
    db.add_message(workspace_id, user_id, "assistant", result["answer"], sources, meta, searched)
    return {"role": "assistant", "content": result["answer"], "sources": sources, "meta": meta, "searched": searched}


@app.get("/api/workspaces/{workspace_id}/export")
def export(ctx: dict = Depends(own_workspace)):
    chat = ctx["chat"]
    lines = [f"# {chat['title']}", ""]
    for message in db.get_messages(chat["id"], ctx["user"]["id"]):
        if message["role"] == "user":
            lines += [f"## {message['content']}", ""]
        else:
            lines += [message["content"], ""]
            for number, source in enumerate(message["sources"], start=1):
                lines.append(f"[{number}] {source['source']}, page {source['page']}")
            lines.append("")
    return PlainTextResponse("\n".join(lines), headers={
        "Content-Disposition": f'attachment; filename="{SAFE_NAME.sub("_", chat["title"])}.md"'})


# ---------------- front end ----------------
@app.get("/{path:path}", include_in_schema=False)
def front_end(path: str):
    """Serve the built React app; unknown paths get index.html so client-side routes work."""
    if path.startswith("api/"):
        raise HTTPException(404, "Not found.")
    root = WEB.resolve()
    target = (WEB / path).resolve()
    if path and target.is_file() and root in target.parents:
        return FileResponse(target)
    index = WEB / "index.html"
    if not index.exists():
        return PlainTextResponse("Front end not built. Run: cd web && npm install && npm run build", 503)
    return FileResponse(index)

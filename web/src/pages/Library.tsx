import { DragEvent, FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fileKind, fileSize, uploadDocument, WorkspaceDetail } from "../api";
import { Close, UploadIcon } from "../icons";
import { useSession } from "../session";

type Job = { key: number; label: string; file?: File; url?: string; stage: "queued" | "uploading" | "reading" | "done" | "failed"; progress: number; message?: string };

const STEP_LABELS = ["Uploaded", "Reading text, tables and pictures", "Splitting into passages and indexing"];

function JobCard({ job }: { job: Job }) {
  const stageIndex = { queued: -1, uploading: 0, reading: 1, done: 3, failed: 3 }[job.stage];
  const width = job.stage === "queued" ? 0 : job.stage === "uploading" ? 10 + job.progress * 30 : job.stage === "reading" ? 70 : 100;
  return (
    <div className="job">
      <div className="row">
        <b title={job.label}>{job.label}</b>
        <span className="mono muted" style={{ fontSize: 12 }}>{job.stage === "done" ? "ready" : job.stage === "failed" ? "failed" : job.stage === "queued" ? "queued" : "working"}</span>
      </div>
      <div className="bar-track" aria-hidden="true"><div className={`bar-fill ${job.stage === "failed" ? "fail" : ""}`} style={{ width: `${width}%` }} /></div>
      {job.stage !== "queued" && job.stage !== "failed" && (
        <ol className="steps">
          {STEP_LABELS.map((label, i) => {
            const done = i < stageIndex || job.stage === "done";
            const now = !done && (i === stageIndex || (job.stage === "reading" && i === 2));
            return <li key={i} className={done ? "done" : now ? "now" : ""}><span aria-hidden="true">{done ? "✓" : now ? "●" : "○"}</span>{label}</li>;
          })}
        </ol>
      )}
      {job.message && <div className={job.stage === "failed" ? "error" : "muted"} style={{ fontSize: 12 }}>{job.message}</div>}
    </div>
  );
}

export default function Library() {
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const { refreshWorkspaces } = useSession();
  const [ws, setWs] = useState<WorkspaceDetail | null>(null);
  const [error, setError] = useState("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [ocr, setOcr] = useState(true);
  const [describe, setDescribe] = useState(false);
  const [link, setLink] = useState("");
  const [over, setOver] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const running = useRef(false);
  const keyRef = useRef(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => api.workspace(id).then(setWs).catch((e) => setError(e.message)), [id]);
  useEffect(() => { setWs(null); setJobs([]); load(); }, [load]);

  const patch = (key: number, change: Partial<Job>) => setJobs((all) => all.map((j) => (j.key === key ? { ...j, ...change } : j)));

  // Process the queue one file at a time so each gets real progress and its own result.
  useEffect(() => {
    if (running.current) return;
    const next = jobs.find((j) => j.stage === "queued");
    if (!next) return;
    running.current = true;
    patch(next.key, { stage: "uploading", progress: 0 });
    uploadDocument(id, { file: next.file, url: next.url }, { ocr, describe }, (fraction) => {
      patch(next.key, fraction >= 1 ? { stage: "reading", progress: 1 } : { progress: fraction });
    })
      .then((r) => {
        const read = r.file.read_as.join(", ");
        patch(next.key, { stage: "done", label: r.file.name, message: `${r.file.pages} ${r.file.pages === 1 ? "page" : "pages"} · read as ${read}${r.warnings.length ? ". " + r.warnings.join(" ") : ""}` });
        load();
        refreshWorkspaces().catch(() => undefined);
      })
      .catch((e) => patch(next.key, { stage: "failed", message: e.message }))
      .finally(() => { running.current = false; setJobs((all) => [...all]); });
  }, [jobs, id, ocr, describe, load, refreshWorkspaces]);

  const addFiles = (files: FileList | File[]) => {
    const items = Array.from(files).map((file) => ({ key: ++keyRef.current, label: file.name, file, stage: "queued" as const, progress: 0 }));
    if (items.length) { setJobs((all) => [...items, ...all]); setPanelOpen(true); }
  };
  const onDrop = (e: DragEvent) => { e.preventDefault(); setOver(false); addFiles(e.dataTransfer.files); };
  const addLink = (e: FormEvent) => {
    e.preventDefault();
    if (!link.trim()) return;
    setJobs((all) => [{ key: ++keyRef.current, label: link.trim(), url: link.trim(), stage: "queued", progress: 0 }, ...all]);
    setLink("");
  };

  const remove = async () => {
    await api.deleteWorkspace(id);
    await refreshWorkspaces();
    navigate("/app");
  };
  const rename = async (e: FormEvent) => {
    e.preventDefault();
    if (!renaming?.trim()) return;
    await api.renameWorkspace(id, renaming.trim());
    setRenaming(null);
    load();
    refreshWorkspaces().catch(() => undefined);
  };

  if (error) return <div className="empty"><h2>Can’t open this library</h2><p>{error}</p><Link className="btn" to="/app">Back to Home</Link></div>;
  if (!ws) return <div className="empty"><span className="spinner" aria-label="Loading" /></div>;

  const ocrFiles = ws.files.filter((f) => f.read_as.includes("OCR")).length;
  const pictureFiles = ws.files.filter((f) => f.read_as.includes("Picture")).length;
  const busy = jobs.filter((j) => j.stage !== "done" && j.stage !== "failed");

  return (
    <div className="library">
      <div className="library-main">
        <div className="library-head">
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="muted" style={{ fontSize: 13 }}>{ws.title}</span>
            <h1>Library</h1>
          </div>
          <div className="actions">
            <button type="button" className="btn btn-sm" onClick={() => setRenaming(ws.title)}>Rename</button>
            <button type="button" className="btn btn-sm" onClick={() => setConfirmDelete(true)}>Delete workspace</button>
            {ws.documents > 0 && <Link className="btn btn-sm" to={`/app/w/${id}`}>Ask questions</Link>}
            <button type="button" className="btn btn-dark btn-sm only-small" onClick={() => setPanelOpen(true)}>Upload files</button>
          </div>
        </div>

        <div className="stats">
          <div className="card stat"><span>Documents</span><b>{ws.documents}</b></div>
          <div className="card stat"><span>Pages indexed</span><b>{ws.pages}</b></div>
          <div className="card stat"><span>Read with OCR</span><b>{ocrFiles} {ocrFiles === 1 ? "file" : "files"}</b></div>
          <div className="card stat"><span>With pictures read</span><b>{pictureFiles} {pictureFiles === 1 ? "file" : "files"}</b></div>
        </div>

        <div className="card table-wrap">
          <table className="files">
            <thead><tr><th>Name</th><th>Pages</th><th>How it was read</th><th>Status</th></tr></thead>
            <tbody>
              {busy.map((j) => (
                <tr key={`job${j.key}`} style={{ background: "#fffbf0" }}>
                  <td><span className="n">{j.label}</span><small>{j.file ? `${fileKind(j.file.name)} · ${fileSize(j.file.size)}` : "Web link"}</small></td>
                  <td className="mono">–</td>
                  <td><span className="muted">–</span></td>
                  <td><span className="status busy"><i />{j.stage === "queued" ? "Waiting" : j.stage === "uploading" ? "Uploading…" : "Reading…"}</span></td>
                </tr>
              ))}
              {[...ws.files].reverse().map((f) => (
                <tr key={f.name}>
                  <td><a className="n" href={api.fileUrl(id, f.name)} target="_blank" rel="noreferrer" style={{ color: "inherit", textDecoration: "none" }}>{f.name}</a><small>{fileKind(f.name)} · {fileSize(f.size)}</small></td>
                  <td className="mono">{f.pages}</td>
                  <td><div className="tags">{f.read_as.map((t) => <span key={t} className={`tag tag-${t}`}>{t === "Picture" ? "Picture read" : t}</span>)}</div></td>
                  <td><span className="status"><i />Ready</span></td>
                </tr>
              ))}
              {!ws.files.length && !busy.length && (
                <tr><td colSpan={4} style={{ padding: 32, textAlign: "center" }} className="muted">No documents yet. Drop files on the right, or paste a link.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <aside className={`upload-panel ${panelOpen ? "open" : ""}`} aria-label="Add documents">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2>{busy.length ? `Adding ${busy.length} ${busy.length === 1 ? "file" : "files"}` : "Add documents"}</h2>
          <button type="button" className="btn btn-sm only-small" aria-label="Close" onClick={() => setPanelOpen(false)}><Close size={14} /></button>
        </div>
        <button type="button" className={`drop ${over ? "over" : ""}`} onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
          <UploadIcon size={22} />
          <span style={{ fontWeight: 500 }}>Drop files or click to choose</span>
          <small>PDF, Word, text, Markdown, HTML and images</small>
        </button>
        <input ref={inputRef} type="file" multiple hidden accept=".pdf,.docx,.txt,.md,.html,.htm,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff"
          onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }} />
        <form className="link-row" onSubmit={addLink}>
          <label className="sr-only" htmlFor="link">Web page or PDF link</label>
          <input id="link" type="url" placeholder="https://… web page or PDF link" value={link} onChange={(e) => setLink(e.target.value)} />
          <button type="submit" className="btn" disabled={!link.trim()}>Add</button>
        </form>
        <label className="toggle">
          <span><b style={{ fontWeight: 500 }}>Read scans and pictures (OCR)</b><small>Scanned pages, image files, words in charts</small></span>
          <input type="checkbox" checked={ocr} onChange={(e) => setOcr(e.target.checked)} />
        </label>
        <label className="toggle">
          <span><b style={{ fontWeight: 500 }}>Describe charts with AI</b><small>Lets you ask about values in charts and diagrams</small></span>
          <input type="checkbox" checked={describe} onChange={(e) => setDescribe(e.target.checked)} />
        </label>
        {jobs.map((j) => <JobCard key={j.key} job={j} />)}
      </aside>

      {confirmDelete && (
        <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && setConfirmDelete(false)}>
          <div className="card modal" role="alertdialog" aria-labelledby="del-title">
            <h2 id="del-title">Delete “{ws.title}”?</h2>
            <p style={{ margin: 0 }} className="muted">This removes its {ws.documents} documents, their index and every answer in it. It can’t be undone.</p>
            <div className="row">
              <button type="button" className="btn" onClick={() => setConfirmDelete(false)}>Cancel</button>
              <button type="button" className="btn btn-dark" onClick={remove}>Delete workspace</button>
            </div>
          </div>
        </div>
      )}
      {renaming !== null && (
        <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && setRenaming(null)}>
          <form className="card modal" onSubmit={rename}>
            <h2>Rename workspace</h2>
            <label className="field">Name<input autoFocus maxLength={60} value={renaming} onChange={(e) => setRenaming(e.target.value)} /></label>
            <div className="row">
              <button type="button" className="btn" onClick={() => setRenaming(null)}>Cancel</button>
              <button type="submit" className="btn btn-dark" disabled={!renaming.trim()}>Save</button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

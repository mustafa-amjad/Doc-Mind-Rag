import { FormEvent, KeyboardEvent, ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api, Evidence, loadSettings, Message, NOT_FOUND, Source, WorkspaceDetail } from "../api";
import { Check, Close, Doc, Up } from "../icons";
import { useSession } from "../session";

type Turn = { question: string; answer?: Message; error?: string };
type Active = { turn: number; index: number } | null;

function toTurns(messages: Message[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of messages) {
    if (m.role === "user") turns.push({ question: m.content });
    else if (turns.length) turns[turns.length - 1].answer = m;
  }
  return turns;
}

const citedNumbers = (text: string) => [...new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))];

/** Markdown-lite: paragraphs, "- " lists, **bold**, and [n] citation buttons. */
function AnswerBody({ text, sources, active, onCite }: { text: string; sources: Source[]; active: number | null; onCite: (n: number) => void }) {
  const inline = (line: string, key: string): ReactNode[] =>
    line.split(/(\*\*[^*]+\*\*|\[\d+\])/g).filter(Boolean).map((part, i) => {
      const cite = part.match(/^\[(\d+)\]$/);
      if (cite) {
        const n = Number(cite[1]);
        const s = sources[n - 1];
        if (!s) return null;
        return (
          <button key={`${key}-${i}`} type="button" className={`cite ${active === n ? "on" : ""}`} onClick={() => onCite(n)}
            aria-label={`Source ${n}: ${s.source}, page ${s.page}`}>{n}</button>
        );
      }
      if (part.startsWith("**") && part.endsWith("**")) return <strong key={`${key}-${i}`}>{part.slice(2, -2)}</strong>;
      return <span key={`${key}-${i}`}>{part}</span>;
    });

  const blocks: ReactNode[] = [];
  let list: string[] = [];
  const flush = () => {
    if (list.length) blocks.push(<ul key={`ul${blocks.length}`}>{list.map((l, i) => <li key={i}>{inline(l, `li${blocks.length}-${i}`)}</li>)}</ul>);
    list = [];
  };
  text.split("\n").forEach((raw) => {
    const line = raw.trim();
    if (/^[-*•]\s+/.test(line)) list.push(line.replace(/^[-*•]\s+/, ""));
    else {
      flush();
      if (line) blocks.push(<p key={`p${blocks.length}`}>{inline(line, `p${blocks.length}`)}</p>);
    }
  });
  flush();
  return <div className="answer-body">{blocks}</div>;
}

/** The index marks tables and pictures with tags; show them as plain labels in the viewer. */
const readable = (s: string) =>
  s.replace(/\[Image description\]\s*/g, "AI description of the picture: ")
    .replace(/\[Image\]\s*/g, "Picture (text read from it):\n")
    .replace(/\[Table\]\s*/g, "Table columns: ");

function Highlighted({ text: raw, sentences: rawSentences }: { text: string; sentences: string[] }) {
  const text = readable(raw);
  const sentences = rawSentences.map(readable);
  const ranges: [number, number][] = [];
  for (const s of sentences) {
    const at = text.indexOf(s);
    if (at >= 0) ranges.push([at, at + s.length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const parts: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach(([start, end], i) => {
    if (start < cursor) return;
    parts.push(text.slice(cursor, start), <mark key={i}>{text.slice(start, end)}</mark>);
    cursor = end;
  });
  parts.push(text.slice(cursor));
  return <div className="passage">{parts}</div>;
}

function Viewer({ workspaceId, answer, index, onPick, onClose, open }: {
  workspaceId: number; answer: Message | undefined; index: number; onPick: (i: number) => void; onClose: () => void; open: boolean;
}) {
  const sources = answer?.sources || [];
  const source = sources[index - 1];
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setEvidence(null);
    setError("");
    if (!answer || !source) return;
    let live = true;
    api.evidence(workspaceId, source, answer.content).then((e) => live && setEvidence(e)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [workspaceId, answer, source]);

  return (
    <aside className={`viewer ${open ? "" : "closed"}`} aria-label="Source viewer">
      {source ? (
        <>
          <div className="viewer-head">
            <span className="cite" style={{ margin: 0 }} aria-hidden="true">{index}</span>
            <div className="t">
              <span>{source.source}</span>
              <small>Page {source.page}{source.ocr ? " · read with OCR" : ""}{source.image ? " · includes a picture" : ""}</small>
            </div>
            <a className="btn btn-sm" href={api.fileUrl(workspaceId, source.source)} target="_blank" rel="noreferrer">Open</a>
            <button type="button" className="btn btn-sm only-small" aria-label="Close source viewer" onClick={onClose}><Close size={14} /></button>
          </div>
          <div className="viewer-body">
            {error && <div className="error">{error}</div>}
            {!evidence && !error && <div className="viewer-empty"><span className="spinner" />Loading page…</div>}
            {evidence?.kind === "pdf" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 16, width: "100%", alignItems: "center" }}>
              <div className="page">
                <img src={evidence.image} alt={`Page ${source.page} of ${source.source}`} />
                {evidence.boxes.map((b, i) => (
                  <span key={i} className="hl" aria-hidden="true" style={{
                    left: `${((b.x0 - 2) / evidence.width) * 100}%`, top: `${((b.top - 2) / evidence.height) * 100}%`,
                    width: `${((b.x1 - b.x0 + 4) / evidence.width) * 100}%`, height: `${((b.bottom - b.top + 4) / evidence.height) * 100}%`,
                  }} />
                ))}
              </div>
              {evidence.boxes.length === 0 && (  // scanned page: no text layer to mark, so show the OCR'd text
                <>
                  <span className="muted" style={{ fontSize: 12, alignSelf: "flex-start" }}>Text read from this page{source.ocr ? " with OCR" : ""}:</span>
                  <Highlighted text={evidence.text} sentences={evidence.sentences} />
                </>
              )}
              </div>
            )}
            {evidence?.kind === "text" && <Highlighted text={evidence.text} sentences={evidence.sentences} />}
          </div>
          {sources.length > 1 && (
            <div className="viewer-foot">
              <span className="muted">Passages used</span>
              {sources.map((s, i) => (
                <button key={i} type="button" className={i + 1 === index ? "on" : ""} onClick={() => onPick(i + 1)}>
                  {i + 1} · p.{s.page} {s.source.length > 18 ? s.source.slice(0, 16) + "…" : s.source}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="viewer-body">
          <div className="viewer-empty"><Doc size={22} />The page behind each answer appears here. Click a numbered citation to see it.</div>
        </div>
      )}
    </aside>
  );
}

export default function Ask() {
  const id = Number(useParams().id);
  const { refreshWorkspaces } = useSession();
  const [params, setParams] = useSearchParams();
  const [ws, setWs] = useState<WorkspaceDetail | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [active, setActive] = useState<Active>(null);
  const [viewerOpen, setViewerOpen] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoAsked = useRef<string | null>(null);

  useEffect(() => {
    setWs(null); setTurns([]); setActive(null); setLoadError("");
    api.workspace(id).then((w) => {
      setWs(w);
      const t = toTurns(w.messages);
      setTurns(t);
      const last = t.length - 1;
      if (last >= 0 && t[last].answer) {
        const first = citedNumbers(t[last].answer!.content)[0];
        if (first) setActive({ turn: last, index: first });
      }
    }).catch((e) => setLoadError(e.message));
  }, [id]);

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [turns.length, busy]);

  const send = useCallback(async (text: string) => {
    const question = text.trim();
    if (!question || busy) return;
    setDraft("");
    setBusy(true);
    let turnIndex = 0;
    setTurns((t) => { turnIndex = t.length; return [...t, { question }]; });
    try {
      const answer = await api.ask(id, question, loadSettings());
      setTurns((t) => t.map((turn, i) => (i === turnIndex ? { ...turn, answer } : turn)));
      const first = citedNumbers(answer.content)[0];
      setActive(first ? { turn: turnIndex, index: first } : null);
      refreshWorkspaces().catch(() => undefined);
    } catch (e) {
      setTurns((t) => t.map((turn, i) => (i === turnIndex ? { ...turn, error: (e as Error).message } : turn)));
    } finally {
      setBusy(false);
    }
  }, [busy, id, refreshWorkspaces]);

  useEffect(() => {
    const q = params.get("q");
    if (q && ws && ws.documents > 0 && autoAsked.current !== q) {
      autoAsked.current = q;
      setParams({}, { replace: true });
      send(q);
    }
  }, [params, ws, send, setParams]);

  const onSubmit = (e: FormEvent) => { e.preventDefault(); send(draft); };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(draft); }
  };
  const cite = (turn: number, index: number) => { setActive({ turn, index }); setViewerOpen(true); };
  const activeAnswer = useMemo(() => (active ? turns[active.turn]?.answer : undefined), [active, turns]);

  if (loadError) return <div className="empty"><h2>Can’t open this workspace</h2><p>{loadError}</p><Link className="btn" to="/app">Back to Home</Link></div>;
  if (!ws) return <div className="empty"><span className="spinner" aria-label="Loading" /></div>;

  return (
    <>
      <header className="topbar">
        <h1>{ws.title}</h1>
        <span className="pill pill-plain"><Doc size={12} />{ws.documents} {ws.documents === 1 ? "document" : "documents"} · {ws.pages} pages</span>
        <div className="actions">
          <Link className="btn btn-sm" to={`/app/w/${id}/library`}>Library</Link>
          <a className="btn btn-sm" href={api.exportUrl(id)} download>Export</a>
        </div>
      </header>
      <div className="ask">
        <div className="thread">
          <div className="thread-scroll" ref={scrollRef}>
            {ws.documents === 0 && (
              <div className="empty">
                <h2>Add documents to start</h2>
                <p>Upload PDFs, Word files, text, web pages or images. DocMind answers only from them.</p>
                <Link className="btn btn-dark" to={`/app/w/${id}/library`}>Add documents</Link>
              </div>
            )}
            {ws.documents > 0 && turns.length === 0 && !busy && (
              <div className="empty">
                <h2>Ask your first question</h2>
                <p>Every answer cites the file and page it came from. If the answer isn’t in your documents, DocMind says so.</p>
              </div>
            )}
            {turns.map((turn, t) => {
              const a = turn.answer;
              const notFound = a && a.content.trim() === NOT_FOUND;
              const cited = a ? citedNumbers(a.content).filter((n) => a.sources?.[n - 1]) : [];
              const citedFiles = new Set(cited.map((n) => a!.sources![n - 1].source)).size;
              return (
                <div className="turn" key={t}>
                  <div className="question"><small>You asked</small><p>{turn.question}</p></div>
                  {!a && !turn.error && <div className="thinking"><span className="spinner" />Searching your documents…</div>}
                  {turn.error && <div className="error">{turn.error}</div>}
                  {a && (
                    <>
                      <div className="answer-meta">
                        {notFound
                          ? <span className="pill pill-grey">Not in your documents</span>
                          : cited.length > 0 && <span className="pill pill-accent"><Check size={13} />Cited from {citedFiles} {citedFiles === 1 ? "source" : "sources"}</span>}
                        {a.searched && <span className="muted" style={{ fontSize: 12 }}>Searched for “{a.searched}”</span>}
                      </div>
                      <AnswerBody text={a.content} sources={a.sources || []} active={active?.turn === t ? active.index : null} onCite={(n) => cite(t, n)} />
                      {cited.length > 0 && (
                        <div className="sources-row">
                          {cited.map((n) => {
                            const s = a.sources![n - 1];
                            return (
                              <button key={n} type="button" className={`source-chip ${active?.turn === t && active.index === n ? "on" : ""}`} onClick={() => cite(t, n)}>
                                <b>{n}</b>{s.source} · p.{s.page}
                              </button>
                            );
                          })}
                        </div>
                      )}
                      <div className="answer-tools">
                        <button type="button" className="btn btn-sm" onClick={() => {
                          navigator.clipboard?.writeText(a.content).then(() => { setCopied(t); setTimeout(() => setCopied(null), 1500); });
                        }}>{copied === t ? "Copied" : "Copy"}</button>
                        {a.meta && <span className="mono muted" style={{ fontSize: 11, alignSelf: "center" }}>{a.meta}</span>}
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
          <div className="composer-wrap">
            <form className="composer" onSubmit={onSubmit}>
              <label className="sr-only" htmlFor="ask-input">Ask a question</label>
              <textarea id="ask-input" rows={1} value={draft} onChange={(e) => setDraft(e.target.value)} onKeyDown={onKey}
                placeholder={ws.documents ? (turns.length ? "Ask a follow-up question" : "Ask a question about your documents") : "Add documents first"}
                disabled={!ws.documents} />
              <div className="bar">
                <span className="pill pill-grey">All {ws.documents} {ws.documents === 1 ? "document" : "documents"}</span>
                <Link className="muted" style={{ fontSize: 12 }} to="/app/settings">Search settings</Link>
                <button type="submit" className="btn btn-dark icon-btn send" aria-label="Send" disabled={busy || !draft.trim() || !ws.documents}>
                  {busy ? <span className="spinner" /> : <Up size={17} />}
                </button>
              </div>
            </form>
          </div>
        </div>
        <Viewer workspaceId={id} answer={activeAnswer} index={active?.index || 0} open={viewerOpen}
          onPick={(i) => active && setActive({ turn: active.turn, index: i })} onClose={() => setViewerOpen(false)} />
      </div>
    </>
  );
}

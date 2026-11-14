import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import { api, chipColor, initials, NOT_FOUND, Recent } from "../api";
import { Up } from "../icons";
import { useSession } from "../session";
import type { ShellContext } from "./Shell";

export default function Home() {
  const { workspaces } = useSession();
  const { openNewWorkspace } = useOutletContext<ShellContext>();
  const navigate = useNavigate();
  const [question, setQuestion] = useState("");
  const [scope, setScope] = useState<number | null>(null);
  const [recent, setRecent] = useState<Recent[]>([]);
  const ready = workspaces.filter((w) => w.documents > 0);
  const target = scope ?? ready[0]?.id ?? null;

  useEffect(() => { api.recent().then(setRecent).catch(() => undefined); }, []);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!question.trim() || target === null) return;
    navigate(`/app/w/${target}?q=${encodeURIComponent(question.trim())}`);
  };

  return (
    <div className="scroll">
      <div className="home">
        <div className="hero">
          <h1>What do you need to know?</h1>
          <p>Answers come only from your documents, with the page they came from.</p>
          <form className="bigask" onSubmit={submit}>
            <label className="sr-only" htmlFor="home-ask">Ask a question</label>
            <input id="home-ask" value={question} onChange={(e) => setQuestion(e.target.value)}
              placeholder={ready.length ? "Ask about policies, contracts, reports…" : "Add documents to a workspace first"} disabled={!ready.length} />
            <div className="bar">
              {ready.length > 0 && <span className="muted" style={{ fontSize: 12 }}>Search in</span>}
              {ready.slice(0, 4).map((w) => (
                <button key={w.id} type="button" className={`scope ${w.id === target ? "on" : ""}`} aria-pressed={w.id === target} onClick={() => setScope(w.id)}>{w.title}</button>
              ))}
              {!ready.length && <button type="button" className="scope" onClick={openNewWorkspace}>+ New workspace</button>}
              <button type="submit" className="btn btn-dark icon-btn" aria-label="Ask" style={{ marginLeft: "auto", width: 44, height: 44, borderRadius: 12 }} disabled={!question.trim() || target === null}><Up size={18} /></button>
            </div>
          </form>
        </div>

        <section className="section" aria-labelledby="ws-heading">
          <div className="section-head">
            <h2 id="ws-heading">Workspaces</h2>
            <button type="button" className="btn btn-sm" onClick={openNewWorkspace}>New workspace</button>
          </div>
          <div className="ws-grid">
            {workspaces.map((w) => (
              <Link key={w.id} to={w.documents ? `/app/w/${w.id}` : `/app/w/${w.id}/library`} className="ws-card">
                <div className="top">
                  <span className="ws-chip" style={{ background: chipColor(w.id) }}>{initials(w.title)}</span>
                  <span className="mono muted" style={{ fontSize: 12 }}>{w.documents} {w.documents === 1 ? "doc" : "docs"} · {w.pages} {w.pages === 1 ? "page" : "pages"}</span>
                </div>
                <span className="name">{w.title}</span>
                <span className="files">{w.file_names.length ? w.file_names.join(", ") : "No documents yet: add some"}</span>
              </Link>
            ))}
            <button type="button" className="ws-card new" onClick={openNewWorkspace}>+ New workspace</button>
          </div>
        </section>

        {recent.length > 0 && (
          <section className="section" aria-labelledby="recent-heading">
            <h2 id="recent-heading" style={{ margin: 0, fontSize: 15, fontWeight: 500 }}>Recent answers</h2>
            <div className="card recent">
              {recent.map((r, i) => {
                const found = r.answer.trim() !== NOT_FOUND && r.source;
                return (
                  <Link key={i} to={`/app/w/${r.workspace_id}`}>
                    <span className="q">{r.question}</span>
                    <span className="a">{found ? r.answer.replace(/\s*\[\d+\]/g, "").replace(/\*\*/g, "") : "Not in your documents"}</span>
                    {found && r.source
                      ? <span className="src">{r.source.source.replace(/\.[a-z]+$/i, "").slice(0, 22)} p.{r.source.page}</span>
                      : <span className="src none">no source</span>}
                  </Link>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

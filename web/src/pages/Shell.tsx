import { FormEvent, useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useMatch, useNavigate } from "react-router-dom";
import { api, chipColor, initials } from "../api";
import { Book, Chat, Chevron, Gear, Home as HomeIcon, LogOut, Plus } from "../icons";
import { useSession } from "../session";

/** Modal that creates a workspace and opens its Library. */
export function NewWorkspace({ onClose }: { onClose: () => void }) {
  const { refreshWorkspaces } = useSession();
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      const ws = await api.createWorkspace(title.trim());
      await refreshWorkspaces();
      onClose();
      navigate(`/app/w/${ws.id}/library`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <div className="backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form className="card modal" onSubmit={submit} aria-labelledby="new-ws-title">
        <h2 id="new-ws-title">New workspace</h2>
        <p className="muted" style={{ margin: 0 }}>A workspace holds a set of documents and the questions you ask about them.</p>
        <label className="field">Name
          <input autoFocus required maxLength={60} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Customer policies" />
        </label>
        {error && <div className="error">{error}</div>}
        <div className="row">
          <button type="button" className="btn" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-dark" disabled={busy || !title.trim()}>Create and add files</button>
        </div>
      </form>
    </div>
  );
}

export type ShellContext = { openNewWorkspace: () => void };

export default function Shell() {
  const { user, workspaces, setUser } = useSession();
  const navigate = useNavigate();
  const wsMatch = useMatch("/app/w/:id/*");
  const currentId = wsMatch ? Number(wsMatch.params.id) : workspaces[0]?.id;
  const current = workspaces.find((w) => w.id === currentId);
  const [menuOpen, setMenuOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const close = (e: MouseEvent) => { if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const signOut = async () => {
    await api.logout().catch(() => undefined);
    setUser(null);
    navigate("/");
  };

  const context: ShellContext = { openNewWorkspace: () => setCreating(true) };

  return (
    <div className="shell">
      <nav className="sidebar" aria-label="DocMind">
        <div className="brand"><Link to="/app">DocMind</Link></div>

        <div className="switcher" ref={menuRef}>
          <button type="button" aria-haspopup="menu" aria-expanded={menuOpen} onClick={() => setMenuOpen((o) => !o)}>
            <span className="ws-chip" style={{ background: current ? chipColor(current.id) : "#45443f" }}>{current ? initials(current.title) : "+"}</span>
            <span className="name">{current ? current.title : "No workspace yet"}</span>
            <Chevron size={14} />
          </button>
          {menuOpen && (
            <div className="menu" role="menu">
              {workspaces.map((w) => (
                <Link key={w.id} role="menuitem" to={`/app/w/${w.id}`} onClick={() => setMenuOpen(false)}>
                  <span className="ws-chip" style={{ background: chipColor(w.id) }}>{initials(w.title)}</span>{w.title}
                </Link>
              ))}
              <button type="button" role="menuitem" onClick={() => { setMenuOpen(false); setCreating(true); }}><Plus size={14} />New workspace</button>
            </div>
          )}
        </div>

        <div className="nav">
          <NavLink to="/app" end><HomeIcon />Home</NavLink>
          {current && <NavLink to={`/app/w/${current.id}`} end><Chat />Ask</NavLink>}
          {current && <NavLink to={`/app/w/${current.id}/library`}><Book />Library</NavLink>}
          <NavLink to="/app/settings"><Gear />Settings</NavLink>
        </div>

        {workspaces.length > 0 && (
          <div>
            <div className="side-label">WORKSPACES</div>
            <div className="threads">
              {workspaces.slice(0, 8).map((w) => (
                <Link key={w.id} to={`/app/w/${w.id}`} className={w.id === currentId && wsMatch ? "active" : ""}>{w.title}</Link>
              ))}
            </div>
          </div>
        )}

        <div className="me">
          <div className="avatar" aria-hidden="true">{initials(user?.username || "")}</div>
          <div className="who"><span>{user?.username}</span><small>Signed in</small></div>
          <button type="button" aria-label="Sign out" title="Sign out" onClick={signOut}><LogOut /></button>
        </div>
      </nav>
      <div className="main">
        <Outlet context={context} />
      </div>
      {creating && <NewWorkspace onClose={() => setCreating(false)} />}
    </div>
  );
}

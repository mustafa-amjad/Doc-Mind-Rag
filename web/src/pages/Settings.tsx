import { FormEvent, ReactNode, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, loadSettings, saveSettings, Settings } from "../api";
import { useSession } from "../session";

type Tab = "general" | "account" | "security" | "privacy";
const TABS: { id: Tab; label: string }[] = [
  { id: "general", label: "General" },
  { id: "account", label: "Account" },
  { id: "security", label: "Security" },
  { id: "privacy", label: "Privacy" },
];

function Row({ title, note, children }: { title: string; note?: string; children?: ReactNode }) {
  return (
    <div className="set-row">
      <div className="set-text"><b>{title}</b>{note && <small>{note}</small>}</div>
      {children && <div className="set-control">{children}</div>}
    </div>
  );
}

function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return <input type="checkbox" role="switch" className="switch" aria-label={label} checked={checked} onChange={(e) => onChange(e.target.checked)} />;
}

function General() {
  const { model } = useSession();
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const update = (change: Partial<Settings>) => {
    const next = { ...settings, ...change };
    setSettings(next);
    saveSettings(next);
  };
  return (
    <>
      <h2>Search</h2>
      <Row title="Hybrid search" note="Combines keyword and meaning search.">
        <Switch label="Hybrid search" checked={settings.hybrid} onChange={(v) => update({ hybrid: v })} />
      </Row>
      <Row title="Re-rank results" note="Puts the most relevant passage first.">
        <Switch label="Re-rank results" checked={settings.rerank} onChange={(v) => update({ rerank: v })} />
      </Row>
      <Row title="Chat memory" note="Follow-up questions use the earlier ones for context.">
        <Switch label="Chat memory" checked={settings.memory} onChange={(v) => update({ memory: v })} />
      </Row>
      <Row title="Passages per answer" note={`${settings.top_k} passages are sent with each question.`}>
        <input type="range" min={1} max={8} value={settings.top_k} aria-label="Passages per answer" onChange={(e) => update({ top_k: Number(e.target.value) })} />
      </Row>
      <p className="set-foot">Saved in this browser. Answers are written by {model || "the configured model"} on Groq.</p>
    </>
  );
}

function Account() {
  const { user, setUser } = useSession();
  const navigate = useNavigate();
  const signOut = async () => {
    await api.logout().catch(() => undefined);
    setUser(null);
    navigate("/");
  };
  return (
    <>
      <h2>Account</h2>
      <Row title="Username" note={user?.username} />
      <Row title="Sign out" note="Sign out of DocMind in this browser.">
        <button type="button" className="btn btn-sm" onClick={signOut}>Sign out</button>
      </Row>
    </>
  );
}

function Security() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [codePassword, setCodePassword] = useState("");
  const [code, setCode] = useState("");
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState("");

  const run = async (key: string, action: () => Promise<string>) => {
    setBusy(key);
    setMessage(null);
    try {
      setMessage({ kind: "ok", text: await action() });
    } catch (e) {
      setMessage({ kind: "error", text: (e as Error).message });
    } finally {
      setBusy("");
    }
  };
  const changePassword = (event: FormEvent) => {
    event.preventDefault();
    run("password", async () => {
      await api.changePassword(current, next);
      setCurrent("");
      setNext("");
      return "Password updated. Other devices were signed out.";
    });
  };
  const newCode = (event: FormEvent) => {
    event.preventDefault();
    run("code", async () => {
      const r = await api.newRecoveryCode(codePassword);
      setCodePassword("");
      setCode(r.recovery_code);
      return "New recovery code created. The old one no longer works.";
    });
  };

  return (
    <>
      <h2>Security</h2>
      {message && <div className={message.kind === "ok" ? "notice" : "error"} role="status">{message.text}</div>}
      <form className="set-block" onSubmit={changePassword}>
        <b>Change password</b>
        <label className="field">Current password
          <input type="password" autoComplete="current-password" required value={current} onChange={(e) => setCurrent(e.target.value)} />
        </label>
        <label className="field">New password<span className="muted" style={{ fontWeight: 400 }}>At least 8 characters</span>
          <input type="password" autoComplete="new-password" required minLength={8} value={next} onChange={(e) => setNext(e.target.value)} />
        </label>
        <button type="submit" className="btn btn-dark" disabled={busy === "password"}>{busy === "password" ? <span className="spinner" /> : "Update password"}</button>
      </form>
      <form className="set-block" onSubmit={newCode}>
        <b>Recovery code</b>
        <small className="muted">Lets you reset a forgotten password. A new code replaces the old one.</small>
        {code ? <div className="code">{code}</div> : (
          <>
            <label className="field">Password
              <input type="password" autoComplete="current-password" required value={codePassword} onChange={(e) => setCodePassword(e.target.value)} />
            </label>
            <button type="submit" className="btn" disabled={busy === "code"}>{busy === "code" ? <span className="spinner" /> : "Create new code"}</button>
          </>
        )}
      </form>
      <Row title="Other devices" note="Sign out everywhere except this browser.">
        <button type="button" className="btn btn-sm" disabled={busy === "others"}
          onClick={() => run("others", async () => { await api.signOutOthers(); return "Signed out of all other devices."; })}>Sign out others</button>
      </Row>
    </>
  );
}

function Privacy() {
  const { refreshWorkspaces } = useSession();
  const navigate = useNavigate();
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const wipe = async () => {
    setBusy(true);
    try {
      await api.deleteAllData();
      await refreshWorkspaces();
      navigate("/app");
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
    }
  };
  return (
    <>
      <h2>Privacy</h2>
      <Row title="What DocMind stores"
        note="Your username, a salted hash of your password, your workspaces and questions, and the files you upload with their search index. Only your account can open them. To write an answer, your question and the matching passages are sent to Groq." />
      <div className="set-block danger">
        <b>Delete all workspaces</b>
        <small className="muted">Removes every workspace, its questions and its uploaded files. Your account stays. This can’t be undone.</small>
        {error && <div className="error">{error}</div>}
        <label className="check"><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} />I understand</label>
        <button type="button" className="btn btn-danger" disabled={!confirm || busy} onClick={wipe}>{busy ? <span className="spinner" /> : "Delete all workspaces"}</button>
      </div>
    </>
  );
}

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("general");
  return (
    <div className="scroll">
      <div className="settings">
        <h1>Settings</h1>
        <div className="settings-body">
          <nav className="settings-tabs" aria-label="Settings sections">
            {TABS.map((t) => (
              <button key={t.id} type="button" className={tab === t.id ? "on" : ""} aria-current={tab === t.id ? "page" : undefined} onClick={() => setTab(t.id)}>{t.label}</button>
            ))}
          </nav>
          <section className="settings-panel">
            {tab === "general" && <General />}
            {tab === "account" && <Account />}
            {tab === "security" && <Security />}
            {tab === "privacy" && <Privacy />}
          </section>
        </div>
      </div>
    </div>
  );
}

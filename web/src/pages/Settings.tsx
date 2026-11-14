import { useState } from "react";
import { loadSettings, saveSettings, Settings } from "../api";
import { useSession } from "../session";

export default function SettingsPage() {
  const { user, model } = useSession();
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const update = (change: Partial<Settings>) => {
    const next = { ...settings, ...change };
    setSettings(next);
    saveSettings(next);
  };
  return (
    <div className="scroll">
      <div className="narrow">
        <h1>Settings</h1>
        <h2>How DocMind searches</h2>
        <label className="toggle">
          <span><b style={{ fontWeight: 500 }}>Hybrid search</b><small>Keyword (BM25) plus meaning (vector) search, merged</small></span>
          <input type="checkbox" checked={settings.hybrid} onChange={(e) => update({ hybrid: e.target.checked })} />
        </label>
        <label className="toggle">
          <span><b style={{ fontWeight: 500 }}>Re-rank results</b><small>A cross-encoder puts the most relevant passage first</small></span>
          <input type="checkbox" checked={settings.rerank} onChange={(e) => update({ rerank: e.target.checked })} />
        </label>
        <label className="toggle">
          <span><b style={{ fontWeight: 500 }}>Chat memory</b><small>Follow-up questions use the earlier ones for context</small></span>
          <input type="checkbox" checked={settings.memory} onChange={(e) => update({ memory: e.target.checked })} />
        </label>
        <label className="range">
          <span>Passages sent to the model: <b>{settings.top_k}</b></span>
          <input type="range" min={1} max={8} value={settings.top_k} onChange={(e) => update({ top_k: Number(e.target.value) })} />
        </label>
        <p className="muted" style={{ margin: 0, fontSize: 13 }}>Saved in this browser. Answers are written by {model || "the configured model"} via Groq.</p>
        <h2>Account</h2>
        <div className="toggle" style={{ cursor: "default" }}><span><b style={{ fontWeight: 500 }}>Signed in as {user?.username}</b><small>Your sign-in lasts 30 days on this device.</small></span></div>
      </div>
    </div>
  );
}

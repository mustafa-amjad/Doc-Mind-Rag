import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api";
import { useSession } from "../session";

function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const { setUser } = useSession();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [code, setCode] = useState("");
  const next = (location.state as { from?: string } | null)?.from || "/app";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (mode === "login") {
        const r = await api.login(username, password);
        setUser(r.user);
        navigate(next, { replace: true });
      } else {
        const r = await api.signup(username, password);
        setCode(r.recovery_code);
        setUser(r.user);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (code) {
    return (
      <div className="auth">
        <div className="card auth-card">
          <Link className="logo" to="/">DocMind</Link>
          <h1>Save your recovery code</h1>
          <p className="muted" style={{ margin: 0 }}>It is the only way to reset a forgotten password, and it is shown once. Keep it in a password manager.</p>
          <div className="code">{code}</div>
          <button type="button" className="btn btn-dark btn-lg" onClick={() => navigate("/app", { replace: true })}>I saved it, continue</button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth">
      <div className="card auth-card">
        <Link className="logo" to="/">DocMind</Link>
        <h1>{mode === "login" ? "Welcome back" : "Create your account"}</h1>
        <form onSubmit={submit}>
          <label className="field">Username
            <input autoFocus autoComplete="username" required value={username} onChange={(e) => setUsername(e.target.value)} />
          </label>
          <label className="field">Password{mode === "signup" && <span className="muted" style={{ fontWeight: 400 }}>At least 8 characters</span>}
            <input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={mode === "signup" ? 8 : undefined}
              value={password} onChange={(e) => setPassword(e.target.value)} />
          </label>
          {error && <div className="error" role="alert">{error}</div>}
          <button type="submit" className="btn btn-dark btn-lg" disabled={busy}>
            {busy ? <span className="spinner" /> : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="muted" style={{ margin: 0, fontSize: 14 }}>
          {mode === "login" ? <>New to DocMind? <Link to="/signup">Create an account</Link></> : <>Already have an account? <Link to="/login">Sign in</Link></>}
        </p>
      </div>
    </div>
  );
}

export const Login = () => <AuthForm mode="login" />;
export const Signup = () => <AuthForm mode="signup" />;

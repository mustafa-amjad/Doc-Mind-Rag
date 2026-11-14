import { createContext, ReactNode, useCallback, useContext, useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { api, ApiError, User, Workspace } from "./api";

type SessionValue = {
  user: User | null | undefined; // undefined = still checking
  model: string;
  workspaces: Workspace[];
  setUser: (user: User | null) => void;
  refreshWorkspaces: () => Promise<Workspace[]>;
};

const SessionContext = createContext<SessionValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null | undefined>(undefined);
  const [model, setModel] = useState("");
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);

  useEffect(() => {
    api.me()
      .then((r) => { setUser(r.user); setModel(r.model); })
      .catch((e) => { if (e instanceof ApiError) setUser(null); else setUser(null); });
  }, []);

  const refreshWorkspaces = useCallback(async () => {
    const list = await api.workspaces();
    setWorkspaces(list);
    return list;
  }, []);

  useEffect(() => {
    if (user) {
      refreshWorkspaces().catch(() => undefined);
      api.me().then((r) => setModel(r.model)).catch(() => undefined);
    } else setWorkspaces([]);
  }, [user, refreshWorkspaces]);

  return (
    <SessionContext.Provider value={{ user, model, workspaces, setUser, refreshWorkspaces }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession outside SessionProvider");
  return value;
}

export function RequireUser({ children }: { children: ReactNode }) {
  const { user } = useSession();
  const location = useLocation();
  if (user === undefined) return <div className="auth"><span className="spinner" aria-label="Loading" /></div>;
  if (user === null) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return <>{children}</>;
}

// Thin client for the DocMind JSON API (same origin, session cookie).

export type User = { id: number; username: string };
export type Source = { source: string; page: number; text: string; ocr: boolean; image: boolean };
export type Message = { role: "user" | "assistant"; content: string; sources?: Source[]; meta?: string; searched?: string };
export type FileEntry = { name: string; size: number; pages: number; read_as: string[]; added_at: number };
export type Workspace = { id: number; title: string; updated_at: number; documents: number; pages: number; file_names: string[] };
export type WorkspaceDetail = Workspace & { files: FileEntry[]; messages: Message[] };
export type Recent = { workspace_id: number; workspace: string; question: string; answer: string; source: Source | null };
export type Evidence =
  | { kind: "pdf"; sentences: string[]; text: string; width: number; height: number; boxes: { x0: number; top: number; x1: number; bottom: number }[]; image: string }
  | { kind: "text"; sentences: string[]; text: string };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : undefined;
  const response = await fetch(path, { credentials: "same-origin", headers, ...init });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, message);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  me: () => request<{ user: User; model: string; types: string[] }>("/api/me"),
  login: (username: string, password: string) =>
    request<{ user: User }>("/api/auth/login", { method: "POST", body: json({ username, password }) }),
  signup: (username: string, password: string) =>
    request<{ user: User; recovery_code: string }>("/api/auth/signup", { method: "POST", body: json({ username, password }) }),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),

  workspaces: () => request<{ workspaces: Workspace[] }>("/api/workspaces").then((r) => r.workspaces),
  createWorkspace: (title: string) => request<Workspace>("/api/workspaces", { method: "POST", body: json({ title }) }),
  renameWorkspace: (id: number, title: string) =>
    request<{ ok: boolean }>(`/api/workspaces/${id}`, { method: "PATCH", body: json({ title }) }),
  deleteWorkspace: (id: number) => request<{ ok: boolean }>(`/api/workspaces/${id}`, { method: "DELETE" }),
  workspace: (id: number) => request<WorkspaceDetail>(`/api/workspaces/${id}`),
  recent: () => request<{ recent: Recent[] }>("/api/recent").then((r) => r.recent),

  ask: (id: number, question: string, settings: Settings) =>
    request<Message>(`/api/workspaces/${id}/ask`, { method: "POST", body: json({ question, ...settings }) }),
  evidence: (id: number, source: Source, answer: string) =>
    request<Evidence>(`/api/workspaces/${id}/evidence`, {
      method: "POST",
      body: json({ source: source.source, page: source.page, passage: source.text, answer }),
    }),
  fileUrl: (id: number, name: string) => `/api/workspaces/${id}/files/${encodeURIComponent(name)}`,
  exportUrl: (id: number) => `/api/workspaces/${id}/export`,
};

/** Upload one file or link. Reports upload progress; resolves when the server has read and indexed it. */
export function uploadDocument(
  id: number,
  item: { file?: File; url?: string },
  options: { ocr: boolean; describe: boolean },
  onUploaded: (fraction: number) => void,
): Promise<{ file: FileEntry; warnings: string[] }> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    if (item.file) form.append("file", item.file);
    if (item.url) form.append("url", item.url);
    form.append("ocr", String(options.ocr));
    form.append("describe", String(options.describe));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/workspaces/${id}/documents`);
    xhr.withCredentials = true;
    xhr.upload.onprogress = (event) => event.lengthComputable && onUploaded(event.loaded / event.total);
    xhr.onload = () => {
      let body: { detail?: string } & Record<string, unknown> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as unknown as { file: FileEntry; warnings: string[] });
      else reject(new ApiError(xhr.status, typeof body.detail === "string" ? body.detail : `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error while uploading."));
    xhr.send(form);
  });
}

// ---------- retrieval settings (kept per browser) ----------
export type Settings = { hybrid: boolean; rerank: boolean; memory: boolean; top_k: number };
const DEFAULTS: Settings = { hybrid: true, rerank: true, memory: true, top_k: 4 };

export function loadSettings(): Settings {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("docmind.settings") || "{}") };
  } catch {
    return DEFAULTS;
  }
}

export function saveSettings(settings: Settings) {
  try {
    localStorage.setItem("docmind.settings", JSON.stringify(settings));
  } catch {
    /* private mode: settings just won't persist */
  }
}

export const NOT_FOUND = "I couldn't find that in the uploaded documents.";

export function initials(title: string) {
  return (title.trim()[0] || "D").toUpperCase();
}

const CHIP_COLORS = ["#0b5d45", "#1e40af", "#9a3412", "#5b21b6", "#0e7490", "#854d0e"];
export function chipColor(id: number) {
  return CHIP_COLORS[id % CHIP_COLORS.length];
}

export function fileSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function fileKind(name: string) {
  const ext = name.split(".").pop()?.toLowerCase() || "";
  if (ext === "pdf") return "PDF";
  if (ext === "docx") return "Word";
  if (["txt", "md"].includes(ext)) return "Text";
  if (["html", "htm"].includes(ext)) return "Web page";
  return "Image";
}

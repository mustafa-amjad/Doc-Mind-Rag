"""SQLite storage: users, chats, messages. Every chat query also checks user_id (no peeking at other users' chats)."""
import json  # sources are saved as JSON text
import sqlite3  # built into Python, no install needed
import time
from contextlib import contextmanager

from docmind import config


@contextmanager
def connect():
    """Open a connection, commit if all went well, always close. One connection per call = thread-safe."""
    conn = sqlite3.connect(config.DB_PATH)  # read the path at call time (tests can change it)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA foreign_keys = ON")  # needed so deleting a chat deletes its messages
    try:
        yield conn
        conn.commit()  # save changes
    finally:
        conn.close()


def init_db() -> None:
    """Create the data folder and tables (safe to call on every start)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                pw_hash TEXT NOT NULL,
                recovery_salt TEXT NOT NULL DEFAULT '',
                recovery_hash TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS login_attempts (
                key TEXT PRIMARY KEY,
                fails INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                meta TEXT NOT NULL DEFAULT '',
                searched TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, pinned, updated_at);
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """
        )
        # upgrade older databases that were created before recovery codes existed
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
        for column in ("recovery_salt", "recovery_hash"):
            if column not in columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")


# ---------------- users ----------------
def create_user(username: str, salt: str, pw_hash: str, recovery_salt: str = "", recovery_hash: str = "") -> dict:
    try:
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, salt, pw_hash, recovery_salt, recovery_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (username, salt, pw_hash, recovery_salt, recovery_hash, time.time()),
            )
            return {"id": cur.lastrowid, "username": username}
    except sqlite3.IntegrityError:  # UNIQUE failed -> name taken
        raise ValueError("That username is already taken.") from None


def get_user(username: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


# ---------------- chats ----------------
def create_chat(user_id: int, title: str) -> int:
    now = time.time()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO chats (user_id, title, pinned, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
            (user_id, title, now, now),
        )
        return cur.lastrowid


def list_chats(user_id: int, limit: int = 40) -> list[dict]:
    """Pinned first, then newest activity first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, pinned, updated_at FROM chats WHERE user_id = ? "
            "ORDER BY pinned DESC, updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat(chat_id: int, user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
        return dict(row) if row else None


def rename_chat(chat_id: int, user_id: int, title: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE chats SET title = ? WHERE id = ? AND user_id = ?", (title, chat_id, user_id))


def set_pinned(chat_id: int, user_id: int, pinned: bool) -> None:
    with connect() as conn:
        conn.execute("UPDATE chats SET pinned = ? WHERE id = ? AND user_id = ?", (int(pinned), chat_id, user_id))


def delete_chat(chat_id: int, user_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        return cur.rowcount > 0  # False means "not yours or not found"


# ---------------- messages ----------------
def add_message(chat_id: int, user_id: int, role: str, content: str,
                sources: list | None = None, meta: str = "", searched: str = "") -> None:
    now = time.time()
    with connect() as conn:
        touched = conn.execute(  # also bumps the chat to the top of "Recent"
            "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?", (now, chat_id, user_id)
        ).rowcount
        if touched == 0:  # chat is not this user's
            raise PermissionError("Chat not found.")
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, meta, searched, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, json.dumps(sources or []), meta, searched, now),
        )


def get_messages(chat_id: int, user_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT m.role, m.content, m.sources, m.meta, m.searched FROM messages m "
            "JOIN chats c ON c.id = m.chat_id WHERE m.chat_id = ? AND c.user_id = ? ORDER BY m.id",
            (chat_id, user_id),
        ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"]),
             "meta": r["meta"], "searched": r["searched"]}
            for r in rows
        ]


# ---------------- login sessions (only a HASH of the token is stored) ----------------
def create_session(user_id: int, token_hash: str, expires_at: float) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))  # clean old ones
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )


def get_session_user(token_hash: str) -> dict | None:
    """Return {id, username} if the token exists and has not expired."""
    with connect() as conn:
        row = conn.execute(
            "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at > ?",
            (token_hash, time.time()),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token_hash: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


# ---------------- password reset + rate limiting ----------------
def get_user_by_id(user_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def set_password(user_id: int, salt: str, pw_hash: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET salt = ?, pw_hash = ? WHERE id = ?", (salt, pw_hash, user_id))


def set_recovery(user_id: int, salt: str, recovery_hash: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE users SET recovery_salt = ?, recovery_hash = ? WHERE id = ?", (salt, recovery_hash, user_id))


def delete_user_sessions(user_id: int) -> None:
    """Log the user out everywhere (used after a password change or reset)."""
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def get_lock_until(key: str) -> float:
    with connect() as conn:
        row = conn.execute("SELECT locked_until FROM login_attempts WHERE key = ?", (key,)).fetchone()
        return row["locked_until"] if row else 0.0


def record_failure(key: str, max_fails: int, lock_seconds: int) -> None:
    """Count a failed attempt. After max_fails the key is locked and the counter starts fresh."""
    with connect() as conn:
        row = conn.execute("SELECT fails FROM login_attempts WHERE key = ?", (key,)).fetchone()
        fails = (row["fails"] if row else 0) + 1
        locked_until = 0.0
        if fails >= max_fails:
            locked_until, fails = time.time() + lock_seconds, 0
        conn.execute(
            "INSERT INTO login_attempts (key, fails, locked_until) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET fails = excluded.fails, locked_until = excluded.locked_until",
            (key, fails, locked_until),
        )


def clear_failures(key: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM login_attempts WHERE key = ?", (key,))

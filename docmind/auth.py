"""Sign up / log in. Passwords are never stored, only a salted PBKDF2 hash."""
import hashlib  # PBKDF2 hashing
import hmac  # safe string compare
import os  # random salt
import re
import secrets  # cryptographically safe random tokens
import time

from docmind import config, db

ITERATIONS = 600_000  # how many times we hash (slow on purpose, makes guessing hard)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")  # letters, numbers, _ and . only


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    """Return (salt_hex, hash_hex). A new random salt is made if none is given."""
    salt = salt or os.urandom(16)  # unique salt per user: same password -> different hash
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    _, digest = hash_password(password, bytes.fromhex(salt_hex))  # hash the attempt with the SAME salt
    return hmac.compare_digest(digest, hash_hex)  # constant-time compare (no timing leaks)


def signup(username: str, password: str) -> dict:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-30 characters: letters, numbers, _ or .")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    salt, pw_hash = hash_password(password)
    return db.create_user(username, salt, pw_hash)  # raises ValueError if the name exists


def login(username: str, password: str) -> dict | None:
    """Return {id, username} if correct, otherwise None."""
    row = db.get_user(username.strip())
    if row is None:
        hash_password(password)  # burn the same time so "no such user" is not detectable by speed
        return None
    if not verify_password(password, row["salt"], row["pw_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


# ---------------- "stay logged in" sessions ----------------
def _hash_token(token: str) -> str:
    """Tokens are long and random, so a plain SHA-256 is enough (we never store the token itself)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    """Make a new random token, save its hash, and return the token (goes into the browser cookie)."""
    token = secrets.token_urlsafe(32)  # 256 bits of randomness
    db.create_session(user_id, _hash_token(token), time.time() + config.SESSION_DAYS * 86400)
    return token


def user_from_token(token: str | None) -> dict | None:
    """Cookie value -> user, or None if missing / unknown / expired."""
    if not token:
        return None
    return db.get_session_user(_hash_token(token))


def end_session(token: str | None) -> None:
    """Log out: the token stops working immediately, even if the cookie is still in the browser."""
    if token:
        db.delete_session(_hash_token(token))

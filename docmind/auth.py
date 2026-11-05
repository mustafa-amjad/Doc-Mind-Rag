"""Sign up / log in. Passwords are never stored, only a salted PBKDF2 hash."""
import hashlib  # PBKDF2 hashing
import hmac  # safe string compare
import math
import os  # random salt
import re
import secrets  # cryptographically safe random tokens
import time

from docmind import config, db

MAX_FAILS = 5  # wrong attempts allowed ...
LOCK_SECONDS = 15 * 60  # ... before the account is locked for 15 minutes
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I so codes are easy to read
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


def generate_recovery_code() -> str:
    """20 random characters (100 bits), shown as XXXXX-XXXXX-XXXXX-XXXXX."""
    chars = "".join(secrets.choice(CODE_ALPHABET) for _ in range(20))
    return "-".join(chars[i:i + 5] for i in range(0, 20, 5))


def _normalize_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())  # ignore dashes, spaces and lowercase


def _new_recovery(user_id: int) -> str:
    """Make a fresh recovery code, save only its salted hash, return the code to show ONCE."""
    code = generate_recovery_code()
    salt, code_hash = hash_password(_normalize_code(code))
    db.set_recovery(user_id, salt, code_hash)
    return code


def _check_password_rules(password: str) -> None:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")


def _check_lock(key: str) -> None:
    until = db.get_lock_until(key)
    if until > time.time():
        raise ValueError(f"Too many failed attempts. Try again in {math.ceil((until - time.time()) / 60)} min.")


def _fail(key: str) -> None:
    db.record_failure(key, MAX_FAILS, LOCK_SECONDS)


def signup(username: str, password: str) -> dict:
    """Create the account. The returned dict has a one-time 'recovery_code' to show the user."""
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-30 characters: letters, numbers, _ or .")
    _check_password_rules(password)
    salt, pw_hash = hash_password(password)
    code = generate_recovery_code()
    recovery_salt, recovery_hash = hash_password(_normalize_code(code))
    user = db.create_user(username, salt, pw_hash, recovery_salt, recovery_hash)  # raises if name exists
    user["recovery_code"] = code
    return user


def login(username: str, password: str) -> dict | None:
    """Return {id, username} if correct, None if wrong. Raises ValueError while locked out."""
    key = f"login:{username.strip().lower()}"
    _check_lock(key)
    row = db.get_user(username.strip())
    if row is None:
        hash_password(password)  # burn the same time so "no such user" is not detectable by speed
        _fail(key)
        return None
    if not verify_password(password, row["salt"], row["pw_hash"]):
        _fail(key)
        return None
    db.clear_failures(key)
    return {"id": row["id"], "username": row["username"]}


# ---------------- forgot / change password ----------------
def reset_password(username: str, recovery_code: str, new_password: str) -> dict:
    """Forgot password: username + recovery code -> new password.
    Logs out every device and returns the user plus a NEW recovery code (the old one is now dead)."""
    _check_password_rules(new_password)
    key = f"reset:{username.strip().lower()}"
    _check_lock(key)
    row = db.get_user(username.strip())
    has_code = bool(row and row["recovery_hash"])  # old accounts may have no code yet
    valid = has_code and verify_password(_normalize_code(recovery_code), row["recovery_salt"], row["recovery_hash"])
    if not valid:
        if not row:
            hash_password(recovery_code)  # same timing for unknown users
        _fail(key)
        raise ValueError("Wrong username or recovery code.")
    salt, pw_hash = hash_password(new_password)
    db.set_password(row["id"], salt, pw_hash)
    db.delete_user_sessions(row["id"])  # anyone using the old password is logged out
    db.clear_failures(key)
    return {"id": row["id"], "username": row["username"], "recovery_code": _new_recovery(row["id"])}


def change_password(user_id: int, current_password: str, new_password: str) -> None:
    """Logged-in user changes the password. All saved logins are removed (log in again to get a fresh one)."""
    _check_password_rules(new_password)
    key = f"change:{user_id}"
    _check_lock(key)
    row = db.get_user_by_id(user_id)
    if row is None or not verify_password(current_password, row["salt"], row["pw_hash"]):
        _fail(key)
        raise ValueError("Current password is wrong.")
    salt, pw_hash = hash_password(new_password)
    db.set_password(user_id, salt, pw_hash)
    db.delete_user_sessions(user_id)
    db.clear_failures(key)


def regenerate_recovery_code(user_id: int, password: str) -> str:
    """Make a new recovery code (needs the password). Also the way for older accounts to get their first one."""
    key = f"change:{user_id}"
    _check_lock(key)
    row = db.get_user_by_id(user_id)
    if row is None or not verify_password(password, row["salt"], row["pw_hash"]):
        _fail(key)
        raise ValueError("Password is wrong.")
    db.clear_failures(key)
    return _new_recovery(user_id)


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

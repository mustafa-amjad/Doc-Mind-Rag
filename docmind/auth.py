"""Sign up / log in. Passwords are never stored, only a salted PBKDF2 hash."""
import hashlib  # PBKDF2 hashing
import hmac  # safe string compare
import os  # random salt
import re

from docmind import db

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

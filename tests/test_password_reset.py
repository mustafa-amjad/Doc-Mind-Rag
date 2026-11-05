import sqlite3

import pytest

from docmind import auth, config, db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(auth, "ITERATIONS", 1000)
    db.init_db()


def make_user(name="mustafa", password="supersecret1"):
    user = auth.signup(name, password)
    return user, user["recovery_code"]


def test_signup_gives_recovery_code_in_expected_format():
    _, code = make_user()
    parts = code.split("-")
    assert len(parts) == 4 and all(len(p) == 5 for p in parts)


def test_reset_changes_password_and_rotates_code():
    user, code = make_user()
    result = auth.reset_password("mustafa", code, "brandnewpass1")
    assert auth.login("mustafa", "supersecret1") is None  # old password dead
    assert auth.login("mustafa", "brandnewpass1")["id"] == user["id"]  # new one works
    assert result["recovery_code"] != code
    with pytest.raises(ValueError):
        auth.reset_password("mustafa", code, "anotherpass12")  # old code is dead too
    auth.reset_password("mustafa", result["recovery_code"], "anotherpass12")  # new code works


def test_code_entry_ignores_case_dashes_and_spaces():
    _, code = make_user()
    sloppy = code.lower().replace("-", " ")
    assert auth.reset_password("mustafa", sloppy, "brandnewpass1")["username"] == "mustafa"


def test_reset_logs_out_other_devices():
    user, code = make_user()
    token = auth.create_session(user["id"])
    auth.reset_password("mustafa", code, "brandnewpass1")
    assert auth.user_from_token(token) is None


def test_wrong_code_and_unknown_user():
    make_user()
    with pytest.raises(ValueError):
        auth.reset_password("mustafa", "AAAAA-AAAAA-AAAAA-AAAAA", "brandnewpass1")
    with pytest.raises(ValueError):
        auth.reset_password("nobody", "AAAAA-AAAAA-AAAAA-AAAAA", "brandnewpass1")


def test_short_new_password_rejected():
    _, code = make_user()
    with pytest.raises(ValueError):
        auth.reset_password("mustafa", code, "short")


def test_login_lockout_after_five_failures_blocks_even_correct_password():
    make_user()
    for _ in range(auth.MAX_FAILS):
        assert auth.login("mustafa", "wrong-password") is None
    with pytest.raises(ValueError, match="Too many"):
        auth.login("mustafa", "supersecret1")


def test_success_resets_the_failure_counter():
    make_user()
    for _ in range(auth.MAX_FAILS - 1):
        auth.login("mustafa", "wrong-password")
    assert auth.login("mustafa", "supersecret1") is not None  # clears the counter
    for _ in range(auth.MAX_FAILS - 1):
        auth.login("mustafa", "wrong-password")
    assert auth.login("mustafa", "supersecret1") is not None  # still not locked


def test_reset_is_locked_after_too_many_wrong_codes():
    _, code = make_user()
    for _ in range(auth.MAX_FAILS):
        with pytest.raises(ValueError):
            auth.reset_password("mustafa", "AAAAA-AAAAA-AAAAA-AAAAA", "brandnewpass1")
    with pytest.raises(ValueError, match="Too many"):
        auth.reset_password("mustafa", code, "brandnewpass1")  # even the right code is blocked


def test_change_password_needs_current_password():
    user, _ = make_user()
    with pytest.raises(ValueError):
        auth.change_password(user["id"], "not-my-password", "brandnewpass1")
    token = auth.create_session(user["id"])
    auth.change_password(user["id"], "supersecret1", "brandnewpass1")
    assert auth.login("mustafa", "brandnewpass1") is not None
    assert auth.user_from_token(token) is None  # saved logins removed


def test_regenerate_code_needs_password_and_replaces_old():
    user, old_code = make_user()
    with pytest.raises(ValueError):
        auth.regenerate_recovery_code(user["id"], "wrong-password")
    new_code = auth.regenerate_recovery_code(user["id"], "supersecret1")
    with pytest.raises(ValueError):
        auth.reset_password("mustafa", old_code, "brandnewpass1")
    assert auth.reset_password("mustafa", new_code, "brandnewpass1")["username"] == "mustafa"


def test_old_database_gets_upgraded(tmp_path, monkeypatch):
    """A database made before recovery codes existed must keep working."""
    old_path = tmp_path / "old.db"
    conn = sqlite3.connect(old_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE COLLATE NOCASE, "
                 "salt TEXT NOT NULL, pw_hash TEXT NOT NULL, created_at REAL NOT NULL)")
    salt, pw_hash = auth.hash_password("supersecret1")
    conn.execute("INSERT INTO users (username, salt, pw_hash, created_at) VALUES ('legacy', ?, ?, 0)", (salt, pw_hash))
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", old_path)
    db.init_db()
    legacy = auth.login("legacy", "supersecret1")  # old user can still log in
    assert legacy is not None
    with pytest.raises(ValueError):  # no recovery code yet, so reset is refused
        auth.reset_password("legacy", "AAAAA-AAAAA-AAAAA-AAAAA", "brandnewpass1")
    code = auth.regenerate_recovery_code(legacy["id"], "supersecret1")  # they can create one while logged in
    assert auth.reset_password("legacy", code, "brandnewpass1")["username"] == "legacy"

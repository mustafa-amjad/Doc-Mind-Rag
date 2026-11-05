import pytest

from docmind import auth, config, db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(auth, "ITERATIONS", 1000)
    db.init_db()


def test_token_logs_user_back_in():
    user = auth.signup("mustafa", "supersecret1")
    token = auth.create_session(user["id"])
    assert auth.user_from_token(token)["username"] == "mustafa"


def test_bad_or_missing_token():
    assert auth.user_from_token(None) is None
    assert auth.user_from_token("") is None
    assert auth.user_from_token("not-a-real-token") is None


def test_logout_kills_token():
    user = auth.signup("alice", "supersecret1")
    token = auth.create_session(user["id"])
    auth.end_session(token)
    assert auth.user_from_token(token) is None


def test_expired_token_is_rejected(monkeypatch):
    user = auth.signup("bobby", "supersecret1")
    monkeypatch.setattr(config, "SESSION_DAYS", -1)  # already expired when created
    token = auth.create_session(user["id"])
    assert auth.user_from_token(token) is None


def test_token_is_never_stored_in_plain_text():
    user = auth.signup("carol", "supersecret1")
    token = auth.create_session(user["id"])
    with db.connect() as conn:
        stored = [row["token_hash"] for row in conn.execute("SELECT token_hash FROM sessions")]
    assert token not in stored and len(stored) == 1


def test_deleting_user_chats_not_needed_but_sessions_are_per_user():
    a = auth.signup("userone", "supersecret1")
    b = auth.signup("usertwo", "supersecret1")
    token_a = auth.create_session(a["id"])
    assert auth.user_from_token(token_a)["id"] == a["id"] != b["id"]

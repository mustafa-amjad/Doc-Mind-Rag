import pytest

from docmind import auth, config, db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Every test gets its own empty database."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(auth, "ITERATIONS", 1000)  # fast hashing for tests
    db.init_db()


def test_signup_login_and_wrong_password():
    auth.signup("mustafa", "supersecret1")
    assert auth.login("mustafa", "supersecret1")["username"] == "mustafa"
    assert auth.login("MUSTAFA", "supersecret1") is not None  # usernames are case-insensitive
    assert auth.login("mustafa", "wrongpass") is None
    assert auth.login("nobody", "whatever12") is None


def test_validation_and_duplicate():
    with pytest.raises(ValueError):
        auth.signup("ab", "supersecret1")  # username too short
    with pytest.raises(ValueError):
        auth.signup("validname", "short")  # password too short
    auth.signup("dupe", "supersecret1")
    with pytest.raises(ValueError):
        auth.signup("DUPE", "supersecret2")  # same name, different case


def test_pinned_first_and_messages_roundtrip():
    user = auth.signup("alice", "supersecret1")
    first = db.create_chat(user["id"], "first")
    second = db.create_chat(user["id"], "second")
    db.set_pinned(first, user["id"], True)
    assert [c["id"] for c in db.list_chats(user["id"])] == [first, second]  # pinned on top
    db.add_message(second, user["id"], "user", "hi")
    db.add_message(second, user["id"], "assistant", "hello", [{"page": 1}], "1s", "hi there")
    msgs = db.get_messages(second, user["id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"] == [{"page": 1}] and msgs[1]["searched"] == "hi there"
    assert db.list_chats(user["id"])[0]["id"] == first  # pinned stays above even after activity


def test_users_cannot_touch_each_others_chats():
    alice = auth.signup("alice", "supersecret1")
    bob = auth.signup("bobby", "supersecret1")
    chat_id = db.create_chat(alice["id"], "private")
    db.add_message(chat_id, alice["id"], "user", "secret")
    assert db.get_chat(chat_id, bob["id"]) is None
    assert db.get_messages(chat_id, bob["id"]) == []
    with pytest.raises(PermissionError):
        db.add_message(chat_id, bob["id"], "user", "hack")
    assert db.delete_chat(chat_id, bob["id"]) is False  # cannot delete
    assert db.delete_chat(chat_id, alice["id"]) is True


def test_delete_chat_removes_messages():
    user = auth.signup("carol", "supersecret1")
    chat_id = db.create_chat(user["id"], "temp")
    db.add_message(chat_id, user["id"], "user", "x")
    db.delete_chat(chat_id, user["id"])
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0  # cascade worked

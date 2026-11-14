"""Web API tests: accounts, workspaces, uploads, asking, evidence and isolation between users.
Uses a tiny fake embedder and a fake LLM, so no model download or API key is needed."""
import pytest
from fastapi.testclient import TestClient

from docmind import config, db, evidence
from tests.test_pipeline_store import FakeEmbedder


class FakeLLM:
    model = "fake-model"

    def rewrite_question(self, question, history):
        return "What is the refund window for gift cards?" if history and "gift" in question else question

    def answer(self, question, chunks, history=None):
        return "Refunds are allowed within 30 days of purchase. [1]" if chunks else "I couldn't find that in the uploaded documents."

    def describe_image(self, image):
        return "A chart."


class FakeReranker:
    def rerank(self, question, chunks, k):
        return chunks[:k]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "indexes")
    db.init_db()
    import server
    monkeypatch.setattr(server, "_models", (FakeEmbedder(), FakeReranker()))
    monkeypatch.setattr(server, "_llm", FakeLLM())
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    server._pipelines.clear()
    return TestClient(server.app)


def sign_up(client, name="alice"):
    r = client.post("/api/auth/signup", json={"username": name, "password": "longpassword1"})
    assert r.status_code == 200, r.text
    return r.json()


TEXT = b"Refunds are allowed within 30 days of purchase. Gift cards cannot be refunded.\nShipping takes five days."


def test_requires_sign_in(client):
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/workspaces").status_code == 401


def test_signup_sets_cookie_that_survives_a_new_client(client):
    body = sign_up(client)
    assert body["recovery_code"]
    cookie = client.cookies.get(config.COOKIE_NAME)
    assert cookie and client.get("/api/me").json()["user"]["username"] == "alice"
    fresh = TestClient(client.app, cookies={config.COOKIE_NAME: cookie})  # a reload / new tab
    assert fresh.get("/api/me").status_code == 200


def test_wrong_password_and_logout(client):
    sign_up(client)
    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/auth/login", json={"username": "alice", "password": "nope-nope-1"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "alice", "password": "longpassword1"}).status_code == 200


def test_upload_ask_and_history(client):
    sign_up(client)
    ws = client.post("/api/workspaces", json={"title": "Policies"}).json()
    r = client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("policy.txt", TEXT)}, data={"ocr": "false"})
    assert r.status_code == 200, r.text
    assert r.json()["file"]["read_as"] == ["Text"] and r.json()["file"]["pages"] == 1
    again = client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("policy.txt", TEXT)})
    assert again.status_code == 409  # same name twice

    answer = client.post(f"/api/workspaces/{ws['id']}/ask", json={"question": "How long for refunds?"}).json()
    assert "30 days" in answer["content"] and answer["sources"][0]["source"] == "policy.txt"
    follow = client.post(f"/api/workspaces/{ws['id']}/ask", json={"question": "And for gift cards?"}).json()
    assert follow["searched"] == "What is the refund window for gift cards?"

    detail = client.get(f"/api/workspaces/{ws['id']}").json()
    assert detail["documents"] == 1 and len(detail["messages"]) == 4
    assert client.get("/api/recent").json()["recent"][0]["question"] == "And for gift cards?"
    assert "# Policies" in client.get(f"/api/workspaces/{ws['id']}/export").text


def test_ask_before_upload_is_refused(client):
    sign_up(client)
    ws = client.post("/api/workspaces", json={"title": "Empty"}).json()
    assert client.post(f"/api/workspaces/{ws['id']}/ask", json={"question": "Anything?"}).status_code == 400


def test_evidence_highlights_supporting_sentence(client):
    sign_up(client)
    ws = client.post("/api/workspaces", json={"title": "P"}).json()
    client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("policy.txt", TEXT)}, data={"ocr": "false"})
    body = client.post(f"/api/workspaces/{ws['id']}/evidence", json={
        "source": "policy.txt", "page": 1, "passage": TEXT.decode(), "answer": "Refunds are allowed within 30 days. [1]"}).json()
    assert body["kind"] == "text" and body["sentences"] == ["Refunds are allowed within 30 days of purchase."]


def test_users_cannot_see_each_others_workspaces(client):
    sign_up(client, "alice")
    ws = client.post("/api/workspaces", json={"title": "Private"}).json()
    client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("policy.txt", TEXT)}, data={"ocr": "false"})
    client.post("/api/auth/logout")
    sign_up(client, "bob")
    assert client.get(f"/api/workspaces/{ws['id']}").status_code == 404
    assert client.post(f"/api/workspaces/{ws['id']}/ask", json={"question": "x"}).status_code == 404
    assert client.get(f"/api/workspaces/{ws['id']}/files/policy.txt").status_code == 404
    assert client.get("/api/workspaces").json()["workspaces"] == []


def test_file_names_cannot_escape_the_workspace(client):
    sign_up(client)
    ws = client.post("/api/workspaces", json={"title": "P"}).json()
    r = client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("../../evil.txt", TEXT)}, data={"ocr": "false"})
    assert r.json()["file"]["name"] == "evil.txt"
    assert client.get(f"/api/workspaces/{ws['id']}/files/..%2F..%2Ftest.db").status_code == 404


def test_delete_workspace_removes_files(client):
    sign_up(client)
    ws = client.post("/api/workspaces", json={"title": "Temp"}).json()
    client.post(f"/api/workspaces/{ws['id']}/documents", files={"file": ("policy.txt", TEXT)}, data={"ocr": "false"})
    folder = config.INDEX_DIR / "1" / str(ws["id"])
    assert folder.exists()
    assert client.delete(f"/api/workspaces/{ws['id']}").status_code == 200
    assert not folder.exists() and client.get(f"/api/workspaces/{ws['id']}").status_code == 404


def test_pick_sentences_ignores_weak_overlap():
    passage = "Refunds are allowed within 30 days of purchase. Shipping takes five days."
    assert evidence.pick_sentences(passage, "Refunds are allowed within 30 days. [1]") == ["Refunds are allowed within 30 days of purchase."]
    assert evidence.pick_sentences(passage, "I couldn't find that.") == []

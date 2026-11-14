import io

import pytest
from docx import Document

from docmind import config, loader


def test_txt_is_split_into_numbered_sections(monkeypatch):
    monkeypatch.setattr(config, "SECTION_CHARS", 200)
    text = "\n".join(f"Paragraph number {i} with some filler words to take space." for i in range(20))
    pages = loader.load_document(text.encode(), "notes.txt")
    assert len(pages) > 1
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))
    assert all(p["source"] == "notes.txt" for p in pages)


def test_docx_text_and_tables_are_read():
    doc = Document()
    doc.add_paragraph("Refunds are allowed within 30 days.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Plan"
    table.rows[0].cells[1].text = "Premium"
    buffer = io.BytesIO()
    doc.save(buffer)
    pages = loader.load_document(buffer.getvalue(), "policy.docx")
    joined = " ".join(p["text"] for p in pages)
    assert "Refunds are allowed" in joined and "Plan | Premium" in joined  # one-row table: plain row


def test_html_keeps_main_text_and_drops_scripts_and_menus():
    html = "<html><body><nav>MENU</nav><script>var x=1;</script><main><p>Real content here.</p></main><footer>FOOT</footer></body></html>"
    text = loader.extract_html_text(html)
    assert "Real content here." in text
    assert "MENU" not in text and "var x" not in text and "FOOT" not in text


def test_unsupported_extension():
    with pytest.raises(ValueError):
        loader.load_document(b"data", "archive.zip")


@pytest.mark.parametrize("url", [
    "ftp://example.com/file", "file:///etc/passwd", "http://localhost/", "http://127.0.0.1/",
    "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://10.0.0.5/", "http://192.168.1.1/",
    "http://user:pass@8.8.8.8/", "http://8.8.8.8:22/", "http://[::ffff:127.0.0.1]/", "not a url",
])
def test_dangerous_urls_are_blocked(url):
    with pytest.raises(ValueError):
        loader.validate_url(url)


def test_public_ip_is_allowed():
    assert loader.validate_url("http://8.8.8.8/page") == "http://8.8.8.8/page"


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code, self.headers, self._body = status, headers or {}, body

    def iter_content(self, size):
        yield self._body

    def raise_for_status(self):
        pass

    def close(self):
        pass


def test_fetch_html(monkeypatch):
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(headers={"Content-Type": "text/html; charset=utf-8"}, body=b"<p>hi</p>"))
    name, body = loader.fetch_url("http://8.8.8.8/docs/page")
    assert name.endswith(".html") and body == b"<p>hi</p>"


def test_redirect_to_private_address_is_blocked(monkeypatch):
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(status=302, headers={"Location": "http://127.0.0.1/admin"}))
    with pytest.raises(ValueError):
        loader.fetch_url("http://8.8.8.8/start")


def test_too_large_and_wrong_type(monkeypatch):
    monkeypatch.setattr(config, "MAX_URL_BYTES", 10)
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(headers={"Content-Type": "text/html"}, body=b"x" * 50))
    with pytest.raises(ValueError):
        loader.fetch_url("http://8.8.8.8/big")
    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: FakeResponse(headers={"Content-Type": "application/zip"}, body=b"x"))
    with pytest.raises(ValueError):
        loader.fetch_url("http://8.8.8.8/pic")

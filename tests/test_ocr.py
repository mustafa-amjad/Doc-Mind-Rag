import io

import pytest
from pypdf import PdfWriter

from docmind import config, loader, ocr


FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",  # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/Library/Fonts/Arial.ttf",
]


def blank_pdf(pages: int) -> bytes:
    """A PDF whose pages have no text layer (like a scan)."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_scanned_pages_skipped_without_ocr():
    assert loader.load_document(blank_pdf(2), "scan.pdf", use_ocr=False) == []


def test_ocr_pages_are_added_and_marked(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda data, numbers: {n: f"scanned text {n}" for n in numbers})
    pages = loader.load_document(blank_pdf(2), "scan.pdf", use_ocr=True)
    assert [p["page"] for p in pages] == [1, 2]
    assert all(p["ocr"] and p["text"].startswith("scanned text") for p in pages)


def test_ocr_page_limit(monkeypatch):
    asked = []
    monkeypatch.setattr(config, "MAX_OCR_PAGES", 2)
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda data, numbers: asked.append(list(numbers)) or {n: "x" for n in numbers})
    loader.load_document(blank_pdf(5), "big.pdf", use_ocr=True)
    assert asked == [[1, 2]]  # only the first 2 pages were sent to OCR


def test_empty_ocr_result_is_dropped(monkeypatch):
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda data, numbers: {n: "" for n in numbers})
    assert loader.load_document(blank_pdf(1), "scan.pdf", use_ocr=True) == []


def test_real_ocr_reads_a_scanned_pdf():
    """Runs only when the OCR libraries and a TrueType font are installed."""
    pytest.importorskip("pypdfium2")
    if not ocr.is_available():
        pytest.skip("OCR libraries not installed")
    pil = pytest.importorskip("PIL.ImageFont")
    from PIL import Image, ImageDraw
    font = None
    for path in FONT_PATHS:
        try:
            font = pil.truetype(path, 40)
            break
        except OSError:
            continue
    if font is None:
        pytest.skip("no TrueType font available")
    image = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(image).text((30, 60), "Refunds are allowed within 30 days.", fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PDF")  # an image-only PDF = a scan
    pages = loader.load_document(buffer.getvalue(), "scan.pdf", use_ocr=True)
    assert pages and pages[0]["ocr"] and "refunds" in pages[0]["text"].lower()


def test_nearly_empty_text_layer_is_ocrd(monkeypatch):
    """A scan with only a stamp / page number in its text layer should still be OCR'd."""
    monkeypatch.setattr(loader, "PdfReader", lambda _: type("R", (), {"pages": [
        type("P", (), {"extract_text": lambda self: "12"})()]})())
    monkeypatch.setattr(ocr, "ocr_pdf_pages", lambda data, numbers: {n: "the real scanned paragraph" for n in numbers})
    pages = loader.load_document(b"%PDF", "scan.pdf", use_ocr=True)
    assert pages == [{"text": "the real scanned paragraph", "source": "scan.pdf", "page": 1, "ocr": True}]


def test_short_text_kept_without_ocr(monkeypatch):
    monkeypatch.setattr(loader, "PdfReader", lambda _: type("R", (), {"pages": [
        type("P", (), {"extract_text": lambda self: "Short title"})()]})())
    assert loader.load_document(b"%PDF", "a.pdf", use_ocr=False)[0]["text"] == "Short title"


def test_texts_handles_both_engine_formats():
    new_style = type("Out", (), {"txts": ("line one", "line two")})()
    legacy = ([[[0, 0], "line one", 0.9], [[0, 0], "line two", 0.8]], 0.1)
    assert ocr._texts(new_style) == ["line one", "line two"]
    assert ocr._texts(legacy) == ["line one", "line two"]
    assert ocr._texts(None) == [] and ocr._texts((None, 0.1)) == []


def test_broken_pdf_gives_friendly_error():
    with pytest.raises(ValueError, match="Could not read"):
        loader.load_document(b"not a pdf at all", "broken.pdf")

import io

import pytest
from pypdf import PdfWriter

from docmind import config, loader, ocr


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
    pytest.importorskip("rapidocr_onnxruntime")
    pil = pytest.importorskip("PIL.ImageFont")
    from PIL import Image, ImageDraw
    try:
        font = pil.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
    except OSError:
        pytest.skip("no TrueType font available")
    image = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(image).text((30, 60), "Refunds are allowed within 30 days.", fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, "PDF")  # an image-only PDF = a scan
    pages = loader.load_document(buffer.getvalue(), "scan.pdf", use_ocr=True)
    assert pages and pages[0]["ocr"] and "refunds" in pages[0]["text"].lower()

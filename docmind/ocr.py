"""Optional OCR for scanned PDFs (pages that have no text layer).
Needs: pip install -r requirements-ocr.txt  (pure pip, no Tesseract install)."""
import threading
from functools import lru_cache

from docmind import config

INSTALL_HINT = "OCR libraries are missing. Run: pip install -r requirements-ocr.txt"
_lock = threading.Lock()  # the OCR engine is not safe for two users at the same moment


def is_available() -> bool:
    """True if both OCR libraries can be imported."""
    try:
        import pypdfium2  # noqa: F401  (renders PDF pages to images)
        import rapidocr_onnxruntime  # noqa: F401  (reads text from images)
        return True
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _engine():
    """Load the OCR model once and reuse it."""
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_pdf_pages(file_bytes: bytes, page_numbers: list[int]) -> dict[int, str]:
    """OCR the given 1-based page numbers. Returns {page_number: text}."""
    if not is_available():
        raise RuntimeError(INSTALL_HINT)
    import numpy as np
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(file_bytes)
    results: dict[int, str] = {}
    try:
        with _lock:  # one OCR job at a time
            engine = _engine()
            for number in page_numbers:
                image = pdf[number - 1].render(scale=config.OCR_SCALE).to_pil().convert("RGB")  # page -> picture
                lines, _ = engine(np.array(image))  # list of [box, text, score] or None
                results[number] = "\n".join(line[1] for line in (lines or [])).strip()
    finally:
        pdf.close()
    return results

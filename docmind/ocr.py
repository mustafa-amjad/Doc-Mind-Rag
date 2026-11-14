"""OCR for scanned PDF pages (no usable text layer), image files and pictures inside documents.
Installed by requirements.txt (pure pip, no Tesseract install).

Works with the maintained `rapidocr` package (v2/v3, supports Python 3.10-3.14) and
falls back to the older `rapidocr_onnxruntime` package (Python <= 3.12) if that is
what is installed."""
import threading
from functools import lru_cache

from docmind import config

INSTALL_HINT = "OCR libraries are missing. Run: pip install -r requirements.txt"
_lock = threading.Lock()  # the OCR engine is not safe for two users at the same moment


def _import_engine_class():
    """Return the RapidOCR class from whichever package is installed, or None."""
    try:
        from rapidocr import RapidOCR  # current package name
        return RapidOCR
    except ImportError:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR  # legacy package name
        return RapidOCR
    except ImportError:
        return None


def is_available() -> bool:
    """True if a PDF renderer and an OCR engine can be imported."""
    try:
        import pypdfium2  # noqa: F401  (renders PDF pages to images)
    except ImportError:
        return False
    return _import_engine_class() is not None


@lru_cache(maxsize=1)
def _engine():
    """Load the OCR model once and reuse it."""
    engine_class = _import_engine_class()
    if engine_class is None:
        raise RuntimeError(INSTALL_HINT)
    try:  # rapidocr >= 2 logs every model load at INFO; keep the server console clean
        return engine_class(params={"Global.log_level": "warning"})
    except TypeError:  # legacy package has no `params` argument
        return engine_class()


def _texts(result) -> list[str]:
    """Pull the recognised lines out of either engine's result format."""
    if result is None:
        return []
    if hasattr(result, "txts"):  # rapidocr >= 2: RapidOCROutput object
        return [t for t in (result.txts or ()) if t]
    lines = result[0] if isinstance(result, tuple) else result  # legacy: (lines, elapsed)
    return [line[1] for line in (lines or []) if len(line) > 1 and line[1]]


def images_available() -> bool:
    """True if loose images (PNG/JPG, pictures inside PDF/DOCX) can be OCR'd. No PDF renderer needed."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return _import_engine_class() is not None


def ocr_images(images: list) -> list[str]:
    """OCR a list of PIL images. Returns one text per image (empty string if nothing was read)."""
    if not images:
        return []
    if not images_available():
        raise RuntimeError(INSTALL_HINT)
    import numpy as np

    texts = []
    with _lock:  # one OCR job at a time
        engine = _engine()
        for image in images:
            try:
                texts.append("\n".join(_texts(engine(np.array(image.convert("RGB"))))).strip())
            except Exception:  # one unreadable picture should not stop the file
                texts.append("")
    return texts


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
                if not 1 <= number <= len(pdf):
                    continue
                page = pdf[number - 1]
                try:
                    bitmap = page.render(scale=config.OCR_SCALE)
                    image = np.array(bitmap.to_pil().convert("RGB"))  # page -> picture
                finally:
                    page.close()
                results[number] = "\n".join(_texts(engine(image))).strip()
    finally:
        pdf.close()
    return results

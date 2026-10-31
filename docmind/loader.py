"""Step 1: read a PDF and give back text page by page."""
import io  # lets us treat raw bytes like a file

from pypdf import PdfReader  # PDF reading library


def load_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Return a list of {text, source, page}. Pages with no text are skipped."""
    reader = PdfReader(io.BytesIO(file_bytes))  # open the PDF from memory (no temp file needed)
    pages = []  # we collect good pages here
    for page_number, page in enumerate(reader.pages, start=1):  # start=1 so humans see page 1, not 0
        text = (page.extract_text() or "").strip()  # extract_text can return None -> use "" instead
        if not text:  # empty page (maybe a scanned image) -> skip it
            continue
        pages.append({"text": text, "source": filename, "page": page_number})
    return pages

"""Step 1: read documents (PDF, DOCX, TXT/MD, HTML, images, web links) and give back text 'pages'.
PDFs have real pages. Other formats are cut into sections of about 3000 characters,
and the section number is used as the 'page' in citations.

Beyond plain text it also reads:
- tables (PDF, DOCX, HTML): every row is written with its column headers, so rows stay searchable
- pictures inside PDF and DOCX files (charts, screenshots): OCR, plus an optional AI description
- image files (PNG, JPG, WEBP, BMP, TIFF, GIF): OCR, plus an optional AI description"""
import hashlib  # spots the same logo repeated on every page
import io  # lets us treat raw bytes like a file
import ipaddress  # checks if an IP address is public or private
import re
import socket  # resolves a website name to its IP addresses
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests  # downloads web pages
from bs4 import BeautifulSoup  # reads HTML
from docx import Document  # reads Word files
from docx.oxml.ns import qn  # Word XML tag names
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader  # PDF reading library

from docmind import config
from docmind import ocr as ocr_engine

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"]
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "html", *IMAGE_TYPES]  # shown in the upload box
IMAGE_TAG = "[Image]"  # marks text that came from a picture
DESCRIPTION_TAG = "[Image description]"  # marks the AI's description of a picture
OCR_MAX_SIDE = 2500  # very large pictures are shrunk before OCR (same text, much faster)

Describer = Callable[[object], str]  # takes a PIL image, returns a description ("" if none)


# ---------------- helpers ----------------
def _clean(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()  # None -> "", squeeze line breaks inside a cell


def format_table(rows: list[list]) -> str:
    """Write a table so each row keeps its meaning after chunking:
    '[Table] Plan | Price' then 'Plan: Basic | Price: $10' for every row."""
    rows = [[_clean(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]  # drop empty rows
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]  # make every row the same length
    keep = [col for col in range(width) if any(row[col] for row in rows)]  # drop empty columns
    rows = [[row[col] for col in keep] for row in rows]
    if len(rows) == 1 or len(keep) == 1:  # no header to repeat -> plain rows
        return "\n".join(" | ".join(cell for cell in row if cell) for row in rows)
    header = [cell or f"Column {number}" for number, cell in enumerate(rows[0], start=1)]
    lines = ["[Table] " + " | ".join(header)]
    for row in rows[1:]:
        lines.append(" | ".join(f"{name}: {cell}" for name, cell in zip(header, row) if cell))
    return "\n".join(lines)


def image_block(ocr_text: str, description: str) -> str:
    """Text for one picture: the words OCR read in it and/or the AI's description of it."""
    parts = []
    if ocr_text:
        parts.append(ocr_text)
    if description:
        parts.append(f"{DESCRIPTION_TAG} {description}")
    return f"{IMAGE_TAG}\n" + "\n".join(parts) if parts else ""


def split_sections(text: str, filename: str) -> list[dict]:
    """Pack paragraphs into sections of up to SECTION_CHARS. Section number = 'page'."""
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]  # non-empty lines
    pages, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) > config.SECTION_CHARS:  # section is full -> start next
            pages.append(current)
            current = ""
        current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        pages.append(current)
    result = []
    for number, section in enumerate(pages, start=1):
        page = {"text": section, "source": filename, "page": number}
        if IMAGE_TAG in section:  # this section includes text from a picture
            page["image"] = True
        result.append(page)
    return result


class ImageReader:
    """Reads pictures for ONE file: OCR (words in the picture) and/or an AI description (what it shows).
    Holds the per-file limits and skips tiny icons and repeated logos."""

    def __init__(self, use_ocr: bool = False, describe: Describer | None = None):
        self.use_ocr = use_ocr
        self.describe = describe
        self.ocr_left = config.MAX_IMAGES_PER_FILE
        self.describe_left = config.MAX_VISION_IMAGES
        self.seen: set[str] = set()  # fingerprints of pictures already read in this file

    @property
    def active(self) -> bool:
        return self.use_ocr or self.describe is not None

    def read(self, images: list, min_pixels: int | None = None) -> list[str]:
        """PIL images -> one text block per useful picture (empty pictures are left out)."""
        if not self.active:
            return []
        min_pixels = config.MIN_IMAGE_PIXELS if min_pixels is None else min_pixels
        picked = []
        for image in images:
            if image is None or min(image.size) < min_pixels:  # icon, bullet, divider line...
                continue
            fingerprint = hashlib.md5(image.tobytes()).hexdigest()
            if fingerprint in self.seen:  # same logo on every page -> read it once
                continue
            self.seen.add(fingerprint)
            image = image.convert("RGB")
            if max(image.size) > OCR_MAX_SIDE:
                image.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
            picked.append(image)
        if not picked:
            return []

        texts = [""] * len(picked)
        if self.use_ocr and self.ocr_left > 0:
            batch = picked[: self.ocr_left]
            self.ocr_left -= len(batch)
            try:
                texts[: len(batch)] = ocr_engine.ocr_images(batch)
            except RuntimeError as error:  # OCR libraries missing
                raise ValueError(str(error)) from None

        blocks = []
        for image, text in zip(picked, texts):
            description = ""
            if self.describe is not None and self.describe_left > 0:
                self.describe_left -= 1
                description = self.describe(image) or ""
            block = image_block(text, description)
            if block:
                blocks.append(block)
        return blocks


def _open_image(data: bytes):
    """Bytes -> PIL image, or None if Pillow cannot read it (EMF/WMF drawings, broken files)."""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception:
        return None


# ---------------- PDF ----------------
def _open_plumber(file_bytes: bytes):
    """pdfplumber finds tables. Optional: without it, tables are read as plain text."""
    try:
        import pdfplumber
        return pdfplumber.open(io.BytesIO(file_bytes))
    except Exception:
        return None


def _pdf_page_with_tables(plumber_page) -> str | None:
    """Page text in reading order with every table written row by row. None if the page has no table."""
    tables = []
    for table in plumber_page.find_tables():
        rows = table.extract()
        if len(rows) >= 2 and max(len(row) for row in rows) >= 2:  # a real table, not a box around text
            tables.append((table.bbox, rows))
    if not tables:
        return None
    tables.sort(key=lambda item: item[0][1])  # top to bottom
    x0, top, x1, bottom = plumber_page.bbox
    parts, cursor = [], top
    for (_, t_top, _, t_bottom), rows in tables:
        if t_top > cursor:  # text between the previous table and this one
            parts.append(plumber_page.crop((x0, cursor, x1, t_top)).extract_text() or "")
        parts.append(format_table(rows))
        cursor = max(cursor, t_bottom)
    if cursor < bottom:  # text after the last table
        parts.append(plumber_page.crop((x0, cursor, x1, bottom)).extract_text() or "")
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _pdf_page_images(page) -> list:
    """Pictures embedded in one PDF page (charts, screenshots, photos)."""
    images = []
    try:
        for item in page.images:
            try:
                images.append(item.image)  # decoded PIL image
            except Exception:  # unusual image encoding -> skip just this picture
                continue
    except Exception:  # damaged page resources
        pass
    return images


def load_pdf(file_bytes: bytes, filename: str, use_ocr: bool = False,
             describe: Describer | None = None) -> list[dict]:
    """Return a list of {text, source, page}. Pages without a text layer are OCR'd if use_ocr is on.
    Tables are kept row by row, and pictures on normal pages are read too."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))  # open the PDF from memory (no temp file needed)
    except Exception as error:  # broken or encrypted file -> friendly message instead of a crash
        raise ValueError(f"Could not read {filename}: {error}") from None
    pictures = ImageReader(use_ocr, describe)
    plumber = _open_plumber(file_bytes)
    pages, needs_ocr = {}, []  # page number -> page dict / pages that look scanned
    try:
        for page_number, page in enumerate(reader.pages, start=1):  # start=1 so humans see page 1, not 0
            try:
                text = (page.extract_text() or "").strip()  # extract_text can return None -> use "" instead
            except Exception:  # a single damaged page should not stop the whole file
                text = ""
            if len(text) < config.OCR_MIN_CHARS:  # empty or nearly empty (a scan with a stamp / page number)
                needs_ocr.append(page_number)
                if text:
                    pages[page_number] = {"text": text, "source": filename, "page": page_number}
                continue

            entry = {"text": text, "source": filename, "page": page_number}
            if plumber is not None and page_number <= min(len(plumber.pages), config.MAX_TABLE_PAGES):
                try:
                    with_tables = _pdf_page_with_tables(plumber.pages[page_number - 1])
                    if with_tables:
                        entry["text"] = with_tables
                except Exception:  # table detection failed -> keep the plain text
                    pass
            if pictures.active:  # charts / screenshots on a normal text page
                blocks = pictures.read(_pdf_page_images(page))
                if blocks:
                    entry["text"] += "\n\n" + "\n\n".join(blocks)
                    entry["image"] = True
            pages[page_number] = entry
    finally:
        if plumber is not None:
            plumber.close()

    if use_ocr and needs_ocr:  # read the scanned pages as pictures
        todo = needs_ocr[: config.MAX_OCR_PAGES]  # safety limit per file
        try:
            scanned = ocr_engine.ocr_pdf_pages(file_bytes, todo)
        except RuntimeError as error:  # OCR libraries missing
            raise ValueError(str(error)) from None
        for page_number, text in scanned.items():
            current = pages.get(page_number, {}).get("text", "")
            if text and len(text) > len(current):  # keep whichever version has more text
                pages[page_number] = {"text": text, "source": filename, "page": page_number, "ocr": True}
    return [pages[number] for number in sorted(pages)]  # keep reading order


# ---------------- Word ----------------
def _docx_images(element, document) -> list:
    """Pictures inside one Word paragraph or table, in order."""
    images = []
    for blip in element.iter(qn("a:blip")):
        part = document.part.related_parts.get(blip.get(qn("r:embed")))
        image = _open_image(part.blob) if part is not None else None
        if image is not None:
            images.append(image)
    return images


def _docx_table_rows(table: Table) -> list[list[str]]:
    rows = []
    for row in table.rows:
        cells, last = [], None
        for cell in row.cells:
            if cell._tc is last:  # merged cell: python-docx repeats it, keep it once
                continue
            last = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return rows


def load_docx(file_bytes: bytes, filename: str, use_ocr: bool = False,
              describe: Describer | None = None) -> list[dict]:
    """Paragraphs, tables and pictures in the order they appear in the document."""
    try:
        document = Document(io.BytesIO(file_bytes))
    except Exception as error:
        raise ValueError(f"Could not read {filename}: {error}") from None
    pictures = ImageReader(use_ocr, describe)
    lines = []
    for element in document.element.body.iterchildren():
        if element.tag == qn("w:p"):
            lines.append(Paragraph(element, document).text)
        elif element.tag == qn("w:tbl"):
            lines.append(format_table(_docx_table_rows(Table(element, document))))
        else:
            continue
        if pictures.active:
            lines.extend(pictures.read(_docx_images(element, document)))
    return split_sections("\n".join(lines), filename)


# ---------------- text and HTML ----------------
def load_text(file_bytes: bytes, filename: str) -> list[dict]:
    return split_sections(file_bytes.decode("utf-8", errors="replace"), filename)  # bad bytes -> "?" not a crash


def extract_html_text(html_text: str) -> str:
    """Visible text only: no scripts, menus or footers. Tables keep their rows, pictures keep their alt text."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()  # remove the tag and everything inside it
    main = soup.find("main") or soup.find("article") or soup.body or soup  # prefer the main content
    for table in reversed(main.find_all("table")):  # inner tables first, so nested tables are not lost
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
                for row in table.find_all("tr")]
        table.replace_with(soup.new_string("\n" + format_table(rows) + "\n"))
    for image in main.find_all("img"):
        alt = _clean(image.get("alt"))
        image.replace_with(soup.new_string(f"\n{IMAGE_TAG} {alt}\n" if len(alt) > 3 else ""))
    text = main.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n", text).strip()  # squeeze blank lines


def load_html(file_bytes: bytes, filename: str) -> list[dict]:
    return split_sections(extract_html_text(file_bytes.decode("utf-8", errors="replace")), filename)


# ---------------- image files ----------------
def load_image(file_bytes: bytes, filename: str, use_ocr: bool = False,
               describe: Describer | None = None) -> list[dict]:
    """PNG / JPG / WEBP / BMP / GIF / TIFF. Every frame of a multi-page TIFF becomes its own page."""
    pictures = ImageReader(use_ocr, describe)
    if not pictures.active:
        raise ValueError(f"{filename} is an image. Turn on OCR or AI image description to read it.")
    try:
        from PIL import Image, ImageSequence
        Image.MAX_IMAGE_PIXELS = 100_000_000  # refuse 'decompression bomb' files instead of freezing
        image = Image.open(io.BytesIO(file_bytes))
        frames = [frame.copy() for _, frame in zip(range(config.MAX_IMAGE_FRAMES), ImageSequence.Iterator(image))]
    except ImportError:
        raise ValueError(ocr_engine.INSTALL_HINT) from None
    except Exception as error:
        raise ValueError(f"Could not read {filename}: {type(error).__name__}") from None
    pages = []
    for number, frame in enumerate(frames, start=1):
        blocks = pictures.read([frame], min_pixels=1)  # a whole file is never 'just an icon'
        if not blocks:
            continue
        text = blocks[0].removeprefix(IMAGE_TAG).strip()
        page = {"text": text, "source": filename, "page": number, "image": True}
        if use_ocr and not text.startswith(DESCRIPTION_TAG):  # some of the text came from OCR
            page["ocr"] = True
        pages.append(page)
    return pages


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text, ".md": load_text,
           ".html": load_html, ".htm": load_html, **{f".{ext}": load_image for ext in IMAGE_TYPES}}
READS_PICTURES = {load_pdf, load_docx, load_image}  # loaders that take use_ocr / describe


def load_document(file_bytes: bytes, filename: str, use_ocr: bool = False,
                  describe: Describer | None = None) -> list[dict]:
    """Pick the right loader from the file extension.
    use_ocr: read scanned PDF pages and the words inside pictures.
    describe: optional function that explains a picture (charts, diagrams) in words."""
    extension = Path(filename).suffix.lower()
    loader = LOADERS.get(extension)
    if loader is None:
        raise ValueError(f"Unsupported file type: {filename}. Use PDF, DOCX, TXT, MD, HTML or an image.")
    if loader in READS_PICTURES:
        return loader(file_bytes, filename, use_ocr, describe)
    return loader(file_bytes, filename)




# ---------------- web links (with SSRF protection) ----------------
ALLOWED_PORTS = {None, 80, 443}  # no scanning of odd ports
CONTENT_TYPES = {"application/pdf": ".pdf", "text/plain": ".txt", "text/markdown": ".md",
                 "text/html": ".html", "application/xhtml+xml": ".html",
                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                 "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
                 "image/bmp": ".bmp", "image/tiff": ".tif"}


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:  # ::ffff:127.0.0.1 trick
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast  # False for private, loopback, link-local, reserved


def validate_url(url: str) -> str:
    """Only public http(s) sites. Blocks localhost, private networks and cloud metadata addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a full link starting with http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("Links with a username or password are not allowed.")
    if parsed.port not in ALLOWED_PORTS:
        raise ValueError("Only the standard web ports (80, 443) are allowed.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror:
        raise ValueError("Could not find that website.") from None
    if not addresses or not all(_is_public(a) for a in addresses):  # EVERY address must be public
        raise ValueError("That address is not allowed (private or internal network).")
    return url


def _name_from_url(url: str, extension: str) -> str:
    parsed = urlparse(url)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{parsed.hostname}{parsed.path}").strip("_")[:80] or "web_page"
    return name if name.lower().endswith(extension) else name + extension


def fetch_url(url: str) -> tuple[str, bytes]:
    """Download a web page / PDF / text file. Returns (filename, bytes) ready for load_document."""
    current = url.strip()
    for _ in range(config.MAX_REDIRECTS + 1):
        validate_url(current)  # check EVERY hop, a redirect could point inside your network
        try:
            response = requests.get(current, timeout=config.URL_TIMEOUT, stream=True, allow_redirects=False,
                                    headers={"User-Agent": "DocMind/1.0"})
            if response.status_code in (301, 302, 303, 307, 308) and response.headers.get("Location"):
                current = urljoin(current, response.headers["Location"])  # follow manually, then re-validate
                response.close()
                continue
            response.raise_for_status()
            body = b""
            for block in response.iter_content(65536):  # read in blocks so a huge file cannot fill memory
                body += block
                if len(body) > config.MAX_URL_BYTES:
                    raise ValueError(f"That page is too large (limit {config.MAX_URL_BYTES // 1_000_000} MB).")
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        except requests.RequestException as error:
            raise ValueError(f"Could not download that link ({type(error).__name__}).") from None
        extension = CONTENT_TYPES.get(content_type)
        if extension is None:
            raise ValueError(f"Unsupported content type: {content_type or 'unknown'}.")
        return _name_from_url(current, extension), body
    raise ValueError("Too many redirects.")

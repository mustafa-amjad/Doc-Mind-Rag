"""Step 1: read documents (PDF, DOCX, TXT/MD, HTML, web links) and give back text 'pages'.
PDFs have real pages. Other formats are cut into sections of about 3000 characters,
and the section number is used as the 'page' in citations."""
import io  # lets us treat raw bytes like a file
import ipaddress  # checks if an IP address is public or private
import re
import socket  # resolves a website name to its IP addresses
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests  # downloads web pages
from bs4 import BeautifulSoup  # reads HTML
from docx import Document  # reads Word files
from pypdf import PdfReader  # PDF reading library

from docmind import config

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "html"]  # shown in the upload box


# ---------------- helpers ----------------
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
    return [{"text": t, "source": filename, "page": number} for number, t in enumerate(pages, start=1)]


# ---------------- one loader per format ----------------
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


def load_docx(file_bytes: bytes, filename: str) -> list[dict]:
    document = Document(io.BytesIO(file_bytes))
    lines = [p.text for p in document.paragraphs]  # normal paragraphs
    for table in document.tables:  # tables are separate in Word files
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))  # one table row = one line
    return split_sections("\n".join(lines), filename)


def load_text(file_bytes: bytes, filename: str) -> list[dict]:
    return split_sections(file_bytes.decode("utf-8", errors="replace"), filename)  # bad bytes -> "?" not a crash


def extract_html_text(html_text: str) -> str:
    """Visible text only: no scripts, menus or footers."""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        tag.decompose()  # remove the tag and everything inside it
    main = soup.find("main") or soup.find("article") or soup.body or soup  # prefer the main content
    text = main.get_text(separator="\n")
    return re.sub(r"\n\s*\n+", "\n", text).strip()  # squeeze blank lines


def load_html(file_bytes: bytes, filename: str) -> list[dict]:
    return split_sections(extract_html_text(file_bytes.decode("utf-8", errors="replace")), filename)


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text, ".md": load_text,
           ".html": load_html, ".htm": load_html}


def load_document(file_bytes: bytes, filename: str) -> list[dict]:
    """Pick the right loader from the file extension."""
    loader = LOADERS.get(Path(filename).suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {filename}. Use PDF, DOCX, TXT, MD or HTML.")
    return loader(file_bytes, filename)


# ---------------- web links (with SSRF protection) ----------------
ALLOWED_PORTS = {None, 80, 443}  # no scanning of odd ports
CONTENT_TYPES = {"application/pdf": ".pdf", "text/plain": ".txt", "text/markdown": ".md",
                 "text/html": ".html", "application/xhtml+xml": ".html"}


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

"""Step 2: cut pages into small chunks the search can find.

Each page is first split at its headings, so one chunk never mixes the end of one section with the
start of the next. One-line sections are packed with their neighbour (a contents page stays one chunk),
and a section longer than CHUNK_SIZE is cut at paragraphs, then lines, then sentences, with a small
overlap so a sentence is not lost at the cut.
"""
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter  # smart cutter

from docmind import config

_NUMBERED = re.compile(r"\d+(\.\d+)*\.?\s+[A-Z]")  # "2. Returns and refunds", "3.1 Gift cards"
_CAPS = re.compile(r"[A-Z0-9][A-Z0-9 &'/\-]+")  # "NORTHWIND WAREHOUSE NOTICE"


def is_heading(line: str) -> bool:
    """A short line that looks like a section title: Markdown '#', a numbered title, or ALL CAPS."""
    line = line.strip()
    if not 3 <= len(line) <= 90 or "|" in line or line[-1] in ".,;:":
        return False  # empty, too long, a table row, or the end of a sentence
    if line.startswith("#"):
        return True
    words = line.split()
    if _NUMBERED.match(line) and len(words) <= 10:
        return True
    return _CAPS.fullmatch(line) is not None and len(words) >= 2 and any(c.isalpha() for c in line)


def split_points(lines: list[str]) -> set[int]:
    """Line numbers where a new section starts. A heading followed by a sub-heading splits once (at the
    first line); three or more heading lines in a row are a contents list, not sections, so no split."""
    flags = [is_heading(line) for line in lines]
    points, i = set(), 0
    while i < len(lines):
        if not flags[i]:
            i += 1
            continue
        run_end = i
        while run_end + 1 < len(lines) and flags[run_end + 1]:
            run_end += 1
        if run_end - i + 1 < 3:
            points.add(i)
        i = run_end + 1
    return points


def split_at_headings(text: str) -> list[str]:
    """Sections that each start at a heading (text before the first heading is its own section)."""
    lines = text.split("\n")
    points = split_points(lines)
    sections, current = [], []
    for number, line in enumerate(lines):
        if number in points and any(part.strip() for part in current):
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if any(part.strip() for part in current):
        sections.append("\n".join(current).strip())
    return sections


def pack_sections(sections: list[str], limit: int) -> list[str]:
    """Join a one-line section (a contents entry, a lone title) to its neighbour when both fit in one
    chunk, so they don't become tiny chunks. A section with a heading AND text keeps its own chunk."""
    packed: list[str] = []
    for section in sections:
        one_line = "\n" not in section or (packed and "\n" not in packed[-1])
        if packed and one_line and len(packed[-1]) + 1 + len(section) <= limit:
            packed[-1] += "\n" + section
        else:
            packed.append(section)
    return packed


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Turn pages into chunks. chunk_id == position in the list == id inside FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,  # max size of one chunk
        chunk_overlap=config.CHUNK_OVERLAP,  # overlap so a sentence is not cut in half
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph first, then line, sentence, word, letter
    )
    chunks = []  # final list
    for page in pages:  # go page by page so we never lose the page number
        pieces = []
        for block in pack_sections(split_at_headings(page["text"]), config.CHUNK_SIZE):
            pieces.extend([block] if len(block) <= config.CHUNK_SIZE else splitter.split_text(block))
        for piece in pieces:
            piece = piece.strip()  # remove spaces at the edges
            if len(piece) < config.MIN_CHUNK_CHARS:  # too tiny to be useful
                continue
            chunks.append(
                {
                    "chunk_id": len(chunks),  # next free number = its index in the list
                    "text": piece,
                    "source": page["source"],  # which file
                    "page": page["page"],  # which page (used for citations)
                    "ocr": bool(page.get("ocr")),  # True if the text came from OCR (can contain small errors)
                    "image": bool(page.get("image")),  # True if the page includes text read from a picture
                }
            )
    return chunks

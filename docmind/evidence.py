"""Find the exact sentences an answer used, so the source viewer can highlight them.

pick_sentences(): which sentences of a retrieved passage support the answer (word overlap).
pdf_highlights(): where those sentences sit on a PDF page, as boxes in PDF points.
"""
import io
import re

STOPWORDS = set(
    "a an and are as at be been but by can do does for from has have how i if in into is it its may "
    "must not of on or our so than that the their them then there these they this to was we were what "
    "when where which who will with within you your".split()
)
_WORD = re.compile(r"[a-z0-9$%.,]+")


def _words(text: str) -> list[str]:
    return [w.strip(".,") for w in _WORD.findall(text.lower()) if w.strip(".,")]


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if w not in STOPWORDS and len(w) > 1}


def split_sentences(text: str) -> list[str]:
    """Sentences, also breaking on line ends (headings, table rows)."""
    parts = []
    for line in text.splitlines():
        parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", line) if s.strip())
    return parts


def pick_sentences(passage: str, answer: str, limit: int = 3) -> list[str]:
    """The passage sentences that share the most content words with the answer (none if the overlap is weak)."""
    target = _content(re.sub(r"\[\d+\]", " ", answer))
    if not target:
        return []
    scored = []
    for sentence in split_sentences(passage):
        words = _content(sentence)
        if not words:
            continue
        hits = len(words & target)
        if hits >= 2:
            scored.append((hits / (len(words) ** 0.5), sentence))
    if not scored:
        return []
    best = max(score for score, _ in scored)
    keep = sorted((s for s in scored if s[0] >= best * 0.6), key=lambda s: -s[0])
    chosen = {sentence for _, sentence in keep[:limit]}
    return [s for s in split_sentences(passage) if s in chosen]  # keep reading order


def pdf_highlights(file_bytes: bytes, page_number: int, sentences: list[str]) -> dict:
    """Boxes (x0, top, x1, bottom in PDF points) covering each sentence on one page, merged per line."""
    import pdfplumber

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page = pdf.pages[page_number - 1]
        words = page.extract_words(use_text_flow=True)
        size = {"width": float(page.width), "height": float(page.height)}
    tokens = [(_words(w["text"]) or [""])[0] for w in words]
    boxes = []
    for sentence in sentences:
        target = _words(sentence)
        if len(target) < 2:
            continue
        span = _find_span(tokens, target)
        if span is None:
            continue
        lines: dict[int, list] = {}
        for w in words[span[0]:span[1]]:
            lines.setdefault(round(w["top"]), []).append(w)
        for line in lines.values():
            boxes.append({"x0": min(w["x0"] for w in line), "top": min(w["top"] for w in line),
                          "x1": max(w["x1"] for w in line), "bottom": max(w["bottom"] for w in line)})
    return {**size, "boxes": boxes}


def _find_span(tokens: list[str], target: list[str]) -> tuple[int, int] | None:
    """Where the sentence's words run on the page. Matches its first 6 words, then covers its full length."""
    probe = target[:min(6, len(target))]
    for start in range(len(tokens) - len(probe) + 1):
        if tokens[start:start + len(probe)] == probe:
            return start, min(len(tokens), start + len(target))
    return None

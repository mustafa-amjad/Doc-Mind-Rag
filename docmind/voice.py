"""Voice: speech-to-text for spoken questions (Whisper on Groq) and the text an answer is read aloud with.

The app reads answers aloud with the browser's own speech voice, so speaking needs no API call.
"""
import os
import re

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")  # Groq speech-to-text model
MAX_SPOKEN_CHARS = 1500  # long answers are cut at a sentence end before they are read aloud


def transcribe(client, audio: bytes, filename: str = "question.wav") -> str:
    """Speech -> text with Whisper on Groq. Returns the transcript ('' if nothing was heard).
    Raises on API errors (the app shows a friendly message)."""
    if not audio:
        return ""
    result = client.audio.transcriptions.create(file=(filename, audio), model=STT_MODEL,
                                                response_format="json", temperature=0)
    text = (getattr(result, "text", "") or "").strip()
    return "" if re.fullmatch(r"[\W_]*", text) else text  # silence comes back as "." or ""


def speech_text(answer: str) -> str:
    """An answer as plain sentences to read aloud: no [1] markers, no markdown, not too long."""
    text = re.sub(r"\s*\[\d+\]", "", answer)  # citation markers
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](link) -> label
    text = re.sub(r"[*_`#>|]+", " ", text)  # bold, italics, code, headings, quotes, table bars
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.MULTILINE)  # list bullets
    text = " ".join(text.split())
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # "jacket ." -> "jacket."
    if len(text) > MAX_SPOKEN_CHARS:
        cut = text[:MAX_SPOKEN_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        text = cut[: end + 1] if end > 200 else cut.rsplit(" ", 1)[0] + "…"
    return text

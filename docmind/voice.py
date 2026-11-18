"""Voice: speech-to-text for spoken questions (Whisper on Groq) and the text an answer is read aloud with.

Whisper invents text when it is given silence or noise ("Thank you.", or a random sentence in another
language), so a recording is checked for actual speech before it is sent, the language is fixed (English by
default) and Whisper's stock filler phrases are dropped. The app reads answers aloud with the browser's own
speech voice, so speaking needs no API call.
"""
import io
import os
import re
import wave

import numpy as np

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")  # Groq speech-to-text model
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")  # ISO-639-1 code; empty = let Whisper guess (less reliable)
STT_PROMPT = "A question about the user's documents."  # nudges Whisper towards a question, not a caption
MIN_SPEECH_SECONDS = 0.3  # less speech than this in a recording counts as "nothing was said"
MAX_SPOKEN_CHARS = 1500  # long answers are cut at a sentence end before they are read aloud

# what Whisper writes for silence or noise; a transcript that is only one of these is not a question
FILLER = {"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
          "thank you so much for watching", "please subscribe", "subtitles by the amara.org community",
          "you", "bye", "okay", "ok", "so", "uh", "um", "hmm"}


def speech_seconds(audio: bytes) -> float | None:
    """Seconds of audio in a 16-bit WAV that are clearly louder than its background (None if not 16-bit WAV).
    Each 30 ms frame counts as speech when it is well above the recording's own noise floor."""
    try:
        with wave.open(io.BytesIO(audio)) as recording:
            if recording.getsampwidth() != 2:
                return None
            rate, channels = recording.getframerate(), recording.getnchannels()
            samples = np.frombuffer(recording.readframes(recording.getnframes()), dtype=np.int16).astype(np.float32)
    except (wave.Error, EOFError):
        return None
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    frame = max(1, int(rate * 0.03))
    count = len(samples) // frame
    if count == 0:
        return 0.0
    rms = np.sqrt((samples[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
    floor = np.percentile(rms, 20)  # the quietest frames are the background
    loud = rms > max(150.0, floor * 1.6)  # 1.6x keeps speech over a loud fan; 150 rejects digital silence
    return float(loud.sum() * frame / rate)


def transcribe(client, audio: bytes, filename: str = "question.wav") -> str:
    """Speech -> text with Whisper on Groq. Returns '' when nothing was said (silence, noise, filler).
    Raises on API errors (the app shows a friendly message)."""
    if not audio:
        return ""
    heard = speech_seconds(audio)
    if heard is not None and heard < MIN_SPEECH_SECONDS:  # silence or background noise: don't ask Whisper
        return ""
    options = {"language": STT_LANGUAGE} if STT_LANGUAGE else {}
    result = client.audio.transcriptions.create(file=(filename, audio), model=STT_MODEL, response_format="json",
                                                temperature=0, prompt=STT_PROMPT, **options)
    text = (getattr(result, "text", "") or "").strip()
    if re.fullmatch(r"[\W_]*", text):  # only punctuation
        return ""
    if re.sub(r"[^\w\s.']", "", text.lower()).strip(" .") in FILLER:
        return ""
    return text


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

"""The LLM: rewrites follow-up questions and writes answers using ONLY the retrieved chunks."""
import re

import groq
from groq import Groq

from docmind import config


def friendly_error(error: Exception) -> str:
    """Turn an API exception into a message a user can act on."""
    if isinstance(error, groq.AuthenticationError):
        return "Groq rejected the API key (invalid or expired). Update GROQ_API_KEY in .env and restart the app."
    if isinstance(error, groq.RateLimitError):
        return "Groq rate limit reached. Wait a moment and try again."
    if isinstance(error, groq.APIConnectionError):
        return "Could not reach Groq. Check your internet connection and try again."
    if isinstance(error, groq.NotFoundError):
        return f"Model '{config.LLM_MODEL}' is not available on Groq. Set LLM_MODEL in .env to a current model."
    if isinstance(error, groq.APIStatusError):
        return f"Groq returned an error (HTTP {error.status_code}). Try again shortly."
    return f"The answer could not be generated ({type(error).__name__})."

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's documents. "
    "Use ONLY the numbered context below. If the answer is not in the context, say exactly: "
    "\"I couldn't find that in the uploaded documents.\" "
    "Cite sources with plain square brackets like [1] or [2] after each claim (never 【1】). "
    "Keep the answer short and clear."
)

REWRITE_PROMPT = (
    "Rewrite the user's last question as ONE standalone search question, using the chat history to "
    "resolve words like 'it', 'that', 'they', 'the second one'. Do NOT answer it. "
    "If it is already standalone, return it unchanged. Output ONLY the question."
)


VISION_PROMPT = (
    "You are indexing a picture from a document so it can be searched later. Describe it factually. "
    "If it is a chart or graph: give the chart type, title, axis labels, units, each series, the key values "
    "you can read, and the main trend or comparison. If it is a diagram or flowchart: list the parts and how "
    "they connect, in order. If it is a table: write it out row by row. If it is a photo or screenshot: say "
    "what it shows. Copy important visible text exactly. Never invent numbers you cannot read. "
    "Plain text only, no preamble, at most 200 words."
)


def image_to_data_url(image) -> str:
    """PIL image -> small JPEG data URL that a vision model accepts."""
    import base64
    import io

    picture = image.convert("RGB")
    picture.thumbnail((config.VISION_MAX_SIDE, config.VISION_MAX_SIDE))  # keeps aspect ratio, only shrinks
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def clean_model_text(text: str) -> str:
    """Drop <think>...</think> reasoning that some models print before the answer."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def normalize_citations(text: str) -> str:
    """Some models cite as 【1】, 【1†L3-L5】 or [1†source]. Turn every form into the plain [1] the UI expects."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    return re.sub(r"\[(\d+)†[^\]]*\]", r"[\1]", text)


def build_context(chunks: list[dict]) -> str:
    """Make a numbered block so the LLM can cite [1], [2]..."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        header = f"[{number}] ({chunk['source']}, page {chunk['page']})"  # label for each chunk
        blocks.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(blocks)


class LLM:
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)  # connect to Groq
        self.model = model

    def rewrite_question(self, question: str, history: list[dict]) -> str:
        """Chat memory for SEARCH: 'what about refunds for it?' -> 'What is the refund policy for gift cards?'"""
        if not history:  # first question, nothing to resolve
            return question
        recent = history[-config.HISTORY_TURNS:]  # only the last few messages
        transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:500]}" for m in recent)  # cut long answers
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": f"Chat history:\n{transcript}\n\nLast question: {question}"},
                ],
                temperature=0,
                max_tokens=120,  # a question is short
            )
            rewritten = (response.choices[0].message.content or "").strip().strip('"')
            return rewritten or question  # empty reply -> keep the original
        except Exception:  # API hiccup must never break the chat, so fall back to the raw question
            return question

    def describe_image(self, image) -> str:
        """Charts and diagrams: ask a vision model what the picture shows. Raises on API errors."""
        if min(image.size) < 32:  # the API refuses tiny images
            return ""
        response = self.client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
            ]}],
            temperature=0,
            max_tokens=600,
        )
        return clean_model_text(response.choices[0].message.content)

    def answer(self, question: str, chunks: list[dict], history: list[dict] | None = None) -> str:
        if not chunks:  # nothing retrieved -> do not waste an API call
            return "I couldn't find that in the uploaded documents."
        memory = [  # chat memory for the ANSWER: recent turns, content only
            {"role": m["role"], "content": m["content"][:800]} for m in (history or [])[-config.HISTORY_TURNS:]
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},  # the rules
                *memory,  # what we talked about before
                {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
            ],
            temperature=0,  # 0 = no creativity, stick to the facts
        )
        return normalize_citations(clean_model_text(response.choices[0].message.content))

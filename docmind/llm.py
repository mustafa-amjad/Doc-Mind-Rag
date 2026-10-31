"""Step 7: the LLM writes the final answer using ONLY the retrieved chunks."""
from groq import Groq

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions about the user's documents. "
    "Use ONLY the numbered context below. If the answer is not in the context, say exactly: "
    "\"I couldn't find that in the uploaded documents.\" "
    "Cite sources like [1] or [2] after each claim. Keep the answer short and clear."
)


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

    def answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:  # nothing retrieved -> do not waste an API call
            return "I couldn't find that in the uploaded documents."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},  # the rules
                {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
            ],
            temperature=0,  # 0 = no creativity, stick to the facts
        )
        return response.choices[0].message.content

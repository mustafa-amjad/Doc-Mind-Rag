"""Step 2: cut pages into small chunks the search can find."""
from langchain_text_splitters import RecursiveCharacterTextSplitter  # smart cutter

from docmind import config


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Turn pages into chunks. chunk_id == position in the list == id inside FAISS."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,  # max size of one chunk
        chunk_overlap=config.CHUNK_OVERLAP,  # overlap so a sentence is not cut in half
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph first, then line, sentence, word, letter
    )
    chunks = []  # final list
    for page in pages:  # go page by page so we never lose the page number
        for piece in splitter.split_text(page["text"]):  # cut this page
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
                }
            )
    return chunks

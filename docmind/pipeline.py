"""The boss file: connects every step into one RAG pipeline (with save / load)."""
from __future__ import annotations  # lets us use type hints without importing heavy libraries

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from docmind import config
from docmind.chunker import chunk_pages
from docmind.hybrid import rrf_fusion
from docmind.keyword_search import KeywordIndex
from docmind.loader import load_document
from docmind.vector_store import VectorStore

if TYPE_CHECKING:  # only for type hints, never imported at runtime (keeps tests light)
    from docmind.embedder import Embedder
    from docmind.llm import LLM
    from docmind.reranker import Reranker


class RAGPipeline:
    def __init__(self, embedder: Embedder, reranker: Reranker, llm: LLM):
        self.embedder = embedder  # shared heavy models (loaded once by the app)
        self.reranker = reranker
        self.llm = llm
        self.chunks: list[dict] = []  # this chat's chunks
        self._vectors: np.ndarray | None = None  # kept so we can save them and add more files later
        self.store: VectorStore | None = None  # vector index
        self.keyword: KeywordIndex | None = None  # keyword index
        self.file_names: list[str] = []  # for the UI
        self.page_count = 0

    @property
    def is_ready(self) -> bool:
        return self.store is not None and len(self.chunks) > 0

    def stats(self) -> dict:
        return {"files": len(self.file_names), "pages": self.page_count, "chunks": len(self.chunks),
                "index": self.store.mode if self.store else "none", "names": list(self.file_names)}

    def _build_indexes(self, chunks: list[dict], vectors: np.ndarray) -> None:
        """Fill FAISS + BM25 from chunks and vectors. Used by ingest AND load (no re-embedding on load)."""
        self.chunks = chunks
        self._vectors = vectors
        mode = "flat" if len(chunks) < config.FLAT_LIMIT else "hnsw"  # small data -> exact, big data -> HNSW
        self.store = VectorStore(vectors.shape[1], mode)
        self.store.add(vectors)
        self.keyword = KeywordIndex([c["text"] for c in chunks])

    def _safe_describer(self, warnings: list[str]):
        """Wrap the vision call so one API problem never stops indexing. After the first failure it stops
        calling the API for the rest of this upload and leaves a warning for the UI."""
        state = {"failed": False}

        def describe(image) -> str:
            if state["failed"]:
                return ""
            try:
                return self.llm.describe_image(image)
            except Exception as error:
                state["failed"] = True
                from docmind.llm import friendly_error  # imported here so tests do not need the groq package
                message = friendly_error(error)
                if "not available" in message or "error (HTTP 400)" in message:
                    message = (f"Vision model '{config.VISION_MODEL}' could not read images. "
                               "Set VISION_MODEL in .env to a Groq model that accepts images.")
                warnings.append(f"AI image description was skipped: {message}")
                return ""

        return describe

    def ingest(self, files: list[tuple[str, bytes]], append: bool = False, ocr: bool = False,
               describe_images: bool = False) -> dict:
        """files = [(filename, pdf_bytes), ...]. append=True adds to the existing documents.
        ocr: read scanned pages and words in pictures. describe_images: let the AI explain charts/diagrams."""
        warnings: list[str] = []
        describe = self._safe_describer(warnings) if describe_images else None
        pages = []
        for filename, data in files:
            pages.extend(load_document(data, filename, ocr, describe))  # every file (PDF, DOCX, images...)
        if not pages:  # all pages empty -> probably scanned images
            raise ValueError(
                (warnings[0] if warnings else "No text found, even with OCR.") if (ocr or describe_images)
                else "No text found. This looks like a scanned PDF or an image: turn on OCR in the Documents panel."
            )
        new_chunks = chunk_pages(pages)  # cut into chunks
        if not new_chunks:
            raise ValueError("Text was found but it was too short to make chunks.")
        new_vectors = self.embedder.encode([c["text"] for c in new_chunks])  # embed ONLY the new chunks

        if append and self.is_ready:  # keep old documents and add the new ones
            offset = len(self.chunks)  # new ids continue after the old ones (id == list position)
            for chunk in new_chunks:
                chunk["chunk_id"] += offset
            chunks = self.chunks + new_chunks
            vectors = np.vstack([self._vectors, new_vectors])  # stack old + new vectors
            self.file_names += [n for n, _ in files if n not in self.file_names]
            self.page_count += len(pages)
        else:  # replace everything
            chunks, vectors = new_chunks, new_vectors
            self.file_names = [n for n, _ in files]
            self.page_count = len(pages)

        self._build_indexes(chunks, vectors)
        stats = self.stats()
        stats["ocr_pages"] = sum(1 for page in pages if page.get("ocr"))  # shown in the UI message
        stats["image_pages"] = sum(1 for page in pages if page.get("image"))
        stats["warnings"] = warnings
        return stats

    # ---------------- save / load (this is what makes old chats usable again) ----------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")
        np.save(folder / "vectors.npy", self._vectors)
        (folder / "meta.json").write_text(
            json.dumps({"file_names": self.file_names, "page_count": self.page_count}), encoding="utf-8"
        )

    def load(self, folder: Path) -> dict:
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy").astype("float32")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.file_names = meta["file_names"]
        self.page_count = meta["page_count"]
        self._build_indexes(chunks, vectors)  # rebuild indexes from saved data (fast, no embedding)
        return self.stats()

    # ---------------- search + answer ----------------
    def retrieve(self, question: str, use_hybrid: bool = True, use_rerank: bool = True,
                 final_k: int | None = None) -> list[dict]:
        """Find the best chunks. The flags let you A/B test each technique."""
        final_k = final_k or config.FINAL_K  # how many chunks to keep
        query_vector = self.embedder.encode([question])[0]  # embed the question
        vector_ids = [i for i, _ in self.store.search(query_vector, config.RETRIEVE_K)]  # meaning search

        if use_hybrid:  # add keyword search and merge
            keyword_ids = [i for i, _ in self.keyword.search(question, config.RETRIEVE_K)]
            fused = rrf_fusion([vector_ids, keyword_ids], config.RRF_K)  # merge by rank
            candidate_ids = [i for i, _ in fused[: config.RETRIEVE_K]]
        else:  # vector only
            candidate_ids = vector_ids

        candidates = [self.chunks[i] for i in candidate_ids]  # ids -> real chunks
        if use_rerank:  # strict teacher picks the final few
            return self.reranker.rerank(question, candidates, final_k)
        return candidates[:final_k]

    def ask(self, question: str, use_hybrid: bool = True, use_rerank: bool = True,
            final_k: int | None = None, history: list[dict] | None = None) -> dict:
        """Full RAG with chat memory: rewrite follow-up, retrieve, generate."""
        history = history or []
        start = time.perf_counter()
        search_query = self.llm.rewrite_question(question, history)  # follow-up -> standalone question
        chunks = self.retrieve(search_query, use_hybrid, use_rerank, final_k)  # step 1: retrieve
        retrieve_ms = (time.perf_counter() - start) * 1000
        answer = self.llm.answer(question, chunks, history)  # step 2: generate (sees recent chat too)
        total_ms = (time.perf_counter() - start) * 1000
        return {"answer": answer, "sources": chunks, "retrieve_ms": retrieve_ms, "total_ms": total_ms,
                "search_query": search_query}

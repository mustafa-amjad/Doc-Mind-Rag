"""The boss file: connects every step into one RAG pipeline."""
import time

from docmind import config
from docmind.chunker import chunk_pages
from docmind.embedder import Embedder
from docmind.hybrid import rrf_fusion
from docmind.keyword_search import KeywordIndex
from docmind.llm import LLM
from docmind.loader import load_pdf
from docmind.reranker import Reranker
from docmind.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, embedder: Embedder, reranker: Reranker, llm: LLM):
        self.embedder = embedder  # shared heavy models (loaded once by the app)
        self.reranker = reranker
        self.llm = llm
        self.chunks: list[dict] = []  # this user's chunks
        self.store: VectorStore | None = None  # vector index (built in ingest)
        self.keyword: KeywordIndex | None = None  # keyword index (built in ingest)

    @property
    def is_ready(self) -> bool:
        return self.store is not None and len(self.chunks) > 0

    def ingest(self, files: list[tuple[str, bytes]]) -> dict:
        """files = [(filename, pdf_bytes), ...]. Builds both indexes. Returns stats for the UI."""
        pages = []
        for filename, data in files:
            pages.extend(load_pdf(data, filename))  # read every PDF
        if not pages:  # all pages empty -> probably scanned images
            raise ValueError("No text found. The PDF may be scanned images (OCR is not supported yet).")

        self.chunks = chunk_pages(pages)  # cut into chunks
        if not self.chunks:
            raise ValueError("Text was found but it was too short to make chunks.")

        texts = [c["text"] for c in self.chunks]
        vectors = self.embedder.encode(texts)  # meaning vectors

        mode = "flat" if len(self.chunks) < config.FLAT_LIMIT else "hnsw"  # small data -> exact, big data -> HNSW
        self.store = VectorStore(self.embedder.dim, mode)
        self.store.add(vectors)  # fill the vector index
        self.keyword = KeywordIndex(texts)  # fill the keyword index
        return {"files": len(files), "pages": len(pages), "chunks": len(self.chunks), "index": mode}

    def retrieve(self, question: str, use_hybrid: bool = True, use_rerank: bool = True, final_k: int | None = None) -> list[dict]:
        """Find the best chunks. The two flags let you A/B test each technique."""
        final_k = final_k or config.FINAL_K  # how many chunks to keep (UI slider overrides the default)
        query_vector = self.embedder.encode([question])[0]  # embed the question
        vector_ids = [i for i, _ in self.store.search(query_vector, config.RETRIEVE_K)]  # meaning search

        if use_hybrid:  # add keyword search and merge
            keyword_ids = [i for i, _ in self.keyword.search(question, config.RETRIEVE_K)]
            fused = rrf_fusion([vector_ids, keyword_ids], config.RRF_K)  # merge by rank
            candidate_ids = [i for i, _ in fused[: config.RETRIEVE_K]]  # keep the top candidates
        else:  # vector only
            candidate_ids = vector_ids

        candidates = [self.chunks[i] for i in candidate_ids]  # ids -> real chunks
        if use_rerank:  # strict teacher picks the final few
            return self.reranker.rerank(question, candidates, final_k)
        return candidates[:final_k]  # no teacher, just take the top few

    def ask(self, question: str, use_hybrid: bool = True, use_rerank: bool = True, final_k: int | None = None) -> dict:
        """Full RAG: retrieve, then generate. Returns answer, sources and timings."""
        start = time.perf_counter()
        chunks = self.retrieve(question, use_hybrid, use_rerank, final_k)  # step 1: retrieve
        retrieve_ms = (time.perf_counter() - start) * 1000
        answer = self.llm.answer(question, chunks)  # step 2: generate
        total_ms = (time.perf_counter() - start) * 1000
        return {"answer": answer, "sources": chunks, "retrieve_ms": retrieve_ms, "total_ms": total_ms}

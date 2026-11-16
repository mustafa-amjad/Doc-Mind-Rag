import zlib

import numpy as np

from docmind.keyword_search import tokenize
from docmind.pipeline import RAGPipeline


class FakeEmbedder:
    """Tiny bag-of-words embedder so tests need no model download."""
    dim = 64

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in tokenize(text):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out


CHUNKS = [
    {"chunk_id": 0, "text": "Refunds are allowed within 30 days of purchase.", "source": "a.pdf", "page": 1},
    {"chunk_id": 1, "text": "Shipping takes five to seven business days.", "source": "a.pdf", "page": 2},
    {"chunk_id": 2, "text": "We accept Visa, Mastercard and PayPal payments.", "source": "a.pdf", "page": 3},
]


def make_pipeline():
    pipe = RAGPipeline(FakeEmbedder(), None, None)
    pipe.file_names, pipe.page_count = ["a.pdf"], 3
    embedder = FakeEmbedder()
    pipe._build_indexes([dict(c) for c in CHUNKS], embedder.encode([c["text"] for c in CHUNKS]))
    return pipe


def test_save_load_roundtrip(tmp_path):
    pipe = make_pipeline()
    pipe.save(tmp_path / "chat1")
    restored = RAGPipeline(FakeEmbedder(), None, None)
    stats = restored.load(tmp_path / "chat1")
    assert stats["chunks"] == 3 and stats["names"] == ["a.pdf"] and restored.is_ready
    before = [c["chunk_id"] for c in pipe.retrieve("refund within 30 days", True, False, 2)]
    after = [c["chunk_id"] for c in restored.retrieve("refund within 30 days", True, False, 2)]
    assert before == after and before[0] == 0  # same answer after reload, refund chunk first


def test_hybrid_finds_exact_word():
    pipe = make_pipeline()
    top = pipe.retrieve("PayPal", use_hybrid=True, use_rerank=False, final_k=1)
    assert top[0]["chunk_id"] == 2


def two_file_pipeline():
    pipe = RAGPipeline(FakeEmbedder(), None, None)
    chunks = [dict(c) for c in CHUNKS] + [
        {"chunk_id": 3, "text": "Refunds for staff purchases take 30 days to process.", "source": "staff.docx", "page": 1}]
    pipe.file_names, pipe.page_count = ["a.pdf", "staff.docx"], 4
    pipe._build_indexes(chunks, FakeEmbedder().encode([c["text"] for c in chunks]))
    return pipe


def test_filter_searches_only_the_chosen_files():
    pipe = two_file_pipeline()
    for hybrid in (True, False):
        found = pipe.retrieve("refunds within 30 days", hybrid, False, 4, sources=["staff.docx"])
        assert [c["source"] for c in found] == ["staff.docx"]  # a.pdf's refund chunk is never a candidate
    everything = pipe.retrieve("refunds within 30 days", True, False, 4)
    assert {c["source"] for c in everything} == {"a.pdf", "staff.docx"}


def test_filter_with_unknown_file_returns_nothing():
    assert two_file_pipeline().retrieve("refunds", True, False, 4, sources=["missing.pdf"]) == []

from docmind.chunker import chunk_pages


def test_chunks_keep_page_and_ids():
    pages = [{"text": "Refunds are allowed within 30 days of purchase. " * 40, "source": "a.pdf", "page": 3}]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1  # long text must be split
    assert all(c["page"] == 3 and c["source"] == "a.pdf" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))  # id == list position

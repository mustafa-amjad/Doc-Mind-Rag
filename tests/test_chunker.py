from docmind.chunker import chunk_pages


def test_chunks_keep_page_and_ids():
    pages = [{"text": "Refunds are allowed within 30 days of purchase. " * 40, "source": "a.pdf", "page": 3}]
    chunks = chunk_pages(pages)
    assert len(chunks) > 1  # long text must be split
    assert all(c["page"] == 3 and c["source"] == "a.pdf" for c in chunks)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))  # id == list position


def test_headings_start_new_chunks():
    body = "Returns are accepted within 30 days of delivery when tags are attached. " * 8  # ~580 chars
    text = "2. Returns and refunds\n" + body + "\n3. Gift cards\nGift cards never expire and have no fees at all."
    chunks = chunk_pages([{"text": text, "source": "h.pdf", "page": 1}])
    assert len(chunks) == 2  # would fit in one 800-character chunk, but the heading starts a new one
    assert chunks[0]["text"].startswith("2. Returns") and chunks[1]["text"].startswith("3. Gift cards")


def test_short_sections_are_packed_together():
    text = "Contents\n1. Orders and payment\n2. Returns and refunds\n3. Gift cards\n4. Shipping"
    chunks = chunk_pages([{"text": text, "source": "c.pdf", "page": 1}])
    assert len(chunks) == 1 and "4. Shipping" in chunks[0]["text"]  # a contents page stays one chunk


def test_heading_detection():
    from docmind.chunker import is_heading

    for line in ["2. Returns and refunds", "3.1 Gift cards", "## Shipping", "NORTHWIND WAREHOUSE NOTICE"]:
        assert is_heading(line), line
    for line in ["Q1", "Items must be unworn.", "Product: Trail Jacket | Price: $189", "2 hours later the order ships",
                 "We ship to the United States, Canada and the United Kingdom"]:
        assert not is_heading(line), line


def test_contents_list_is_not_split():
    from docmind.chunker import split_at_headings

    text = ("Customer Policy Handbook\nContents\n1. Orders and payment\n2. Returns and refunds\n3. Gift cards\n"
            "4. Shipping\nThis sample handbook is fictional.")
    assert len(split_at_headings(text)) == 1
    nested = "Intro text here.\n2. Returns\n2.1 Policy\nReturns are accepted within 30 days."
    assert split_at_headings(nested) == ["Intro text here.", "2. Returns\n2.1 Policy\nReturns are accepted within 30 days."]

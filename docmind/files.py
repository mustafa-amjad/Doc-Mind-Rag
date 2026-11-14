"""How an indexed file was read, shared by the Streamlit app and the web API."""


def how_read(chunks: list[dict], name: str) -> list[str]:
    """How a file was read: Text, Table, Picture and/or OCR (shown as tags in both apps)."""
    mine = [c for c in chunks if c["source"] == name]
    tags = []
    if any("[Table]" in c["text"] for c in mine):
        tags.append("Table")
    if any(c.get("image") for c in mine):
        tags.append("Picture")
    if any(c.get("ocr") for c in mine):
        tags.append("OCR")
    # "Text" = the file has its own typed text (not only words read from scans or pictures)
    typed = any(not c.get("ocr") and len(c["text"].split("[Image]")[0].strip()) > 40 for c in mine)
    if typed or not tags:
        tags.insert(0, "Text")
    return tags

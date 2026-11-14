"""Tables, image files, and pictures inside PDF / Word files."""
import io
import zlib

import pytest
from docx import Document
from PIL import Image, ImageDraw

from docmind import chunker, config, loader, ocr
from docmind.llm import clean_model_text, image_to_data_url
from docmind.pipeline import RAGPipeline

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",  # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/Library/Fonts/Arial.ttf",
]


# ---------------- helpers that build test documents in memory ----------------
def picture(width=300, height=200, color=(200, 30, 30)) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(image).rectangle((20, 20, width - 20, height - 20), fill=color)
    return image


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def text_pdf(with_table: bool = True, with_image: bool = True) -> bytes:
    """A real PDF with a text layer, a ruled 3x3 table and an embedded picture (written by hand)."""
    ops = ["BT /F1 11 Tf 50 760 Td (Quarterly sales report for the northern and southern regions.) Tj ET"]
    if with_table:
        x_lines, y_lines = [50, 200, 350, 500], [700, 670, 640, 610]
        for x in x_lines:
            ops.append(f"{x} 610 m {x} 700 l S")
        for y in y_lines:
            ops.append(f"50 {y} m 500 {y} l S")
        cells = [["Region", "Sales", "Growth"], ["North", "120", "5%"], ["South", "95", "-2%"]]
        for row_number, row in enumerate(cells):
            for col_number, value in enumerate(row):
                ops.append(f"BT /F1 10 Tf {x_lines[col_number] + 8} {y_lines[row_number] - 20} Td ({value}) Tj ET")
        ops.append("BT /F1 11 Tf 50 580 Td (Text written below the table stays in order.) Tj ET")
    if with_image:
        ops.append("q 300 0 0 200 50 300 cm /Im1 Do Q")
    content = "\n".join(ops).encode()

    pixels = picture().tobytes()
    image_data = zlib.compress(pixels)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> /XObject << /Im1 6 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /XObject /Subtype /Image /Width 300 /Height 200 /ColorSpace /DeviceRGB "
        b"/BitsPerComponent 8 /Filter /FlateDecode /Length %d >>\nstream\n" % len(image_data)
        + image_data + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
    for offset in offsets:
        out.write(b"%010d 00000 n \n" % offset)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
    return out.getvalue()


def docx_with_table_and_pictures(pictures: list[Image.Image]) -> bytes:
    document = Document()
    document.add_paragraph("Our price list is below.")
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [("Plan", "Price"), ("Basic", "$10"), ("Pro", "$25")]):
        row.cells[0].text, row.cells[1].text = values
    document.add_paragraph("The chart after this line shows growth.")
    for image in pictures:
        document.add_picture(io.BytesIO(png_bytes(image)))
    document.add_paragraph("End of document.")
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def fake_ocr(monkeypatch):
    """Pretend OCR: every picture reads as 'chart text N'. Records how many pictures were sent."""
    calls = []

    def run(images):
        calls.append(len(images))
        return [f"chart text {sum(calls) - len(images) + n}" for n in range(1, len(images) + 1)]

    monkeypatch.setattr(ocr, "ocr_images", run)
    return calls


# ---------------- tables ----------------
def test_format_table_repeats_headers_on_every_row():
    text = loader.format_table([["Plan", "Price", ""], ["Basic", "$10", ""], ["", "", ""], ["Pro", "$25", None]])
    assert text.splitlines() == ["[Table] Plan | Price", "Plan: Basic | Price: $10", "Plan: Pro | Price: $25"]


def test_format_table_edge_cases():
    assert loader.format_table([]) == ""
    assert loader.format_table([["Only", "one row"]]) == "Only | one row"
    assert loader.format_table([["", "Price"], ["Basic", "$10"]]).splitlines()[0] == "[Table] Column 1 | Price"
    assert loader.format_table([["a\nb", "c"], ["d", "e"]]).splitlines()[0] == "[Table] a b | c"  # no line breaks in cells


def test_html_tables_keep_headers_and_images_keep_alt_text():
    html = """<main><p>Intro</p><table><tr><th>City</th><th>Population</th></tr>
              <tr><td>Lahore</td><td>13 million</td></tr></table>
              <img src="x.png" alt="Bar chart of population by city"><img src="y.png" alt=""></main>"""
    text = loader.extract_html_text(html)
    assert "City: Lahore | Population: 13 million" in text
    assert "[Image] Bar chart of population by city" in text
    assert text.index("Intro") < text.index("City: Lahore")


def test_docx_tables_are_in_reading_order_with_headers():
    pages = loader.load_document(docx_with_table_and_pictures([]), "prices.docx")
    text = pages[0]["text"]
    assert "Plan: Pro | Price: $25" in text
    assert text.index("price list") < text.index("Plan: Basic") < text.index("chart after this line")
    assert "image" not in pages[0]


def test_pdf_table_is_read_row_by_row_in_order():
    pytest.importorskip("pdfplumber")
    pages = loader.load_document(text_pdf(with_image=False), "report.pdf")
    text = pages[0]["text"]
    assert "Region: North | Sales: 120 | Growth: 5%" in text
    assert "Region: South | Sales: 95 | Growth: -2%" in text
    assert text.index("Quarterly sales") < text.index("Region: North") < text.index("below the table")


def test_pdf_without_tables_keeps_plain_text():
    pages = loader.load_document(text_pdf(with_table=False, with_image=False), "plain.pdf")
    assert pages[0]["text"].startswith("Quarterly sales report") and "[Table]" not in pages[0]["text"]


# ---------------- pictures inside PDF and Word ----------------
def test_pdf_picture_on_a_text_page_is_read(fake_ocr):
    pages = loader.load_document(text_pdf(), "report.pdf", use_ocr=True)
    assert len(pages) == 1 and pages[0]["image"] is True
    assert "Quarterly sales" in pages[0]["text"] and "[Image]\nchart text 1" in pages[0]["text"]
    assert not pages[0].get("ocr")  # the page itself is not a scan


def test_pdf_pictures_ignored_when_ocr_is_off(fake_ocr):
    pages = loader.load_document(text_pdf(), "report.pdf", use_ocr=False)
    assert "[Image]" not in pages[0]["text"] and fake_ocr == []


def test_docx_pictures_are_read_where_they_appear(fake_ocr):
    pages = loader.load_document(docx_with_table_and_pictures([picture()]), "deck.docx", use_ocr=True)
    text = pages[0]["text"]
    assert pages[0]["image"] is True
    assert text.index("chart after this line") < text.index("chart text 1") < text.index("End of document")


def test_repeated_logo_and_tiny_icons_are_read_once(fake_ocr):
    logo = picture()
    icon = picture(40, 40, (0, 0, 255))
    loader.load_document(docx_with_table_and_pictures([logo, logo.copy(), icon]), "logos.docx", use_ocr=True)
    assert sum(fake_ocr) == 1


def test_picture_limit_per_file(fake_ocr, monkeypatch):
    monkeypatch.setattr(config, "MAX_IMAGES_PER_FILE", 2)
    images = [picture(color=(n * 40, 10, 10)) for n in range(4)]
    loader.load_document(docx_with_table_and_pictures(images), "many.docx", use_ocr=True)
    assert sum(fake_ocr) == 2


def test_ai_description_is_added_and_limited(monkeypatch):
    monkeypatch.setattr(config, "MAX_VISION_IMAGES", 1)
    described = []

    def describe(image):
        described.append(image.size)
        return "A bar chart: North 120, South 95."

    images = [picture(color=(10, 200, 10)), picture(color=(10, 10, 200))]
    pages = loader.load_document(docx_with_table_and_pictures(images), "deck.docx", describe=describe)
    assert len(described) == 1
    assert "[Image description] A bar chart: North 120, South 95." in pages[0]["text"]


# ---------------- image files ----------------
def test_image_file_needs_ocr_or_description():
    with pytest.raises(ValueError, match="Turn on OCR"):
        loader.load_document(png_bytes(picture()), "chart.png")


def test_image_file_is_read_with_ocr(fake_ocr):
    pages = loader.load_document(png_bytes(picture()), "chart.png", use_ocr=True)
    assert pages == [{"text": "chart text 1", "source": "chart.png", "page": 1, "image": True, "ocr": True}]


def test_image_file_with_description_only():
    pages = loader.load_document(png_bytes(picture()), "diagram.jpg", describe=lambda image: "A flowchart.")
    assert pages == [{"text": "[Image description] A flowchart.", "source": "diagram.jpg", "page": 1, "image": True}]


def test_small_image_file_is_still_read(fake_ocr):
    pages = loader.load_document(png_bytes(picture(60, 40)), "label.png", use_ocr=True)
    assert pages and pages[0]["text"] == "chart text 1"


def test_multi_page_tiff_gives_one_page_per_frame(fake_ocr):
    frames = [picture(color=(200, 0, 0)), picture(color=(0, 200, 0))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "TIFF", save_all=True, append_images=frames[1:])
    pages = loader.load_document(buffer.getvalue(), "scan.tiff", use_ocr=True)
    assert [p["page"] for p in pages] == [1, 2]


def test_broken_image_gives_friendly_error():
    with pytest.raises(ValueError, match="Could not read"):
        loader.load_document(b"not an image", "photo.jpg", use_ocr=True)


def test_image_links_are_accepted(monkeypatch):
    class Response:
        status_code, headers = 200, {"Content-Type": "image/png"}

        def iter_content(self, size):
            yield b"png"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(loader.requests, "get", lambda *a, **k: Response())
    name, _ = loader.fetch_url("http://8.8.8.8/chart")
    assert name.endswith(".png")


def test_real_ocr_reads_an_image_file():
    """Runs only when the OCR libraries and a TrueType font are installed."""
    if not ocr.images_available():
        pytest.skip("OCR libraries not installed")
    from PIL import ImageFont
    font = None
    for path in FONT_PATHS:
        try:
            font = ImageFont.truetype(path, 40)
            break
        except OSError:
            continue
    if font is None:
        pytest.skip("no TrueType font available")
    image = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(image).text((30, 60), "Refunds are allowed within 30 days.", fill="black", font=font)
    pages = loader.load_document(png_bytes(image), "notice.png", use_ocr=True)
    assert pages and "refunds" in pages[0]["text"].lower()


# ---------------- chunks, pipeline and vision helpers ----------------
def test_chunks_carry_the_picture_flag():
    chunks = chunker.chunk_pages([
        {"text": "[Image]\nRevenue grew 20 percent in the third quarter.", "source": "a.png", "page": 1, "image": True},
        {"text": "Plain paragraph with enough words to make a chunk.", "source": "b.txt", "page": 1},
    ])
    assert [c["image"] for c in chunks] == [True, False]


def test_vision_failure_becomes_one_warning():
    calls = []

    class BrokenLLM:
        def describe_image(self, image):
            calls.append(1)
            raise RuntimeError("model gone")

    pipeline = RAGPipeline(embedder=None, reranker=None, llm=BrokenLLM())
    warnings = []
    describe = pipeline._safe_describer(warnings)
    assert describe(picture()) == "" and describe(picture()) == ""
    assert len(calls) == 1 and len(warnings) == 1 and "skipped" in warnings[0]


def test_vision_helpers():
    assert clean_model_text("<think>hmm</think>\nA pie chart.") == "A pie chart."
    url = image_to_data_url(picture(4000, 1000))
    assert url.startswith("data:image/jpeg;base64,")
    import base64
    shrunk = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(shrunk.size) == config.VISION_MAX_SIDE

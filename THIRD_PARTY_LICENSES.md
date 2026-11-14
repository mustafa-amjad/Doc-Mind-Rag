# Third-Party Licenses

DocMind uses the open-source components below. Licenses listed are what each project states at the time of writing.
**Verify each one on its own page before commercial use.** This is not legal advice.

| Component | Purpose | License |
|---|---|---|
| Streamlit | UI | Apache 2.0 |
| FAISS (faiss-cpu) | Vector search | MIT |
| sentence-transformers | Embeddings + re-ranker runtime | Apache 2.0 |
| all-MiniLM-L6-v2 | Embedding model | Apache 2.0 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Re-ranker model | Apache 2.0 on its model card. **Note:** trained on MS MARCO data, which has research-oriented terms. Review before paid/commercial use, or swap in another re-ranker. |
| rank-bm25 | Keyword search | Apache 2.0 |
| pypdf | PDF reading | BSD-3-Clause |
| pdfplumber | Finding tables in PDFs | MIT |
| python-docx | Word file reading | MIT |
| beautifulsoup4 | HTML text extraction | MIT |
| requests | Downloading web links | Apache 2.0 |
| langchain-text-splitters | Chunking | MIT |
| Groq Python SDK | LLM API client | Apache 2.0 |
| python-dotenv | Config | BSD-3-Clause |
| pypdfium2 (optional OCR) | Renders PDF pages to images | Apache 2.0 / BSD-3 (PDFium). Verify on its page. |
| rapidocr + onnxruntime (optional OCR) | Reads text from images | Apache 2.0 / MIT (rapidocr downloads PaddleOCR model files, Apache 2.0) |
| Pillow | Reading image files | MIT-CMU (HPND) |
| Vision model via Groq (default `qwen/qwen3.8-27b`, optional) | Describing charts and diagrams | Set by the model publisher. Verify on the model card before commercial use. |
| gpt-oss-120b by OpenAI (via Groq, default `LLM_MODEL`) | Answer generation | Apache 2.0 |

Using Groq's API is also subject to Groq's own Terms of Service and data policy.

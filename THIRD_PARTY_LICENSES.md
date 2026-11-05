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
| python-docx | Word file reading | MIT |
| beautifulsoup4 | HTML text extraction | MIT |
| requests | Downloading web links | Apache 2.0 |
| langchain-text-splitters | Chunking | MIT |
| Groq Python SDK | LLM API client | Apache 2.0 |
| python-dotenv | Config | BSD-3-Clause |
| Llama 3.3 70B (via Groq) | Answer generation | Llama 3.3 Community License + Meta Acceptable Use Policy. Show "Built with Llama" when you make a product available. |

Using Groq's API is also subject to Groq's own Terms of Service and data policy.

# LinkedIn post (copy everything between the lines)

---

Most RAG demos break the moment you ask a real question.

I built a RAG app that doesn't. 👇

DocMind RAG: upload PDFs, ask questions, get answers with page citations.

Basic RAG has 3 problems. Here is how I fixed each one:

❌ Misses exact words (IDs, names, codes)
✅ Hybrid search: BM25 + vector search, merged with Reciprocal Rank Fusion

❌ Right chunk is found but ranked 7th
✅ Cross-encoder re-ranking picks the best 4

❌ LLM makes things up
✅ "Use only the context" prompt + citations like [1] [2]

Also added:
→ FAISS HNSW index (auto-switches to exact search for small data)
→ UI toggles to turn hybrid and re-ranking ON/OFF, so you can see the difference yourself
→ Unit tests + clean project structure

Stack: Python, FAISS, sentence-transformers, BM25, Groq (Llama 3.3 70B), Streamlit

The biggest lesson: the LLM is not the hard part. Retrieval is.
A bad librarian makes a smart kid look dumb.

Code + README in the comments 👇

This is project #1 in my Gen AI series. Next: Agentic AI.

Mustafa Amjad

#RAG #GenAI #LLM #Python #MachineLearning #AIEngineering #Streamlit #OpenSource

---

## First comment (paste right after posting)

GitHub: https://github.com/YOUR_USERNAME/docmind-rag
Star it if it helps ⭐ Tell me what to add next.

## Posting tips
- Add a 20-30 second screen recording (toggle hybrid OFF vs ON on the same question). Video beats text.
- Post Tuesday to Thursday, US morning time.
- Reply to every comment in the first hour.

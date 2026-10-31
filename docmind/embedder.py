"""Step 3: turn text into vectors (lists of numbers) that capture meaning."""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)  # downloads the model the first time

    @property
    def dim(self) -> int:
        """How many numbers each vector has (384 for MiniLM)."""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return a (len(texts), dim) float32 array, each row has length 1."""
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,  # process 32 texts at once (faster)
            normalize_embeddings=True,  # length 1 -> inner product == cosine similarity
            show_progress_bar=False,  # keep the Streamlit logs clean
        )
        return np.asarray(vectors, dtype="float32")  # FAISS only accepts float32

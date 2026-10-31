"""Step 4a: vector search with FAISS (search by MEANING)."""
import faiss
import numpy as np

from docmind import config


class VectorStore:
    def __init__(self, dim: int, mode: str = "hnsw"):
        self.mode = mode  # remember which index we built (shown in the UI)
        if mode == "hnsw":
            # HNSW = friend-of-a-friend graph. Fast + accurate on big data.
            self.index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION  # build quality
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH  # search quality
        else:
            # Flat = check every vector. 100% exact, perfect for small data.
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(vectors)  # ids are assigned 0,1,2... in the order we add

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_id, score), ...] best first."""
        k = min(k, self.index.ntotal)  # cannot ask for more results than we stored
        if k == 0:  # empty index
            return []
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)  # FAISS wants a 2D array
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]  # -1 means "no result"

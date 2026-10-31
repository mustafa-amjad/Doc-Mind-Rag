"""Step 4b: keyword search with BM25 (search by EXACT WORDS)."""
import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words. 'Error-404!' -> ['error', '404']."""
    return re.findall(r"\w+", text.lower())


class KeywordIndex:
    def __init__(self, texts: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])  # build word statistics for every chunk

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_id, score), ...] best first. Chunks with score 0 are dropped."""
        tokens = tokenize(query)
        if not tokens:  # query had no words (only symbols)
            return []
        scores = self.bm25.get_scores(tokens)  # one score per chunk
        top = np.argsort(scores)[::-1][:k]  # sort high -> low, keep first k
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]  # score 0 = no word matched

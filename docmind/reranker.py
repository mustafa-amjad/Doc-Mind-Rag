"""Step 6: a slow-but-smart cross-encoder re-orders the candidates."""
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)  # reads (question, chunk) TOGETHER, so it is more accurate

    def rerank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        """Score every (query, chunk) pair and return the best top_n chunks."""
        if not chunks:
            return []
        pairs = [[query, chunk["text"]] for chunk in chunks]  # the model needs pairs
        scores = self.model.predict(pairs)  # one relevance score per pair
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)  # best first
        results = []
        for score, chunk in ranked[:top_n]:  # keep only the top_n
            item = dict(chunk)  # copy so we do not change the original chunk
            item["rerank_score"] = float(score)  # save the score for the UI
            results.append(item)
        return results

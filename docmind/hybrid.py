"""Step 5: merge keyword + vector results with Reciprocal Rank Fusion (RRF)."""


def rrf_fusion(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Each list is chunk ids, best first. A chunk gets 1/(k + position) from every list it appears in.
    Chunks that are high in BOTH lists get the biggest total. Scores are NOT compared directly
    (BM25 and cosine live on different scales), only the ranks are used. That is the whole trick.
    """
    scores: dict[int, float] = {}  # chunk_id -> total RRF score
    for ranked_ids in rank_lists:  # one pass per search method
        for position, chunk_id in enumerate(ranked_ids, start=1):  # position 1 = best
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)  # add this list's vote
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)  # best total first

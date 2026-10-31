from docmind.hybrid import rrf_fusion


def test_chunk_in_both_lists_wins():
    fused = rrf_fusion([[1, 2, 3], [3, 1, 4]])  # 1 and 3 appear in both lists
    ids = [chunk_id for chunk_id, _ in fused]
    assert set(ids[:2]) == {1, 3}  # both-list chunks beat single-list chunks


def test_empty_input():
    assert rrf_fusion([[], []]) == []


def test_score_formula():
    fused = dict(rrf_fusion([[7]], k=60))
    assert abs(fused[7] - 1 / 61) < 1e-9  # position 1 -> 1/(60+1)

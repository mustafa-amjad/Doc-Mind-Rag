from docmind.keyword_search import KeywordIndex, tokenize


def test_tokenize():
    assert tokenize("Error-404!") == ["error", "404"]


def test_exact_word_is_found():
    index = KeywordIndex(["refund policy for orders", "shipping takes days", "error 404 page missing"])
    hits = index.search("error 404", k=3)
    assert hits[0][0] == 2  # the chunk with the exact words ranks first


def test_empty_query():
    assert KeywordIndex(["hello world"]).search("!!!", k=3) == []

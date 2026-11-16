import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_answers", Path(__file__).parent.parent / "scripts" / "eval_answers.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

SOURCES = [{"source": "handbook.pdf", "page": 3, "text": ""}, {"source": "notice.pdf", "page": 1, "text": ""}]


def test_pass_needs_text_and_the_right_citation():
    case = {"expect": ["30 days"], "cite": "handbook.pdf", "page": 3}
    assert ev.check(case, "You have 30 days to return it [1].", SOURCES) == []
    assert ev.check(case, "You have 30 days to return it [2].", SOURCES) == ["does not cite handbook.pdf p.3"]
    assert ev.check(case, "You have a month [1].", SOURCES) == ["missing 30 days"]


def test_alternatives_and_whole_words():
    case = {"expect": [["uk", "united kingdom"]]}
    assert ev.check(case, "We ship to the UK.", SOURCES) == []
    assert ev.check(case, "We ship in bulk.", SOURCES) != []


def test_not_found_and_must_not():
    case = {"not_found": True, "must_not": ["Paris"]}
    assert ev.check(case, "I couldn’t find that in the uploaded documents.", SOURCES) == []
    assert ev.check(case, "Paris.", SOURCES) == ["expected the not-found sentence", "must not say Paris"]

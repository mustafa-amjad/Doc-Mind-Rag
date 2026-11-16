"""Answer test: run DocMind end to end on the sample documents and score every answer.

Uses the same pipeline as the app (OCR, chart descriptions, hybrid search, re-rank, chat memory) and
the Groq key from .env. Re-run it after any change to chunking, retrieval or the prompt.

Usage:
    python scripts/eval_answers.py                      # eval/docs + eval/answers.json
    python scripts/eval_answers.py --no-describe        # skip the AI chart descriptions (fewer API calls)
Exit code 1 if any case fails. Results are saved to eval/last_run.json.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "import docmind" works from here

from docmind import config, ocr  # noqa: E402
from docmind.embedder import Embedder  # noqa: E402
from docmind.llm import LLM, NOT_FOUND  # noqa: E402
from docmind.pipeline import RAGPipeline  # noqa: E402
from docmind.reranker import Reranker  # noqa: E402


def normalize(text: str) -> str:
    """Lowercase, plain spaces, plain apostrophes and hyphens, no markdown bold."""
    text = re.sub("[   ]", " ", text).replace("‑", "-").replace("’", "'")
    return " ".join(text.replace("**", "").lower().split())


def contains(answer: str, term: str) -> bool:
    """term appears as a whole word or phrase ('uk' does not match 'bulk')."""
    return re.search(r"(?<!\w)" + re.escape(normalize(term)) + r"(?!\w)", answer) is not None


def cited(answer: str, sources: list[dict]) -> list[dict]:
    """Sources the answer actually points at with [n]."""
    numbers = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return [sources[n - 1] for n in sorted(numbers) if 1 <= n <= len(sources)]


def check(case: dict, answer: str, sources: list[dict]) -> list[str]:
    """Reasons the case failed (empty list = pass)."""
    text = normalize(answer)
    if case.get("not_found"):
        problems = [] if text == normalize(NOT_FOUND) else ["expected the not-found sentence"]
    else:
        problems = []
        for item in case.get("expect", []):
            options = item if isinstance(item, list) else [item]
            if not any(contains(text, option) for option in options):
                problems.append("missing " + " / ".join(options))
        if case.get("cite"):
            hits = [s for s in cited(answer, sources) if s["source"] == case["cite"]
                    and (case.get("page") is None or int(s["page"]) == case["page"])]
            if not hits:
                where = case["cite"] + (f" p.{case['page']}" if case.get("page") else "")
                problems.append(f"does not cite {where}")
    problems += [f"must not say {term}" for term in case.get("must_not", []) if contains(text, term)]
    return problems


def ask_with_retry(pipe: RAGPipeline, question: str, history: list[dict]) -> dict:
    for attempt in range(6):  # Groq free tier: wait and retry on rate limits
        try:
            return pipe.ask(question, history=history)
        except Exception as error:
            if "rate_limit" not in str(error) and "429" not in str(error) or attempt == 5:
                raise
            time.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--docs", default=str(ROOT / "eval" / "docs"))
    parser.add_argument("--cases", default=str(ROOT / "eval" / "answers.json"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "last_run.json"))
    parser.add_argument("--no-describe", action="store_true", help="skip AI chart descriptions")
    parser.add_argument("--pause", type=float, default=6.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()
    if not config.GROQ_API_KEY:
        sys.exit("Set GROQ_API_KEY in .env first.")

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    files = sorted(p for p in Path(args.docs).iterdir() if p.is_file())
    pipe = RAGPipeline(Embedder(config.EMBED_MODEL), Reranker(config.RERANK_MODEL),
                       LLM(config.GROQ_API_KEY, config.LLM_MODEL))
    stats = pipe.ingest([(f.name, f.read_bytes()) for f in files], ocr=ocr.is_available(),
                        describe_images=not args.no_describe)
    print(f"Indexed {stats['files']} files, {stats['pages']} pages, {stats['chunks']} chunks. Model: {config.LLM_MODEL}")

    answers, results = {}, []
    for case in cases:
        history = []
        if case.get("follow_up_of"):
            earlier = answers[case["follow_up_of"]]
            history = [{"role": "user", "content": earlier["question"]},
                       {"role": "assistant", "content": earlier["answer"]}]
        result = ask_with_retry(pipe, case["question"], history)
        problems = check(case, result["answer"], result["sources"])
        answers[case["id"]] = {"question": case["question"], "answer": result["answer"]}
        results.append({"id": case["id"], "question": case["question"], "pass": not problems, "problems": problems,
                        "answer": result["answer"], "searched_for": result["search_query"],
                        "cited": [f"{s['source']} p.{s['page']}" for s in cited(result["answer"], result["sources"])]})
        print(f"{'PASS' if not problems else 'FAIL'}  {case['id']:>2}. {case['question']}")
        for problem in problems:
            print(f"        {problem}\n        answer: {result['answer'][:300]}")
        time.sleep(args.pause)

    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    Path(args.out).write_text(json.dumps({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "model": config.LLM_MODEL,
        "settings": {"chunk_size": config.CHUNK_SIZE, "retrieve_k": config.RETRIEVE_K, "final_k": config.FINAL_K,
                     "hybrid": True, "rerank": True, "chart_descriptions": not args.no_describe},
        "passed": passed, "total": len(results), "results": results}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

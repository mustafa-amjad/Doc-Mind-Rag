"""Measure retrieval quality for 4 setups. Needs NO Groq key (only retrieval is tested).

Usage:
    python scripts/evaluate.py --pdf data/uploads/sample.pdf --questions eval/questions.json --k 4
"""
import argparse  # reads command-line options
import csv  # saves results
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so "import docmind" works from here

from docmind import config  # noqa: E402
from docmind.embedder import Embedder  # noqa: E402
from docmind.pipeline import RAGPipeline  # noqa: E402
from docmind.reranker import Reranker  # noqa: E402

CONFIGS = {  # name -> (use_hybrid, use_rerank)
    "Vector only": (False, False),
    "Vector + Re-rank": (False, True),
    "Hybrid": (True, False),
    "Hybrid + Re-rank": (True, True),
}


def is_hit(chunk: dict, case: dict) -> bool:
    """A chunk is correct if it has the 'must_contain' text; if the case has none, use 'expected_page'."""
    if case.get("must_contain"):
        return case["must_contain"].lower() in chunk["text"].lower()
    return chunk["page"] == case.get("expected_page")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="PDF to index")
    parser.add_argument("--questions", required=True, help="JSON file with test questions")
    parser.add_argument("--k", type=int, default=4, help="how many chunks count as 'retrieved'")
    parser.add_argument("--out", default="eval/results.csv")
    args = parser.parse_args()

    cases = json.loads(Path(args.questions).read_text(encoding="utf-8"))  # [{question, must_contain|expected_page}]
    pipeline = RAGPipeline(Embedder(config.EMBED_MODEL), Reranker(config.RERANK_MODEL), llm=None)
    pdf_path = Path(args.pdf)
    print("Indexing", pdf_path.name, pipeline.ingest([(pdf_path.name, pdf_path.read_bytes())]))

    rows = []
    for name, (use_hybrid, use_rerank) in CONFIGS.items():
        hits, reciprocal_ranks = 0, []
        for case in cases:
            results = pipeline.retrieve(case["question"], use_hybrid, use_rerank, args.k)  # best first
            rank = next((i for i, chunk in enumerate(results, start=1) if is_hit(chunk, case)), None)
            if rank:  # found inside top k
                hits += 1
                reciprocal_ranks.append(1 / rank)  # rank 1 -> 1.0, rank 2 -> 0.5 ...
            else:
                reciprocal_ranks.append(0.0)
        rows.append({"setup": name, f"hit@{args.k}": round(hits / len(cases), 3),
                     "MRR": round(sum(reciprocal_ranks) / len(cases), 3)})

    print(f"\n{'Setup':<20}{'Hit@' + str(args.k):<10}{'MRR':<8}")  # simple table
    for row in rows:
        print(f"{row['setup']:<20}{row[f'hit@{args.k}']:<10}{row['MRR']:<8}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved:", args.out)


if __name__ == "__main__":
    main()

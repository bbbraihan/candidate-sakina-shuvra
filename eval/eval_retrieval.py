"""
Retrieval evaluation — recall@3 and recall@5 against corpus/labels.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Retriever


LABELS_FILE = Path(__file__).resolve().parent.parent / "corpus" / "labels.json"


def compute_recall(
    retrieved_ids: list[str], expected_ids: list[str], k: int
) -> float:
    if not expected_ids:
        return 1.0
    top_k_ids = set(retrieved_ids[:k])
    hits = sum(1 for eid in expected_ids if eid in top_k_ids)
    return hits / len(expected_ids)


def run_retrieval_eval(retriever: Retriever) -> dict:
    with open(LABELS_FILE) as f:
        data = json.load(f)

    queries = data["queries"]
    recall3_scores, recall5_scores = [], []
    rows = []

    for item in queries:
        qid = item["query_id"]
        query = item["query"]
        expected = item["expected_ids"]

        docs = retriever.search(query, top_k=5)
        retrieved_ids = [d["id"] for d in docs]

        r3 = compute_recall(retrieved_ids, expected, 3)
        r5 = compute_recall(retrieved_ids, expected, 5)
        recall3_scores.append(r3)
        recall5_scores.append(r5)

        rows.append(
            {
                "query_id": qid,
                "query": query[:60],
                "expected": expected,
                "retrieved_top5": retrieved_ids,
                "recall@3": round(r3, 3),
                "recall@5": round(r5, 3),
            }
        )
        print(
            f"  {qid}: recall@3={r3:.2f}  recall@5={r5:.2f}  "
            f"retrieved={retrieved_ids}"
        )

    mean_r3 = sum(recall3_scores) / len(recall3_scores)
    mean_r5 = sum(recall5_scores) / len(recall5_scores)
    print(f"\n  mean recall@3 = {mean_r3:.3f}")
    print(f"  mean recall@5 = {mean_r5:.3f}")

    return {
        "rows": rows,
        "mean_recall@3": round(mean_r3, 3),
        "mean_recall@5": round(mean_r5, 3),
    }


if __name__ == "__main__":
    r = Retriever()
    run_retrieval_eval(r)

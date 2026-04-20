"""
Faithfulness eval — LLM-as-judge.

For 5 queries: retrieve top-5, generate a response, then ask the LLM:
  "Does this response cite ONLY the following sources?"
Score: fraction faithful out of 5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama

from src.config import OLLAMA_MODEL
from src.generator import generate
from src.retriever import Retriever

# 5 representative queries
FAITHFULNESS_QUERIES = [
    "I feel overwhelmed with anxiety and can't sleep at night",
    "My mother passed away recently and I can't stop crying",
    "I used to feel close to Allah but now my prayers feel empty",
    "Life keeps throwing one hardship after another at me",
    "I've made so many sins I feel like Allah won't forgive me",
]

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a list of allowed source references and "
    "an assistant response, decide whether the response cites ONLY those references. "
    "A response is faithful if it does not mention any verse, hadith, or source "
    "that is NOT in the allowed list. "
    "Respond with valid JSON only: {\"faithful\": true|false, \"reason\": \"...\"}"
)


def judge_faithfulness(
    response: str, allowed_refs: list[str]
) -> dict:
    allowed_str = "\n".join(f"- {r}" for r in allowed_refs)
    user_msg = (
        f"ALLOWED REFERENCES:\n{allowed_str}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "Is the response faithful? Reply JSON only."
    )
    try:
        result = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = result["message"]["content"].strip()
        # Extract JSON even if wrapped in markdown code fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as exc:
        return {"faithful": None, "reason": f"Judge error: {exc}"}


def run_faithfulness_eval(retriever: Retriever) -> dict:
    results = []
    faithful_count = 0
    total = 0

    for query in FAITHFULNESS_QUERIES:
        docs = retriever.search(query, top_k=5)
        allowed_refs = [d["reference"] for d in docs]
        response, latency_ms, _ = generate(query, docs)
        verdict = judge_faithfulness(response, allowed_refs)

        is_faithful = verdict.get("faithful")
        if is_faithful is True:
            faithful_count += 1
        if is_faithful is not None:
            total += 1

        results.append(
            {
                "query": query[:60],
                "allowed_refs": allowed_refs,
                "faithful": is_faithful,
                "reason": verdict.get("reason", ""),
                "latency_ms": round(latency_ms, 1),
            }
        )
        print(
            f"  {'✓' if is_faithful else '✗' if is_faithful is False else '?'} "
            f"{query[:55]}…  faithful={is_faithful}"
        )

    score = faithful_count / total if total else 0.0
    print(f"\n  Faithfulness score: {faithful_count}/{total} = {score:.2f}")
    return {"results": results, "faithfulness_score": round(score, 2), "faithful": faithful_count, "total": total}


if __name__ == "__main__":
    r = Retriever()
    run_faithfulness_eval(r)

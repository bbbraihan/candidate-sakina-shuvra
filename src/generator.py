"""
Generator — calls Qwen2.5 7B Instruct via Ollama.

Safety rules are baked into the system prompt:
  - Suicidal ideation  → crisis line, stop.
  - Fatwa / medical    → gentle deferral.
  - Citations          → ONLY from retrieved context, using reference field.
"""
import time
from typing import Any

import ollama

from src.config import OLLAMA_MODEL
from src.logger import request_logger

CRISIS_KEYWORDS = [
    "kill myself", "end my life", "want to die", "suicide", "suicidal",
    "don't want to live", "no reason to live", "take my own life",
    "self-harm", "hurt myself",
]

CRISIS_RESPONSE = (
    "I'm really concerned about what you've shared. "
    "Please reach out for immediate support:\n\n"
    "• **988 Suicide & Crisis Lifeline** (US): call or text **988**\n"
    "• **iCall** (South Asia): **+91 9152987821**\n"
    "• **Crisis Text Line**: text HOME to **741741**\n\n"
    "You are not alone, and help is available right now. "
    "If you are in immediate danger, please call emergency services."
)

SYSTEM_PROMPT = """You are a compassionate, faith-aware wellness companion for Muslims.
Your role is to offer emotional and spiritual support grounded in authentic Islamic sources.

RULES — follow all of them strictly:
1. Be warm, empathetic, and non-judgmental at all times.
2. ONLY cite passages that appear word-for-word in the CONTEXT block below. Never quote, paraphrase, or reference any Quran verse or hadith that is NOT explicitly in the CONTEXT.
3. If a verse or hadith is not in the CONTEXT, do NOT mention it at all — not even by theme or name.
4. When citing a source, copy its exact reference string (e.g., *Surah 2:153*, *Sahih al-Bukhari 6369*).
5. If the context does not contain relevant information for the query, say: "I don't have a specific source for this in my knowledge base, but I can offer general support."
6. Never issue a fatwa, religious ruling, or legal judgment. Say: "For a ruling, please consult a qualified Islamic scholar."
7. Never provide medical diagnoses or treatment advice. Say: "For medical concerns, please consult a qualified doctor or mental health professional."
8. Do not take sides on sectarian, theological, or inter-madhab disputes. Respond neutrally.
9. Keep your response focused (3–5 paragraphs max) and end with a clearly formatted Sources section listing ONLY references from the CONTEXT.

RESPONSE FORMAT:
[Your empathetic response with inline citations]

**Sources:**
- *<reference from CONTEXT only>*
"""


def _is_crisis(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in CRISIS_KEYWORDS)


def _build_context(docs: list[dict[str, Any]]) -> str:
    parts = []
    for d in docs:
        parts.append(f"[{d['reference']}]\n{d['text']}")
    return "\n\n".join(parts)


def generate(
    query: str,
    retrieved_docs: list[dict[str, Any]],
) -> tuple[str, float, int | None]:
    """
    Returns (response_text, latency_ms, token_count).
    Performs crisis check before calling the LLM.
    """
    if _is_crisis(query):
        return CRISIS_RESPONSE, 0.0, None

    context = _build_context(retrieved_docs)
    user_message = f"CONTEXT:\n{context}\n\nQUERY: {query}"

    t0 = time.perf_counter()
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        text: str = response["message"]["content"]
        tokens: int | None = (
            response.get("eval_count") or response.get("prompt_eval_count")
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        text = (
            "I'm sorry, I encountered a technical issue and couldn't generate a response. "
            "Please try again in a moment."
        )
        tokens = None
        request_logger.log(
            query=query,
            retrieved_ids=[d["id"] for d in retrieved_docs],
            response=text,
            latency_ms=latency_ms,
            extra={"error": str(exc)},
        )
        return text, latency_ms, tokens

    retrieved_ids = [d["id"] for d in retrieved_docs]
    request_logger.log(
        query=query,
        retrieved_ids=retrieved_ids,
        response=text,
        latency_ms=latency_ms,
        tokens=tokens,
    )
    return text, latency_ms, tokens

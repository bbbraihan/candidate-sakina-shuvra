"""
Structured JSON logger for every request.
Fields: timestamp, query, retrieved_ids, response_snippet, latency_ms, tokens
"""
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import LOG_FILE


def _ensure_log_dir() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


class RequestLogger:
    def __init__(self) -> None:
        _ensure_log_dir()
        # Also mirror to stderr at INFO level for visibility
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            level=logging.INFO,
        )
        self._log = logging.getLogger("wellness_companion")

    def log(
        self,
        query: str,
        retrieved_ids: list[str],
        response: str,
        latency_ms: float,
        tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": query,
            "retrieved_ids": retrieved_ids,
            "response_snippet": response[:200],
            "latency_ms": round(latency_ms, 1),
            "tokens": tokens,
        }
        if extra:
            record.update(extra)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._log.info(
            "query=%r ids=%s latency=%.0fms tokens=%s",
            query[:60],
            retrieved_ids,
            latency_ms,
            tokens,
        )


# Singleton
request_logger = RequestLogger()

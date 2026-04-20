"""
Central configuration — all values overridable via environment variables.
A .env file at the repo root is loaded automatically if present.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

CORPUS_DIR: Path = Path(os.getenv("CORPUS_DIR", str(BASE_DIR / "dataset")))
INDEX_DIR: Path = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "faiss_index")))
LOG_FILE: Path = Path(os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "requests.jsonl")))

# ── Models ─────────────────────────────────────────────────────────────────
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Voice ──────────────────────────────────────────────────────────────────
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")

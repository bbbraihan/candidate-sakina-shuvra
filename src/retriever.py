"""
Retriever — builds a FAISS index over the corpus and exposes search().

Index is cached to INDEX_DIR on first build; subsequent startups load from disk.
"""
import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import CORPUS_DIR, EMBED_MODEL, INDEX_DIR, TOP_K

CORPUS_FILES = ["quran.jsonl", "hadith.jsonl", "articles.jsonl"]
INDEX_FILE = INDEX_DIR / "corpus.index"
META_FILE = INDEX_DIR / "corpus_meta.pkl"


def _normalise(doc: dict[str, Any]) -> dict[str, Any]:
    """Ensure every doc has a `reference` field for citations."""
    if "reference" not in doc:
        # Articles: use title as reference
        doc["reference"] = doc.get("title", doc["id"])
    return doc


def _load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for fname in CORPUS_FILES:
        fpath = corpus_dir / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(_normalise(json.loads(line)))
    return docs


def _build_index(
    docs: list[dict[str, Any]], model: SentenceTransformer
) -> faiss.IndexFlatIP:
    # Enrich embedding text with theme tags for better semantic matching
    def _embed_text(d: dict) -> str:
        theme = d.get("theme", "")
        base = d["text"]
        return f"{theme}. {base}" if theme else base

    texts = [_embed_text(d) for d in docs]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on L2-normalised vecs
    index.add(embeddings.astype(np.float32))
    return index


class Retriever:
    def __init__(self, force_rebuild: bool = False) -> None:
        self._model = SentenceTransformer(EMBED_MODEL)
        if not force_rebuild and INDEX_FILE.exists() and META_FILE.exists():
            self._index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "rb") as f:
                self._docs = pickle.load(f)
            print(f"[Retriever] Loaded cached index ({len(self._docs)} docs)")
        else:
            print("[Retriever] Building index from corpus …")
            self._docs = _load_corpus(CORPUS_DIR)
            if not self._docs:
                raise RuntimeError(f"No documents found in {CORPUS_DIR}")
            self._index = _build_index(self._docs, self._model)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(INDEX_FILE))
            with open(META_FILE, "wb") as f:
                pickle.dump(self._docs, f)
            print(f"[Retriever] Built & cached index ({len(self._docs)} docs)")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Return top_k docs sorted by cosine similarity."""
        # Prefix boosts semantic alignment with bge-small retrieval
        prefixed = f"Represent this wellness question for searching relevant Islamic guidance: {query}"
        q_vec = self._model.encode(
            [prefixed], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self._index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self._docs[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results

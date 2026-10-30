# Module 11 - Vector Databases
# Shared embedding helpers - lets every 0X_*.py file in this folder embed text
# against Gemini, with no other code changes needed. When no key is set at all,
# get_embed_fn() falls back to a deterministic offline mock_embed() so the
# ChromaDB demos and exercises still run.

import hashlib
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Gemini embeddings (from ai.google.dev)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIM = 3072   # gemini-embedding-001 output dimension

EmbedFn = Callable[[list[str]], list[list[float]]]


def has_gemini_access() -> bool:
    """True when GEMINI_API_KEY is set."""
    value = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(value and "..." not in value and len(value) >= 12)


def gemini_embed(texts: list[str], model: str = GEMINI_EMBEDDING_MODEL) -> list[list[float]]:
    """Embed texts using Gemini's embedding API. Needs GEMINI_API_KEY."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    embeddings = []
    for text in texts:
        result = genai.embed_content(model=model, content=text)
        embeddings.append(result["embedding"])
    return embeddings


def mock_embed(texts: list[str], dim: int = 64) -> list[list[float]]:
    """Deterministic offline embedding: signed hashed bag-of-words, L2-normalised.
    The same text always maps to the same vector (stable across processes, so it
    is safe to persist), and texts sharing words land close together - enough to
    exercise vector DB logic without any network call. Not a real semantic model."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float64)
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
            vec[h % dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0], norm = 1.0, 1.0   # avoid zero vectors (undefined cosine distance)
        vectors.append((vec / norm).tolist())
    return vectors


def get_embed_fn() -> tuple[EmbedFn, str]:
    """Return (embed_fn, label): the real embedder when a key is available,
    otherwise the offline mock. The label is "gemini" or "mock"."""
    if has_gemini_access():
        return gemini_embed, "gemini"
    return mock_embed, "mock"

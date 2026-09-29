"""Shared embedding backend for Module 08.

OpenAI remains the first choice to match the original exercises. When an
OpenAI key is unavailable, the configured Gemini key from Module 06 is reused
so every online Module 08 demo can still produce real semantic embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


MODULE_DIR = Path(__file__).resolve().parent
PHASE2_DIR = MODULE_DIR.parent

load_dotenv(MODULE_DIR / ".env")
load_dotenv(PHASE2_DIR / "Module06_LLM_APIs" / ".env", override=False)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
)
GEMINI_EMBEDDING_DIM = int(os.getenv("GEMINI_EMBEDDING_DIM", "1536"))


def _has_usable_key(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    placeholders = ("...", "your_", "replace", "masukkan", "isi_")
    return bool(value and not any(marker in value for marker in placeholders))


def selected_embedding_provider() -> str | None:
    requested = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if requested and requested not in {"openai", "gemini"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'gemini'.")

    if requested:
        key_name = "OPENAI_API_KEY" if requested == "openai" else "GEMINI_API_KEY"
        if not _has_usable_key(key_name):
            raise ValueError(f"{key_name} is required for EMBEDDING_PROVIDER={requested}.")
        return requested

    if _has_usable_key("OPENAI_API_KEY"):
        return "openai"
    if _has_usable_key("GEMINI_API_KEY"):
        return "gemini"
    return None


def has_embedding_provider() -> bool:
    return selected_embedding_provider() is not None


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """Return one float32 embedding per input text."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    provider = selected_embedding_provider()
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            input=texts,
            model=model or OPENAI_EMBEDDING_MODEL,
        )
        vectors = sorted(response.data, key=lambda item: item.index)
        return np.array([item.embedding for item in vectors], dtype=np.float32)

    if provider == "gemini":
        from google import genai
        from google.genai import types

        with genai.Client(api_key=os.environ["GEMINI_API_KEY"]) as client:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=GEMINI_EMBEDDING_DIM,
                ),
            )
        matrix = np.array(
            [embedding.values for embedding in response.embeddings],
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    raise RuntimeError("Set OPENAI_API_KEY or GEMINI_API_KEY for Module 08.")

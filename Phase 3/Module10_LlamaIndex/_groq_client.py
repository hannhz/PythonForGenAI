# Module 10 - LlamaIndex
# Shared model factory - lets every 0X_*.py file in this folder configure
# LlamaIndex's Settings.llm / Settings.embed_model against Groq + Gemini,
# with no other code changes needed in the calling file. If no key is set at
# all, an offline LLM + embedding model (deterministic, no network) can be used
# so the core logic of every file can still be exercised.
#
# NOTE: needs `pip install llama-index` (core + OpenAI-like LLM integration).
# Groq uses the OpenAILike class because the plain OpenAI class rejects
# non-OpenAI model names. Gemini embeddings are handled separately.
# This module imports llama_index lazily, so it can be imported without it.

import os
import re
import sys
import hashlib
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return. Force UTF-8 stdout so a print() of
# model output never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Groq (primary chat) ──────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_CONTEXT_WINDOW = 131_072

# ── Gemini (embeddings) ─────────────────────────────────────────────────────
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")


def _has_groq() -> bool:
    """True when a real Groq key is configured (not a placeholder)."""
    value = os.getenv("GROQ_API_KEY", "").strip()
    return bool(value and "..." not in value and len(value) >= 12)


def has_llm_access() -> bool:
    """True when GROQ_API_KEY is set."""
    return _has_groq()


def _missing_package(pip_name: str) -> None:
    raise SystemExit(f"Missing package. Install it with: pip install {pip_name}") from None


# ── Real models ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1, max_tokens: int = 1024):
    """LlamaIndex LLM for Groq, or None when no key is set."""
    if not _has_groq():
        return None

    try:
        from llama_index.llms.openai_like import OpenAILike
    except ImportError:
        _missing_package("llama-index-llms-openai-like")

    return OpenAILike(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        api_base=GROQ_BASE_URL,
        is_chat_model=True,
        context_window=GROQ_CONTEXT_WINDOW,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embed_model():
    """LlamaIndex embedding model for Gemini, or None when no key is set."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None

    try:
        from llama_index.embeddings.gemini import GeminiEmbedding
    except ImportError:
        _missing_package("llama-index-embeddings-gemini")

    return GeminiEmbedding(
        model_name=GEMINI_EMBEDDING_MODEL,
        api_key=gemini_key,
    )


# ── Offline models (no network) ──────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the this to was what "
    "when where which who why with does do did can i you we they not".split()
)


def _tokens(text: str) -> list[str]:
    """Lowercased word tokens with a crude plural strip and stopwords removed."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOPWORDS]


def _hash_vector(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words hashing embedding, L2-normalised. Texts that share
    words get a higher cosine score - good enough to test retrieval logic offline,
    but NOT semantically meaningful like a real embedding model."""
    vector = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


_CONTEXT_BLOCK_RE = re.compile(r"-{5,}[ \t]*\n(.*?)\n[ \t]*-{5,}", re.S)
_QUERY_RE = re.compile(r"(?:Query|Question|original query is as follows):\s*(.*?)\s*(?:\n\s*(?:Answer|We have)|$)", re.S)
_METADATA_LINE_RE = re.compile(r"^[a-z_]+: [^\n]*$", re.M)


def _extractive_answer(prompt: str) -> str:
    """What the offline LLM 'answers': the 1-2 context sentences that share the
    most words with the query. Parses LlamaIndex's default QA/refine/summarise
    prompts (context between dashed lines, 'Query: ...' afterwards)."""
    blocks = _CONTEXT_BLOCK_RE.findall(prompt)
    queries = _QUERY_RE.findall(prompt)
    context = blocks[0] if blocks else prompt
    query = queries[-1] if queries else ""

    context = _METADATA_LINE_RE.sub("", context)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", context) if s.strip()]
    if not sentences:
        return "[offline mock LLM] (no context)"

    query_tokens = set(_tokens(query))
    scored = [(len(query_tokens & set(_tokens(s))), i, s) for i, s in enumerate(sentences)]
    best = sorted((t for t in scored if t[0] > 0), key=lambda t: (-t[0], t[1]))[:2]
    chosen = [s for _, _, s in sorted(best, key=lambda t: t[1])] or [sentences[0]]
    return "[offline mock LLM] " + " ".join(" ".join(chosen).split())[:400]


_OFFLINE_CACHE: dict[str, Any] = {}


def get_offline_llm():
    """Offline LlamaIndex LLM: answers by extracting the most relevant context
    sentences. Makes no network call."""
    if "llm" in _OFFLINE_CACHE:
        return _OFFLINE_CACHE["llm"]
    try:
        from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
        from llama_index.core.llms.callbacks import llm_completion_callback
    except ImportError:
        _missing_package("llama-index")

    class OfflineExtractiveLLM(CustomLLM):
        context_window: int = 3900
        num_output: int = 256
        model_name: str = "offline-extractive-llm"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=self.context_window,
                num_output=self.num_output,
                model_name=self.model_name,
            )

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            return CompletionResponse(text=_extractive_answer(prompt))

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
            text = _extractive_answer(prompt)
            yield CompletionResponse(text=text, delta=text)

    _OFFLINE_CACHE["llm"] = OfflineExtractiveLLM()
    return _OFFLINE_CACHE["llm"]


def get_offline_embed_model(dim: int = 256):
    """Offline LlamaIndex embedding model (hashed bag-of-words). Makes no network call."""
    key = f"embed_{dim}"
    if key in _OFFLINE_CACHE:
        return _OFFLINE_CACHE[key]
    try:
        from llama_index.core.embeddings import BaseEmbedding
    except ImportError:
        _missing_package("llama-index")

    class HashEmbedding(BaseEmbedding):
        dim: int = 256

        def _get_query_embedding(self, query: str) -> list[float]:
            return _hash_vector(query, self.dim)

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return _hash_vector(query, self.dim)

        def _get_text_embedding(self, text: str) -> list[float]:
            return _hash_vector(text, self.dim)

    _OFFLINE_CACHE[key] = HashEmbedding(model_name="offline-hash-embedding", dim=dim)
    return _OFFLINE_CACHE[key]


# ── Settings helpers ─────────────────────────────────────────────────────────

def _apply_chunking(chunk_size: int | None, chunk_overlap: int | None) -> None:
    from llama_index.core import Settings

    if chunk_size is not None:
        Settings.chunk_size = chunk_size
    if chunk_overlap is not None:
        Settings.chunk_overlap = chunk_overlap


def configure_settings(
    temperature: float = 0.1,
    max_tokens: int = 1024,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> bool:
    """Point Settings.llm / Settings.embed_model at Groq + Gemini.
    Returns False (and changes nothing) when no API key is set."""
    if not has_llm_access():
        return False
    from llama_index.core import Settings

    Settings.llm = get_llm(temperature=temperature, max_tokens=max_tokens)
    Settings.embed_model = get_embed_model()
    _apply_chunking(chunk_size, chunk_overlap)
    return True


def configure_offline_settings(chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
    """Point Settings at the offline LLM + hashing embedding model."""
    from llama_index.core import Settings

    Settings.llm = get_offline_llm()
    Settings.embed_model = get_offline_embed_model()
    _apply_chunking(chunk_size, chunk_overlap)


def configure_settings_or_offline(
    temperature: float = 0.1,
    max_tokens: int = 1024,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    """Configure Settings with Groq + Gemini if keys exist, otherwise the
    offline models. Returns "real" or "offline" and prints which one is active."""
    if configure_settings(temperature, max_tokens, chunk_size, chunk_overlap):
        print(f"[real] LLM={GROQ_MODEL} | embeddings={GEMINI_EMBEDDING_MODEL}")
        return "real"
    configure_offline_settings(chunk_size, chunk_overlap)
    print(
        "[offline] No GROQ_API_KEY set - "
        "using the offline mock LLM + hashing embeddings (answers are extractive, not generated). "
        "Add a key to a .env file in this folder for real model output."
    )
    return "offline"


def get_models_or_offline() -> tuple[Any, Any, str]:
    """(llm, embed_model, "real" | "offline") without touching global Settings."""
    if has_llm_access():
        return get_llm(), get_embed_model(), "real"
    return get_offline_llm(), get_offline_embed_model(), "offline"

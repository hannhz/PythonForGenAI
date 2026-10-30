# Module 13 - LangGraph & Agents
# Shared chat-model factory - lets every 0X_*.py file in this folder build a
# LangChain chat model from Groq, with no other code changes.
# Groq speaks the OpenAI-compatible chat API, so ChatOpenAI is enough.
#
# NOTE: get_chat_model() needs the langchain-openai package. Every demo in this
# module also runs offline with a scripted fake LLM, so a key is only needed
# for the optional "--real" paths.

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return. Force UTF-8 stdout so a print() of a
# model reply never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Groq (primary chat provider)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

LANGGRAPH_INSTALL_HINT = "pip install langgraph langchain-core langchain-openai python-dotenv"
NO_KEY_HINT = "GROQ_API_KEY is not set. Add one to a .env file in this folder."


def _has_groq() -> bool:
    """True when a real Groq key is configured (not a placeholder)."""
    value = os.getenv("GROQ_API_KEY", "").strip()
    return bool(value and "..." not in value and len(value) >= 12)


def has_llm_access() -> bool:
    """True if GROQ_API_KEY is configured."""
    return _has_groq()


def get_chat_model(model: str | None = None, temperature: float = 0.0):
    """LangChain ChatOpenAI routed through Groq. Returns None when no key is
    configured.

    Raises ImportError (with an install hint) if langchain-openai is missing."""
    if not has_llm_access():
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(f"langchain-openai is not installed. Run: {LANGGRAPH_INSTALL_HINT}") from exc

    return ChatOpenAI(
        model=model or GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
        temperature=temperature,
    )

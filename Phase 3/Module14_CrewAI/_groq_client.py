# Module 14 - CrewAI
# Shared LLM factory - lets every 0X_*.py file in this folder run against Groq,
# with no other code changes needed in the calling file.
#
# get_llm() builds the `llm` object that is passed to Agent(llm=...):
# - Default: langchain_openai.ChatOpenAI pointed at Groq's OpenAI-compatible API
# - CREWAI_LLM_STYLE=native: crewai.LLM("openai/gpt-oss-20b", ...) instead
#
# NOTE: CrewAI makes MANY LLM calls per kickoff (several per task - reasoning
# steps, tool calls, delegation), so every crew.kickoff() in this module makes
# REAL API calls. Groq's free tier is 30 RPM / 14,400 RPD - generous enough for
# demos, but keep an eye on usage for the hierarchical process in
# 06_hierarchical_process.py and 07_exercises.py.

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# Windows consoles default to a codepage (e.g. cp1252) that can't print every
# Unicode character a model might return (emoji, arrows, checkmarks). Force
# UTF-8 stdout so a print(result) never crashes with UnicodeEncodeError.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Groq (primary chat provider)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

NO_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Add one to a .env file in this folder (see .env.example)."
)


def _has_groq() -> bool:
    """True when a real Groq key is configured (not a placeholder)."""
    value = os.getenv("GROQ_API_KEY", "").strip()
    return bool(value and "..." not in value and len(value) >= 12)


def has_llm_access() -> bool:
    return _has_groq()


def describe_llm() -> str:
    """Human-readable 'provider / model' label for printing in demos."""
    if not _has_groq():
        return "no LLM configured"
    return f"groq / {GROQ_MODEL}"


def require_crewai() -> None:
    """Exit cleanly (no traceback) when the crewai package is not installed."""
    if importlib.util.find_spec("crewai") is None:
        raise SystemExit(
            "The 'crewai' package is not installed.\n"
            "Install it with:  pip install crewai langchain-openai python-dotenv"
        )


def load_sibling(module_name: str) -> ModuleType:
    """Import another numbered file of this module (e.g. '03_tools_for_agents').
    File names starting with a digit can't be used in a normal `import`
    statement, so importlib is used instead."""
    return importlib.import_module(module_name)


def _native_llm() -> Any:
    """crewai.LLM routed through Groq's OpenAI-compatible API."""
    from crewai import LLM

    return LLM(
        model=f"openai/{GROQ_MODEL}",
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )


def _langchain_llm() -> Any:
    """The book's `ChatOpenAI(model=..., api_key=...)`, pointed at Groq."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
    )


def get_llm(offline_ok: bool = False) -> Any | None:
    """LLM object for Agent(llm=...).

    Returns None when no API key is configured. With offline_ok=True it returns
    a placeholder crewai.LLM instead (never used for a real call) so Agent / Task /
    Crew objects can still be constructed and inspected without a key.

    NOTE: CrewAI 1.15+ requires crewai.LLM (not langchain ChatOpenAI), so we
    always use the native LLM format."""
    if not _has_groq():
        return _native_llm_placeholder() if offline_ok else None

    return _native_llm()


def _native_llm_placeholder() -> Any:
    from crewai import LLM

    return LLM(model="openai/gpt-oss-20b", api_key="sk-offline-placeholder")

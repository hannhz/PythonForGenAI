"""Shared text-provider fallback for Phase 2.

The course examples use the Anthropic Messages API.  This adapter exposes the
small subset of that API used by Modules 06-07, but sends requests through
Groq's OpenAI-compatible endpoint when GROQ_API_KEY is available.  Set
LLM_PROVIDER=anthropic to force the original provider.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
from dotenv import load_dotenv
from openai import OpenAI


PHASE2_DIR = Path(__file__).resolve().parent
load_dotenv()
# Reuse the existing Module 06 .env without copying secrets into every module.
load_dotenv(PHASE2_DIR / "Module06_LLM_APIs" / ".env", override=False)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def has_usable_key(name: str) -> bool:
    value = os.getenv(name, "").strip()
    return bool(value and "..." not in value and len(value) >= 12)


def selected_provider() -> str | None:
    requested = os.getenv("LLM_PROVIDER", "").strip().lower()
    if requested:
        if requested not in {"groq", "anthropic"}:
            raise ValueError("LLM_PROVIDER must be 'groq' or 'anthropic'.")
        key_name = "GROQ_API_KEY" if requested == "groq" else "ANTHROPIC_API_KEY"
        if not has_usable_key(key_name):
            raise ValueError(f"{key_name} is required when LLM_PROVIDER={requested}.")
        return requested

    # Prefer the free Groq path so a present-but-unfunded Anthropic key does
    # not cause a billing error.
    if has_usable_key("GROQ_API_KEY"):
        return "groq"
    if has_usable_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def has_text_provider() -> bool:
    return selected_provider() is not None


def get_openai_compatible_client() -> tuple[OpenAI, str, str]:
    """Return (client, model, provider) for Groq first, then OpenAI."""
    if has_usable_key("GROQ_API_KEY"):
        return (
            OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL),
            GROQ_MODEL,
            "groq",
        )
    if has_usable_key("OPENAI_API_KEY"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"]), "gpt-4o", "openai"
    raise RuntimeError("Set GROQ_API_KEY or a valid OPENAI_API_KEY.")


def _block_value(block: Any, name: str, default: Any = None) -> Any:
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


def _to_openai_messages(messages: list[dict], system: str = "") -> list[dict]:
    converted: list[dict] = []
    if system:
        converted.append({"role": "system", "content": system})

    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content:
            block_type = _block_value(block, "type")
            if block_type == "text":
                text_parts.append(str(_block_value(block, "text", "")))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": _block_value(block, "id"),
                    "type": "function",
                    "function": {
                        "name": _block_value(block, "name"),
                        "arguments": json.dumps(_block_value(block, "input", {})),
                    },
                })
            elif block_type == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": _block_value(block, "tool_use_id"),
                    "content": str(_block_value(block, "content", "")),
                })

        if role == "assistant":
            converted.append({
                "role": "assistant",
                "content": "\n".join(text_parts) or None,
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
        elif tool_results:
            converted.extend(tool_results)
        else:
            converted.append({"role": role, "content": "\n".join(text_parts)})
    return converted


def _to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    } for tool in tools]


def _normalise_response(response: Any) -> SimpleNamespace:
    choice = response.choices[0]
    message = choice.message
    blocks: list[SimpleNamespace] = []
    if message.content:
        blocks.append(SimpleNamespace(type="text", text=message.content))
    for tool_call in message.tool_calls or []:
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=tool_call.id,
            name=tool_call.function.name,
            input=json.loads(tool_call.function.arguments or "{}"),
        ))
    usage = response.usage
    return SimpleNamespace(
        content=blocks,
        stop_reason="tool_use" if message.tool_calls else "end_turn",
        usage=SimpleNamespace(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        ),
    )


class _GroqStream:
    def __init__(self, stream: Any):
        self._stream = stream
        self._input_tokens = 0
        self._output_tokens = 0

    def __enter__(self) -> "_GroqStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        close = getattr(self._stream, "close", None)
        if close:
            close()

    @property
    def text_stream(self):
        for chunk in self._stream:
            if chunk.usage:
                self._input_tokens = chunk.usage.prompt_tokens
                self._output_tokens = chunk.usage.completion_tokens
            text = chunk.choices[0].delta.content if chunk.choices else None
            if text:
                yield text

    def get_final_message(self) -> SimpleNamespace:
        return SimpleNamespace(usage=SimpleNamespace(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        ))


class _GroqMessages:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def create(
        self,
        *,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_openai_messages(messages, system),
            "max_completion_tokens": max_tokens,
        }
        openai_tools = _to_openai_tools(tools)
        if openai_tools:
            kwargs["tools"] = openai_tools
        return _normalise_response(self._client.chat.completions.create(**kwargs))

    def stream(
        self,
        *,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 1024,
        **_: Any,
    ) -> _GroqStream:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=_to_openai_messages(messages, system),
            max_completion_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        return _GroqStream(stream)


class GroqAnthropicAdapter:
    """Anthropic-like `.messages` surface backed by Groq."""

    def __init__(self, model: str = GROQ_MODEL):
        client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL)
        self.messages = _GroqMessages(client, model)


def get_text_client():
    provider = selected_provider()
    if provider == "groq":
        return GroqAnthropicAdapter()
    if provider == "anthropic":
        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    raise RuntimeError("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")

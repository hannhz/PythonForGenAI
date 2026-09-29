# Module 06 - OpenAI & Anthropic APIs
# 6.4 Tool Calling (Function Calling) - OpenAI
#
# NOTE: This script makes REAL, billed API calls to OpenAI (two calls per
# run). It requires a valid OPENAI_API_KEY in a .env file in this folder.

from openai import OpenAI
import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_openai_compatible_client, has_usable_key

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"],
            },
        },
    }
]


def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o": {
            "context_tokens": 128_000,
            "input_usd_per_million_tokens": 2.50,
        },
        "claude-sonnet-5": {
            "context_tokens": 1_000_000,
            "input_usd_per_million_tokens": 2.00,
        },
    }
    return db.get(model_name, {"error": "unknown model"})


def run_tool_call_demo() -> None:
    client, model, provider = get_openai_compatible_client()
    print(f"Provider: {provider}")

    messages = [{"role": "user", "content": "What is gpt-4o's context window?"}]

    response = client.chat.completions.create(
        model=model,
        tools=TOOLS,
        messages=messages,
    )

    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = get_model_info(**args)

        # Append assistant message and tool result
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        final = client.chat.completions.create(model=model, messages=messages)
        print(final.choices[0].message.content)


if __name__ == "__main__":
    if not (has_usable_key("GROQ_API_KEY") or has_usable_key("OPENAI_API_KEY")):
        print("Set GROQ_API_KEY or a valid OPENAI_API_KEY in a .env file.")
    else:
        run_tool_call_demo()

# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Basic message call
# The openai package follows a nearly identical pattern to Anthropic's SDK.
#
# NOTE: This script makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from openai import OpenAI
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_openai_compatible_client, has_usable_key

load_dotenv()


def basic_call() -> None:
    client, model, provider = get_openai_compatible_client()
    print(f"Provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": "You are a concise technical assistant."},
            {"role": "user", "content": "What is the difference between RAG and fine-tuning?"},
        ],
    )

    print(response.choices[0].message.content)
    print(f"Tokens used: {response.usage.total_tokens}")


if __name__ == "__main__":
    if not (has_usable_key("GROQ_API_KEY") or has_usable_key("OPENAI_API_KEY")):
        print("Set GROQ_API_KEY or a valid OPENAI_API_KEY in a .env file.")
    else:
        basic_call()

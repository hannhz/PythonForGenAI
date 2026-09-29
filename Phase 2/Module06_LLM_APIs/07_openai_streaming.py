# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Streaming with OpenAI
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


def stream_response() -> None:
    client, model, provider = get_openai_compatible_client()
    print(f"Provider: {provider}")

    stream = client.chat.completions.create(
        model=model,
        max_tokens=512,
        stream=True,
        messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)

    print()


if __name__ == "__main__":
    if not (has_usable_key("GROQ_API_KEY") or has_usable_key("OPENAI_API_KEY")):
        print("Set GROQ_API_KEY or a valid OPENAI_API_KEY in a .env file.")
    else:
        stream_response()

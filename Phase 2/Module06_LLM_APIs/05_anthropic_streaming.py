# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Streaming responses
# Streaming lets you display tokens as they arrive instead of waiting for the
# full response - essential for chat UIs.
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()


def stream_response() -> None:
    client = get_text_client()

    with client.messages.stream(
        model="claude-sonnet-5",
        max_tokens=512,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": "List 5 use cases for vector databases."}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print()  # newline after stream ends

    # Access final message and usage after stream completes
    final = stream.get_final_message()
    print(f"\nTotal tokens: {final.usage.input_tokens + final.usage.output_tokens}")


if __name__ == "__main__":
    if not has_text_provider():
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()}")
        stream_response()

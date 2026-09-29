# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Basic message call
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


def basic_call() -> None:
    client = get_text_client()

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[
            {"role": "user", "content": "What is retrieval-augmented generation?"}
        ],
    )

    # Response text is in the first content block
    print(message.content[0].text)

    # Usage stats
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"Output tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    if not has_text_provider():
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()}")
        basic_call()

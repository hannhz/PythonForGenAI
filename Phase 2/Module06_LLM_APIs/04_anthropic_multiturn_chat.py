# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Multi-turn conversation
# The Anthropic API is stateless - you must send the full conversation history
# on every call. Build the history yourself.
#
# NOTE: This is an interactive CLI chat loop that makes REAL, billed API calls.
# Run it directly (python 04_anthropic_multiturn_chat.py) and type "exit" to quit.

import anthropic
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()

client = get_text_client() if has_text_provider() else None


def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=system,
            messages=history,
        )

        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})
        print(f"Assistant ({selected_provider()}): {assistant_text}\n")


if __name__ == "__main__":
    if client is None:
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()} (type 'exit' to quit)")
        chat(system="You are a helpful Python tutor.")

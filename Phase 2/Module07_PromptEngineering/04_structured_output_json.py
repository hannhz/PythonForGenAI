# Module 07 - Prompt Engineering
# 7.5 Structured Output (JSON Mode)
# Getting models to return valid, parseable JSON is one of the most common
# production requirements. There are two approaches.
#
# NOTE: Both demos make REAL, billed API calls (Anthropic and OpenAI
# respectively). They require ANTHROPIC_API_KEY / OPENAI_API_KEY in a .env
# file in this folder to run.

import json
import os
import sys
from pathlib import Path

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import (
    get_openai_compatible_client,
    get_text_client,
    has_text_provider,
    has_usable_key,
    selected_provider,
)

load_dotenv()


# ── Approach 1: Prompt-enforced JSON (works with any model) ──────────────────

ANTHROPIC_SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

COMPANY_TEXTS = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude "
    "AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San "
    "Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:
    client = get_text_client()
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=256,
        thinking={"type": "disabled"},
        system=ANTHROPIC_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = resp.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def run_anthropic_demo() -> None:
    for text in COMPANY_TEXTS:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print()


# ── Approach 2: OpenAI JSON mode (response_format enforces valid JSON) ───────

def run_openai_json_mode_demo() -> None:
    client, model, provider = get_openai_compatible_client()
    print(f"JSON mode provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},   # enforces valid JSON
        messages=[
            {
                "role": "system",
                "content": """Extract entities. Return JSON with this schema:
{"people": [string], "organizations": [string], "locations": [string]}""",
            },
            {
                "role": "user",
                "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla.",
            },
        ],
    )

    result = json.loads(response.choices[0].message.content)
    print(result)
    # {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}


if __name__ == "__main__":
    if has_text_provider():
        print(f"=== Approach 1: prompt-enforced JSON ({selected_provider()}) ===")
        run_anthropic_demo()
    else:
        print("Skipping prompt-enforced demo - no Groq/Anthropic key set.")

    if has_usable_key("GROQ_API_KEY") or has_usable_key("OPENAI_API_KEY"):
        print("=== Approach 2: OpenAI-compatible json_object mode ===")
        run_openai_json_mode_demo()
    else:
        print("Skipping JSON mode - no Groq/valid OpenAI key set.")

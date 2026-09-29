# Module 07 - Prompt Engineering
# 7.3 Few-Shot Prompting
# Few-shot prompting provides examples of the desired input -> output mapping
# inside the prompt. It is the fastest way to teach a model an unusual format
# or task without fine-tuning.
#
# NOTE: This script makes REAL API calls through Groq or Anthropic (one per
# test input). It requires a valid provider key in a .env file.

import json
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()

FEW_SHOT_SYSTEM = """You are a data extractor. Given a raw AI benchmark result string,
extract: model name, task, and score as a JSON object.

Model-name normalization rules:
- Use lowercase letters.
- Replace spaces with hyphens.
- Copy version numbers exactly: "3.1" MUST remain "3.1", never "3-1".

Task extraction rules:
- Copy the complete task name after the word "on".
- Preserve qualifiers such as "science", "math", and "Verified".
- Do not shorten or paraphrase the task name.

Examples:

Input: "GPT-4o scored 87.3% on the MMLU science subset"
Output: {"model": "gpt-4o", "task": "MMLU science", "score": 87.3}

Input: "Claude Sonnet 4.5 achieved 92.1 on HumanEval"
Output: {"model": "claude-sonnet-4-5", "task": "HumanEval", "score": 92.1}

Input: "Gemini 1.5 Pro: 78.9% accuracy on GSM8K math"
Output: {"model": "gemini-1.5-pro", "task": "GSM8K math", "score": 78.9}

Before responding, silently verify that version dots and the complete task
name are preserved. Return ONLY the JSON object. No explanation."""

TEST_INPUTS = [
    "GPT-4o-mini reached 82.0% on MMLU",
    "Llama 3.1 70B: 86.4 on TruthfulQA",
    "Claude Opus 4.5 scored 96.7% on SWE-bench Verified",
]


def run_few_shot_demo() -> None:
    client = get_text_client()

    for text in TEST_INPUTS:
        resp = client.messages.create(
            model="claude-sonnet-5",
            # Reasoning models can spend part of this budget before producing
            # their final answer. A larger limit prevents an empty response.
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=FEW_SHOT_SYSTEM,
            messages=[{"role": "user", "content": text}],
        )
        print(f"Input: {text}")

        text_blocks = [
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", "").strip()
        ]
        if not text_blocks:
            print("Output: ERROR - provider returned no text. Try running this input again.\n")
            continue

        raw_output = "\n".join(text_blocks).strip()
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            print(f"Output: ERROR - response is not valid JSON ({exc.msg})")
            print(f"Raw output: {raw_output}\n")
            continue

        print(f"Output: {json.dumps(parsed_output, ensure_ascii=False)}\n")


if __name__ == "__main__":
    if not has_text_provider():
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()}")
        run_few_shot_demo()

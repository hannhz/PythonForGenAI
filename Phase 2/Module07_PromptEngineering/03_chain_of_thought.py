# Module 07 - Prompt Engineering
# 7.4 Chain-of-Thought Prompting
# Chain-of-thought (CoT) prompting asks the model to reason step by step before
# giving a final answer. It consistently improves accuracy on tasks requiring
# multi-step reasoning - math, logic, planning.
#
# NOTE: This script makes REAL, billed API calls to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()

# Without CoT - model jumps to answer, more likely to be wrong
DIRECT_PROMPT = (
    "If a model costs $3.00 per million input tokens and $15.00 per million "
    "output tokens, and a request uses 2,400 input tokens and 800 output "
    "tokens, what is the total cost in USD?"
)

# With CoT - model reasons through each step
COT_PROMPT = """If a model costs $3.00 per million input tokens and $15.00 per million output tokens,
and a request uses 2,400 input tokens and 800 output tokens,
what is the total cost in USD?

Think through this step by step before giving the final answer."""

# Zero-shot CoT: just adding "think step by step"
ZERO_SHOT_COT = """Solve this problem. Think step by step, showing each calculation.
Finally, state: ANSWER: $X.XXXXXX

Problem: A pipeline makes 50 API calls per hour. Each call uses an average of 1,200 input tokens
and 400 output tokens. The model costs $3.00/M input and $15.00/M output.
What is the daily cost?"""

PLAIN_TEXT_SYSTEM = """Write calculations as plain terminal-friendly text.
Do not use LaTeX commands or math delimiters. Use / for division and * for
multiplication. Keep the explanation concise."""

# Structured CoT with XML tags - makes it easy to parse the final answer
STRUCTURED_SYSTEM = """Solve problems using this exact format:

Use plain terminal-friendly text only. Do not use LaTeX commands or math delimiters.

<thinking>
Step-by-step reasoning here.
</thinking>

<answer>
The final answer only, no reasoning.
</answer>"""

STRUCTURED_QUESTION = (
    "A RAG pipeline retrieves 5 documents, each 400 tokens. The query is 50 "
    "tokens. The model has a 4096 token limit for context. How many tokens "
    "remain for the response?"
)


def compare_direct_vs_cot() -> None:
    client = get_text_client()

    for label, prompt in [("Direct", DIRECT_PROMPT), ("CoT", COT_PROMPT), ("Zero-shot CoT", ZERO_SHOT_COT)]:
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=PLAIN_TEXT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"=== {label} ===")
        text = response_text(resp)
        print(text if text else "Provider returned no visible text; try again.")
        print()


def response_text(response) -> str:
    """Collect visible text blocks without assuming content[0] exists."""
    return "\n".join(
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
        and getattr(block, "text", "").strip()
    ).strip()


def structured_cot_demo() -> None:
    client = get_text_client()

    resp = client.messages.create(
        model="claude-sonnet-5",
        # Groq reasoning models may use part of this budget internally before
        # returning visible text. A larger limit avoids an empty response.
        max_tokens=1024,
        thinking={"type": "disabled"},
        system=STRUCTURED_SYSTEM,
        messages=[{"role": "user", "content": STRUCTURED_QUESTION}],
    )

    text = response_text(resp)
    if not text:
        print("Reasoning: provider returned no visible text")
        print("Answer:    try running this section again")
        return

    # Extract sections. IGNORECASE also accepts <Thinking>/<Answer>.
    thinking = re.search(
        r"<thinking\s*>(.*?)</thinking\s*>", text, re.DOTALL | re.IGNORECASE
    )
    answer = re.search(
        r"<answer\s*>(.*?)</answer\s*>", text, re.DOTALL | re.IGNORECASE
    )

    if thinking and answer:
        print("Reasoning:", thinking.group(1).strip())
        print("Answer:   ", answer.group(1).strip())
    else:
        # The calculation is still useful when a model ignores the XML format.
        print("Response:")
        print(text)


if __name__ == "__main__":
    if not has_text_provider():
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()}")
        compare_direct_vs_cot()
        print("=" * 40)
        structured_cot_demo()

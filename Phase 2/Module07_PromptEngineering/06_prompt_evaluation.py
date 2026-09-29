# Module 07 - Prompt Engineering
# 7.7 Prompt Evaluation
# A prompt that works once is not a prompt - it is luck. Systematically
# evaluate prompts across a test set before using them in production.
#
# NOTE: evaluate_prompt() makes one REAL, billed API call per test case. It
# requires a valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]   # at least one must appear in response
    must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase], client=None) -> dict:
    """Run a prompt against test cases and return pass rate + details."""
    if not cases:
        raise ValueError("At least one evaluation case is required.")
    client = client or get_text_client()
    results = []

    for case in cases:
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=256,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": case.input_text}],
        )
        text = resp.content[0].text.strip()

        # Check keyword hit
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)

        # Check JSON validity if required
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    pass_rate = sum(r["passed"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "results": results}


# Test a classification prompt
CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

TEST_CASES = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]


if __name__ == "__main__":
    if not has_text_provider():
        print("Set GROQ_API_KEY or ANTHROPIC_API_KEY in a .env file.")
    else:
        print(f"Provider: {selected_provider()}")
        report = evaluate_prompt(CLASSIFY_SYSTEM, TEST_CASES)
        print(f"Pass rate: {report['pass_rate']:.0%}")
        for r in report["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['input']!r} -> {r['response_preview']!r}")

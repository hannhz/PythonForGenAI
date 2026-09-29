# Module 07 - Prompt Engineering
# 7.8 Module 07 Exercises
#
# Exercises 2 and 3 (tiers 1-2) are fully testable offline. Exercises 1 and 4,
# and tier 3 of exercise 3, need a real ANTHROPIC_API_KEY to actually call the
# model - the code is complete and correct, just gated behind a key check.

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_provider import get_text_client, has_text_provider, selected_provider

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


# ── Shared: a small PromptTemplate (same as section 7.6) ──────────────────────

@dataclass
class PromptTemplate:
    name: str
    system: str
    user: str
    version: str = "1.0"
    required_vars: list[str] = field(default_factory=list)

    def __post_init__(self):
        formatter = Formatter()
        combined = self.system + self.user
        self.required_vars = [
            fname for _, fname, _, _ in formatter.parse(combined)
            if fname is not None
        ]

    def render(self, **kwargs: Any) -> tuple[str, str]:
        missing = set(self.required_vars) - set(kwargs)
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return self.system.format(**kwargs), self.user.format(**kwargs)


# ── Exercise 1 ───────────────────────────────────────────────────────────────
# Write three versions of a system prompt for a "code review assistant" -
# basic, intermediate, and expert-level. Evaluate all three on the same 5
# code snippets and compare output quality.

CODE_REVIEW_BASIC = "You are a code reviewer. Review the code and point out any issues."

CODE_REVIEW_INTERMEDIATE = """You are a code reviewer. Review the given code for:
- Bugs and correctness issues
- Readability
- Basic security concerns

List each issue you find with a short explanation."""

CODE_REVIEW_EXPERT = """You are a senior software engineer performing a production code review.

Check for:
- Correctness bugs (logic errors, edge cases, off-by-one errors)
- Security vulnerabilities (injection, unsafe deserialization, secrets in code)
- Performance issues (unnecessary loops, N+1 queries, blocking calls)
- Maintainability (naming, duplication, missing error handling)

Rules:
- Be direct, no filler praise.
- For every issue: state Impact, then give the corrected code.
- If the code has no issues, say so in one sentence.

Format: numbered list, one issue per item."""

CODE_REVIEW_CASES = [
    ("def add(a, b):\n    return a + b", ("correct", "no issue", "looks good")),
    (
        "def get_user(id):\n    return db.execute(f'SELECT * FROM users WHERE id={id}')",
        ("sql injection", "parameterized", "parameterised"),
    ),
    ("def divide(a, b):\n    return a / b", ("zero", "zerodivision")),
    (
        "results = []\nfor item in items:\n    for other in items:\n        results.append(item == other)",
        ("quadratic", "o(n", "performance"),
    ),
    (
        "password = 'admin123'\ndef login(pw):\n    return pw == password",
        ("hard-coded", "hardcoded", "secret", "password"),
    ),
]


def compare_review_prompts(client=None) -> dict[str, dict]:
    """Evaluate all three prompts against the same five code snippets.

    The keyword rubric makes the comparison repeatable. A real run makes
    15 model calls, so it is only started when an API key is available.
    """
    client = client or get_text_client()
    prompts = {
        "basic": CODE_REVIEW_BASIC,
        "intermediate": CODE_REVIEW_INTERMEDIATE,
        "expert": CODE_REVIEW_EXPERT,
    }
    report = {}
    for label, system in prompts.items():
        print(f"\n=== {label} ===")
        case_results = []
        for case_number, (snippet, expected_terms) in enumerate(CODE_REVIEW_CASES, 1):
            resp = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=300,
                thinking={"type": "disabled"},
                system=system,
                messages=[{"role": "user", "content": snippet}],
            )
            output = resp.content[0].text
            matched = [term for term in expected_terms if term in output.lower()]
            passed = bool(matched)
            case_results.append({"case": case_number, "passed": passed, "matched": matched})
            print(f"Case {case_number}: {'PASS' if passed else 'FAIL'} | matched={matched}")
        pass_rate = sum(item["passed"] for item in case_results) / len(case_results)
        report[label] = {"pass_rate": pass_rate, "cases": case_results}
        print(f"Pass rate: {pass_rate:.0%}")
    return report


# ── Exercise 2 ───────────────────────────────────────────────────────────────
# Build a PromptLibrary class that stores named PromptTemplate instances,
# supports saving/loading to JSON, and tracks which version of each template
# produced the last evaluation run.

class PromptLibrary:
    """In-memory + JSON-persisted store of named PromptTemplate instances,
    with tracking of the last evaluation run per template."""

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._last_eval: dict[str, dict] = {}   # name -> {version, pass_rate, ...}

    def add(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        return self._templates[name]

    def names(self) -> list[str]:
        return list(self._templates.keys())

    def record_evaluation(self, name: str, pass_rate: float) -> None:
        """Call this after running an evaluation against a template, to track
        which version produced that result."""
        template = self._templates[name]
        self._last_eval[name] = {"version": template.version, "pass_rate": pass_rate}

    def last_evaluation(self, name: str) -> Optional[dict]:
        return self._last_eval.get(name)

    def save(self, path: str) -> None:
        data = {
            "templates": {
                name: {"name": t.name, "system": t.system, "user": t.user, "version": t.version}
                for name, t in self._templates.items()
            },
            "last_eval": self._last_eval,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "PromptLibrary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lib = cls()
        for t in data["templates"].values():
            lib.add(PromptTemplate(name=t["name"], system=t["system"], user=t["user"], version=t["version"]))
        lib._last_eval = data.get("last_eval", {})
        return lib


# ── Exercise 3 ───────────────────────────────────────────────────────────────
# Implement automatic JSON repair: safe_json_parse(text) -> dict that first
# tries json.loads, then strips markdown fences, then asks the model to fix
# the JSON if it is still invalid.

def safe_json_parse(text: str, client=None, model: str = "claude-sonnet-5") -> dict:
    """Tier 1: direct parse. Tier 2: strip markdown fences. Tier 3 (needs a
    client): ask the model to repair the JSON."""
    # Tier 1
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tier 2 - strip ```json ... ``` or ``` ... ``` fences
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Tier 3 - ask the model to repair it
    if client is None:
        raise ValueError("JSON could not be parsed and no client was provided for repair.")

    resp = client.messages.create(
        model=model,
        max_tokens=512,
        thinking={"type": "disabled"},
        system="Fix the following text so it becomes valid JSON. Return ONLY the corrected JSON, nothing else.",
        messages=[{"role": "user", "content": text}],
    )
    repaired = resp.content[0].text.strip()
    repaired = re.sub(r"^```(?:json)?\s*|\s*```$", "", repaired)
    return json.loads(repaired)   # let this raise if the model still couldn't fix it


# ── Exercise 4 ───────────────────────────────────────────────────────────────
# Design a CoT prompt: given a list of 10 LLM evaluation scores across 3
# tasks, rank the models and write a 2-sentence recommendation. Verify it
# works correctly on at least 3 different inputs.

RANKING_COT_SYSTEM = """You rank LLMs based on evaluation score records.

Think step by step:
1. Group records by model and list each model's average score.
2. Rank models from highest to lowest average.
3. Identify the strongest and weakest evaluated task for each model.

Then write exactly 2 sentences recommending which model to use and why.

Format your response as:
<reasoning>
...step-by-step reasoning...
</reasoning>
<recommendation>
...exactly 2 sentences...
</recommendation>"""

RANKING_TEST_INPUTS = [
    """10 score records (0-100) across qa, summarise, and code:
gpt-4o,qa,88
gpt-4o,summarise,82
gpt-4o,code,91
claude-sonnet-5,qa,91
claude-sonnet-5,summarise,89
claude-sonnet-5,code,93
gemini-1.5-pro,qa,85
gemini-1.5-pro,summarise,90
gemini-1.5-pro,code,80
gpt-4o,qa,87""",
    """10 score records (0-100) across qa, summarise, and code:
model-a,qa,70
model-a,summarise,95
model-a,code,60
model-b,qa,90
model-b,summarise,60
model-b,code,95
model-c,qa,80
model-c,summarise,80
model-c,code,80
model-b,qa,92""",
    """10 score records (0-100) across qa, summarise, and code:
fast-model,qa,75
fast-model,summarise,75
fast-model,code,75
big-model,qa,95
big-model,summarise,95
big-model,code,95
cheap-model,qa,65
cheap-model,summarise,70
cheap-model,code,68
fast-model,code,77""",
]


def validate_ranking_response(text: str) -> tuple[bool, str]:
    """Check that the recommendation tag contains exactly two sentences."""
    match = re.search(r"<recommendation>(.*?)</recommendation>", text, re.DOTALL)
    if not match:
        return False, "recommendation tag not found"
    recommendation = match.group(1).strip()
    sentences = re.findall(r"[^.!?]+[.!?](?:\s|$)", recommendation)
    if len(sentences) != 2:
        return False, f"expected 2 sentences, found {len(sentences)}"
    return True, recommendation


def run_ranking_cot_demo(client=None) -> list[bool]:
    client = client or get_text_client()
    validations = []
    for i, scores_text in enumerate(RANKING_TEST_INPUTS, 1):
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=512,
            thinking={"type": "disabled"},
            system=RANKING_COT_SYSTEM,
            messages=[{"role": "user", "content": scores_text}],
        )
        text = resp.content[0].text
        valid, detail = validate_ranking_response(text)
        validations.append(valid)
        print(f"--- Input {i} ---")
        print("Validation:", "PASS" if valid else "FAIL")
        print("Recommendation:", detail)
        print()
    return validations


if __name__ == "__main__":
    has_key = has_text_provider()
    if has_key:
        print(f"Provider: {selected_provider()}")

    print("=== Exercise 1: compare_review_prompts (needs API key) ===")
    if has_key:
        compare_review_prompts()
    else:
        print("Skipped - no GROQ_API_KEY/ANTHROPIC_API_KEY set.")

    print("\n=== Exercise 2: PromptLibrary (offline) ===")
    DATA_DIR.mkdir(exist_ok=True)
    library = PromptLibrary()
    library.add(PromptTemplate(
        name="qa", version="1.0",
        system="You are a {domain} expert.",
        user="Question: {question}",
    ))
    library.record_evaluation("qa", pass_rate=0.8)
    lib_path = DATA_DIR / "prompt_library.json"
    library.save(str(lib_path))
    reloaded = PromptLibrary.load(str(lib_path))
    print("Templates:", reloaded.names())
    print("Last eval for 'qa':", reloaded.last_evaluation("qa"))

    print("\n=== Exercise 3: safe_json_parse (tiers 1-2 offline) ===")
    print(safe_json_parse('{"a": 1}'))                          # tier 1
    print(safe_json_parse('```json\n{"a": 1}\n```'))            # tier 2
    if has_key:
        client = get_text_client()
        print(safe_json_parse('{"a": 1,}', client=client))      # tier 3 (trailing comma)
    else:
        print("Tier 3 (model repair) skipped - no text-provider key set.")

    print("\n=== Exercise 4: ranking CoT prompt (needs API key) ===")
    if has_key:
        run_ranking_cot_demo()
    else:
        print("Skipped - no GROQ_API_KEY/ANTHROPIC_API_KEY set.")

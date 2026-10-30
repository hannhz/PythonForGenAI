# Module 09 - LangChain
# 9.5 Output Parsers
# Parsers turn raw model output into Python values. StrOutputParser returns
# plain text, JsonOutputParser returns a dict, and with_structured_output()
# makes the model itself return a validated Pydantic object.
#
# NOTE: The offline parser demo needs no key. The structured-output demo makes
# a REAL, billed API call and requires a valid OPENAI_API_KEY (or
# OPENROUTER_API_KEY / ANTHROPIC_API_KEY) in a .env file in this folder.
# with_structured_output() is schema-enforced on OpenAI models.

from pydantic import BaseModel, Field

from _openrouter_client import ensure_ready, get_chat_model, is_ready

# Print a clear message (instead of a traceback) when a library is missing.
if __name__ == "__main__" and not ensure_ready(["langchain_core"]):
    raise SystemExit(0)

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ── Pydantic schema for structured output ──────────────────────────────────────

class ModelComparison(BaseModel):
    """Structured answer for a model-comparison question."""

    models: list[str] = Field(description="Model names compared")
    best_for_cost: str = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str = Field(description="Overall recommendation")


# ── Structured output chain ────────────────────────────────────────────────────

def build_comparison_chain(llm):
    """prompt | llm | JsonOutputParser -> dict -> ModelComparison.

    Uses a plain parser instead of with_structured_output() so that a model
    returning invalid JSON raises OutputParserException *inside the chain*,
    which .with_fallbacks() can catch and retry on a second model."""
    parser = JsonOutputParser(pydantic_object=ModelComparison)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI model comparison expert. "
                   "Return ONLY valid JSON matching this schema:\n{format_instructions}"),
        ("human", "Compare these models for a startup on a budget: {models}"),
    ])
    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
        | (lambda data: ModelComparison(**data))
    )


def offline_parser_demo() -> None:
    """Show the two plain parsers on a canned model reply (no API call)."""
    reply = (
        '{"models": ["gpt-4o", "gpt-4o-mini"], "best_for_cost": "gpt-4o-mini", '
        '"best_for_quality": "gpt-4o", "recommendation": "Start with gpt-4o-mini."}'
    )

    text = StrOutputParser().invoke(AIMessage(content=reply))
    print("StrOutputParser  ->", type(text).__name__)

    data = JsonOutputParser(pydantic_object=ModelComparison).parse(reply)
    print("JsonOutputParser ->", type(data).__name__, "with keys", list(data))

    comparison = ModelComparison(**data)   # validate the dict against the schema
    print("Validated        ->", comparison.best_for_cost)


if __name__ == "__main__":
    print("=== Offline parser demo ===")
    offline_parser_demo()

    print("\n=== Structured output (real model) ===")
    if is_ready(llm="auto"):
        from _openrouter_client import OPENROUTER_FALLBACK_MODEL

        # Small free models (e.g. Gemma) sometimes return output that fails
        # schema validation, or the provider rate-limits / 5xx's. Try the
        # primary model first, then transparently fall back to a second model.
        models_to_try = [
            ("primary", get_chat_model("auto")),
            ("fallback", get_chat_model("auto", model=OPENROUTER_FALLBACK_MODEL)),
        ]

        result = None
        for label, llm in models_to_try:
            if llm is None:
                continue
            if label == "fallback":
                print(f"(trying fallback model: {OPENROUTER_FALLBACK_MODEL})")
            try:
                result = build_comparison_chain(llm).invoke({
                    "models": "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"
                })
                break
            except Exception as exc:
                print(f"{label.capitalize()} model failed: {exc}")
                if label == "fallback":
                    print("No more fallback models available.")
        else:
            result = None

        if result is not None:
            print(f"Best for cost: {result.best_for_cost}")
            print(f"Best for quality: {result.best_for_quality}")
            print(f"Recommendation: {result.recommendation}")
    else:
        ensure_ready(llm="auto")
        print("Skipped the real-model demo.")

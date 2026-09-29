# Module 06 - OpenAI & Anthropic APIs
# 6.5 Vision - Images as Input
# Anthropic and Gemini accept images alongside text in the same message.
#
# NOTE: This script makes a REAL API call. Set GEMINI_API_KEY (recommended for
# this course fallback) or a funded ANTHROPIC_API_KEY in this folder's .env.

import anthropic
import os
import base64
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

anthropic_client = (
    anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if os.getenv("ANTHROPIC_API_KEY")
    else None
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
}


def _has_usable_key(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    placeholders = ("...", "masukkan", "your_", "replace", "isi_")
    return bool(value and not any(marker in value for marker in placeholders))


def vision_provider() -> str | None:
    """Choose a provider independently from the Groq text fallback."""
    requested = os.getenv("VISION_PROVIDER", "").strip().lower()
    if requested and requested not in {"gemini", "anthropic"}:
        raise ValueError("VISION_PROVIDER must be 'gemini' or 'anthropic'.")

    if requested == "gemini":
        if not _has_usable_key("GEMINI_API_KEY"):
            raise ValueError(
                "Replace the GEMINI_API_KEY placeholder in .env with your real API key."
            )
        return "gemini"
    if requested == "anthropic":
        if anthropic_client is None:
            raise ValueError("ANTHROPIC_API_KEY is required when VISION_PROVIDER=anthropic.")
        return "anthropic"

    # Prefer Gemini for vision. Groq can remain the text provider.
    if _has_usable_key("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("LLM_PROVIDER", "").strip().lower() == "anthropic" and anthropic_client:
        return "anthropic"
    return None


def _gemini_client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Google GenAI SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _describe_with_gemini(data: bytes, media_type: str, prompt: str) -> str:
    from google.genai import types
    from google.genai.errors import ServerError

    with _gemini_client() as client:
        models = [GEMINI_MODEL]
        if GEMINI_MODEL != "gemini-2.5-flash":
            models.append("gemini-2.5-flash")

        for index, model in enumerate(models):
            try:
                chat = client.chats.create(model=model)
                response = chat.send_message(
                    [
                        prompt,
                        types.Part.from_bytes(data=data, mime_type=media_type),
                    ]
                )
                if response.text:
                    return response.text
            except ServerError:
                if index == len(models) - 1:
                    raise
    return "(no text returned by Gemini)"


def _describe_with_anthropic(content: dict, prompt: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=512,
        thinking={"type": "disabled"},
        messages=[{
            "role": "user",
            "content": [content, {"type": "text", "text": prompt}],
        }],
    )
    return response.content[0].text


# Option A: URL (fastest)
def describe_image_url(url: str) -> str:
    prompt = "Describe what you see in this image."
    if vision_provider() == "anthropic":
        return _describe_with_anthropic(
            {"type": "image", "source": {"type": "url", "url": url}}, prompt
        )

    request = Request(url, headers={"User-Agent": "Python-For-GenAI/1.0"})
    with urlopen(request, timeout=30) as response:
        data = response.read()
        media_type = response.headers.get_content_type()
    return _describe_with_gemini(data, media_type, prompt)


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lstrip(".").lower()
    if ext not in MEDIA_TYPES:
        raise ValueError(f"Unsupported image extension: .{ext}")
    media_type = MEDIA_TYPES[ext]
    prompt = "What is in this image?"

    if vision_provider() == "anthropic":
        b64 = base64.standard_b64encode(data).decode()
        return _describe_with_anthropic(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64,
                },
            },
            prompt,
        )
    return _describe_with_gemini(data, media_type, prompt)


if __name__ == "__main__":
    try:
        provider = vision_provider()
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        raise SystemExit(1) from None
    if provider is None:
        print("Vision provider is not configured. Add GEMINI_API_KEY and "
              "VISION_PROVIDER=gemini to this folder's .env file.")
    else:
        print(f"Vision provider: {provider}")
        text = describe_image_url(
            "https://raw.githubusercontent.com/python-pillow/Pillow/main/"
            "Tests/images/hopper.jpg"
        )
        print(text)

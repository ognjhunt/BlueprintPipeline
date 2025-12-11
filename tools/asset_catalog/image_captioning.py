"""Utility helpers for captioning asset thumbnails with a lightweight vision model."""

import importlib.util
from pathlib import Path
from typing import Optional


DEFAULT_CAPTION_PROMPT = (
    "Provide a concise caption for this asset thumbnail. "
    "Focus on the object type, material, colors, and style in one sentence."
)


def caption_thumbnail(
    image_path: str,
    api_key: Optional[str] = None,
    model: str = "gemini-1.5-flash",
    prompt: str = DEFAULT_CAPTION_PROMPT,
) -> str:
    """Generate a caption for a thumbnail using Gemini Vision when available.

    Falls back to a filename-based caption when the Gemini SDK is unavailable or
    when no API key is provided, so the utility remains usable in offline or
    testing environments.
    """

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Thumbnail not found: {image_path}")

    genai_spec = importlib.util.find_spec("google.generativeai")
    if genai_spec is None:
        return _fallback_caption(path)

    import google.generativeai as genai

    if not api_key:
        return _fallback_caption(path)

    genai.configure(api_key=api_key, client_options=None)
    model_client = genai.GenerativeModel(model)

    try:
        response = model_client.generate_content(
            [
                prompt,
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": path.read_bytes(),
                    }
                },
            ],
            stream=False,
        )
        text = getattr(response, "text", None) or _extract_response_text(response)
    except Exception:
        return _fallback_caption(path)

    if text:
        cleaned = text.strip()
        if cleaned:
            return cleaned

    return _fallback_caption(path)


def _extract_response_text(response: object) -> str:
    """Extract text content from a Gemini response object."""
    candidate = getattr(response, "candidates", None)
    if not candidate:
        return ""

    parts = getattr(candidate[0], "content", None)
    if not parts or not getattr(parts, "parts", None):
        return ""

    for part in parts.parts:
        maybe_text = getattr(part, "text", None)
        if maybe_text:
            return str(maybe_text)
    return ""


def _fallback_caption(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    cleaned = stem.strip()
    return cleaned or "asset thumbnail"

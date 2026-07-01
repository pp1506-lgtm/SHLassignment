"""
gemini_client.py
Thin wrapper around the Gemini Flash API using google-genai SDK (new).
Handles retries, JSON parsing, and timeout enforcement.
"""
import json
import os
import re
import time
from typing import Optional

from google import genai
from google.genai import types

GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 2048
TEMPERATURE = 0.2  # Low temperature for consistent, grounded responses
REQUEST_TIMEOUT = 22  # seconds — well under 30s API timeout


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    return genai.Client(api_key=api_key)


_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = _get_client()
    return _client


def chat_completion(
    system_prompt: str,
    messages: list[dict],
    *,
    timeout: int = REQUEST_TIMEOUT,
) -> dict:
    """
    Send conversation to Gemini and parse JSON response.
    Returns parsed dict matching the agent response schema.
    Falls back to a safe default on any error so the API never 500s.
    """
    client = get_client()

    # Build contents list in google-genai format
    contents = []

    # Inject system as first user turn (no native system role in basic chat)
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=f"[SYSTEM INSTRUCTIONS]\n{system_prompt}\n[END SYSTEM INSTRUCTIONS]\n\nBegin.")]
        )
    )
    contents.append(
        types.Content(
            role="model",
            parts=[types.Part(text="Understood. I am ready to help with SHL assessment selection.")]
        )
    )

    # Add conversation messages
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        response_mime_type="application/json",
    )

    start = time.time()
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )
        raw = response.text.strip()
        return _parse_response(raw)
    except Exception as exc:
        # Return a safe clarification response rather than crashing
        return {
            "state": "CLARIFY",
            "reply": (
                "I'm experiencing a brief technical issue. "
                "Could you please describe the role you're hiring for?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
            "_error": str(exc),
        }


def _parse_response(raw: str) -> dict:
    """
    Extract and validate JSON from model output.
    The model is instructed to return bare JSON, but may wrap it in ```json blocks.
    """
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.M)
    raw = re.sub(r"\s*```$", "", raw, flags=re.M)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return _fallback_clarify()
        else:
            return _fallback_clarify()

    return _validate_and_normalise(parsed)


def _validate_and_normalise(parsed: dict) -> dict:
    """Ensure all required fields exist and have valid types."""
    valid_states = {"CLARIFY", "RECOMMEND", "COMPARE", "REFINE", "REFUSE", "DONE"}
    state = parsed.get("state", "CLARIFY")
    if state not in valid_states:
        state = "CLARIFY"

    reply = str(parsed.get("reply", "Could you tell me more about the role?"))

    raw_recs = parsed.get("recommendations", [])
    if not isinstance(raw_recs, list):
        raw_recs = []

    recs = []
    for r in raw_recs:
        if not isinstance(r, dict):
            continue
        name = r.get("name", "")
        url = r.get("url", "")
        test_type = r.get("test_type", "K")
        if name and url and "shl.com" in url:
            recs.append({"name": name, "url": url, "test_type": test_type})

    recs = recs[:10]
    eoc = bool(parsed.get("end_of_conversation", False))

    return {
        "state": state,
        "reply": reply,
        "recommendations": recs,
        "end_of_conversation": eoc,
    }


def _fallback_clarify() -> dict:
    return {
        "state": "CLARIFY",
        "reply": "Could you describe the role you're hiring for in a bit more detail?",
        "recommendations": [],
        "end_of_conversation": False,
    }

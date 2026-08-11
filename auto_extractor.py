"""
auto_extractor.py  —  Person A's Backend Module
──────────────────────────────────────────────────
Auto-Schema Inference & Extraction Engine.

Takes raw unstructured text (e.g., pasted from a ChatGPT / Gemini response),
uses an LLM to automatically detect what columns/attributes exist,
and extracts every record into a structured pandas DataFrame.

Public API
----------
    extract_auto_dataset(raw_text, api_key, provider, model) -> pd.DataFrame
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd


# ─── Extraction Prompt ────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a precise data-extraction engine.
Your task is to read unstructured text and pull out EVERY structured record
into a clean JSON format.

INSTRUCTIONS
1. Read the text carefully.
2. Identify what kind of data records are described
   (products, people, statistics, rankings, etc.).
3. Determine the most appropriate column names from the attributes
   mentioned in the text.  Use clean, human-readable Title Case names.
4. Extract EVERY record — do not skip any.
5. For any attribute that is missing for a particular record, use null.
6. Numeric values must be numbers (int or float), NOT strings.
   Monetary values may remain as formatted strings (e.g. "$3.9 million").

OUTPUT FORMAT  (return ONLY this JSON — nothing else)
{{
  "columns": ["Column1", "Column2", "Column3"],
  "records": [
    {{"Column1": "value1", "Column2": 123, "Column3": null}},
    {{"Column1": "value2", "Column2": 456, "Column3": "abc"}}
  ]
}}

RULES
- Do NOT include any text before or after the JSON.
- Do NOT wrap in markdown code fences.
- If the text has no extractable data, return: {{"columns": [], "records": []}}

TEXT TO ANALYSE
───────────────
{raw_text}
───────────────
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_auto_dataset(
    raw_text: str,
    api_key: str,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
) -> pd.DataFrame:
    """
    Auto-detect schema from raw text and extract structured records.

    Parameters
    ----------
    raw_text : str
        The unstructured text pasted by the user.
    api_key : str
        The API key for the selected LLM provider.
    provider : str
        ``"gemini"`` or ``"openai"``.
    model : str
        The model identifier (e.g. ``"gemini-2.0-flash"``, ``"gpt-4o-mini"``).

    Returns
    -------
    pd.DataFrame
        A DataFrame with the auto-detected columns and extracted records.
        Returns an empty DataFrame if no data could be extracted.
    """
    if not raw_text or not raw_text.strip():
        return pd.DataFrame()

    prompt = _EXTRACTION_PROMPT.format(raw_text=raw_text.strip())

    # ── Call the appropriate LLM ──────────────────────────────────────────
    if provider == "openai":
        raw_json = _call_openai(prompt, api_key, model)
    else:
        raw_json = _call_gemini(prompt, api_key, model)

    # ── Parse & build DataFrame ───────────────────────────────────────────
    records = _parse_llm_response(raw_json)
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


# ─── LLM Callers ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Google Gemini via the google-genai SDK and return raw text."""
    from google import genai
    from google.genai import types as genai_types

    clean_key = api_key.strip()

    # Key format validation
    if clean_key.startswith("sk-"):
        raise ValueError(
            "You pasted an OpenAI API key (starts with 'sk-') into the Gemini key field. "
            "Please switch the provider in the sidebar to '🟢 OpenAI (GPT)'."
        )
    if clean_key.startswith("tvly-"):
        raise ValueError(
            "You pasted a Tavily web search key into the Gemini key field. "
            "Please enter a valid Gemini API key from Google AI Studio."
        )

    # Build de-duplicated model list
    candidates = _build_model_candidates(model, [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ])

    # Try both v1beta and v1 API versions across all candidate models
    api_versions = ["v1beta", "v1"]
    last_error: Optional[Exception] = None

    for api_ver in api_versions:
        try:
            client = genai.Client(
                api_key=clean_key,
                http_options={"api_version": api_ver},
            )
        except Exception:
            client = genai.Client(api_key=clean_key)

        for candidate in candidates:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                return response.text.strip()
            except Exception as exc:
                err_str = str(exc)
                if "404" in err_str or "NOT_FOUND" in err_str:
                    last_error = exc
                    continue
                # For non-404 errors (e.g. quota 429 or auth 401), raise immediately
                raise

    if last_error:
        raise last_error
    return "{}"


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI GPT and return raw text."""
    import openai

    client = openai.OpenAI(api_key=api_key.strip())

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data extraction engine. "
                    "Return ONLY valid JSON — no markdown, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=8192,
    )
    return response.choices[0].message.content.strip()


# ─── Response Parsing ─────────────────────────────────────────────────────────

def _parse_llm_response(raw_json: str) -> List[Dict[str, Any]]:
    """
    Parse the JSON string returned by the LLM.

    Handles multiple possible shapes:
      - {"columns": [...], "records": [...]}
      - {"records": [...]}
      - [...]  (plain array)
      - {"data": [...]}

    Returns a list of record dicts, or an empty list on failure.
    """
    # Strip any markdown fences the LLM might have added
    text = raw_json.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: try to find JSON inside the response
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    # Normalise shape
    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        # Preferred: {"records": [...]}
        if "records" in parsed and isinstance(parsed["records"], list):
            return parsed["records"]
        # Alt: {"data": [...]}
        if "data" in parsed and isinstance(parsed["data"], list):
            return parsed["data"]
        # Alt: {"results": [...]}
        if "results" in parsed and isinstance(parsed["results"], list):
            return parsed["results"]
        # Single record dict → wrap
        return [parsed]

    return []


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_model_candidates(selected: str, fallbacks: List[str]) -> List[str]:
    """Build de-duplicated model fallback list with selected model first."""
    candidates = [selected] + fallbacks
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

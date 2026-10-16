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
Your task is to read unstructured text and pull out EVERY structured entity/record
into a clean JSON format.

INSTRUCTIONS
1. Read the text carefully.
2. Extract EXACTLY ONE record per unique entity/product/subject described in the text.
   - Do NOT miss any distinct entity (e.g. if 6 smartphones are described, return exactly 6 records).
   - Do NOT create duplicate rows for the same entity.
3. Determine the most appropriate column names from the attributes mentioned in the text.
   - The first column MUST be the Entity/Product/Subject Name (e.g., "Model Name" or "Phone" or "Product").
   - Use clean, human-readable Title Case column names.
4. Extract every record — do not skip any.
5. For any attribute that is missing for a particular record, use null.
6. Numeric values must be numbers (int or float) where clear.
   Monetary and formatted values may remain as strings (e.g. "$1,299", "6.9-inch").

OUTPUT FORMAT  (return ONLY this JSON — nothing else)
{{
  "columns": ["Product Name", "Display", "Processor", "Price"],
  "records": [
    {{"Product Name": "Samsung Galaxy S26 Ultra", "Display": "6.9-inch", "Processor": "Snapdragon 8 Elite", "Price": "$1,299"}},
    {{"Product Name": "Apple iPhone 17 Pro Max", "Display": "6.9-inch", "Processor": "A19 Pro", "Price": "$1,199"}}
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
    model: str = "gemini-flash-lite-latest",
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
        The model identifier.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the auto-detected columns and extracted records.
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

    df = pd.DataFrame(records)
    # Deduplicate by primary entity column if present
    if not df.empty and len(df.columns) > 0:
        df = df.drop_duplicates(subset=[df.columns[0]], keep="first").reset_index(drop=True)

    return df


# ─── LLM Callers ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """
    Call Google Gemini with multi-layered fallback strategy:
      1. SDK call with JSON mode
      2. SDK call without JSON mode (plain text)
      3. Direct REST HTTP call via urllib
    """
    from google import genai
    from google.genai import types as genai_types
    import urllib.request
    import json

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

    candidates = _build_model_candidates(model, [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-pro-latest",
    ])

    last_error: Optional[Exception] = None

    # ── Attempt 1 & 2: SDK with and without response_mime_type ───────────────
    try:
        client = genai.Client(api_key=clean_key)
    except Exception:
        client = None

    if client is not None:
        for candidate in candidates:
            # 1A. Try with JSON mime type
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
                if response and response.text:
                    return response.text.strip()
            except Exception as exc:
                last_error = exc

            # 1B. Try without JSON mime type (plain text mode)
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as exc:
                last_error = exc

    # ── Attempt 3: Direct REST HTTP Call (guaranteed fallback) ───────────────
    for candidate in candidates:
        for api_ver in ["v1beta", "v1"]:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{candidate}:generateContent?key={clean_key}"
            payload = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates_res = data.get("candidates", [])
                    if candidates_res:
                        parts = candidates_res[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except Exception as exc:
                last_error = exc
                continue

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

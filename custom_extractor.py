"""
custom_extractor.py  —  Person B's Backend Module (Merged)
─────────────────────────────────────────────────────────
Custom Column Extraction Engine.

Takes raw unstructured text AND user-specified column names,
uses Pydantic to strictly map only those requested columns,
and returns a structured pandas DataFrame.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import create_model


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_custom_dataset(
    raw_text: str,
    columns: Optional[str] = None,
    api_key: str = "",
    provider: str = "gemini",
    model: str = "gemini-flash-lite-latest",
    user_columns_str: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract structured records using user-specified column names.

    Parameters
    ----------
    raw_text : str
        The unstructured text pasted by the user.
    columns : str, optional
        Comma-separated column names provided by the user.
    api_key : str
        The API key for the selected LLM provider.
    provider : str
        ``"gemini"`` or ``"openai"``.
    model : str
        The model identifier.
    user_columns_str : str, optional
        Alias for columns.
    """
    col_str = columns if columns is not None else user_columns_str
    if not col_str:
        raise ValueError("Please provide at least one valid column name.")

    col_list = [col.strip() for col in col_str.split(",") if col.strip()]
    if not col_list:
        raise ValueError("Please provide at least one valid column name.")

    # 1. Dynamically build a Pydantic model with those fields
    row_fields = {col: (Optional[str], None) for col in col_list}
    DynamicRowModel = create_model("DynamicRowModel", **row_fields)
    DatasetSchema = create_model("DatasetSchema", items=(List[DynamicRowModel], ...))

    # 2. Build prompt
    prompt = f"""Extract EXACTLY ONE record per unique entity/subject mentioned in the text matching these target fields: {', '.join(col_list)}.
Do NOT omit any distinct entity and do NOT create duplicate records for the same entity.
If a specific column value is missing for an entry in the text, return null.
Format your output as a JSON object with a single key "items" containing a list of the extracted records.

Raw Text:
{raw_text}"""

    # 3. Call LLM
    if provider == "openai":
        raw_json = _call_openai(prompt, api_key, model)
    else:
        raw_json = _call_gemini(prompt, api_key, model, DatasetSchema)

    # 4. Parse JSON & Return DataFrame
    records = _parse_llm_response(raw_json)
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if not df.empty and len(df.columns) > 0:
        df = df.drop_duplicates(subset=[df.columns[0]], keep="first").reset_index(drop=True)

    return df


# ─── LLM Callers ─────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str, model: str, schema: Any) -> str:
    """
    Call Google Gemini with multi-layered fallback strategy:
      1A. SDK call with JSON mode + schema
      1B. SDK call with JSON mode
      1C. SDK call without JSON mode (plain text)
      2. Direct REST HTTP call via urllib
    """
    from google import genai
    from google.genai import types as genai_types

    clean_key = api_key.strip()

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

    try:
        client = genai.Client(api_key=clean_key)
    except Exception:
        client = None

    if client is not None:
        for candidate in candidates:
            # 1A. Try with JSON mime type + schema
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as exc:
                last_error = exc

            # 1B. Try with JSON mime type only
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
                
            # 1C. Try without JSON mime type (plain text mode)
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

    # 2. Direct REST HTTP Call (guaranteed fallback)
    for candidate in candidates:
        for api_ver in ["v1beta", "v1"]:
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{candidate}:generateContent?key={clean_key}"
            payload_dict = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json"
                }
            }
            try:
                payload_dict["generationConfig"]["responseSchema"] = schema.model_json_schema()
            except Exception:
                pass
                
            payload = json.dumps(payload_dict).encode("utf-8")

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
                    "You are a precise data-extraction engine. "
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_llm_response(raw_json: str) -> List[Dict[str, Any]]:
    """Parse JSON output into list of records."""
    text = raw_json.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    if isinstance(parsed, list):
        return parsed

    if isinstance(parsed, dict):
        if "items" in parsed and isinstance(parsed["items"], list):
            return parsed["items"]
        if "records" in parsed and isinstance(parsed["records"], list):
            return parsed["records"]
        if "data" in parsed and isinstance(parsed["data"], list):
            return parsed["data"]
        if "results" in parsed and isinstance(parsed["results"], list):
            return parsed["results"]
        return [parsed]

    return []

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

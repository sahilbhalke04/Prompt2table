"""
enricher.py  —  Data Enrichment & Missing Value Imputation
──────────────────────────────────────────────────────────
Identifies rows with missing values ('—', null, None, '') and uses Gemini or OpenAI
to backfill missing cells IN-PLACE matched strictly by 4-tier entity index/name.
Guarantees:
  - Strictly preserves original row count and row order.
  - Multi-tiered model fallbacks (flash-lite, 3.5, 3.6, pro, REST).
  - Detailed reporting of filled cells (no silent failure).
  - 0 duplicate rows, 0 shifted rows, and 0 dropped entities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class _NpEncoder(json.JSONEncoder):
    """Handle numpy int64/float64 that standard json.dumps can't serialize."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _call_gemini_enricher(prompt: str, api_key: str, model: Optional[str] = None) -> str:
    """
    Call Google Gemini with multi-layered fallback strategy:
      1. SDK call with JSON mode
      2. SDK call without JSON mode (plain text)
      3. Direct REST HTTP call via urllib
    """
    from google import genai
    from google.genai import types as genai_types
    import urllib.request

    clean_key = api_key.strip()

    # Active working candidate models in priority order
    base_candidates = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-pro-latest",
    ]

    candidates = []
    if model and model.strip():
        m = model.strip()
        candidates.append(m)
        for c in base_candidates:
            if c != m:
                candidates.append(c)
    else:
        candidates = base_candidates

    last_error: Optional[Exception] = None

    # ── Attempt 1 & 2: SDK with and without response_mime_type ───────────────
    try:
        client = genai.Client(api_key=clean_key)
    except Exception as exc:
        client = None
        last_error = exc

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
                if response and response.text and response.text.strip():
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
                if response and response.text and response.text.strip():
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
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                    "maxOutputTokens": 8192,
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates_res = data.get("candidates", [])
                    if candidates_res:
                        parts = candidates_res[0].get("content", {}).get("parts", [])
                        if parts:
                            txt = parts[0].get("text", "").strip()
                            if txt:
                                return txt
            except Exception as exc:
                last_error = exc
                continue

    if last_error:
        err_msg = str(last_error)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            raise RuntimeError(
                "Gemini API quota exceeded (429). "
                "Your free tier limit for the requested model is reached. "
                "Switch to 'gemini-flash-lite-latest' or 'gemini-3.5-flash-lite' in the sidebar."
            ) from last_error
        raise RuntimeError(f"Gemini API call failed: {err_msg}") from last_error

    raise RuntimeError("No response received from Gemini for data auto-fill.")


def _call_openai_enricher(prompt: str, api_key: str, model: Optional[str] = None) -> str:
    """Call OpenAI GPT for data enrichment."""
    import openai

    clean_key = api_key.strip()
    chosen_model = model.strip() if model and model.strip() else "gpt-4o-mini"

    try:
        client = openai.OpenAI(api_key=clean_key)
        response = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert data enrichment assistant. Return valid JSON array of objects only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI enrichment failed: {str(exc)}") from exc

    raise RuntimeError("No response received from OpenAI.")


def fill_missing_values(
    df: pd.DataFrame,
    api_key: str,
    provider: str = "gemini",
    model: Optional[str] = None,
    return_count: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, int]]:
    """
    Identifies rows with missing values or '—' placeholders and asks 
    the LLM to infer and backfill missing cells in-place using entity context.
    
    Guarantees:
      - Strictly preserves original row count and row order.
      - Uses 4-tier entity matching (__row_index__, __entity_name__, value scan, position).
      - Never modifies the primary entity name or overwrites unrelated rows.
      - Never appends duplicate rows.
      - Returns tuple (df, filled_cells_count) if return_count=True.
    """
    if df is None or df.empty:
        return (df, 0) if return_count else df

    # 1. Identify which rows have missing / empty values
    missing_mask = (
        df.isin(["—", "-", "null", "None", "", "NaN", "nan"]).any(axis=1)
        | df.isnull().any(axis=1)
    )
    missing_indices = df[missing_mask].index.tolist()

    if not missing_indices:
        return (df, 0) if return_count else df  # Nothing to fill!

    primary_col = df.columns[0]

    # 2. Extract incomplete records with their primary entity name and explicit index
    incomplete_records = []
    for idx in missing_indices:
        row_dict = {
            "__row_index__": int(idx),
            "__entity_name__": str(df.at[idx, primary_col]),
        }
        for col in df.columns:
            val = df.at[idx, col]
            if val in ["—", "-", "null", "None", "", "NaN", "nan"] or pd.isna(val):
                row_dict[col] = None
            else:
                row_dict[col] = val
        incomplete_records.append(row_dict)

    # 3. Call LLM to fill missing fields
    prompt = f"""You are an expert data enrichment assistant.
Below is a JSON list of records where some attributes are null/missing.
Each record has an identifier '__row_index__' and a primary subject name '__entity_name__'.

TASK:
For each record, infer and fill in ONLY the missing (null) values based on real-world knowledge and row context.
Keep all non-null values EXACTLY as they are.
Preserve '__row_index__' and '__entity_name__'.
Return a JSON array of objects containing the completed records.

Incomplete Records:
{json.dumps(incomplete_records, indent=2, cls=_NpEncoder)}

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects containing the filled records."""

    clean_prov = provider.strip().lower() if provider else "gemini"
    if "openai" in clean_prov:
        response_text = _call_openai_enricher(prompt, api_key, model=model)
    else:
        response_text = _call_gemini_enricher(prompt, api_key, model=model)

    if not response_text:
        raise RuntimeError("AI model returned an empty response. Please try again.")

    # 4. Parse response JSON
    clean_json = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.MULTILINE)
    clean_json = re.sub(r"\s*```\s*$", "", clean_json, flags=re.MULTILINE).strip()

    filled_data = None
    try:
        filled_data = json.loads(clean_json)
    except Exception:
        # Fallback regex search for JSON array
        match = re.search(r"\[.*\]", clean_json, re.DOTALL)
        if match:
            try:
                filled_data = json.loads(match.group(0))
            except Exception:
                pass

    if isinstance(filled_data, dict):
        for v in filled_data.values():
            if isinstance(v, list):
                filled_data = v
                break
        if isinstance(filled_data, dict):
            filled_data = [filled_data]

    if not isinstance(filled_data, list):
        raise ValueError(
            f"Unable to parse AI auto-fill response into a list of records. "
            f"Snippet: {response_text[:200]}"
        )

    # Convert working copy to object dtype to prevent pandas type conflicts on assignment
    df_result = df.astype(object).copy()
    cells_filled = 0

    # 5. Robust 4-tier target matching to strictly update ONLY the target missing cells
    for i, record in enumerate(filled_data):
        if not isinstance(record, dict):
            continue

        target_idx = None

        # Tier 1: exact __row_index__ (cast to int)
        if "__row_index__" in record:
            try:
                row_idx = int(record["__row_index__"])
                if row_idx in missing_indices:
                    target_idx = row_idx
            except (ValueError, TypeError):
                pass

        # Tier 2: __entity_name__
        if target_idx is None and "__entity_name__" in record and record["__entity_name__"]:
            name_val = str(record["__entity_name__"]).strip().lower()
            matches = df_result[
                df_result[primary_col].astype(str).str.strip().str.lower() == name_val
            ].index
            if len(matches) > 0:
                target_idx = matches[0]

        # Tier 3: Scan all values in record to see if one matches an entity name in missing rows
        if target_idx is None:
            for val in record.values():
                if isinstance(val, str) and val.strip():
                    v_lower = val.strip().lower()
                    matches = df_result.loc[missing_indices][
                        df_result.loc[missing_indices, primary_col].astype(str).str.strip().str.lower() == v_lower
                    ].index
                    if len(matches) > 0:
                        target_idx = matches[0]
                        break

        # Tier 4: Positional fallback within missing_indices
        if target_idx is None and i < len(missing_indices):
            target_idx = missing_indices[i]

        # Apply updates to missing cells only
        if target_idx is not None:
            for col in df_result.columns:
                if col == primary_col or col.startswith("__"):
                    continue  # Never overwrite the entity name or internal keys!

                matched_key = next(
                    (k for k in record.keys() if str(k).strip().lower() == str(col).strip().lower()),
                    None,
                )
                if matched_key:
                    new_val = record[matched_key]
                    curr_val = df_result.at[target_idx, col]
                    if curr_val in ["—", "-", "null", "None", "", "NaN", "nan"] or pd.isna(curr_val):
                        if new_val is not None and str(new_val).strip() not in ["", "—", "-", "null", "None"]:
                            df_result.at[target_idx, col] = str(new_val)
                            cells_filled += 1

    # 6. Standardize leftover empty / NA cells with "—"
    df_result = df_result.fillna("—")
    for col in df_result.columns:
        df_result.loc[df_result[col].isin(["", "nan", "NaN", "None", "null"]), col] = "—"

    if return_count:
        return df_result, cells_filled
    return df_result

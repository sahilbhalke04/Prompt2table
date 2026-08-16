"""
enricher.py  —  Data Enrichment & Missing Value Imputation
──────────────────────────────────────────────────────────
Identifies rows with missing values ('—', null, None) and uses Gemini
to backfill missing cells IN-PLACE matched strictly by 4-tier entity index/name.
Guarantees 0 duplicate rows, 0 shifted rows, and 0 dropped entities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from google import genai
from google.genai import types


def fill_missing_values(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Identifies rows with missing values or '—' placeholders and asks 
    Gemini to infer and backfill missing cells in-place using entity context.
    
    Guarantees:
      - Strictly preserves original row count and row order.
      - Uses 4-tier entity matching (__row_index__, __entity_name__, value scan, position).
      - Never modifies the primary entity name or overwrites unrelated rows.
      - Never appends duplicate rows.
    """
    if df is None or df.empty:
        return df

    # Make a clean working copy
    df_result = df.copy()

    # 1. Identify which rows have missing / empty values
    missing_mask = (
        df_result.isin(["—", "-", "null", "None", "", "NaN", "nan"]).any(axis=1)
        | df_result.isnull().any(axis=1)
    )
    missing_indices = df_result[missing_mask].index.tolist()

    if not missing_indices:
        return df_result  # Nothing to fill!

    primary_col = df_result.columns[0]

    # 2. Extract incomplete records with their primary entity name and explicit index
    incomplete_records = []
    for idx in missing_indices:
        row_dict = {
            "__row_index__": int(idx),
            "__entity_name__": str(df_result.at[idx, primary_col]),
        }
        for col in df_result.columns:
            val = df_result.at[idx, col]
            if val in ["—", "-", "null", "None", "", "NaN", "nan"] or pd.isna(val):
                row_dict[col] = None
            else:
                row_dict[col] = val
        incomplete_records.append(row_dict)

    # 3. Call LLM to fill missing fields
    clean_key = api_key.strip()
    client = genai.Client(api_key=clean_key)

    prompt = f"""You are an expert data enrichment assistant.
Below is a JSON list of records where some attributes are null/missing.
Each record has an identifier '__row_index__' and a primary subject name '__entity_name__'.

TASK:
For each record, infer and fill in ONLY the missing (null) values based on real-world knowledge and row context.
Keep all non-null values EXACTLY as they are.
Preserve '__row_index__' and '__entity_name__'.
Return a JSON array of objects containing the completed records.

Incomplete Records:
{json.dumps(incomplete_records, indent=2)}

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects containing the filled records."""

    candidates = ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-pro-latest"]
    response_text = ""
    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            if response and response.text:
                response_text = response.text.strip()
                break
        except Exception:
            continue

    if not response_text:
        return df_result

    # 4. Parse response JSON
    try:
        clean_json = re.sub(r"^```(?:json)?\s*", "", response_text, flags=re.MULTILINE)
        clean_json = re.sub(r"\s*```\s*$", "", clean_json, flags=re.MULTILINE)
        filled_data = json.loads(clean_json)

        if isinstance(filled_data, dict):
            for v in filled_data.values():
                if isinstance(v, list):
                    filled_data = v
                    break
            if isinstance(filled_data, dict):
                filled_data = [filled_data]
    except Exception:
        return df_result

    if not isinstance(filled_data, list):
        return df_result

    # 5. Robust 4-tier target matching to strictly update ONLY the target missing cells
    for i, record in enumerate(filled_data):
        if not isinstance(record, dict):
            continue

        target_idx = None

        # Tier 1: exact __row_index__ (cast to int — Gemini may return "2" instead of 2)
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
                        if new_val is not None and str(new_val).strip() != "":
                            df_result.at[target_idx, col] = str(new_val)

    # 6. Replace any leftover NA with "—" and return exactly df_result
    return df_result.fillna("—")
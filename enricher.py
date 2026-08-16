"""
enricher.py  —  Data Enrichment & Missing Value Imputation
──────────────────────────────────────────────────────────
Identifies rows with missing values ('—', null, None) and uses Gemini
to backfill missing cells IN-PLACE matching by explicit row ID and
primary entity name. Guarantees 0 duplicated rows and 0 dropped entities.
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
    Gemini to infer and backfill missing cells in-place using row context.
    
    Guarantees:
      - The exact same number of rows (no duplicates created)
      - All original entities (Samsung, iPhone, etc.) are strictly preserved
      - Only the missing (null/—) cells are filled
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

    # 2. Extract incomplete records, tagging each with its explicit row index
    primary_col = df_result.columns[0]
    incomplete_records = []
    for idx in missing_indices:
        row_dict = {"__row_id__": int(idx)}
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
Below is a JSON list of {len(incomplete_records)} database records where some attributes are null/missing.
Each record has an identifier '__row_id__' and a primary subject name under '{primary_col}'.

CRITICAL INSTRUCTIONS:
1. For each record, infer and fill in ONLY the missing (null) values based on real-world knowledge and row context.
2. Keep all non-null values EXACTLY as they are.
3. You MUST preserve the exact '__row_id__' for each record.
4. Do NOT drop any records and do NOT duplicate any records. Return exactly {len(incomplete_records)} records.

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

    # 5. Robust In-place update by __row_id__ and primary entity name
    for record in filled_data:
        if not isinstance(record, dict):
            continue

        target_idx = None
        # Match by explicit __row_id__
        if "__row_id__" in record and record["__row_id__"] in df_result.index:
            target_idx = record["__row_id__"]
        elif primary_col in record and record[primary_col]:
            # Fallback: match by primary entity name
            name_val = str(record[primary_col]).strip().lower()
            matches = df_result[df_result[primary_col].astype(str).str.strip().str.lower() == name_val].index
            if len(matches) > 0:
                target_idx = matches[0]

        if target_idx is not None:
            for col in df_result.columns:
                if col == "__row_id__":
                    continue
                matched_key = next(
                    (k for k in record.keys() if str(k).strip().lower() == str(col).strip().lower()),
                    None,
                )
                if matched_key:
                    new_val = record[matched_key]
                    curr_val = df_result.at[target_idx, col]
                    # Only update if the cell was originally missing
                    if curr_val in ["—", "-", "null", "None", "", "NaN", "nan"] or pd.isna(curr_val):
                        if new_val is not None:
                            df_result.at[target_idx, col] = str(new_val)

    # 6. Ensure no NaN remain and return
    return df_result.fillna("—")
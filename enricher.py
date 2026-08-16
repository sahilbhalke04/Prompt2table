import json
import pandas as pd
from google import genai
from google.genai import types

def fill_missing_values(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Identifies rows with missing values or '—' placeholders and asks 
    Gemini to infer and backfill missing cells using row context.
    """
    # 1. Replace '—' placeholders with None for uniform detection
    df_working = df.replace("—", None)
    
    # 2. Locate rows with missing values
    missing_rows_mask = df_working.isnull().any(axis=1)
    if not missing_rows_mask.any():
        return df  # Nothing to fill!

    # 3. Extract incomplete rows into JSON
    incomplete_records = df_working[missing_rows_mask].to_dict(orient="records")

    # 4. Call Gemini to infer missing fields
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert data enrichment assistant.
    Below is a list of database records where some attributes are null/missing.
    
    TASK: Infer and fill in ALL missing (null) values based on the context provided by each record's existing populated fields.
    
    Incomplete Records:
    {json.dumps(incomplete_records, indent=2)}
    
    Return ONLY a valid JSON array of objects with the exact same structure where all null values are replaced with accurate inferred values.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
    except Exception:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

    # 5. Update original dataset with enriched values
    filled_data = json.loads(response.text)
    filled_df = pd.DataFrame(filled_data)

    df_working.update(filled_df)
    
    return df_working.fillna("—")
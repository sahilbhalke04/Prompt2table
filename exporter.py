"""
exporter.py  —  Export Engine
────────────────────────────────────────────────
Data Export & Cleaning Utilities.

Cleans missing values (None/null), converts the extracted JSON
into a pandas DataFrame, and formats it into downloadable
CSV bytes or JSON strings.
"""

from __future__ import annotations

import pandas as pd


def json_to_dataframe(raw_data: list[dict], expected_columns: list[str] = None) -> pd.DataFrame:
    """
    Convert a list of raw JSON dictionaries to a pandas DataFrame.
    
    Ensures that the output DataFrame contains exactly the expected columns,
    padding with missing values where necessary, and replacing any remaining
    missing values with the "—" placeholder.

    Parameters
    ----------
    raw_data : list[dict]
        A list of dictionaries representing the extracted records.
    expected_columns : list[str], optional
        A list of column names that the resulting DataFrame must have.
        Missing columns will be added with None values.

    Returns
    -------
    pd.DataFrame
        The constructed DataFrame with missing values filled.
    """
    if not raw_data:
        return pd.DataFrame(columns=expected_columns if expected_columns else [])
    df = pd.DataFrame(raw_data)
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = None
        df = df[expected_columns]
    return df.fillna("—")


def convert_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes suitable for st.download_button.

    Parameters
    ----------
    df : pd.DataFrame
        The extracted dataset.

    Returns
    -------
    bytes
        UTF-8 encoded CSV content.
    """
    return df.to_csv(index=False).encode('utf-8')


def convert_to_json(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a JSON string suitable for st.download_button.

    Parameters
    ----------
    df : pd.DataFrame
        The extracted dataset.

    Returns
    -------
    str
        Pretty-printed JSON string (array of objects).
    """
    return df.to_json(orient="records", indent=2)

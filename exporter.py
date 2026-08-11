"""
exporter.py  —  Person B's Export Engine (STUB)
────────────────────────────────────────────────
Data Export & Cleaning Utilities.

Cleans missing values (None/null), converts the extracted JSON
into a pandas DataFrame, and formats it into downloadable
CSV bytes or JSON strings.

Interface Contract
──────────────────
Person B must implement the functions below with these EXACT signatures
so that app.py can call them seamlessly.

Public API
----------
    convert_to_csv(df)  -> bytes
    convert_to_json(df) -> str
"""

from __future__ import annotations

import pandas as pd


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

    Notes
    -----
    - Should handle None/NaN values gracefully (replace with empty string
      or keep as-is depending on preference).
    - Should NOT include the DataFrame index in the CSV output.
    """
    # ──────────────────────────────────────────────────────────────────────
    # TODO: Person B — implement this function.
    #
    # Suggested approach:
    #   return df.to_csv(index=False).encode("utf-8")
    # ──────────────────────────────────────────────────────────────────────
    raise NotImplementedError(
        "exporter.convert_to_csv() is not yet implemented. "
        "Person B: please implement this function!"
    )


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

    Notes
    -----
    - Should use ``orient="records"`` for clean JSON output.
    - Should pretty-print with indent=2 for readability.
    """
    # ──────────────────────────────────────────────────────────────────────
    # TODO: Person B — implement this function.
    #
    # Suggested approach:
    #   return df.to_json(orient="records", indent=2)
    # ──────────────────────────────────────────────────────────────────────
    raise NotImplementedError(
        "exporter.convert_to_json() is not yet implemented. "
        "Person B: please implement this function!"
    )

"""
custom_extractor.py  —  Person B's Backend Module (STUB)
─────────────────────────────────────────────────────────
Custom Column Extraction Engine.

Takes raw unstructured text AND user-specified column names,
uses Pydantic to strictly map only those requested columns,
and returns a structured pandas DataFrame.

Interface Contract
──────────────────
Person B must implement the function below with this EXACT signature
so that app.py can call it seamlessly.

Public API
----------
    extract_custom_dataset(raw_text, columns, api_key, provider, model) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd


def extract_custom_dataset(
    raw_text: str,
    columns: str,
    api_key: str,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
) -> pd.DataFrame:
    """
    Extract structured records using user-specified column names.

    Parameters
    ----------
    raw_text : str
        The unstructured text pasted by the user.
    columns : str
        Comma-separated column names provided by the user.
        Example: ``"Name, Price, Top Speed"``
    api_key : str
        The API key for the selected LLM provider.
    provider : str
        ``"gemini"`` or ``"openai"``.
    model : str
        The model identifier (e.g. ``"gemini-2.0-flash"``, ``"gpt-4o-mini"``).

    Returns
    -------
    pd.DataFrame
        A DataFrame with EXACTLY the columns the user specified,
        populated with extracted data.  Missing values should be ``None``.
    """
    # ──────────────────────────────────────────────────────────────────────
    # TODO: Person B — implement this function.
    #
    # Suggested approach:
    #   1. Parse the `columns` string into a list: ["Name", "Price", "Top Speed"]
    #   2. Dynamically build a Pydantic model with those fields.
    #   3. Send the raw_text + column list to the LLM asking it to extract
    #      records that match ONLY those columns.
    #   4. Parse the JSON response and validate with Pydantic.
    #   5. Return as pd.DataFrame.
    # ──────────────────────────────────────────────────────────────────────
    raise NotImplementedError(
        "custom_extractor.extract_custom_dataset() is not yet implemented. "
        "Person B: please implement this function!"
    )

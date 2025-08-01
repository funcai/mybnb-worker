"""

The tool filters a pandas DataFrame of apartment listings by:
  • exact number of rooms (`rooms`)
  • maximum monthly rent (`rent_monthly`)

Apartment metadata such as `rooms` and `rent_monthly` is expected to live in the
`facts` column of the DataFrame as a Python `dict`.

Usage:
    python agent_filter.py <apartments.jsonl>
Then start chatting – type natural-language queries like:
    "Find me 3-room apartments under 1800 euros"
Type `exit` or `quit` to leave.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_apartment_df(path: str) -> pd.DataFrame:
    """Read an apartments JSON-Lines file into a DataFrame."""
    df = pd.read_json(path)
    city_names = []
    for address in df["address"]:
        city_names.append(address["city"].lower())
    df["city_name"] = city_names
    return df


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def filter_apartments(
    df: pd.DataFrame,
    rooms: int = None,
    rent_monthly: float = None,
    area_m2: int = None,
    beds: int = None,
    deposit: float = None,
    currency: str = None,
    furnished: bool = None,
    availableFrom: str = None,
    city_name: str = None,
) -> pd.DataFrame:
    """Return rows that match all non-None constraints."""
    mask = pd.Series(True, index=df.index)

    # Filter by location first, as it's a top-level column
    if city_name is not None:
        mask &= df["city_name"] == city_name.lower()

    # Filter by facts dictionary
    def check_facts(d):
        if not isinstance(d, dict):
            return False
        if rooms is not None and d.get("rooms") != rooms:
            return False
        if rent_monthly is not None and d.get("rent_monthly", float("inf")) > rent_monthly:
            return False
        if area_m2 is not None and d.get("area_m2", 0) < area_m2:
            return False
        if beds is not None and d.get("beds") != beds:
            return False
        if deposit is not None and d.get("deposit", float("inf")) > deposit:
            return False
        if currency is not None and d.get("currency") != currency:
            return False
        if furnished is not None and d.get("furnished") != furnished:
            return False
        if availableFrom is not None and d.get("availableFrom") != availableFrom:
            return False
        return True

    mask &= df["facts"].apply(check_facts)
    return df[mask]

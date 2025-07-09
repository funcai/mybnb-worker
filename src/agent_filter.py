"""
Apartment Filtering Agent using Ollama (gemma3n:e4b)

This script spins up a simple CLI agent backed by Ollama that knows one tool:
`filter_apartments`.

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
# import sys
from typing import Any, Dict

import pandas as pd
# from langchain.agents import Tool, initialize_agent
# from langchain_community.llms import Ollama

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


# ---------------------------------------------------------------------------
# Tool wrapper that LangChain can call
# ---------------------------------------------------------------------------

# def filter_apartments_tool(query: str) -> str:
#     """LangChain-compatible wrapper for `filter_apartments`.

#     Expects *query* to be a JSON string with any of the following keys.
#     Any missing keys will be ignored.
#         {
#           "location": str | None,
#           "area_m2": int | None,
#           "rooms": int | None,
#           "beds": int | None,
#           "rent_monthly": float | None,
#           "deposit": float | None,
#           "currency": str | None,
#           "furnished": bool | None,
#           "availableFrom": str | None
#         }
#     Returns matching apartments as JSON-Lines text, or a friendly
#     message if none are found.
#     """
#     # Accept either a dict (already parsed) or a JSON string. Be forgiving of
#     # extra text the agent may append (e.g. "query" tokens).
#     if isinstance(query, dict):
#         params = query
#     else:
#         # Try direct JSON parse first
#         try:
#             params: Dict[str, Any] = json.loads(query)
#         except json.JSONDecodeError:
#             # Fallback: extract the first JSON object substring if the string
#             # contains extra tokens (e.g. "{...}query").
#             try:
#                 start = query.index("{")
#                 end = query.rindex("}") + 1
#                 params = json.loads(query[start:end])
#             except Exception as exc:
#                 return (
#                     "Invalid input for filter_apartments. Provide JSON with keys "
#                     "'rooms'(int) and 'rent_monthly'(float). Error: " + str(exc)
#                 )

#     # Validate and coerce types
#     try:
#         rooms = int(params["rooms"])
#         rent = float(params["rent_monthly"])
#     except (ValueError, KeyError, TypeError) as exc:
#         return (
#             "Invalid parameters. Expected keys 'rooms'(int) and 'rent_monthly'(float). "
#             "Error: " + str(exc)
#         )

#     result = filter_apartments(rooms, rent, apartment_df)
#     if result.empty:
#         return "No apartments matched the criteria."

#     # Return as JSONL so callers can easily parse it further.
#     return result.to_json(orient="records", lines=True)


# ---------------------------------------------------------------------------
# CLI agent
# ---------------------------------------------------------------------------

# def main() -> None:
#     if len(sys.argv) != 2:
#         print("Usage: python agent_filter.py <apartments.jsonl>")
#         sys.exit(1)

#     data_path = sys.argv[1]
#     global apartment_df  # so the tool wrapper can see it
#     apartment_df = load_apartment_df(data_path)

#     # Use a very low temperature for deterministic responses
#     llm = Ollama(model="gemma3n:e4b", temperature=0.1)

#     tools = [
#         Tool(
#             name="filter_apartments",
#             func=filter_apartments_tool,
#             description=(
#                 "Filter apartments by exact number of rooms and maximum monthly rent. "
#                 "Input must be a JSON string with keys 'rooms' (int) and 'rent_monthly' (float)."
#             ),
#         )
#     ]

#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent="zero-shot-react-description",
#         verbose=True,
#     )

#     print("\nChat with the apartment agent. Type 'exit' to quit.\n")
#     while True:
#         try:
#             user_query = input(">> ")
#         except (EOFError, KeyboardInterrupt):
#             break

#         if user_query.strip().lower() in {"exit", "quit"}:
#             break

#         response = agent.run(user_query)
#         print(response)


# if __name__ == "__main__":
#     main()

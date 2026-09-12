"""
Static Gulf Coast reference datasets (ports, lighthouses, companies).
Loaded from local CSV — no network.
"""

from pathlib import Path

import pandas as pd

_REF_DIR = Path(__file__).resolve().parent


def load_ports() -> pd.DataFrame:
    return pd.read_csv(_REF_DIR / "ports.csv")


def load_lighthouses() -> pd.DataFrame:
    return pd.read_csv(_REF_DIR / "lighthouses.csv")


def load_companies() -> pd.DataFrame:
    return pd.read_csv(_REF_DIR / "companies.csv")


def load_debris_reference() -> dict:
    import json
    with open(_REF_DIR / "real_marine_debris.json", encoding="utf-8") as f:
        return json.load(f)

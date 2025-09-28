"""Persistence utilities for writing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import utils


def write_json(obj: Any, path: Path) -> None:
    """Write a JSON object to disk."""
    utils.ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV."""
    utils.ensure_directory(path.parent)
    df.to_csv(path, index=False)

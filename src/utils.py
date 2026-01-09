from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
import pandas as pd


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def select_date_range(df, start: Optional[str], end: Optional[str]):
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df


def calc_drawdown(nav_series) -> Iterable[float]:
    peak = None
    dds = []
    last_dd = 0.0
    for v in nav_series:
        if v is None or (hasattr(v, "__float__") and pd.isna(v)):
            dds.append(last_dd)
            continue
        peak = v if peak is None else max(peak, v)
        if peak == 0:
            last_dd = 0.0
        else:
            last_dd = float(v / peak - 1.0)
        dds.append(last_dd)
    return dds

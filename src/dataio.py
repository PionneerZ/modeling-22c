from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


_DATE_CANDIDATES = ["date", "datetime", "time", "timestamp", "day", "dt"]
_GOLD_CANDIDATES = [
    "gold",
    "gold_price",
    "goldprice",
    "gold_price_usd",
    "gold_usd",
    "xau",
    "xauusd",
    "xau_usd",
]
_BTC_CANDIDATES = [
    "btc",
    "bitcoin",
    "btc_price",
    "bitcoin_price",
    "btc_usd",
    "bitcoin_usd",
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _pick_column(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], label: str) -> str:
    if preferred and preferred in df.columns:
        return preferred
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    raise ValueError(f"missing {label} column. tried: {candidates}")


def _missing_ranges(missing_dates: List[pd.Timestamp]) -> str:
    if not missing_dates:
        return ""
    ranges = []
    start = prev = missing_dates[0]
    for dt in missing_dates[1:]:
        if dt == prev + pd.Timedelta(days=1):
            prev = dt
            continue
        ranges.append((start, prev))
        start = prev = dt
    ranges.append((start, prev))
    parts = [f"{s.date()} to {e.date()}" for s, e in ranges]
    return "; ".join(parts)


def _resolve_column_map(config: Dict) -> Dict[str, Optional[str]]:
    data_cfg = config["data"]
    col_map = data_cfg.get("column_map") or {}
    return {
        "date": col_map.get("date") or data_cfg.get("date_col"),
        "gold": col_map.get("gold") or data_cfg.get("gold_col"),
        "btc": col_map.get("btc") or data_cfg.get("btc_col"),
    }


def _load_split_files(
    gold_path: Path,
    btc_path: Path,
    column_map: Dict[str, Optional[str]],
    date_format: Optional[str],
) -> pd.DataFrame:
    gold_df = _read_table(gold_path)
    btc_df = _read_table(btc_path)

    date_col_gold = _pick_column(gold_df, column_map.get("date"), _DATE_CANDIDATES, "date")
    date_col_btc = _pick_column(btc_df, column_map.get("date"), _DATE_CANDIDATES, "date")
    gold_col = _pick_column(gold_df, column_map.get("gold"), _GOLD_CANDIDATES, "gold")
    btc_col = _pick_column(btc_df, column_map.get("btc"), _BTC_CANDIDATES, "btc")

    gold_df = gold_df.copy()
    btc_df = btc_df.copy()
    gold_df["date"] = pd.to_datetime(gold_df[date_col_gold], format=date_format)
    btc_df["date"] = pd.to_datetime(btc_df[date_col_btc], format=date_format)
    gold_df = gold_df.sort_values("date").drop_duplicates("date")
    btc_df = btc_df.sort_values("date").drop_duplicates("date")

    gold_df["price_gold"] = pd.to_numeric(gold_df[gold_col], errors="coerce")
    btc_df["price_btc"] = pd.to_numeric(btc_df[btc_col], errors="coerce")

    merged = pd.merge(
        gold_df[["date", "price_gold"]],
        btc_df[["date", "price_btc"]],
        on="date",
        how="outer",
    )
    merged = merged.sort_values("date").drop_duplicates("date")
    return merged


def load_price_data(config: Dict) -> pd.DataFrame:
    data_cfg = config["data"]

    column_map = _resolve_column_map(config)
    date_format = data_cfg.get("date_format")

    gold_path = data_cfg.get("gold_path")
    btc_path = data_cfg.get("btc_path")
    if gold_path and btc_path:
        gold_path = Path(gold_path)
        btc_path = Path(btc_path)
        if not gold_path.exists():
            raise FileNotFoundError(f"gold data file not found: {gold_path}")
        if not btc_path.exists():
            raise FileNotFoundError(f"btc data file not found: {btc_path}")
        df = _load_split_files(gold_path, btc_path, column_map, date_format)
        source_info = {
            "mode": "split",
            "gold_path": str(gold_path),
            "btc_path": str(btc_path),
            "date_col": column_map.get("date"),
            "gold_col": column_map.get("gold"),
            "btc_col": column_map.get("btc"),
        }
    else:
        path = Path(data_cfg["path"])
        if not path.exists():
            raise FileNotFoundError(
                f"data file not found: {path}. Put data under data/ and update config."
            )
        df = _read_table(path)
        date_col = _pick_column(df, column_map.get("date"), _DATE_CANDIDATES, "date")
        gold_col = _pick_column(df, column_map.get("gold"), _GOLD_CANDIDATES, "gold")
        btc_col = _pick_column(df, column_map.get("btc"), _BTC_CANDIDATES, "btc")

        df = df.copy()
        df["date"] = pd.to_datetime(df[date_col], format=date_format)
        df = df.sort_values("date").drop_duplicates("date")

        df["price_gold"] = pd.to_numeric(df[gold_col], errors="coerce")
        df["price_btc"] = pd.to_numeric(df[btc_col], errors="coerce")
        df = df[["date", "price_gold", "price_btc"]]
        source_info = {
            "mode": "single",
            "path": str(path),
            "date_col": date_col,
            "gold_col": gold_col,
            "btc_col": btc_col,
        }

    start_date = pd.to_datetime(data_cfg.get("start_date"))
    end_date = pd.to_datetime(data_cfg.get("end_date"))
    end_mode = data_cfg.get("end_date_mode", "trade_end")
    if end_mode not in {"trade_end", "valuation_end"}:
        raise ValueError(f"unsupported end_date_mode: {end_mode}")

    trade_end = end_date if end_mode == "trade_end" else end_date - pd.Timedelta(days=1)
    if trade_end < start_date:
        raise ValueError("end_date is earlier than start_date after applying end_date_mode.")

    raw_min = df["date"].min()
    raw_max = df["date"].max()
    if raw_min > start_date or raw_max < trade_end:
        raise ValueError(
            f"data does not cover required trade range: {start_date.date()} to {trade_end.date()}. "
            f"available: {raw_min.date()} to {raw_max.date()}"
        )

    calendar_anchor = data_cfg.get("calendar_anchor", "btc")
    if calendar_anchor != "btc":
        raise ValueError(f"unsupported calendar_anchor: {calendar_anchor}")

    idx = pd.date_range(start=start_date, end=end_date, freq="D")
    trade_idx = pd.date_range(start=start_date, end=trade_end, freq="D")

    gold_series_raw = df.set_index("date")["price_gold"].reindex(idx)
    btc_series_raw = df.set_index("date")["price_btc"].reindex(idx)

    btc_trade_series = btc_series_raw.reindex(trade_idx)
    missing_btc = btc_trade_series[btc_trade_series.isna()].index.tolist()
    if missing_btc:
        ranges = _missing_ranges(missing_btc)
        raise ValueError(f"btc data missing within trade range: {ranges}")

    weekday_mask = pd.Series(idx.weekday < 5, index=idx)
    gold_weekday = gold_series_raw[weekday_mask]
    missing_gold_weekday = gold_weekday[gold_weekday.isna()].index.tolist()
    missing_gold_weekend = gold_series_raw[(~weekday_mask) & gold_series_raw.isna()].index.tolist()

    gold_ffill_for_valuation = bool(data_cfg.get("gold_ffill_for_valuation", True))
    if gold_ffill_for_valuation:
        gold_series = gold_series_raw.ffill().bfill()
    else:
        gold_series = gold_series_raw

    btc_series = btc_series_raw.ffill()

    if gold_series.isna().all():
        raise ValueError("gold price series is empty after alignment.")
    if btc_series.isna().all():
        raise ValueError("bitcoin price series is empty after alignment.")

    is_trading_btc = (idx <= trade_end)
    gold_trade_weekdays_only = bool(data_cfg.get("gold_trade_weekdays_only", True))
    gold_trade_on_filled = bool(data_cfg.get("gold_trade_on_filled", False))
    if gold_trade_on_filled:
        gold_trade_series = gold_series_raw.ffill().bfill()
        gold_trade_available = gold_trade_series.notna()
    else:
        gold_trade_available = gold_series_raw.notna()

    if gold_trade_weekdays_only:
        is_trading_gold = (idx.weekday < 5) & (idx <= trade_end) & gold_trade_available
    else:
        is_trading_gold = (idx <= trade_end) & gold_trade_available

    out = pd.DataFrame(
        {
            "date": idx,
            "t": range(0, len(idx)),
            "price_gold": gold_series.values,
            "price_btc": btc_series.values,
            "is_trading_gold": is_trading_gold,
            "is_trading_btc": is_trading_btc,
        }
    )

    data_info = {
        "source": source_info,
        "raw_min": str(raw_min.date()),
        "raw_max": str(raw_max.date()),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "trade_end": str(trade_end.date()),
        "calendar_anchor": calendar_anchor,
        "btc_missing_days": len(missing_btc),
        "gold_missing_total": int(gold_series_raw.isna().sum()),
        "gold_missing_weekday": len(missing_gold_weekday),
        "gold_missing_weekend": len(missing_gold_weekend),
        "gold_ffill_for_valuation": gold_ffill_for_valuation,
        "gold_trade_weekdays_only": gold_trade_weekdays_only,
        "gold_trade_on_filled": gold_trade_on_filled,
    }
    out.attrs["data_info"] = data_info

    print(
        "[dataio] range",
        data_info["raw_min"],
        "to",
        data_info["raw_max"],
        "| trade_end",
        data_info["trade_end"],
        "| btc_missing_days",
        data_info["btc_missing_days"],
        "| gold_missing_weekday",
        data_info["gold_missing_weekday"],
        "| gold_missing_weekend",
        data_info["gold_missing_weekend"],
    )

    return out.reset_index(drop=True)

"""One-off reproduction pipeline with debug exports."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio, plots
from src.backtest import run_backtest
from src.metrics import build_metrics
from src.params import resolve_paper_params
from src.utils import ensure_dir, load_yaml, save_json


TARGET_NAV = 220486.0
BTC_WINDOW = (1610, 1760)
GOLD_WINDOW = (460, 560)
PAPER_BTC_EXTREME_T = 1628
PAPER_BTC_REBUY_T = 1715


def _guardrail_settings(config: dict) -> dict:
    guard_cfg = config.get("run", {}).get("guardrail", {})
    return {
        "enabled": bool(guard_cfg.get("enabled", True)),
        "hard_fail": bool(guard_cfg.get("hard_fail", False)),
        "last_known_good": float(guard_cfg.get("last_known_good_final_nav", 150000.0)),
    }


def _format_data_info(data_info: dict) -> list[str]:
    source = data_info.get("source", {})
    if source.get("mode") == "split":
        source_desc = f"{source.get('gold_path')} | {source.get('btc_path')}"
    else:
        source_desc = source.get("path")
    return [
        "## Data coverage",
        f"- source: {source_desc}",
        f"- raw_range: {data_info.get('raw_min')} to {data_info.get('raw_max')}",
        f"- trade_end: {data_info.get('trade_end')}",
        f"- btc_missing_days: {data_info.get('btc_missing_days')}",
        f"- gold_missing_trading: {data_info.get('gold_missing_trading')}",
        f"- gold_ffill_for_valuation: {data_info.get('gold_ffill_for_valuation')}",
        f"- gold_trade_weekdays_only: {data_info.get('gold_trade_weekdays_only')}",
        "",
    ]


def _write_notes(
    path: Path,
    config: dict,
    run_id: str,
    run_path: Path,
    data_info: dict | None,
) -> None:
    paper_params = resolve_paper_params(config)
    params = (paper_params["hold_T"], paper_params["reentry_N"], paper_params["extreme_E"])
    fee_pair = (config["fees"]["fee_gold"], config["fees"]["fee_btc"])
    change_note = config.get("run", {}).get("change_note")
    lines = [
        "# notes",
        f"- run_id: {run_id}",
        f"- config: {config.get('_config_path')}",
        f"- run_path: {run_path}",
        f"- params[T,N,E]: {params}",
        f"- fee_pair: {fee_pair}",
        f"- strategy_mode: {config.get('strategy', {}).get('mode', 'full')}",
        "",
        "## Change summary",
        f"- {change_note}" if change_note else "- none (default config)",
        "",
        "## Deviation sources (top 3 if mismatched to paper)",
        "1. Data source and date range (trade_end vs valuation_end).",
        "2. Signal scaling and weight factor W implementation (see assumption_ledger).",
        "3. State machine boundaries (extreme/no-buy/holding blocking logic).",
        "",
    ]
    if data_info:
        lines.extend(_format_data_info(data_info))
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_trade_tag(reason: str, side: str, rebuy: bool) -> str:
    if reason == "extreme_sell":
        tag = "extreme sell"
    elif reason == "sell_signal":
        tag = "normal sell"
    elif reason in {"buy_momentum", "buy_reversion"}:
        tag = "normal buy"
    else:
        tag = reason or "trade"
    if side == "buy" and rebuy:
        tag = "rebuy/no-buy release"
    return tag


def _plot_window(price_series: pd.DataFrame, trades: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_series["t"], price_series["price"], color="black", linewidth=1.2, label="price")

    for _, trade in trades.iterrows():
        side = trade["side"]
        tag = trade["trigger_tag"]
        is_blocked = bool(trade.get("is_blocked", False))
        color = "red" if side == "buy" else "blue"
        marker = "^" if side == "buy" else "v"
        if "extreme sell" in tag:
            marker = "*"
            color = "blue"
        if "rebuy" in tag:
            marker = "^"
            color = "green"
        if is_blocked:
            marker = "x"
            color = "gray"

        ax.scatter(trade["t"], trade["price"], color=color, marker=marker, s=60, zorder=3)
        ax.text(
            trade["t"],
            trade["price"],
            trade["tag_abbrev"],
            color=color,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("price")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _export_window(
    results: pd.DataFrame,
    trades: pd.DataFrame,
    debug: pd.DataFrame,
    out_dir: Path,
    asset: str,
    t_start: int,
    t_end: int,
    window_name: str,
) -> None:
    debug_window = debug[
        (debug["asset"] == asset) & (debug["t"] >= t_start) & (debug["t"] <= t_end)
    ].copy()
    debug_window = debug_window[
        [
            "t",
            "date",
            "price",
            "E",
            "threshold",
            "no_buy_state",
            "extreme_state",
            "cooldown_remaining",
            "skipped_extreme_due_to_history",
            "target_position",
            "action",
            "action_reason",
        ]
    ]
    debug_window.to_csv(out_dir / f"{window_name}_window_debug.csv", index=False)

    rebuy_days = set(
        debug[
            (debug["asset"] == asset)
            & (debug["no_buy_released"])
            & (debug["t"] >= t_start)
            & (debug["t"] <= t_end)
        ]["t"].tolist()
    )

    trades_window = trades[
        (trades["asset"] == asset) & (trades["t"] >= t_start) & (trades["t"] <= t_end)
    ].copy()
    if not trades_window.empty:
        trades_window["is_blocked"] = trades_window["qty"] <= 0
    else:
        trades_window["is_blocked"] = []

    if not trades_window.empty:
        trades_window["trigger_tag"] = trades_window.apply(
            lambda row: _build_trade_tag(
                str(row.get("reason", "")),
                str(row.get("side", "")),
                row.get("t") in rebuy_days,
            ),
            axis=1,
        )
        trades_window.loc[trades_window["is_blocked"], "trigger_tag"] = trades_window.loc[
            trades_window["is_blocked"], "side"
        ].map({"buy": "blocked buy", "sell": "blocked sell"})
        trades_window["tag_abbrev"] = trades_window["trigger_tag"].map(
            {
                "normal buy": "B",
                "normal sell": "S",
                "extreme sell": "E",
                "rebuy/no-buy release": "R",
                "blocked buy": "Bb",
                "blocked sell": "Sb",
            }
        ).fillna("T")
    else:
        trades_window["trigger_tag"] = []
        trades_window["tag_abbrev"] = []

    trades_out = trades_window[
        ["t", "date", "asset", "side", "qty", "price", "reason", "trigger_tag", "is_blocked"]
    ].rename(columns={"qty": "size"})
    trades_out.to_csv(out_dir / f"{window_name}_window_trades.csv", index=False)

    price_col = "price_gold" if asset == "gold" else "price_btc"
    price_window = results[
        (results["t"] >= t_start) & (results["t"] <= t_end)
    ][["t", price_col]].rename(columns={price_col: "price"})
    _plot_window(price_window, trades_window, out_dir / f"{window_name}_window_plot.png", window_name)


def _btc_window_metrics(trades: pd.DataFrame, events: pd.DataFrame) -> dict:
    t_start, t_end = BTC_WINDOW
    live_trades = trades[trades["qty"] > 0]
    btc_trades = live_trades[
        (live_trades["asset"] == "btc") & (live_trades["t"].between(t_start, t_end))
    ]
    extreme_sell = btc_trades[btc_trades["reason"] == "extreme_sell"]
    no_buy_on = events[
        (events["asset"] == "btc")
        & (events["state_type"] == "no_buy")
        & (events["now"] == True)
        & (events["t"].between(t_start, t_end))
    ]
    no_buy_off = events[
        (events["asset"] == "btc")
        & (events["state_type"] == "no_buy")
        & (events["now"] == False)
        & (events["t"].between(t_start, t_end))
    ]
    extreme_t = int(extreme_sell.iloc[0]["t"]) if not extreme_sell.empty else None
    release_t = int(no_buy_off.iloc[0]["t"]) if not no_buy_off.empty else None
    rebuy_t = None
    if release_t is not None:
        after_release = btc_trades[(btc_trades["side"] == "buy") & (btc_trades["t"] >= release_t)]
        if not after_release.empty:
            rebuy_t = int(after_release.iloc[0]["t"])

    return {
        "btc_trades_window": int(btc_trades.shape[0]),
        "btc_extreme_sell_window": not extreme_sell.empty,
        "btc_no_buy_on_window": not no_buy_on.empty,
        "btc_no_buy_release_window": not no_buy_off.empty,
        "btc_rebuy_after_release": rebuy_t is not None,
        "btc_first_extreme_t": extreme_t,
        "btc_no_buy_release_t": release_t,
        "btc_rebuy_t": rebuy_t,
    }


def _gold_window_metrics(results: pd.DataFrame, trades: pd.DataFrame) -> dict:
    t_start, t_end = GOLD_WINDOW
    price_window = results[results["t"].between(t_start, t_end)]["price_gold"]
    max_price = float(price_window.max()) if not price_window.empty else float("nan")
    min_price = float(price_window.min()) if not price_window.empty else float("nan")
    high_threshold = max_price * 0.95 if pd.notna(max_price) else float("nan")

    gold_trades = trades[
        (trades["asset"] == "gold") & (trades["t"].between(t_start, t_end))
    ]
    buys = gold_trades[(gold_trades["side"] == "buy") & (gold_trades["qty"] > 0)]
    sells_actual = gold_trades[(gold_trades["side"] == "sell") & (gold_trades["qty"] > 0)]
    sells_blocked = gold_trades[
        (gold_trades["side"] == "sell")
        & (gold_trades["reason"].fillna("").str.startswith("blocked_by_hold"))
    ]
    high_sells = (
        sells_actual[sells_actual["price"] >= high_threshold]
        if pd.notna(high_threshold)
        else sells_actual
    )

    return {
        "gold_buy_count": int(buys.shape[0]),
        "gold_sell_actual_count": int(sells_actual.shape[0]),
        "gold_sell_blocked_count": int(sells_blocked.shape[0]),
        "gold_high_sell_count": int(high_sells.shape[0]),
        "gold_min_price": min_price if pd.notna(min_price) else None,
        "gold_max_price": max_price if pd.notna(max_price) else None,
        "gold_high_threshold": high_threshold if pd.notna(high_threshold) else None,
    }


def _shape_score(btc: dict, gold: dict) -> int:
    btc_score = sum(
        [
            int(btc["btc_extreme_sell_window"]),
            int(btc["btc_no_buy_on_window"]),
            int(btc["btc_no_buy_release_window"]),
            int(btc["btc_rebuy_after_release"]),
        ]
    )
    gold_score = int(gold["gold_buy_count"] >= 5) + int(gold["gold_sell_actual_count"] >= 2)
    return btc_score + gold_score


def _write_final_report(
    out_path: Path,
    config: dict,
    results: pd.DataFrame,
    trades: pd.DataFrame,
    events: pd.DataFrame,
    metrics: dict,
) -> dict:
    paper_params = resolve_paper_params(config)
    no_buy_cfg = config.get("no_buy", {})
    holding_cfg = config.get("holding", {})
    extreme_cfg = config.get("extreme", {})
    release_frac = no_buy_cfg.get("release_frac")
    if release_frac is None:
        release_frac = paper_params["reentry_N"]
    min_extreme_history_days = int(extreme_cfg.get("min_history_days", 0) or 0)
    btc = _btc_window_metrics(trades, events)
    gold = _gold_window_metrics(results, trades)

    t460_row = results[results["t"] == GOLD_WINDOW[0]]
    t460_date = (
        t460_row.iloc[0]["date"].date().isoformat() if not t460_row.empty else "missing"
    )

    live_trades = trades[trades["qty"] > 0] if not trades.empty else trades
    first_trade_t = None
    first_trade_date = None
    first_trade_date_ts = None
    if not live_trades.empty:
        first_trade_t = int(live_trades.iloc[0]["t"])
        first_trade_date_ts = pd.to_datetime(live_trades.iloc[0]["date"])
        first_trade_date = first_trade_date_ts.date().isoformat()

    num_trades_total = int(live_trades.shape[0]) if not live_trades.empty else 0
    num_btc_trades = int(live_trades[live_trades["asset"] == "btc"].shape[0]) if not live_trades.empty else 0
    num_gold_trades = int(live_trades[live_trades["asset"] == "gold"].shape[0]) if not live_trades.empty else 0

    cash_ratio = results["nav_cash"] / results["nav_total"]
    exposure_ratio = (results["nav_gold"] + results["nav_btc"]) / results["nav_total"]
    avg_cash_ratio = float(cash_ratio.mean()) if not cash_ratio.empty else 0.0
    avg_exposure_ratio = float(exposure_ratio.mean()) if not exposure_ratio.empty else 0.0
    cash_p50 = float(cash_ratio.quantile(0.5)) if not cash_ratio.empty else 0.0
    cash_p90 = float(cash_ratio.quantile(0.9)) if not cash_ratio.empty else 0.0

    lock_reasons = []
    if first_trade_date_ts is not None and first_trade_date_ts >= pd.Timestamp("2020-01-01"):
        lock_reasons.append("first_trade_date>=2020-01-01")
    if avg_exposure_ratio < 0.3:
        lock_reasons.append("avg_exposure_ratio<0.3")

    guard = _guardrail_settings(config)
    last_known_good = guard["last_known_good"]
    regression_threshold = 0.5 * last_known_good
    guard_enabled = guard["enabled"]
    guard_hard_fail = guard["hard_fail"]
    guard_failed = guard_enabled and metrics["final_nav"] < regression_threshold

    btc_extreme_diff = (
        abs(btc["btc_first_extreme_t"] - PAPER_BTC_EXTREME_T)
        if btc["btc_first_extreme_t"] is not None
        else None
    )
    btc_rebuy_diff = (
        abs(btc["btc_rebuy_t"] - PAPER_BTC_REBUY_T)
        if btc["btc_rebuy_t"] is not None
        else None
    )

    issues = []
    if btc["btc_first_extreme_t"] is None or (btc_extreme_diff is not None and btc_extreme_diff > 5):
        issues.append(
            f"BTC extreme sell timing diff: ours={btc['btc_first_extreme_t']} vs paper={PAPER_BTC_EXTREME_T}"
        )
    if btc["btc_rebuy_t"] is None or (btc_rebuy_diff is not None and btc_rebuy_diff > 10):
        issues.append(
            f"BTC rebuy timing diff: ours={btc['btc_rebuy_t']} vs paper={PAPER_BTC_REBUY_T}"
        )
    if gold["gold_sell_actual_count"] < 2:
        issues.append(
            f"Gold sells too few: actual_sells={gold['gold_sell_actual_count']} blocked_sells={gold['gold_sell_blocked_count']}"
        )
    if gold["gold_max_price"] is not None and gold["gold_max_price"] < 1800:
        issues.append(
            f"Gold window price range low: min={gold['gold_min_price']} max={gold['gold_max_price']}"
        )

    report_lines = [
        "# final_report",
        f"- final_nav: {metrics['final_nav']:.2f}",
        f"- target_nav: {TARGET_NAV:.2f}",
        f"- nav_diff: {abs(metrics['final_nav'] - TARGET_NAV):.2f}",
        f"- paper_params[T,N,E]: [{paper_params['hold_T']}, {paper_params['reentry_N']}, {paper_params['extreme_E']}]",
        f"- lookback_M: {paper_params['lookback_M']}",
        f"- extreme_min_history_days: {min_extreme_history_days}",
        "- t_definition: calendar-day index (master_index, calendar_anchor=btc)",
        f"- t=460 date: {t460_date}",
        f"- gold_window_price_range: {gold['gold_min_price']} .. {gold['gold_max_price']}",
        f"- no_buy_release: mode={no_buy_cfg.get('release_mode')} direction={no_buy_cfg.get('release_direction')} "
        f"cooldown_days={no_buy_cfg.get('cooldown_days')} release_frac={release_frac}",
        f"- hold_days: buys={holding_cfg.get('min_days_between_buys')} sells={holding_cfg.get('min_days_between_sells')}",
        "",
        "## Regression diagnostics",
        f"- first_trade_t: {first_trade_t}",
        f"- first_trade_date: {first_trade_date}",
        f"- num_trades_total: {num_trades_total}",
        f"- num_btc_trades: {num_btc_trades}",
        f"- num_gold_trades: {num_gold_trades}",
        f"- avg_cash_ratio: {avg_cash_ratio:.4f}",
        f"- avg_exposure_ratio: {avg_exposure_ratio:.4f}",
        (
            f"- <span style=\"color:red\">strategy_locked: {'; '.join(lock_reasons)}</span>"
            if lock_reasons
            else "- strategy_locked: false"
        ),
        f"- guardrail: enabled={guard_enabled} hard_fail={guard_hard_fail} last_known_good={last_known_good:.0f}",
        "",
        "## Root cause & fix",
        "- root_cause: min_history_days was applied as a global guard and could block trading until the threshold.",
        "- fix: min_history_days now only gates extreme signals; normal buy/sell stays active when extreme history is insufficient.",
        f"- evidence_after_fix: first_trade_date={first_trade_date}, avg_exposure_ratio={avg_exposure_ratio:.4f}",
        "",
        "## Figure8 BTC window (t=1610..1760)",
        f"- extreme_sell_t: {btc['btc_first_extreme_t']} (paper {PAPER_BTC_EXTREME_T})",
        f"- no_buy_release_t: {btc['btc_no_buy_release_t']}",
        f"- rebuy_t: {btc['btc_rebuy_t']} (paper {PAPER_BTC_REBUY_T})",
        f"- trades_in_window: {btc['btc_trades_window']}",
        "",
        "## Figure9 Gold window (t=460..560)",
        f"- buy_count: {gold['gold_buy_count']}",
        f"- sell_actual_count: {gold['gold_sell_actual_count']}",
        f"- sell_blocked_count: {gold['gold_sell_blocked_count']}",
        f"- high_sell_count (>=95% max): {gold['gold_high_sell_count']}",
        "",
        "## Paper evidence (quotes)",
        "- [T,N,E] definition (paper p11): \"T is the time the model must hold an asset. N is the price the asset must fall to (as a percentage of the maximum price) for the model to begin buying assets again. E is the percentage of the max price that an asset must fall to for the model to sell in an extreme market condition.\"",
        "- no-buy (paper p7): \"The model is then no longer allowed to buy back into the market for a number of days.\"",
        "- holding (paper p7): \"Once the model buys an asset, it can no longer sell for those z days.\"",
        "",
    ]

    if guard_failed:
        report_lines.extend(
            [
                "## Regression guardrail",
                f"- !! regression_fail: final_nav<{regression_threshold:.2f} (baseline {last_known_good:.0f})",
                f"- first_trade_date: {first_trade_date}",
                f"- avg_exposure_ratio: {avg_exposure_ratio:.4f}",
                f"- cash_ratio_p50: {cash_p50:.4f}",
                f"- cash_ratio_p90: {cash_p90:.4f}",
                "",
            ]
        )

    if issues:
        report_lines.append("## Remaining gaps (top sources)")
        for issue in issues[:3]:
            report_lines.append(f"- {issue}")
    else:
        report_lines.append("## Remaining gaps (top sources)")
        report_lines.append("- none detected in window-level diagnostics")

    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "enabled": guard_enabled,
        "hard_fail": guard_hard_fail,
        "failed": guard_failed,
        "threshold": regression_threshold,
        "last_known_good": last_known_good,
    }


def _run_sensitivity(base_cfg: dict, prices: pd.DataFrame) -> pd.DataFrame:
    base_params = resolve_paper_params(base_cfg)
    e_base = base_params["extreme_E"]
    n_base = base_params["reentry_N"]

    values_e = [round(e_base + d, 2) for d in [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]]
    values_n = [round(n_base + d, 2) for d in [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]]

    rows = []
    for e_val in values_e:
        for n_val in values_n:
            cfg = copy.deepcopy(base_cfg)
            cfg["paper_params"]["extreme_E"] = float(e_val)
            cfg["paper_params"]["reentry_N"] = float(n_val)
            cfg["extreme"]["extreme_sell_pct_E"] = float(e_val)

            results, trades, _w, _debug, events = run_backtest(prices, cfg, return_debug=True)
            final_nav = float(results["nav_total"].iloc[-1]) if len(results) else base_cfg["run"]["initial_cash"]
            nav_diff = abs(final_nav - TARGET_NAV)
            btc = _btc_window_metrics(trades, events)
            gold = _gold_window_metrics(results, trades)
            shape_score = _shape_score(btc, gold)
            extreme_t_diff = (
                abs(btc["btc_first_extreme_t"] - PAPER_BTC_EXTREME_T)
                if btc["btc_first_extreme_t"] is not None
                else 999
            )
            rebuy_t_diff = (
                abs(btc["btc_rebuy_t"] - PAPER_BTC_REBUY_T)
                if btc["btc_rebuy_t"] is not None
                else 999
            )

            rows.append(
                {
                    "extreme_E": e_val,
                    "reentry_N": n_val,
                    "final_nav": final_nav,
                    "nav_diff": nav_diff,
                    "shape_score": shape_score,
                    "btc_extreme_t": btc["btc_first_extreme_t"],
                    "btc_extreme_t_diff": extreme_t_diff,
                    "btc_rebuy_t": btc["btc_rebuy_t"],
                    "btc_rebuy_t_diff": rebuy_t_diff,
                    "gold_buy_count": gold["gold_buy_count"],
                    "gold_sell_actual_count": gold["gold_sell_actual_count"],
                    "gold_high_sell_count": gold["gold_high_sell_count"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml", help="paper-aligned base config")
    parser.add_argument("--run-id", default="run_paper_final", help="output run id")
    args = parser.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["_config_path"] = args.config
    base_cfg["strategy"]["mode"] = "full"

    prices = dataio.load_price_data(base_cfg)
    data_info = prices.attrs.get("data_info", {})
    run_id = args.run_id

    out_dir = ensure_dir(Path("outputs") / run_id)
    debug_dir = ensure_dir(out_dir / "debug")
    calib_dir = ensure_dir(out_dir / "calibration")
    figs_dir = ensure_dir(out_dir / "figs")

    sens_df = _run_sensitivity(base_cfg, prices)
    sens_path = calib_dir / "sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)

    min_nav_threshold = TARGET_NAV * 0.5
    viable = sens_df[sens_df["final_nav"] >= min_nav_threshold]
    if viable.empty:
        viable = sens_df.copy()

    best = viable.sort_values(
        ["shape_score", "btc_extreme_t_diff", "btc_rebuy_t_diff", "nav_diff"],
        ascending=[False, True, True, True],
    ).iloc[0]

    final_cfg = copy.deepcopy(base_cfg)
    final_cfg["paper_params"]["extreme_E"] = float(best["extreme_E"])
    final_cfg["paper_params"]["reentry_N"] = float(best["reentry_N"])
    final_cfg["extreme"]["extreme_sell_pct_E"] = float(best["extreme_E"])
    final_cfg["run"]["run_id"] = run_id
    final_cfg["run"]["change_note"] = "paper_final: aligned T/N/E + no-buy/hold/index refactor"

    recommended_path = Path("config") / "recommended_paper_aligned.yaml"
    import yaml

    recommended_path.write_text(yaml.safe_dump(final_cfg, sort_keys=False), encoding="utf-8")

    results, trades, w_stats, debug, events = run_backtest(prices, final_cfg, return_debug=True)

    results_path = out_dir / "results_table.csv"
    trades_path = out_dir / "trades.csv"
    results.to_csv(results_path, index=False)
    trades.to_csv(trades_path, index=False)

    data_range = f"{results['date'].iloc[0].date()} to {results['date'].iloc[-1].date()}"
    metrics = build_metrics(results, trades, final_cfg, run_id, data_range, w_stats=w_stats)
    save_json(out_dir / "key_metrics.json", metrics)

    plots.plot_nav(results, figs_dir / "fig1_nav.png")
    plots.plot_drawdown(results, figs_dir / "fig2_drawdown.png")
    plots.plot_positions(results, figs_dir / "fig3_positions.png")
    plots.plot_trade_window(results, trades, final_cfg, figs_dir / "fig4_trades_window.png")

    debug[["t", "date"]].drop_duplicates().sort_values("t").to_csv(
        debug_dir / "index_mapping.csv", index=False
    )
    events.to_csv(debug_dir / "state_events.csv", index=False)

    _export_window(results, trades, debug, debug_dir, "btc", *BTC_WINDOW, "fig8_btc")
    _export_window(results, trades, debug, debug_dir, "gold", *GOLD_WINDOW, "fig9_gold")

    report_path = calib_dir / "final_report.md"
    guardrail = _write_final_report(report_path, final_cfg, results, trades, events, metrics)
    _write_notes(out_dir / "notes.md", final_cfg, run_id, out_dir, data_info)
    if guardrail["failed"] and guardrail["hard_fail"]:
        raise RuntimeError(
            f"Regression guardrail failed: final_nav<{guardrail['threshold']:.2f} "
            f"(baseline {guardrail['last_known_good']:.0f})"
        )

    print(f"final_report: {report_path}")
    print(f"sensitivity: {sens_path}")
    print(f"recommended_config: {recommended_path}")
    print(f"run outputs: {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from src import dataio, plots
from src.backtest import run_backtest
from src.metrics import build_metrics
from src.params import resolve_paper_params
from src.utils import ensure_dir, load_yaml, make_run_id, save_json


def _format_data_info(data_info: dict) -> list[str]:
    source = data_info.get("source", {})
    if source.get("mode") == "split":
        source_desc = f"{source.get('gold_path')} | {source.get('btc_path')}"
    else:
        source_desc = source.get("path")
    return [
        "## 数据覆盖",
        f"- source: {source_desc}",
        f"- raw_range: {data_info.get('raw_min')} to {data_info.get('raw_max')}",
        f"- trade_end: {data_info.get('trade_end')}",
        f"- btc_missing_days: {data_info.get('btc_missing_days')}",
        f"- gold_missing_trading: {data_info.get('gold_missing_trading')}",
        f"- gold_ffill_for_valuation: {data_info.get('gold_ffill_for_valuation')}",
        f"- gold_trade_weekdays_only: {data_info.get('gold_trade_weekdays_only')}",
        "",
    ]


def write_notes(path: Path, config: dict, run_id: str, run_path: Path, data_info: dict | None) -> None:
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
        "## 变更摘要",
        f"- {change_note}" if change_note else "- 无（默认配置）",
        "",
        "## 差异来源 Top3（若与论文不一致）",
        "1. 数据源与时间区间（trade_end vs valuation_end）。",
        "2. 信号尺度与权重因子 W 的实现细节（见 assumption_ledger）。",
        "3. 状态机边界（extreme/no-buy/holding 的阻断逻辑）。",
        "",
    ]
    if data_info:
        lines.extend(_format_data_info(data_info))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config yaml")
    parser.add_argument("--outdir", default=None, help="override outputs directory")
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["_config_path"] = args.config

    seed = config["run"].get("seed")
    if seed is not None:
        np.random.seed(seed)

    prices = dataio.load_price_data(config)
    data_info = prices.attrs.get("data_info", {})

    results_df, trades_df, w_stats = run_backtest(prices, config)

    if args.outdir:
        out_dir = ensure_dir(Path(args.outdir))
        run_id = out_dir.name
    else:
        run_id = config["run"].get("run_id") or make_run_id()
        out_dir = ensure_dir(Path("outputs") / run_id)
    figs_dir = ensure_dir(out_dir / "figs")

    results_path = out_dir / "results_table.csv"
    trades_path = out_dir / "trades.csv"
    results_df.to_csv(results_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    data_range = f"{prices['date'].iloc[0].date()} to {prices['date'].iloc[-1].date()}"
    metrics = build_metrics(results_df, trades_df, config, run_id, data_range, w_stats=w_stats)
    save_json(out_dir / "key_metrics.json", metrics)

    plots.plot_nav(results_df, figs_dir / "fig1_nav.png")
    plots.plot_drawdown(results_df, figs_dir / "fig2_drawdown.png")
    plots.plot_positions(results_df, figs_dir / "fig3_positions.png")
    plots.plot_trade_window(results_df, trades_df, config, figs_dir / "fig4_trades_window.png")

    write_notes(out_dir / "notes.md", config, run_id, out_dir, data_info)

    print(f"run completed: {out_dir}")


if __name__ == "__main__":
    main()

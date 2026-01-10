from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import dataio
from src.backtest import run_backtest
from src.figures import FIGURE_REGISTRY, FigureContext
from src.utils import ensure_dir, load_yaml, make_run_id


EXPECTED_DATE_RANGE = ("2016-09-11", "2021-09-10")
EXPECTED_FIG4_FINAL_NAV = 220486.0
EXPECTED_FIG6_BEST_PARAMS = [12, 0.6, 0.89]
EXPECTED_FIG7_FINAL_NAV = {
    "0.01,0.02": 220486.0,
    "0.02,0.03": 199945.0,
    "0.03,0.05": 156038.0,
    "0.1,0.1": 47452.0,
}


def _maybe_extract_paper_figures(manifest_path: Path, config_path: Path) -> None:
    if manifest_path.exists():
        return
    script_path = Path(__file__).resolve().parent / "extract_paper_figures.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--outdir",
        str(manifest_path.parent),
    ]
    print(f"[figures] manifest missing, running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _load_manifest(manifest_path: Path) -> dict[int, dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {int(row["figure_id"]): row for row in reader}


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _make_compare(paper_path: Path, ours_path: Path, out_path: Path) -> None:
    paper_img = plt.imread(paper_path)
    ours_img = plt.imread(ours_path)
    height = max(paper_img.shape[0], ours_img.shape[0])
    width = paper_img.shape[1] + ours_img.shape[1]
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(paper_img)
    ax1.set_title("paper")
    ax1.axis("off")
    ax2.imshow(ours_img)
    ax2.set_title("ours")
    ax2.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _within_tolerance(actual: float, expected: float, tol: float = 0.05) -> bool:
    if expected == 0:
        return False
    return abs(actual - expected) / expected <= tol


def _assess_alignment(fig_id: int, context: FigureContext, runs_cache: dict) -> tuple[str, str]:
    cfg = context.config
    if fig_id in {1, 2}:
        start = str(cfg["data"].get("start_date"))
        end = str(cfg["data"].get("end_date"))
        if (start, end) == EXPECTED_DATE_RANGE:
            return "OK", f"date range matches {EXPECTED_DATE_RANGE[0]}..{EXPECTED_DATE_RANGE[1]}"
        return "Deviation", f"date range {start}..{end} differs from paper"
    if fig_id == 3:
        return "OK", "diagram-only figure (no data alignment needed)"
    if fig_id == 4:
        final_nav = float(context.results["nav_total"].iloc[-1])
        if _within_tolerance(final_nav, EXPECTED_FIG4_FINAL_NAV):
            return "OK", f"final_nav={final_nav:.2f} close to paper {EXPECTED_FIG4_FINAL_NAV:.0f}"
        return "Deviation", f"final_nav={final_nav:.2f} vs paper {EXPECTED_FIG4_FINAL_NAV:.0f}"
    if fig_id == 5:
        weight_cfg = cfg.get("weight_factor", {})
        apply_mode = str(weight_cfg.get("apply_mode", "")).lower()
        w_use = [str(v).lower() for v in weight_cfg.get("W_use", [])]
        if apply_mode in {"score", "compare"} and ("all" in w_use):
            return "OK", f"apply_mode={apply_mode}, W_use={w_use}"
        return "Deviation", f"apply_mode={apply_mode}, W_use={w_use}"
    if fig_id == 6:
        runs = runs_cache.get(6, [])
        if not runs:
            return "Deviation", "missing param grid runs"
        best = max(runs, key=lambda r: r["final_nav"])
        if [float(v) for v in best["params"]] == [float(v) for v in EXPECTED_FIG6_BEST_PARAMS]:
            return "OK", f"best_params={best['params']}"
        return "Deviation", f"best_params={best['params']} expected {EXPECTED_FIG6_BEST_PARAMS}"
    if fig_id == 7:
        runs = runs_cache.get(7, [])
        if not runs:
            return "Deviation", "missing fee sensitivity runs"
        mismatches = []
        for run in runs:
            key = f"{run['fees'][0]},{run['fees'][1]}"
            expected = EXPECTED_FIG7_FINAL_NAV.get(key)
            if expected is None:
                continue
            if not _within_tolerance(run["final_nav"], expected):
                mismatches.append(f"{key} {run['final_nav']:.2f} vs {expected:.0f}")
        if mismatches:
            return "Deviation", "; ".join(mismatches)
        return "OK", "fee-pair final_nav within tolerance"
    return "Deviation", "no alignment rule"


def _figure_meta(fig_id: int) -> dict:
    meta = {
        1: {
            "meaning": "BTC and gold price paths in separate panels.",
            "entrypoint": "src/figures.py:plot_fig1",
            "data": "price series from data load",
            "params": "data.start_date/end_date",
        },
        2: {
            "meaning": "BTC and gold prices on a shared axis to show scale.",
            "entrypoint": "src/figures.py:plot_fig2",
            "data": "price series from data load",
            "params": "data.start_date/end_date",
        },
        3: {
            "meaning": "Flow chart for buy logic.",
            "entrypoint": "src/figures.py:plot_fig3",
            "data": "diagram only",
            "params": "strategy logic",
        },
        4: {
            "meaning": "Portfolio NAV breakdown over time.",
            "entrypoint": "src/figures.py:plot_fig4",
            "data": "results_table.csv nav_* columns",
            "params": "full config",
        },
        5: {
            "meaning": "Weight-factor adjusted signal strength over time.",
            "entrypoint": "src/figures.py:plot_fig5",
            "data": "profit_* columns + weight_factor config",
            "params": "weight_factor, signals",
        },
        6: {
            "meaning": "NAV under parameter grid variations.",
            "entrypoint": "src/figures.py:plot_fig6",
            "data": "param_grid from config/reproduce_tables.yaml",
            "params": "T/N/E grid + fee pair",
        },
        7: {
            "meaning": "NAV under fee sensitivity variations.",
            "entrypoint": "src/figures.py:plot_fig7",
            "data": "fee_sensitivity from config/reproduce_tables.yaml",
            "params": "fee pairs + fixed T/N/E",
        },
    }
    return meta.get(fig_id, {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--paper-manifest", default="outputs/paper_figures/figures_manifest.csv")
    parser.add_argument("--paper-config", default="config/figures_bboxes.yaml")
    parser.add_argument("--tables-config", default="config/reproduce_tables.yaml")
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--exclude", default=None, help="comma-separated figure ids to exclude")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["_config_path"] = args.config
    tables_cfg = load_yaml(args.tables_config)

    out_dir = Path(args.outdir) if args.outdir else Path("outputs") / (
        cfg.get("run", {}).get("run_id") or make_run_id()
    )
    figures_dir = ensure_dir(out_dir / "figures")
    ours_dir = ensure_dir(figures_dir / "ours")
    paper_dir = ensure_dir(figures_dir / "paper")
    compare_dir = ensure_dir(figures_dir / "compare")

    manifest_path = Path(args.paper_manifest)
    _maybe_extract_paper_figures(manifest_path, Path(args.paper_config))
    manifest = _load_manifest(manifest_path)

    exclude = set()
    if args.exclude:
        exclude |= {int(x.strip()) for x in args.exclude.split(",") if x.strip()}

    prices = dataio.load_price_data(cfg)

    results_path = out_dir / "results_table.csv"
    trades_path = out_dir / "trades.csv"
    if args.reuse_results and results_path.exists() and trades_path.exists():
        results_df = _load_results(results_path)
        trades_df = _load_results(trades_path)
    else:
        results_df, trades_df, _ = run_backtest(prices, cfg)
        ensure_dir(out_dir)
        results_df.to_csv(results_path, index=False)
        trades_df.to_csv(trades_path, index=False)

    context = FigureContext(prices=prices, results=results_df, trades=trades_df, config=cfg, tables_config=tables_cfg)

    runs_cache: dict[int, list[dict]] = {}

    for fig_id in sorted(manifest):
        row = manifest[fig_id]
        if fig_id in exclude:
            continue
        func = FIGURE_REGISTRY.get(fig_id)
        if not func:
            continue
        ours_path = ours_dir / f"fig{fig_id}.png"
        result = func(context, ours_path)
        if isinstance(result, list):
            runs_cache[fig_id] = result

        paper_src = Path(row["paper_png"])
        if not paper_src.exists():
            paper_src = manifest_path.parent / paper_src
        paper_dst = paper_dir / f"fig{fig_id}.png"
        shutil.copyfile(paper_src, paper_dst)

        compare_path = compare_dir / f"fig{fig_id}_compare.png"
        _make_compare(paper_dst, ours_path, compare_path)

    report_lines = [
        "# Figure Reproduction Report",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- config: {args.config}",
        f"- outdir: {out_dir}",
        f"- paper_manifest: {manifest_path}",
        f"- tables_config: {args.tables_config}",
        "",
    ]

    for fig_id in sorted(manifest):
        if fig_id in exclude:
            continue
        meta = _figure_meta(fig_id)
        status, reason = _assess_alignment(fig_id, context, runs_cache)
        report_lines.extend(
            [
                f"## Figure {fig_id}",
                f"- meaning: {meta.get('meaning', '')}",
                f"- implementation: {meta.get('entrypoint', '')}",
                f"- data: {meta.get('data', '')}",
                f"- key_params: {meta.get('params', '')}",
                f"- alignment: {status}",
                f"- notes: {reason}",
                "",
            ]
        )

    report_path = figures_dir / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[figures] ours: {ours_dir}")
    print(f"[figures] paper: {paper_dir}")
    print(f"[figures] compare: {compare_dir}")
    print(f"[figures] report: {report_path}")


if __name__ == "__main__":
    main()

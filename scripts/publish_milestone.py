from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.params import resolve_paper_params
from src.utils import ensure_dir, load_yaml


def _read_config_from_notes(notes_path: Path) -> str | None:
    if not notes_path.exists():
        return None
    for line in notes_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- config:"):
            return line.split(":", 1)[1].strip()
    return None


def _ensure_section(cfg: dict, key: str, default: dict) -> None:
    if key not in cfg or cfg[key] is None:
        cfg[key] = default.copy()


def _set_default(d: dict, key: str, value) -> None:
    if key not in d or d[key] is None:
        d[key] = value


def _expand_effective_config(cfg: dict) -> dict:
    expanded = json.loads(json.dumps(cfg))
    paper = resolve_paper_params(expanded)

    _ensure_section(expanded, "signals", {})
    _ensure_section(expanded, "buy_logic", {})
    _ensure_section(expanded, "weight_factor", {})
    _ensure_section(expanded, "no_buy", {})
    _ensure_section(expanded, "holding", {})
    _ensure_section(expanded, "buy_sizing", {})
    _ensure_section(expanded, "selling", {})
    _ensure_section(expanded, "execution", {})
    _ensure_section(expanded, "extreme", {})

    signals = expanded["signals"]
    _set_default(signals, "reversion_mode", "below_ma")
    _set_default(signals, "score_mode", "max")
    _set_default(signals, "score_threshold", 0.0)
    _set_default(signals, "ma_include_current", True)
    _set_default(signals, "require_full_window", False)
    if signals.get("reversion_mode") == "below_ma_pct" and signals.get("reversion_pct") is None:
        signals["reversion_pct"] = float(paper["reentry_N"])

    buy_logic = expanded["buy_logic"]
    _set_default(buy_logic, "mode", "single")
    _set_default(buy_logic, "rebalance_on_buy", False)
    _set_default(buy_logic, "rebalance_ignore_hold", True)
    if buy_logic.get("rebalance_threshold") is None:
        buy_logic["rebalance_threshold"] = float(signals.get("score_threshold", 0.0) or 0.0)

    weight = expanded["weight_factor"]
    _set_default(weight, "apply_mode", "compare")
    _set_default(weight, "W_use", [])
    _set_default(weight, "use_t_plus_one", True)

    no_buy = expanded["no_buy"]
    _set_default(no_buy, "scope", "asset")
    _set_default(no_buy, "rebuy_on_release", False)
    _set_default(no_buy, "max_price_ref", "extreme_window")
    _set_default(no_buy, "release_mode", "hybrid")
    _set_default(no_buy, "release_direction", "drop")
    _set_default(no_buy, "cooldown_days", 0)
    if no_buy.get("release_frac") is None:
        no_buy["release_frac"] = float(paper["reentry_N"])

    holding = expanded["holding"]
    _set_default(holding, "min_days_between_buys", 0)
    if holding.get("min_days_between_sells") is None:
        holding["min_days_between_sells"] = int(paper["hold_T"])
    _set_default(holding, "sell_hold_ref", "entry")

    buy_sizing = expanded["buy_sizing"]
    _set_default(buy_sizing, "mode", "score")
    _set_default(buy_sizing, "fixed_fraction", 0.5)
    _set_default(buy_sizing, "buy_scale", 10.0)
    _set_default(buy_sizing, "buy_cap", 1.0)
    _set_default(buy_sizing, "buy_min_cash_reserve", 0.0)

    selling = expanded["selling"]
    _set_default(selling, "avg_cost_method", "weighted_avg")
    _set_default(selling, "skip_if_buy_signal", True)

    execution = expanded["execution"]
    _set_default(execution, "sell_then_buy", True)

    extreme = expanded["extreme"]
    _set_default(extreme, "min_history_days", 0)
    _set_default(extreme, "avg_mode", "positive_history")
    if not extreme.get("avg_window_days"):
        extreme["avg_window_days"] = int(paper["lookback_M"])

    expanded["paper_params"] = {
        "hold_T": int(paper["hold_T"]),
        "reentry_N": float(paper["reentry_N"]),
        "extreme_E": float(paper["extreme_E"]),
        "lookback_M": int(paper["lookback_M"]),
    }

    return expanded


def _write_effective_config(config_path: Path, out_path: Path) -> None:
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(config_path)
    effective = _expand_effective_config(cfg)
    out_path.write_text(
        yaml.safe_dump(effective, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _write_trades_head(src: Path, dst: Path, head: int) -> None:
    with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for i, row in enumerate(reader):
            writer.writerow(row)
            if i >= head:
                break


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _collect_figures(run_dir: Path, patterns: list[str]) -> list[Path]:
    found = []
    for pat in patterns:
        for path in run_dir.glob(pat):
            if path.is_file():
                found.append(path)
    seen = []
    for path in found:
        if path not in seen:
            seen.append(path)
    return seen


def _copy_figures(run_dir: Path, milestone_dir: Path, patterns: list[str]) -> list[str]:
    figures_dir = ensure_dir(milestone_dir / "figures")
    copied = []
    for src in _collect_figures(run_dir, patterns):
        rel = src.relative_to(run_dir)
        if rel.parts[0] in ("figures", "figs"):
            rel = Path(*rel.parts[1:])
        if rel.parts:
            dst = figures_dir / rel
        else:
            dst = figures_dir / src.name
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(milestone_dir)).replace("\\", "/"))
    return copied


def _update_index(index_path: Path, name: str, summary_rel: str) -> None:
    lines = []
    if index_path.exists():
        lines = index_path.read_text(encoding="utf-8").splitlines()
    if "## Index" not in lines:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("## Index")
    entry = f"- {name}: {summary_rel}"
    if entry not in lines:
        lines.append(entry)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="path to outputs/run_xxx")
    parser.add_argument("--name", required=True, help="milestone name, e.g. M01_fixpack_220k")
    parser.add_argument("--milestones-dir", default="outputs/_milestones")
    parser.add_argument("--config", default=None, help="override config path (yaml)")
    parser.add_argument("--command", default=None, help="override repro command")
    parser.add_argument("--goal", default=None, help="goal/acceptance summary")
    parser.add_argument("--note", default=None, help="extra note for summary")
    parser.add_argument("--trades-head", type=int, default=200)
    parser.add_argument(
        "--figs",
        default=None,
        help="comma-separated glob patterns relative to run dir",
    )
    parser.add_argument("--pip-freeze", action="store_true", help="include pip freeze in env.txt")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    milestones_dir = Path(args.milestones_dir)
    out_dir = milestones_dir / args.name
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"milestone already exists: {out_dir}")

    ensure_dir(out_dir)
    copied_files = []

    def copy_if_exists(src: Path, dst_name: str) -> None:
        if src.exists():
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            copied_files.append(dst_name)

    copy_if_exists(run_dir / "key_metrics.json", "key_metrics.json")
    copy_if_exists(run_dir / "results_table.csv", "results_table.csv")

    trades_src = run_dir / "trades.csv"
    if trades_src.exists():
        trades_dst = out_dir / "trades_head.csv"
        _write_trades_head(trades_src, trades_dst, args.trades_head)
        copied_files.append("trades_head.csv")

    state_candidates = [
        run_dir / "debug" / "state_events.csv",
        run_dir / "state_events.csv",
    ]
    for src in state_candidates:
        if src.exists():
            shutil.copy2(src, out_dir / "state_events.csv")
            copied_files.append("state_events.csv")
            break

    config_path = args.config or _read_config_from_notes(run_dir / "notes.md")
    resolved_config = None
    if config_path:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path
        if config_path.exists():
            resolved_config = config_path
            _write_effective_config(config_path, out_dir / "effective_config.yaml")
            copied_files.append("effective_config.yaml")

    fig_patterns = args.figs.split(",") if args.figs else [
        "figures/compare/fig*.png",
        "figures/report.md",
        "figs/fig4_trades_window.png",
    ]
    copied_figs = _copy_figures(run_dir, out_dir, fig_patterns)

    repro_dir = ensure_dir(out_dir / "repro")
    command = args.command
    if not command and resolved_config:
        cfg_rel = _relative(resolved_config, root)
        command = f"python scripts/run.py --config {cfg_rel} --outdir {_relative(run_dir, root)}"
    if not command:
        command = "unknown"
    (repro_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    sha = _git_sha(root)
    (repro_dir / "git_sha.txt").write_text(sha + "\n", encoding="utf-8")

    env_lines = [f"python: {sys.version.split()[0]}"]
    if args.pip_freeze:
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            env_lines.append("")
            env_lines.append(freeze.strip())
        except Exception:
            env_lines.append("")
            env_lines.append("pip_freeze: failed")
    (repro_dir / "env.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    metrics = None
    metrics_path = out_dir / "key_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    summary_lines = [
        f"# {args.name}",
        f"- published_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- source_run: {_relative(run_dir, root)}",
        f"- config: {_relative(resolved_config, root) if resolved_config else 'unknown'}",
        f"- command: {command}",
        f"- git_sha: {sha}",
        "",
    ]
    if args.goal:
        summary_lines.extend(["## Goal", f"- {args.goal}", ""])
    if args.note:
        summary_lines.extend(["## Notes", f"- {args.note}", ""])
    if metrics:
        summary_lines.extend(
            [
                "## Key Metrics",
                f"- final_nav: {metrics.get('final_nav')}",
                f"- ROI: {metrics.get('ROI')}",
                f"- maxDD: {metrics.get('maxDD')}",
                f"- trades_count: {metrics.get('trades_count')}",
                "",
            ]
        )
    if copied_files or copied_figs:
        summary_lines.append("## Included Files")
        for item in copied_files:
            summary_lines.append(f"- {item}")
        for item in copied_figs:
            summary_lines.append(f"- {item}")
        summary_lines.append("")

    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    index_path = milestones_dir / "README.md"
    summary_rel = f"{args.name}/summary.md"
    _update_index(index_path, args.name, summary_rel)

    print(f"milestone created: {out_dir}")


if __name__ == "__main__":
    main()

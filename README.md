# 2229059 Gold & Bitcoin Backtest (Paper Reproduction)

This repo reproduces Team #2229059's gold/bitcoin trading model, paper figures, and tables.
It is written for team use: one backtest entrypoint, one figure repro pipeline, and a
tracked milestone window under `outputs/_milestones/` for PR reviews and comparisons.

## 0) Repo layout (what to use)
- `scripts/`: runnable entrypoints
- `src/`: core logic (backtest / strategy / indicators / metrics)
- `config/`: all configs (edit these, not the code)
- `docs/`: authoritative docs (start at `docs/DOCS_INDEX.md`)
- `outputs/`: local runs (ignored; safe to delete)
- `outputs/_milestones/`: tracked milestone snapshots (reviewable)
- `assets/papers/`: paper PDFs

## 1) Install
```bash
python -m pip install -r requirements.txt
```

## 2) Run a core backtest (default)
```bash
python scripts/run.py --config config/base.yaml --outdir outputs/run_base
```
Outputs (local, not tracked):
- `outputs/run_base/results_table.csv`
- `outputs/run_base/trades.csv`
- `outputs/run_base/key_metrics.json`
- `outputs/run_base/figs/`
- `outputs/run_base/notes.md`

## 3) Reproduce paper figures (Figure 1-7 + compare report)
```bash
python scripts/reproduce_all_figures.py --config config/fixpack.yaml --outdir outputs/run_figures
```
Outputs:
- `outputs/run_figures/figures/ours/`
- `outputs/run_figures/figures/paper/`
- `outputs/run_figures/figures/compare/`
- `outputs/run_figures/figures/report.md`

If you need to re-extract from the PDF:
```bash
python scripts/extract_paper_figures.py --config config/figures_bboxes.yaml --outdir outputs/paper_figures
```

## 4) Tables (param grid + fee sensitivity)
```bash
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml --outdir outputs/param_grid
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml --outdir outputs/fee_sweep
```

## 5) Publish a milestone (tracked)
Milestones are the only tracked artifacts under `outputs/_milestones/`.
```bash
python scripts/publish_milestone.py \
  --run outputs/run_fixpack \
  --name M01_fixpack_220k \
  --config config/fixpack.yaml
```
This creates a minimal evidence pack in `outputs/_milestones/M01_fixpack_220k/`
and updates `outputs/_milestones/README.md`.

## 6) Keep outputs tidy (local cleanup)
```bash
python scripts/clean_outputs.py --keep 5 --dry-run
python scripts/clean_outputs.py --keep 5
```

## 7) Tests
```bash
python -m unittest discover -s tests
```

## Docs
Start here: `docs/DOCS_INDEX.md`

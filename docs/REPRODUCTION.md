# Reproduction

This project targets the Team #2229059 gold/bitcoin trading model and its paper figures/tables.

## Core backtest
```bash
python scripts/run.py --config config/base.yaml --outdir outputs/run_base
```
Outputs: `outputs/run_base/results_table.csv`, `trades.csv`, `key_metrics.json`, and `figs/`.

## Parameter grid (Figure 6 table)
```bash
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml --outdir outputs/param_grid
```
Output: `outputs/param_grid/param_sweep.csv`.

## Fee sensitivity (Figure 7 table)
```bash
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml --outdir outputs/fee_sweep
```
Output: `outputs/fee_sweep/fee_sensitivity.csv`.

## Paper figure windows (Figure 8/9)
If you need the paper window overlays, use the fixpack configuration:
```bash
python scripts/run.py --config config/fixpack.yaml --outdir outputs/run_fixpack
```

## Optional paper-aligned profile
```bash
python scripts/run.py --config config/recommended_paper_aligned.yaml --outdir outputs/run_paper_aligned
```

## Notes
- Output schema is defined in `docs/interface_contract.md`.
- Assumptions are tracked in `docs/assumption_ledger.md`.
- Full figure reproduction (paper vs ours) is documented in `docs/FIGURES_REPRO.md`.
- Milestones are tracked under `outputs/_milestones/` (see `scripts/publish_milestone.py`).

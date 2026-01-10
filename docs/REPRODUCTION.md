# Reproduction

This project targets the Team #2229059 gold/bitcoin trading model and its paper figures/tables.

## Core backtest
```bash
python scripts/run.py --config config/base.yaml
```
Outputs: `outputs/run_<id>/results_table.csv`, `trades.csv`, `key_metrics.json`, and `figs/`.

## Parameter grid (Figure 6 table)
```bash
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml
```
Output: `outputs/param_grid/param_sweep.csv`.

## Fee sensitivity (Figure 7 table)
```bash
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml
```
Output: `outputs/fee_sweep/fee_sensitivity.csv`.

## Paper figure windows (Figure 8/9)
If you need the paper window overlays, use the fixpack configuration:
```bash
python scripts/run.py --config config/fixpack.yaml
```

## Optional paper-aligned profile
```bash
python scripts/run.py --config config/recommended_paper_aligned.yaml
```

## Notes
- Output schema is defined in `docs/interface_contract.md`.
- Assumptions are tracked in `docs/assumption_ledger.md`.
- Full figure reproduction (paper vs ours) is documented in `docs/FIGURES_REPRO.md`.

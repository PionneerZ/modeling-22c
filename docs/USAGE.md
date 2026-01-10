# Usage

## Install
```bash
python -m pip install -r requirements.txt
```

## Run a single backtest
```bash
python scripts/run.py --config config/base.yaml --outdir outputs/run_base
```
Outputs:
- `outputs/run_base/results_table.csv`
- `outputs/run_base/trades.csv`
- `outputs/run_base/key_metrics.json`
- `outputs/run_base/figs/`
- `outputs/run_base/notes.md`

## Parameter grid (Figure 6 table)
```bash
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml --outdir outputs/param_grid
```
Output:
- `outputs/param_grid/param_sweep.csv`

## Fee sensitivity (Figure 7 table)
```bash
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml --outdir outputs/fee_sweep
```
Output:
- `outputs/fee_sweep/fee_sensitivity.csv`

## Tests
```bash
python -m unittest discover -s tests
```

## Data
- Primary sources: `problem/BCHAIN-MKPRU.csv` and `problem/LBMA-GOLD.csv`.
- Configure paths and column mapping in `config/base.yaml` under `data`.

## Figures
- Paper figure reproduction uses a separate pipeline. See `docs/FIGURES_REPRO.md`.

## Milestones (tracked)
`outputs/` is local and ignored. Publish milestones to the tracked window:
```bash
python scripts/publish_milestone.py --run outputs/run_base --name M01_run_base --config config/base.yaml
```

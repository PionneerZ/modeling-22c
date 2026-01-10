# Usage

## Install
```bash
python -m pip install -r requirements.txt
```

## Run a single backtest
```bash
python scripts/run.py --config config/base.yaml
```
Outputs:
- `outputs/run_<id>/results_table.csv`
- `outputs/run_<id>/trades.csv`
- `outputs/run_<id>/key_metrics.json`
- `outputs/run_<id>/figs/`
- `outputs/run_<id>/notes.md`

## Parameter grid (Figure 6 table)
```bash
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml
```
Output:
- `outputs/param_grid/param_sweep.csv`

## Fee sensitivity (Figure 7 table)
```bash
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml
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

# 2229059 Gold & Bitcoin Day Trading

Quickstart:
```bash
python -m pip install -r requirements.txt
python scripts/run.py --config config/base.yaml
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml
python -m unittest discover -s tests
```

Outputs:
- `outputs/run_<id>/` (single backtest)
- `outputs/param_grid/param_sweep.csv` (Figure 6 table)
- `outputs/fee_sweep/fee_sensitivity.csv` (Figure 7 table)

Docs:
- `docs/USAGE.md`
- `docs/CONFIG.md`
- `docs/REPRODUCTION.md`
- `docs/FIGURES_REPRO.md`
- `docs/interface_contract.md`
- `docs/assumption_ledger.md`
- `docs/spec_extracted.md`

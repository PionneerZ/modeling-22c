# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core logic (data loading, indicators, strategy, backtest, metrics, plots).
- `scripts/`: entry points (`run.py`, `batch_param_grid.py`, `batch_sensitivity.py`).
- `config/`: YAML configs (`base.yaml`, `reproduce_tables.yaml`).
- `data/`: input data and notes (`sample_prices.csv`, `README.md`).
- `outputs/`: generated artifacts (`run_<id>/`, `param_grid/`, `fee_sweep/`).
- `docs/`: Chinese handoff/spec/assumptions/repro docs.
- `tests/`: unit tests (unittest).
- `requirements.txt`: Python dependencies.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`  
  Install runtime dependencies.
- `python scripts/run.py --config config/base.yaml`  
  Run a full backtest and generate `outputs/run_<id>/`.
- `python scripts/batch_param_grid.py --config config/reproduce_tables.yaml`  
  Reproduce the parameter grid table to `outputs/param_grid/param_sweep.csv`.
- `python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml`  
  Reproduce the fee sensitivity table to `outputs/fee_sweep/fee_sensitivity.csv`.
- `python -m unittest discover -s tests`  
  Run all unit tests.

## Coding Style & Naming Conventions
- Python 3; 4-space indentation; keep identifiers in English.
- Prefer config-driven behavior (edit `config/*.yaml` instead of hardcoding).
- Output field names are fixed by contract (see `docs/interface_contract.md`).
- No formatting or lint tooling is configured; keep edits minimal and readable.

## Reproduction References (must read before logic changes)
- Original paper: `无水印-2229059.pdf`.
- Spec extraction: `docs/spec_extracted.md`.
- Modeling memo: `论文1复刻memo.pdf`.
- Baseline replication notebook: `replication1.html`.
- Ambiguities and deviations: `docs/assumption_ledger.md`.
- Output contract: `docs/interface_contract.md`.

## Reproduction Priority
- Target 100% reproduction of paper results and figures; treat mismatches as bugs.
- Manual parameter tuning is allowed, but changes must be recorded in config and
  summarized in `outputs/run_<id>/notes.md` and `docs/assumption_ledger.md`.

## Testing Guidelines
- Framework: `unittest`.
- Test files live in `tests/` and use `test_*.py` naming.
- Add tests for behavior changes that affect signals, fees, or state logic.

## Commit & Pull Request Guidelines
- Git tooling is not available in this environment, so no local history to infer conventions.
- Use clear, imperative commit messages (e.g., `Add fee sensitivity batch script`).
- PRs should include: summary of changes, how to run/verify, and any data/config impacts.

## Configuration & Data Notes
- Do not fetch external data; use local files under `data/` or `problem/` only.
- Replace `data/sample_prices.csv` with real data and update `config/base.yaml` mappings.
- Treat `outputs/` as generated artifacts; avoid manual edits.
- Use the paper data in `problem/` as the primary source when reproducing
  results (`problem/BCHAIN-MKPRU.csv`, `problem/LBMA-GOLD.csv`).

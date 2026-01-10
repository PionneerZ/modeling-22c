# Project Goals
This file records the final goals for this repo. Use it as the stable target for
future changes and reviews.

## Primary References
- `assets/papers/paper.pdf` (original paper).
- `docs/spec_extracted.md` (paper formulas and thresholds).
- `assets/papers/memo.pdf` (modeling interpretation and pitfalls).
- `docs/interface_contract.md` (output contract).
- `docs/assumption_ledger.md` (ambiguities and deviations).

## Reproduction Targets
- Implement the paper's trading logic (momentum, mean reversion, profitability,
  selling, extreme, no-buy, holding, daily order: sell then buy), aligned with
  `docs/spec_extracted.md` and `assets/papers/memo.pdf`.
- Use the paper time range 2016-09-11 to 2021-09-10, initial cash 1000, BTC
  calendar anchor, and t=0 on start_date; gold trades only on trading days but
  is valued every day.
- Reproduce Table 4.3 and Table 4.4 final amounts exactly, with [12, 0.6, 0.89]
  -> 220486 as the headline target. Manual parameter tuning is allowed, but all
  changes must live in config and be documented in `outputs/run_<id>/notes.md`
  and `docs/assumption_ledger.md`.
- Produce paper-aligned figures: NAV (Figure 4), buy/sell windows for BTC
  t=1610..1760 and Gold t=460..560 (Figures 8/9), plus drawdown and position
  plots.

## Macro Strategy
- Align preprocessing and calendar handling with the paper spec and assumption
  ledger (master timeline, gold tradable flag, gold valuation fill, BTC daily
  anchor).
- Validate model formulas against the memo's interpretation (buy signal as
  event intersection, profitability/weight logic, fixed margin L, holding T).
- Iterate parameter tuning through `config/*.yaml` until tables and curves
  match the paper; treat residual gaps as blocking issues.

## Deliverables and Interfaces
- `scripts/run.py` generates a complete run package: `results_table.csv`,
  `trades.csv`, `key_metrics.json`, `figs/fig1_nav.png` to
  `figs/fig4_trades_window.png`, and `notes.md`.
- `scripts/batch_param_grid.py` outputs `outputs/param_grid/param_sweep.csv` in
  the Table 4.3 format.
- `scripts/batch_sensitivity.py` outputs
  `outputs/fee_sweep/fee_sensitivity.csv` in the Table 4.4 format.
- Output field names and meanings strictly follow
  `docs/interface_contract.md`.

## Configuration and Data
- Keep behavior config-driven via `config/*.yaml`; avoid hardcoded parameters.
- Use local data only (`problem/` or `data/`); do not fetch external data.
- Ensure date parsing, column mapping, and trading calendar rules remain
  explicit and configurable.

## Quality and Diagnostics
- All unit tests under `tests/` pass; add tests when changing signals, fees, or
  state logic.
- Maintain `docs/assumption_ledger.md` for any paper ambiguities or
  implementation deviations.
- Keep `docs/REPRODUCTION.md` up to date with the minimal reproducibility steps.

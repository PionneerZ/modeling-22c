# Experiments (non-production tools)

These scripts are exploratory/diagnostic and are not part of the core reproduction flow.

- `data_sanity.py`: print data coverage, missing ranges, and anchor points.
- `diagnose_run_diff.py`: compare two runs and export NAV/event diffs.
- `export_effective_config.py`: dump an expanded config with resolved defaults.
- `locate_paper_windows.py`: locate paper window indices for figure alignment.
- `repro_commands.md`: command snippets for experimental runs.
- `run_paper_final.py`: one-off reproduction pipeline with debug exports.
- `run_with_debug.py`: run backtest with full debug output to `outputs/<run>/debug/`.

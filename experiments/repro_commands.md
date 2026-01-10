# Repro Commands

## A) High-NAV baseline (reference)
- Run: `python scripts/run.py --config config/highnav_ref.yaml`
- Effective config: `python experiments/export_effective_config.py --config config/highnav_ref.yaml --out outputs/run_highnav_ref/effective_config.json`

## B) Paper-aligned baseline
- Run: `python experiments/run_paper_final.py --config config/base.yaml --run-id run_paper_ref`
- Effective config: `python experiments/export_effective_config.py --config config/base.yaml --out outputs/run_paper_ref/effective_config.json`

## C) Fixpack (NAV restored + Figure8 alignment)
- Run: `python experiments/run_with_debug.py --config config/fixpack.yaml`
- Effective config: `python experiments/export_effective_config.py --config config/fixpack.yaml --out outputs/run_fixpack/effective_config.json`

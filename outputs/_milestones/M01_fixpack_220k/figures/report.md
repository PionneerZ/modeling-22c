# Figure Reproduction Report
- generated_at: 2026-01-10T11:04:49
- config: config/fixpack.yaml
- outdir: outputs\run_fixpack
- paper_manifest: outputs\paper_figures\figures_manifest.csv
- tables_config: config/reproduce_tables.yaml

## Figure 1
- meaning: BTC and gold price paths in separate panels.
- implementation: src/figures.py:plot_fig1
- data: price series from data load
- key_params: data.start_date/end_date
- alignment: OK
- notes: date range matches 2016-09-11..2021-09-10

## Figure 2
- meaning: BTC and gold prices on a shared axis to show scale.
- implementation: src/figures.py:plot_fig2
- data: price series from data load
- key_params: data.start_date/end_date
- alignment: OK
- notes: date range matches 2016-09-11..2021-09-10

## Figure 3
- meaning: Flow chart for buy logic.
- implementation: src/figures.py:plot_fig3
- data: diagram only
- key_params: strategy logic
- alignment: OK
- notes: diagram-only figure (no data alignment needed)

## Figure 4
- meaning: Portfolio NAV breakdown over time.
- implementation: src/figures.py:plot_fig4
- data: results_table.csv nav_* columns
- key_params: full config
- alignment: Deviation
- notes: final_nav=235754.32 vs paper 220486

## Figure 5
- meaning: Weight-factor adjusted signal strength over time.
- implementation: src/figures.py:plot_fig5
- data: profit_* columns + weight_factor config
- key_params: weight_factor, signals
- alignment: Deviation
- notes: apply_mode=compare, W_use=['btc']

## Figure 6
- meaning: NAV under parameter grid variations.
- implementation: src/figures.py:plot_fig6
- data: param_grid from config/reproduce_tables.yaml
- key_params: T/N/E grid + fee pair
- alignment: Deviation
- notes: best_params=[5, 0.4, 0.89] expected [12, 0.6, 0.89]

## Figure 7
- meaning: NAV under fee sensitivity variations.
- implementation: src/figures.py:plot_fig7
- data: fee_sensitivity from config/reproduce_tables.yaml
- key_params: fee pairs + fixed T/N/E
- alignment: Deviation
- notes: 0.01,0.02 235754.32 vs 220486; 0.03,0.05 129971.13 vs 156038; 0.1,0.1 93806.76 vs 47452

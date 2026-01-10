# Configuration

Main config file: `config/base.yaml`.

## Common configs
- `config/base.yaml`: default run configuration.
- `config/fixpack.yaml`: paper-aligned window settings for Figure 8/9 and NAV tuning.
- `config/recommended_paper_aligned.yaml`: optional paper-aligned profile.
- `config/reproduce_tables.yaml`: parameter grid and fee sensitivity inputs.

## run
- `initial_cash`: starting capital.
- `seed`: random seed for reproducibility.
- `run_id`: optional fixed output folder name.
- `change_note`: run summary recorded in `outputs/run_<id>/notes.md`.

## data
- `path`: single merged CSV path (fallback).
- `gold_path` / `btc_path`: split data paths (preferred for paper data).
- `column_map`: column names for date/gold/btc.
- `date_format`: format string for parsing dates.
- `start_date` / `end_date`: backtest window.
- `end_date_mode`: `trade_end` or `valuation_end`.
- `calendar_anchor`: `btc` (daily) or `business` (trading days).
- `gold_ffill_for_valuation`: forward-fill gold for valuation.
- `gold_trade_weekdays_only`: restrict gold trades to weekdays.
- `gold_trade_on_filled`: allow trading on filled prices.
- `gold_trade_fill_method`: `ffill` or `ffill_bfill` when trading on filled prices.

## fees
- `fee_gold`, `fee_btc`: transaction fee rates.

## momentum / paper_params / thresholds
- `momentum.n_window`: momentum window (legacy alias to `paper_params.lookback_M`).
- `paper_params`: `hold_T`, `reentry_N`, `extreme_E`, `lookback_M`.
- `thresholds`: gradient and MA diff thresholds.

## weight_factor
- `W_C`: scale factor for W = C / t^2.
- `W_use`: list of assets to apply weight (`gold`, `btc`, or `all`).
- `apply_mode`: `compare`, `score`, or `none`.
- `use_t_plus_one`: use t+1 to avoid division by zero.

## selling / extreme / no_buy / holding
- `selling.margin_L`: profit margin threshold for selling.
- `selling.avg_cost_method`: `weighted_avg` or `fifo`.
- `extreme.*`: extreme detection and sell configuration.
- `no_buy.*`: post-extreme cooldown and re-entry configuration.
- `holding.*`: min days between buys/sells and hold reference.

## buy_sizing / execution / strategy
- `buy_sizing`: `score` or `fixed_fraction` modes and parameters.
- `execution.sell_then_buy`: enforce sell-before-buy ordering.
- `strategy.mode`: `full` or `baseline`.

## signals / buy_logic
- `signals.*`: MA inclusion, reversion mode, score aggregation and thresholds.
- `buy_logic.*`: single vs multi buy selection and optional rebalance.

## plots
- `plots.window_btc_t` / `plots.window_gold_t`: t-window for trade plots.
- `plots.window_btc_date` / `plots.window_gold_date`: date-window alternative.

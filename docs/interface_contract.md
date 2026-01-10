# 输出契约（Interface Contract）

## outputs/run_<id>/results_table.csv（日线）
必含字段（含义）：
- `date`：日期（ISO）。
- `price_gold`, `price_btc`：当日价格。
- `cash`, `gold_qty`, `btc_qty`：现金与持仓数量。
- `nav_total`, `nav_cash`, `nav_gold`, `nav_btc`：净值拆分。
- `dd`：回撤（当前净值/历史峰值 - 1）。
- `state_extreme`：极端状态（none/gold/btc/both）。
- `state_no_buy`：no-buy 状态（none/gold/btc/both）。
- `hold_left_gold`, `hold_left_btc`：剩余不可卖天数。
- `profit_gold`, `profit_btc`：利润评分。
- `chosen_buy_asset`：当日买入选择（gold/btc/none）。
- `sell_flag_gold`, `sell_flag_btc`：卖出信号标记（True/False）。
- `t`（可选）：自然日序号（Calendar Day），**t=0** 为 start_date 当天。

## outputs/run_<id>/trades.csv（交易日志）
必含字段（含义）：
- `date`：交易日期。
- `asset`：gold/btc。
- `side`：buy/sell。
- `qty`：数量。
- `price`：成交价。
- `notional`：成交额。
- `fee_paid`：手续费。
- `cash_change`：现金变化（买为负，卖为正）。
- `reason`：触发原因（buy_momentum/buy_reversion/sell_signal/extreme_sell/baseline_buy/blocked_by_*）。
- `signal_score`：触发强度。
- `t`（可选）：交易日序号（与 results_table 对齐）。

## outputs/run_<id>/key_metrics.json
必含字段：
- `final_nav` / `ROI` / `maxDD`（含 `max_drawdown` 别名）
- `trades_count`
- `params`（T/N/E 等）
- `fee_pair`
- `run_id`
- `data_range`
- `calendar_mode`
- `end_date_mode`（trade_end / valuation_end）

## outputs/run_<id>/figs/
- `fig1_nav.png`：净值曲线（对标 Figure 4）。
- `fig2_drawdown.png`：回撤曲线。
- `fig3_positions.png`：仓位比例。
- `fig4_trades_window.png`（可选）：区间买卖点覆盖（Figure 8/9）。

## outputs/run_<id>/figures/ (paper figure reproduction)
- `ours/figX.png`：生成图（Figure 1–7）。
- `paper/figX.png`：论文图（裁剪自 PDF）。
- `compare/figX_compare.png`：paper vs ours 并排对比图。
- `report.md`：逐图复现说明与对齐备注。

## outputs/paper_figures/
- `pages/page_<n>.png`：包含 Figure 的整页截图。
- `figX_paper.png`：裁剪后的论文图。
- `figures_manifest.csv`：figure_id/page/bbox/title 清单。

## outputs/run_<id>/notes.md
运行口径、参数、变更摘要与差异来源。

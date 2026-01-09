# notes
- run_id: run_20260107_165646
- config: config\base.yaml
- run_path: outputs\run_20260107_165646
- params[T,N,E]: (12, 0.6, 0.89)
- fee_pair: (0.01, 0.02)
- strategy_mode: full

## 变更摘要
- iter4: extreme triggers only when holding asset (avoid lockout)

## 差异来源 Top3（若与论文不一致）
1. 数据源与时间区间（trade_end vs valuation_end）。
2. 信号尺度与权重因子 W 的实现细节（见 assumption_ledger）。
3. 状态机边界（extreme/no-buy/holding 的阻断逻辑）。

## 数据覆盖
- source: problem\LBMA-GOLD.csv | problem\BCHAIN-MKPRU.csv
- raw_range: 2016-09-11 to 2021-09-10
- trade_end: 2021-09-10
- btc_missing_days: 0
- gold_missing_weekday: 50
- gold_missing_weekend: 522
- gold_ffill_for_valuation: True
- gold_trade_weekdays_only: True

# notes
- run_id: run_20260107_160244
- config: config/base.yaml
- run_path: outputs\run_20260107_160244
- params[T,N,E]: (12, 0.6, 0.89)
- fee_pair: (0.01, 0.02)
- strategy_mode: full

## 变更摘要
- 已实现完整策略（动量/均值回归/极端行情/no-buy/holding/先卖后买）。

## 差异来源 Top3（如与论文数值不一致）
1. 数据源与时间范围可能不同（示例数据仅用于跑通流程）。
2. 盈利性映射与权重因子 W 的实现细节（详见 assumption_ledger）。
3. 极端行情与 no-buy 的边界处理（如持仓锁定与触发条件）。
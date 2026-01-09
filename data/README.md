数据文件说明：
- `sample_prices.csv` 为示例数据（仅用于跑通流程）。
- 实际复现请替换为真实数据，并保持字段：`date`, `gold`, `btc`。
- gold 可在周末缺失（空值），由 `gold_ffill` 决定是否前向填充。

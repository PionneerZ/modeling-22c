# 假设与口径补齐（Assumption Ledger）
仅记录原文未明确但实现必须确定的细节。

| 项目 | 现实现口径 | 依据/原因 | 影响 |
| --- | --- | --- | --- |
| 数据来源 | 使用 `problem/BCHAIN-MKPRU.csv` 与 `problem/LBMA-GOLD.csv`（分表） | 用户提供原文数据 | 需配置 `data.gold_path/btc_path` 与 `column_map` |
| 日期解析 | `date_format="%m/%d/%y"` | CSV 日期格式为 9/11/16 | 影响对齐与 t 起点 |
| 日历锚点 | `calendar_anchor=btc` 自然日连续 | BTC 每天可交易 | t 与 Figure 8/9 对齐 |
| t 计数 | t=0 为 `start_date` 当天；W 使用 t（t=0 按 1 处理） | 避免 t=0 除零 | 影响 W 的尺度 |
| 交易/估值截止 | `end_date=2021-09-10` 且 `end_date_mode=trade_end` | 论文时间区间（PDF p1/p7） | 对齐回测区间 |
| gold 周末/节假日 | 仅交易日可交易；估值使用 ffill | 论文说明 gold 仅周一至周五交易 | 周末不产生交易 |
| W 使用位置 | `W_C=100000`，仅对 BTC 生效，`apply_mode=compare` | 论文 3.9：BTC 信号更重 | BTC 更易被选中 |
| 买量映射 | 固定投入现金 60%，no-buy 解除时全仓回买 | 结合 `replication1.html` 与论文回买描述 | 回买更激进 |
| 阈值尺度 | `thr_grad=1`、`thr_ma_diff=1`，`score_threshold=0` | 论文 3.1 条件与无额外阈值 | 买入触发更频繁 |
| no-buy 释放 | `no_buy.mode=price` + `scope=global`，`rebuy_pct_N` 使用 N，`rebuy_on_release=true`，`max_price_ref=max`，`rebuy_fraction=1.0` | 论文 4.3：N 为回买阈值 | 解除时触发回买 |
| extreme 激活 | 仅对 BTC 启用；仅持仓时触发；平均梯度取正值均值；`min_history_days=420` | 论文 3.5 聚焦 BTC 极端行情，避免负梯度抵消 | 降低早期误触发 |
| 平均成本 | `avg_cost_method=weighted_avg` | 原文强调 average price paid | 影响卖出触发 |
| 阻断记录 | `blocked_by_*` 写入 `trades.csv` | 便于诊断状态机 | 诊断可追溯 |
| MA 窗口 | 使用含当日价格的均值，允许不足窗口 | 论文公式含 `P_t` 且早期缺数据 | 早期信号更强 |
| 回归阈值 | 使用 `P < MA` | 论文 3.2（无 N 门槛） | 回归信号更密集 |
| 信号强度 | W 仅用于比较资产；不设额外买入阈值 | 论文 3.9 + 3.1/3.2 | 信号直接驱动买入 |

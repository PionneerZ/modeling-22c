# 假设与口径补齐（Assumption Ledger）
仅记录原文未明确但实现必须确定的细节。

| 项目 | 现实现口径 | 依据/原因 | 影响 |
| --- | --- | --- | --- |
| 数据来源 | 使用 `problem/BCHAIN-MKPRU.csv` 与 `problem/LBMA-GOLD.csv`（分表） | 用户提供原文数据 | 需配置 `data.gold_path/btc_path` 与 `column_map` |
| 日期解析 | `date_format="%m/%d/%y"` | CSV 日期格式为 9/11/16 | 影响对齐与 t 起点 |
| 日历锚点 | `calendar_anchor=btc` 自然日连续 | BTC 每天可交易 | t 与 Figure 8/9 对齐 |
| t 计数 | t=0 为 `start_date` 当天；W 使用 t（t=0 按 1 处理） | 避免 t=0 除零 | 影响 W 的尺度 |
| 交易/估值截止 | `end_date=2021-09-10` 且 `end_date_mode=trade_end` | 参考 `replication1.html` | 对齐回测区间 |
| gold 周末/节假日 | 允许使用填充值交易；估值使用 ffill+bfill | 参考 `replication1.html` | 交易频率提高 |
| W 使用位置 | `W_C=100000`，对 BTC/GOLD 同时生效，`apply_mode=score` | 参考 `replication1.html` | 早期权重更大 |
| 买量映射 | 固定投入现金 50% | 参考 `replication1.html` | 交易规模固定化 |
| 阈值尺度 | `thr_grad=0`、`thr_ma_diff=0`，`score_threshold=100` | 参考 `replication1.html` | 买入触发更稀疏 |
| no-buy 释放 | `no_buy.mode=days` 且 `no_buy_days=0` | 先对齐 notebook 逻辑 | 暂不阻断回买 |
| extreme 激活 | 仅在持有该资产时触发 extreme | 避免空仓被极端状态锁死 | 影响卖点与 no-buy |
| 平均成本 | `avg_cost_method=weighted_avg` | 原文强调 average price paid | 影响卖出触发 |
| 阻断记录 | `blocked_by_*` 写入 `trades.csv` | 便于诊断状态机 | 诊断可追溯 |
| MA 窗口 | 使用前 n 天均值（不含当日价格） | 参考 `replication1.html` | 信号时点偏移 |
| 回归阈值 | 使用 `P < MA * N`（N 来自参数） | 参考 `replication1.html` | 回归信号密度依赖 N |
| 信号强度 | `score = W * (mom + rev)` 且 `score > 100` 才买 | 参考 `replication1.html` | 买入触发更稀疏 |

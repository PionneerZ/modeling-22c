# 假设与口径补齐（Assumption Ledger）
仅记录原文未明确但实现必须确定的细节。默认口径以 `config/base.yaml` 为准；若使用 `config/fixpack.yaml`，请参考文末的 override 说明。

| 项目 | 现实现口径 | 依据/原因 | 影响 |
| --- | --- | --- | --- |
| 数据来源 | 使用 `problem/BCHAIN-MKPRU.csv` 与 `problem/LBMA-GOLD.csv`（分表） | 用户提供原文数据 | 需配置 `data.gold_path/btc_path` 与 `column_map` |
| 日期解析 | `date_format="%m/%d/%y"` | CSV 日期格式为 9/11/16 | 影响对齐与 t 起点 |
| 日历锚点 | `calendar_anchor=btc` 自然日（每日）索引 | 论文 Figure 8 的 t=1610..1760 覆盖 5 年区间，若使用交易日索引总长度不足 1610 | t 与 Figure 8/9 对齐 |
| t 计数 | t=0 为 `start_date` 当天（自然日序号） | 以日历日序号作为统一索引 | 影响 W 的尺度与窗口定位 |
| 参数映射 | `paper_params` 显式映射：T=持有期、N=回买阈值、E=极端卖出阈值 | 论文 4.3 对 [T,N,E] 的定义 | 避免参数串线 |
| 交易/估值截止 | `end_date=2021-09-10` 且 `end_date_mode=trade_end` | 论文时间区间（PDF p1/p7） | 对齐回测区间 |
| gold 周末/节假日 | 仅交易日可交易；估值使用 ffill | 论文说明 gold 仅周一至周五交易 | 周末不产生交易 |
| W 使用位置 | `W_C=100000`，对 gold/btc 的 `profit_score` 同步缩放（`apply_mode=score`，`W_use=all`） | 论文 3.9：噪声归一化因子 W= C/t^2 | 资金投入随时间衰减，BTC 仍因波动更大而占优 |
| 买量映射 | `mode=fixed_fraction`，每次买入使用 `cash*0.5` | 复刻 notebook 的交易口径 | 买入节奏更接近复现代码 |
| 阈值尺度 | `thr_grad=0`、`thr_ma_diff=0`，`score_threshold=10` | 复刻口径下为匹配 gold 窗口买入密度调低阈值 | 控制买入频率 |
| no-buy 释放 | `scope=asset`；`release_mode=hybrid`；`release_direction=drop`；价格需跌至 `N×max_price` 且冷却天数到期 | 论文 3.6/4.3：no-buy 为天数 + N 为回买阈值（跌至最大值百分比） | 解除时触发回买 |
| no-buy 冷却天数 | `cooldown_days=60`（自然日） | 结合极端卖出时点与 Figure 8 的回买位置，避免阻塞回买 | 控制禁买时长 |
| extreme 触发资产 | gold/btc 均启用 | 论文未限制资产类型，BTC 波动大自然更易触发 | 极端卖出以 BTC 为主 |
| extreme 平均窗口 | `avg_mode=window`，窗口长度取 `lookback_M`，`min_history_days=1600` | 论文 3.5：平均历史梯度；避免早期样本触发非论文极端事件 | 极端触发更贴近论文图 |
| holding 规则 | `min_days_between_sells=T`，`sell_hold_ref=last_buy`；不限制买入冷却 | 复刻 notebook 使用 `last_buy_date_idx` 控制卖出 | 频繁买入会推迟卖出 |
| 平均成本 | `avg_cost_method=weighted_avg` | 原文强调 average price paid | 影响卖出触发 |
| 阻断记录 | `blocked_by_*` 写入 `trades.csv` | 便于诊断状态机 | 诊断可追溯 |
| MA 窗口 | 不含当日价格，且要求窗口足长 | 复刻 notebook 的 MA 计算 | 早期信号更少 |
| 回归阈值 | 使用 `P < MA * N` | 复刻 notebook 的 N 口径 | 回归信号更稀疏 |
| 信号强度 | W 作用于 `profit_score`；买入需 `score>10` | 复刻 notebook 触发条件 | 避免低强度噪声交易 |
| fixpack override | `no_buy.release_frac=0.75`；`buy_sizing.fixed_fraction=0.55`；`extreme.min_history_days=420` | 仅用于 `config/fixpack.yaml`（非默认） | 提升 NAV，同时保留 Figure8 关键事件 |

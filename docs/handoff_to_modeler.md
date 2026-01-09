# 统计交接：诊断 / 稳健性 / 差距归因

## 0. C 要做什么（任务清单）
- 诊断：确认状态机是否按口径运行（extreme/no-buy/holding/卖先买后）。
- 稳健性：复现 4.3/4.4，验证趋势是否与论文一致。
- 差距归因：若 final amount 偏离论文，定位来源并给出最小改动建议。

## 1. 数据字段“读法”（与契约一致）
- `results_table.csv` 重点字段：
  - `state_extreme` / `state_no_buy`：状态机主状态（none/gold/btc/both）。
  - `hold_left_gold` / `hold_left_btc`：剩余不可卖天数。
  - `profit_gold` / `profit_btc`：利润评分，用于资产选择与买量。
  - `chosen_buy_asset`：当日最终选择买入资产。
  - `sell_flag_gold` / `sell_flag_btc`：当日卖出信号标记。
  - `t`：日序号（t=0 为 start_date）。
- `trades.csv` 重点字段：
  - `reason`：`buy_momentum`/`buy_reversion`/`sell_signal`/`extreme_sell`/`blocked_by_*`。
  - `signal_score`：交易触发强度（用来解释买卖点）。

## 2. 诊断清单（至少 10 条，可执行）
1) `trades_count` 过高 → 看 `trades.csv` 中 `buy_*` 是否过密 → 过密说明阈值过低或 no-buy 失效 → 调整 `thresholds.*` 或 `no_buy.*`。
2) 费用侵蚀明显 → 对比 `fee_pair` 与 `ROI` → 费率小但 ROI 低说明高频交易 → 检查 `sell_signal` 与 `hold_left_*` 是否导致频繁平仓。
3) extreme 触发过久 → 看 `results_table.csv:state_extreme` 持续段 → 若长时间为 both，说明触发条件过敏或未退出 → 检查 `extreme_C` 与 `extreme_sell_pct_E`。
4) no-buy 长期阻断 → 看 `state_no_buy` 与 `trades.csv:blocked_by_no_buy` → 长期阻断会导致空仓 → 调整 `no_buy.mode/no_buy_days`。
5) holding 是否真正生效 → 比较 `hold_left_*` 与 `sell_flag_*` → 若 sell_flag 为 True 但无 sell 交易，说明被 holding 阻断 → 检查 `blocked_by_hold` 记录。
6) cash 在大跌段是否上升 → 选 BTC 大跌窗口（如 t≈500/1050/1625）看 `nav_cash` 变化 → 若不升，说明卖出逻辑缺失 → 检查 sell 与 extreme。
7) profit_gold vs profit_btc 比较是否偏 BTC → 看 `profit_*` 与 `chosen_buy_asset` → 若 gold 经常被选，可能 W 作用位置不对 → 检查 `weight_factor.W_use`。
8) sell_flag 与实际卖出一致性 → `sell_flag_*` 为 True 但 trades 无 sell → 可能被 calendar/hold/no-buy 阻断 → 查 `trades.csv:reason`。
9) 周末 gold 是否“估值连续但不交易” → 周末 `is_trading_gold=False` 且 `price_gold` 连续 → 若交易发生则逻辑错误 → 检查 `data.gold_trade_weekdays_only`。
10) t 与 Figure 8/9 窗口是否对齐 → `fig4_trades_window.png` 买卖点应出现在 t=1610..1760 / t=460..560 → 若偏移，检查 t 从 0 起算。

## 3. 稳健性批跑（命令 + 输出 + 怎么读）
- 参数网格：`python scripts/batch_param_grid.py --config config/reproduce_tables.yaml`
  - 输出：`outputs/param_grid/param_sweep.csv`
  - 关注最优组合是否为 `[12, 0.6, 0.89]`，趋势应接近论文。
- 费用敏感性：`python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml`
  - 输出：`outputs/fee_sweep/fee_sensitivity.csv`
  - 关注 fee 越高 final amount 越低的趋势。

## 4. 与原文不一致时的 Top3 排查优先级
1) 口径问题 → 证据：`notes.md:raw_range` 与 `end_date_mode` → 去看 `config/base.yaml:data.*`。
2) 信号尺度与 W 位置 → 证据：`profit_*` 分布偏小/偏大 → 去看 `config/base.yaml:thresholds/weight_factor` 与 `src/backtest.py`。
3) 状态机边界 → 证据：`state_no_buy` 或 `state_extreme` 长期占用 → 去看 `src/backtest.py` 的 no-buy/extreme 释放逻辑。

## 5. 可执行建模规范（基于 `论文1复刻memo.pdf` + `replication1.html`）
### 5.1 数据与日历
- 数据源：`BCHAIN-MKPRU.csv` 与 `LBMA-GOLD.csv`；日期格式 `%m/%d/%y`。
- 时间区间：`2016-09-11` 到 `2021-09-10`，按自然日创建主时间轴。
- BTC：用 `ffill` 补齐缺失。
- Gold：记录 `Gold_Tradable`（原始是否有价），估值用 `ffill/bfill`；Notebook 实际交易价使用 `ffill` 后的 `Price_Gold`（不按交易日阻断）。
- t 计数：`start_date` 当天为 t=0；Notebook 从 `t >= MA_window` 开始进入交易循环。

### 5.2 指标与买入信号
- MA：`A_t(n) = mean(P_{t-i}, i=0..n-1)`（Notebook 为简单均值）。
- Gradient：`G_t(n) = P_t - P_{t-n+1}`。
- Momentum 买入事件：`G_t > thr_grad` 且 `P_t - A_t > thr_ma_diff`（memo 为“事件交集”）。
- Mean Reversion 买入事件：`P_t < A_t * N`（N 为回买阈值）。
- 信号强度：`score = W * (mom_component + rev_component)`，
  - mom_component = `G_t`（仅当满足动量事件）
  - rev_component = `(A_t - P_t)^2`（仅当满足回归事件）
- 权重：`W = C / t^2`（t 从 1 开始计）。
- 买入触发：`score > score_threshold`（Notebook 使用 100）。

### 5.3 卖出与持有
- Holding：买入后至少持有 `T` 天（`t - last_buy_date_idx >= T`）。
- 卖出条件：`(1 - fee) * P - avg_cost > L`，满足即清仓卖出。
- 日内顺序：先卖后买。
- 仅做多（Long-only）。

### 5.4 仓位与交易执行
- Notebook 资金分配：每次触发买入则投入当前现金的 50%。
- 若同日 BTC 与 Gold 都触发，则两笔都会执行（各自基于当时现金）。
- 加权平均成本包含买入手续费。

### 5.5 映射到现有 `src/`
- 数据与日历：`src/dataio.py:load_price_data` + `config.data.*`。
- MA/Gradient/信号：`src/indicators.py:weighted_moving_average/price_gradient/compute_asset_signals`。
- 权重 W：`src/backtest.py`（当前仅用于比较/买入打分）。
- 买入执行与资金分配：`src/strategy.py:size_buy` 与 `src/backtest.py` 的 buy 分支。
- 卖出条件与持有：`src/strategy.py:should_sell`、`src/backtest.py` 的 hold 逻辑、`src/portfolio.py` 的 avg_cost。

## 6. 当前实现 vs `replication1.html` 差异清单（改动前必看）
### 6.1 信号
- MA：当前为加权 MA（`w_scheme=linear`），Notebook 为简单均值。
- Momentum 条件：当前 `thr_grad/thr_ma_diff` 默认 1000，Notebook 为 `>0`（并用 `score>100` 门槛）。
- Reversion 条件：当前 `P < MA`，Notebook 为 `P < MA * N`（N=0.6）。
- 信号强度：当前取 `max(mom, rev)`，Notebook 为 `mom + rev` 并全程乘 `W`。

### 6.2 仓位/买入
- 当前每日只选一个资产（`choose_buy_asset`），Notebook 可同时买 BTC+Gold。
- 当前买入规模按 `score/price` 映射，Notebook 固定投入 50% 现金。
- 当前 `W_C=1` 且默认只作用 gold；Notebook `C=100000` 且作用两资产。

### 6.3 卖出门槛
- 当前 `margin_L=0` 且若同日存在买入信号则跳过卖出；Notebook `L=10` 且不跳过。
- 当前手续费分资产（gold/btc），Notebook 单一费率 `0.02`。

### 6.4 日内顺序
- 两者均为先卖后买，但当前实现会因 `buy_signal` 抑制卖出，Notebook 不抑制。
- 当前从 t=0 开始计算信号，Notebook 从 `t>=MA_window` 开始回测循环。

### 6.5 交易日与数据对齐
- 当前 gold 仅交易日可交易（周末阻断），Notebook 用 ffill 后价格每日可交易。
- 当前 `end_date=2021-09-11` + `valuation_end`，Notebook 止于 `2021-09-10`。

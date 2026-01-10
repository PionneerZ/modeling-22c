> Deprecated: see docs/USAGE.md and docs/REPRODUCTION.md.

# 复现入口与排错指南

## 0. 这仓库复现什么
- 复现对象：Team #2229059 的黄金/比特币日交易策略。
- 目标产物：一次回测输出 + 参数网格表（4.3）+ 费用敏感性表（4.4）。
- 输出契约以 `docs/interface_contract.md` 为准，关键输出：`results_table.csv`、`trades.csv`、`key_metrics.json`、`figs/fig1_nav.png`~`figs/fig4_trades_window.png`、`outputs/param_grid/param_sweep.csv`、`outputs/fee_sweep/fee_sensitivity.csv`。

## 1. 30 秒跑通（最短路径）
```bash
python -m pip install -r requirements.txt
python scripts/run.py --config config/base.yaml
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml
python -m unittest discover -s tests
```
- `run.py`：输出在 `outputs/run_<id>/`，生成 `results_table.csv`、`trades.csv`、`key_metrics.json`、`figs/fig1_nav.png`~`figs/fig4_trades_window.png`、`notes.md`。
- `batch_param_grid.py`：输出在 `outputs/param_grid/param_sweep.csv`（含 T/N/E 组合）。
- `batch_sensitivity.py`：输出在 `outputs/fee_sweep/fee_sensitivity.csv`（含 fee 组合）。
- 单测：只校验逻辑一致性，不校验论文数值。

## 2. outputs 怎么读
| 文件 | 用途 | 写作/验证怎么用 |
| --- | --- | --- |
| `results_table.csv` | 每日净值/状态机轨迹 | 诊断策略是否按口径运行（state_*、hold_left_*、chosen_buy_asset、sell_flag_*、t）。 |
| `trades.csv` | 交易明细与阻断原因 | 解释交易逻辑（reason、signal_score），核对卖出/阻断是否符合状态机。 |
| `key_metrics.json` | 汇总指标 | 论文 Results 段落引用（final_nav、ROI、maxDD、trades_count）。 |
| `figs/*` | 图形输出 | Figure 4/8/9 对标（fig1/fig2/fig3/fig4）。 |

## 3. 数据准备与口径（精确说明）
- 数据文件：`problem/BCHAIN-MKPRU.csv`（Date, Value）与 `problem/LBMA-GOLD.csv`（Date, USD (PM)）。
- 配置入口：`config/base.yaml`，重点字段：
  - `data.gold_path` / `data.btc_path`
  - `data.column_map`（date/gold/btc 列映射）
  - `data.date_format`（当前为 `%m/%d/%y`）
  - `data.start_date` / `data.end_date` / `data.end_date_mode`
- 时间与索引：以 BTC 自然日为日历，`t` 从 `start_date` 当天开始计数，**t=0**。
- gold 缺失：周末与假日价格允许缺失，估值用 ffill/bfill 补齐，但 **不允许交易**（详见 `docs/assumption_ledger.md`）。
- 日内顺序：先检查 extreme/no-buy，若 extreme 生效只走 extreme 卖出并跳过其余流程；否则 **先卖后买**（来自 PDF 3.8）。
- 参数对应：`[T,N,E]` 对应持有天数、回买阈值、极端卖出阈值（对标 PDF 4.3/4.4）。

## 4. 常见错误与排查（可操作）
- 列找不到/日期解析失败 → 列名或 `date_format` 不匹配 → 看 `config/base.yaml:data.column_map` 与 CSV 表头 → 修正映射或格式。
- 报“数据区间不覆盖” → 原始数据缺天 → 看 `notes.md` 的 `raw_range` → 补齐数据或改 `start_date/end_date`。
- 输出为空/几乎无交易 → no-buy 长期阻断或阈值过高 → 看 `results_table.csv:state_no_buy` 与 `trades.csv:reason` → 调整 `no_buy.mode/no_buy_days` 或阈值。
- 交易次数爆炸/费用侵蚀 → 阈值过低或卖出过频 → 看 `trades_count` 与 `reason` → 调整 `thresholds.*` 或 `holding.hold_days_T`。
- 净值异常（负数/回撤>100%） → 数据 NaN 或估值连贯性问题 → 看 `results_table.csv:price_*` 是否 NaN → 检查数据缺失处理与 `gold_ffill_for_valuation`。
- 与论文 220486 差距极大 → 重点排查口径/信号尺度/状态机：
  1) 口径（区间与数据源）→ `config/base.yaml:data.*`、`notes.md:raw_range`；
  2) 信号尺度与 W → `config/base.yaml:thresholds/weight_factor`；
  3) 状态机边界 → `src/backtest.py` 中 extreme/no-buy/holding 顺序。

## 5. 给队友的快速入口
- 写作 A：看 `outputs/run_<id>/figs/*`、`outputs/run_<id>/key_metrics.json`、`outputs/param_grid/param_sweep.csv`、`outputs/fee_sweep/fee_sensitivity.csv`。
- 统计 C：先看 `results_table.csv` 的 `state_* / hold_left_* / chosen_buy_asset / sell_flag_* / t`，再看 `trades.csv` 的 `reason / signal_score`，优先跑 `batch_param_grid` 与 `batch_sensitivity`。

## 6. 复现对齐报告（当前版本）
### 6.1 数据与区间
- 数据文件：`problem/BCHAIN-MKPRU.csv`（Date, Value）与 `problem/LBMA-GOLD.csv`（Date, USD (PM)）。
- 覆盖范围：2016-09-11 ~ 2021-09-10（BTC 无 2021-09-11）。
- `end_date_mode=valuation_end`（交易截止 2021-09-10，估值到 2021-09-11）。
- `trade_end` 模式目前无法跑通（BTC 缺 2021-09-11）。

### 6.2 主结果对齐
- 本次 run_id：`run_20260107_172021`
- 本次 `final_nav`：29457.23
- 论文目标 `final amount`：220486
- 绝对误差：-191028.77
- 相对误差：-86.64%

### 6.3 表 4.3 逐行对比
| T | N | E | 论文 Final | 本次 Final | 差值 | 相对误差 |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 0.4 | 0.86 | 14023 | 5098.08 | -8924.92 | -63.64% |
| 5 | 0.4 | 0.89 | 24183 | 29435.36 | 5252.36 | 21.72% |
| 5 | 0.4 | 0.895 | 24183 | 28478.80 | 4295.80 | 17.76% |
| 5 | 0.6 | 0.86 | 117416 | 5098.08 | -112317.92 | -95.66% |
| 5 | 0.6 | 0.89 | 185291 | 29435.36 | -155855.64 | -84.11% |
| 5 | 0.6 | 0.895 | 169463 | 28478.80 | -140984.20 | -83.19% |
| 5 | 0.65 | 0.86 | 106137 | 5098.08 | -101038.92 | -95.20% |
| 5 | 0.65 | 0.89 | 157249 | 29435.36 | -127813.64 | -81.28% |
| 5 | 0.65 | 0.895 | 140421 | 28478.80 | -111942.20 | -79.72% |
| 6 | 0.4 | 0.86 | 13307 | 5098.08 | -8208.92 | -61.69% |
| 6 | 0.4 | 0.89 | 24183 | 29435.36 | 5252.36 | 21.72% |
| 6 | 0.4 | 0.895 | 24183 | 28478.80 | 4295.80 | 17.76% |
| 6 | 0.6 | 0.86 | 90167 | 5098.08 | -85068.92 | -94.35% |
| 6 | 0.6 | 0.89 | 181230 | 29435.36 | -151794.64 | -83.76% |
| 6 | 0.6 | 0.895 | 169178 | 28478.80 | -140699.20 | -83.17% |
| 6 | 0.65 | 0.86 | 82855 | 5098.08 | -77756.92 | -93.85% |
| 6 | 0.65 | 0.89 | 135476 | 29435.36 | -106040.64 | -78.27% |
| 6 | 0.65 | 0.895 | 123393 | 28478.80 | -94914.20 | -76.92% |
| 12 | 0.4 | 0.86 | 72670 | 7828.48 | -64841.52 | -89.23% |
| 12 | 0.4 | 0.89 | 24183 | 29457.23 | 5274.23 | 21.81% |
| 12 | 0.4 | 0.895 | 24183 | 26271.28 | 2088.28 | 8.64% |
| 12 | 0.6 | 0.86 | 72670 | 7828.48 | -64841.52 | -89.23% |
| 12 | 0.6 | 0.89 | 220486 | 29457.23 | -191028.77 | -86.64% |
| 12 | 0.6 | 0.895 | 178076 | 26271.28 | -151804.72 | -85.25% |
| 12 | 0.65 | 0.86 | 72670 | 7828.48 | -64841.52 | -89.23% |
| 12 | 0.65 | 0.89 | 175510 | 29457.23 | -146052.77 | -83.22% |
| 12 | 0.65 | 0.895 | 162657 | 26271.28 | -136385.72 | -83.85% |

### 6.4 表 4.4 逐行对比
| gold fee | btc fee | 论文 Final | 本次 Final | 差值 | 相对误差 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 0.02 | 220486 | 29457.23 | -191028.77 | -86.64% |
| 0.02 | 0.03 | 199945 | 24492.49 | -175452.51 | -87.75% |
| 0.03 | 0.05 | 156038 | 15685.23 | -140352.77 | -89.95% |
| 0.1 | 0.1 | 47452 | 5064.43 | -42387.57 | -89.33% |

### 6.5 结论与下一步
- 当前 **未能数值完全复现**，整体低于论文约 86%。
- 残留差距 Top3：
  1) 信号尺度与阈值归一化（`thresholds.*` 与 W 的作用位置）。
  2) no-buy/holding/extreme 的边界联动（是否应允许更快回买）。
  3) 买量映射函数（profit → 买入比例）的实现形式。
- 下一步最小改动建议：
  1) 将阈值改为“相对变化/百分比”，而不是绝对美元差值；
  2) 评估 W 是否应用于信号阈值而非买量；
  3) 用论文 Figure 8/9 的买卖点时序反推 no-buy 长度与 extreme 触发条件。





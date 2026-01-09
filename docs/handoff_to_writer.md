# 写作交接：Results / Discussion

## 0. A 需要交付什么（对齐论文写作）
- Results：净值曲线（Figure 4 对标）、回撤、仓位变化，以及两张表（4.3 参数网格、4.4 费用敏感性）。
- Discussion：假设与局限、费用敏感性趋势、为何与论文数值存在差距。

## 1. 证据索引（按图/表一对一）
| 证据 | 对应论文内容 | 输出路径 |
| --- | --- | --- |
| fig1_nav.png | Figure 4：净值走势与论文主结果对标 | `outputs/run_<id>/figs/fig1_nav.png` |
| fig2_drawdown.png | 回撤趋势（辅助说明风险） | `outputs/run_<id>/figs/fig2_drawdown.png` |
| fig3_positions.png | 现金/黄金/比特币仓位比例变化 | `outputs/run_<id>/figs/fig3_positions.png` |
| fig4_trades_window.png | Figure 8/9：区间买卖点（BTC t=1610..1760，Gold t=460..560） | `outputs/run_<id>/figs/fig4_trades_window.png` |
| param_sweep.csv | Table 4.3 参数网格（[T,N,E] 含义：持有天数/回买阈值/极端卖出阈值） | `outputs/param_grid/param_sweep.csv` |
| fee_sensitivity.csv | Table 4.4 费用敏感性 | `outputs/fee_sweep/fee_sensitivity.csv` |

> 当前默认对齐运行：`outputs/run_20260107_172021/`。

## 2. 指标引用（字段名必须精确）
- `key_metrics.json`：
  - `final_nav`（最终金额）
  - `ROI`（收益率）
  - `maxDD`（最大回撤）
  - `trades_count`（交易次数）
- `trades.csv`：
  - `reason`：说明交易触发原因（`buy_momentum`/`buy_reversion`/`sell_signal`/`extreme_sell`/`blocked_by_*`）。
  - `signal_score`：用于解释“信号强度”。

可直接写入的示例句：
1) “在 `reason=extreme_sell` 的交易中，模型在价格回撤到 `E*max_price` 时触发卖出，避免继续回撤。”
2) “`signal_score` 在 BTC 上显著高于 gold 的区间，对应 Figure 8 中更密集的买卖点。”
3) “当 `blocked_by_calendar` 出现时，说明该日为 gold 非交易日，仅估值不交易。”

## 3. A 要改图/换区间/重跑的操作
1) 换数据文件：将原文数据放入 `problem/`（或 `data/`），更新 `config/base.yaml` 的 `data.gold_path`/`data.btc_path` 与 `data.column_map`，再跑 `python scripts/run.py --config config/base.yaml`。
2) 换手续费：修改 `config/reproduce_tables.yaml` 的 `fee_sensitivity.fee_pairs` 或 `config/base.yaml` 的 `fees.*`，再跑 `python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml`。
3) 补某段区间的买卖点图：改 `config/base.yaml` 的 `plots.window_btc_t` / `plots.window_gold_t`，再跑 `run.py` 生成 `fig4_trades_window.png`。

## 4. 写作必须提示的假设/差异（可直接粘贴）
- “由于原文未明确买量映射函数，本文实现采用 `score/price` 的线性映射并限制 `buy_cap`，该选择可能降低高波动区间的资金投入强度。”
- “BTC 数据在 2021-09-11 缺失，本文采用 `end_date_mode=valuation_end` 将估值延伸到 9/11，但交易截止在 9/10，该处理可能影响最终金额。”
- “论文未明确 no-buy 的释放规则，本文以 `no_buy_days=30` 进行时间阻断，可能导致回买节奏与原文不完全一致。”
- “噪声归一化因子 W 的作用位置未完全明确，本文采用‘对 gold 进行下权重’的实现以体现 BTC 更高权重。”
- “t 从 0 计数（start_date 当天为 t=0），并在 Figure 8/9 区间中使用该口径。”

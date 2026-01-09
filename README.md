# 2229059 Gold & Bitcoin Day Trading 复现工程

这份仓库是我负责的“工程集成 + 可复现交付”部分，用于复现 Team #2229059 的黄金/比特币日交易策略，并产出与论文一致的结果结构（回测输出 + 参数网格表 + 费用敏感性表）。

## 一句话目标
- 跑通 `run.py` 得到一次回测输出；
- 跑通两张表（4.3 参数网格、4.4 费用敏感性）；
- 输出字段与口径统一，队友可以直接写作/诊断。

## 30 秒上手
```bash
python -m pip install -r requirements.txt
python scripts/run.py --config config/base.yaml
python scripts/batch_param_grid.py --config config/reproduce_tables.yaml
python scripts/batch_sensitivity.py --config config/reproduce_tables.yaml
python -m unittest discover -s tests
```
- 单次回测输出：`outputs/run_<id>/`
- 参数网格表：`outputs/param_grid/param_sweep.csv`
- 费用敏感性表：`outputs/fee_sweep/fee_sensitivity.csv`

## 产物与对接
- 关键输出契约以 `docs/interface_contract.md` 为准。
- 一次回测应生成：`results_table.csv`、`trades.csv`、`key_metrics.json`、`figs/fig1_nav.png`、`figs/fig2_drawdown.png`、`figs/fig3_positions.png`、`notes.md`。
- 参数表与费用表口径来源：`docs/spec_extracted.md`（含页码）。

## 口径与假设（必须对齐）
- 模型公式与状态机顺序：`docs/spec_extracted.md`。
- 输出字段与命名：`docs/interface_contract.md`。
- 原文未明处的实现补齐：`docs/assumption_ledger.md`。

## 数据准备
- 数据放在 `data/`，默认示例为 `data/sample_prices.csv`。
- 配置入口：`config/base.yaml`，重点字段：
  - `data.path`, `data.date_col`, `data.gold_col`, `data.btc_col`, `data.date_format`
  - `fees.*`, `holding.hold_days_T`, `no_buy.rebuy_pct_N`, `extreme.extreme_sell_pct_E`
- gold 周末缺失：`execution.gold_ffill=true` 仅用于估值，周末不交易（见 `docs/assumption_ledger.md`）。

## 队友入口
- 写作队友 A：直接看 `docs/handoff_to_writer.md`。
- 统计队友 C：直接看 `docs/handoff_to_modeler.md`。
- 全队入口：`docs/reproduce.md`（跑通 + 输出如何读 + 排错）。

## 目录结构（简版）
- `src/`：核心逻辑（加载数据、信号、策略、回测、指标、画图）
- `scripts/`：执行入口（单次回测 / 参数网格 / 费用敏感性）
- `config/`：配置
- `data/`：数据
- `outputs/`：生成产物
- `docs/`：口径与交接文档
- `tests/`：单元测试

> 注意：不要改动输出文件名/字段名，否则队友无法对接。

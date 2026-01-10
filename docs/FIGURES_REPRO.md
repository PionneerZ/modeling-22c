# Figures Reproduction

This pipeline regenerates all paper figures (except Figure 8/9) and produces side-by-side comparisons.

## One-command reproduction
```bash
python scripts/reproduce_all_figures.py --config config/fixpack.yaml --outdir outputs/run_fixpack
```
This command will:
- extract paper figures to `outputs/paper_figures/` (if missing)
- generate `ours` figures
- copy `paper` figures
- render `compare` side-by-side images
- write a per-run report to `outputs/run_fixpack/figures/report.md`

## Outputs
- `outputs/paper_figures/figures_manifest.csv`
- `outputs/paper_figures/fig1_paper.png` ... `fig7_paper.png`
- `outputs/<run>/figures/ours/figX.png`
- `outputs/<run>/figures/paper/figX.png`
- `outputs/<run>/figures/compare/figX_compare.png`
- `outputs/<run>/figures/report.md`

## Paper figure extraction
If you want to re-extract with custom bounding boxes:
```bash
python scripts/extract_paper_figures.py --config config/figures_bboxes.yaml --outdir outputs/paper_figures
```
Edit `config/figures_bboxes.yaml` to adjust page/bbox values.

## Window figures (Figure 8/9)
Window plots accept either t-window or date-window configuration:
- `plots.window_btc_t` / `plots.window_gold_t`
- `plots.window_btc_date` / `plots.window_gold_date`

The plotting logic prefers t-window when both are provided.

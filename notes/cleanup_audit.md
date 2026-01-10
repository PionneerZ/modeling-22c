# Cleanup Audit (Phase A)

Scope: root reference files and specified PDFs.

## Root reference files
- paper_ascii.pdf
  - referenced by: config/figures_bboxes.yaml
  - decision: delete after config points to assets/papers/paper.pdf
- paper_text.txt
  - referenced by: none found
  - decision: delete (intermediate text extract)
- memo_ascii.pdf
  - referenced by: none found
  - decision: delete (intermediate extract)
- memo_text.txt
  - referenced by: none found
  - decision: delete (intermediate extract)
- replication1.html
  - referenced by: AGENTS.md, PROJECT_GOALS.md, docs/handoff_to_modeler.md
  - decision: delete (intermediate notebook capture); update docs to remove references
- replication_buy_code.html.txt
  - referenced by: none found
  - decision: delete (intermediate extract)
- replication_buy_lines.txt
  - referenced by: none found
  - decision: delete (intermediate extract)
- replication_buy_snippet.txt
  - referenced by: none found
  - decision: delete (intermediate extract)
- replication_last_buy.txt
  - referenced by: none found
  - decision: delete (intermediate extract)
- replication_snippet.txt
  - referenced by: none found
  - decision: delete (intermediate extract)

## Source PDFs
- 无水印-2229059.pdf
  - referenced by: AGENTS.md, PROJECT_GOALS.md
  - decision: move to assets/papers/paper.pdf and update references
- 论文1复刻memo.pdf
  - referenced by: AGENTS.md, PROJECT_GOALS.md, docs/handoff_to_modeler.md
  - decision: move to assets/papers/memo.pdf and update references

## Other PDFs
- problem/2022_MCM_Problem_C.pdf
  - referenced by: none found
  - decision: delete (unrelated to this project)

## Experiments cleanup
- experiments/ablation_matrix.py
  - decision: delete (one-off analysis, not required for reproducible runs)
- experiments/auto_calibrate.py
  - decision: delete (sweep helper, not required for PR)
- experiments/calibrate_min_delta.py
  - decision: delete (calibration helper, not required for PR)
- experiments/calibrate_to_paper.py
  - decision: delete (large search script, not required for PR)
- experiments/window_debug.py
  - decision: delete (overlaps with run_with_debug and figure window tooling)

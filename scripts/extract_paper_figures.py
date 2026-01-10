from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys

import fitz

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_yaml


_FIGURE_RE = re.compile(r"^Figure\s+(\d+)\s*:\s*(.+)", re.IGNORECASE)


def _find_pdf(root: Path) -> Path:
    candidates = list(root.rglob("*.pdf"))
    if not candidates:
        raise FileNotFoundError("no pdf files found under repo root")

    def score(path: Path) -> int:
        name = path.name.lower()
        score_val = 0
        if "paper" in name:
            score_val += 5
        if "ascii" in name:
            score_val += 3
        if "2229059" in name or "22c" in name:
            score_val += 2
        if "memo" in name:
            score_val -= 2
        return score_val

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[0]


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _caption_blocks(page) -> list[dict]:
    blocks = page.get_text("dict")["blocks"]
    captions = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        text = "".join(
            span["text"] for line in block.get("lines", []) for span in line.get("spans", [])
        )
        text = _normalize_text(text)
        if not text.lower().startswith("figure "):
            continue
        match = _FIGURE_RE.match(text)
        if not match:
            continue
        captions.append(
            {
                "figure_id": int(match.group(1)),
                "title": match.group(2).strip(),
                "bbox": block.get("bbox"),
                "text": text,
            }
        )
    return captions


def _assign_images_to_captions(page, captions: list[dict]) -> dict[int, list[list[float]]]:
    blocks = page.get_text("dict")["blocks"]
    images = [b for b in blocks if b.get("type") == 1]
    if not images or not captions:
        return {}

    captions = sorted(captions, key=lambda c: c["bbox"][1])
    assigned: dict[int, list[list[float]]] = {}
    prev_y1 = 0.0
    for caption in captions:
        cap_y0 = float(caption["bbox"][1])
        cap_y1 = float(caption["bbox"][3])
        picked = []
        for img in images:
            x0, y0, x1, y1 = img["bbox"]
            if y1 <= cap_y0 + 5 and y0 >= prev_y1 - 5:
                picked.append([float(x0), float(y0), float(x1), float(y1)])
        if picked:
            assigned[int(caption["figure_id"])] = picked
        prev_y1 = cap_y1
    return assigned


def _expand_bbox(bbox: list[float], margin: float) -> list[float]:
    x0, y0, x1, y1 = bbox
    return [x0 - margin, y0 - margin, x1 + margin, y1 + margin]


def _bbox_union(bboxes: list[list[float]]) -> list[float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="path to paper pdf (optional)")
    parser.add_argument("--config", default="config/figures_bboxes.yaml")
    parser.add_argument("--outdir", default="outputs/paper_figures")
    parser.add_argument("--dpi", type=int, default=None)
    parser.add_argument("--exclude", default=None, help="comma-separated figure ids to exclude")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}

    pdf_path = args.pdf or cfg.get("pdf_path")
    if not pdf_path:
        pdf_path = str(_find_pdf(root))
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        pdf_path = root / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"pdf not found: {pdf_path}")

    dpi = args.dpi or int(cfg.get("dpi", 200))

    exclude = set(int(x) for x in (cfg.get("exclude_figures") or []))
    if args.exclude:
        exclude |= {int(x.strip()) for x in args.exclude.split(",") if x.strip()}

    manual_figs = cfg.get("figures") or {}

    out_dir = ensure_dir(args.outdir)
    pages_dir = ensure_dir(out_dir / "pages")

    figure_specs: dict[int, dict] = {}
    pages_with_figures: set[int] = set()

    for fig_id_str, spec in manual_figs.items():
        fig_id = int(fig_id_str)
        if fig_id in exclude:
            continue
        page = int(spec["page"])
        bboxes = [[float(v) for v in box] for box in spec.get("bboxes", [])]
        if not bboxes:
            continue
        figure_specs[fig_id] = {
            "page": page,
            "bboxes": bboxes,
            "title": spec.get("title", ""),
            "source": "manual",
        }
        pages_with_figures.add(page)

    auto_missing = []
    with fitz.open(pdf_path) as doc:
        for page_index in range(doc.page_count):
            page_num = page_index + 1
            page = doc[page_index]
            captions = _caption_blocks(page)
            if not captions:
                continue
            pages_with_figures.add(page_num)

            img_map = _assign_images_to_captions(page, captions)
            for caption in captions:
                fig_id = int(caption["figure_id"])
                if fig_id in exclude or fig_id in figure_specs:
                    continue
                bboxes = img_map.get(fig_id)
                if not bboxes:
                    auto_missing.append(fig_id)
                    continue
                figure_specs[fig_id] = {
                    "page": page_num,
                    "bboxes": bboxes,
                    "title": caption.get("title", ""),
                    "caption": caption.get("text", ""),
                    "source": "auto",
                }

        scale = dpi / 72.0
        for page_num in sorted(pages_with_figures):
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            page_path = pages_dir / f"page_{page_num:02d}.png"
            pix.save(page_path.as_posix())

        manifest_rows = []
        for fig_id in sorted(figure_specs):
            spec = figure_specs[fig_id]
            page_num = int(spec["page"])
            page = doc[page_num - 1]
            union_bbox = _bbox_union(spec["bboxes"])
            union_bbox = _expand_bbox(union_bbox, margin=1.0)
            rect = fitz.Rect(union_bbox)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect)
            out_path = out_dir / f"fig{fig_id}_paper.png"
            pix.save(out_path.as_posix())

            manifest_rows.append(
                {
                    "figure_id": fig_id,
                    "page": page_num,
                    "title": spec.get("title", ""),
                    "caption": spec.get("caption", ""),
                    "bbox_list": json.dumps(spec["bboxes"]),
                    "paper_png": str(out_path),
                    "source": spec.get("source", ""),
                }
            )

    if auto_missing:
        print(f"[extract] warning: missing bboxes for figures: {sorted(set(auto_missing))}")

    manifest_path = out_dir / "figures_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure_id", "page", "title", "caption", "bbox_list", "paper_png", "source"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[extract] pdf: {pdf_path}")
    print(f"[extract] pages: {len(pages_with_figures)} saved to {pages_dir}")
    print(f"[extract] figures: {len(manifest_rows)} manifest {manifest_path}")


if __name__ == "__main__":
    main()

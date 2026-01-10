from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("run_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--keep", type=int, default=5, help="keep newest N run_* folders")
    parser.add_argument("--prune-others", action="store_true", help="also remove non-run folders")
    parser.add_argument("--drop-paper", action="store_true", help="remove outputs/paper_figures")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    if not out_dir.exists():
        print(f"outputs dir not found: {out_dir}")
        return

    skip = {"_milestones"}
    if not args.drop_paper:
        skip.add("paper_figures")

    run_dirs = [p for p in out_dir.iterdir() if _is_run_dir(p)]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    keep_set = set(run_dirs[: args.keep])
    to_remove = [p for p in run_dirs if p not in keep_set]

    if args.prune_others:
        for p in out_dir.iterdir():
            if p.name in skip or p in run_dirs:
                continue
            to_remove.append(p)

    if args.dry_run:
        print("would remove:")
        for p in to_remove:
            print(f"- {p}")
        return

    for p in to_remove:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    print(f"removed {len(to_remove)} item(s) from {out_dir}")


if __name__ == "__main__":
    main()

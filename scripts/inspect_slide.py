"""Look before you embed: which tiles would the pipeline keep for one slide?

    python scripts/inspect_slide.py configs/embed.yaml /path/to/slide.svs [--out audit.html]

Runs the exact tiling + filters from the embedding config without encoding anything,
prints the keep/drop funnel per filter, and writes an interactive HTML overlay (kept tiles
green, dropped tiles coloured by the filter that dropped them, sliders to retune the
thresholds live). Open the HTML in a browser.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wsi_patching import visualize_audit

from wsisurv.config import load_config
from wsisurv.patching.pipeline import build_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("slide")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    p = build_pipeline([args.slide], cfg)
    p.dry_run(num_workers=1)
    p.print_audit()
    out = args.out or f"{Path(args.slide).stem}_audit.html"
    visualize_audit(p, args.slide, save_path=out)
    print(f"overlay -> {out}")


if __name__ == "__main__":
    main()

"""Stage 1: every slide in the cohort table -> <output_dir>/<slide_id>.pt

    python scripts/embed_slides.py configs/embed.yaml

Each .pt holds {embeddings (N, D) float16, coords (N, 2) in the target_mpp grid,
slide_id, encoder, mpp, tile_size}. Re-running skips slides that are done, so a killed
job simply resumes. A manifest CSV lists every slide with its status.
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from wsisurv.config import load_config
from wsisurv.data.cohort import load_slide_table
from wsisurv.encoders.foundation import FoundationEncoder
from wsisurv.patching.pipeline import build_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed")


def resolve(cfg) -> pd.DataFrame:
    slides = load_slide_table(cfg["cohort_table"])
    root, out = Path(cfg.get("root_data_dir", ".")), Path(cfg["output_dir"])
    slides["wsi_path"] = [str(p if Path(p).is_absolute() else root / p) for p in slides["wsi_path"]]
    slides["status"] = "todo"
    slides.loc[[not Path(p).exists() for p in slides["wsi_path"]], "status"] = "missing_wsi"
    slides.loc[[(out / f"{s}.pt").exists() for s in slides["slide_id"]], "status"] = "done"
    # wsi_patching names slides by file stem; two files with one stem would be merged
    stems = slides["wsi_path"].map(lambda p: Path(p).stem)
    if stems.duplicated().any():
        raise ValueError(f"WSI file stems collide: {stems[stems.duplicated()].tolist()[:5]}")
    log.info("Slides: %s", slides["status"].value_counts().to_dict())
    for _, r in slides[slides.status == "missing_wsi"].head(10).iterrows():
        log.warning("missing WSI for %s: %s", r.slide_id, r.wsi_path)
    return slides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. --set use_gpu=false")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    slides = resolve(cfg)
    slides.to_csv(out / "manifest.csv", index=False)
    todo = slides[slides.status == "todo"]
    if todo.empty:
        log.info("Nothing to do.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FoundationEncoder(cfg["model_name"], cfg.get("weights_path")).to(device).eval()
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    n = cfg.get("slides_per_batch", 4)
    for start in range(0, len(todo), n):
        chunk = todo.iloc[start:start + n]
        slide_id_by_stem = {Path(p).stem: s for p, s in zip(chunk.wsi_path, chunk.slide_id)}
        acc = defaultdict(lambda: {"emb": [], "xy": []})

        pipeline = build_pipeline(chunk.wsi_path.tolist(), cfg)
        # Batches of different slides arrive interleaved; wsi_id (file stem) says whose they are.
        for wsi_id, tiles, coords, _meta in pipeline.stream(num_workers=cfg.get("num_workers", 4),
                                                            gracefully_handle_producer_errors=True):
            if tiles.shape[0] == 0:
                continue
            with torch.autocast(**amp):
                emb = encoder(tiles.to(device))
            acc[slide_id_by_stem[wsi_id]]["emb"].append(emb.float().cpu())
            acc[slide_id_by_stem[wsi_id]]["xy"].append(coords.cpu())

        for slide_id in chunk.slide_id:
            if slide_id not in acc:
                log.warning("%s produced no tiles (failed to open, or everything filtered)", slide_id)
                continue
            emb = torch.cat(acc[slide_id]["emb"]).half()
            torch.save({"embeddings": emb, "coords": torch.cat(acc[slide_id]["xy"]),
                        "slide_id": slide_id, "encoder": cfg["model_name"],
                        "mpp": cfg["target_mpp"], "tile_size": cfg["tile_size"]},
                       out / f"{slide_id}.pt")
            log.info("saved %s: %d tiles x %d", slide_id, *emb.shape)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

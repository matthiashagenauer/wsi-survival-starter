"""WSI -> tissue tiles, with the `wsi_patching` library (github.com/amspath/wsi-patching-pipeline).

A pipeline is a chain of stages ending in a writer:

    WSIGrid            which slides, at what resolution (here: exactly `target_mpp` µm/pixel)
    PatchExtractor     plans the tile grid, reads regions, cuts + batches tiles
    *Filter            drop background / pen marks / slide edges
    TorchStreamWriter  yields (wsi_id, tiles[N,3,H,W] float 0–255, coords[N,2], metadata)

`fallback_mode="resample"` reads the finest pyramid level at or below the target and
resizes, so a 40x (0.25 mpp) and a 20x (0.5 mpp) scanner give tiles of the same physical
size. With "nearest" they silently would not — and the model learns the scanner.
"""
from __future__ import annotations

from wsi_patching import (
    LowContrastBackgroundFilter,
    OtsuFilter,
    PatchExtractor,
    PenArtifactFilter,
    RemoveEdgeTiles,
    TorchStreamWriter,
    WSIGrid,
)


def build_pipeline(slides: list[str], cfg: dict):
    """Tiles at `target_mpp`, background dropped. Every knob is in the embedding config."""
    f = cfg.get("filters", {})
    p = (WSIGrid(slides=slides, resolution=cfg["target_mpp"], unit="mpp",
                 use_gpu=cfg.get("use_gpu", True), fallback_mode="resample")
         .then(PatchExtractor(tile_size=cfg["tile_size"], stride=cfg["tile_size"],
                              max_batch_size=cfg.get("max_batch_size", 64))))
    if f.get("low_contrast_range", 0.2) is not None:
        p = p.then(LowContrastBackgroundFilter(range_threshold=f.get("low_contrast_range", 0.2)))
    if f.get("otsu_min_tissue_fraction") is not None:
        p = p.then(OtsuFilter(min_tissue_fraction=f["otsu_min_tissue_fraction"]))
    if f.get("pen_filter", False):
        p = p.then(PenArtifactFilter())
    if f.get("remove_edge_depth"):
        p = p.then(RemoveEdgeTiles(depth=f["remove_edge_depth"]))
    return p.to(TorchStreamWriter(layout="NCHW"))

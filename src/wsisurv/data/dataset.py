"""Patient-level bags of tile embeddings.

A patient's bag is the concatenation of the tiles of all their slides. Training samples
a fixed number of tiles per bag (`n_tiles`) so bags can be batched — the Cox loss needs
several patients per batch. Evaluation uses every tile, one patient at a time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_slide_embeddings(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["embeddings"].float()


def discretize_times(time: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index of each time; edges = inner cut points (len n_bins-1)."""
    return np.searchsorted(edges, time, side="right")


def make_bin_edges(train_time: np.ndarray, train_event: np.ndarray, n_bins: int) -> np.ndarray:
    """Inner cut points at quantiles of the *observed event* times of the training fold
    (so each bin holds ~the same number of events). Computed per fold — never on the test set."""
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(train_time[train_event == 1], q)


class PatientBagDataset(Dataset):
    def __init__(self, patients: pd.DataFrame, embedding_dir: str | Path, n_tiles: int | None = None,
                 bin_edges: np.ndarray | None = None, cache: bool = True, seed: int = 0):
        self.patients = patients
        self.ids = patients.index.tolist()
        self.embedding_dir = Path(embedding_dir)
        self.n_tiles = n_tiles
        self.cache = {} if cache else None
        self.rng = np.random.default_rng(seed)
        self.time = patients["time"].to_numpy(dtype=np.float32)
        self.event = patients["event"].to_numpy(dtype=np.float32)
        self.bins = (discretize_times(self.time, bin_edges) if bin_edges is not None
                     else np.zeros(len(self.ids), dtype=np.int64))

    def __len__(self):
        return len(self.ids)

    def _bag(self, pid: str) -> torch.Tensor:
        if self.cache is not None and pid in self.cache:
            return self.cache[pid]
        slides = self.patients.at[pid, "slide_ids"]
        bag = torch.cat([load_slide_embeddings(self.embedding_dir / f"{s}.pt") for s in slides])
        if self.cache is not None:
            self.cache[pid] = bag
        return bag

    def __getitem__(self, i):
        pid = self.ids[i]
        bag = self._bag(pid)
        if self.n_tiles is not None:
            # with replacement only when the bag is smaller than n_tiles
            idx = self.rng.choice(len(bag), self.n_tiles, replace=len(bag) < self.n_tiles)
            bag = bag[torch.from_numpy(idx)]
        return {
            "features": bag,
            "time": torch.tensor(self.time[i]),
            "event": torch.tensor(self.event[i]),
            "bin": torch.tensor(self.bins[i], dtype=torch.long),
            "patient_id": pid,
        }

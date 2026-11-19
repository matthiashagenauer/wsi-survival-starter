"""Patient-level cross-validation splits.

Two rules that matter more than any model choice:
  1. Split by patient, never by slide — slides of one patient are near-duplicates.
  2. Stratify by event (and by site/cohort if you have several), so every test fold
     contains enough events to rank. A fold with 3 events gives a meaningless C-index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Fold:
    k: int
    train: list[str]
    val: list[str]   # empty when val_fraction == 0
    test: list[str]


def _strata(patients: pd.DataFrame, stratify_on: list[str]) -> np.ndarray:
    return patients[stratify_on].astype(str).agg("|".join, axis=1).to_numpy()


def patient_folds(patients: pd.DataFrame, n_folds: int = 5, seed: int = 0,
                  stratify_on: list[str] | None = None, val_fraction: float = 0.0) -> list[Fold]:
    stratify_on = stratify_on or ["event"]
    ids = patients.index.to_numpy()
    strata = _strata(patients, stratify_on)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for k, (tr, te) in enumerate(skf.split(ids, strata)):
        train, val = ids[tr], np.array([], dtype=ids.dtype)
        if val_fraction > 0:
            train, val = train_test_split(train, test_size=val_fraction, random_state=seed,
                                          stratify=strata[tr])
        folds.append(Fold(k, list(train), list(val), list(ids[te])))
        assert not set(train) & set(ids[te]) and not set(val) & set(ids[te])
    return folds

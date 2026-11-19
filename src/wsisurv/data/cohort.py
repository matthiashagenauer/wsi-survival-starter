"""The cohort table: one row per slide, survival per patient.

Required columns
    patient_id  – who the slide belongs to. Splits, labels and predictions are per patient.
    slide_id    – unique, filename-safe; names the embedding file `<slide_id>.pt`.
    time        – follow-up time (one unit for the whole table, e.g. months).
    event       – 1 = event observed (e.g. death), 0 = censored (alive at last follow-up).
Optional
    wsi_path    – path to the slide file, absolute or relative to `root_data_dir`
                  (only needed for embedding).
    any other column (age, stage, cohort, ...) is carried along and can be used as a
    clinical covariate or as a stratification variable for the splits.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REQUIRED = ("patient_id", "slide_id", "time", "event")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class CohortError(ValueError):
    pass


def load_slide_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise CohortError(f"{path}: missing columns {missing}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)

    bad = df.loc[~df["slide_id"].str.match(_SAFE_ID), "slide_id"].tolist()
    if bad:
        raise CohortError(f"slide_id must match {_SAFE_ID.pattern} (used as a filename): {bad[:5]}")
    dup = df.loc[df["slide_id"].duplicated(), "slide_id"].tolist()
    if dup:
        raise CohortError(f"duplicate slide_id: {dup[:5]}")

    df["event"] = df["event"].astype(int)
    if not df["event"].isin([0, 1]).all():
        raise CohortError("event must be 0 (censored) or 1 (event)")
    if (df["time"] < 0).any() or df["time"].isna().any():
        raise CohortError("time must be present and non-negative")
    return df


def patient_table(slides: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient. Survival must agree across a patient's slides."""
    per_patient = slides.groupby("patient_id")
    conflicting = per_patient[["time", "event"]].nunique().gt(1).any(axis=1)
    if conflicting.any():
        raise CohortError(f"patients with conflicting time/event across slides: "
                          f"{conflicting[conflicting].index.tolist()[:5]}")

    extra = [c for c in slides.columns if c not in {"patient_id", "slide_id", "wsi_path"}]
    patients = per_patient[extra].first()
    patients["slide_ids"] = per_patient["slide_id"].agg(list)
    return patients


def keep_embedded(patients: pd.DataFrame, embedding_dir: str | Path) -> pd.DataFrame:
    """Drop slides without a `.pt`, then patients without any slide. Says who was dropped."""
    embedding_dir = Path(embedding_dir)
    have = {p.stem for p in embedding_dir.glob("*.pt")}
    patients = patients.copy()
    patients["slide_ids"] = patients["slide_ids"].apply(lambda ids: [s for s in ids if s in have])
    empty = patients["slide_ids"].str.len() == 0
    if empty.any():
        log.warning("%d/%d patients have no embedded slide and are dropped, e.g. %s",
                    empty.sum(), len(patients), patients.index[empty][:5].tolist())
    patients = patients[~empty]
    log.info("Cohort: %d patients, %d events (%.0f%%), median follow-up %.1f",
             len(patients), patients["event"].sum(), 100 * patients["event"].mean(),
             patients["time"].median())
    return patients

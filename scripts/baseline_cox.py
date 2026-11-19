"""Baselines on the *same* folds as the MIL model. Run these first.

    python scripts/baseline_cox.py --config configs/baseline_cox.yaml

  * clinical   – ridge Cox on clinical covariates only (if `clinical_covariates` is set).
                 The bar a WSI model has to clear to be clinically interesting.
  * mean_pool  – each patient = mean of all tile embeddings -> standardise -> PCA -> ridge Cox.
                 Often within a few C-index points of attention MIL.
  * both       – mean_pool + clinical: does the slide add information beyond the clinic?
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from wsisurv.config import load_config
from wsisurv.data.cohort import keep_embedded, load_slide_table, patient_table
from wsisurv.data.dataset import load_slide_embeddings
from wsisurv.data.splits import patient_folds
from wsisurv.evaluation import summarize
from wsisurv.metrics import as_surv, eval_times, survival_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def mean_embeddings(patients, embedding_dir) -> np.ndarray:
    return np.stack([torch.cat([load_slide_embeddings(Path(embedding_dir) / f"{s}.pt") for s in ids]).mean(0).numpy()
                     for ids in patients["slide_ids"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]

    patients = keep_embedded(patient_table(load_slide_table(cfg["cohort_table"])), cfg["embedding_dir"])
    folds = patient_folds(patients, cfg["n_folds"], cfg["seed"], cfg.get("stratify_on"),
                          cfg.get("val_fraction", 0.0))
    emb = pd.DataFrame(mean_embeddings(patients, cfg["embedding_dir"]), index=patients.index)
    clin = cfg.get("clinical_covariates") or []
    n_pc = cfg.get("pca_components", 32)

    def features(name, df):
        parts = []
        if name in {"mean_pool", "both"}:
            parts.append(emb.loc[df.index].add_prefix("e"))
        if name in {"clinical", "both"}:
            parts.append(df[clin].astype(float))
        return pd.concat(parts, axis=1).to_numpy()

    variants = ["mean_pool"] + (["clinical", "both"] if clin else [])
    for name in variants:
        out_dir = run_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = []
        for fold in folds:
            tr = patients.loc[fold.train + fold.val]   # no early stopping here, use all non-test
            te = patients.loc[fold.test]
            steps = [StandardScaler()]
            if name != "clinical":
                steps.append(PCA(n_components=min(n_pc, len(tr) - 1), random_state=0))
            steps.append(CoxPHSurvivalAnalysis(alpha=cfg.get("ridge_alpha", 1.0)))
            model = make_pipeline(*steps).fit(features(name, tr), as_surv(tr["time"], tr["event"]))

            x_te = features(name, te)
            times = eval_times(tr["time"].to_numpy(), tr["event"].to_numpy(),
                               te["time"].to_numpy(), te["event"].to_numpy())
            surv = np.stack([fn(times) for fn in model.predict_survival_function(x_te)])
            m = survival_metrics(tr["time"], tr["event"], te["time"], te["event"], model.predict(x_te),
                                 surv_at_times=surv, times=times)
            metrics.append({**m, "fold": fold.k})
        summarize(metrics, out_dir, title=f"baseline: {name}")


if __name__ == "__main__":
    main()

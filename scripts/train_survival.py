"""Stage 2: k-fold cross-validated MIL survival model.

    python scripts/train_survival.py --config configs/survival_abmil.yaml
    python scripts/train_survival.py --config configs/survival_abmil.yaml --set loss=cox batch_size=32

Writes runs/<run_name>/: fold_<k>/{model.pt, history.csv, test_predictions.csv},
fold_metrics.csv, summary.{csv,txt}, km_risk_groups.png, config.yaml.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from wsisurv.config import load_config
from wsisurv.data.cohort import keep_embedded, load_slide_table, patient_table
from wsisurv.data.splits import patient_folds
from wsisurv.evaluation import km_plot, summarize
from wsisurv.training.trainer import train_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="override config keys: key=value ...")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="run only these folds")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    patients = keep_embedded(patient_table(load_slide_table(cfg["cohort_table"])), cfg["embedding_dir"])
    folds = patient_folds(patients, cfg["n_folds"], cfg["seed"], cfg.get("stratify_on"),
                          cfg.get("val_fraction", 0.0))

    metrics, preds = [], []
    for fold in folds:
        if args.folds is not None and fold.k not in args.folds:
            continue
        m, p = train_fold(cfg, patients, fold, cfg["embedding_dir"], run_dir / f"fold_{fold.k}")
        metrics.append(m)
        preds.append(p)

    preds = pd.concat(preds)
    preds.to_csv(run_dir / "oof_predictions.csv", index=False)
    summarize(metrics, run_dir, title=cfg["run_name"])
    p = km_plot(preds, run_dir / "km_risk_groups.png", title=cfg["run_name"])
    print(f"log-rank p (pooled out-of-fold risk groups) = {p:.3g}")


if __name__ == "__main__":
    main()

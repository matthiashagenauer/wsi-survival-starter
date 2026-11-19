"""End-to-end on a tiny synthetic cohort: generator -> CV training (both losses) -> metrics."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsisurv.metrics import survival_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_perfect_and_reversed_ranking():
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20, dtype=int)
    event[::4] = 0
    assert survival_metrics(time, event, time, event, -time)["c_index"] == pytest.approx(1.0)
    assert survival_metrics(time, event, time, event, time)["c_index"] == pytest.approx(0.0)


@pytest.mark.parametrize("loss", ["nll", "cox"])
def test_cross_validation_runs(tmp_path, loss):
    data = tmp_path / "syn"
    subprocess.run([sys.executable, ROOT / "scripts/make_synthetic_cohort.py", "--out", data,
                    "--patients", "60", "--dim", "16"], check=True)
    subprocess.run([sys.executable, ROOT / "scripts/train_survival.py", "--config", ROOT / "configs/synthetic.yaml",
                    "--set", f"cohort_table={data}/cohort.csv", f"embedding_dir={data}/embeddings",
                    f"output_dir={tmp_path}/runs", "run_name=t", "n_folds=2", "epochs=2", f"loss={loss}",
                    "batch_size=8", "n_tiles=32"], check=True)
    summary = pd.read_csv(tmp_path / "runs/t/summary.csv", index_col=0)
    assert 0 <= summary.at["c_index", "mean"] <= 1
    assert np.isfinite(summary.at["ibs", "mean"])
    assert len(pd.read_csv(tmp_path / "runs/t/oof_predictions.csv")) == 60

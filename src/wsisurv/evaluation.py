"""Cross-validation summary: per-fold metrics -> mean ± std, Kaplan-Meier plot by risk group.

Report the metric per fold and average. Do *not* compute one C-index over the concatenated
out-of-fold risks: each fold's model has its own risk scale, so pooled scores are not
comparable. Risk *groups* (each defined by its own fold's training median) can be pooled,
which is what the KM plot / log-rank test below do.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.compare import compare_survival  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402

from wsisurv.metrics import as_surv  # noqa: E402

METRICS = ["c_index", "c_index_uno", "mean_td_auc", "ibs", "ibs_km_reference"]


def summarize(fold_metrics: list[dict], out_dir: Path, title: str = "") -> pd.DataFrame:
    df = pd.DataFrame(fold_metrics)
    df.to_csv(out_dir / "fold_metrics.csv", index=False)
    cols = [c for c in METRICS if c in df]
    summary = df[cols].agg(["mean", "std"]).T
    summary.to_csv(out_dir / "summary.csv")
    lines = [f"== {title} ==" if title else "", df[["fold", "n", "events", *cols]].round(4).to_string(index=False), ""]
    lines += [f"{c:18s} {summary.at[c, 'mean']:.4f} ± {summary.at[c, 'std']:.4f}" for c in cols]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)
    return summary


def km_plot(preds: pd.DataFrame, out_path: Path, title: str = "") -> float:
    """KM curves of the pooled out-of-fold high/low risk groups + log-rank p-value."""
    y = as_surv(preds["time"], preds["event"])
    if preds["high_risk"].nunique() < 2:
        return float("nan")
    _, p = compare_survival(y, preds["high_risk"].to_numpy())
    fig, ax = plt.subplots(figsize=(5, 4))
    for g, name, color in [(0, "low risk", "#2a6f97"), (1, "high risk", "#c8553d")]:
        sub = preds[preds["high_risk"] == g]
        t, s = kaplan_meier_estimator(sub["event"].astype(bool).to_numpy(), sub["time"].to_numpy())
        ax.step(np.r_[0, t], np.r_[1, s], where="post", color=color, label=f"{name} (n={len(sub)})")
    ax.set(xlabel="time", ylabel="survival probability", ylim=(0, 1.02),
           title=f"{title}  log-rank p={p:.2g}".strip())
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(p)

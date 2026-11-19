"""Turning model outputs into (a) a risk score and (b) a survival curve S(t | x).

The C-index only needs (a): it is a ranking metric. The Brier score needs (b): calibrated
probabilities. A Cox model gives only (a) directly; (b) requires a baseline hazard,
estimated here with Breslow's estimator on held-in data.
"""
from __future__ import annotations

import numpy as np
import torch


# ---------- discrete-time (NLL) head ----------

def discrete_survival(logits: torch.Tensor) -> torch.Tensor:
    """(B, K) hazard logits -> (B, K) survival at the end of each bin."""
    return torch.cumprod(1 - torch.sigmoid(logits), dim=1)


def discrete_risk(logits: torch.Tensor) -> torch.Tensor:
    """Scalar risk = -sum_k S_k (higher = worse). Equivalent to -(expected bin of survival)."""
    return -discrete_survival(logits).sum(dim=1)


def discrete_curve_at(surv: np.ndarray, bin_edges: np.ndarray, t_max: float, times: np.ndarray) -> np.ndarray:
    """Evaluate the per-bin survival on arbitrary `times` by linear interpolation between
    bin boundaries [0, edges..., t_max] with S=1 at t=0. surv: (n, K)."""
    knots = np.concatenate([[0.0], bin_edges, [t_max]])
    values = np.concatenate([np.ones((len(surv), 1)), surv], axis=1)
    return np.stack([np.interp(times, knots, v) for v in values])


# ---------- Cox head ----------

def breslow_baseline(risk: np.ndarray, time: np.ndarray, event: np.ndarray):
    """Breslow estimate of the baseline cumulative hazard H0(t).

    Returns (event_times, H0 at those times). Fit it on training (or better: validation)
    predictions — on training predictions of a flexible model the spread of exp(risk) is
    overconfident, so the curves will be too.
    """
    exp_r = np.exp(risk - risk.max())   # shift for stability; undone below
    times = np.unique(time[event == 1])
    increments = []
    for t in times:
        d = ((time == t) & (event == 1)).sum()
        increments.append(d / exp_r[time >= t].sum())
    h0 = np.cumsum(increments) * np.exp(-risk.max())
    return times, h0


def cox_curve_at(risk: np.ndarray, baseline, times: np.ndarray) -> np.ndarray:
    """S(t|x) = exp(-H0(t) * exp(risk)). risk: (n,) -> (n, len(times))."""
    bt, h0 = baseline
    h0_at = np.concatenate([[0.0], h0])[np.searchsorted(bt, times, side="right")]
    return np.exp(-np.outer(np.exp(risk), h0_at))

"""Survival metrics (thin wrappers around scikit-survival).

| metric            | needs           | question it answers                                  |
|-------------------|-----------------|------------------------------------------------------|
| Harrell's C       | risk score      | of comparable pairs, how often is the earlier death ranked riskier? |
| Uno's C (IPCW)    | risk score      | same, re-weighted so heavy censoring does not bias it; up to horizon tau |
| time-dependent AUC| risk score      | at time t, does the score separate "dead by t" from "alive at t"? |
| (integrated) Brier| survival curve  | squared error of predicted S(t) vs observed status, IPCW-weighted;  |
|                   |                 | lower is better, compare to the Kaplan-Meier (no-skill) reference    |

The IPCW metrics estimate the censoring distribution from the *training* fold and can only
be evaluated at times where it is > 0, so evaluation times are clipped to the training
follow-up range.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

log = logging.getLogger(__name__)


def as_surv(time, event):
    return Surv.from_arrays(event=np.asarray(event).astype(bool), time=np.asarray(time, dtype=float))


def eval_times(train_time, train_event, test_time, test_event, n: int = 50) -> np.ndarray:
    """Time grid between the 10th and 80th percentile of test event times, inside
    the range where both test follow-up and the training censoring KM are defined."""
    ev = test_time[test_event == 1]
    lo, hi = np.percentile(ev, [10, 80]) if len(ev) >= 2 else (test_time.min(), test_time.max())
    hi = min(hi, test_time.max() * 0.999, train_time.max() * 0.999)
    lo = max(lo, test_time.min())
    return np.linspace(lo, hi, n) if hi > lo else np.array([lo])


def km_reference_curve(train_time, train_event, times) -> np.ndarray:
    """No-skill model: everyone gets the training Kaplan-Meier curve."""
    t, s = kaplan_meier_estimator(train_event.astype(bool), train_time)
    return np.concatenate([[1.0], s])[np.searchsorted(t, times, side="right")]


def _safe(name, fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(fn())
    except (ValueError, ZeroDivisionError) as e:   # e.g. too few events in a small fold
        log.warning("%s not computable: %s", name, e)
        return float("nan")


def survival_metrics(train_time, train_event, test_time, test_event, risk,
                     surv_at_times: np.ndarray | None = None, times: np.ndarray | None = None) -> dict:
    """All metrics for one test fold. `surv_at_times` (n_test, len(times)) enables the Brier score."""
    train_time, train_event = np.asarray(train_time, float), np.asarray(train_event, int)
    test_time, test_event = np.asarray(test_time, float), np.asarray(test_event, int)
    risk = np.asarray(risk, float)
    y_tr, y_te = as_surv(train_time, train_event), as_surv(test_time, test_event)
    if times is None:
        times = eval_times(train_time, train_event, test_time, test_event)
    tau = times.max()

    out = {
        "n": len(test_time), "events": int(test_event.sum()),
        "c_index": _safe("Harrell C", lambda: concordance_index_censored(
            test_event.astype(bool), test_time, risk)[0]),
        "c_index_uno": _safe("Uno C", lambda: concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0]),
        "mean_td_auc": _safe("td-AUC", lambda: cumulative_dynamic_auc(y_tr, y_te, risk, times)[1]),
    }
    if surv_at_times is not None and len(times) > 1:
        km = np.tile(km_reference_curve(train_time, train_event, times), (len(test_time), 1))
        out["ibs"] = _safe("IBS", lambda: integrated_brier_score(y_tr, y_te, surv_at_times, times))
        out["ibs_km_reference"] = _safe("IBS (KM)", lambda: integrated_brier_score(y_tr, y_te, km, times))
    return out

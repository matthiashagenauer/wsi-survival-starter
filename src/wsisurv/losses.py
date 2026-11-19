"""Survival losses.

Two families, and the difference matters for MIL:

* **Cox partial likelihood** (`cox_ph_loss`). The model outputs one risk score per patient.
  The loss compares each patient who had an event with everyone still at risk at that
  time — *within the batch*. So it is not a sum over independent samples: batch size 1 gives
  no signal, and small batches give a noisy estimate of the risk sets. Needs batches of
  bags (here: tile-subsampled bags, batch_size >= ~32) and several events per batch.

* **Discrete-time negative log-likelihood** (`nll_discrete_loss`). Follow-up is cut into K
  bins; the model outputs K hazard logits h_k = P(event in bin k | survived to k). Each
  patient's loss is independent of the others, so it works at batch size 1 with full
  bags. This is the default in most recent pathology work (Mahmood lab: PORPOISE, MCAT,
  SurvPath, ...; Zadeh & Schmid 2021).

Convention everywhere: event = 1 means the event was observed, 0 means censored.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), averaged over events.

    risk: (B,) log-hazard ratios; higher = shorter survival expected.
    """
    risk, time, event = risk.reshape(-1), time.reshape(-1), event.reshape(-1).float()
    if event.sum() == 0:
        return risk.sum() * 0.0   # no event in batch: no information, keep the graph

    order = torch.argsort(time, descending=True)
    risk, time, event = risk[order], time[order], event[order]
    # log sum_{j: t_j >= t_i} exp(r_j); with descending time this is a running logsumexp...
    log_cum = torch.logcumsumexp(risk, dim=0)
    # ...taken at the *last* member of each tied-time group (Breslow: ties share a risk set)
    _, counts = torch.unique_consecutive(time, return_counts=True)
    last = torch.cumsum(counts, 0) - 1
    log_risk_set = log_cum[last].repeat_interleave(counts)
    return -((risk - log_risk_set) * event).sum() / event.sum()


def nll_discrete_loss(logits: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor,
                      alpha: float = 0.0) -> torch.Tensor:
    """Discrete-time survival NLL.

    logits:  (B, K) hazard logits.
    bin_idx: (B,) bin of the event / censoring time.
    alpha:   optional up-weighting of the uncensored term (the Mahmood-lab code exposes it;
             0 = plain likelihood).

    Event in bin y:    -log S(y-1) - log h(y)     (survived bins < y, then had the event)
    Censored in bin y: -log S(y)                  (survived up to and including bin y)
    """
    event = event.reshape(-1).float()
    bin_idx = bin_idx.reshape(-1, 1).long()
    log_h = F.logsigmoid(logits)          # log h_k
    log_1mh = F.logsigmoid(-logits)       # log (1 - h_k)
    log_s = torch.cumsum(log_1mh, dim=1)  # log S_k = sum_{j<=k} log(1-h_j)
    log_s_pad = F.pad(log_s, (1, 0), value=0.0)   # log S_{-1} = 0

    log_s_before = log_s_pad.gather(1, bin_idx).squeeze(1)       # S(y-1)
    log_s_at = log_s_pad.gather(1, bin_idx + 1).squeeze(1)       # S(y)
    log_h_at = log_h.gather(1, bin_idx).squeeze(1)

    uncensored = -event * (log_s_before + log_h_at)
    censored = -(1 - event) * log_s_at
    loss = (1 - alpha) * (censored + uncensored) + alpha * uncensored
    return loss.mean()

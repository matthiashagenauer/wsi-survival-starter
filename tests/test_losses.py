import numpy as np
import torch

from wsisurv.losses import cox_ph_loss, nll_discrete_loss
from wsisurv.survival import breslow_baseline, discrete_survival


def brute_force_cox(risk, time, event):
    """Textbook Breslow partial likelihood, loop by loop."""
    total = 0.0
    for i in range(len(risk)):
        if event[i]:
            at_risk = time >= time[i]
            total -= risk[i] - np.log(np.exp(risk[at_risk]).sum())
    return total / event.sum()


def test_cox_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    risk = rng.normal(size=40)
    time = rng.integers(1, 10, size=40).astype(float)   # many ties
    event = (rng.random(40) < 0.6).astype(float)
    got = cox_ph_loss(torch.tensor(risk), torch.tensor(time), torch.tensor(event)).item()
    assert np.isclose(got, brute_force_cox(risk, time, event))


def test_cox_no_events_is_zero_with_grad():
    r = torch.zeros(5, requires_grad=True)
    loss = cox_ph_loss(r, torch.arange(5.0), torch.zeros(5))
    loss.backward()
    assert loss.item() == 0 and r.grad is not None


def test_nll_matches_probabilities():
    logits = torch.randn(3, 4)
    h = torch.sigmoid(logits)
    s = discrete_survival(logits)
    # patient 0: event in bin 2 -> P = S1 * h2 ; patient 1: censored in bin 1 -> P = S1 ; patient 2: event bin 0 -> h0
    bins, event = torch.tensor([2, 1, 0]), torch.tensor([1.0, 0.0, 1.0])
    expected = -torch.stack([torch.log(s[0, 1] * h[0, 2]), torch.log(s[1, 1]), torch.log(h[2, 0])]).mean()
    assert torch.isclose(nll_discrete_loss(logits, bins, event), expected, atol=1e-6)


def test_breslow_no_covariates_is_nelson_aalen():
    time = np.array([1.0, 2, 2, 3, 5])
    event = np.array([1, 1, 0, 1, 0])
    t, h0 = breslow_baseline(np.zeros(5), time, event)
    assert np.allclose(t, [1, 2, 3])
    assert np.allclose(h0, np.cumsum([1 / 5, 1 / 4, 1 / 2]))

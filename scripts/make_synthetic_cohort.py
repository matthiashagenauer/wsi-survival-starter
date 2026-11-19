"""Fake cohort with a known answer, to check the whole training/evaluation path on a laptop.

    python scripts/make_synthetic_cohort.py --out data/synthetic

Each tile embedding is noise plus, for a fraction f of tiles, an "aggressive morphology"
direction. Hazard grows with f (and with age). Survival times are exponential; censoring is
random drop-out plus end of study. A working model should reach C ≈ 0.7+; C ≈ 0.5 means
something in the pipeline is broken.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)

    direction = rng.normal(size=args.dim)
    direction /= np.linalg.norm(direction)
    rows = []
    for i in range(args.patients):
        pid = f"P{i:04d}"
        frac = rng.beta(1.0, 4.0)                          # share of aggressive tiles
        age = rng.normal(65, 10)
        hazard = 0.02 * np.exp(6.0 * frac + 0.03 * (age - 65))
        t_event = rng.exponential(1 / hazard)
        t_cens = min(rng.exponential(120), 100.0)          # drop-out, study ends at 100 months
        time, event = min(t_event, t_cens), int(t_event <= t_cens)

        for s in range(rng.integers(1, 3)):                # 1–2 slides per patient
            n = int(rng.integers(150, 600))
            x = rng.normal(size=(n, args.dim)) + rng.normal(scale=0.5, size=args.dim)  # slide offset
            aggressive = rng.random(n) < frac
            x[aggressive] += 2.5 * direction
            sid = f"{pid}_{chr(65 + s)}"
            coords = np.stack(np.unravel_index(np.arange(n), (32, n // 32 + 1)), 1) * 224
            torch.save({"embeddings": torch.from_numpy(x).half(), "coords": torch.from_numpy(coords),
                        "slide_id": sid, "encoder": "synthetic", "mpp": 0.5, "tile_size": 224,
                        "tile_is_aggressive": torch.from_numpy(aggressive)},  # ground truth, fake data only
                       out / "embeddings" / f"{sid}.pt")
            rows.append({"patient_id": pid, "slide_id": sid, "wsi_path": f"{sid}.svs",
                         "time": round(time, 2), "event": event, "age": round(age, 1),
                         "true_aggressive_fraction": round(frac, 4)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "cohort.csv", index=False)
    per_patient = df.drop_duplicates("patient_id")
    from sksurv.metrics import concordance_index_censored
    oracle = concordance_index_censored(per_patient.event.astype(bool), per_patient.time,
                                        per_patient.true_aggressive_fraction)[0]
    print(f"oracle C-index (true risk) = {oracle:.3f}")
    print(f"{len(per_patient)} patients, {len(df)} slides, "
          f"{per_patient.event.mean():.0%} events -> {out}")


if __name__ == "__main__":
    main()

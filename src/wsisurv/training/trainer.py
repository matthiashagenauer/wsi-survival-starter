"""Train + evaluate one cross-validation fold of the MIL survival model."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from wsisurv.data.dataset import PatientBagDataset, make_bin_edges
from wsisurv.data.splits import Fold
from wsisurv.losses import cox_ph_loss, nll_discrete_loss
from wsisurv.metrics import eval_times, survival_metrics
from wsisurv.models.mil import ABMIL
from wsisurv.survival import (
    breslow_baseline,
    cox_curve_at,
    discrete_curve_at,
    discrete_risk,
    discrete_survival,
)

log = logging.getLogger(__name__)


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embedding_dim(embedding_dir: Path, slide_id: str) -> int:
    return torch.load(embedding_dir / f"{slide_id}.pt", map_location="cpu", weights_only=False)["embeddings"].shape[1]


def _loss(cfg, out, batch):
    if cfg["loss"] == "cox":
        return cox_ph_loss(out.squeeze(-1), batch["time"], batch["event"])
    return nll_discrete_loss(out, batch["bin"], batch["event"], alpha=cfg.get("nll_alpha", 0.0))


def _risk(cfg, out):
    return out.squeeze(-1) if cfg["loss"] == "cox" else discrete_risk(out)


@torch.no_grad()
def predict(model, ds: PatientBagDataset, device) -> np.ndarray:
    """Full bags, one patient at a time. Returns raw outputs (n, n_outputs)."""
    model.eval()
    outs = []
    for i in range(len(ds)):
        x = ds[i]["features"].unsqueeze(0).to(device)
        outs.append(model(x).float().cpu())
    return torch.cat(outs).numpy()


def train_fold(cfg: dict, patients: pd.DataFrame, fold: Fold, embedding_dir: str | Path,
               out_dir: str | Path) -> tuple[dict, pd.DataFrame]:
    embedding_dir, out_dir = Path(embedding_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(cfg.get("device", "auto"))
    seed = cfg.get("seed", 0) + fold.k
    torch.manual_seed(seed)

    tr, va, te = (patients.loc[ids] for ids in (fold.train, fold.val, fold.test))
    is_nll = cfg["loss"] == "nll"
    edges = (make_bin_edges(tr["time"].to_numpy(), tr["event"].to_numpy(), cfg["n_bins"])
             if is_nll else None)

    train_ds = PatientBagDataset(tr, embedding_dir, n_tiles=cfg["n_tiles"], bin_edges=edges, seed=seed)
    eval_ds = {name: PatientBagDataset(df, embedding_dir, n_tiles=None, bin_edges=edges)
               for name, df in (("train", tr), ("val", va), ("test", te)) if len(df)}
    eval_ds["train"].cache = train_ds.cache   # share loaded bags

    bs = cfg["batch_size"]
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) > bs,
                        num_workers=cfg.get("num_workers", 0))

    input_dim = embedding_dim(embedding_dir, tr["slide_ids"].iloc[0][0])
    model = ABMIL(input_dim, n_outputs=cfg["n_bins"] if is_nll else 1, hidden_dim=cfg["hidden_dim"],
                  dropout=cfg["dropout"], pooling=cfg.get("pooling", "attention")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best = {"score": -np.inf, "state": None, "epoch": cfg["epochs"] - 1}
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            loss = _loss(cfg, model(batch["features"]), batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}

        # Model selection only on the *validation* split, never on the test fold.
        if "val" in eval_ds:
            val_risk = _risk(cfg, torch.from_numpy(predict(model, eval_ds["val"], device))).numpy()
            m = survival_metrics(tr["time"], tr["event"], va["time"], va["event"], val_risk)
            row["val_c_index"] = m["c_index"]
            if m["c_index"] > best["score"]:
                best = {"score": m["c_index"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
        history.append(row)
        log.info("fold %d epoch %3d  %s", fold.k, epoch,
                 "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"))

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    torch.save({"model": model.state_dict(), "config": cfg, "bin_edges": edges, "epoch": best["epoch"]},
               out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # ---------- evaluation on the held-out fold ----------
    raw = {name: predict(model, ds, device) for name, ds in eval_ds.items()}
    risk = {name: _risk(cfg, torch.from_numpy(r)).numpy() for name, r in raw.items()}
    times = eval_times(tr["time"].to_numpy(), tr["event"].to_numpy(), te["time"].to_numpy(), te["event"].to_numpy())

    if is_nll:
        surv_te = discrete_curve_at(discrete_survival(torch.from_numpy(raw["test"])).numpy(),
                                    edges, tr["time"].max(), times)
    else:
        # Baseline hazard from the val split when there is one (less overfit than train).
        ref = "val" if "val" in risk else "train"
        ref_df = va if ref == "val" else tr
        baseline = breslow_baseline(risk[ref], ref_df["time"].to_numpy(), ref_df["event"].to_numpy())
        surv_te = cox_curve_at(risk["test"], baseline, times)

    metrics = survival_metrics(tr["time"], tr["event"], te["time"], te["event"], risk["test"],
                               surv_at_times=surv_te, times=times)
    metrics.update(fold=fold.k, best_epoch=best["epoch"])

    # Risk groups use the *training* median as cut-off (a test-set median would leak).
    cut = np.median(risk["train"])
    preds = pd.DataFrame({"patient_id": te.index, "fold": fold.k, "time": te["time"].to_numpy(),
                          "event": te["event"].to_numpy(), "risk": risk["test"],
                          "high_risk": (risk["test"] > cut).astype(int)})
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    return metrics, preds

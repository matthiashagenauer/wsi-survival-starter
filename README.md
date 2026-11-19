# wsi-survival-starter

Whole-slide images → tile embeddings → patient-level survival prediction.
A small, readable starting point: every file is short enough to read in one sitting.

```
slides (.svs/.mrxs/.tif)                         cohort.csv (patient_id, slide_id, wsi_path, time, event)
   │  wsi_patching: read at 0.5 µm/px, cut 224 px tiles, drop background
   ▼
tiles (N × 3 × 224 × 224)
   │  frozen foundation model (UNI2 / Virchow2 / H-optimus-1)
   ▼
embeddings/<slide_id>.pt   (N × 1536, + tile coords)          ← Stage 1, GPU, once
   │  all slides of a patient = one bag
   ▼
attention MIL (ABMIL) → risk score / hazard per time bin        ← Stage 2, minutes per run
   │  Cox or discrete-time NLL loss, patient-level 5-fold CV
   ▼
C-index (Harrell, Uno), time-dependent AUC, integrated Brier score, KM curves by risk group
```

## Install

```bash
conda create -n wsisurv python=3.12 -y && conda activate wsisurv
pip install -e ".[gpu,dev]"        # GPU machine / HPC
pip install -e ".[patching,dev]"   # laptop (CPU tiling)
huggingface-cli login              # after accepting the terms of UNI2-h / Virchow2 / H-optimus-1
```

## Quick start (no slides needed)

```bash
make test        # unit tests: losses vs. brute force, metrics, end-to-end CV
make synthetic   # fake cohort with a known answer -> baselines + MIL (Cox and NLL), ~2 min on CPU
cat runs/synthetic/*/summary.txt
```
The synthetic cohort prints its oracle C-index (≈0.71). Models should land at ≈0.60–0.65;
≈0.50 means something is broken.

## Real data

1. **Cohort table** — copy `data/cohort_template.csv` to `data/cohort.csv`. One row per slide;
   `time`/`event` per patient (event = 1 death/progression, 0 censored). Extra columns
   (age, stage, site) can be used as covariates or for stratified splits.
2. **Look at a slide first** — `make inspect SLIDE=/path/to/slide.svs` writes an HTML
   overlay of kept/dropped tiles with sliders for the filter thresholds.
3. **Embed** — set paths + encoder in `configs/embed.yaml`, then `sbatch jobs/embed.slurm`
   (or `make embed`). Resumable; `manifest.csv` lists every slide's status.
4. **Baselines** — `make baseline` (mean-pooled embedding + ridge Cox; clinical-only Cox).
5. **MIL** — `make train` (NLL) or `make train CONFIG=configs/survival_abmil_cox.yaml`.
   Override anything: `--set lr=1e-4 n_bins=8 pooling=mean`.

## Layout

| path | what |
|------|------|
| `src/wsisurv/patching/pipeline.py` | `wsi_patching` pipeline: WSIGrid → PatchExtractor → filters → TorchStreamWriter |
| `src/wsisurv/encoders/foundation.py` | foundation-model wrapper incl. each model's input normalisation |
| `src/wsisurv/data/` | cohort table validation, patient-level folds, bag dataset, time binning |
| `src/wsisurv/models/mil.py` | gated attention MIL (and mean-pool switch) |
| `src/wsisurv/losses.py` | Cox partial likelihood (Breslow ties), discrete-time NLL |
| `src/wsisurv/survival.py` | outputs → risk score / survival curve (Breslow baseline hazard) |
| `src/wsisurv/metrics.py` | Harrell / Uno C, td-AUC, IBS (+ Kaplan–Meier reference) |
| `src/wsisurv/training/trainer.py` | one CV fold: train, select, predict, evaluate |
| `scripts/` | `embed_slides`, `inspect_slide`, `train_survival`, `baseline_cox`, `make_synthetic_cohort` |

## Things that will bite you

- **Split by patient**, stratify by event (and site). Never tune on the test fold —
  set `val_fraction` if you want model selection.
- **Report per-fold mean ± std**, not a C-index over pooled out-of-fold risks.
- **C-index is a ranking metric**; it says nothing about calibration. Always also report
  IBS next to its Kaplan–Meier reference (`ibs_km_reference`): if they are equal, the
  predicted curves are no better than "everyone gets the population curve".
- **Small cohorts are noisy**: 100 patients / 40 events gives a C-index CI of roughly ±0.08.
  Compare to the baselines on the same folds before believing a gain.
- **Choose the endpoint deliberately**: OS includes non-cancer deaths; for TCGA, use the
  curated endpoints of Liu et al. 2018 (TCGA-CDR), which recommend PFI/DSS over OS for several cancer types.
- **Site/scanner confounding**: if survival differs between hospitals, the model can learn
  the hospital. Stratify folds by site and check per-site C-index.
- **Magnification**: always tile at a fixed µm/px (`target_mpp`), never at a pyramid level.

# Common runs. Override the config with e.g. `make train CONFIG=configs/survival_abmil_cox.yaml`
PY ?= python
CONFIG ?= configs/survival_abmil.yaml

.PHONY: install test synthetic embed inspect train baseline

install:            ## editable install incl. slide reading (GPU machine: pip install -e ".[gpu,dev]")
	pip install -e ".[patching,dev]"

test:
	pytest -q

synthetic:          ## whole training/eval path on fake data, ~1 min on a laptop CPU
	$(PY) scripts/make_synthetic_cohort.py --out data/synthetic
	$(PY) scripts/baseline_cox.py --config configs/baseline_cox.yaml --set cohort_table=data/synthetic/cohort.csv embedding_dir=data/synthetic/embeddings output_dir=runs/synthetic "clinical_covariates=[age]"
	$(PY) scripts/train_survival.py --config configs/synthetic.yaml
	$(PY) scripts/train_survival.py --config configs/synthetic.yaml --set loss=cox batch_size=32 run_name=abmil_cox

inspect:            ## make inspect SLIDE=/path/to/slide.svs
	$(PY) scripts/inspect_slide.py configs/embed.yaml $(SLIDE)

embed:
	$(PY) scripts/embed_slides.py configs/embed.yaml

train:
	$(PY) scripts/train_survival.py --config $(CONFIG)

baseline:
	$(PY) scripts/baseline_cox.py --config configs/baseline_cox.yaml

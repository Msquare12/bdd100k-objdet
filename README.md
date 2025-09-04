# BDD100K — Object Detection (Data Analysis → Training → Evaluation)

End-to-end project for BDD100K object detection:
- **Part 1: Data Analysis (EDA)** — Streamlit dashboard in Docker
- **Part 2: Model Training** — Object Detection training
- **Part 3: Evaluation** — metrics + failure slices

---

## Quick Start: Data Analysis (Docker)

Use the containerized dashboard; dataset is mounted from host.

```bash
# 1) edit dataset paths in docker-compose.yml (volumes section)
# 2) build + run
docker compose build eda
docker compose up eda   # open http://localhost:8501
# reuse parquet later:
SKIP_INGEST=1 docker compose up eda
````

Full details: **docker/README.md**

Data Analysis Insights Report: **reports/eda/INSIGHTS.md**

---

## Repository Layout

```
docker/               # Dockerfile + entrypoint + usage
requirements/
  data.txt            # EDA dependencies
scripts/
  run_data_ingest.sh  # JSON → parquet
  run_streamlit.sh    # local Streamlit (non-Docker)
src/
  data/               # parsers, split analysis
  vis/                # Streamlit app + exporters
reports/
  eda/
    INSIGHTS.md       # analysis notes + figures
    assets/           # screenshots/overlays used in the doc
    insights/         # exported CSVs
```

(Heavy data is gitignored; mount via Compose.)

---

## Part 1 : Data Analysis (EDA)

* Dashboard: `src/vis/streamlit_app.py`
* Figures/CSVs + write-up: `reports/eda/INSIGHTS.md`
* How to reproduce figures: see the bottom of **INSIGHTS.md**

---

## Part 2 : Model Training

Planned files (to be added):

* `requirements/model.txt` — torch/ultralytics deps
* `configs/data.yaml`, `configs/yolo.yaml` — dataset & hyperparams
* `scripts/run_train_subset.sh` — quick 1-epoch sanity run
* `scripts/run_train.sh` — full training run
* `reports/model/` — logs/checkpoints (gitignored)

Expected usage:

```bash
# subset sanity (to be added)
bash scripts/run_train_subset.sh
# full training (to be added)
bash scripts/run_train.sh
```

---

## Part 3 : Evaluation

Planned files (to be added):

* `scripts/run_eval.sh` — evaluation on val set
* `src/eval/` — COCO metrics, PR curves, failure clustering
* `reports/eval/` — metrics + qualitative outputs (gitignored)

Expected usage:

```bash
# evaluate last/best weights (to be added)
bash scripts/run_eval.sh
```

---
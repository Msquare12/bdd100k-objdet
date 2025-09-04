# BDD100K Data Analysis – Docker (Compose)

Runs the Streamlit EDA app in a self-contained container.  
Dataset is **mounted** (not baked into the image).

**Files used**
- `docker/Dockerfile.data`
- `docker/entrypoint.sh`
- `docker-compose.yml` (at repo root)

---

## 0) One-time: set your dataset paths

Edit `docker-compose.yml` volumes (left side = your host paths):

```yaml
volumes:
  - "/ABS/PATH/.../bdd100k/images/100k:/data/images:ro"
  - "/ABS/PATH/.../bdd100k/labels:/data/labels:ro"
  - "./reports:/app/reports"
````

---

## 1) Build the image

```bash
docker compose build eda
```

(Optional clean rebuild)

```bash
docker compose build --no-cache eda
```

---

## 2) Run WITH ingest (first run or regenerate parquet)

Generates `reports/eda/annotations.parquet` inside the mounted `./reports` folder, then serves Streamlit.

```bash
docker compose up eda
```

Open: [http://localhost:8501](http://localhost:8501)

Stop: `Ctrl+C` (or `docker compose down`)

---

## 3) Run WITHOUT ingest (reuse existing parquet)

Skips JSON → parquet step and just serves the dashboard.

```bash
SKIP_INGEST=1 docker compose up eda
```

---

## 4) 'Optional' overrides (when needed)

```bash
# Change port
ST_PORT=8600 docker compose up eda

# Custom label filenames
TRAIN_JSON=my_train.json VAL_JSON=my_val.json docker compose up eda
```

Defaults inside the container:

* `BDD_IMG_ROOT=/data/images`
* `BDD_LABELS_DIR=/data/labels`
* `TRAIN_JSON=bdd100k_labels_images_train.json`
* `VAL_JSON=bdd100k_labels_images_val.json`
* `ST_PORT=8501`
* `SKIP_INGEST=0`

---

## 5) Outputs

* Parquet + exports persist under `./reports/...` on your host.



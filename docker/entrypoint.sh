#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden via env)
: "${BDD_IMG_ROOT:=/data/images}"
: "${BDD_LABELS_DIR:=/data/labels}"
: "${TRAIN_JSON:=bdd100k_labels_images_train.json}"
: "${VAL_JSON:=bdd100k_labels_images_val.json}"
: "${OUT_PARQUET:=reports/eda/annotations.parquet}"
: "${ST_PORT:=8501}"
: "${SKIP_INGEST:=0}"

echo "[INFO] IMG_ROOT=$BDD_IMG_ROOT  LABELS_DIR=$BDD_LABELS_DIR"
echo "[INFO] TRAIN_JSON=$TRAIN_JSON  VAL_JSON=$VAL_JSON"
echo "[INFO] OUT_PARQUET=$OUT_PARQUET  SKIP_INGEST=$SKIP_INGEST"

# Basic checks
if [[ ! -d "$BDD_IMG_ROOT" ]]; then
  echo "[ERR] Images root not mounted: $BDD_IMG_ROOT" >&2; exit 1
fi
if [[ ! -f "$BDD_LABELS_DIR/$TRAIN_JSON" || ! -f "$BDD_LABELS_DIR/$VAL_JSON" ]]; then
  echo "[ERR] Label JSONs not found in $BDD_LABELS_DIR" >&2
  echo "      Expect: $TRAIN_JSON and $VAL_JSON" >&2
  exit 1
fi

# Optional ingest (skips if parquet exists)
if [[ "$SKIP_INGEST" != "1" ]]; then
  mkdir -p "$(dirname "$OUT_PARQUET")"
  if [[ -f "$OUT_PARQUET" ]]; then
    echo "[INFO] Parquet exists → $OUT_PARQUET (skipping re-ingest)"
  else
    echo "[INFO] Ingesting labels → $OUT_PARQUET"
    python -m src.data.bdd_parser \
      --train_labels "$BDD_LABELS_DIR/$TRAIN_JSON" \
      --val_labels   "$BDD_LABELS_DIR/$VAL_JSON" \
      --images_root  "$BDD_IMG_ROOT" \
      --out_path     "$OUT_PARQUET"
  fi
else
  echo "[INFO] SKIP_INGEST=1 → not running parser"
fi

# Launch Streamlit
exec streamlit run src/vis/streamlit_app.py --server.port "$ST_PORT" --server.address 0.0.0.0

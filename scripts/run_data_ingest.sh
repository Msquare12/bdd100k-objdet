#!/usr/bin/env bash
set -euo pipefail

# YOUR paths (from your message)
IMG_ROOT="assignment_data_bdd/bdd100k_images_100k/bdd100k/images/100k"
LBL_TRAIN="assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
LBL_VAL="assignment_data_bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"

# Run
python -m src.data.bdd_parser \
  --train_labels "$LBL_TRAIN" \
  --val_labels "$LBL_VAL" \
  --images_root "$IMG_ROOT" \
  --out_path reports/eda/annotations.parquet

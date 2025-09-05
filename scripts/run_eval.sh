#!/usr/bin/env bash
set -euo pipefail

# Path to the trained run
RUN_DIR="runs/subset_yolov8s3"
WEIGHTS="$RUN_DIR/weights/best.pt"

# Use the config you trained with
DATA_YAML="configs/data_bdd_yolo_subset.yaml"

# Eval + save predictions (txt + conf) and standard plots
yolo detect val \
  model="$WEIGHTS" \
  data="$DATA_YAML" \
  imgsz=1280 device=0 \
  save_txt=True save_conf=True save=True plots=True \
  project=runs name=evaluate_subset

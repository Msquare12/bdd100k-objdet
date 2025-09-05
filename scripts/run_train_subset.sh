#!/usr/bin/env bash
set -euo pipefail

# 0) Make a small subset using our main data (for the 1st time only)
# python scripts/make_subset.py --root data/yolo_bdd --out data/yolo_bdd_subset --train_k 2000 --val_k 500 --copy

# 1) Train YOLOv8-s for 1 epoch
yolo detect train \
  model=yolov8s.pt \
  data=configs/data_bdd_yolo_subset.yaml \
  imgsz=1280 epochs=1 batch=6 device=0 \
  mosaic=1.0 copy_paste=0.5 mixup=0.0 scale=0.5 translate=0.1 fliplr=0.5 flipud=0.0 \
  project=runs name=subset_yolov8s \

# # # 2) Validate
# yolo detect val \
#   model=runs/detect/subset_yolov8s2/weights/best.pt \
#   data=configs/data_bdd_yolo_subset.yaml \
#   imgsz=960 device=0

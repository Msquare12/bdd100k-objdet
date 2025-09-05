#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=yolov8s.pt \
  data=configs/data_bdd_yolo.yaml \
  imgsz=1280 epochs=50 batch=6 device=0 \
  mosaic=1.0 copy_paste=0.5 mixup=0.0 scale=0.5 translate=0.1 fliplr=0.5 flipud=0.0 \
  project=runs name=full_yolov8s \

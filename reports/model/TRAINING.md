# Part 2 — Model Training (YOLOv8)

This doc explains **how to train** the chosen model. See **MODEL.md** for model choice & reasoning.

---

## 0) Environment

> CUDA 11.3 (per my machine). Adjust if your CUDA differs.

```bash
# PyTorch (CUDA 11.3)
pip uninstall -y torch torchvision torchaudio
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

# YOLO & deps
pip install -r requirements/model.txt
````

---

## 1) Export BDD → YOLO (one-time)

```bash
python -m src.data.export_yolo \
  --ann reports/eda/annotations.parquet \
  --out data/yolo_bdd \
  --splits train val \
  --img_root /ABS/PATH/to/bdd100k/images/100k
```

Creates:

```
data/yolo_bdd/
  images/{train,val}/   # symlinks
  labels/{train,val}/   # .txt
configs/data_bdd_yolo.yaml
```

---

## 2) Subset sanity (1 epoch only)

```bash
# build a small subset
python scripts/make_subset.py --root data/yolo_bdd --out data/yolo_bdd_subset --train_k 2000 --val_k 500 --copy

# train 1 epoch
yolo detect train \
  model=yolov8s.pt \
  data=configs/data_bdd_yolo_subset.yaml \
  imgsz=1280 epochs=1 batch=16 device=0 \
  mosaic=1.0 copy_paste=0.5 mixup=0.0 scale=0.5 translate=0.1 fliplr=0.5 flipud=0.0 \
  project=runs name=subset_yolov8s \
```

Outputs under `runs/detect/subset_yolov8s/`.

---

## 3) Full training (YOLOv8-s)

> Adjust batch/imgsz to your GPU (12 GB used imgsz=1280, batch≈16).

```bash
yolo detect train \
  model=yolov8s.pt \
  data=configs/data_bdd_yolo.yaml \
  imgsz=1280 epochs=100 batch=16 device=0 \
  mosaic=1.0 copy_paste=0.5 mixup=0.0 scale=0.5 translate=0.1 fliplr=0.5 flipud=0.0 \
  project=runs name=full_yolov8s \
```

Validate later:

```bash
yolo detect val \
  model=runs/detect/full_yolov8s/weights/best.pt \
  data=configs/data_bdd_yolo.yaml \
  imgsz=1280 device=0
```

---

## 4) Quick Documentation Details:

* `reports/model/MODEL.md` (model choice reasoning, architecture etc)
* `reports/model/TRAINING.md` (this file)
* `configs/data_bdd_yolo*.yaml`, `scripts/make_subset.py`, `scripts/run_train*.sh` (scripts for training purpose)

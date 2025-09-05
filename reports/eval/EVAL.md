# Part 3 — Evaluation & Visualization

**Setup**

* Trained **1 epoch** on a **2k/500** subset (time/compute bound) with `yolov8s.pt`, `imgsz=1280`, tiny-object augs.
* Validated on **500** val images (subset).

**Ultralytics val (subset, 1 epoch)**
Overall: **mAP50 = 0.375**, **mAP50-95 = 0.202**, **P = 0.514**, **R = 0.361**.
Per-class: `car` best; **tiny-heavy** classes remain low — `traffic light` (mAP50-95 ≈ **0.16**), `traffic sign` (≈ **0.26**). Low-support long-tail (`rider/motor/bike`) show lowest recall.

**Why (consistent with our EDA)**

* Tiny objects dominate (`traffic light` \~88% small; `sign` \~75% small) → low AP\_small.
* Occlusion/truncation common for `rider/motor` → recall drops.
* No train↔val drift → gaps are **task difficulty**, not split mismatch.

---

## Slice metrics (ours)

* CSV: `reports/eval/metrics_slices.csv` (per-class precision/recall + small-object recall proxy).
* Generate:

```bash
bash scripts/run_eval.sh
python -m src.eval.match_eval \
  --ann reports/eda/annotations.parquet \
  --pred_dir runs/evaluate_subset/labels \
  --out_csv reports/eval/metrics_slices.csv \
  --restrict_to_pred
```

*Note:* single-threshold IoU=0.5; precision can look low since we include all saved predictions. Optionally add `--conf_min 0.25`.

---

## Qualitative (GT vs Pred overlays)

Outputs: `reports/eval/qual/*.jpg` (focus on tiny light/sign scenes).

```bash
python -m src.vis.overlay_pred_gt \
  --ann reports/eda/annotations.parquet \
  --img_root /ABS/PATH/to/bdd100k/images/100k \
  --pred_dir runs/evaluate_subset/labels \
  --out_dir reports/eval/qual \
  --max_images 20 \
  --restrict_to_pred \
  --conf_min 0.25
```

---

## Improvements to try (data → model)

* **Tiny objects:** `imgsz 1280–1536`, `rect=True`, NMS IoU **0.55–0.65**, consider **tiling** at eval.
* **Augs:** keep `mosaic=1.0`, `copy_paste=0.5`, add mild blur/noise; verify **stride-8** coverage.
* **Sampling:** upweight rare contexts (dusk/rain) if extending beyond subset.
* **Monitoring:** track **AP\_small** and recall on `occluded=true` / `edge_touch=true` slices.

---

## Run order

```bash
# 1) Save YOLO val predictions
bash scripts/run_eval.sh

# 2) Compute slice metrics
python -m src.eval.match_eval \
  --ann reports/eda/annotations.parquet \
  --pred_dir runs/evaluate_subset/labels \
  --out_csv reports/eval/metrics_slices.csv \
  --restrict_to_pred

# 3) Export qualitative overlays (images)
python -m src.vis.overlay_pred_gt \
  --ann reports/eda/annotations.parquet \
  --img_root /ABS/PATH/to/bdd100k/images/100k \
  --pred_dir runs/evaluate_subset/labels \
  --out_dir reports/eval/qual --max_images 20 --restrict_to_pred --conf_min 0.25
```

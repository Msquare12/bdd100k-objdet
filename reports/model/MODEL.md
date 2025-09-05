# Model Choice

**Selected:** YOLOv8-s (Ultralytics), COCO-pretrained → fine-tune on BDD100K.

## Why this model (Based on our EDA)
- **Tiny objects:** `traffic light` (≈88% small) and `traffic sign` (≈75% small). YOLOv8’s **anchor-free** head with **DFL** (distribution focal loss) and **stride-8** output works well for small boxes; we’ll train with **imgsz 960–1280** and small-object augs (mosaic/copy-paste).
- **Crowded scenes:** decoupled cls/box head + NMS tuning handle overlaps; we’ll monitor crowded slices.
- **Occlusion/truncation:** robust aug stack (translate/scale/perspective) helps; our drift was negligible → no split rebalancing required.
- **Practicality:** fast to fine-tune, strong COCO priors, easy reproducibility (1-line CLI), light enough for quick subset runs.

## Architecture (short overview)
- **Backbone:** C2f blocks + SPPF → efficient feature extraction.
- **Neck:** PAN-FPN (multi-scale fusion at strides 8/16/32).
- **Head:** decoupled classification & regression, **anchor-free** with **DFL** for precise box localization.
- **Training knobs we’ll use:** `imgsz=1280 mosaic=1.0 copy_paste=0.5 scale=0.5 translate=0.1 fliplr=0.5`.

## Alternatives
- **RT-DETR**: strong accuracy, simpler post-proc; heavier to fine-tune quickly.
- **YOLOv5/v7/v9/v10**: also viable; v8 gives cleaner anchor-free head + tooling for quick demos.
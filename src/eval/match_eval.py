import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLASS_NAMES = [
    "car","bus","truck","person","rider","bike","motor","traffic light","traffic sign","train"
]
NAME2ID = {c:i for i,c in enumerate(CLASS_NAMES)}

def xywhn_to_xyxyn(cx, cy, w, h):
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def IoU(a, b):
    # a: (N,4), b: (M,4), both normalized xyxy
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0: return np.zeros((N,M), dtype=np.float32)
    inter = np.zeros((N,M), dtype=np.float32)
    for i in range(N):
        x1 = np.maximum(a[i,0], b[:,0]); y1 = np.maximum(a[i,1], b[:,1])
        x2 = np.minimum(a[i,2], b[:,2]); y2 = np.minimum(a[i,3], b[:,3])
        w = np.clip(x2 - x1, 0, 1); h = np.clip(y2 - y1, 0, 1)
        inter[i] = w*h
    area_a = (a[:,2]-a[:,0]) * (a[:,3]-a[:,1])
    area_b = (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    union = area_a[:,None] + area_b[None,:] - inter
    union = np.clip(union, 1e-9, 1e9)
    return inter / union

def load_preds(pred_dir: Path):
    # YOLO txt: cls cx cy w h conf  (normalized)
    D = {}
    for p in sorted((pred_dir).glob("*.txt")):
        stem = p.stem  # image stem
        lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if not lines:
            D[stem] = (np.zeros((0,6),dtype=np.float32))
            continue
        arr = []
        for l in lines:
            parts = l.split()
            cls = int(float(parts[0])); cx,cy,w,h,conf = map(float, parts[1:6])
            x1,y1,x2,y2 = xywhn_to_xyxyn(cx,cy,w,h)
            arr.append([cls, x1,y1,x2,y2, conf])
        D[stem] = np.array(arr, dtype=np.float32)
    return D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="reports/eda/annotations.parquet")
    ap.add_argument("--pred_dir", required=True, help="e.g., runs/detect/evaluate_subset/labels/val")
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out_csv", default="reports/eval/metrics_slices.csv")
    ap.add_argument("--restrict_to_pred", action="store_true",
                    help="Evaluate only images that have prediction files in pred_dir")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_csv).parent; out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ann)
    df = df[df["split"] == args.split].copy()
    # Keep only classes we train on
    df = df[df["category"].isin(CLASS_NAMES)]

    # Build GT per image-stem
    df["stem"] = df["image"].map(lambda s: Path(s).stem)
    gt_dict = {}
    for stem, g in df.groupby("stem"):
        boxes = g[["cx_norm","cy_norm","w_norm","h_norm"]].to_numpy(dtype=np.float32)
        xyxy = np.stack([xywhn_to_xyxyn(*row) for row in boxes], axis=0) if len(boxes) else np.zeros((0,4), dtype=np.float32)
        cls  = g["category"].map(NAME2ID).to_numpy(dtype=np.int32)
        small = (g["size_bucket"] == "small").to_numpy()
        occ   = g["occluded"].to_numpy()
        trunc = g["truncated"].to_numpy()
        gt_dict[stem] = {"xyxy": xyxy, "cls": cls, "small": small, "occ": occ, "trunc": trunc}

    preds = load_preds(pred_dir)
    pred_stems = set(preds.keys())
    if args.restrict_to_pred:
        df = df[df["stem"].isin(pred_stems)].copy()
    print(f"[INFO] GT images after restrict: {df['stem'].nunique()} | pred files: {len(pred_stems)}")

    rows = []
    for stem, GT in gt_dict.items():
        P = preds.get(stem, np.zeros((0,6),dtype=np.float32))
        # per class greedy matching
        for c in range(len(CLASS_NAMES)):
            gt_idx = np.where(GT["cls"] == c)[0]
            pd_idx = np.where((P[:,0] == c))[0]
            gt_boxes = GT["xyxy"][gt_idx]
            pd_boxes = P[pd_idx][:,1:5]
            pd_conf  = P[pd_idx][:,5] if len(pd_idx) else np.zeros(0, dtype=np.float32)

            iou_mat = IoU(pd_boxes, gt_boxes)
            matched_gt = set()
            tp = 0; fp = 0
            # greedy by conf
            order = np.argsort(-pd_conf)
            for j in order:
                if iou_mat.shape[0] == 0: fp += 1; continue
                k = np.argmax(iou_mat[j]) if iou_mat.shape[1] else None
                if k is not None and iou_mat[j, k] >= args.iou and k not in matched_gt:
                    matched_gt.add(k); tp += 1
                else:
                    fp += 1
            fn = len(gt_idx) - tp

            # slices
            if len(gt_idx):
                small_mask = GT["small"][gt_idx]
                occ_mask   = GT["occ"][gt_idx]
                trunc_mask = GT["trunc"][gt_idx]
                rows.append({
                    "class": CLASS_NAMES[c],
                    "image": stem,
                    "gt_count": int(len(gt_idx)),
                    "tp": int(tp), "fp": int(fp), "fn": int(fn),
                    "recall": tp / max(1, (tp+fn)),
                    "precision": tp / max(1, (tp+fp)),
                    "gt_small": int(small_mask.sum()),
                    "gt_occ": int(occ_mask.sum()),
                    "gt_trunc": int(trunc_mask.sum()),
                    "tp_small": int(min(tp, small_mask.sum())),  # coarse proxy
                })

    res = pd.DataFrame(rows)
    if res.empty:
        print("[WARN] No matches computed. Check pred_dir and file stems.")
        return

    # Aggregate
    agg = (res.groupby("class")
              .agg(gt=("gt_count","sum"),
                   tp=("tp","sum"), fp=("fp","sum"), fn=("fn","sum"),
                   recall=("recall","mean"), precision=("precision","mean"),
                   gt_small=("gt_small","sum"), gt_occ=("gt_occ","sum"), gt_trunc=("gt_trunc","sum"))
              .reset_index())

    # Small-object recall proxy per class (focus lights/signs)
    per_img_small = (res.groupby("class")
                       .apply(lambda d: (d["tp_small"].sum()/max(1,d["gt_small"].sum())) if d["gt_small"].sum()>0 else np.nan)
                       .reset_index(name="recall_small_proxy"))

    out = agg.merge(per_img_small, on="class", how="left")
    out = out.sort_values("gt", ascending=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote", args.out_csv)
    print(out.head(12))

if __name__ == "__main__":
    main()

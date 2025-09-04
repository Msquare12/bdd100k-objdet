import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --- Helpers -----------------------------------------------------------------
COCO_SMALL = 32 ** 2
COCO_MEDIUM = 96 ** 2

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def im_size_or_default(p: Path, default=(1280, 720)):
    try:
        with Image.open(p) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception:
        return default


def clamp_box(x1, y1, x2, y2, W, H, eps=1e-3):
    ox1, oy1, ox2, oy2 = x1, y1, x2, y2
    x1 = float(np.clip(x1, 0, W))
    y1 = float(np.clip(y1, 0, H))
    x2 = float(np.clip(x2, 0, W))
    y2 = float(np.clip(y2, 0, H))
    was_clamped = (abs(ox1 - x1) > eps) or (abs(oy1 - y1) > eps) or (abs(ox2 - x2) > eps) or (abs(oy2 - y2) > eps)
    return x1, y1, x2, y2, was_clamped


def size_bucket(area_px: float):
    if area_px < COCO_SMALL:
        return "small"
    if area_px < COCO_MEDIUM:
        return "medium"
    return "large"


def iter_bdd_annotations(
    json_path: Path, split_name: str, images_root: Path
) -> Iterable[Dict[str, Any]]:
    """Yield one flattened annotation row per bbox in BDD json."""
    data = json.loads(Path(json_path).read_text())
    for item in data:
        name = item.get("name")
        attrs = item.get("attributes", {})
        labels = item.get("labels", []) or []

        # resolve image path: BDD keeps exactly the file name under split dir
        img_path = None
        # try split folder
        p1 = images_root / split_name / name
        if p1.suffix.lower() not in IMG_EXTS:
            # Some BDD dumps are nested; fallback to scanning split dir
            p1 = next(
                (p for p in (images_root / split_name).glob("**/*") if p.name == name),
                None,
            )
        img_path = p1 if p1 and p1.exists() else None

        if img_path:
            W, H = im_size_or_default(img_path)
        else:
            # fall back to canonical BDD resolution if missing
            W, H = (1280, 720)

        for lab in labels:
            cat = lab.get("category")
            box = lab.get("box2d")
            if not box:
                continue
            x1, y1, x2, y2 = (
                box.get("x1", 0.0),
                box.get("y1", 0.0),
                box.get("x2", 0.0),
                box.get("y2", 0.0),
            )
            x1, y1, x2, y2, was_clamped = clamp_box(x1, y1, x2, y2, W, H)
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            
            cx, cy = x1 + w * 0.5, y1 + h * 0.5
            area = w * h
            area_norm = area / float(W * H)
            aspect = w / h if h > 0 else np.nan

            # edge-touch flags (useful to find truncated/near-border boxes)
            eps = 1e-3
            touch_left   = x1 <= eps
            touch_top    = y1 <= eps
            touch_right  = x2 >= (W - eps)
            touch_bottom = y2 >= (H - eps)
            touch_any = touch_left or touch_top or touch_right or touch_bottom
            edge_sides = ",".join([s for s, b in [
                ("left", touch_left), ("top", touch_top), ("right", touch_right), ("bottom", touch_bottom)
            ] if b])

            attr = lab.get("attributes", {})
            occluded = bool(attr.get("occluded", False))
            truncated = bool(attr.get("truncated", False))
            tlc = attr.get("trafficLightColor", "none")
            manual_shape = bool(lab.get("manualShape", False))
            manual_attr = bool(lab.get("manualAttributes", False))

            yield {
                "image": name,
                "split": split_name,
                "img_w": W,
                "img_h": H,
                "img_path": str(img_path) if img_path else "",
                "category": cat,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "w": w,
                "h": h,
                "cx": cx, "cy": cy, "cx_norm": cx / W, "cy_norm": cy / H,
                "w_norm": w / W, "h_norm": h / H,
                "area": area,
                "area_norm": area_norm,
                "aspect": aspect,
                "size_bucket": size_bucket(area),
                "occluded": occluded,
                "truncated": truncated,
                "trafficLightColor": tlc,
                "edge_touch": touch_any, "edge_sides": edge_sides, "was_clamped": was_clamped,
                "global_timeofday": attrs.get("timeofday", ""),
                "global_scene": attrs.get("scene", ""),
                "global_weather": attrs.get("weather", ""),
                "ann_id": lab.get("id", -1),
                "manual_shape": manual_shape,
                "manual_attr": manual_attr,
            }


def parse_split(json_path: Path, split: str, images_root: Path) -> pd.DataFrame:
    rows = list(tqdm(iter_bdd_annotations(json_path, split, images_root)))
    if not rows:
        return pd.DataFrame(
            columns=[
                "image",
                "split",
                "img_w",
                "img_h",
                "img_path",
                "category",
                "x1",
                "y1",
                "x2",
                "y2",
                "w",
                "h",
                "area",
                "area_norm",
                "aspect",
                "size_bucket",
                "occluded",
                "truncated",
                "trafficLightColor",
                "global_timeofday",
                "global_scene",
                "global_weather",
                "ann_id",
                "manual_shape",
                "manual_attr",
            ]
        )
    return pd.DataFrame.from_records(rows)


# --- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("BDD JSON → parquet for EDA")
    ap.add_argument("--train_labels", type=Path, required=True)
    ap.add_argument("--val_labels", type=Path, required=True)
    ap.add_argument("--images_root", type=Path, required=True, help=".../images/100k")
    ap.add_argument("--out_path", type=Path, default=Path("reports/eda/annotations.parquet"))
    args = ap.parse_args()

    # train
    print("[INFO] Parsing TRAIN ...")
    df_tr = parse_split(args.train_labels, "train", args.images_root)
    # val
    print("[INFO] Parsing VAL ...")
    df_va = parse_split(args.val_labels, "val", args.images_root)

    df = pd.concat([df_tr, df_va], ignore_index=True)
    # Basic sanity flags
    df["is_light_or_sign"] = df["category"].isin(["traffic light", "traffic sign"])
    df["is_vehicle"] = df["category"].isin(["car", "bus", "truck", "train", "rider", "motor", "bike"])

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"[OK] Wrote {len(df):,} rows → {args.out_path}")

    # Also a small image-level table (one row per image), useful for dashboard filters
    img_df = (
        df.groupby(["image", "split", "img_w", "img_h", "global_timeofday", "global_scene", "global_weather"], as_index=False)
          .size()
          .rename(columns={"size": "num_boxes"})
    )
    img_out = args.out_path.with_name("images_index.parquet")
    img_df.to_parquet(img_out, index=False)
    print(f"[OK] Wrote {len(img_df):,} image rows → {img_out}")


if __name__ == "__main__":
    main()

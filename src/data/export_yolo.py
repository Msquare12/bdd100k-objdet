import argparse
from pathlib import Path
import os
import pandas as pd
from typing import Optional

# Final class list (index = YOLO class id)
CLASS_NAMES = [
    "car", "bus", "truck", "person", "rider",
    "bike", "motor", "traffic light", "traffic sign", "train"
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

def yolo_line(cls_id, cx, cy, w, h):
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except OSError:
        # Fallback: copy if symlink not permitted
        import shutil
        shutil.copy2(src, dst)

def resolve_image_path(img_path_col, split, image, img_root: Optional[Path] = None):
    # 1) use img_path from parquet if valid
    if isinstance(img_path_col, str):
        p = Path(img_path_col)
        if p.exists():
            return p
    # 2) try <img_root>/<split>/<image>
    if img_root:
        p = img_root / split / image
        if p.exists():
            return p
        # 3) glob fallback inside split
        hits = list((img_root / split).glob(f"**/{image}"))
        if hits:
            return hits[0]
    return None




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", type=Path, default=Path("reports/eda/annotations.parquet"),
                    help="Parquet produced by bdd_parser.py")
    ap.add_argument("--out", type=Path, default=Path("data/yolo_bdd"),
                    help="Output root: will create images/{split} (symlinks) and labels/{split}")
    ap.add_argument("--splits", nargs="*", default=["train", "val"], help="Which splits to export")
    ap.add_argument("--no_symlink", action="store_true", help="Copy images instead of symlinks")
    ap.add_argument("--img_root", type=Path, required=False,
                    help="Root to BDD images/100k (e.g., .../bdd100k/images/100k)")
    args = ap.parse_args()

    df = pd.read_parquet(args.ann)
    df = df[df["split"].isin(args.splits)].copy()

    out_img_root = args.out / "images"
    out_lab_root = args.out / "labels"
    out_img_root.mkdir(parents=True, exist_ok=True)
    out_lab_root.mkdir(parents=True, exist_ok=True)

    n_imgs_written = 0
    n_boxes_written = 0
    n_skipped_cls = 0

    # We rely on columns from bdd_parser: image, split, img_path, cx_norm, cy_norm, w_norm, h_norm, category
    for (split, image), rows in df.groupby(["split", "image"], sort=False):
        img_path_col = rows["img_path"].iloc[0] if "img_path" in rows.columns else ""
        img_root = args.img_root if args.img_root else None
        img_path = resolve_image_path(img_path_col, split, image, img_root)


        # write label file
        lab_path = out_lab_root / split / (Path(image).stem + ".txt")
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for _, r in rows.iterrows():
            cat = str(r["category"])
            if cat not in CLASS_TO_ID:
                n_skipped_cls += 1
                continue
            cls_id = CLASS_TO_ID[cat]
            lines.append(yolo_line(cls_id, float(r["cx_norm"]), float(r["cy_norm"]),
                                   float(r["w_norm"]), float(r["h_norm"])))
        lab_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_boxes_written += len(lines)

        # link/copy image (optional but convenient)
        if img_path:
            dst = out_img_root / split / Path(image).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if args.no_symlink:
                import shutil
                shutil.copy2(img_path, dst)
            else:
                safe_symlink(Path(img_path), dst)
            n_imgs_written += 1

    print(f"[OK] labels → {out_lab_root} | images → {out_img_root}")
    print(f"     images linked/copied: {n_imgs_written:,}")
    print(f"     boxes written:        {n_boxes_written:,}")
    if n_skipped_cls:
        print(f"     skipped (unknown class): {n_skipped_cls:,}")
    print("[OK] classes:", CLASS_NAMES)

if __name__ == "__main__":
    main()

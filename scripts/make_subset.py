#!/usr/bin/env python
from pathlib import Path
import random, shutil, argparse

def pick(src_lbl_dir: Path, k: int):
    files = sorted([p for p in src_lbl_dir.glob("*.txt")])
    k = min(k, len(files))
    return random.sample(files, k) if k < len(files) else files

def link(img_src: Path, img_dst: Path, copy=False):
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if copy:
            shutil.copy2(img_src, img_dst)
        else:
            img_dst.symlink_to(img_src)
    except Exception:
        shutil.copy2(img_src, img_dst)  # fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/yolo_bdd", help="export_yolo output root")
    ap.add_argument("--out",  default="data/yolo_bdd_subset", help="subset root")
    ap.add_argument("--train_k", type=int, default=2000)
    ap.add_argument("--val_k",   type=int, default=500)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    args = ap.parse_args()

    src = Path(args.root); dst = Path(args.out)
    for split, k in [("train", args.train_k), ("val", args.val_k)]:
        lbl_src = src / "labels" / split
        img_src = src / "images" / split
        lbl_dst = dst / "labels" / split
        img_dst = dst / "images" / split
        lbl_dst.mkdir(parents=True, exist_ok=True); img_dst.mkdir(parents=True, exist_ok=True)

        picks = pick(lbl_src, k)
        for lp in picks:
            lp_dst = lbl_dst / lp.name
            shutil.copy2(lp, lp_dst)
            ip = img_src / (lp.stem + ".jpg")
            if not ip.exists():
                # try common alt extensions
                alts = list(img_src.glob(f"{lp.stem}.*"))
                if not alts: continue
                ip = alts[0]
            link(ip, img_dst / ip.name, copy=args.copy)

    print(f"[OK] subset at {dst}")

if __name__ == "__main__":
    main()

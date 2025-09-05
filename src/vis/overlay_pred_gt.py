import argparse
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

ID2NAME = ["car","bus","truck","person","rider","bike","motor","traffic light","traffic sign","train"]


CLASS_COLORS = {
    "traffic light": (255,60,60),
    "traffic sign":  (255,160,0),
    "car": (0,220,0), "bus": (0,180,0), "truck": (0,140,0),
    "person": (60,130,255), "rider": (60,90,255),
    "bike": (120,120,255), "motor": (100,100,255), "train": (180,0,180)
}

def denorm_xy(x, y, W, H):
    return max(0, int(x * W)), max(0, int(y * H))

def draw_box(draw, xyxy, color, text=None, width=2):
    draw.rectangle(xyxy, outline=color, width=width)
    if text:
        x1,y1,_,_ = xyxy
        draw.text((x1+3,y1+3), text, fill=color)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="reports/eda/annotations.parquet")
    ap.add_argument("--img_root", required=True, help=".../bdd100k/images/100k")
    ap.add_argument("--pred_dir", required=True, help="runs/evaluate_subset/labels")
    ap.add_argument("--out_dir", default="reports/eval/qual")
    ap.add_argument("--max_images", type=int, default=30)
    ap.add_argument("--restrict_to_pred", action="store_true")
    ap.add_argument("--single_stem", default=None, help="Render just this image stem (no .jpg) (For debugging))")
    ap.add_argument("--conf_min", type=float, default=0.25, help="Min conf for drawing preds")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.ann); df = df[df["split"]=="val"].copy()
    df["stem"] = df["image"].map(lambda s: Path(s).stem)

    # build set of stems that actually have a pred file
    pred_stems = {p.stem for p in Path(args.pred_dir).glob("*.txt")}
    if args.restrict_to_pred:
        df = df[df["stem"].isin(pred_stems)].copy()

    # optional single image debug
    if args.single_stem:
        df = df[df["stem"] == args.single_stem].copy()


    # pick images with lots of tiny lights/signs BUT ensure they have predictions
    hard = df[df["category"].isin(["traffic light","traffic sign"]) & (df["size_bucket"]=="small")]
    picks = (hard.groupby(["image","split","stem"]).size()
                .reset_index(name="n")
                .sort_values("n", ascending=False)
                .head(args.max_images))

    for _, r in picks.iterrows():
        img_name, sp = r["image"], r["split"]
        stem = Path(img_name).stem
        # load image
        img_path = Path(args.img_root) / sp / img_name
        if not img_path.exists():
            hits = list((Path(args.img_root)/sp).glob(f"**/{img_name}"))
            if not hits: continue
            img_path = hits[0]
        im = Image.open(img_path).convert("RGB")
        W,H = im.size
        draw = ImageDraw.Draw(im)

        # GT
        sub = df[(df["image"]==img_name) & (df["split"]==sp)]
        for _, g in sub.iterrows():
            x1,y1,x2,y2 = g["cx_norm"]-g["w_norm"]/2, g["cy_norm"]-g["h_norm"]/2, g["cx_norm"]+g["w_norm"]/2, g["cy_norm"]+g["h_norm"]/2
            x1p, y1p = denorm_xy(x1, y1, W, H)
            x2p, y2p = denorm_xy(x2, y2, W, H)
            x1p, y1p = max(0, x1p), max(0, y1p)
            x2p, y2p = min(W - 1, x2p), min(H - 1, y2p)

            color = CLASS_COLORS.get(g["category"], (0, 255, 0))
            draw_box(draw, (x1p, y1p, x2p, y2p), color, text=g["category"], width=2)

        # PRED
        pfile = Path(args.pred_dir)/f"{stem}.txt"
        if pfile.exists():
            lines = [l.strip() for l in pfile.read_text().splitlines() if l.strip()]
            for l in lines:
                c, cx, cy, w, h, conf = l.split()[:6]
                cls_id = int(float(c))
                cx, cy, w, h, conf = map(float, (cx, cy, w, h, conf))
                if conf < args.conf_min:
                    continue
                # xywh (normalized) -> xyxy (pixels)
                x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
                x1, y1 = max(0,int(x1)), max(0,int(y1))
                x2, y2 = min(W-1,int(x2)), min(H-1,int(y2))

                # draw preds in RED, thicker stroke
                draw_box(draw, (x1,y1,x2,y2), (255,0,0), text=f"{ID2NAME[cls_id]} {conf:.2f}", width=3)

        im.save(out/f"{sp}__{img_name}")

    print("[OK] wrote overlays to", out)

if __name__ == "__main__":
    main()

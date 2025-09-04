import pandas as pd
from pathlib import Path

INP = Path("reports/eda/annotations.parquet")
OUT_DIR = Path("reports/eda"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_parquet(INP)

    # Per-image class counts (for boxes/image metrics)
    per_img = (df.groupby(["split","category","image"])
                 .size().reset_index(name="boxes_per_image"))
    per_img_stats = (per_img.groupby(["split","category"])
                          .agg(img_count=("image","nunique"),
                               boxes_per_image_mean=("boxes_per_image","mean"))
                          .reset_index())

    # Core per-class stats
    base = (df.groupby(["split","category"])
              .agg(ann_count=("image","count"),
                   small_rate=("size_bucket", lambda s: (s=="small").mean()),
                   medium_rate=("size_bucket", lambda s: (s=="medium").mean()),
                   large_rate=("size_bucket", lambda s: (s=="large").mean()),
                   occluded_rate=("occluded","mean"),
                   truncated_rate=("truncated","mean"),
                   edge_touch_rate=("edge_touch","mean"),
                   area_px_median=("area","median"),
                   area_norm_median=("area_norm","median"),
                   aspect_median=("aspect","median"))
              .reset_index())

    prof = base.merge(per_img_stats, on=["split","category"], how="left")

    # Wide compare table (train vs val) with simple drift flags
    wide = prof.pivot(index="category", columns="split")
    # flatten MultiIndex cols
    wide.columns = [f"{a}_{b}" for a,b in wide.columns]
    wide = wide.reset_index()

    # Ratios and deltas
    def safe_ratio(a,b): 
        return (a / b) if (b > 0) else float("inf")
    wide["count_ratio_train_val"] = wide.apply(lambda r: safe_ratio(r.get("ann_count_train",0), r.get("ann_count_val",0)), axis=1)
    for col in ["small_rate","occluded_rate","truncated_rate","edge_touch_rate","area_norm_median","boxes_per_image_mean"]:
        wide[f"{col}_delta_train_minus_val"] = wide.get(f"{col}_train",0) - wide.get(f"{col}_val",0)

    # Simple drift flags (tune thresholds if you like)
    wide["flag_imbalance"] = ( (wide["count_ratio_train_val"] > 1.5) | (wide["count_ratio_train_val"] < (1/1.5)) )
    wide["flag_small_rate_drift"] = wide["small_rate_delta_train_minus_val"].abs() > 0.07
    wide["flag_occlusion_drift"] = wide["occluded_rate_delta_train_minus_val"].abs() > 0.07

    # Save
    prof.to_csv(OUT_DIR/"class_profile_by_split.csv", index=False)
    wide.to_csv(OUT_DIR/"class_split_compare.csv", index=False)
    print("[OK] wrote:", OUT_DIR/"class_profile_by_split.csv", "and", OUT_DIR/"class_split_compare.csv")

if __name__ == "__main__":
    main()

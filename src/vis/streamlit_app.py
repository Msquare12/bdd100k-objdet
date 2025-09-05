import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw
from datetime import datetime

# A tiny util to safe-plot empty frames
def _safe(df, default_empty=True):
    return df if len(df) else (pd.DataFrame() if default_empty else df)

@st.cache_data
def load_tables(ann_path: str, img_index_path: str):
    df = pd.read_parquet(ann_path)
    img_df = pd.read_parquet(img_index_path)
    return df, img_df



def export_overlays(current_q: pd.DataFrame, img_root: str, out_dir: Path, max_images: int = 40):
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = (current_q[["image","split"]].drop_duplicates().head(max_images)).itertuples(index=False)
    saved = 0
    for img, sp in picks:
        rows = current_q[(current_q["image"] == img) & (current_q["split"] == sp)]
        image_path = Path(img_root) / sp / img
        if not image_path.exists():
            matches = list((Path(img_root) / sp).glob(f"**/{img}"))
            image_path = matches[0] if matches else None
        if not image_path or not image_path.exists():
            continue
        im = overlay_boxes(image_path, rows, max_boxes=200)
        name_safe = f"{sp}__{img.replace('/', '_')}"
        # ensure we keep an extension
        if "." not in Path(name_safe).name.split("_")[-1]:
            name_safe = name_safe + (image_path.suffix or ".jpg")
        im.save(out_dir / name_safe)
        saved += 1
    return saved



def overlay_boxes(img_path: Path, rows: pd.DataFrame, max_boxes=50):
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for _, r in rows.head(max_boxes).iterrows():
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        color = (0, 200, 0) if r["category"] in {"car", "bus", "truck"} else (200, 0, 0)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
        draw.text((x1 + 2, y1 + 2), r["category"], fill=color)
    return im


def main():
    st.set_page_config(page_title="BDD100K EDA", layout="wide")
    st.title("BDD100K Dataset Object Detection EDA")

    def _apply_pending_preset():
        preset = st.session_state.pop("_preset_updates", None)
        if preset:
            for k, v in preset.items():
                st.session_state[k] = v

    _apply_pending_preset()


    # Paths (can be overridden from sidebar)
    default_repo = Path(".")
    default_reports = default_repo / "reports" / "eda"
    ann_path = st.sidebar.text_input("Annotations parquet", str(default_reports / "annotations.parquet"))
    img_index_path = st.sidebar.text_input("Images index parquet", str(default_reports / "images_index.parquet"))
    img_root = st.sidebar.text_input("Images root (…/images/100k)", os.environ.get("BDD_IMG_ROOT", ""))

    df, img_df = load_tables(ann_path, img_index_path)

    tab_overview, tab_split, tab_spatial, tab_interesting = st.tabs(
        ["Overview", "Train vs Val", "Spatial maps", "Interesting samples"]
    )

    def _queue_preset(**kwargs):
        # Store desired widget values and rerun; they’ll be applied on top via _apply_pending_preset()
        st.session_state["_preset_updates"] = kwargs
        st.rerun()
        # st.experimental_rerun()


    # Sidebar Filters
    splits = st.sidebar.multiselect(
        "Split",
        sorted(df["split"].unique()),
        default=sorted(df["split"].unique()),
        key="f_split",
    )
    cats = st.sidebar.multiselect(
        "Category",
        sorted(df["category"].unique()),
        default=sorted(df["category"].unique()),
        key="f_cat",
    )
    time_of_day = st.sidebar.multiselect(
        "Time of day",
        sorted(df["global_timeofday"].unique()),
        default=sorted(df["global_timeofday"].unique()),
        key="f_tod",
    )

    occl = st.sidebar.selectbox("Occluded", ["any", "true", "false"], index=0, key="f_occl")
    trunc = st.sidebar.selectbox("Truncated", ["any", "true", "false"], index=0, key="f_trunc")

    # traffic light color filter (green/red slicing)
    tlc_vals = ["any"] + sorted([x for x in df["trafficLightColor"].unique() if isinstance(x, str)])
    tlc = st.sidebar.selectbox("Traffic-light color", tlc_vals, index=0, key="f_tlc")

    with st.sidebar.expander("Advanced filters"):
        only_small = st.checkbox("Only small objects (COCO <32 px)", value=False, key="f_small")
        only_edge  = st.checkbox("Only boxes touching border", value=False, key="f_edge")

    # Applying filters
    q = df[df["split"].isin(splits) & df["category"].isin(cats)]
    if time_of_day:
        q = q[q["global_timeofday"].isin(time_of_day)]
    if occl != "any":
        q = q[q["occluded"] == (occl == "true")]
    if trunc != "any":
        q = q[q["truncated"] == (trunc == "true")]
    if tlc != "any":
        q = q[q["trafficLightColor"] == tlc]
    if only_small:
        q = q[q["size_bucket"] == "small"]
    if only_edge:
        q = q[q["edge_touch"]]


    #Quick presets
    st.sidebar.markdown("---")
    # st.sidebar.markdown("---")
    st.sidebar.subheader("Quick presets (Will alter all above settings)")

    # def _set_state(**kwargs):
    #     for k, v in kwargs.items():
    #         st.session_state[k] = v
    #     st.experimental_rerun()

    # Useful values
    _all_splits = sorted(df["split"].unique())
    _all_cats   = sorted(df["category"].unique())
    _tod_uni    = sorted(df["global_timeofday"].unique())
    _nightish   = [x for x in _tod_uni if any(t in x.lower() for t in ["night", "dusk", "dawn"])]

    colA, colB = st.sidebar.columns(2)
    if colA.button("Tiny traffic lights"):
        _queue_preset(
            f_split=_all_splits,
            f_cat=["traffic light"],
            f_tod=[],
            f_occl="any",
            f_trunc="any",
            f_tlc="any",
            f_small=True,
            f_edge=False,
        )

    if colB.button("Tiny traffic signs"):
        _queue_preset(
            f_split=_all_splits,
            f_cat=["traffic sign"],
            f_tod=[],
            f_occl="any",
            f_trunc="any",
            f_tlc="any",
            f_small=True,
            f_edge=False,
        )

    if colA.button("Occluded cars at night"):
        _queue_preset(
            f_split=_all_splits,
            f_cat=["car"],
            f_tod=_nightish if _nightish else [],
            f_occl="true",
            f_trunc="any",
            f_tlc="any",
            f_small=False,
            f_edge=False,
        )

    if colB.button("Border-truncated boxes"):
        _queue_preset(
            f_split=_all_splits,
            f_cat=_all_cats,
            f_tod=[],
            f_occl="any",
            f_trunc="true",
            f_tlc="any",
            f_small=False,
            f_edge=True,
        )

    #Green lights far away
    if st.sidebar.button("Green lights (far)"):
        _queue_preset(
            f_split=_all_splits,
            f_cat=["traffic light"],
            f_tod=[],
            f_occl="any",
            f_trunc="any",
            f_tlc="green",
            f_small=True,
            f_edge=False,
        )


    with tab_overview:
        st.subheader("Summary")
        total = int(len(q))
        img_count = int(q["image"].nunique()) if total > 0 else 0
        median_area = int(q["area"].median()) if total > 0 else 0
        small = (q["size_bucket"] == "small").mean() if total > 0 else 0.0
        medium = (q["size_bucket"] == "medium").mean() if total > 0 else 0.0
        large = (q["size_bucket"] == "large").mean() if total > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total annotations", f"{total:,}")
        c2.metric("Images (filtered)", f"{img_count:,}")
        c3.metric("Median bbox area (px^2)", f"{median_area:,}")
        c4.metric(
            "Small / Medium / Large",
            f"{small:.2%} / {medium:.2%} / {large:.2%}"
        )


        st.subheader("Distributions")
        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(px.histogram(q, x="area", nbins=60, title="BBox area (px^2)"), use_container_width=True)
            st.plotly_chart(px.histogram(q, x="area_norm", nbins=60, title="BBox area as % of image"), use_container_width=True)
        with colB:
            st.plotly_chart(px.histogram(q, x="aspect", nbins=60, title="BBox aspect ratio (w/h)"), use_container_width=True)
            st.plotly_chart(px.bar(q.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False),
                                x="category", y="count", title="Per-class counts"), use_container_width=True)

        st.subheader("Attribute slices")
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(px.bar(q.groupby("global_timeofday").size().reset_index(name="count"),
                                x="global_timeofday", y="count", title="Time of day"), use_container_width=True)
        col2.plotly_chart(px.bar(q.groupby("global_weather").size().reset_index(name="count"),
                                x="global_weather", y="count", title="Weather"), use_container_width=True)
        col3.plotly_chart(px.bar(q.groupby("global_scene").size().reset_index(name="count"),
                                x="global_scene", y="count", title="Scene"), use_container_width=True)

        st.subheader("Qualitative browser")
        if total == 0:
            st.info("No annotations match the current filters. Try widening the selection (e.g., include more splits or categories).")
        else:
            st.caption("Pick an image from the filtered set and view GT boxes.")
            # Build choices from filtered set
            img_candidates = q["image"].unique().tolist()
            pick = st.selectbox("Image", img_candidates[:500])  # limit list for performance
            if pick:
                rows = q[q["image"] == pick]
                # Determine full path
                split = rows["split"].iloc[0]
                # Try standard location
                image_path = Path(img_root) / split / pick
                if not image_path.exists():
                    # fallback: search
                    matches = list((Path(img_root) / split).glob(f"**/{pick}"))
                    image_path = matches[0] if matches else None

                if image_path and image_path.exists():
                    st.image(overlay_boxes(image_path, rows), caption=str(image_path))
                else:
                    st.warning("Couldn't locate the image on disk. Check Images root path in the sidebar.")

        # st.divider()
        # st.caption("Export the current filtered annotations as CSV (for quick offline analysis).")
        # st.download_button("Download CSV", q.to_csv(index=False).encode("utf-8"), file_name="filtered_annotations.csv", mime="text/csv")

        # st.divider()
        # st.subheader("Export current slice")
        # exp_col1, exp_col2 = st.columns([2,1])
        # exp_col1.caption("Save both CSV and overlaid images for the filters you’ve set (handy for your report).")
        # export_n = exp_col2.number_input("Max images to export", min_value=5, max_value=200, value=40, step=5)

        # export_dir = Path("reports/eda/exports") / datetime.now().strftime("%Y%m%d_%H%M%S")
        # if st.button("Export CSV + overlays"):
        #     export_dir.mkdir(parents=True, exist_ok=True)
        #     # CSV
        #     q.to_csv(export_dir / "annotations_filtered.csv", index=False)
        #     saved = export_overlays(q, img_root, export_dir, max_images=int(export_n))
        #     st.success(f"Saved CSV and {saved} overlaid images to: {export_dir}")




    #Train vs Val tab (per-class stats & drift flags)
    with tab_split:
        st.subheader("Per-class stats by split")

        # Use the current filters (q), not the full df
        g = (q.groupby(["split","category"])
            .agg(ann_count=("image","count"),
                    small_rate=("size_bucket", lambda s: (s=="small").mean()),
                    occluded_rate=("occluded","mean"),
                    truncated_rate=("truncated","mean"),
                    area_norm_median=("area_norm","median"))
            .reset_index())

        if g.empty:
            st.info("No annotations match the current filters for split comparison.")
        else:
            left, right = st.columns(2)
            left.plotly_chart(
                px.bar(g, x="category", y="ann_count", color="split", barmode="group", title="Counts by class & split"),
                use_container_width=True
            )
            right.plotly_chart(
                px.bar(g, x="category", y="small_rate", color="split", barmode="group", title="Small-object rate by class"),
                use_container_width=True
            )

            # Second row: Occlusion & Truncation rates
            col3, col4 = st.columns(2)
            col3.plotly_chart(
                px.bar(g, x="category", y="occluded_rate", color="split",
                    barmode="group", title="Occlusion rate by class"),
                use_container_width=True
            )
            col4.plotly_chart(
                px.bar(g, x="category", y="truncated_rate", color="split",
                    barmode="group", title="Truncation rate by class"),
                use_container_width=True
            )

            for fig in [col3, col4]: pass

            st.caption("Drift quick-checks (delta = train − val)")

            # Build a proper wide table and FLATTEN columns
            metrics = ["ann_count","small_rate","occluded_rate","truncated_rate","area_norm_median"]
            wide = g.pivot_table(index="category", columns="split", values=metrics, aggfunc="first")
            wide.columns = [f"{m}_{s}" for (m, s) in wide.columns.to_flat_index()]  # flatten
            wide = wide.reset_index()

            # Helpers that survive missing columns (e.g., only train selected)
            def colget(name):
                return wide[name] if name in wide.columns else pd.Series([np.nan]*len(wide))

            tbl = pd.DataFrame({
                "category": wide["category"],
                "count_train": colget("ann_count_train").fillna(0).astype(int),
                "count_val":   colget("ann_count_val").fillna(0).astype(int),
                "count_ratio_train_val": (colget("ann_count_train") / colget("ann_count_val").replace({0: np.nan})).fillna(np.inf),
                "small_rate_delta": (colget("small_rate_train") - colget("small_rate_val")).fillna(0),
                "occluded_rate_delta": (colget("occluded_rate_train") - colget("occluded_rate_val")).fillna(0),
                "truncated_rate_delta": (colget("truncated_rate_train") - colget("truncated_rate_val")).fillna(0),
                "area_norm_median_delta": (colget("area_norm_median_train") - colget("area_norm_median_val")).fillna(0),
            })

            st.dataframe(tbl.sort_values("count_ratio_train_val", ascending=False), use_container_width=True)
            st.download_button("Download split compare CSV", tbl.to_csv(index=False).encode("utf-8"),
                            file_name="train_val_compare.csv", mime="text/csv")
        

    #Spatial maps tab (center heatmaps, per class)
    with tab_spatial:
        st.subheader("Spatial distribution (bbox centers)")
        sel_class = st.selectbox("Class", sorted(df["category"].unique()))
        st.session_state["spatial_sel_class"] = sel_class
        qq = df[df["category"] == sel_class]
        if len(qq) == 0:
            st.info("No samples for this class in the filtered dataset.")
        else:
            st.plotly_chart(
                px.density_heatmap(qq, x="cx_norm", y="cy_norm", facet_col="split",
                                nbinsx=50, nbinsy=50, range_x=[0,1], range_y=[0,1],
                                title=f"Center heatmap for '{sel_class}' (train vs val)"),
                use_container_width=True
            )
            st.caption("Tip: concentrated blobs indicate systematic placement (e.g., traffic lights near the top).")


    #Interesting samples tab (auto-curated lists + viewer)
    with tab_interesting:

        st.caption("Quick set: focus the lists below to common slices")
        ch1, ch2, ch3, ch4 = st.columns(4)
        if ch1.button("Cars @ night (occluded)"):
            _queue_preset(f_cat=["car"], f_tod=_nightish if _nightish else [], f_occl="true", f_trunc="any", f_tlc="any", f_small=False, f_edge=False)
        if ch2.button("Tiny lights"):
            _queue_preset(f_cat=["traffic light"], f_tod=[], f_occl="any", f_trunc="any", f_tlc="any", f_small=True, f_edge=False)
        if ch3.button("Truncated near border"):
            _queue_preset(f_cat=_all_cats, f_tod=[], f_occl="any", f_trunc="true", f_tlc="any", f_small=False, f_edge=True)
        if ch4.button("Crowded scenes"):
            _queue_preset(f_cat=_all_cats, f_tod=[], f_occl="any", f_trunc="any", f_tlc="any", f_small=False, f_edge=False)


        st.subheader("Interesting / unique samples")

        K = st.slider("How many to list", min_value=5, max_value=50, value=10, step=1)
        sel_class_2 = st.selectbox("Focus class (for class-specific list only)", sorted(df["category"].unique()))

        # Candidates
        crowded = (df.groupby(["image","split"]).size().reset_index(name="n").sort_values("n", ascending=False).head(200))
        class_crowded = (df[df["category"] == sel_class_2]
                        .groupby(["image","split"]).size().reset_index(name="n")
                        .sort_values("n", ascending=False).head(200))
        tiny = (df[df["size_bucket"]=="small"]
                .groupby(["image","split"]).size().reset_index(name="n")
                .sort_values("n", ascending=False).head(200))
        occl_heavy = (df.groupby(["image","split"])
                        .agg(n=("image","count"), occl=("occluded","mean"))
                        .reset_index().query("n>=5").sort_values("occl", ascending=False).head(200))
        trunc_heavy = (df.groupby(["image","split"])
                        .agg(n=("image","count"), trunc=("truncated","mean"), edge=("edge_touch","mean"))
                        .reset_index().query("n>=5").sort_values(["trunc","edge"], ascending=False).head(200))

        # Rare attribute combo: (timeofday, weather, scene)
        img_level = (df.groupby(["image","split","global_timeofday","global_weather","global_scene"], as_index=False)
                    .size().rename(columns={"size":"num_boxes"}))
        combo_freq = (img_level.groupby(["global_timeofday","global_weather","global_scene"])
                                .size().reset_index(name="combo_freq"))
        rare = (img_level.merge(combo_freq, on=["global_timeofday","global_weather","global_scene"], how="left")
                        .sort_values(["combo_freq","num_boxes"], ascending=[True, False]).head(200))

        # Show short tables
        c1, c2 = st.columns(2)
        c1.write("Most crowded (overall)"); c1.dataframe(crowded.head(K), use_container_width=True, height=300)
        c2.write(f"Most crowded '{sel_class_2}'"); c2.dataframe(class_crowded.head(K), use_container_width=True, height=300)

        c3, c4 = st.columns(2)
        c3.write("Tiny-object rich"); c3.dataframe(tiny.head(K), use_container_width=True, height=300)
        c4.write("Occlusion / truncation heavy"); 
        show_trunc = trunc_heavy[["image","split","n","trunc","edge"]].rename(columns={"trunc":"trunc_rate","edge":"edge_touch_rate"})
        c4.dataframe(show_trunc.head(K), use_container_width=True, height=300)

        # Quick viewer: pick any image from concatenated candidate pool
        st.markdown("### Preview an image from above lists")
        pool = pd.concat([
            crowded[["image","split"]], class_crowded[["image","split"]], tiny[["image","split"]],
            occl_heavy[["image","split"]], trunc_heavy[["image","split"]], rare[["image","split"]]
        ]).drop_duplicates().head(1000)
        st.session_state["interesting_pool"] = pool[["image", "split"]].drop_duplicates()
        if len(pool) == 0:
            st.info("No candidates under current filters.")
        else:
            pick_row = st.selectbox("Pick image", pool.apply(lambda r: f"{r['split']} / {r['image']}", axis=1).tolist())
            split = pick_row.split(" / ")[0]; name = pick_row.split(" / ")[1]
            rows = df[(df["image"] == name) & (df["split"] == split)]
            image_path = Path(img_root) / split / name
            if not image_path.exists():
                matches = list((Path(img_root) / split).glob(f"**/{name}"))
                image_path = matches[0] if matches else None
            if image_path and image_path.exists():
                st.image(overlay_boxes(image_path, rows), caption=str(image_path))
            else:
                st.warning("Image not found on disk for preview.")

    #Export block
    st.divider()
    st.subheader("Export current slice")

    exp_col1, exp_col2 = st.columns([2, 1])
    exp_col1.caption("Save CSV + overlaid images for the selected target (sidebar filters, interesting samples, or spatial class).")
    export_n = exp_col2.number_input("Max images to export(between 1 to 200)", min_value=1, max_value=200, value=10, step=1)

    target = st.selectbox(
        "Export target",
        ["Sidebar filters", "Spatial maps tab class", "Interesting Sample (top-K)"],
        index=0,
    )

    # Decide which dataframe to export
    if target == "Sidebar filters":
        exp_df = q.copy()
    elif target == "Interesting Sample (top-K)":
        if "interesting_pool" in st.session_state:
            keys = st.session_state["interesting_pool"]
            # Intersect with q so it respects the sidebar filters too
            exp_df = q.merge(keys, on=["image", "split"], how="inner")
        else:
            st.warning("No interesting sample available yet. Visit the 'Interesting samples' tab first.")
            exp_df = q.iloc[0:0]  # empty
    elif target == "Spatial maps tab class":
        if "spatial_sel_class" in st.session_state:
            selc = st.session_state["spatial_sel_class"]
            exp_df = q[q["category"] == selc].copy()
        else:
            st.warning("No spatial class selected yet. Visit the 'Spatial maps' tab first.")
            exp_df = q.iloc[0:0]
    else:
        exp_df = q.copy()

    # Safety guard
    if exp_df.empty:
        st.info("Nothing to export for the chosen target + filters.")
    else:
        export_dir = Path("reports/eda/exports") / datetime.now().strftime("%Y%m%d_%H%M%S")
        if st.button("Export CSV + overlays"):
            export_dir.mkdir(parents=True, exist_ok=True)
            exp_df.to_csv(export_dir / "annotations_filtered.csv", index=False)
            saved = export_overlays(exp_df, img_root, export_dir, max_images=int(export_n))
            st.success(f"Saved CSV and {saved} overlaid images to: {export_dir}")




if __name__ == "__main__":
    main()

"""
Predict cell types on new TIF images using ITERATIVE refinement.

Same as predict_cell_types.py but instead of a fixed 2-pass approach,
iterates predictions until neighbor-celltype features stabilize:

  Round 1: predict with neighbor celltypes = -1 (unknown)
  Round 2: fill neighbor celltypes from round 1, re-predict
  Round 3: fill neighbor celltypes from round 2, re-predict
  ...
  Stop when predictions no longer change OR max_rounds reached.

Usage:
  python predict_cell_types_iterative.py \
    --tif path/to/new_image.tif \
    --model-dir trained_model \
    --um-per-px 1.0 --gpu \
    --max-rounds 10

Output (per TIF, per model):
  predictions/
    {stem}_RandomForest_predictions.csv
    {stem}_RandomForest_overlay.png
    {stem}_XGBoost_predictions.csv
    {stem}_XGBoost_overlay.png
    {stem}_LightGBM_predictions.csv
    {stem}_LightGBM_overlay.png
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, disk
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes, label as ndlabel
from collections import deque, defaultdict

from .extract_features import (
    ensure_rgb_uint8,
    to_grayscale_float,
    segment_cellpose_sam,
    build_tissue_mask,
    compute_layer_index_edt,
    build_cell_adjacency,
    extract_all_features,
    count_vascular_poles,
    compute_pole_features,
    compute_neighbor_celltypes,
    extract_cnn_embedding_features,
    CELL_CLASSES,
    LABEL_TO_NAME,
)


# Display palette (RGB)
DISPLAY_PALETTE = {
    0: (255, 105, 180),  # root_cap - pink
    1: (0,   0,   255),  # epidermis - blue
    2: (255, 255,   0),  # exodermis - yellow
    3: (0,   200,   0),  # cortex - green
    4: (255, 165,   0),  # endodermis - orange
    5: (128,   0, 128),  # pericycle - purple
    6: (255,   0,   0),  # xylem - red
    7: (255, 255, 255),  # phloem - white
    8: (0,   255, 255),  # stele - cyan
}


def _load_font(size):
    for path in [
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


MODEL_NAMES = ["RandomForest", "XGBoost", "LightGBM"]


def load_models(model_dir):
    """Load all three models + scalers + encoder.

    Supports two layouts:
      - Per-model scalers: feature_scaler_RandomForest.joblib, etc. (iterative training)
      - Shared scaler: feature_scaler.joblib (old training)
    """
    model_dir = Path(model_dir)
    feature_cols = joblib.load(model_dir / "feature_columns.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")

    # Check for shared scaler (old layout)
    shared_scaler_path = model_dir / "feature_scaler.joblib"
    shared_scaler = None
    if shared_scaler_path.exists():
        shared_scaler = joblib.load(shared_scaler_path)

    models = {}
    scalers = {}
    for name in MODEL_NAMES:
        model_path = model_dir / f"model_{name}.joblib"
        if not model_path.exists():
            print(f"  WARNING: {model_path} not found, skipping {name}")
            continue
        models[name] = joblib.load(model_path)

        # Per-model scaler (iterative training) or shared scaler
        per_model_scaler_path = model_dir / f"feature_scaler_{name}.joblib"
        if per_model_scaler_path.exists():
            scalers[name] = joblib.load(per_model_scaler_path)
        elif shared_scaler is not None:
            scalers[name] = shared_scaler
        else:
            raise FileNotFoundError(
                f"No scaler found for {name} in {model_dir}")
        print(f"  Loaded {name} from {model_path}")

    print(f"  Classes: {list(le.classes_)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Models loaded: {list(models.keys())}")
    return models, scalers, feature_cols, le


def save_prediction_overlay(img_rgb, masks, cell_predictions, out_path,
                             le, alpha=0.5, label_cells=False):
    """
    Overlay predicted cell-type colors on TIF, with a legend.

    label_cells=True also prints an abbreviated cell-type name inside every
    cell. Off by default: on a dense cross-section the per-cell text overlaps
    and hides the image. The color + legend already carry the same information.
    """
    H, W = masks.shape
    gray = to_grayscale_float(img_rgb)
    base = np.stack([gray, gray, gray], axis=-1)

    # Build color overlay from predictions
    color_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cid, pred_name in cell_predictions.items():
        label_int = CELL_CLASSES.get(pred_name, -1)
        if label_int >= 0 and label_int in DISPLAY_PALETTE:
            color_rgb[masks == cid] = DISPLAY_PALETTE[label_int]

    has_color = color_rgb.sum(axis=-1) > 0
    mask_f = has_color[..., None].astype(np.float32)
    blended = base * (1.0 - alpha * mask_f) + color_rgb.astype(np.float32) * (alpha * mask_f)

    # Cell boundaries
    boundaries = find_boundaries(masks, mode="outer")
    blended[boundaries] = [255, 255, 255]

    im = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    # Optional: draw the cell-type name inside each cell (--label-cells)
    if label_cells:
        props = {p.label: p for p in regionprops(masks)}
        med_area = float(np.median([p.area for p in props.values()])) if props else 100.0
        base_fs = max(7, min(14, int(np.sqrt(med_area) * 0.25)))
        font = _load_font(base_fs)

        for cid, pred_name in cell_predictions.items():
            if cid not in props:
                continue
            cy, cx = props[cid].centroid
            x, y = int(round(cx)), int(round(cy))
            # Abbreviate: first 4 chars
            abbr = pred_name[:4]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                draw.text((x + dx, y + dy), abbr, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((x, y), abbr, fill=(255, 255, 255), font=font, anchor="mm")

    # Legend
    legend_font = _load_font(13)
    legend_x = W - 140
    legend_y = 10
    present_types = sorted(set(cell_predictions.values()))
    n_legend = len(present_types)
    draw.rectangle(
        [legend_x - 4, legend_y - 4, W - 4, legend_y + n_legend * 20 + 4],
        fill=(0, 0, 0),
    )
    for i, name in enumerate(present_types):
        label_int = CELL_CLASSES.get(name, -1)
        color = DISPLAY_PALETTE.get(label_int, (128, 128, 128))
        ly = legend_y + i * 20
        draw.rectangle([legend_x, ly, legend_x + 14, ly + 14], fill=color,
                       outline=(255, 255, 255))
        draw.text((legend_x + 20, ly), name, fill=(255, 255, 255), font=legend_font)

    im.save(str(out_path))


def anatomical_postprocess(df, layer_lookup, adjacency, masks, le=None, y_proba=None,
                           rf_proba_lookup=None):
    """
    Enforce known root anatomical ring order using the ADJACENCY GRAPH.

    Root anatomy (outside -> inside):
      root_cap -> epidermis -> exodermis -> cortex -> endodermis -> pericycle -> stele

    Rules (layer + adjacency based):
      - Exodermis = ONE layer only (epidermis_layer + 1)
      - Endodermis = ONE layer only (the layer neighboring pericycle/stele)
      - Pericycle = ONE layer only (just inside endodermis)
      - Cortex fills the multiple layers between exodermis and endodermis
      - Recover phloem and xylem from stele using probability scores
    """
    predictions = dict(zip(df["cell_id"].values, df["predicted_cell_type"].values))

    # Build probability lookup for phloem recovery
    proba_lookup = {}
    if y_proba is not None and le is not None:
        class_names = list(le.classes_)
        for i, cid in enumerate(df["cell_id"].values):
            proba_lookup[int(cid)] = dict(zip(class_names, y_proba[i]))

    # -- Step 1: Identify tissue groups from model predictions --
    epidermis_cells = {cid for cid, pred in predictions.items()
                       if pred == "epidermis"}
    root_cap_cells = {cid for cid, pred in predictions.items()
                      if pred == "root_cap"}
    outer_cells = epidermis_cells | root_cap_cells

    inner_types = ("pericycle", "stele", "xylem", "phloem")
    inner_cells = {cid for cid, pred in predictions.items()
                   if pred in inner_types}

    # -- Step 2: Find exodermis ring -- SINGLE LAYER only --
    epi_layer_counts = defaultdict(int)
    for cid in epidermis_cells:
        lv = layer_lookup.get(cid, -1)
        if lv >= 0:
            epi_layer_counts[lv] += 1
    main_epi_layer = max(epi_layer_counts, key=epi_layer_counts.get) if epi_layer_counts else 0

    epi_neighbor_cells = set()
    for epi_cid in epidermis_cells:
        for nbr in adjacency.get(epi_cid, set()):
            if nbr not in outer_cells and nbr not in inner_cells:
                epi_neighbor_cells.add(nbr)

    epi_nbr_layer_counts = defaultdict(int)
    for cid in epi_neighbor_cells:
        lv = layer_lookup.get(cid, -1)
        if lv >= 0 and lv > main_epi_layer:
            epi_nbr_layer_counts[lv] += 1

    if epi_nbr_layer_counts:
        exo_layer = max(epi_nbr_layer_counts, key=epi_nbr_layer_counts.get)
    else:
        exo_layer = main_epi_layer + 1

    n_exo_promoted = 0
    n_exo_demoted = 0
    for cid, pred in list(predictions.items()):
        lv = layer_lookup.get(cid, -1)
        if lv == exo_layer and cid not in outer_cells and cid not in inner_cells:
            if pred != "exodermis":
                n_exo_promoted += 1
            predictions[cid] = "exodermis"
        elif pred == "exodermis":
            predictions[cid] = "cortex"
            n_exo_demoted += 1

    n_final_exo = sum(1 for p in predictions.values() if p == "exodermis")
    print(f"    Post-processing: exodermis restricted to layer {exo_layer} "
          f"(epi_layer={main_epi_layer}): {n_final_exo} exodermis cells, "
          f"promoted {n_exo_promoted} into exo, demoted {n_exo_demoted} to cortex")

    # -- Step 3: Find endodermis ring -- SINGLE LAYER only --
    cells_neighboring_inner = set()
    for inner_cid in inner_cells:
        for nbr in adjacency.get(inner_cid, set()):
            if nbr not in inner_cells and nbr not in outer_cells:
                cells_neighboring_inner.add(nbr)

    exo_cells_final = {cid for cid, pred in predictions.items()
                       if pred == "exodermis"}
    endo_candidates = cells_neighboring_inner - exo_cells_final

    endo_layer_counts = defaultdict(int)
    for cid in endo_candidates:
        lv = layer_lookup.get(cid, -1)
        if lv >= 0:
            endo_layer_counts[lv] += 1
    endo_layer = max(endo_layer_counts, key=endo_layer_counts.get) if endo_layer_counts else -1

    n_relabeled_endo = 0
    for cid in endo_candidates:
        lv = layer_lookup.get(cid, -1)
        if lv == endo_layer:
            if predictions.get(cid) != "endodermis":
                n_relabeled_endo += 1
            predictions[cid] = "endodermis"

    n_fixed_endo = 0
    for cid, pred in list(predictions.items()):
        if pred == "endodermis":
            lv = layer_lookup.get(cid, -1)
            if lv != endo_layer:
                predictions[cid] = "cortex"
                n_fixed_endo += 1

    n_final_endo = sum(1 for p in predictions.values() if p == "endodermis")
    print(f"    Post-processing: endodermis restricted to layer {endo_layer}: "
          f"{n_final_endo} endodermis cells, relabeled {n_relabeled_endo} into endo, "
          f"demoted {n_fixed_endo} to cortex")

    # -- Step 4: Pericycle -- SINGLE RING only (adjacency-based) --
    endo_cells_final = {cid for cid, pred in predictions.items()
                        if pred == "endodermis"}

    peri_candidates = set()
    for endo_cid in endo_cells_final:
        endo_lv = layer_lookup.get(endo_cid, -1)
        for nbr in adjacency.get(endo_cid, set()):
            nbr_lv = layer_lookup.get(nbr, -1)
            if nbr_lv > endo_lv and nbr not in endo_cells_final:
                nbr_pred = predictions.get(nbr)
                if nbr_pred not in ("epidermis", "root_cap", "exodermis", "cortex",
                                    "endodermis"):
                    peri_candidates.add(nbr)

    peri_layer_counts = defaultdict(int)
    for cid in peri_candidates:
        lv = layer_lookup.get(cid, -1)
        if lv >= 0:
            peri_layer_counts[lv] += 1
    peri_layer = max(peri_layer_counts, key=peri_layer_counts.get) if peri_layer_counts else (endo_layer + 1 if endo_layer >= 0 else -1)

    allowed_pericycle = set()
    n_peri_promoted = 0
    for cid in peri_candidates:
        lv = layer_lookup.get(cid, -1)
        if lv == peri_layer:
            allowed_pericycle.add(cid)
            if predictions.get(cid) != "pericycle":
                n_peri_promoted += 1
            predictions[cid] = "pericycle"

    n_peri_demoted = 0
    for cid, pred in list(predictions.items()):
        if pred == "pericycle" and cid not in allowed_pericycle:
            predictions[cid] = "stele"
            n_peri_demoted += 1

    n_final_peri = sum(1 for p in predictions.values() if p == "pericycle")
    print(f"    Post-processing: pericycle restricted to endodermis-adjacent cells "
          f"in layer {peri_layer} (endo_layer={endo_layer}): {n_final_peri} pericycle "
          f"cells, promoted {n_peri_promoted} into peri, demoted {n_peri_demoted} to stele")

    # -- Step 5: Fix stray epidermis deep inside the root --
    n_fixed_epi = 0
    for cid, pred in list(predictions.items()):
        if pred == "epidermis":
            lv = layer_lookup.get(cid, -1)
            if lv >= main_epi_layer + 3:
                nbrs = adjacency.get(cid, set())
                nbr_preds = [predictions.get(n) for n in nbrs
                             if n in predictions and predictions.get(n) != "epidermis"]
                if nbr_preds:
                    from collections import Counter
                    predictions[cid] = Counter(nbr_preds).most_common(1)[0][0]
                    n_fixed_epi += 1
    if n_fixed_epi > 0:
        print(f"    Post-processing: fixed {n_fixed_epi} stray epidermis cells")

    # -- Step 6: Recover phloem and xylem from inner tissue using probabilities --
    if proba_lookup:
        def _effective_prob(cid, cls_name):
            """Return the best available probability for xylem/phloem."""
            probs = proba_lookup.get(cid, {})
            model_prob = probs.get(cls_name, 0.0)
            if rf_proba_lookup and model_prob < 0.05:
                rf_probs = rf_proba_lookup.get(cid, {})
                rf_prob = rf_probs.get(cls_name, 0.0)
                if rf_prob > model_prob:
                    return rf_prob
            return model_prob

        # Recover phloem from stele (and xylem)
        n_recovered_phloem = 0
        n_recovered_phloem_rf = 0
        for cid, pred in list(predictions.items()):
            if pred in ("stele", "xylem"):
                phloem_prob = _effective_prob(cid, "phloem")
                probs = proba_lookup.get(cid, {})
                current_prob = probs.get(pred, 0.0)
                if phloem_prob >= 0.12 and phloem_prob >= current_prob - 0.25:
                    predictions[cid] = "phloem"
                    n_recovered_phloem += 1
                    if probs.get("phloem", 0.0) < 0.05:
                        n_recovered_phloem_rf += 1
        if n_recovered_phloem > 0:
            msg = f"    Post-processing: recovered {n_recovered_phloem} phloem cells from stele/xylem"
            if n_recovered_phloem_rf > 0:
                msg += f" ({n_recovered_phloem_rf} via RF fallback)"
            print(msg)

        # Recover xylem from stele
        n_recovered_xylem = 0
        n_recovered_xylem_rf = 0
        for cid, pred in list(predictions.items()):
            if pred == "stele":
                xylem_prob = _effective_prob(cid, "xylem")
                probs = proba_lookup.get(cid, {})
                current_prob = probs.get(pred, 0.0)
                if xylem_prob >= 0.08 and xylem_prob >= current_prob - 0.30:
                    predictions[cid] = "xylem"
                    n_recovered_xylem += 1
                    if probs.get("xylem", 0.0) < 0.05:
                        n_recovered_xylem_rf += 1
        if n_recovered_xylem > 0:
            msg = f"    Post-processing: recovered {n_recovered_xylem} xylem cells from stele"
            if n_recovered_xylem_rf > 0:
                msg += f" ({n_recovered_xylem_rf} via RF fallback)"
            print(msg)

        # Xylem strand growing
        n_strand_grown = 0
        for _round in range(3):
            grown_this_round = 0
            current_xylem = {cid for cid, p in predictions.items() if p == "xylem"}
            for cid, pred in list(predictions.items()):
                if pred != "stele":
                    continue
                nbrs = adjacency.get(cid, set())
                n_xylem_nbrs = sum(1 for n in nbrs if n in current_xylem)
                if n_xylem_nbrs >= 2:
                    xylem_prob = _effective_prob(cid, "xylem")
                    if xylem_prob >= 0.05:
                        predictions[cid] = "xylem"
                        grown_this_round += 1
            n_strand_grown += grown_this_round
            if grown_this_round == 0:
                break
        if n_strand_grown > 0:
            print(f"    Post-processing: grew xylem strands by {n_strand_grown} cells")

    # Update dataframe
    df["predicted_cell_type"] = df["cell_id"].map(predictions)
    return df, predictions


def _prepare_features(df_base, feature_cols, scaler):
    """Align feature columns and scale. Returns X_scaled, handling NaN."""
    for col in feature_cols:
        if col not in df_base.columns:
            df_base[col] = 0.0

    X = df_base[feature_cols].values.astype(np.float32)

    # Handle NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = np.nanmean(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    X_scaled = scaler.transform(X)
    return X_scaled


def iterative_predict(model, df_base, adjacency, feature_cols, scaler, le,
                      max_rounds=10):
    """
    Iteratively predict cell types until neighbor-celltype features converge.

    Round 1: predict with neighbor celltypes = -1 (unknown)
    Round N: fill neighbor celltypes from round N-1 predictions, re-predict
    Stop when no predictions change or max_rounds reached.

    Returns: (df, pred_names, y_proba) from the final round.
    """
    neighbor_ct_cols = ["radial_inward_neighbor_celltype", "radial_outward_neighbor_celltype",
                        "tangential_cw_neighbor_celltype", "tangential_ccw_neighbor_celltype"]
    has_neighbor_ct = any(c in feature_cols for c in neighbor_ct_cols)

    prev_pred_names = None
    best_pred_names = None
    best_proba = None
    best_df = None
    best_n_changed = float("inf")
    prev_n_changed = float("inf")
    n_stalled = 0  # count rounds where n_changed doesn't decrease

    for round_num in range(1, max_rounds + 1):
        df_round = df_base.copy()

        if has_neighbor_ct:
            if round_num == 1:
                # First round: no neighbor celltype info
                df_round = compute_neighbor_celltypes(
                    df_round, adjacency, cell_type_labels=None
                )
            else:
                # Use previous round's predictions as neighbor celltypes
                prev_labels = {}
                for i, cid in enumerate(df_base["cell_id"].values):
                    prev_labels[int(cid)] = CELL_CLASSES.get(prev_pred_names[i], -1)

                df_round = compute_neighbor_celltypes(
                    df_round, adjacency, cell_type_labels=prev_labels
                )

        X_scaled = _prepare_features(df_round, feature_cols, scaler)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        pred_names = le.inverse_transform(y_pred)

        # Count changes from previous round
        if prev_pred_names is not None:
            n_changed = sum(1 for a, b in zip(prev_pred_names, pred_names) if a != b)
            n_total = len(pred_names)
            print(f"      Round {round_num}: {n_changed}/{n_total} cells changed "
                  f"({100.0 * n_changed / n_total:.2f}%)")

            if n_changed == 0:
                print(f"      Converged at round {round_num}!")
                best_pred_names = pred_names.copy()
                best_proba = y_proba.copy()
                best_df = df_round.copy()
                break

            # Track best round (fewest changes = most stable)
            if n_changed < best_n_changed:
                best_n_changed = n_changed
                best_pred_names = pred_names.copy()
                best_proba = y_proba.copy()
                best_df = df_round.copy()
                n_stalled = 0
            else:
                n_stalled += 1

            # Detect oscillation: if no improvement for 3 rounds, stop
            if n_stalled >= 3:
                print(f"      Oscillation detected (no improvement for 3 rounds), "
                      f"using best round with {best_n_changed} changes.")
                break

            prev_n_changed = n_changed
        else:
            print(f"      Round 1: initial prediction ({len(pred_names)} cells)")
            best_pred_names = pred_names.copy()
            best_proba = y_proba.copy()
            best_df = df_round.copy()

        prev_pred_names = pred_names.copy()

    else:
        print(f"      Reached max rounds ({max_rounds}), "
              f"using best round with {best_n_changed} changes.")

    return best_df, best_pred_names, best_proba


def load_image(tif_path):
    """Read a TIFF and return it as an RGB uint8 array (z-stacks max-projected)."""
    img_raw = imread(str(tif_path))
    return ensure_rgb_uint8(img_raw, stack_mode="max")


# ═══════════════════════════════════════════════════════════════
# PIPELINE STAGES
#
# predict_single_tif() below runs these in order. They are also exposed
# separately so a caller that pays for GPU time by the second (a web app on
# ZeroGPU, say) can wrap only stage_segment and stage_embed, the two steps
# that actually touch the GPU, and run the rest on CPU.
# ═══════════════════════════════════════════════════════════════

def stage_segment(img_rgb, gpu=True, cellpose_model=None):
    """GPU stage 1: Cellpose-SAM. Returns the int32 label mask."""
    return segment_cellpose_sam(img_rgb, use_gpu=gpu, model=cellpose_model)


def stage_features(masks, img_rgb, um_per_px=1.0, verbose=True):
    """CPU stage: tissue mask, layer index, debris removal, handcrafted
    features.

    Debris removal edits ``masks`` in place, so the (possibly modified) mask is
    returned alongside the feature table.

    Returns ``(masks, df_base, layer_lookup, adjacency, n_layers)``, or
    ``(masks, None, ...)`` if every cell was debris.
    """
    if verbose:
        print("    Computing layer index...")
    tissue = build_tissue_mask(masks)
    layer_lookup, n_layers, adjacency = compute_layer_index_edt(masks, tissue)
    if verbose:
        print(f"    {n_layers} layers")

    # Filter out debris cells outside the main tissue body
    n_cells = int(masks.max())
    debris_cells = set()
    for cid in range(1, n_cells + 1):
        cell_px = masks == cid
        n_total = cell_px.sum()
        if n_total == 0:
            continue
        n_in_tissue = (cell_px & tissue).sum()
        if n_in_tissue / n_total < 0.5:
            debris_cells.add(cid)
    if debris_cells:
        if verbose:
            print(f"    Removed {len(debris_cells)} debris cells outside tissue")
        for cid in debris_cells:
            masks[masks == cid] = 0
            layer_lookup.pop(cid, None)
        n_cells = int(masks.max())

    if n_cells == 0:
        return masks, None, layer_lookup, adjacency, n_layers

    if verbose:
        print("    Extracting features...")
    df_base = extract_all_features(masks, img_rgb, um_per_px, layer_lookup,
                                   adjacency, tissue_mask=tissue)
    df_base["n_layers_total"] = n_layers
    return masks, df_base, layer_lookup, adjacency, n_layers


def stage_embed(masks, img_rgb, df_base, gpu=True, cnn_weights=None,
                dinov2_model=None, verbose=True):
    """GPU stage 2: fine-tuned DINOv2 per-cell embeddings, merged into
    ``df_base``. Returns the merged table (unchanged if embeddings are
    unavailable)."""
    if verbose:
        print("    Extracting CNN embeddings...")
    cnn_df = extract_cnn_embedding_features(masks, img_rgb, use_gpu=gpu,
                                             weights_path=cnn_weights,
                                             model=dinov2_model)
    if cnn_df is not None:
        df_base = df_base.merge(cnn_df, on="cell_id", how="left")
        emb_cols = [c for c in df_base.columns if c.startswith("cnn_emb_")]
        df_base[emb_cols] = df_base[emb_cols].fillna(0.0)
    return df_base


def stage_classify(df_base, masks, img_rgb, layer_lookup, adjacency,
                   models_dict, scalers, feature_cols, le,
                   out_dir="predictions", stem="image", source_name=None,
                   um_per_px=1.0, max_rounds=10, label_cells=False):
    """CPU stage: iterative prediction with each model plus the weighted
    ensemble, anatomical post-processing, and per-model CSV + overlay PNG.

    Returns the concatenated per-cell table across all models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_name is None:
        source_name = f"{stem}.tif"

    all_dfs = []
    model_probas = {}  # model_name -> (cell_ids, y_proba)
    for model_name, model in models_dict.items():
        print(f"\n    --- {model_name} (iterative, max {max_rounds} rounds) ---")

        model_scaler = scalers[model_name]
        df, pred_names, y_proba = iterative_predict(
            model, df_base, adjacency, feature_cols, model_scaler, le,
            max_rounds=max_rounds,
        )

        pred_conf = y_proba.max(axis=1)
        df["predicted_cell_type"] = pred_names
        df["prediction_confidence"] = np.round(pred_conf, 4)
        df["source_file"] = source_name
        df["um_per_px"] = um_per_px
        df["model"] = model_name

        # Report (before post-processing)
        print(f"    Model predictions:")
        for ct, cnt in pd.Series(pred_names).value_counts().items():
            print(f"      {ct}: {cnt}")

        # Anatomical post-processing
        rf_fallback = None
        if model_name != "RandomForest" and "RandomForest" in model_probas:
            rf_cids, rf_proba_arr = model_probas["RandomForest"]
            rf_class_names = list(le.classes_)
            rf_fallback = {}
            for i, cid in enumerate(rf_cids):
                rf_fallback[int(cid)] = dict(zip(rf_class_names, rf_proba_arr[i]))

        print(f"    Applying anatomical post-processing...")
        df, cell_predictions = anatomical_postprocess(
            df, layer_lookup, adjacency, masks, le=le, y_proba=y_proba,
            rf_proba_lookup=rf_fallback
        )

        # Report (after post-processing)
        final_preds = df["predicted_cell_type"].values
        print(f"    Final predictions:")
        for ct, cnt in pd.Series(final_preds).value_counts().items():
            print(f"      {ct}: {cnt}")

        # Count vascular poles
        final_labels = {}
        for cid, pred in cell_predictions.items():
            final_labels[cid] = CELL_CLASSES.get(pred, -1)
        pole_info = count_vascular_poles(final_labels, adjacency, masks)
        n_ph = pole_info["phloem"]["n_poles"]
        n_xy = pole_info["xylem"]["n_poles"]
        print(f"    Vascular poles: phloem={n_ph}, xylem={n_xy}")

        df["n_phloem_poles"] = n_ph
        df["n_xylem_poles"] = n_xy

        # Store probabilities for ensemble
        model_probas[model_name] = (df["cell_id"].values.copy(), y_proba.copy())

        # Save per-model CSV
        csv_path = out_dir / f"{stem}_{model_name}_predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")

        # Save per-model overlay PNG
        overlay_path = out_dir / f"{stem}_{model_name}_overlay.png"
        save_prediction_overlay(img_rgb, masks, cell_predictions,
                                overlay_path, le, label_cells=label_cells)
        print(f"    Saved: {overlay_path}")

        all_dfs.append(df)

    # -- Ensemble voting (average probabilities across models) --
    if len(model_probas) >= 2:
        print(f"\n    --- Ensemble ({len(model_probas)} models) ---")

        ref_model = list(model_probas.keys())[0]
        ref_cell_ids = model_probas[ref_model][0]
        n_classes = model_probas[ref_model][1].shape[1]

        minority_classes = {"xylem", "phloem"}
        minority_idx = set()
        for cls_name in minority_classes:
            try:
                minority_idx.add(list(le.classes_).index(cls_name))
            except ValueError:
                pass
        rf_trust_threshold = 0.20

        avg_proba = np.zeros((len(ref_cell_ids), n_classes), dtype=np.float64)
        n_models = len(model_probas)
        rf_proba = model_probas.get("RandomForest", (None, None))[1]

        if rf_proba is not None and minority_idx:
            rf_top_idx = np.argmax(rf_proba, axis=1)
            rf_top_conf = rf_proba[np.arange(len(rf_proba)), rf_top_idx]
            rf_minority_mask = np.array([
                (idx in minority_idx and conf >= rf_trust_threshold)
                for idx, conf in zip(rf_top_idx, rf_top_conf)
            ])
            n_rf_trusted = rf_minority_mask.sum()
            print(f"    RF-trust rule: {n_rf_trusted} cells where RF predicts minority class")

            for mname, (cids, proba) in model_probas.items():
                weight = np.ones(len(ref_cell_ids), dtype=np.float64)
                if mname == "RandomForest":
                    weight[rf_minority_mask] = 2.0
                avg_proba += proba * weight[:, np.newaxis]
            total_weight = np.full(len(ref_cell_ids), float(n_models), dtype=np.float64)
            total_weight[rf_minority_mask] += 1.0
            avg_proba /= total_weight[:, np.newaxis]
        else:
            for mname, (cids, proba) in model_probas.items():
                avg_proba += proba
            avg_proba /= n_models

        ens_pred_idx = np.argmax(avg_proba, axis=1)
        ens_pred_names = le.inverse_transform(ens_pred_idx)
        ens_conf = avg_proba.max(axis=1)

        df_ens = df_base.copy()
        df_ens["predicted_cell_type"] = ens_pred_names
        df_ens["prediction_confidence"] = np.round(ens_conf, 4)
        df_ens["source_file"] = source_name
        df_ens["um_per_px"] = um_per_px
        df_ens["model"] = "Ensemble"

        print(f"    Ensemble predictions:")
        for ct, cnt in pd.Series(ens_pred_names).value_counts().items():
            print(f"      {ct}: {cnt}")

        print(f"    Applying anatomical post-processing...")
        df_ens, ens_cell_predictions = anatomical_postprocess(
            df_ens, layer_lookup, adjacency, masks, le=le, y_proba=avg_proba
        )

        final_ens_preds = df_ens["predicted_cell_type"].values
        print(f"    Final predictions:")
        for ct, cnt in pd.Series(final_ens_preds).value_counts().items():
            print(f"      {ct}: {cnt}")

        ens_final_labels = {}
        for cid, pred in ens_cell_predictions.items():
            ens_final_labels[cid] = CELL_CLASSES.get(pred, -1)
        ens_pole_info = count_vascular_poles(ens_final_labels, adjacency, masks)
        n_ph_ens = ens_pole_info["phloem"]["n_poles"]
        n_xy_ens = ens_pole_info["xylem"]["n_poles"]
        print(f"    Vascular poles: phloem={n_ph_ens}, xylem={n_xy_ens}")

        df_ens["n_phloem_poles"] = n_ph_ens
        df_ens["n_xylem_poles"] = n_xy_ens

        ens_csv_path = out_dir / f"{stem}_Ensemble_predictions.csv"
        df_ens.to_csv(ens_csv_path, index=False)
        print(f"    Saved: {ens_csv_path}")

        ens_overlay_path = out_dir / f"{stem}_Ensemble_overlay.png"
        save_prediction_overlay(img_rgb, masks, ens_cell_predictions,
                                ens_overlay_path, le, label_cells=label_cells)
        print(f"    Saved: {ens_overlay_path}")

        all_dfs.append(df_ens)

    return pd.concat(all_dfs, ignore_index=True)


def predict_single_tif(tif_path, models_dict, scalers, feature_cols, le,
                        um_per_px=1.0, gpu=True, out_dir="predictions",
                        max_rounds=10, cnn_weights=None, label_cells=False,
                        cellpose_model=None, dinov2_model=None):
    """
    Full pipeline: segment -> features -> iterative predict -> overlay.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(tif_path).stem.replace(".aivia", "")

    print(f"\n  Processing: {tif_path}")

    # Load TIF
    img_rgb = load_image(tif_path)
    print(f"    Image shape: {img_rgb.shape}")

    # Segment
    print("    Segmenting (Cellpose-SAM)...")
    masks = stage_segment(img_rgb, gpu=gpu, cellpose_model=cellpose_model)
    n_cells = int(masks.max())
    print(f"    {n_cells} cells found")

    if n_cells == 0:
        print("    No cells found, skipping.")
        return None

    # BFS layer index, debris removal, handcrafted features
    masks, df_base, layer_lookup, adjacency, n_layers = stage_features(
        masks, img_rgb, um_per_px=um_per_px)
    if df_base is None:
        print("    No cells left after debris removal, skipping.")
        return None

    # CNN embeddings
    df_base = stage_embed(masks, img_rgb, df_base, gpu=gpu,
                          cnn_weights=cnn_weights, dinov2_model=dinov2_model)

    return stage_classify(
        df_base, masks, img_rgb, layer_lookup, adjacency,
        models_dict, scalers, feature_cols, le,
        out_dir=out_dir, stem=stem, source_name=Path(tif_path).name,
        um_per_px=um_per_px, max_rounds=max_rounds, label_cells=label_cells,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predict cell types on new TIF images (iterative refinement)"
    )
    parser.add_argument("--tif", default=None,
                        help="Single TIF path")
    parser.add_argument("--tif-dir", default=None,
                        help="Directory of TIF files (batch mode)")
    parser.add_argument("--model-dir", required=True,
                        help="Directory with trained model artifacts")
    parser.add_argument("--um-per-px", type=float, default=1.0,
                        help="Microns per pixel (default 1.0)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for Cellpose")
    parser.add_argument("--out-dir", default="predictions",
                        help="Output directory")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Max iterative prediction rounds (default 10)")
    parser.add_argument("--cnn-weights", default=None,
                        help="Path to fine-tuned DINOv2 backbone.pt (must "
                             "match what the models were trained on). "
                             "If omitted, uses pretrained DINOv2.")
    args = parser.parse_args()

    if not args.tif and not args.tif_dir:
        parser.error("Provide --tif for single image or --tif-dir for batch")

    print("=" * 60)
    print("CELL TYPE PREDICTION (ITERATIVE)")
    print("=" * 60)

    print("\n[1] Loading models...")
    models_dict, scalers, feature_cols, le = load_models(args.model_dir)

    if not models_dict:
        print("  ERROR: No models found. Run train_classifier.py first.")
        return

    # Collect TIF paths
    if args.tif:
        tif_paths = [Path(args.tif)]
    else:
        tif_dir = Path(args.tif_dir)
        tif_paths = sorted(tif_dir.glob("*.tif"))
        print(f"\n  Found {len(tif_paths)} TIF files in {tif_dir}")

    print(f"\n[2] Predicting cell types with {len(models_dict)} models "
          f"({', '.join(models_dict.keys())}), iterative, max {args.max_rounds} rounds...")
    all_tables = []
    for tif_path in tif_paths:
        try:
            df = predict_single_tif(
                str(tif_path), models_dict, scalers, feature_cols, le,
                um_per_px=args.um_per_px, gpu=args.gpu, out_dir=args.out_dir,
                max_rounds=args.max_rounds,
                cnn_weights=args.cnn_weights,
                label_cells=getattr(args, "label_cells", False),
            )
            if df is not None:
                all_tables.append(df)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Combined CSV (all models x all images)
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined_path = Path(args.out_dir) / "all_predictions.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined predictions: {combined_path} ({len(combined)} rows)")

    print(f"\n{'=' * 60}")
    print(f"DONE. Results in {args.out_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

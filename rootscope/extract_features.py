"""
Extract cell features from TIF/BMP pairs into a single table.
Segments cells with Cellpose-SAM, computes BFS layer index,
extracts shape/size/intensity features from TIF, and assigns
cell type from BMP color overlay.

Usage:
  # Full pipeline (all pairs)
  python extract_features.py \
    --metadata metadata_with_tif_sizes3.csv \
    --tif-dir ./tif --bmp-dir ./bmp --gpu

  # Single species/stage
  python extract_features.py \
    --metadata metadata_with_tif_sizes3.csv \
    --tif-dir ./tif --bmp-dir ./bmp \
    --species Cannum --stage Maturation --gpu

  # Quick layer-index check on a single TIF (no BMP needed)
  python extract_features.py --single tif/Sarcanum_Meristem1.aivia.tif --gpu

Output:
  - all_cell_features.csv          (all pairs combined)
  - features_{species}_{stage}.csv (per group)
  - layers_*.png                   (layer overlay per image)
  - debug_classmask_*.png          (BMP class overlay per image)
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label as ndlabel
from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import (
    binary_closing, binary_opening, remove_small_objects, disk, binary_erosion,
)
from skimage.segmentation import find_boundaries
from collections import deque, defaultdict
from cellpose import models
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CELL_CLASSES = {
    "root_cap":   0,
    "epidermis":  1,
    "exodermis":  2,
    "cortex":     3,
    "endodermis": 4,
    "pericycle":  5,
    "xylem":      6,
    "phloem":     7,
    "stele":      8,
}
LABEL_TO_NAME = {v: k for k, v in CELL_CLASSES.items()}

# HSV ranges in OpenCV scale (H:0-180, S:0-255, V:0-255)
COLOR_RANGES = {
    "phloem":     [(0, 180, 0, 30, 210, 255)],
    "cortex":     [(35, 85, 50, 255, 40, 255)],
    "epidermis":  [(100, 130, 50, 255, 80, 255)],
    "stele":      [(80, 100, 50, 255, 100, 255)],
    "exodermis":  [(20, 38, 40, 255, 50, 255)],
    "endodermis": [(5, 22, 80, 255, 60, 255)],
    "pericycle":  [(125, 155, 20, 255, 30, 255)],
    "root_cap":   [(150, 175, 40, 255, 80, 255)],
    "xylem":      [(0, 8, 150, 255, 120, 255),
                   (175, 180, 150, 255, 120, 255)],
}
COLOR_PROCESS_ORDER = [
    "phloem", "cortex", "epidermis", "stele", "exodermis",
    "endodermis", "pericycle", "root_cap", "xylem",
]


# ═══════════════════════════════════════════════════════════════
# IMAGE LOADING
# ═══════════════════════════════════════════════════════════════

def _normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    lo = float(np.percentile(img_f, 1))
    hi = float(np.percentile(img_f, 99))
    img_f = (img_f - lo) / (hi - lo + 1e-8)
    return (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_2d(img, mode="max"):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            return img
        return img[img.shape[0] // 2] if mode == "mid" else img.max(axis=0)
    if img.ndim == 4:
        return img[img.shape[0] // 2] if mode == "mid" else img.max(axis=0)
    raise ValueError(f"Unsupported ndim={img.ndim}, shape={img.shape}")


def ensure_rgb_uint8(img, stack_mode="max"):
    img2 = to_2d(img, mode=stack_mode)
    if img2.ndim == 2:
        g = _normalize_to_uint8(img2)
        return np.stack([g, g, g], axis=-1)
    if img2.ndim == 3:
        if img2.shape[-1] == 1:
            g = _normalize_to_uint8(img2[..., 0])
            return np.stack([g, g, g], axis=-1)
        return _normalize_to_uint8(img2[..., :3])
    raise ValueError(f"Unsupported shape after reduction: {img2.shape}")


def to_grayscale_float(img_rgb_uint8):
    return (0.299 * img_rgb_uint8[..., 0].astype(np.float32) +
            0.587 * img_rgb_uint8[..., 1].astype(np.float32) +
            0.114 * img_rgb_uint8[..., 2].astype(np.float32))


# ═══════════════════════════════════════════════════════════════
# METADATA PARSER
# ═══════════════════════════════════════════════════════════════

def _parse_size(s):
    m = re.match(r"(\d+)\s*x\s*(\d+)", str(s).strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _parse_scale(s):
    # Match µ (micro sign U+00B5), μ (Greek mu U+03BC), and u
    m = re.match(r"([\d.]+)\s*[uμµ\u00b5\u03bc]m/px", str(s).strip())
    return float(m.group(1)) if m else None


def parse_metadata(csv_path, tif_dir, bmp_dir):
    tif_dir = Path(tif_dir)
    bmp_dir = Path(bmp_dir)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df["bmp_w"], df["bmp_h"] = zip(*df["bmp_size"].apply(_parse_size))
    df["tif_w"], df["tif_h"] = zip(*df["tif_size"].apply(_parse_size))
    df["um_per_px"] = df["scale_ratio"].apply(_parse_scale)
    df["bmp_path"] = df["bmp_filename"].apply(lambda f: str(bmp_dir / f))
    df["tif_path"] = df["tif_matched"].apply(lambda f: str(tif_dir / f))
    print(f"  Loaded {len(df)} pairs from {csv_path}")
    return df


def get_pairs(df, species=None, stage=None):
    subset = df.copy()
    if species:
        subset = subset[subset["species"].str.lower() == species.lower()]
    if stage:
        subset = subset[subset["stage"].str.lower() == stage.lower()]
    return subset.to_dict("records")


# ═══════════════════════════════════════════════════════════════
# CELLPOSE-SAM SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def load_cellpose_model(use_gpu=True):
    """Build the Cellpose-SAM model. Expensive (downloads/loads weights), so
    long-running callers such as a web server should build it once and pass it
    to ``segment_cellpose_sam(model=...)``."""
    return models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")


def segment_cellpose_sam(img_rgb_uint8, use_gpu=True, diameter=None,
                          cellprob_threshold=0.0, flow_threshold=0.4,
                          min_size=30, model=None):
    if model is None:
        model = load_cellpose_model(use_gpu=use_gpu)
    masks, _, _ = model.eval(
        img_rgb_uint8,
        channels=[0, 0],
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
    )
    return masks.astype(np.int32)


# ═══════════════════════════════════════════════════════════════
# TISSUE MASK & BFS LAYER INDEX
# ═══════════════════════════════════════════════════════════════

def build_tissue_mask(masks, close_radius=5, open_radius=1,
                       min_tissue_obj=500):
    """
    Build a clean tissue mask with NO internal holes.
    Only the outer boundary should exist so BFS assigns layers correctly.
    """
    tissue = masks > 0

    if min_tissue_obj > 0:
        tissue = remove_small_objects(tissue, min_size=min_tissue_obj)
    if open_radius > 0:
        tissue = binary_opening(tissue, footprint=disk(open_radius))
    if close_radius > 0:
        tissue = binary_closing(tissue, footprint=disk(close_radius))

    # Keep only the largest connected component
    labeled, n_components = ndlabel(tissue)
    if n_components > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        tissue = labeled == largest

    # Fill ALL internal holes
    tissue = binary_fill_holes(tissue)
    tissue = binary_closing(tissue, footprint=disk(close_radius))
    tissue = binary_fill_holes(tissue)

    return tissue.astype(bool)


def build_cell_adjacency(masks):
    """Build adjacency graph: which cells touch which."""
    H, W = masks.shape
    adjacency = defaultdict(set)
    ys, xs = np.nonzero(find_boundaries(masks, mode="inner"))
    for y, x in zip(ys, xs):
        a = int(masks[y, x])
        if a == 0:
            continue
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                b = int(masks[ny, nx])
                if b != 0 and b != a:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
    return adjacency


def _normalized_radial_position(centroids_y, centroids_x, tissue_mask):
    """
    Compute normalized radial position for each cell: 0 = outermost, 1 = center.

    For each cell, find its angle from the tissue center, then find the
    boundary distance at that angle. Normalize cell distance by that
    local boundary distance.

    This corrects for irregular (non-circular) root shapes so that cells
    in the same concentric ring get the same normalized depth regardless
    of which side of the root they're on.
    """
    from scipy.ndimage import gaussian_filter1d as gf1d

    # Tissue center of mass
    ys, xs = np.where(tissue_mask)
    center_y = ys.mean()
    center_x = xs.mean()

    # Compute angle and distance from center for all cells
    dy = centroids_y - center_y
    dx = centroids_x - center_x
    angles = np.arctan2(dy, dx)
    dists = np.sqrt(dy**2 + dx**2)

    # Build boundary profile: 95th-pctl cell distance at each angle
    n_angle_bins = 72  # 5° per bin
    angle_bins = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
    max_dist_per_angle = np.full(n_angle_bins, np.median(dists))

    for i in range(n_angle_bins):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1])
        if mask.sum() > 0:
            max_dist_per_angle[i] = np.percentile(dists[mask], 95)

    # Circular smoothing of boundary profile
    ext = np.concatenate([max_dist_per_angle[-5:],
                          max_dist_per_angle,
                          max_dist_per_angle[:5]])
    max_dist_per_angle = gf1d(ext, sigma=2.0)[5:-5]

    # Normalize each cell's distance
    norm = np.zeros(len(centroids_y))
    for i in range(len(centroids_y)):
        bin_idx = np.clip(
            np.searchsorted(angle_bins, angles[i]) - 1, 0, n_angle_bins - 1
        )
        boundary_r = max_dist_per_angle[bin_idx]
        if boundary_r > 1.0:
            norm[i] = 1.0 - dists[i] / boundary_r
        else:
            norm[i] = 0.0

    return np.clip(norm, 0, 1)


def compute_layer_index_edt(masks, tissue_mask):
    """
    Assign layer index using EDT (Euclidean Distance Transform) directly.

    Layer 0 = outermost cells (touching tissue boundary).
    Each successive ring inward increments by 1.

    Strategy:
      1. Compute EDT from the tissue boundary (distance of each pixel
         from the nearest background pixel).
      2. For each cell, compute its median EDT distance.
      3. Estimate ring width from the median cell diameter.
      4. Bin cells into layers by dividing their EDT distance by ring width.

    This avoids BFS spiraling by assigning layers purely from geometry.
    """
    adjacency = build_cell_adjacency(masks)

    props = regionprops(masks)
    if not props:
        return {}, 0, adjacency

    # EDT: distance from tissue boundary (0 at edge, increases inward)
    edt = ndimage.distance_transform_edt(tissue_mask)

    # For each cell, compute median EDT distance of its pixels
    cids = []
    edt_dists = []
    areas = []
    for p in props:
        cid = p.label
        cell_pixels = masks == cid
        med_dist = float(np.median(edt[cell_pixels]))
        cids.append(int(cid))
        edt_dists.append(med_dist)
        areas.append(p.area)

    cids = np.array(cids)
    edt_dists = np.array(edt_dists)
    areas = np.array(areas)

    # Estimate ring width = median cell diameter (equivalent circle)
    med_diam = float(np.median(np.sqrt(areas / np.pi) * 2))
    # Use 0.8x diameter as ring step (cells overlap slightly between rings)
    ring_width = max(med_diam * 0.8, 3.0)

    # Assign layer = floor(edt_distance / ring_width)
    layer_lookup = {}
    for i in range(len(cids)):
        layer_lookup[int(cids[i])] = int(edt_dists[i] / ring_width)

    n_layers = max(layer_lookup.values()) + 1 if layer_lookup else 0
    return layer_lookup, n_layers, adjacency


# ═══════════════════════════════════════════════════════════════
# BMP COLOR TO CLASS LABEL
# ═══════════════════════════════════════════════════════════════

def create_class_mask_from_bmp(bmp_path):
    bmp_rgb = np.array(Image.open(str(bmp_path)).convert("RGB"))
    bmp_hsv = cv2.cvtColor(bmp_rgb, cv2.COLOR_RGB2HSV)

    h, w = bmp_hsv.shape[:2]
    class_mask = np.full((h, w), -1, dtype=np.int8)
    confidence = np.zeros((h, w), dtype=np.float32)
    sat = bmp_hsv[:, :, 1].astype(np.float32) / 255.0
    val = bmp_hsv[:, :, 2].astype(np.float32) / 255.0

    for class_name in COLOR_PROCESS_ORDER:
        ranges = COLOR_RANGES[class_name]
        label = CELL_CLASSES[class_name]
        combined = np.zeros((h, w), dtype=bool)
        for (h_lo, h_hi, s_lo, s_hi, v_lo, v_hi) in ranges:
            lower = np.array([h_lo, s_lo, v_lo])
            upper = np.array([h_hi, s_hi, v_hi])
            combined |= cv2.inRange(bmp_hsv, lower, upper) > 0

        conf = val * (1.0 - sat) if class_name == "phloem" else sat * val
        update = combined & (conf >= confidence)
        class_mask[update] = label
        confidence[update] = conf[update]

    return class_mask


def resize_class_mask_to_tif(class_mask, tif_h, tif_w):
    return cv2.resize(
        class_mask.astype(np.float32), (tif_w, tif_h),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int8)


def assign_cell_type(cell_masks, class_mask_resized):
    cell_ids = np.unique(cell_masks)
    cell_ids = cell_ids[cell_ids > 0]
    labels = {}
    confidences = {}
    for cid in cell_ids:
        pixels = class_mask_resized[cell_masks == cid]
        valid = pixels[pixels >= 0]
        if len(valid) == 0:
            labels[cid] = -1
            confidences[cid] = 0.0
            continue
        counts = np.bincount(valid.astype(int), minlength=9)
        best = counts.argmax()
        labels[cid] = int(best)
        confidences[cid] = float(counts[best]) / len(valid)
    return labels, confidences


# ═══════════════════════════════════════════════════════════════
# VASCULAR POLE COUNTING
# ═══════════════════════════════════════════════════════════════

def count_vascular_poles(cell_type_labels, adjacency, masks):
    """
    Count phloem and xylem poles (spatially connected clusters).
    A pole is a group of adjacent cells of the same vascular type.

    Returns:
        n_phloem_poles, n_xylem_poles,
        phloem_pole_map (cell_id -> pole_id),
        xylem_pole_map (cell_id -> pole_id),
        phloem_pole_centroids [(cy, cx), ...],
        xylem_pole_centroids [(cy, cx), ...]
    """
    props_lookup = {p.label: p for p in regionprops(masks)}

    results = {}
    for vtype, vlabel in [("phloem", CELL_CLASSES["phloem"]),
                          ("xylem", CELL_CLASSES["xylem"])]:
        # Find cells of this type
        type_cells = {cid for cid, lbl in cell_type_labels.items()
                      if lbl == vlabel}

        # BFS to find connected clusters via adjacency
        visited = set()
        pole_map = {}
        pole_centroids = []
        pole_id = 0

        for start in type_cells:
            if start in visited:
                continue
            # BFS from this cell
            cluster = []
            queue = deque([start])
            visited.add(start)
            while queue:
                cid = queue.popleft()
                cluster.append(cid)
                pole_map[cid] = pole_id
                for nbr in adjacency.get(cid, set()):
                    if nbr in type_cells and nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)

            # Compute centroid of this pole
            cys, cxs = [], []
            for cid in cluster:
                if cid in props_lookup:
                    cy, cx = props_lookup[cid].centroid
                    cys.append(cy)
                    cxs.append(cx)
            if cys:
                pole_centroids.append((np.mean(cys), np.mean(cxs)))
            pole_id += 1

        results[vtype] = {
            "n_poles": pole_id,
            "pole_map": pole_map,
            "pole_centroids": pole_centroids,
        }

    return results


def compute_pole_features(cell_id, centroid_y, centroid_x, pole_info):
    """Compute per-cell features related to vascular poles."""
    phloem_info = pole_info.get("phloem", {})
    xylem_info = pole_info.get("xylem", {})

    n_phloem_poles = phloem_info.get("n_poles", 0)
    n_xylem_poles = xylem_info.get("n_poles", 0)

    # Distance to nearest phloem pole centroid
    phloem_centroids = phloem_info.get("pole_centroids", [])
    if phloem_centroids:
        dists = [np.sqrt((centroid_y - cy)**2 + (centroid_x - cx)**2)
                 for cy, cx in phloem_centroids]
        dist_to_nearest_phloem = round(float(min(dists)), 2)
    else:
        dist_to_nearest_phloem = -1.0

    # Distance to nearest xylem pole centroid
    xylem_centroids = xylem_info.get("pole_centroids", [])
    if xylem_centroids:
        dists = [np.sqrt((centroid_y - cy)**2 + (centroid_x - cx)**2)
                 for cy, cx in xylem_centroids]
        dist_to_nearest_xylem = round(float(min(dists)), 2)
    else:
        dist_to_nearest_xylem = -1.0

    return {
        "n_phloem_poles": n_phloem_poles,
        "n_xylem_poles": n_xylem_poles,
        "dist_to_nearest_phloem_pole": dist_to_nearest_phloem,
        "dist_to_nearest_xylem_pole": dist_to_nearest_xylem,
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_all_features(masks, img_rgb, um_per_px, layer_lookup, adjacency,
                         tissue_mask=None, pole_info=None):
    from scipy.stats import skew, kurtosis

    gray = to_grayscale_float(img_rgb)

    # Root center = center-of-mass of all cells
    all_cells = masks > 0
    if all_cells.any():
        img_cy, img_cx = ndimage.center_of_mass(all_cells)
    else:
        img_cy, img_cx = masks.shape[0] / 2, masks.shape[1] / 2

    ys, xs = np.where(all_cells)
    max_radius = np.sqrt((ys - img_cy)**2 + (xs - img_cx)**2).max() if len(ys) > 0 else 1.0

    # EDT from tissue boundary (continuous distance, better than layer_index for ML)
    if tissue_mask is not None:
        edt = ndimage.distance_transform_edt(tissue_mask)
        edt_max = float(edt.max()) if edt.max() > 0 else 1.0
    else:
        edt = None
        edt_max = 1.0

    props = regionprops(masks, intensity_image=gray)

    # Pre-compute per-cell areas, layers, intensities, and centroids for neighbor features
    cell_areas = {}
    cell_intensities = {}
    cell_centroids = {}  # {cid: (cy, cx)}
    for p in props:
        cell_areas[p.label] = p.area
        cell_px = masks == p.label
        cell_intensities[p.label] = float(gray[cell_px].mean())
        cell_centroids[p.label] = p.centroid  # (row, col) = (y, x)

    # Check if image is truly multi-channel (not grayscale duplicated to RGB)
    is_color = not np.array_equal(img_rgb[..., 0], img_rgb[..., 1])
    r_ch = img_rgb[..., 0].astype(np.float32)
    g_ch = img_rgb[..., 1].astype(np.float32)
    b_ch = img_rgb[..., 2].astype(np.float32)

    # Pre-compute n_layers for layer_fraction
    n_layers = max(layer_lookup.values()) + 1 if layer_lookup else 1

    # Pre-compute per-layer cell counts and areas (thin ring vs thick cortex)
    layer_cell_counts = defaultdict(int)
    layer_cell_areas = defaultdict(list)
    for cid, lv in layer_lookup.items():
        if lv >= 0:
            layer_cell_counts[lv] += 1
            if cid in cell_areas:
                layer_cell_areas[lv].append(cell_areas[cid])

    # Pre-compute max cells in any layer (for thin-ring detection)
    max_layer_cell_count = max(layer_cell_counts.values()) if layer_cell_counts else 1

    # Global area stats for relative size features
    all_areas = np.array([p.area for p in props])
    global_median_area = float(np.median(all_areas)) if len(all_areas) > 0 else 1.0
    global_mean_area = float(np.mean(all_areas)) if len(all_areas) > 0 else 1.0
    global_std_area = float(np.std(all_areas)) if len(all_areas) > 0 else 1.0

    # Pre-compute which cells touch background (no cell = masks==0)
    # This directly checks each cell's border pixels for adjacency to empty space.
    # Epidermis cells always touch background; exodermis/endodermis do NOT.
    H, W = masks.shape
    touches_bg = set()

    # For each cell, check if any pixel at its inner boundary is adjacent to masks==0
    inner_boundary = find_boundaries(masks, mode="inner")
    bys, bxs = np.nonzero(inner_boundary)
    for by, bx in zip(bys, bxs):
        cid_here = int(masks[by, bx])
        if cid_here == 0 or cid_here in touches_bg:
            continue
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = by + dy, bx + dx
            if 0 <= ny < H and 0 <= nx < W:
                if masks[ny, nx] == 0:
                    touches_bg.add(cid_here)
                    break
            else:
                # Cell is at image edge = touches background
                touches_bg.add(cid_here)
                break

    # Pre-compute which cells neighbor a boundary cell (= exodermis signal)
    # Exodermis is always adjacent to epidermis (which touches background)
    neighbors_boundary = set()
    for cid in touches_bg:
        for nbr in adjacency.get(cid, set()):
            if nbr not in touches_bg:
                neighbors_boundary.add(nbr)

    # Pre-compute Sobel gradient magnitude for cell wall contrast feature
    _sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    _sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(_sobel_x**2 + _sobel_y**2)

    records = []
    for p in props:
        cid = p.label
        cy, cx = p.centroid
        dist_px = np.sqrt((cy - img_cy)**2 + (cx - img_cx)**2)

        area = p.area
        perim = p.perimeter if p.perimeter > 0 else 1.0
        compactness = (4.0 * np.pi * area) / (perim ** 2)

        major = p.major_axis_length if p.major_axis_length > 0 else 1.0
        minor = p.minor_axis_length if p.minor_axis_length > 0 else 1.0

        cell_px = masks == cid
        gray_vals = gray[cell_px]

        # EDT-based depth features
        if edt is not None:
            edt_vals = edt[cell_px]
            edt_median = float(np.median(edt_vals))
            edt_normalized = round(edt_median / edt_max, 4)
        else:
            edt_median = 0.0
            edt_normalized = 0.0

        # --- Neighbor context features ---
        neighbors = adjacency.get(cid, set())
        n_neighbors = len(neighbors)
        neighbor_layers = [layer_lookup.get(n, -1) for n in neighbors if layer_lookup.get(n, -1) >= 0]
        neighbor_areas = [cell_areas.get(n, 0) for n in neighbors if n in cell_areas]

        mean_neighbor_layer = float(np.mean(neighbor_layers)) if neighbor_layers else -1.0
        std_neighbor_layer = float(np.std(neighbor_layers)) if len(neighbor_layers) > 1 else 0.0
        mean_neighbor_area = float(np.mean(neighbor_areas)) if neighbor_areas else 0.0

        my_layer = layer_lookup.get(cid, -1)
        layer_fraction = round(my_layer / max(n_layers - 1, 1), 4) if my_layer >= 0 else 0.0
        is_boundary_cell = 1 if my_layer == 0 else 0

        # Layer difference from neighbors (helps distinguish tissue boundaries)
        layer_diff_from_neighbors = float(np.mean([abs(my_layer - nl) for nl in neighbor_layers])) if neighbor_layers and my_layer >= 0 else 0.0

        # Area ratio vs neighbors (tissue types have characteristic size differences)
        area_ratio_to_neighbors = round(area / mean_neighbor_area, 4) if mean_neighbor_area > 0 else 1.0

        # --- Ring topology features (epidermis/exodermis/endodermis) ---

        # 1. Does this cell touch the tissue background?
        #    Epidermis = YES, exodermis/endodermis = NO
        cell_touches_background = 1 if cid in touches_bg else 0

        # 2. How many cells share this layer? Thin rings (epi/exo/endo)
        #    have fewer cells than thick cortex
        cells_in_same_layer = layer_cell_counts.get(my_layer, 0) if my_layer >= 0 else 0

        # 3. Neighbors inward vs outward — ring cells have neighbors
        #    on both sides; cortex cells mostly have same-layer neighbors
        inner_neighbor_count = 0
        outer_neighbor_count = 0
        same_layer_neighbor_count = 0
        if my_layer >= 0:
            for nl in neighbor_layers:
                if nl > my_layer:
                    inner_neighbor_count += 1
                elif nl < my_layer:
                    outer_neighbor_count += 1
                else:
                    same_layer_neighbor_count += 1

        frac_neighbors_same_layer = round(
            same_layer_neighbor_count / max(n_neighbors, 1), 4
        )
        frac_neighbors_inner = round(
            inner_neighbor_count / max(n_neighbors, 1), 4
        )
        frac_neighbors_outer = round(
            outer_neighbor_count / max(n_neighbors, 1), 4
        )

        # 4. Layer from inside (endodermis is always close to center)
        layer_from_inside = (n_layers - 1 - my_layer) if my_layer >= 0 else -1

        # 5. Intensity gradient: difference between this cell and
        #    inner/outer neighbors (structural tissue differences)
        my_intensity = cell_intensities.get(cid, 0.0)
        inner_nbr_intensities = [cell_intensities.get(n, 0) for n in neighbors
                                  if layer_lookup.get(n, -1) > my_layer and n in cell_intensities]
        outer_nbr_intensities = [cell_intensities.get(n, 0) for n in neighbors
                                  if layer_lookup.get(n, -1) < my_layer and n in cell_intensities]
        radial_intensity_gradient = 0.0
        if inner_nbr_intensities and outer_nbr_intensities:
            radial_intensity_gradient = float(np.mean(inner_nbr_intensities)) - float(np.mean(outer_nbr_intensities))
        elif inner_nbr_intensities:
            radial_intensity_gradient = float(np.mean(inner_nbr_intensities)) - my_intensity
        elif outer_nbr_intensities:
            radial_intensity_gradient = my_intensity - float(np.mean(outer_nbr_intensities))

        # 6. Neighbor layer range (ring cells span exactly 2 adjacent layers,
        #    cortex cells often have neighbors all at the same layer)
        neighbor_layer_range = (max(neighbor_layers) - min(neighbor_layers)) if neighbor_layers else 0

        # 7. Is this cell adjacent to a boundary cell?
        #    Exodermis = YES (neighbors epidermis), cortex/endodermis = NO
        cell_neighbors_boundary = 1 if cid in neighbors_boundary else 0

        # 8. Min layer among neighbors — exodermis neighbors include
        #    epidermis (layer 0-1), cortex neighbors are all mid-layers
        min_neighbor_layer = min(neighbor_layers) if neighbor_layers else -1
        max_neighbor_layer = max(neighbor_layers) if neighbor_layers else -1

        # 9. How many of this cell's neighbors touch background?
        #    Epidermis: many neighbors also touch bg; exodermis: some do;
        #    deeper cells: none do
        n_neighbors_touching_bg = sum(1 for n in neighbors if n in touches_bg)

        # --- Texture features ---
        intensity_cv = round(float(gray_vals.std() / (gray_vals.mean() + 1e-8)), 4)
        intensity_skewness = round(float(skew(gray_vals)), 4) if len(gray_vals) > 2 else 0.0
        intensity_kurtosis = round(float(kurtosis(gray_vals)), 4) if len(gray_vals) > 2 else 0.0

        # Intensity percentiles (more robust than min/max)
        intensity_p10 = round(float(np.percentile(gray_vals, 10)), 2)
        intensity_p90 = round(float(np.percentile(gray_vals, 90)), 2)

        # --- Shape features ---
        area_perimeter_ratio = round(area / perim, 4)
        equivalent_diameter = round(float(p.equivalent_diameter), 2)

        # Angular position around root center (helps distinguish radially asymmetric tissues)
        angular_position = round(float(np.arctan2(cy - img_cy, cx - img_cx)), 4)

        # --- Enhanced polar coordinate features ---
        # Cyclical encoding of angle (avoids discontinuity at -pi/+pi)
        sin_angular = round(float(np.sin(angular_position)), 4)
        cos_angular = round(float(np.cos(angular_position)), 4)
        norm_r = dist_px / max_radius if max_radius > 0 else 0
        radial_x_sin = round(norm_r * sin_angular, 4)
        radial_x_cos = round(norm_r * cos_angular, 4)

        # --- Directional neighbor features (polar-based) ---
        # Classify neighbors as inward/outward (radially) or CW/CCW (tangentially)
        # relative to root center, then summarize their morphology
        inward_nbr_areas = []
        outward_nbr_areas = []
        cw_nbr_areas = []
        ccw_nbr_areas = []
        inward_nbr_intensities = []
        outward_nbr_intensities = []
        cw_nbr_intensities = []
        ccw_nbr_intensities = []

        my_angle = np.arctan2(cy - img_cy, cx - img_cx)
        my_dist = dist_px

        for nbr in neighbors:
            if nbr not in cell_centroids:
                continue
            ny, nx = cell_centroids[nbr]
            nbr_dist = np.sqrt((ny - img_cy)**2 + (nx - img_cx)**2)
            nbr_angle = np.arctan2(ny - img_cy, nx - img_cx)
            nbr_area = cell_areas.get(nbr, 0)
            nbr_intens = cell_intensities.get(nbr, 0.0)

            # Radial classification: inward (closer to center) vs outward
            if nbr_dist < my_dist - 1.0:
                inward_nbr_areas.append(nbr_area)
                inward_nbr_intensities.append(nbr_intens)
            elif nbr_dist > my_dist + 1.0:
                outward_nbr_areas.append(nbr_area)
                outward_nbr_intensities.append(nbr_intens)

            # Tangential classification: clockwise vs counter-clockwise
            angle_diff = nbr_angle - my_angle
            # Normalize to [-pi, pi]
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            if angle_diff > 0.01:    # CCW
                ccw_nbr_areas.append(nbr_area)
                ccw_nbr_intensities.append(nbr_intens)
            elif angle_diff < -0.01:  # CW
                cw_nbr_areas.append(nbr_area)
                cw_nbr_intensities.append(nbr_intens)

        radial_inward_nbr_area = round(float(np.mean(inward_nbr_areas)), 2) if inward_nbr_areas else 0.0
        radial_outward_nbr_area = round(float(np.mean(outward_nbr_areas)), 2) if outward_nbr_areas else 0.0
        cw_nbr_area = round(float(np.mean(cw_nbr_areas)), 2) if cw_nbr_areas else 0.0
        ccw_nbr_area = round(float(np.mean(ccw_nbr_areas)), 2) if ccw_nbr_areas else 0.0
        radial_inward_nbr_intensity = round(float(np.mean(inward_nbr_intensities)), 4) if inward_nbr_intensities else 0.0
        radial_outward_nbr_intensity = round(float(np.mean(outward_nbr_intensities)), 4) if outward_nbr_intensities else 0.0
        cw_nbr_intensity = round(float(np.mean(cw_nbr_intensities)), 4) if cw_nbr_intensities else 0.0
        ccw_nbr_intensity = round(float(np.mean(ccw_nbr_intensities)), 4) if ccw_nbr_intensities else 0.0

        # (Polar-direction neighbor cell types are computed as a
        # post-processing step via compute_neighbor_celltypes(), not here.)

        # --- Exodermis-specific features ---
        # Exodermis cells are: large, hexagonal, single-layer ring

        # 1. Relative size: exodermis cells are BIGGER than other cell types
        area_zscore = round((area - global_mean_area) / max(global_std_area, 1.0), 4)
        area_ratio_to_global_median = round(area / max(global_median_area, 1.0), 4)

        # 2. Size rank within same layer (exodermis cells are among the largest)
        layer_areas = layer_cell_areas.get(my_layer, [])
        if layer_areas and len(layer_areas) > 1:
            area_percentile_in_layer = round(
                float(np.searchsorted(np.sort(layer_areas), area)) / len(layer_areas), 4
            )
        else:
            area_percentile_in_layer = 0.5

        # 3. Size ratio to inner/outer neighbor cells
        inner_nbr_areas = [cell_areas.get(n, 0) for n in neighbors
                           if layer_lookup.get(n, -1) > my_layer and n in cell_areas]
        outer_nbr_areas = [cell_areas.get(n, 0) for n in neighbors
                           if layer_lookup.get(n, -1) < my_layer and n in cell_areas]
        area_ratio_to_inner = round(
            area / max(float(np.mean(inner_nbr_areas)), 1.0), 4
        ) if inner_nbr_areas else 1.0
        area_ratio_to_outer = round(
            area / max(float(np.mean(outer_nbr_areas)), 1.0), 4
        ) if outer_nbr_areas else 1.0

        # 4. Hexagonality: regular hexagon has compactness ≈ 0.9069
        #    and ~6 neighbors, low eccentricity, high solidity
        HEXAGON_COMPACTNESS = 0.9069
        hexagonality = round(1.0 - abs(compactness - HEXAGON_COMPACTNESS), 4)

        # 5. Convex hull vertex count — hexagons have ~6 vertices
        try:
            from skimage.measure import approximate_polygon
            coords = p.coords  # (N, 2) array of pixel coordinates
            hull = p.convex_image
            # Count corners of convex hull
            from skimage.measure import find_contours
            contours = find_contours(hull.astype(float), 0.5)
            if contours:
                approx = approximate_polygon(contours[0], tolerance=2.0)
                n_vertices = len(approx) - 1  # -1 because first==last
            else:
                n_vertices = 0
        except Exception:
            n_vertices = 0

        # --- Thin-ring / boundary features (endodermis & exodermis) ---

        # 1. Ratio of cells in this layer to the thickest layer
        #    Endodermis/exodermis rings are thin → low ratio; cortex is thick → high ratio
        layer_cell_count_ratio = round(
            cells_in_same_layer / max(max_layer_cell_count, 1), 4
        ) if my_layer >= 0 else 0.0

        # 2. Cell counts in adjacent inner/outer layers
        #    At endodermis: outer layer (cortex) has MANY cells, inner (pericycle) has FEW
        #    At exodermis: outer layer (epidermis) has fewer, inner (cortex) has MANY
        inner_layer = my_layer + 1 if my_layer >= 0 else -1
        outer_layer = my_layer - 1 if my_layer >= 0 else -1
        inner_layer_cell_count = layer_cell_counts.get(inner_layer, 0) if inner_layer >= 0 else 0
        outer_layer_cell_count = layer_cell_counts.get(outer_layer, 0) if outer_layer >= 0 else 0

        # 3. Gradient in cell count: big jump means tissue boundary
        #    Endodermis: outer has many cortex cells, inner has few pericycle → large positive
        #    Exodermis: outer has fewer epidermis, inner has many cortex → large negative
        layer_count_gradient = outer_layer_cell_count - inner_layer_cell_count

        # 4. Am I between layers of very different cell counts?
        #    Ring cells sit between a thick layer and a thin layer
        layer_count_asymmetry = round(
            abs(outer_layer_cell_count - inner_layer_cell_count) /
            max(outer_layer_cell_count + inner_layer_cell_count, 1), 4
        )

        # 5. Mean area in adjacent layers (endodermis neighbors large cortex + small pericycle)
        inner_layer_mean_area = float(np.mean(layer_cell_areas.get(inner_layer, [0]))) if inner_layer >= 0 and inner_layer in layer_cell_areas else 0.0
        outer_layer_mean_area = float(np.mean(layer_cell_areas.get(outer_layer, [0]))) if outer_layer >= 0 and outer_layer in layer_cell_areas else 0.0
        adjacent_layer_area_ratio = round(
            outer_layer_mean_area / max(inner_layer_mean_area, 1.0), 4
        ) if inner_layer_mean_area > 0 else 1.0

        # --- Vascular distinction features (xylem vs stele) ---

        # 1. Wall thickness proxy: mean intensity of boundary pixels (2px inner ring)
        cell_mask = (masks == cid)
        eroded_mask = binary_erosion(cell_mask, disk(2))
        wall_ring = cell_mask & ~eroded_mask
        wall_pixels = gray[wall_ring]
        wall_thickness_proxy = round(float(wall_pixels.mean()), 2) if wall_pixels.size > 0 else 0.0

        # 2. Wall-to-lumen ratio: wall mean intensity / interior mean intensity
        interior_pixels = gray[eroded_mask]
        if interior_pixels.size > 0:
            wall_to_lumen_ratio = round(
                float(wall_pixels.mean()) / max(float(interior_pixels.mean()), 1e-8), 4
            ) if wall_pixels.size > 0 else 1.0
        else:
            wall_to_lumen_ratio = 1.0

        # 3. Local area rank in stele: percentile of area among inner 40% of layers
        if n_layers > 0 and layer_from_inside >= 0 and (layer_from_inside / n_layers) < 0.4:
            inner_cell_areas = []
            for lv, areas_list in layer_cell_areas.items():
                lfi = (n_layers - 1 - lv) if lv >= 0 else -1
                if lfi >= 0 and (lfi / n_layers) < 0.4:
                    inner_cell_areas.extend(areas_list)
            if len(inner_cell_areas) > 1:
                local_area_rank_in_stele = round(
                    float(np.searchsorted(np.sort(inner_cell_areas), area)) / len(inner_cell_areas), 4
                )
            else:
                local_area_rank_in_stele = 0.5
        else:
            local_area_rank_in_stele = 0.5

        # 4. Neighbor area std: heterogeneous neighbors suggest phloem
        neighbor_area_std = round(float(np.std(neighbor_areas)), 2) if len(neighbor_areas) > 1 else 0.0

        # 5. Cell wall contrast: mean Sobel gradient magnitude at cell boundary
        wall_sobel = sobel_mag[wall_ring]
        cell_wall_contrast = round(float(wall_sobel.mean()), 2) if wall_sobel.size > 0 else 0.0

        # --- Enhanced xylem vs stele features ---

        # 6. Lumen darkness: mean intensity of deep interior (erode 3px)
        #    Xylem vessels have large dark lumens; stele parenchyma is more uniform
        eroded_mask_3 = binary_erosion(cell_mask, disk(3))
        deep_interior = gray[eroded_mask_3]
        lumen_darkness = round(float(deep_interior.mean()), 2) if deep_interior.size > 0 else float(gray_vals.mean())

        # 7. Wall-lumen intensity gap: difference between wall and deep interior
        #    Xylem has high gap (bright wall, dark lumen); stele has low gap
        wall_lumen_gap = round(float(wall_pixels.mean()) - lumen_darkness, 2) if wall_pixels.size > 0 else 0.0

        # 8. Interior intensity std: variation inside the cell
        #    Xylem vessels may have heterogeneous interiors (secondary wall patterns)
        interior_intensity_std = round(float(interior_pixels.std()), 2) if interior_pixels.size > 1 else 0.0

        # 9. Fraction of dark interior pixels: fraction below 25th percentile of image
        #    Xylem lumens are often the darkest structures
        if deep_interior.size > 0:
            dark_threshold = float(np.percentile(gray, 25))
            frac_dark_interior = round(float((deep_interior < dark_threshold).sum()) / deep_interior.size, 4)
        else:
            frac_dark_interior = 0.0

        # 10. Wall ring intensity at multiple erosion depths (profile)
        #     Xylem: intensity drops sharply from wall → interior (thick wall, empty lumen)
        #     Stele: intensity is relatively uniform across rings
        eroded_1 = binary_erosion(cell_mask, disk(1))
        ring_1 = cell_mask & ~eroded_1  # outermost 1px ring
        ring_2 = eroded_1 & ~eroded_mask  # 1-2px ring
        ring1_intensity = round(float(gray[ring_1].mean()), 2) if gray[ring_1].size > 0 else 0.0
        ring2_intensity = round(float(gray[ring_2].mean()), 2) if gray[ring_2].size > 0 else 0.0
        # Radial intensity drop from wall to interior
        wall_interior_gradient = round(ring1_intensity - lumen_darkness, 2)

        # 11. Neighbor wall thickness similarity: mean wall_thickness of neighbors
        #     Xylem cells cluster with other thick-walled cells
        nbr_wall_vals = []
        for nbr in adjacency.get(cid, set()):
            nbr_mask = (masks == nbr)
            nbr_eroded = binary_erosion(nbr_mask, disk(2))
            nbr_wall = nbr_mask & ~nbr_eroded
            nbr_wall_px = gray[nbr_wall]
            if nbr_wall_px.size > 0:
                nbr_wall_vals.append(float(nbr_wall_px.mean()))
        mean_neighbor_wall_thickness = round(float(np.mean(nbr_wall_vals)), 2) if nbr_wall_vals else 0.0
        wall_thickness_vs_neighbors = round(wall_thickness_proxy - mean_neighbor_wall_thickness, 2)

        # 12. Stele-relative area: ratio of this cell's area to mean area
        #     of cells in the inner 40% of layers (xylem vessels are often larger)
        if n_layers > 0 and layer_from_inside >= 0 and (layer_from_inside / n_layers) < 0.4:
            inner_cell_areas_all = []
            for lv, areas_list in layer_cell_areas.items():
                lfi = (n_layers - 1 - lv) if lv >= 0 else -1
                if lfi >= 0 and (lfi / n_layers) < 0.4:
                    inner_cell_areas_all.extend(areas_list)
            stele_mean_area = float(np.mean(inner_cell_areas_all)) if inner_cell_areas_all else float(area)
            area_ratio_to_stele_mean = round(float(area) / max(stele_mean_area, 1.0), 4)
        else:
            area_ratio_to_stele_mean = 1.0

        rec = {
            "cell_id": int(cid),
            "layer_index": my_layer,
            "centroid_y": round(cy, 2),
            "centroid_x": round(cx, 2),
            "dist_from_centroid_px": round(dist_px, 2),
            "dist_from_centroid_um": round(dist_px * um_per_px, 2),
            "normalized_radius": round(dist_px / max_radius, 4) if max_radius > 0 else 0,
            "edt_distance_px": round(edt_median, 2),
            "edt_normalized": edt_normalized,
            "area_px": int(area),
            "area_um2": round(area * um_per_px ** 2, 2),
            "perimeter_px": round(perim, 2),
            "perimeter_um": round(perim * um_per_px, 2),
            "eccentricity": round(p.eccentricity, 4),
            "solidity": round(p.solidity, 4),
            "aspect_ratio": round(major / minor, 4),
            "compactness": round(compactness, 4),
            "major_axis_px": round(major, 2),
            "minor_axis_px": round(minor, 2),
            "orientation_rad": round(p.orientation, 4),
            "extent": round(p.extent, 4),
            "mean_intensity": round(float(gray_vals.mean()), 2),
            "std_intensity": round(float(gray_vals.std()), 2),
            "min_intensity": round(float(gray_vals.min()), 2),
            "max_intensity": round(float(gray_vals.max()), 2),
            "median_intensity": round(float(np.median(gray_vals)), 2),
            "intensity_range": round(float(gray_vals.max() - gray_vals.min()), 2),
            "neighbors_count": n_neighbors,
            # --- New features ---
            "layer_fraction": layer_fraction,
            "is_boundary_cell": is_boundary_cell,
            "mean_neighbor_layer": round(mean_neighbor_layer, 2),
            "std_neighbor_layer": round(std_neighbor_layer, 2),
            "mean_neighbor_area": round(mean_neighbor_area, 2),
            "layer_diff_from_neighbors": round(layer_diff_from_neighbors, 4),
            "area_ratio_to_neighbors": area_ratio_to_neighbors,
            "intensity_cv": intensity_cv,
            "intensity_skewness": intensity_skewness,
            "intensity_kurtosis": intensity_kurtosis,
            "intensity_p10": intensity_p10,
            "intensity_p90": intensity_p90,
            "area_perimeter_ratio": area_perimeter_ratio,
            "equivalent_diameter": equivalent_diameter,
            "angular_position": angular_position,
            # Enhanced polar features
            "sin_angular_position": sin_angular,
            "cos_angular_position": cos_angular,
            "radial_x_sin": radial_x_sin,
            "radial_x_cos": radial_x_cos,
            # Directional neighbor features (polar-based)
            "radial_inward_neighbor_area": radial_inward_nbr_area,
            "radial_outward_neighbor_area": radial_outward_nbr_area,
            "cw_neighbor_area": cw_nbr_area,
            "ccw_neighbor_area": ccw_nbr_area,
            "radial_inward_neighbor_intensity": radial_inward_nbr_intensity,
            "radial_outward_neighbor_intensity": radial_outward_nbr_intensity,
            "cw_neighbor_intensity": cw_nbr_intensity,
            "ccw_neighbor_intensity": ccw_nbr_intensity,
            # --- Ring topology features (epi/exo/endo) ---
            "touches_background": cell_touches_background,
            "cells_in_same_layer": cells_in_same_layer,
            "frac_neighbors_same_layer": frac_neighbors_same_layer,
            "frac_neighbors_inner": frac_neighbors_inner,
            "frac_neighbors_outer": frac_neighbors_outer,
            "inner_neighbor_count": inner_neighbor_count,
            "outer_neighbor_count": outer_neighbor_count,
            "layer_from_inside": layer_from_inside,
            "radial_intensity_gradient": round(radial_intensity_gradient, 4),
            "neighbor_layer_range": neighbor_layer_range,
            "neighbors_boundary_cell": cell_neighbors_boundary,
            "min_neighbor_layer": min_neighbor_layer,
            "max_neighbor_layer": max_neighbor_layer,
            "n_neighbors_touching_bg": n_neighbors_touching_bg,
            # Exodermis-specific: large, hexagonal, single-layer
            "area_zscore": area_zscore,
            "area_ratio_to_global_median": area_ratio_to_global_median,
            "area_percentile_in_layer": area_percentile_in_layer,
            "area_ratio_to_inner": area_ratio_to_inner,
            "area_ratio_to_outer": area_ratio_to_outer,
            "hexagonality": hexagonality,
            "n_vertices": n_vertices,
            # Thin-ring / boundary features (endodermis & exodermis)
            "layer_cell_count_ratio": layer_cell_count_ratio,
            "inner_layer_cell_count": inner_layer_cell_count,
            "outer_layer_cell_count": outer_layer_cell_count,
            "layer_count_gradient": layer_count_gradient,
            "layer_count_asymmetry": layer_count_asymmetry,
            "adjacent_layer_area_ratio": adjacent_layer_area_ratio,
            # Vascular distinction features
            "wall_thickness_proxy": wall_thickness_proxy,
            "wall_to_lumen_ratio": wall_to_lumen_ratio,
            "local_area_rank_in_stele": local_area_rank_in_stele,
            "neighbor_area_std": neighbor_area_std,
            "cell_wall_contrast": cell_wall_contrast,
            # Enhanced xylem features
            "lumen_darkness": lumen_darkness,
            "wall_lumen_gap": wall_lumen_gap,
            "interior_intensity_std": interior_intensity_std,
            "frac_dark_interior": frac_dark_interior,
            "ring1_intensity": ring1_intensity,
            "ring2_intensity": ring2_intensity,
            "wall_interior_gradient": wall_interior_gradient,
            "mean_neighbor_wall_thickness": mean_neighbor_wall_thickness,
            "wall_thickness_vs_neighbors": wall_thickness_vs_neighbors,
            "area_ratio_to_stele_mean": area_ratio_to_stele_mean,
            # (neighbor celltypes added separately via compute_neighbor_celltypes)
        }

        # Add vascular pole features if available
        if pole_info:
            rec.update(compute_pole_features(cid, cy, cx, pole_info))

        # Only add per-channel features if image is actually color
        if is_color:
            r_vals = r_ch[cell_px]
            g_vals = g_ch[cell_px]
            b_vals = b_ch[cell_px]
            rec["mean_intensity_r"] = round(float(r_vals.mean()), 2)
            rec["mean_intensity_g"] = round(float(g_vals.mean()), 2)
            rec["mean_intensity_b"] = round(float(b_vals.mean()), 2)

        records.append(rec)

    return pd.DataFrame(records)


def compute_neighbor_celltypes(df, adjacency, cell_type_labels=None,
                               center=None):
    """
    For each cell, find the nearest adjacent neighbor in each polar
    direction (radial inward, radial outward, tangential CW, tangential CCW)
    relative to the tissue center, and look up that neighbor's cell type.

    This is rotation-invariant and biologically meaningful for cross-sections.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: cell_id, centroid_y, centroid_x.
    adjacency : dict
        {cell_id: set of neighbor cell_ids}.
    cell_type_labels : dict or None
        {cell_id: int_label} mapping. If None, all set to -1.
    center : tuple (cy, cx) or None
        Tissue center coordinates. If None, computed as mean of all centroids.

    Returns
    -------
    pd.DataFrame
        The input df with 4 new columns added:
        radial_inward_neighbor_celltype, radial_outward_neighbor_celltype,
        tangential_cw_neighbor_celltype, tangential_ccw_neighbor_celltype.
        Values are integer-encoded cell types (-1 = no neighbor/background).
    """
    if cell_type_labels is None:
        cell_type_labels = {}

    # Build centroid lookup
    centroid_lookup = {}
    for _, row in df.iterrows():
        cid = int(row["cell_id"])
        centroid_lookup[cid] = (row["centroid_y"], row["centroid_x"])

    # Compute tissue center from centroids if not provided
    if center is None:
        all_cy = [v[0] for v in centroid_lookup.values()]
        all_cx = [v[1] for v in centroid_lookup.values()]
        center_y = float(np.mean(all_cy))
        center_x = float(np.mean(all_cx))
    else:
        center_y, center_x = center

    inward_ct = []
    outward_ct = []
    cw_ct = []
    ccw_ct = []

    for _, row in df.iterrows():
        cid = int(row["cell_id"])
        cy, cx = row["centroid_y"], row["centroid_x"]
        neighbors = adjacency.get(cid, set())

        my_dist = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
        my_angle = np.arctan2(cy - center_y, cx - center_x)

        inward_nbr_ct = -1
        outward_nbr_ct = -1
        cw_nbr_ct = -1
        ccw_nbr_ct = -1
        # Track the most extreme neighbor in each direction
        best_inward_dr = 0.0   # most negative radial delta (closest to center)
        best_outward_dr = 0.0  # most positive radial delta (farthest from center)
        best_cw_angle = 0.0    # most negative angle delta (clockwise)
        best_ccw_angle = 0.0   # most positive angle delta (counter-clockwise)

        for nbr in neighbors:
            if nbr not in centroid_lookup:
                continue
            ny, nx = centroid_lookup[nbr]
            nbr_dist = np.sqrt((ny - center_y)**2 + (nx - center_x)**2)
            nbr_angle = np.arctan2(ny - center_y, nx - center_x)
            nbr_ct = cell_type_labels.get(nbr, -1)

            # Radial delta: positive = outward, negative = inward
            dr = nbr_dist - my_dist
            # Angular delta, normalized to [-pi, pi]
            dangle = nbr_angle - my_angle
            if dangle > np.pi:
                dangle -= 2 * np.pi
            elif dangle < -np.pi:
                dangle += 2 * np.pi

            # Classify: radial if |dr| dominates, tangential if |dangle * dist| dominates
            # Use angular arc length vs radial displacement for fair comparison
            arc_length = abs(dangle) * my_dist if my_dist > 1.0 else abs(dangle)
            is_radial = abs(dr) >= arc_length

            if is_radial:
                if dr < best_inward_dr:
                    best_inward_dr = dr
                    inward_nbr_ct = nbr_ct
                if dr > best_outward_dr:
                    best_outward_dr = dr
                    outward_nbr_ct = nbr_ct
            else:
                if dangle < best_cw_angle:
                    best_cw_angle = dangle
                    cw_nbr_ct = nbr_ct
                if dangle > best_ccw_angle:
                    best_ccw_angle = dangle
                    ccw_nbr_ct = nbr_ct

        inward_ct.append(inward_nbr_ct)
        outward_ct.append(outward_nbr_ct)
        cw_ct.append(cw_nbr_ct)
        ccw_ct.append(ccw_nbr_ct)

    df = df.copy()
    df["radial_inward_neighbor_celltype"] = inward_ct
    df["radial_outward_neighbor_celltype"] = outward_ct
    df["tangential_cw_neighbor_celltype"] = cw_ct
    df["tangential_ccw_neighbor_celltype"] = ccw_ct
    return df


def extract_cnn_embedding_features(masks, img_rgb, use_gpu=True, weights_path=None,
                                    model=None):
    """
    Extract DINOv2 CNN embeddings for all cells. Returns a DataFrame
    with cell_id and cnn_emb_* columns, or None if not available.

    If `weights_path` is given, loads a fine-tuned backbone state_dict
    produced by finetune_dinov2.py (expects a sibling meta.json that
    records the architecture).
    """
    try:
        from .cnn_embeddings import extract_cnn_embeddings
        return extract_cnn_embeddings(masks, img_rgb, use_gpu=use_gpu,
                                      weights_path=weights_path, model=model)
    except ImportError:
        import warnings
        warnings.warn("cnn_embeddings module not found — skipping CNN features.")
        return None
    except Exception as e:
        import warnings
        warnings.warn(f"CNN embedding extraction failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def _load_font(font_size):
    for path in [
        "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, font_size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def make_layer_colormap(n_layers):
    base_colors = [
        (255,   0,   0),  # red
        (  0,   0, 255),  # blue
        (255, 255,   0),  # yellow
        (128,   0, 128),  # purple
        (  0, 180,   0),  # green
        (255, 165,   0),  # orange
        (  0, 255, 255),  # cyan
        (255, 105, 180),  # pink
        (  0, 255,   0),  # lime
        (180,   0,   0),  # dark red
        (  0, 100, 255),  # light blue
        (200, 200,   0),  # olive
    ]
    if n_layers <= 0:
        return [base_colors[0]]
    if n_layers <= len(base_colors):
        return base_colors[:n_layers]
    # Tile if more layers than base colors
    repeats = (n_layers // len(base_colors)) + 1
    tiled = (base_colors * repeats)[:n_layers]
    return tiled


def save_layer_overlay_png(img_rgb, masks, layer_lookup, n_layers,
                            out_path, alpha=0.45):
    """
    Overlay layer colors on grayscale image with layer number on every cell.
    """
    if n_layers == 0:
        Image.fromarray(img_rgb).save(str(out_path))
        return

    H, W = masks.shape
    colors = make_layer_colormap(n_layers)

    # Build per-pixel layer image
    layer_img = np.zeros((H, W), dtype=np.int32)
    for cid, lv in layer_lookup.items():
        if lv >= 0:
            layer_img[masks == cid] = lv + 1  # 0=background

    # Colorize layers
    color_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for lv in range(1, n_layers + 1):
        m = layer_img == lv
        if m.any():
            color_rgb[m] = colors[lv - 1]

    # Blend onto grayscale base
    gray = to_grayscale_float(img_rgb)
    base = np.stack([gray, gray, gray], axis=-1)
    mask_f = (layer_img > 0)[..., None].astype(np.float32)
    blended = base * (1.0 - alpha * mask_f) + color_rgb.astype(np.float32) * (alpha * mask_f)

    # Draw cell boundaries
    boundaries = find_boundaries(masks, mode='outer')
    blended[boundaries] = [255, 255, 255]

    im = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    # Adaptive font size based on median cell size
    props = regionprops(masks)
    med_area = float(np.median([p.area for p in props])) if props else 100.0
    base_font_size = max(8, min(18, int(np.sqrt(med_area) * 0.35)))
    font_cache = {}

    for p in props:
        cid = p.label
        lv = layer_lookup.get(cid, -1)
        if lv < 0:
            continue

        cy, cx = p.centroid
        ratio = float(p.area) / max(med_area, 1.0)
        fs = max(8, min(base_font_size, int(round(base_font_size * np.sqrt(ratio)))))
        if fs not in font_cache:
            font_cache[fs] = _load_font(fs)
        font = font_cache[fs]

        x, y = int(round(cx)), int(round(cy))
        text = str(lv)

        # Black outline then white text
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            draw.text((x+dx, y+dy), text, fill=(0,0,0), font=font, anchor="mm")
        draw.text((x, y), text, fill=(255,255,255), font=font, anchor="mm")

    # Legend
    legend_font = _load_font(12)
    legend_x, legend_y = 10, max(10, H - (n_layers + 1) * 16 - 10)
    draw.rectangle(
        [legend_x - 2, legend_y - 2, legend_x + 110, legend_y + n_layers * 16 + 2],
        fill=(0, 0, 0)
    )
    for i in range(n_layers):
        color = colors[i]
        ly = legend_y + i * 16
        draw.rectangle([legend_x, ly, legend_x + 12, ly + 12], fill=color)
        draw.text((legend_x + 16, ly), f"Layer {i}", fill=(255,255,255), font=legend_font)

    # Root center marker
    all_cells = masks > 0
    if all_cells.any():
        ctr_y, ctr_x = ndimage.center_of_mass(all_cells)
        r = 5
        draw.ellipse([ctr_x-r, ctr_y-r, ctr_x+r, ctr_y+r],
                      fill=(255, 0, 0), outline=(255, 255, 255))

    im.save(str(out_path))


def save_debug_overlay(class_mask, out_path):
    palette = {
        0: (255, 105, 180),  # root_cap - pink
        1: (0, 0, 255),      # epidermis - blue
        2: (255, 255, 0),    # exodermis - yellow
        3: (0, 200, 0),      # cortex - green
        4: (255, 165, 0),    # endodermis - orange
        5: (128, 0, 128),    # pericycle - purple
        6: (255, 0, 0),      # xylem - red
        7: (255, 255, 255),  # phloem - white
        8: (0, 255, 255),    # stele - cyan
    }
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in palette.items():
        rgb[class_mask == label] = color
    Image.fromarray(rgb).save(str(out_path))


# ═══════════════════════════════════════════════════════════════
# SINGLE-TIF MODE (quick layer-index visualization)
# ═══════════════════════════════════════════════════════════════

def process_single_tif(tif_path, gpu=True, out_dir="output_layers",
                        diameter=None):
    """Segment one TIF and visualize BFS layer indices."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(tif_path).stem

    print(f"Processing: {tif_path}")

    # Load
    img_raw = imread(str(tif_path))
    img_rgb = ensure_rgb_uint8(img_raw, stack_mode="max")
    print(f"  Image shape: {img_rgb.shape}")

    # Segment
    print("  Running Cellpose-SAM...")
    masks = segment_cellpose_sam(img_rgb, use_gpu=gpu, diameter=diameter)
    n_cells = int(masks.max())
    print(f"  {n_cells} cells found")

    if n_cells == 0:
        print("  No cells found, skipping.")
        return

    # BFS layer index
    print("  Computing EDT layer index...")
    tissue = build_tissue_mask(masks)
    layer_lookup, n_layers, adjacency = compute_layer_index_edt(masks, tissue)
    print(f"  {n_layers} layers")

    # Layer overlay
    overlay_path = out_dir / f"{stem}_layers.png"
    save_layer_overlay_png(img_rgb, masks, layer_lookup, n_layers, overlay_path)
    print(f"  Saved: {overlay_path}")

    # Save masks
    masks_path = out_dir / f"{stem}_masks.npy"
    np.save(str(masks_path), masks)

    # Save CSV
    csv_path = out_dir / f"{stem}_layers.csv"
    props = {p.label: p for p in regionprops(masks)}
    all_cells = masks > 0
    ctr_y, ctr_x = ndimage.center_of_mass(all_cells) if all_cells.any() else (0, 0)

    with open(csv_path, 'w') as f:
        f.write("cell_id,layer_index,centroid_y,centroid_x,distance_from_center,area,neighbors\n")
        for cid in sorted(layer_lookup.keys()):
            if cid not in props:
                continue
            p = props[cid]
            cy, cx = p.centroid
            dist = np.sqrt((cy - ctr_y)**2 + (cx - ctr_x)**2)
            nbrs = len(adjacency.get(cid, set()))
            f.write(f"{cid},{layer_lookup[cid]},{cy:.1f},{cx:.1f},{dist:.1f},{p.area},{nbrs}\n")
    print(f"  Saved: {csv_path}")
    print("Done!")


# ═══════════════════════════════════════════════════════════════
# FULL PIPELINE (TIF+BMP pairs)
# ═══════════════════════════════════════════════════════════════

def process_pair(tif_path, bmp_path, tif_h, tif_w, um_per_px,
                 species, stage, source_file, gpu=True, out_dir=None,
                 cnn_weights=None):
    """
    Process a single TIF/BMP pair:
      1. Segment cells with Cellpose-SAM
      2. Compute BFS layer index
      3. Extract all features from TIF
      4. Assign cell type from BMP color
    """
    img_raw = imread(str(tif_path))
    img_rgb = ensure_rgb_uint8(img_raw, stack_mode="max")

    # Segment
    print(f"    Segmenting (Cellpose-SAM)...")
    masks = segment_cellpose_sam(img_rgb, use_gpu=gpu)
    n_cells = int(masks.max())
    print(f"    {n_cells} cells found")

    if n_cells == 0:
        return None

    # BFS layer index
    print(f"    Computing EDT layer index...")
    tissue = build_tissue_mask(masks)
    layer_lookup, n_layers, adjacency = compute_layer_index_edt(masks, tissue)
    print(f"    {n_layers} layers")

    # BMP class labels (extract BEFORE features so we can compute pole info)
    print(f"    Extracting cell types from BMP...")
    class_mask_bmp = create_class_mask_from_bmp(bmp_path)

    for cls_name, lbl in CELL_CLASSES.items():
        count = int((class_mask_bmp == lbl).sum())
        if count > 0:
            print(f"      BMP '{cls_name}': {count} px")

    # Resize BMP class mask to match actual mask dimensions
    class_mask_resized = resize_class_mask_to_tif(
        class_mask_bmp, masks.shape[0], masks.shape[1]
    )

    # Assign cell type by majority vote
    cell_labels, cell_conf = assign_cell_type(masks, class_mask_resized)

    # Count vascular poles (phloem and xylem clusters)
    print(f"    Counting vascular poles...")
    pole_info = count_vascular_poles(cell_labels, adjacency, masks)
    n_ph = pole_info["phloem"]["n_poles"]
    n_xy = pole_info["xylem"]["n_poles"]
    print(f"      Phloem poles: {n_ph}, Xylem poles: {n_xy}")

    # Extract features (with pole info)
    print(f"    Extracting features...")
    df = extract_all_features(masks, img_rgb, um_per_px, layer_lookup, adjacency,
                              tissue_mask=tissue, pole_info=pole_info)

    df["cell_type_label"] = df["cell_id"].map(cell_labels)
    df["cell_type_confidence"] = df["cell_id"].map(cell_conf).round(4)
    df["cell_type"] = df["cell_type_label"].map(LABEL_TO_NAME)

    # Add neighbor cell type features (using BMP ground-truth labels)
    print(f"    Computing neighbor cell type features...")
    df = compute_neighbor_celltypes(df, adjacency, cell_type_labels=cell_labels)

    # Add CNN embedding features
    print(f"    Extracting CNN embeddings...")
    cnn_df = extract_cnn_embedding_features(masks, img_rgb, use_gpu=gpu,
                                             weights_path=cnn_weights)
    if cnn_df is not None:
        df = df.merge(cnn_df, on="cell_id", how="left")
        # Fill NaN embeddings with 0 for cells that might have been skipped
        emb_cols = [c for c in df.columns if c.startswith("cnn_emb_")]
        df[emb_cols] = df[emb_cols].fillna(0.0)

    # Metadata columns
    df["species"] = species
    df["stage"] = stage
    df["source_file"] = source_file
    df["n_layers_total"] = n_layers
    df["um_per_px"] = um_per_px

    # Report
    typed = df[df["cell_type_label"] >= 0]
    untyped = df[df["cell_type_label"] < 0]
    print(f"    Classified: {len(typed)}/{n_cells}, Unclassified: {len(untyped)}")
    if len(typed) > 0:
        for ct, cnt in typed["cell_type"].value_counts().items():
            print(f"      {ct}: {cnt}")

    # Save debug overlays
    if out_dir:
        stem = Path(tif_path).stem
        debug_path = Path(out_dir) / f"debug_classmask_{stem}.png"
        save_debug_overlay(class_mask_resized, debug_path)

        layer_path = Path(out_dir) / f"layers_{stem}.png"
        save_layer_overlay_png(img_rgb, masks, layer_lookup, n_layers, layer_path)
        print(f"    Saved layer overlay: {layer_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Cell segmentation, BFS layer index, and feature extraction"
    )
    # Single-TIF mode
    parser.add_argument("--single", default=None,
                        help="Single TIF path (quick layer-index check, no BMP needed)")
    # Full pipeline mode
    parser.add_argument("--metadata", default=None, help="Path to metadata CSV")
    parser.add_argument("--tif-dir", default=None, help="Directory with TIF files")
    parser.add_argument("--bmp-dir", default=None, help="Directory with BMP files")
    parser.add_argument("--species", default=None, help="Filter by species")
    parser.add_argument("--stage", default=None, help="Filter by stage")
    # Common options
    parser.add_argument("--gpu", action="store_true", help="Use GPU for Cellpose")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Cell diameter in pixels (auto if not set)")
    parser.add_argument("--out-dir", default="feature_outputs", help="Output directory")
    parser.add_argument("--cnn-weights", default=None,
                        help="Path to fine-tuned DINOv2 backbone .pt "
                             "(produced by finetune_dinov2.py). If omitted, "
                             "uses pretrained DINOv2 ViT-S/14.")
    args = parser.parse_args()

    # ── Single-TIF mode ──
    if args.single:
        process_single_tif(args.single, gpu=args.gpu, out_dir=args.out_dir,
                           diameter=args.diameter)
        return

    # ── Full pipeline mode ──
    if not args.metadata:
        parser.error("Provide --single for quick check, or --metadata for full pipeline")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("CELL FEATURE EXTRACTION PIPELINE")
    print("=" * 65)

    print("\n[1] Loading metadata...")
    df_meta = parse_metadata(args.metadata, args.tif_dir, args.bmp_dir)
    pairs = get_pairs(df_meta, species=args.species, stage=args.stage)
    print(f"  {len(pairs)} pairs to process")

    groups = {}
    for p in pairs:
        key = (p["species"], p["stage"])
        groups.setdefault(key, []).append(p)

    all_tables = []

    for (species, stage), group_pairs in groups.items():
        print(f"\n{'='*65}")
        print(f"  Species: {species}, Stage: {stage} ({len(group_pairs)} pairs)")
        print(f"{'='*65}")

        group_tables = []

        for idx, pair in enumerate(tqdm(group_pairs, desc=f"{species}/{stage}")):
            tif_path = pair["tif_path"]
            bmp_path = pair["bmp_path"]
            um_per_px = pair["um_per_px"] or 1.0
            tif_w = pair["tif_w"]
            tif_h = pair["tif_h"]

            if not Path(tif_path).exists():
                print(f"  SKIP TIF not found: {tif_path}")
                continue
            if not Path(bmp_path).exists():
                print(f"  SKIP BMP not found: {bmp_path}")
                continue

            print(f"\n  [{idx+1}/{len(group_pairs)}] {Path(tif_path).name}")
            print(f"    TIF size: {tif_w}x{tif_h}, "
                  f"BMP size: {pair['bmp_w']}x{pair['bmp_h']}, "
                  f"scale: {um_per_px} um/px")

            try:
                table = process_pair(
                    tif_path=tif_path, bmp_path=bmp_path,
                    tif_h=tif_h, tif_w=tif_w, um_per_px=um_per_px,
                    species=species, stage=stage,
                    source_file=Path(tif_path).name,
                    gpu=args.gpu, out_dir=out_dir,
                    cnn_weights=args.cnn_weights,
                )
                if table is not None and len(table) > 0:
                    group_tables.append(table)
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

        if group_tables:
            group_df = pd.concat(group_tables, ignore_index=True)
            group_path = out_dir / f"features_{species}_{stage}.csv"
            cols = [c for c in group_df.columns if c != "cell_type"] + ["cell_type"]
            group_df = group_df[cols]
            group_df.to_csv(group_path, index=False)
            print(f"\n  Saved {len(group_df)} cells to {group_path}")
            all_tables.append(group_df)

    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        combined_path = out_dir / "all_cell_features.csv"
        cols = [c for c in combined.columns if c != "cell_type"] + ["cell_type"]
        combined = combined[cols]
        combined.to_csv(combined_path, index=False)

        print(f"\n{'='*65}")
        print(f"DONE. Total: {len(combined)} cells from {len(all_tables)} groups")
        print(f"Saved to: {combined_path}")
        print(f"\nClass distribution:")
        for ct, cnt in combined["cell_type"].value_counts().items():
            print(f"  {ct}: {cnt}")
        untyped = combined["cell_type"].isna().sum()
        if untyped > 0:
            print(f"  unclassified: {untyped}")
        print(f"{'='*65}")
    else:
        print("\nNo features extracted. Check file paths and metadata.")


if __name__ == "__main__":
    main()

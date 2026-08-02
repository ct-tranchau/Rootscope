"""
Overlay BMP cell-type color labels onto TIF images for visual verification.

For each TIF/BMP pair in the metadata, produces a PNG showing:
  - Top: TIF image with cell-type color overlay + legend
  - Bottom (or side): original BMP for comparison

Usage:
  python overlay_labels_on_tif.py \
    --metadata metadata_with_tif_sizes3.csv \
    --tif-dir ./tif --bmp-dir ./bmp \
    --out-dir label_overlays
"""

import argparse
import re
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from skimage.io import imread


# ── Cell type config (must match extract_features.py) ──
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

# HSV ranges (OpenCV scale: H 0-180, S 0-255, V 0-255)
COLOR_RANGES = {
    "phloem":     [(0, 180, 0, 30, 210, 255)],
    "cortex":     [(35, 85, 50, 255, 40, 255)],
    "epidermis":  [(100, 130, 50, 255, 80, 255)],
    "stele":      [(80, 100, 50, 255, 100, 255)],
    "exodermis":  [(22, 38, 60, 255, 100, 255)],
    "endodermis": [(10, 22, 100, 255, 100, 255)],
    "pericycle":  [(125, 150, 30, 255, 40, 200)],
    "root_cap":   [(150, 175, 40, 255, 80, 255)],
    "xylem":      [(0, 8, 150, 255, 120, 255),
                   (175, 180, 150, 255, 120, 255)],
}
COLOR_PROCESS_ORDER = [
    "phloem", "cortex", "epidermis", "stele", "exodermis",
    "endodermis", "pericycle", "root_cap", "xylem",
]

# Display palette (RGB) for overlay
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


def _normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    lo, hi = float(np.percentile(img_f, 1)), float(np.percentile(img_f, 99))
    img_f = (img_f - lo) / (hi - lo + 1e-8)
    return (np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)


def to_2d(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            return img
        return img.max(axis=0)
    if img.ndim == 4:
        return img.max(axis=0)
    raise ValueError(f"Unsupported ndim={img.ndim}")


def ensure_rgb_uint8(img):
    img2 = to_2d(img)
    if img2.ndim == 2:
        g = _normalize_to_uint8(img2)
        return np.stack([g, g, g], axis=-1)
    if img2.ndim == 3:
        if img2.shape[-1] == 1:
            g = _normalize_to_uint8(img2[..., 0])
            return np.stack([g, g, g], axis=-1)
        return _normalize_to_uint8(img2[..., :3])
    raise ValueError(f"Unsupported shape: {img2.shape}")


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


def colorize_class_mask(class_mask):
    """Convert class mask to RGB image using display palette."""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in DISPLAY_PALETTE.items():
        rgb[class_mask == label] = color
    return rgb


def overlay_on_tif(tif_rgb, class_mask_colored, alpha=0.45):
    """Blend class mask colors onto TIF image where mask is not background."""
    has_label = class_mask_colored.sum(axis=-1) > 0
    out = tif_rgb.astype(np.float32).copy()
    mask_f = has_label[..., None].astype(np.float32)
    out = out * (1.0 - alpha * mask_f) + class_mask_colored.astype(np.float32) * (alpha * mask_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_legend(draw, x, y, font):
    """Draw cell-type color legend."""
    for label in sorted(DISPLAY_PALETTE.keys()):
        name = LABEL_TO_NAME[label]
        color = DISPLAY_PALETTE[label]
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(0, 0, 0))
        draw.text((x + 20, y), name, fill=(255, 255, 255), font=font)
        draw.text((x + 19, y - 1), name, fill=(0, 0, 0), font=font)
        draw.text((x + 20, y), name, fill=(255, 255, 255), font=font)
        y += 18
    return y


def make_overlay_png(tif_path, bmp_path, out_path, tif_h=None, tif_w=None):
    """
    Create a verification PNG:
      Top half: TIF with BMP labels overlaid (semi-transparent colors + legend)
      Bottom half: original BMP image for reference
    """
    # Load TIF
    img_raw = imread(str(tif_path))
    tif_rgb = ensure_rgb_uint8(img_raw)
    th, tw = tif_rgb.shape[:2]

    # Load BMP and create class mask
    class_mask = create_class_mask_from_bmp(bmp_path)
    bmp_rgb = np.array(Image.open(str(bmp_path)).convert("RGB"))
    bh, bw = bmp_rgb.shape[:2]

    # Resize class mask to TIF dimensions
    class_mask_resized = cv2.resize(
        class_mask.astype(np.float32), (tw, th),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int8)

    # Colorize and overlay on TIF
    class_colored = colorize_class_mask(class_mask_resized)
    tif_overlaid = overlay_on_tif(tif_rgb, class_colored, alpha=0.5)

    # Resize BMP to same width as TIF for stacking
    scale = tw / bw
    new_bh = int(bh * scale)
    bmp_resized = cv2.resize(bmp_rgb, (tw, new_bh), interpolation=cv2.INTER_LINEAR)

    # Add title bars
    title_h = 30
    font = _load_font(16)
    small_font = _load_font(12)

    # Total canvas: title + TIF overlay + gap + title + BMP
    gap = 4
    total_h = title_h + th + gap + title_h + new_bh
    canvas = np.zeros((total_h, tw, 3), dtype=np.uint8)

    # Place TIF overlay
    y_offset = title_h
    canvas[y_offset:y_offset + th, :, :] = tif_overlaid

    # Place BMP
    y_bmp = title_h + th + gap + title_h
    canvas[y_bmp:y_bmp + new_bh, :, :] = bmp_resized

    # Draw on PIL
    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)

    # Title: TIF + label overlay
    tif_name = Path(tif_path).name
    bmp_name = Path(bmp_path).name
    draw.rectangle([0, 0, tw, title_h], fill=(30, 30, 30))
    draw.text((10, 6), f"TIF + BMP labels: {tif_name}", fill=(255, 255, 255), font=font)

    # Title: BMP reference
    y_title2 = title_h + th + gap
    draw.rectangle([0, y_title2, tw, y_title2 + title_h], fill=(30, 30, 30))
    draw.text((10, y_title2 + 6), f"BMP reference: {bmp_name}", fill=(255, 255, 255), font=font)

    # Legend on TIF overlay (top-right)
    legend_x = tw - 130
    legend_y = title_h + 10
    # Background box for legend
    n_classes = len(DISPLAY_PALETTE)
    legend_h = n_classes * 18 + 6
    draw.rectangle([legend_x - 4, legend_y - 4, tw - 4, legend_y + legend_h],
                   fill=(0, 0, 0, 180))
    draw_legend(draw, legend_x, legend_y, small_font)

    im.save(str(out_path))
    return out_path


def _parse_size(s):
    m = re.match(r"(\d+)\s*x\s*(\d+)", str(s).strip())
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay BMP cell-type labels on TIF images for verification"
    )
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--tif-dir", required=True, help="Directory with TIF files")
    parser.add_argument("--bmp-dir", required=True, help="Directory with BMP files")
    parser.add_argument("--out-dir", default="label_overlays", help="Output directory")
    parser.add_argument("--species", default=None, help="Filter by species")
    parser.add_argument("--stage", default=None, help="Filter by stage")
    parser.add_argument("--gpu", action="store_true", help="Accepted for consistency (not used)")
    args = parser.parse_args()

    import pandas as pd

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_dir = Path(args.tif_dir)
    bmp_dir = Path(args.bmp_dir)

    df = pd.read_csv(args.metadata)
    df.columns = df.columns.str.strip().str.lower()

    if args.species:
        df = df[df["species"].str.lower() == args.species.lower()]
    if args.stage:
        df = df[df["stage"].str.lower() == args.stage.lower()]

    # Only keep rows with "outlined" in BMP filename (the labeled ones)
    df_outlined = df[df["bmp_filename"].str.lower().str.contains("outlined")]
    if len(df_outlined) == 0:
        print("No 'Outlined' BMPs found, using all rows.")
        df_outlined = df

    print(f"Processing {len(df_outlined)} TIF/BMP pairs...")

    for idx, row in df_outlined.iterrows():
        tif_path = tif_dir / row["tif_matched"]
        bmp_path = bmp_dir / row["bmp_filename"]

        if not tif_path.exists():
            print(f"  SKIP TIF not found: {tif_path}")
            continue
        if not bmp_path.exists():
            print(f"  SKIP BMP not found: {bmp_path}")
            continue

        species = row["species"]
        stage = row["stage"]
        stem = tif_path.stem.replace(".aivia", "")
        out_name = f"verify_{species}_{stage}_{stem}.png"
        out_path = out_dir / out_name

        print(f"  [{idx+1}] {tif_path.name} + {bmp_path.name}")
        try:
            make_overlay_png(tif_path, bmp_path, out_path)
            print(f"    -> {out_path}")
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nDone! Overlays saved to: {out_dir}/")


if __name__ == "__main__":
    main()

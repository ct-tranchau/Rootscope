"""
Extract DINOv2 ViT-S/14 embeddings for each cell in a segmentation mask.

For each cell, crops a patch around the cell's bounding box (with padding),
resizes to 224x224, and passes through DINOv2 to get a 384-dim embedding.

Returns a DataFrame with columns cnn_emb_0 ... cnn_emb_383 indexed by cell_id.
"""

import warnings
import numpy as np
import pandas as pd
from skimage.measure import regionprops


def extract_cnn_embeddings(masks, img_rgb, batch_size=64, use_gpu=True,
                           padding=16, weights_path=None):
    """
    Extract DINOv2 ViT-S/14 embeddings for each cell in the segmentation mask.

    Parameters
    ----------
    masks : np.ndarray (H, W) int32
        Cell segmentation mask (0 = background, >0 = cell IDs).
    img_rgb : np.ndarray (H, W, 3) uint8
        Original RGB image.
    batch_size : int
        Number of cells to process at once.
    use_gpu : bool
        Move model to CUDA if available.
    padding : int
        Extra pixels around each cell's bounding box for context.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, cnn_emb_0, cnn_emb_1, ..., cnn_emb_383
    """
    try:
        import torch
        import torchvision.transforms as T
    except ImportError:
        warnings.warn(
            "PyTorch not available — skipping CNN embedding extraction. "
            "Install torch + torchvision to enable DINOv2 features."
        )
        return None

    # Determine device
    device = torch.device("cpu")
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"    CNN embeddings: using GPU ({torch.cuda.get_device_name(0)})")
    else:
        if use_gpu:
            print("    CNN embeddings: GPU requested but not available, using CPU")
        else:
            print("    CNN embeddings: using CPU")

    # Determine which architecture to load. If weights_path is provided and
    # a sibling meta.json exists (written by finetune_dinov2.py), honor the
    # `arch` / `hub_name` recorded there. Otherwise default to ViT-S/14.
    hub_name = "dinov2_vits14_reg"
    if weights_path is not None:
        from pathlib import Path as _P
        import json as _json
        meta_path = _P(weights_path).parent / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as _f:
                    _meta = _json.load(_f)
                hub_name = _meta.get("hub_name", hub_name)
            except Exception:
                pass

    print(f"    CNN embeddings: loading {hub_name}...")
    try:
        model = torch.hub.load(
            'facebookresearch/dinov2', hub_name, verbose=False
        )
    except Exception as e:
        warnings.warn(
            f"Failed to load DINOv2 model: {e}. "
            "Skipping CNN embedding extraction."
        )
        return None

    if weights_path is not None:
        try:
            state = torch.load(weights_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"    CNN embeddings: loaded fine-tuned weights from "
                  f"{weights_path} (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            warnings.warn(f"Failed to load fine-tuned weights ({e}); using pretrained.")

    model = model.to(device)
    model.eval()

    # DINOv2 preprocessing: resize to 224x224, normalize with ImageNet stats
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    H, W = masks.shape
    props = regionprops(masks)

    if not props:
        return None

    # Handle grayscale: if all channels are identical, it's grayscale
    # DINOv2 expects 3-channel input; grayscale is already repeated to RGB
    # by ensure_rgb_uint8, but double-check
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb, img_rgb, img_rgb], axis=-1)
    elif img_rgb.shape[-1] == 1:
        img_rgb = np.concatenate([img_rgb, img_rgb, img_rgb], axis=-1)

    # Collect cell patches
    cell_ids = []
    patches = []

    for p in props:
        cid = p.label
        # Bounding box: (min_row, min_col, max_row, max_col)
        r0, c0, r1, c1 = p.bbox

        # Add padding for context
        r0_pad = max(0, r0 - padding)
        c0_pad = max(0, c0 - padding)
        r1_pad = min(H, r1 + padding)
        c1_pad = min(W, c1 + padding)

        patch = img_rgb[r0_pad:r1_pad, c0_pad:c1_pad].copy()

        # Ensure patch is not empty
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            continue

        cell_ids.append(cid)
        patches.append(patch)

    if not patches:
        return None

    # Process in batches
    all_embeddings = []
    n_cells = len(patches)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_tensors = []

            for patch in patches[start:end]:
                tensor = transform(patch)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(device)
            embeddings = model(batch)  # (batch_size, 384)
            all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)  # (n_cells, 384)
    emb_dim = all_embeddings.shape[1]

    # Build DataFrame
    emb_cols = [f"cnn_emb_{i}" for i in range(emb_dim)]
    df = pd.DataFrame(all_embeddings, columns=emb_cols)
    df.insert(0, "cell_id", cell_ids)

    print(f"    CNN embeddings: extracted {emb_dim}-dim embeddings for "
          f"{len(cell_ids)} cells")

    return df

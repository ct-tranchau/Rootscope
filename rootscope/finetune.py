"""
Fine-tune DINOv2 on labeled cell crops so it produces embeddings tailored
to this confocal dataset (root-tip cross-sections, 9 cell types).

Approach:
  * Load DINOv2 ViT-{s,b,l}/14 pretrained backbone
  * Freeze all blocks except the last `--unfreeze-blocks`
  * Attach a Linear classifier head on the CLS token
  * Train on (cell crop -> cell_type) with class weights and image-based split
  * Save the fine-tuned backbone state_dict so cnn_embeddings.py can load it

Crops are taken from the original TIF using centroid + major_axis_px from the
features CSV, so no re-segmentation is needed.

Usage (SLURM-friendly):
  python finetune_dinov2.py \
    --features feature_outputs/all_cell_features.csv \
    --tif-dir tif \
    --out-dir dinov2_finetuned \
    --arch vits14 \
    --unfreeze-blocks 2 \
    --epochs 20 \
    --batch-size 64

Then for inference, cnn_embeddings.py can load:
  dinov2_finetuned/backbone.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader

from .extract_features import ensure_rgb_uint8 as _ensure_rgb_uint8_from_ef


CELL_CLASSES = [
    "root_cap", "epidermis", "exodermis", "cortex", "endodermis",
    "pericycle", "xylem", "phloem", "stele",
]
NAME_TO_LABEL = {n: i for i, n in enumerate(CELL_CLASSES)}

ARCH_TO_HUB = {
    "vits14": ("dinov2_vits14_reg", 384),
    "vitb14": ("dinov2_vitb14_reg", 768),
    "vitl14": ("dinov2_vitl14_reg", 1024),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _ensure_rgb_uint8(img):
    # Delegates to extract_features.ensure_rgb_uint8 which handles Z-stacks
    # (shape (Z,H,W) or (Z,H,W,C)) by max-projection and normalizes with
    # the same percentile clipping used elsewhere in the pipeline.
    return _ensure_rgb_uint8_from_ef(img, stack_mode="max")


class CellCropDataset(Dataset):
    """
    Loads crops lazily from TIFs. Caches up to `cache_size` images in RAM.
    """
    def __init__(self, df, tif_dir, transform, cache_size=8, min_crop=48):
        self.df = df.reset_index(drop=True)
        self.tif_dir = Path(tif_dir)
        self.transform = transform
        self.cache_size = cache_size
        self.min_crop = min_crop
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return len(self.df)

    def _load_tif(self, source_file):
        if source_file in self._cache:
            return self._cache[source_file]
        tif_path = self.tif_dir / source_file
        img = imread(str(tif_path))
        img = _ensure_rgb_uint8(img)
        if len(self._cache) >= self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        self._cache[source_file] = img
        self._cache_order.append(source_file)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_tif(row["source_file"])
        H, W = img.shape[:2]
        cx = float(row["centroid_x"])
        cy = float(row["centroid_y"])
        maj = float(row.get("major_axis_px", 0) or 0)
        side = max(self.min_crop, int(maj * 2.2) + 16)
        half = side // 2
        x0 = max(0, int(cx - half))
        y0 = max(0, int(cy - half))
        x1 = min(W, int(cx + half))
        y1 = min(H, int(cy + half))
        patch = img[y0:y1, x0:x1]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            patch = np.zeros((max(patch.shape[0], 4),
                              max(patch.shape[1], 4), 3), dtype=np.uint8)
        pil = Image.fromarray(patch)
        tensor = self.transform(pil)
        label = int(row["cell_type_label"])
        return tensor, label


class DinoClassifier(nn.Module):
    def __init__(self, backbone, emb_dim, n_classes, dropout=0.0):
        super().__init__()
        self.backbone = backbone
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(emb_dim, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


def _set_trainable(backbone, unfreeze_blocks):
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_blocks <= 0:
        return
    blocks = backbone.blocks
    for blk in blocks[-unfreeze_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
    for p in backbone.norm.parameters():
        p.requires_grad = True


def _compute_class_weights(labels, n_classes):
    """
    Softened inverse-frequency weights: w_c = sqrt(N_total / (n_classes * N_c)).
    Much gentler than pure inverse-frequency, which on this dataset pushed
    xylem/phloem weights >7x and caused the model to collapse to those classes.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1
    w = np.sqrt(counts.sum() / (n_classes * counts))
    w = w / w.mean()
    return w


def _eval(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    acc = (all_pred == all_true).mean()
    per_class = {}
    for i, name in enumerate(CELL_CLASSES):
        mask = all_true == i
        if mask.sum() > 0:
            per_class[name] = (all_pred[mask] == all_true[mask]).mean()
    return acc, per_class, all_pred, all_true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--tif-dir", required=True)
    ap.add_argument("--out-dir", default="dinov2_finetuned")
    ap.add_argument("--arch", default="vits14", choices=list(ARCH_TO_HUB))
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--unfreeze-blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--warmup-epochs", type=int, default=2,
                    help="Train only the classifier head for this many "
                         "epochs before unfreezing the backbone.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load + filter features ----
    print(f"Loading {args.features}")
    df = pd.read_csv(args.features, usecols=[
        "cell_id", "centroid_x", "centroid_y", "major_axis_px",
        "cell_type_label", "cell_type_confidence", "cell_type", "source_file",
    ])
    df = df[df["cell_type_label"] >= 0].copy()
    if "cell_type_confidence" in df.columns:
        df = df[df["cell_type_confidence"] >= args.min_confidence].copy()
    df = df.dropna(subset=["centroid_x", "centroid_y", "cell_type_label", "source_file"])
    df["cell_type_label"] = df["cell_type_label"].astype(int)
    print(f"  Usable labeled cells: {len(df)}")

    # ---- Image-level split ----
    images = np.array(sorted(df["source_file"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(images)
    n_val = max(1, int(len(images) * args.val_fraction))
    val_imgs = set(images[:n_val])
    train_imgs = set(images[n_val:])
    df_train = df[df["source_file"].isin(train_imgs)].reset_index(drop=True)
    df_val = df[df["source_file"].isin(val_imgs)].reset_index(drop=True)
    print(f"  Train: {len(df_train)} cells ({len(train_imgs)} images)")
    print(f"  Val:   {len(df_val)} cells ({len(val_imgs)} images)")

    # ---- Transforms ----
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds_train = CellCropDataset(df_train, args.tif_dir, train_tf)
    ds_val = CellCropDataset(df_val, args.tif_dir, eval_tf)

    ld_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    ld_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- Model ----
    hub_name, emb_dim = ARCH_TO_HUB[args.arch]
    print(f"Loading DINOv2 backbone: {hub_name} (emb_dim={emb_dim})")
    backbone = torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)
    _set_trainable(backbone, args.unfreeze_blocks)

    model = DinoClassifier(backbone, emb_dim, len(CELL_CLASSES)).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,}")

    class_weights = _compute_class_weights(
        df_train["cell_type_label"].values, n_classes=len(CELL_CLASSES)
    )
    print(f"  Class weights: {dict(zip(CELL_CLASSES, np.round(class_weights, 3)))}")
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]

    def _make_optimizer(head_lr, bb_lr):
        return torch.optim.AdamW(
            [
                {"params": head_params, "lr": head_lr},
                {"params": bb_params, "lr": bb_lr},
            ],
            weight_decay=args.weight_decay,
        )

    # Linear-probe warmup: for the first `warmup_epochs` epochs train only the
    # head (bb_lr=0) so the classifier stops producing garbage logits before we
    # let gradients flow into the backbone. Without this the backbone gets
    # swamped by huge early head gradients and the whole thing collapses to
    # the majority class.
    warmup_epochs = args.warmup_epochs

    # ---- Training loop ----
    history = []
    best_val = 0.0
    best_epoch = -1
    n_stalled = 0

    optimizer = _make_optimizer(args.lr_head, 0.0)
    scheduler = None

    for epoch in range(1, args.epochs + 1):
        if epoch == warmup_epochs + 1:
            # Warmup finished: unlock backbone and rebuild optimizer + scheduler
            print(f"  Warmup done, enabling backbone LR={args.lr_backbone}")
            optimizer = _make_optimizer(args.lr_head, args.lr_backbone)
            remaining = max(1, args.epochs - warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining,
            )

        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_seen = 0
        n_correct = 0
        for x, y in ld_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
            n_correct += (logits.argmax(dim=1) == y).sum().item()
        if scheduler is not None:
            scheduler.step()
        train_loss = loss_sum / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)

        val_acc, val_per_class, _, _ = _eval(model, ld_val, device)
        dt = time.time() - t0
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_acc": val_acc,
            "per_class": val_per_class, "time_s": dt,
        })
        per_class_str = " ".join(
            f"{n[:4]}={v:.2f}" for n, v in val_per_class.items()
        )
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f}  {per_class_str}  ({dt:.0f}s)",
              flush=True)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            n_stalled = 0
            torch.save(model.backbone.state_dict(), out_dir / "backbone.pt")
            torch.save(model.head.state_dict(), out_dir / "head.pt")
            with open(out_dir / "meta.json", "w") as f:
                json.dump({
                    "arch": args.arch,
                    "hub_name": hub_name,
                    "emb_dim": emb_dim,
                    "classes": CELL_CLASSES,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val,
                    "unfreeze_blocks": args.unfreeze_blocks,
                    "min_confidence": args.min_confidence,
                    "val_images": sorted(val_imgs),
                }, f, indent=2)
        else:
            n_stalled += 1
            if n_stalled >= args.patience:
                print(f"Early stop: no val improvement for {args.patience} epochs.")
                break

    # ---- Final report ----
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_acc: {best_val:.4f} at epoch {best_epoch}")
    print(f"Saved backbone: {out_dir/'backbone.pt'}")
    print(f"Saved head:     {out_dir/'head.pt'}")
    print(f"Meta:           {out_dir/'meta.json'}")


if __name__ == "__main__":
    main()

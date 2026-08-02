#!/usr/bin/env python
"""
STEP 20 — Fine-tune DINOv2.

Fine-tune the DINOv2 ViT backbone on labeled cell crops so its embeddings are
tailored to this confocal root-tip dataset. Saves backbone.pt, which step 30
(training) and prediction both consume.

Run:
    python pipeline/20_finetune_dinov2.py \
        --features feature_outputs_finetuned/all_cell_features.csv \
        --tif-dir tif \
        --out-dir dinov2_finetuned \
        --arch vits14 --unfreeze-blocks 2 --epochs 20 --batch-size 64

(Thin wrapper around rootscope.finetune — see that module for all flags.)
"""
from rootscope.finetune import main

if __name__ == "__main__":
    main()

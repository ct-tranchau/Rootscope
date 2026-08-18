#!/usr/bin/env python
"""
STEP 10: Feature extraction.

Segment TIF/BMP pairs with Cellpose-SAM, compute the BFS tissue-layer index,
extract shape/size/intensity features, and read the ground-truth cell type from
the BMP color overlay. Produces the feature table used to train the model.

Run:
    python pipeline/10_extract_features.py \
        --metadata metadata_with_tif_sizes3.csv \
        --tif-dir ./tif --bmp-dir ./bmp \
        --out-dir feature_outputs_finetuned --gpu

(Thin wrapper around rootscope.extract_features; see that module for all flags.)
"""
from rootscope.extract_features import main

if __name__ == "__main__":
    main()

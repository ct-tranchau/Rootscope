#!/usr/bin/env python
"""
STEP 40 — Predict cell types on new images (research/reproducibility form).

Segments a new TIF, extracts features, and predicts cell types with iterative
refinement, writing per-cell CSVs + labeled overlays. This is the explicit,
all-flags form used in the paper; end users can instead just run the
`rootscope` command (see the top-level README).

Run:
    python pipeline/40_predict_cell_types.py \
        --tif path/to/image.tif \
        --model-dir trained_model_iterative_cnn_finetuned \
        --cnn-weights dinov2_finetuned/backbone.pt \
        --um-per-px 1.0 --out-dir predictions_iterative_finetuned \
        --max-rounds 10 --gpu

(Thin wrapper around rootscope.predict — see that module for all flags.)
"""
from rootscope.predict import main

if __name__ == "__main__":
    main()

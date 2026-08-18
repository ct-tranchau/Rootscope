#!/usr/bin/env python
"""
STEP 30: Train the iterative cell-type classifier (with DINOv2 embeddings).

Trains the RandomForest / XGBoost / LightGBM ensemble with iterative
self-training: each round fills in neighbor cell-types from the previous round's
cross-validated predictions and retrains until predictions converge. Uses the
cnn_emb_* columns from the feature table. Saves the model artifacts that
prediction loads.

Run:
    python pipeline/30_train_iterative_cnn.py \
        --features feature_outputs_finetuned/all_cell_features.csv \
        --out-dir trained_model_iterative_cnn_finetuned \
        --n-estimators 500 --max-rounds 10

(Thin wrapper around rootscope.train; see that module for all flags.)
"""
from rootscope.train import main

if __name__ == "__main__":
    main()

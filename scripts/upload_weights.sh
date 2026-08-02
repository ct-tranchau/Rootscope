#!/usr/bin/bash
#
# Maintainer script: publish the trained Rootscope weights to the Hugging Face
# Hub repo that rootscope/weights.py downloads from.
#
# Prerequisites
#   1. Create the model repo on huggingface.co (New -> Model), named to match
#      DEFAULT_HF_REPO in rootscope/weights.py.
#   2. Log in once:   hf auth login      (paste a token with WRITE permission)
#
# Usage
#   bash scripts/upload_weights.sh              # full bundle, all 3 models (~460 MB)
#   bash scripts/upload_weights.sh --no-rf      # skip RandomForest (~126 MB)
#
# RandomForest is uploaded by DEFAULT and should stay that way: the ensemble in
# predict.py contains an RF-trust rule (it double-weights RandomForest wherever
# RF predicts a minority class), and RF is the only bagging model of the three,
# so it decorrelates from XGBoost and LightGBM, which tend to agree with each
# other. Publishing without it changes the ensemble, not just the download size.
#
# The files are uploaded under their exact required names at the repo root,
# which is the layout rootscope/weights.py expects. Nothing is copied locally.

set -euo pipefail

REPO="ct-tranchau/Rootscope"
SRC="/projects/songli_lab/Nina/Image_analysis/Test_050326"
MODELS="$SRC/trained_model_iterative_cnn_finetuned"
DINO="$SRC/dinov2_finetuned"

WITH_RF=1
[[ "${1:-}" == "--no-rf" ]] && WITH_RF=0

command -v hf >/dev/null 2>&1 || {
    echo "ERROR: the 'hf' CLI is not on PATH. Install it with:"
    echo "    pip install -U huggingface_hub"
    exit 1
}

echo "=== Uploading Rootscope weights to https://huggingface.co/$REPO ==="

# ── required: the XGBoost + LightGBM ensemble and its metadata (~43 MB) ──
for f in feature_columns.joblib \
         label_encoder.joblib \
         model_XGBoost.joblib \
         feature_scaler_XGBoost.joblib \
         model_LightGBM.joblib \
         feature_scaler_LightGBM.joblib; do
    echo "  -> $f"
    hf upload "$REPO" "$MODELS/$f" "$f"
done

# ── required: the fine-tuned DINOv2 backbone (~85 MB) ──
echo "  -> backbone.pt"
hf upload "$REPO" "$DINO/backbone.pt" "backbone.pt"

# ── RandomForest (~350 MB) -- part of the published ensemble by default ──
if [[ "$WITH_RF" == "1" ]]; then
    for f in model_RandomForest.joblib feature_scaler_RandomForest.joblib; do
        echo "  -> $f"
        hf upload "$REPO" "$MODELS/$f" "$f"
    done
else
    echo "  (skipping RandomForest -- --no-rf was passed)"
fi

echo
echo "Done. Verify with:"
echo "    python -c \"from huggingface_hub import HfApi; print([s.rfilename for s in HfApi().model_info('$REPO').siblings])\""

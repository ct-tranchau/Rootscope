# Model weights

The trained weights are **not** stored in this Git repository; they are too
large for GitHub (the RandomForest model alone is ~350 MB). They live on the
Hugging Face Hub at
**[ct-tranchau/Rootscope](https://huggingface.co/ct-tranchau/Rootscope)**, and
Rootscope downloads them automatically on first use, caching them in
`~/.cache/huggingface`. Downloads resume if interrupted.

## Files

| File | ~Size | Required? |
|------|-------|-----------|
| `feature_columns.joblib` | tiny | yes |
| `label_encoder.joblib` | tiny | yes |
| `model_XGBoost.joblib` | ~14 MB | yes |
| `feature_scaler_XGBoost.joblib` | tiny | yes |
| `model_LightGBM.joblib` | ~29 MB | yes |
| `feature_scaler_LightGBM.joblib` | tiny | yes |
| `model_RandomForest.joblib` | ~350 MB | yes (published) |
| `feature_scaler_RandomForest.joblib` | tiny | yes (published) |
| `backbone.pt` (fine-tuned DINOv2) | ~88 MB | strongly recommended |

All three models are part of the published ensemble. RandomForest matters beyond
being a third vote: the ensemble double-weights it wherever it predicts a
minority class (the RF-trust rule in `predict.py`), and it is the only bagging
model of the three, so it decorrelates from XGBoost and LightGBM, which tend to
agree with each other.

Prediction still runs if `model_RandomForest.joblib` is missing, falling back
to the XGBoost + LightGBM ensemble, but results will differ from the published
model.

## For maintainers: publishing the weights

Run [`scripts/upload_weights.sh`](../scripts/upload_weights.sh), which uploads
every file above to the Hub repo named in `DEFAULT_HF_REPO`
([`rootscope/weights.py`](../rootscope/weights.py)) under these exact names.
Prerequisites: create the model repo on huggingface.co, then `hf auth login`
with a **Write** token.

To publish somewhere else instead, set `ROOTSCOPE_HF_REPO` (a different Hub repo)
or `ROOTSCOPE_MODELS_URL` (any base URL where `<base-url>/model_LightGBM.joblib`
downloads directly).

## For users: using your own weights

Drop the files into this `models/` folder, or point at them explicitly:

```bash
rootscope --tif image.tif \
    --model-dir /path/to/models \
    --cnn-weights /path/to/backbone.pt
```

or

```bash
export ROOTSCOPE_MODEL_DIR=/path/to/models
export ROOTSCOPE_CNN_WEIGHTS=/path/to/backbone.pt
```

# Model weights (V4)

Shipped in this folder so prediction works offline. On the Hugging Face model
repo `ct-tranchau/Rootscope` the same files live under `v4/`.

| file | size | what |
|---|---|---|
| `backbone.pt` | 346 MB | DINOv2 ViT-B/14, all 12 blocks fine-tuned on cell crops |
| `meta.json` | tiny | architecture + hub name; `cnn_embeddings.py` reads it, **keep it beside `backbone.pt`** |
| `lgbm_s42.joblib` | ~170 MB | LightGBM bundle, seed 42: `round_models` (six `(model, scaler)` pairs, one per refinement round), `fcols` (915 names, in model order), `classes` (9, in model order) |
| `lgbm_s1.joblib`, `lgbm_s7.joblib` | ~170 MB each | seeds 1 and 7. Optional: seed 42 alone predicts; all three reproduce the manuscript's 3-seed average |
| `lgbm_geom_s42.joblib`, `lgbm_geom_s1.joblib`, `lgbm_geom_s7.joblib` | ~180 MB each | geometry-only bundles (v4.1): same six-round structure, `fcols` = 68 shape / position / neighbourhood columns + the 51 context columns, no intensity and no embedding columns. `classify_v4.classify(geometry=True)` averages their probabilities with the full model's on every cell the full model places in the outer layers (cortex, epidermis, exodermis, root cap), so those layers are decided mostly by geometry and do not depend on the stain. Trained on the same 100 sections with the same script as the full bundles. Held-out: 0.880 -> 0.885, exodermis precision 0.82 -> 0.90. Without these files the package silently falls back to the v4.0 behaviour |

915 = 96 morphometric + 768 embedding + 51 context columns that `context.py`
rebuilds every refinement round from the previous round's predictions
(36 `nbp_*` neighbour probabilities, 6 `agree_*` flags, 9 layer-distance
columns). Those 51 are **never read from a feature table**; a bundle whose
`fcols` lists them is only valid together with the loop in `classify_v4.py`.

## Not committed to git

`.gitignore` excludes `models/*.joblib` and `models/*.pt`. Publish them with
`scripts/upload_weights.sh` into the `v4/` folder of the Hub repo and let
`rootscope/weights.py` fetch them.

## Why the bundles are 160 MB

Each holds all six per-round models, because inference replays the same
round-specific model the training run used at that round. A bundle saved with
only the final-round model (as the research grid's `runs10/` did) cannot be
replayed and `classify_v4.load_models` refuses it.

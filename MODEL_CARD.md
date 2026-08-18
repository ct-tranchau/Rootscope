---
license: mit
tags:
  - biology
  - plant-biology
  - microscopy
  - image-segmentation
  - cell-type-classification
  - cellpose
  - dinov2
library_name: rootscope
pipeline_tag: image-classification
---

# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

Trained model weights for **[RootScope](https://github.com/ct-tranchau/Rootscope)**.

RootScope takes a raw confocal root-tip cross-section TIFF, segments every cell
with **Cellpose-SAM**, describes each cell with hand-crafted morpho-topological
features plus fine-tuned **DINOv2** embeddings, and classifies it into one of
nine anatomical cell types using an iterative tree-based ensemble.

## Install

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
conda env create -f environment.yml
conda activate rootscope
pip install .
```

## Run

```bash
rootscope --tif my_image.tif --out results/
```

`results/` gets a per-cell CSV and a labeled overlay PNG, per model plus the
ensemble.

Or from Python:

```python
from rootscope import predict_tif
df = predict_tif("my_image.tif", out_dir="results/")
```

## Cell types

`root_cap` · `epidermis` · `exodermis` · `cortex` · `endodermis` · `pericycle` ·
`stele` · `xylem` · `phloem`

## Files

| File | Size |
|------|------|
| `model_RandomForest.joblib` | 349 MB |
| `backbone.pt` (fine-tuned DINOv2) | 88 MB |
| `model_LightGBM.joblib` | 29 MB |
| `model_XGBoost.joblib` | 14 MB |
| scalers, feature columns, label encoder | small |

## Performance

Held-out test accuracy, 480 features, 9 classes:

| Model | Test |
|-------|------|
| LightGBM | **0.965** |
| XGBoost | 0.959 |
| RandomForest | 0.937 |

Inference: ~1.5-3 min per image on a single GPU.

## Notes

- Set `--um-per-px` to your image's real scale. It is not auto-detected, and
  the default of 1.0 distorts every size feature.
- Input must be a raw image, not a segmentation mask.
- CPU works but is far slower (~15-30 min per image).
- `scikit-learn` is pinned to 1.7.2, the version these models were saved with.

## Contact

tnchau@vt.edu

# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

RootScope segments every cell in a confocal root cross-section with Cellpose-SAM and assigns each one to one of nine anatomical cell types, refining the labels with neighbor context until they converge.

No install needed: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ct-tranchau/Rootscope/blob/main/notebooks/Rootscope_Colab.ipynb) - upload a TIFF, get the overlay and per-cell CSV on a free GPU.

## Install

Needs Python 3.10+.

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
pip install .
```

For a clean conda environment instead: `conda env create -f environment.yml && conda activate rootscope && pip install .`
(use `environment-cpu.yml` on macOS or without an NVIDIA GPU). To reproduce the exact
validated versions, `pip install -r requirements.txt` first.

Works on Linux/Windows with an NVIDIA GPU (~2 min per image) and on CPU or Apple
Silicon (15-30 min). macOS Intel is unsupported. The first run downloads the
trained weights (~460 MB) from the
[Hub](https://huggingface.co/ct-tranchau/Rootscope) into `~/.cache/huggingface`.

## Predict

```bash
rootscope --tif my_image.tif --out results/     # one image
rootscope --tif-dir my_tifs/  --out results/    # a folder
```

```python
from rootscope import predict_tif
df = predict_tif("my_image.tif", out_dir="results/", gpu=True)   # one row per cell
```

| Flag | Meaning |
|------|---------|
| `--cpu` / `--gpu` | force CPU or GPU (default: auto-detect) |
| `--um-per-px 0.5` | microns per pixel of your image (default 1.0) |
| `--label-cells` | write the cell-type name inside each cell (off by default: the text hides dense sections) |
| `--model-dir DIR` | use your own trained weights instead of the download |
| `--cnn-weights backbone.pt` | use your own fine-tuned DINOv2 backbone |
| `--max-rounds N` | iterative-refinement rounds (default 10) |

Per image, `results/` gets `<image>_<Model>_predictions.csv` (one row per cell:
id, centroid, features, predicted `cell_type`) and `<image>_<Model>_overlay.png`
(cells colored by type). Batch mode also writes `all_predictions.csv`. See
[`examples/README.md`](examples/README.md).

## How it works

```
 raw TIFF
    ▼  Cellpose-SAM               segment every cell
    ▼  layer index + adjacency    position within the root-tip tissue
    ▼  shape/size/intensity       handcrafted per-cell features
    ▼  DINOv2 embeddings          fine-tuned deep features per cell crop
    ▼  iterative ensemble         RandomForest + XGBoost + LightGBM,
                                  refining neighbor-context each round
 per-cell cell-type predictions  +  labeled overlay
```

Each round fills in every cell's neighbor cell-types from the previous round and
re-predicts, until the predictions stop changing.

## More

- **Web app** - `webapp/` is a Gradio front end for Hugging Face Spaces with
  ZeroGPU. Run locally with `pip install gradio && python webapp/app.py`; see
  [`webapp/README.md`](webapp/README.md) to deploy.
- **Retraining** - `pipeline/` holds the numbered research workflow behind the
  published model (features → DINOv2 fine-tuning → training → prediction). See
  [`pipeline/README.md`](pipeline/README.md).
- **Model weights** - hosted on the
  [Hub](https://huggingface.co/ct-tranchau/Rootscope): RandomForest (~350 MB),
  LightGBM (~29 MB, best single model), XGBoost (~14 MB), the fine-tuned DINOv2
  `backbone.pt` (~85 MB), plus feature names and label encoder. Point elsewhere
  with `--model-dir` / `--cnn-weights`, or `ROOTSCOPE_MODEL_DIR`,
  `ROOTSCOPE_CNN_WEIGHTS`, `ROOTSCOPE_HF_REPO`, `ROOTSCOPE_MODELS_URL`.

## Contact

TRAN CHAU - tnchau@vt.edu

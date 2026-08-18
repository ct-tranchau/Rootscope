# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

RootScope uses Cellpose-SAM to segment every cell and describes each one with morpho-topological features and fine-tuned DINOv2 embeddings. Classification is refined through an iterative process in which predictions from neighboring cells are fed back as new features, updating every label until the classification converges. Each cell is then assigned to one of nine anatomical types.

## Install

Needs **Python 3.10 or newer**. Check with `python -V` before you start; conda's
`base` environment is often much newer than that and is not a good place to
install into.

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
```

Then pick one of the three:

### Option A - conda (recommended, brings its own Python 3.10)

Linux or Windows with an NVIDIA GPU:

```bash
conda env create -f environment.yml
conda activate rootscope
pip install .
```

macOS, or any machine without an NVIDIA GPU:

```bash
conda env create -f environment-cpu.yml
conda activate rootscope
pip install .
```

The two files differ by one line. `environment.yml` requires an NVIDIA
metapackage that does not exist for macOS, so on a Mac it cannot be solved at
all; that is what `environment-cpu.yml` is for.

### Option B - pip, whatever Python you already have

```bash
pip install .
```

pip picks dependency versions that have prebuilt packages for your Python, so
this works on 3.13 and 3.14 as well. It does not reproduce the exact validated
version set, so results can differ slightly from the published ones.

### Option C - pip, exact validated versions

```bash
pip install -r requirements.txt     # needs Python 3.10, 3.11 or 3.12
pip install .
```

`requirements.txt` pins the precise set this release was tested with. Those
pins predate Python 3.13, so on 3.13 or newer pip finds nothing prebuilt to
match and tries to compile numpy, scipy and PyTorch from source. That needs a
Fortran compiler and OpenBLAS, and usually ends in:

```
ERROR: Dependency lookup for OpenBLAS with method 'pkg-config' failed
```

If you see that, you are on a Python that is too new for these pins. Use Option
A or B instead.

### Platform notes

| Machine | Works | Notes |
|---|---|---|
| Linux + NVIDIA GPU | yes | `environment.yml`, ~2 min per image |
| Linux, CPU only | yes | `environment-cpu.yml`, 15-30 min per image |
| Windows + NVIDIA GPU | yes | `environment.yml` |
| macOS, Apple Silicon | yes | `environment-cpu.yml`, 15-30 min per image |
| macOS, Intel | **no** | PyTorch no longer ships macOS Intel builds |

No GPU, or not wanting to install anything at all? Both routes below run the
same pipeline with nothing to set up. See "Run it in the browser" below.

The first prediction downloads the trained model weights (~460 MB) once from
[huggingface.co/ct-tranchau/Rootscope](https://huggingface.co/ct-tranchau/Rootscope)
and caches them in `~/.cache/huggingface`, so later runs start instantly.
Cellpose-SAM downloads its own segmentation weights automatically on first use,
so you need an internet connection the first time you run Rootscope.

---

## Predict
### One image

```bash
rootscope --tif my_image.tif --out results/
```
See [`examples/README.md`](examples/README.md)
### A folder of images

```bash
rootscope --tif-dir my_tifs/ --out results/
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--cpu` / `--gpu` | force CPU or GPU (default: auto-detect) |
| `--um-per-px 0.5` | microns per pixel of your image (default 1.0) |
| `--label-cells` | write the cell-type name inside each cell on the overlay (default: off) |
| `--model-dir DIR` | use your own trained weights instead of the download |
| `--cnn-weights backbone.pt` | use your own fine-tuned DINOv2 backbone |
| `--max-rounds N` | iterative-refinement rounds (default 10) |

By default the overlay shows colored cells plus a legend. `--label-cells` also
prints an abbreviated type name inside every cell, handy when zooming into one
region, but on a dense cross-section the text overlaps and hides the image,
which is why it is off by default.

### From Python

```python
from rootscope import predict_tif

df = predict_tif("my_image.tif", out_dir="results/", gpu=True)
print(df.head())        # one row per cell, with predicted cell type
```

---

## Output

For each input image, `results/` contains, per model:

- `<image>_<Model>_predictions.csv`: one row per cell (id, centroid, features,
  predicted `cell_type`)
- `<image>_<Model>_overlay.png`: original image with cells colored by predicted
  type

Batch mode also writes a combined `all_predictions.csv`.

---

## How it works

```
 raw TIFF
    │
    ▼  Cellpose-SAM               segment every cell
    ▼  layer index + adjacency    position within the root-tip tissue
    ▼  shape/size/intensity       handcrafted per-cell features
    ▼  DINOv2 embeddings          fine-tuned deep features per cell crop
    ▼  iterative ensemble         RandomForest + XGBoost + LightGBM,
    │                             refining neighbor-context each round
    ▼
 per-cell cell-type predictions  +  labeled overlay
```

The classifier is *iterative*: each round fills in each cell's neighbor
cell-types from the previous round's predictions and re-predicts, until the
predictions stop changing.

---

## Run it in the browser

### Google Colab (free GPU, nothing to install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ct-tranchau/Rootscope/blob/main/notebooks/Rootscope_Colab.ipynb)

[`notebooks/Rootscope_Colab.ipynb`](notebooks/Rootscope_Colab.ipynb) runs the
whole pipeline on Colab's free T4: upload a TIFF, get the overlay and per-cell
CSV, or batch a folder from Google Drive. It reads the pixel size from OME
metadata so you do not have to remember it.

### Web app

`webapp/` holds a Gradio front end for uploading a TIFF in the browser and getting
the overlay and per-cell CSV back, designed to run on a Hugging Face Space with
ZeroGPU, in the style of the
[Cellpose-SAM Space](https://huggingface.co/spaces/mouseland/cellpose).

```bash
conda activate rootscope
pip install gradio
python webapp/app.py
```

Only segmentation and the DINOv2 embeddings run on the GPU; feature
extraction, the iterative ensemble, and overlay drawing run on the CPU. See
[`webapp/README.md`](webapp/README.md) for the deploy steps.

---

## Retraining / full pipeline

The `pipeline/` folder contains the complete, numbered research workflow used to
build the published model (feature extraction → DINOv2 fine-tuning → training →
prediction). See [`pipeline/README.md`](pipeline/README.md). Most users only
need the `rootscope` command above.

---

## Model weights

The trained models are too large for GitHub, so they are hosted on the Hugging
Face Hub and downloaded automatically on first use:

**[huggingface.co/ct-tranchau/Rootscope](https://huggingface.co/ct-tranchau/Rootscope)**

| File | Size | |
|------|------|--|
| `model_RandomForest.joblib` + scaler | ~350 MB | bagging model; the ensemble double-weights it on minority classes |
| `model_LightGBM.joblib` + scaler | ~29 MB | best single model on held-out test data |
| `model_XGBoost.joblib` + scaler | ~14 MB | |
| `backbone.pt` | ~85 MB | fine-tuned DINOv2 backbone |
| `feature_columns.joblib`, `label_encoder.joblib` | small | 480 feature names, 9 class labels |

You do not need to download these by hand. To use your own instead, pass
`--model-dir` / `--cnn-weights`, or set `ROOTSCOPE_MODEL_DIR` /
`ROOTSCOPE_CNN_WEIGHTS`. To host them elsewhere, set `ROOTSCOPE_HF_REPO` (another
Hub repo) or `ROOTSCOPE_MODELS_URL` (any base URL).

## Contact

TRAN CHAU - tnchau@vt.edu

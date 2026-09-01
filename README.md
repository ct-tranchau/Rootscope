# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

RootScope segments every cell in a confocal root cross-section with Cellpose-SAM and assigns each one to one of nine anatomical cell types, refining the labels with neighbor context until they converge.

No install needed: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ct-tranchau/Rootscope/blob/main/notebooks/Rootscope_Colab.ipynb) - upload a TIFF, get the overlay and per-cell CSV on a free GPU.

## Install

Needs Python 3.10-3.12. Clone the repo first:

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
```

### With conda (recommended)

GPU:

```bash
conda env create -f environment.yml
conda activate rootscope
pip install .
```

CPU:

```bash
conda env create -f environment-cpu.yml
conda activate rootscope
pip install .
```

### With pip

```bash
pip install -r requirements.txt
pip install .
```

`requirements.txt` pins the validated versions. For a specific CUDA build,
install torch from [pytorch.org](https://pytorch.org/get-started/locally/) first.

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
| `--model-version v1` | use the earlier 480-feature model (default: v2) |
| `--model-dir DIR` | use your own trained weights instead of the download |
| `--cnn-weights backbone.pt` | use your own fine-tuned DINOv2 backbone |
| `--max-rounds N` | iterative-refinement rounds (default 10) |

Per image, `results/` gets `<image>_<Model>_predictions.csv` (one row per cell:
id, centroid, features, predicted `cell_type`) and `<image>_<Model>_overlay.png`
(cells colored by type). Batch mode also writes `all_predictions.csv`. See
[`examples/README.md`](examples/README.md).

## Speed and hardware

| Machine | Time per image |
|---------|----------------|
| NVIDIA GPU (Linux/Windows) | ~2 min |
| CPU or Apple Silicon | 15-30 min |
| macOS Intel | not supported - use Colab or the web app |

## Model weights

Nothing to download by hand. The first prediction fetches the trained weights
(~460 MB) from [Hugging Face](https://huggingface.co/ct-tranchau/Rootscope) into
`~/.cache/huggingface`, so the first run needs internet; later runs start
instantly.

## More

- **Web app** - try it in the browser at
  [huggingface.co/spaces/ct-tranchau/Rootscope](https://huggingface.co/spaces/ct-tranchau/Rootscope).
- **Model weights** - hosted on the
  [Hub](https://huggingface.co/ct-tranchau/Rootscope), downloaded on first use.
  Two versions are published:

  | | features | backbone | on the Hub |
  |---|---|---|---|
  | **v2** (default) | 864 = 96 morphometric + 768 embeddings | fine-tuned DINOv2 ViT-B/14 | repo root |
  | v1 | 480 = 96 morphometric + 384 embeddings | fine-tuned DINOv2 ViT-S/14 | `v1/` folder |

## Contact

TRAN CHAU - tnchau@vt.edu

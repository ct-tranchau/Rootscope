"""
Rootscope: cell-type prediction for confocal root-tip cross-section images.

Give it a TIFF, it segments the cells (Cellpose-SAM), extracts shape / size /
intensity / layer features plus fine-tuned DINOv2 embeddings, and predicts the
cell type of every cell with an iterative ensemble (RandomForest / XGBoost /
LightGBM).

Typical use (command line):

    rootscope --tif my_image.tif --out results/

Typical use (Python):

    from rootscope import predict_tif
    df = predict_tif("my_image.tif", out_dir="results/", gpu=True)
"""

__version__ = "0.1.0"

from .api import predict_tif, predict_folder  # noqa: E402,F401

__all__ = ["predict_tif", "predict_folder", "__version__"]

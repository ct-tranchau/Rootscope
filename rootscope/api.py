"""
High-level Python API for Rootscope.

    from rootscope import predict_tif, predict_folder

    df = predict_tif("image.tif", out_dir="results/", gpu=True)
    df = predict_folder("my_tifs/", out_dir="results/", gpu=True)

Both return a pandas DataFrame of per-cell predictions and also write
per-image CSVs + labeled overlay PNGs into ``out_dir``.
"""

from pathlib import Path

import pandas as pd

from . import predict as _predict
from .weights import resolve_cnn_weights, resolve_model_dir


def _load(model_dir=None, cnn_weights=None):
    mdir = resolve_model_dir(model_dir)
    cnn = resolve_cnn_weights(cnn_weights)
    models_dict, scalers, feature_cols, le = _predict.load_models(str(mdir))
    if not models_dict:
        raise RuntimeError(f"No usable models found in {mdir}.")
    return models_dict, scalers, feature_cols, le, cnn


def predict_tif(
    tif,
    out_dir="results",
    gpu=True,
    model_dir=None,
    cnn_weights=None,
    um_per_px=1.0,
    max_rounds=10,
):
    """Segment + predict cell types for a single TIFF. Returns a DataFrame."""
    models_dict, scalers, feature_cols, le, cnn = _load(model_dir, cnn_weights)
    df = _predict.predict_single_tif(
        str(tif), models_dict, scalers, feature_cols, le,
        um_per_px=um_per_px, gpu=gpu, out_dir=str(out_dir),
        max_rounds=max_rounds, cnn_weights=str(cnn) if cnn else None,
    )
    return df


def predict_folder(
    tif_dir,
    out_dir="results",
    gpu=True,
    model_dir=None,
    cnn_weights=None,
    um_per_px=1.0,
    max_rounds=10,
    pattern="*.tif",
):
    """Segment + predict for every TIFF in a folder. Returns a combined
    DataFrame and writes ``all_predictions.csv`` into ``out_dir``."""
    models_dict, scalers, feature_cols, le, cnn = _load(model_dir, cnn_weights)
    tif_paths = sorted(Path(tif_dir).glob(pattern))
    if not tif_paths:
        raise FileNotFoundError(f"No files matching {pattern} in {tif_dir}")

    tables = []
    for tp in tif_paths:
        try:
            df = _predict.predict_single_tif(
                str(tp), models_dict, scalers, feature_cols, le,
                um_per_px=um_per_px, gpu=gpu, out_dir=str(out_dir),
                max_rounds=max_rounds, cnn_weights=str(cnn) if cnn else None,
            )
            if df is not None:
                tables.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED on {tp.name}: {e}")

    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "all_predictions.csv", index=False)
    return combined

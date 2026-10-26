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
from .weights import is_v4, resolve_cnn_weights, resolve_model_dir


def _load(model_dir=None, cnn_weights=None):
    """Resolve weights for the selected model version.

    v4 returns ``(model_dir, None, None, None, cnn)``: its classifiers are
    loaded inside the v4 stage (one bundle per seed, each carrying its own
    scaler and feature list), so there is nothing to pre-load here.
    """
    mdir = resolve_model_dir(model_dir)
    cnn = resolve_cnn_weights(cnn_weights)
    if is_v4():
        return mdir, None, None, None, cnn
    models_dict, scalers, feature_cols, le = _predict.load_models(str(mdir))
    if not models_dict:
        raise RuntimeError(f"No usable models found in {mdir}.")
    return models_dict, scalers, feature_cols, le, cnn


def _gpu_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _run_one(tif, loaded, out_dir, gpu, um_per_px, max_rounds, label_cells,
             no_exodermis=None, smooth_layers=True, species=None):
    if gpu is None:                      # library default: use a GPU if there is one
        gpu = _gpu_available()
    models_dict, scalers, feature_cols, le, cnn = loaded
    if is_v4():
        return _predict.predict_single_tif_v4(
            str(tif), model_dir=str(models_dict), um_per_px=um_per_px, gpu=gpu,
            out_dir=str(out_dir), max_rounds=max_rounds,
            cnn_weights=str(cnn) if cnn else None, label_cells=label_cells,
            no_exodermis=no_exodermis, smooth_layers=smooth_layers, species=species,
        )
    if um_per_px is None:                # v2 path: OME pixel size if present, else the old 1.0 default
        um_per_px = _predict.read_um_per_px(tif) or 1.0
    return _predict.predict_single_tif(
        str(tif), models_dict, scalers, feature_cols, le,
        um_per_px=um_per_px, gpu=gpu, out_dir=str(out_dir),
        max_rounds=max_rounds, cnn_weights=str(cnn) if cnn else None,
        label_cells=label_cells,
    )


def predict_tif(
    tif,
    out_dir="results",
    gpu=None,
    model_dir=None,
    cnn_weights=None,
    um_per_px=None,
    max_rounds=None,
    label_cells=False,
    species=None,
    no_exodermis=None,
    smooth_layers=True,
):
    """Segment + predict cell types for a single TIFF. Returns a DataFrame.

    ``um_per_px=None`` reads the pixel size from the TIFF's OME metadata and
    fails loudly if it is absent -- a wrong scale distorts every size feature.
    ``max_rounds=None`` uses the version's own setting (6 for v4, 10 for v2).
    ``species`` (optional, e.g. "Pisum sativum") is looked up in the species
    table and used for the exodermis decision and the report (v4 only).
    ``no_exodermis``: None decides from the image and the species table;
    True/False is your own call and overrides that (v4 only).
    ``smooth_layers`` removes single cells of another class inside the
    epidermis, exodermis or cortex layer (v4 only).
    Next to the CSV, ``<stem>_LightGBM_report.txt`` says how far to trust the
    result.
    """
    loaded = _load(model_dir, cnn_weights)
    return _run_one(tif, loaded, out_dir, gpu, um_per_px,
                    _predict.default_rounds(max_rounds), label_cells,
                    no_exodermis=no_exodermis, smooth_layers=smooth_layers,
                    species=species)


def predict_folder(
    tif_dir,
    out_dir="results",
    gpu=None,
    model_dir=None,
    cnn_weights=None,
    um_per_px=None,
    max_rounds=None,
    label_cells=False,
    pattern="*.tif",
    species=None,
    no_exodermis=None,
    smooth_layers=True,
):
    """Segment + predict for every TIFF in a folder. Returns a combined
    DataFrame and writes ``all_predictions.csv`` into ``out_dir``.
    ``species`` and ``no_exodermis`` apply to every image in the folder."""
    loaded = _load(model_dir, cnn_weights)
    rounds = _predict.default_rounds(max_rounds)
    tif_paths = sorted(Path(tif_dir).glob(pattern))
    if not tif_paths:
        raise FileNotFoundError(f"No files matching {pattern} in {tif_dir}")

    tables = []
    for tp in tif_paths:
        try:
            df = _run_one(tp, loaded, out_dir, gpu, um_per_px, rounds, label_cells,
                          no_exodermis=no_exodermis, smooth_layers=smooth_layers,
                          species=species)
            if df is not None:
                tables.append(df)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAILED on {tp.name}: {e}")
            traceback.print_exc()

    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "all_predictions.csv", index=False)
    return combined

"""
Command-line entry point for Rootscope:  `rootscope ...`

Examples
--------
    # one image (GPU auto-detected)
    rootscope --tif image.tif --out results/

    # a whole folder of TIFFs
    rootscope --tif-dir my_tifs/ --out results/

    # force CPU (slow), or point at your own weights
    rootscope --tif image.tif --cpu
    rootscope --tif image.tif --model-dir /path/to/models --cnn-weights /path/to/backbone.pt
"""

import argparse
import sys

from . import __version__, api


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rootscope",
        description="Predict cell types in confocal root-tip TIFF images "
                    "(segmentation + prediction in one step).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tif", help="Path to a single TIFF image.")
    src.add_argument("--tif-dir", help="Folder of TIFF images (batch mode).")

    p.add_argument("--out", "--out-dir", dest="out_dir", default="results",
                   help="Output folder for CSVs + overlays (default: results/).")
    p.add_argument("--um-per-px", type=float, default=None,
                   help="Microns per pixel. Default: read from the TIFF's OME "
                        "metadata (PhysicalSizeX); the run stops if it is "
                        "missing, because a wrong scale distorts every "
                        "size feature.")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Max iterative refinement rounds (default: 6 for v4, "
                        "10 for v1/v2 -- the settings each model was "
                        "evaluated with).")
    p.add_argument("--species", default=None,
                   help="Species name, e.g. 'Pisum sativum' (optional). Used for "
                        "the exodermis decision and the report (v4 only).")
    ex = p.add_mutually_exclusive_group()
    ex.add_argument("--no-exodermis", dest="no_exodermis", action="store_const",
                    const=True, default=None,
                    help="Your own call: the root has no exodermis. Without this "
                         "flag the decision comes from the image and the species "
                         "table (v4 only).")
    ex.add_argument("--has-exodermis", dest="no_exodermis", action="store_const",
                    const=False,
                    help="Your own call: the root has an exodermis (v4 only).")
    p.add_argument("--no-layer-smoothing", dest="smooth_layers",
                   action="store_false", default=True,
                   help="Disable the along-the-layer smoothing of epidermis, "
                        "exodermis and cortex (v4 only).")
    p.add_argument("--label-cells", action="store_true",
                   help="Write the cell-type name inside each cell on the "
                        "overlay. Off by default (the text overlaps on dense "
                        "sections); the color legend already shows the types.")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", dest="gpu", action="store_true", default=None,
                   help="Force GPU (default: auto-detect).")
    g.add_argument("--cpu", dest="gpu", action="store_false",
                   help="Force CPU (slow).")

    p.add_argument("--model-version", choices=("v1", "v2", "v4"), default=None,
                   help="Which published model to use. v4 (default): the "
                        "manuscript model -- 915 features (864 + 51 rebuilt "
                        "every refinement round), one LightGBM per round, "
                        "3 seeds averaged, radial-order prior; 0.881 accuracy "
                        "on the clean held-out split. v2: 864 features, "
                        "RF/XGB/LGBM ensemble, 10 rounds. v1: 480 features, "
                        "DINOv2 ViT-S/14.")
    p.add_argument("--model-dir", default=None,
                   help="Folder with trained model weights (default: "
                        "auto-download/cache).")
    p.add_argument("--cnn-weights", default=None,
                   help="Fine-tuned DINOv2 backbone.pt (default: "
                        "auto-download/cache).")
    p.add_argument("--pattern", default="*.tif",
                   help="Glob for --tif-dir mode (default: *.tif).")
    p.add_argument("--version", action="version",
                   version=f"rootscope {__version__}")
    return p


def main(argv=None):
    import os as _os
    args = build_parser().parse_args(argv)
    # set before anything resolves weights, so the classifiers and the backbone
    # are taken from the same version's folder
    if args.model_version:
        _os.environ["ROOTSCOPE_MODEL_VERSION"] = args.model_version
    use_gpu = _gpu_available() if args.gpu is None else args.gpu
    if args.gpu is None:
        print(f"[rootscope] GPU {'detected' if use_gpu else 'not found (using CPU)'}.")

    try:
        if args.tif:
            df = api.predict_tif(
                args.tif, out_dir=args.out_dir, gpu=use_gpu,
                model_dir=args.model_dir, cnn_weights=args.cnn_weights,
                um_per_px=args.um_per_px, max_rounds=args.max_rounds,
                label_cells=args.label_cells, no_exodermis=args.no_exodermis,
                smooth_layers=args.smooth_layers, species=args.species,
            )
            n = 0 if df is None else len(df)
            print(f"\n[rootscope] Done. {n} cell predictions written to {args.out_dir}/")
        else:
            df = api.predict_folder(
                args.tif_dir, out_dir=args.out_dir, gpu=use_gpu,
                model_dir=args.model_dir, cnn_weights=args.cnn_weights,
                um_per_px=args.um_per_px, max_rounds=args.max_rounds,
                label_cells=args.label_cells, pattern=args.pattern,
                no_exodermis=args.no_exodermis, smooth_layers=args.smooth_layers,
                species=args.species,
            )
            n = 0 if df is None else len(df)
            print(f"\n[rootscope] Done. {n} total predictions written to {args.out_dir}/")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[rootscope] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

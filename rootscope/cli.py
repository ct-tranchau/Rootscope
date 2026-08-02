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
    p.add_argument("--um-per-px", type=float, default=1.0,
                   help="Microns per pixel (default: 1.0).")
    p.add_argument("--max-rounds", type=int, default=10,
                   help="Max iterative refinement rounds (default: 10).")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--gpu", dest="gpu", action="store_true", default=None,
                   help="Force GPU (default: auto-detect).")
    g.add_argument("--cpu", dest="gpu", action="store_false",
                   help="Force CPU (slow).")

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
    args = build_parser().parse_args(argv)
    use_gpu = _gpu_available() if args.gpu is None else args.gpu
    if args.gpu is None:
        print(f"[rootscope] GPU {'detected' if use_gpu else 'not found (using CPU)'}.")

    try:
        if args.tif:
            df = api.predict_tif(
                args.tif, out_dir=args.out_dir, gpu=use_gpu,
                model_dir=args.model_dir, cnn_weights=args.cnn_weights,
                um_per_px=args.um_per_px, max_rounds=args.max_rounds,
            )
            n = 0 if df is None else len(df)
            print(f"\n[rootscope] Done — {n} cell predictions written to {args.out_dir}/")
        else:
            df = api.predict_folder(
                args.tif_dir, out_dir=args.out_dir, gpu=use_gpu,
                model_dir=args.model_dir, cnn_weights=args.cnn_weights,
                um_per_px=args.um_per_px, max_rounds=args.max_rounds,
                pattern=args.pattern,
            )
            n = 0 if df is None else len(df)
            print(f"\n[rootscope] Done — {n} total predictions written to {args.out_dir}/")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[rootscope] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

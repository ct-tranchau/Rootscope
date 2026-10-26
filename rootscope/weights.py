"""
Model-weight resolution for Rootscope.

The trained classifier models and the fine-tuned DINOv2 backbone are too large
to ship inside the pip/conda package, so Rootscope resolves them at run time in
this priority order:

  1. An explicit path you pass (``--model-dir`` / ``--cnn-weights``).
  2. Environment variables ``ROOTSCOPE_MODEL_DIR`` / ``ROOTSCOPE_CNN_WEIGHTS``.
  3. A local ``models/`` folder next to the installed package.
  4. Auto-download from the Hugging Face Hub repo ``DEFAULT_HF_REPO``, cached in
     ``~/.cache/huggingface`` (override with ``ROOTSCOPE_HF_REPO``).
  5. Auto-download from a plain base URL, if ``ROOTSCOPE_MODELS_URL`` is set,
     an escape hatch for self-hosting the weights somewhere else.

The very first prediction downloads the weights once; every run after that uses
the cache. Downloads via the Hub resume if interrupted and are checksummed, so
a dropped connection does not mean starting over.

Note: the Cellpose-SAM segmentation weights are NOT handled here; the
``cellpose`` library downloads and caches those itself on first use.
"""

# `X | None` annotations below need this on Python 3.9,
# which pyproject still declares as the supported floor.
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Where the published weights live: a Hugging Face model repo whose root
# contains the .joblib model files and backbone.pt.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_HF_REPO = "ct-tranchau/Rootscope"

# Two published model versions live in one Hub repo:
#   v2 -> "v2/" folder   96 morphometric + 768 DINOv2 ViT-B/14  = 864 features
#   v1 -> "v1/" folder   96 morphometric + 384 DINOv2 ViT-S/14  = 480 features
# v2 is the default: it is the model evaluated in the manuscript and the one the
# hosted demo runs. Its backbone is heavier, but on a GPU the embedding stage is
# a few seconds either way -- end to end the two versions measured within 2% of
# each other, because Cellpose segmentation dominates and is identical.
# v1 is kept for reproducing earlier results. Select with --model-version or
# ROOTSCOPE_MODEL_VERSION.
# v4 -> "v4/" folder   96 morphometric + 768 DINOv2 ViT-B/14, PLUS 51 columns the
#                      classifier rebuilds every refinement round (36 soft-neighbor
#                      probabilities, 6 agreement flags, 9 layer-distance features),
#                      = 915 features; one LightGBM per round, 3 seeds averaged,
#                      radial-order prior after the last round. This is the model
#                      in the manuscript (0.881 accuracy, 0.852 macro-F1 on the
#                      clean 26-section split). Default since 4.0.0.
MODEL_VERSIONS = ("v1", "v2", "v4")
DEFAULT_MODEL_VERSION = "v4"


def model_version() -> str:
    v = os.environ.get("ROOTSCOPE_MODEL_VERSION", DEFAULT_MODEL_VERSION).lower()
    if v not in MODEL_VERSIONS:
        raise ValueError(
            f"ROOTSCOPE_MODEL_VERSION must be one of {MODEL_VERSIONS}, got '{v}'."
        )
    return v


def _hf_prefix() -> str:
    """Subfolder inside the Hub repo holding the selected version's weights.
    Every version has a folder named after it (v1/, v2/, v4/); the repo root
    holds only the model card. v2 lived at the root until 2026-09-21."""
    return f"{model_version()}/"


def is_v4() -> bool:
    return model_version() == "v4"

# Optional escape hatch: a plain base URL under which each file name below is
# directly downloadable, e.g. "https://zenodo.org/records/XXXXXXX/files".
# Only used if the environment variable ROOTSCOPE_MODELS_URL is set.
DEFAULT_MODELS_URL = ""

# Classifier artifacts.
#
# All three models are published and downloaded by default. RandomForest is the
# large one (~350 MB), and it is listed as OPTIONAL only so that prediction
# degrades gracefully to XGBoost + LightGBM if a user supplies their own
# --model-dir without it. Do not treat that as a reason to omit it from a
# release: the ensemble double-weights RandomForest wherever it predicts a
# minority class (see the RF-trust rule in predict.py), and RF is the only
# bagging model of the three, so it decorrelates from the two boosting models.
MODEL_FILES_REQUIRED = [
    "feature_columns.joblib",
    "label_encoder.joblib",
    "model_XGBoost.joblib",
    "feature_scaler_XGBoost.joblib",
    "model_LightGBM.joblib",
    "feature_scaler_LightGBM.joblib",
]
MODEL_FILES_OPTIONAL = [
    "model_RandomForest.joblib",
    "feature_scaler_RandomForest.joblib",
]
CNN_WEIGHTS_FILE = "backbone.pt"

# v4 artifacts. One LightGBM bundle per training seed; each holds the per-round
# models, the scaler, the 915 feature names and the class order. Seed 42 alone
# gives a working predictor; all three reproduce the manuscript's 3-seed average.
# v4.1 adds the geometry-only bundles (lgbm_geom_s*), averaged with the full
# model on the outer layers; seed 42 of those is required so that the published
# model is the validated one (without it classify_v4 falls back to v4.0).
V4_MODEL_FILES_REQUIRED = ["lgbm_s42.joblib", "lgbm_geom_s42.joblib"]
V4_MODEL_FILES_OPTIONAL = ["lgbm_s1.joblib", "lgbm_s7.joblib",
                           "lgbm_geom_s1.joblib", "lgbm_geom_s7.joblib"]


def _required_files():
    return V4_MODEL_FILES_REQUIRED if is_v4() else MODEL_FILES_REQUIRED


def _optional_files():
    return V4_MODEL_FILES_OPTIONAL if is_v4() else MODEL_FILES_OPTIONAL


def _cache_dir() -> Path:
    root = os.environ.get("ROOTSCOPE_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "rootscope"
    return base


def _package_models_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def _hf_repo() -> str:
    return os.environ.get("ROOTSCOPE_HF_REPO", DEFAULT_HF_REPO).strip()


def _models_base_url() -> str:
    return os.environ.get("ROOTSCOPE_MODELS_URL", DEFAULT_MODELS_URL).rstrip("/")


def _has_required_models(d: Path) -> bool:
    return d.is_dir() and all((d / f).exists() for f in _required_files())


# ── Hugging Face Hub ─────────────────────────────────────────────────────────

def _hf_snapshot(repo: str, patterns) -> Path | None:
    """Download the matching files from the Hub and return the local folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[rootscope] huggingface_hub is not installed, so the model weights "
            "cannot be downloaded automatically.\n"
            "            Install it with `pip install huggingface_hub`, or pass "
            "--model-dir / --cnn-weights explicitly."
        )
        return None

    try:
        local = snapshot_download(repo_id=repo, allow_patterns=list(patterns))
        return Path(local)
    except Exception as e:  # noqa: BLE001
        print(f"[rootscope] Could not fetch weights from Hugging Face repo '{repo}': {e}")
        return None


# ── plain-URL fallback ───────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        sys.stdout.write(
            f"\r    {dest.name}: {done/1e6:6.1f} / {total_size/1e6:6.1f} MB "
            f"({pct:5.1f}%)"
        )
        sys.stdout.flush()

    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, tmp, _hook)  # noqa: S310 (trusted release URL)
    sys.stdout.write("\n")
    tmp.replace(dest)


def _download_set(base_url: str, files, dest_dir: Path, skip_missing: bool):
    for name in files:
        target = dest_dir / name
        if target.exists():
            continue
        url = f"{base_url}/{name}"
        try:
            _download(url, target)
        except Exception as e:  # noqa: BLE001
            if skip_missing:
                print(f"    (optional) skipped {name}: {e}")
            else:
                raise


# ── public API ───────────────────────────────────────────────────────────────

def resolve_model_dir(user_arg: str | None = None) -> Path:
    """Return a directory containing the classifier artifacts, downloading
    them on first use if necessary."""
    # 1. explicit argument
    if user_arg:
        d = Path(user_arg)
        if not _has_required_models(d):
            raise FileNotFoundError(
                f"--model-dir '{d}' does not contain the required model files "
                f"for model version {model_version()} ({', '.join(_required_files())})."
            )
        return d

    # 2. environment variable
    env = os.environ.get("ROOTSCOPE_MODEL_DIR")
    if env and _has_required_models(Path(env)):
        return Path(env)

    # 3. models/ folder shipped alongside the package
    pkg = _package_models_dir()
    if _has_required_models(pkg):
        return pkg

    # 4. legacy cache from a previous plain-URL download
    cache = _cache_dir() / "models"
    if _has_required_models(cache):
        return cache

    # 5. Hugging Face Hub (the normal path)
    repo = _hf_repo()
    if repo:
        print(f"[rootscope] Fetching model weights from Hugging Face '{repo}' (first run only)...")
        pref = _hf_prefix()
        # Name the files rather than globbing: "*.joblib" also matches
        # "v1/*.joblib", which would drag the other version's weights down too.
        snap = _hf_snapshot(
            repo, [f"{pref}{f}" for f in _required_files() + _optional_files()]
        )
        if snap is not None and _has_required_models(snap / pref if pref else snap):
            return snap / pref if pref else snap

    # 6. plain base URL, if configured
    base_url = _models_base_url()
    if base_url:
        print(f"[rootscope] Fetching model weights into {cache} (first run only)...")
        _download_set(base_url, _required_files(), cache, skip_missing=False)
        _download_set(base_url, _optional_files(), cache, skip_missing=True)
        return cache

    raise RuntimeError(
        "Rootscope could not find or download the trained model weights.\n"
        "Do ONE of the following:\n"
        "  • Check your internet connection (weights come from the Hugging Face "
        f"repo '{_hf_repo()}'), or\n"
        "  • Put the model files in a folder and pass --model-dir <folder>, or\n"
        "  • export ROOTSCOPE_MODEL_DIR=/path/to/models, or\n"
        "  • export ROOTSCOPE_MODELS_URL=<base url> to self-host them.\n"
        f"Required files ({model_version()}): {', '.join(_required_files())}"
    )


def resolve_cnn_weights(user_arg: str | None = None) -> Path | None:
    """Return the path to the fine-tuned DINOv2 backbone, or None to fall back
    to pretrained DINOv2 (lower accuracy)."""
    if user_arg:
        p = Path(user_arg)
        if not p.exists():
            raise FileNotFoundError(f"--cnn-weights '{p}' not found.")
        return p

    env = os.environ.get("ROOTSCOPE_CNN_WEIGHTS")
    if env and Path(env).exists():
        return Path(env)

    pkg = _package_models_dir() / CNN_WEIGHTS_FILE
    if pkg.exists():
        return pkg

    cache = _cache_dir() / "models" / CNN_WEIGHTS_FILE
    if cache.exists():
        return cache

    repo = _hf_repo()
    if repo:
        pref = _hf_prefix()
        # meta.json travels with the backbone: without it cnn_embeddings falls
        # back to ViT-S/14 and silently produces the wrong embedding width.
        snap = _hf_snapshot(repo, [f"{pref}{CNN_WEIGHTS_FILE}", f"{pref}meta.json"])
        if snap is not None and (snap / pref / CNN_WEIGHTS_FILE).exists():
            return snap / pref / CNN_WEIGHTS_FILE

    base_url = _models_base_url()
    if base_url:
        try:
            _download(f"{base_url}/{CNN_WEIGHTS_FILE}", cache)
            return cache
        except Exception as e:  # noqa: BLE001
            print(f"[rootscope] Could not download {CNN_WEIGHTS_FILE}: {e}")

    print(
        "[rootscope] WARNING: fine-tuned DINOv2 backbone not found, falling "
        "back to pretrained DINOv2. Predictions will be less accurate than the "
        "published model. Provide --cnn-weights backbone.pt to fix this."
    )
    return None

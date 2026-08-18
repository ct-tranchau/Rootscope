"""
RootScope on Hugging Face Spaces: upload a root-tip TIFF, get segmented and
classified cells back.

Modelled on the Cellpose-SAM Space (huggingface.co/spaces/mouseland/cellpose),
with one important difference: only two steps of the RootScope pipeline touch
the GPU (Cellpose-SAM segmentation and the DINOv2 embeddings). Those are the
only ones wrapped in @spaces.GPU. Feature extraction, the iterative
RandomForest/XGBoost/LightGBM refinement, anatomical post-processing and
overlay drawing all run on the CPU worker, off the GPU clock.

Run locally with:  python webapp/app.py
"""

import functools
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from rootscope import predict as rs
from rootscope.cnn_embeddings import load_dinov2
from rootscope.extract_features import load_cellpose_model
from rootscope.weights import resolve_cnn_weights, resolve_model_dir

# ── ZeroGPU shim ─────────────────────────────────────────────────────────────
# `spaces` only exists on Hugging Face. Locally the decorator becomes a no-op
# so the same file runs on a workstation or a lab GPU node.
try:
    import spaces
except ImportError:  # noqa: BLE001
    class _NoSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def wrap(fn):
                return fn
            return wrap

    spaces = _NoSpaces()  # type: ignore[assignment]


print = functools.partial(print, flush=True)  # noqa: A001 (Spaces log stdout)

# Everything the UI serves back (previews, overlays, CSVs, ZIPs) is written
# under here, and this one directory is handed to launch(allowed_paths=...).
# Gradio refuses to serve files from arbitrary locations on disk, so results
# must live somewhere it has been told about.
TMP_ROOT = Path(tempfile.gettempdir()) / "rootscope_runs"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

MAX_PIXELS = 40_000_000  # ~6300x6300; refuse anything larger

# The Diagnostics panel is a developer tool, not something a visitor should
# see, so it is off by default. It earned its keep once already: it is how the
# ZeroGPU "GPU task aborted" failure was pinned to the XGBoost import rather
# than to the pipeline, so it stays in the file behind a switch instead of
# being deleted. To turn it on, add ROOTSCOPE_DIAGNOSTICS=1 in the Space's
# Settings → Variables and restart; no code change needed.
SHOW_DIAGNOSTICS = os.environ.get("ROOTSCOPE_DIAGNOSTICS") == "1"
MODEL_CHOICES = ["Ensemble", "RandomForest", "LightGBM", "XGBoost"]

# ── fetch weights at startup, load nothing ───────────────────────────────────
# Downloading here means the first visitor does not wait for ~550 MB. Note that
# these calls only put files on disk; they unpickle nothing.
print("[rootscope-web] fetching weights...")
MODEL_DIR = resolve_model_dir()
CNN_WEIGHTS = resolve_cnn_weights()

# NOTE: the classifiers are deliberately NOT loaded here.
#
# Unpickling XGBoost builds a Booster, which probes for CUDA devices, which
# registers state ZeroGPU's snapshot/restore cannot reproduce. Once that has
# happened in this process every @spaces.GPU task aborts, verified on this
# Space, where the diagnostics' bare torch.zeros(2048, 2048, device="cuda")
# came back "GPU task aborted" on a healthy CUDA 13.0 / torch 2.11 box.
#
# Having the model merely resident is enough, so loading it lazily on the
# classification path would not help: the GPU stages run first and would
# already be poisoned on the second request. Every classifier therefore lives
# in a subprocess that exits when it is done. See classify_worker.py.
CLASSIFY_WORKER = Path(__file__).resolve().parent / "classify_worker.py"

# NOTE: nothing is placed on CUDA at import.
#
# Hugging Face documents the opposite - models "must be placed on cuda at the
# root module level" - and that is what this file did originally. Moving the
# GPU models here was blamed for the ZeroGPU aborts at the time; that was the
# wrong culprit. The classifiers were the problem (see above), and once they
# left the process the diagnostics passed with the GPU models still lazy.
#
# So this arrangement is no longer load-bearing, only untested against the
# alternative. Module-level CUDA loading is what HF recommends and would skip
# rebuilding Cellpose-SAM per call, but the fork that runs each @spaces.GPU
# task discards anything these helpers cache anyway, and a measured run came in
# at 24 s end to end. Not worth churning until something needs the speed.
_GPU_MODELS = {}


def _gpu_cellpose():
    if "cellpose" not in _GPU_MODELS:
        print("[rootscope-web] building Cellpose-SAM on GPU (first GPU call)")
        _GPU_MODELS["cellpose"] = load_cellpose_model(use_gpu=True)
    return _GPU_MODELS["cellpose"]


def _gpu_dinov2():
    if "dinov2" not in _GPU_MODELS:
        print("[rootscope-web] building DINOv2 on GPU (first GPU call)")
        _GPU_MODELS["dinov2"] = load_dinov2(
            weights_path=str(CNN_WEIGHTS) if CNN_WEIGHTS else None, use_gpu=True)
    return _GPU_MODELS["dinov2"]


print(f"[rootscope-web] ready: weights at {MODEL_DIR}")
print("[rootscope-web] no classifier loaded in this process (ZeroGPU safe); "
      "classification runs in a subprocess")
print("[rootscope-web] GPU models load on first use (no CUDA touched at import)")


# ── the two GPU stages ───────────────────────────────────────────────────────
# Duration is tiered by image size, as the Cellpose-SAM Space does, so a small
# image does not reserve a long slot against the user's daily GPU quota.

# ZeroGPU bills the wall-clock of each @spaces.GPU call against the visitor's
# daily quota, and a call that overruns its reservation is killed. Rather than
# bucket into fixed tiers, ask for a duration computed from the actual input;
# ZeroGPU accepts a callable taking the same arguments as the function.
#
# Calibrated from CPU measurements (segmentation 95 s at 700x700 / 148 s at
# 910x910; embeddings 43 s for 215 cells / 80 s for 448 cells) then floored
# generously, since a GPU should be far quicker but an overrun is fatal while
# over-reserving only costs queue priority.

# The floor is 120 s, not 60: the first call on a fresh ZeroGPU allocation pays
# CUDA context init and the transfer of the 1.15 GB Cellpose-SAM checkpoint
# before any real work starts. Overrunning kills the job; over-reserving only
# costs queue priority.
def _segment_duration(img_rgb):
    px = max(img_rgb.shape[:2])
    return int(min(300, max(150, px / 1000 * 90)))


def _embed_duration(masks, img_rgb, df_base):
    n = int(masks.max())
    return int(min(300, max(150, n / 400 * 90)))


@spaces.GPU(duration=_segment_duration)
def _segment(img_rgb):
    import traceback
    try:
        return rs.stage_segment(img_rgb, gpu=True, cellpose_model=_gpu_cellpose())
    except Exception:
        # Without this the only symptom is ZeroGPU reporting "aborted".
        print("[rootscope-web] SEGMENTATION FAILED\n" + traceback.format_exc())
        raise


@spaces.GPU(duration=_embed_duration)
def _embed(masks, img_rgb, df_base):
    import traceback
    try:
        return rs.stage_embed(masks, img_rgb, df_base, gpu=True,
                              dinov2_model=_gpu_dinov2())
    except Exception:
        print("[rootscope-web] EMBEDDING FAILED\n" + traceback.format_exc())
        raise


# ── ZeroGPU diagnostics ──────────────────────────────────────────────────────
# "GPU task aborted" means the worker process died, so nothing it printed
# survives. Each step below is its own @spaces.GPU call: whichever one fails to
# come back is the culprit.

@spaces.GPU(duration=60)
def _diag_alloc():
    import torch
    x = torch.zeros(2048, 2048, device="cuda")
    return f"{torch.cuda.get_device_name(0)}, alloc ok, sum={float(x.sum())}"


@spaces.GPU(duration=300)
def _diag_alloc_long():
    """Same trivial work, maximum reservation.

    Before running the task, ZeroGPU has to materialise every CUDA tensor this
    process registered at startup - which here means the 1.15 GB Cellpose-SAM
    checkpoint and the DINOv2 backbone. If that restore is what blows the
    budget, a trivial allocation still fails at 60 s but passes at 300 s.
    """
    import torch
    x = torch.zeros(2048, 2048, device="cuda")
    return f"{torch.cuda.get_device_name(0)}, alloc ok at 300s, sum={float(x.sum())}"


@spaces.GPU(duration=60)
def _diag_devices():
    import torch
    out = []
    try:
        p = next(_gpu_cellpose().net.parameters())
        out.append(f"cellpose net on {p.device} ({p.dtype})")
    except Exception as e:
        out.append(f"cellpose net ERR {type(e).__name__}: {e}")
    try:
        p = next(_gpu_dinov2().parameters())
        out.append(f"dinov2 on {p.device}")
    except Exception as e:
        out.append(f"dinov2 ERR {type(e).__name__}: {e}")
    out.append(f"torch {torch.__version__}")
    return "; ".join(out)


@spaces.GPU(duration=120)
def _diag_cellpose():
    import numpy as np
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    m = rs.stage_segment(img, gpu=True, cellpose_model=_gpu_cellpose())
    return f"segmented 256x256 noise -> {int(m.max())} labels, dtype {m.dtype}"


@spaces.GPU(duration=120)
def _diag_dino():
    import numpy as np
    from rootscope.cnn_embeddings import extract_cnn_embeddings
    masks = np.zeros((128, 128), dtype=np.int32)
    masks[20:60, 20:60] = 1
    masks[70:110, 70:110] = 2
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    df = extract_cnn_embeddings(masks, img, use_gpu=True, model=_gpu_dinov2())
    return f"embedded {len(df)} cells, {df.shape[1] - 1} dims"


def env_report():
    """What actually got installed. Runs in the main process, so it still
    reports even when every @spaces.GPU task aborts."""
    import importlib, torch
    lines = [
        f"torch            {torch.__version__}",
        f"torch.version.cuda {torch.version.cuda}",
        f"cuda.is_available  {torch.cuda.is_available()}",
        f"cuda.device_count  {torch.cuda.device_count()}",
    ]
    for mod in ("torchvision", "spaces", "gradio", "cellpose", "numpy",
                "sklearn", "cv2", "xgboost", "lightgbm"):
        try:
            m = importlib.import_module(mod)
            lines.append(f"{mod:16s} {getattr(m, '__version__', '?')}")
        except Exception as e:
            lines.append(f"{mod:16s} IMPORT FAILED: {type(e).__name__}: {e}")
    import os
    lines.append(f"ZEROGPU env      "
                 f"{ {k: v for k, v in os.environ.items() if 'ZERO' in k.upper() or 'SPACES' in k.upper()} }")
    return "\n".join(lines)


def run_diagnostics():
    steps = [
        ("1. CUDA allocation (60s)", _diag_alloc),
        ("1b. CUDA allocation (300s)", _diag_alloc_long),
        ("2. model devices", _diag_devices),
        ("3. Cellpose-SAM on GPU", _diag_cellpose),
        ("4. DINOv2 on GPU", _diag_dino),
    ]
    lines = []
    for name, fn in steps:
        try:
            lines.append(f"PASS  {name}: {fn()}")
        except Exception as e:
            lines.append(f"FAIL  {name}: {type(e).__name__}: {e}")
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────

def read_um_per_px(tif_path):
    """Pull the pixel size out of the TIFF metadata, if it has any.

    RootScope does not read the scale from the file, and the 1.0 default
    silently distorts every size-derived feature, so prefill the box rather
    than let the user forget.
    """
    try:
        with tifffile.TiffFile(tif_path) as tf:
            if tf.ome_metadata:
                m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', tf.ome_metadata)
                if m:
                    return round(float(m.group(1)), 6)
            tags = tf.pages[0].tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                if num:
                    px_per_unit = num / den
                    unit = int(tags["ResolutionUnit"].value)
                    if unit == 3:      # centimetre
                        return round(10_000.0 / px_per_unit, 6)
                    if unit == 2:      # inch
                        return round(25_400.0 / px_per_unit, 6)
    except Exception as e:  # noqa: BLE001
        print(f"[rootscope-web] could not read pixel size: {e}")
    return None


def _preview_png(img_rgb):
    """Write a display-only PNG of the (max-projected) image.

    The gr.Image component is filepath-typed, so whatever we show becomes its
    value. That displayed PNG must never become the thing we segment. The
    original TIFF path is kept in a State instead.
    """
    tmp = tempfile.NamedTemporaryFile(suffix="_preview.png", delete=False,
                                      dir=TMP_ROOT)
    Image.fromarray(img_rgb.astype(np.uint8)).save(tmp.name)
    return tmp.name


def on_upload(filepath):
    """Show the uploaded image and prefill um/px from its metadata."""
    if not filepath:
        return None, gr.update(), "Upload a root-tip TIFF to begin.", None
    try:
        img_rgb = rs.load_image(filepath)
    except Exception as e:  # noqa: BLE001
        return None, gr.update(), f"Could not read that file: {e}", None

    preview = _preview_png(img_rgb)
    h, w = img_rgb.shape[:2]
    if Path(filepath).suffix.lower() not in (".tif", ".tiff"):
        return (preview, gr.update(),
                f"That file is `{Path(filepath).suffix}`, not a TIFF. RootScope "
                f"expects the original 16-bit confocal TIFF. A PNG or JPEG has "
                f"already lost bit depth and metadata.", filepath)
    scale = read_um_per_px(filepath)
    if scale is None:
        note = (f"Loaded {w}×{h}. **No pixel size in the metadata**. Set "
                f"microns/pixel yourself; leaving it at 1.0 distorts every "
                f"size-derived feature.")
        return preview, gr.update(), note, filepath
    note = f"Loaded {w}×{h}. Pixel size from metadata: **{scale} µm/px**."
    return preview, gr.update(value=scale), note, filepath


def _classify_subprocess(workdir, **job):
    """Run the CPU classification stage in a child process and return its table.

    The child loads the classifiers itself and exits, so no XGBoost Booster is
    ever built in this process. See classify_worker.py for why that matters.

    The job goes over a pickle in the run's own directory rather than through a
    pipe: `masks` and `img_rgb` are a few MB each and a file is easier to
    inspect when something goes wrong.
    """
    job_path = workdir / "_classify_job.pkl"
    result_path = workdir / "_classify_result.pkl"
    job["out_dir"] = str(workdir)
    job["model_dir"] = str(MODEL_DIR)
    with open(job_path, "wb") as f:
        pickle.dump(job, f, protocol=pickle.HIGHEST_PROTOCOL)

    proc = subprocess.run(
        [sys.executable, str(CLASSIFY_WORKER), str(job_path), str(result_path)],
        # so `import rootscope` resolves the same copy the parent imported
        cwd=str(CLASSIFY_WORKER.parent),
        capture_output=True, text=True, timeout=1800,
    )
    # The child's stdout is the classification log, so surface it in the Space
    # logs as if it had run here.
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print("[classify-worker stderr]\n" + proc.stderr, end="")

    if not result_path.exists():
        raise gr.Error(
            f"Classification subprocess died without a result "
            f"(exit {proc.returncode}). See the Space logs."
        )
    with open(result_path, "rb") as f:
        result = pickle.load(f)

    for p in (job_path, result_path):
        p.unlink(missing_ok=True)   # never let these reach the results ZIP

    if not result.get("ok"):
        print("[rootscope-web] CLASSIFICATION FAILED\n"
              + result.get("traceback", "(no traceback)"))
        raise gr.Error("Classification failed. See the Space logs.")
    return result["df"]


def _summarize(df, model_name):
    sub = df[df["model"] == model_name]
    counts = sub["predicted_cell_type"].value_counts()
    summary = pd.DataFrame({
        "cell type": counts.index,
        "cells": counts.values,
        "% of cells": (100 * counts.values / counts.values.sum()).round(1),
        "mean confidence": [
            round(float(sub.loc[sub["predicted_cell_type"] == ct,
                                "prediction_confidence"].mean()), 3)
            for ct in counts.index
        ],
    })
    return sub, summary


def _render(results, model_name):
    """Build the display for one model from an already-computed run."""
    workdir = Path(results["workdir"])
    stem = results["stem"]
    overlay_path = workdir / f"{stem}_{model_name}_overlay.png"
    csv_path = workdir / f"{stem}_{model_name}_predictions.csv"

    sub, summary = _summarize(results["df"], model_name)
    stats = (f"**{model_name}** · {len(sub)} cells · {results['n_layers']} "
             f"tissue layers · {results['um_per_px']} µm/px")
    if len(sub) and "n_xylem_poles" in sub.columns:
        row = sub.iloc[0]
        stats += (f" · xylem poles: {int(row['n_xylem_poles'])} · "
                  f"phloem poles: {int(row['n_phloem_poles'])}")

    return (str(overlay_path), summary, stats,
            gr.DownloadButton(value=str(csv_path), visible=True))


def run_rootscope(tif_path, tif_file, um_per_px, which_model, label_cells,
                  max_rounds, progress=gr.Progress()):
    # tif_path comes from the State that on_upload fills; tif_file is whatever
    # the File component is holding right now. Either alone is fragile: the
    # State is empty if Run is pressed before on_upload finishes, and it is
    # dropped on a page reload. Three runs died on "Upload a TIFF first" in the
    # Space logs with a file plainly selected, so take whichever we have.
    tif_path = tif_path or tif_file
    if not tif_path:
        raise gr.Error("Upload a TIFF first.")
    if not um_per_px or float(um_per_px) <= 0:
        raise gr.Error("Microns per pixel must be greater than 0.")

    progress(0.02, desc="Reading image")
    img_rgb = rs.load_image(tif_path)
    h, w = img_rgb.shape[:2]
    if h * w > MAX_PIXELS:
        raise gr.Error(
            f"Image is {w}×{h} ({h * w / 1e6:.0f} MP), over this demo's "
            f"{MAX_PIXELS / 1e6:.0f} MP limit. Crop it, or run RootScope locally."
        )

    # Not cleaned up: Gradio serves the downloads straight out of this
    # directory, so it has to outlive the request. Spaces are ephemeral, but on
    # a long-lived server add a reaper for old TMP_ROOT/run_* directories.
    workdir = Path(tempfile.mkdtemp(prefix="run_", dir=TMP_ROOT))
    stem = Path(tif_path).stem.replace(".aivia", "")

    # ---- GPU: segmentation ----
    progress(0.10, desc="Segmenting with Cellpose-SAM (GPU)")
    masks = _segment(img_rgb)
    n_cells = int(masks.max())
    if n_cells == 0:
        raise gr.Error("Cellpose-SAM found no cells in this image.")

    # ---- CPU: tissue layers + handcrafted features ----
    progress(0.35, desc=f"{n_cells} cells, layer index and features")
    masks, df_base, layer_lookup, adjacency, n_layers = rs.stage_features(
        masks, img_rgb, um_per_px=float(um_per_px))
    if df_base is None:
        raise gr.Error("Every detected object was filtered out as debris "
                       "outside the tissue body.")

    # ---- GPU: DINOv2 embeddings ----
    progress(0.50, desc="DINOv2 embeddings (GPU)")
    df_base = _embed(masks, img_rgb, df_base)

    # ---- CPU: iterative classification, post-processing, overlays ----
    progress(0.65, desc="Classifying cells (iterative ensemble)")
    df = _classify_subprocess(
        workdir,
        df_base=df_base, masks=masks, img_rgb=img_rgb,
        layer_lookup=layer_lookup, adjacency=adjacency,
        stem=stem, source_name=Path(tif_path).name,
        um_per_px=float(um_per_px), max_rounds=int(max_rounds),
        label_cells=bool(label_cells),
    )

    progress(0.95, desc="Packaging results")
    zip_path = workdir / f"{stem}_rootscope_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(workdir.iterdir()):
            if f.suffix in (".png", ".csv"):
                z.write(f, arcname=f.name)

    available = [m for m in MODEL_CHOICES
                 if (workdir / f"{stem}_{m}_overlay.png").exists()]
    chosen = which_model if which_model in available else available[0]

    results = {"workdir": str(workdir), "stem": stem, "df": df,
               "n_layers": n_layers, "um_per_px": float(um_per_px),
               "available": available}
    overlay_path, summary, stats, csv_button = _render(results, chosen)
    return (overlay_path, summary, stats, csv_button,
            gr.DownloadButton(value=str(zip_path), visible=True),
            gr.update(choices=available, value=chosen),
            results)


def switch_model(results, model_name):
    """Swap the displayed model without recomputing, since every model's overlay and
    CSV was already written to the run's temp dir."""
    if not results or model_name not in results.get("available", []):
        return gr.update(), gr.update(), gr.update(), gr.update()
    return _render(results, model_name)


LEGEND = """
<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:13px;margin-top:6px">
  <span><b style="color:#FF69B4">■</b> root cap</span>
  <span><b style="color:#0000FF">■</b> epidermis</span>
  <span><b style="color:#CCCC00">■</b> exodermis</span>
  <span><b style="color:#00C800">■</b> cortex</span>
  <span><b style="color:#FFA500">■</b> endodermis</span>
  <span><b style="color:#800080">■</b> pericycle</span>
  <span><b style="color:#FF0000">■</b> xylem</span>
  <span><b style="color:#888888">■</b> phloem</span>
  <span><b style="color:#00BFBF">■</b> stele</span>
</div>
"""

# Gradio 6 moved `theme` from the Blocks constructor to launch(); Gradio 5
# only accepts it on Blocks. Pass it wherever this version wants it.
_THEME = gr.themes.Soft()
_THEME_ON_LAUNCH = int(gr.__version__.split(".")[0]) >= 6
_blocks_kwargs = {} if _THEME_ON_LAUNCH else {"theme": _THEME}

TITLE = ("RootScope: Cross-species Root Cell-Type Classification from "
         "Confocal Microscopy Images")

with gr.Blocks(title=TITLE, **_blocks_kwargs) as demo:
    gr.Markdown(
        f"# {TITLE}\n"
        "RootScope uses Cellpose-SAM to segment every cell and describes each "
        "one with morpho-topological features and fine-tuned DINOv2 "
        "embeddings. Classification is refined through an iterative process in "
        "which predictions from neighboring cells are fed back as new "
        "features, updating every label until the classification converges. "
        "Each cell is then assigned to one of nine anatomical types.\n\n"
        "[github](https://github.com/ct-tranchau/Rootscope) · "
        "[model weights](https://huggingface.co/ct-tranchau/Rootscope) · "
        "[model card](https://github.com/ct-tranchau/Rootscope/blob/main/MODEL_CARD.md)"
    )

    tif_state = gr.State(None)    # the real TIFF; never the displayed preview
    results_state = gr.State(None)  # last completed run, for switching models

    with gr.Row():
        with gr.Column(scale=1):
            # NOT a gr.Image: an Image component re-encodes uploads to PNG,
            # which strips the OME metadata and the original bit depth. A File
            # component passes the actual TIFF through untouched.
            tif_upload = gr.File(label="Input TIFF",
                                 file_types=[".tif", ".tiff"],
                                 file_count="single", type="filepath")
            input_image = gr.Image(label="Preview", type="filepath",
                                   interactive=False, height=300)
            status = gr.Markdown("Upload a root-tip TIFF to begin.")

            um_per_px = gr.Number(
                label="Microns per pixel",
                value=1.0,
                info="Read from the file's metadata when present. This is not "
                     "optional. The 1.0 default distorts every size-derived "
                     "feature.",
            )
            model_pick = gr.Dropdown(
                MODEL_CHOICES, value="Ensemble", label="Model to display",
                info="All four are computed in one run; this only picks which "
                     "one you see. Ensemble is the published result.",
            )
            with gr.Accordion("Advanced", open=False):
                label_cells = gr.Checkbox(
                    label="Write cell-type names on the overlay", value=False,
                    info="Readable when zoomed into one region; on a dense "
                         "section the text overlaps and hides the image.",
                )
                max_rounds = gr.Slider(
                    1, 20, value=10, step=1, label="Max refinement rounds",
                    info="The classifier re-predicts using neighbours' types "
                         "until predictions stop changing.",
                )
            run_btn = gr.Button("Run RootScope", variant="primary")
            if SHOW_DIAGNOSTICS:
                with gr.Accordion("Diagnostics", open=False):
                    env_btn = gr.Button("Report environment (no GPU needed)")
                    diag_btn = gr.Button("Test the GPU step by step")
                    diag_out = gr.Textbox(label="Result", lines=7,
                                          show_copy_button=True)

        with gr.Column(scale=1):
            overlay = gr.Image(label="Predicted cell types", type="filepath",
                               height=420)
            gr.HTML(LEGEND)
            stats_md = gr.Markdown()
            summary = gr.Dataframe(label="Cells per type", interactive=False,
                                   wrap=True)
            with gr.Row():
                csv_btn = gr.DownloadButton("Download predictions (CSV)",
                                            visible=False)
                zip_btn = gr.DownloadButton("Download everything (ZIP)",
                                            visible=False)

    # On a Space the TIFFs sit next to app.py; in a git checkout they are one
    # level up in examples/.
    here = Path(__file__).parent
    examples_dir = next((d for d in (here / "examples", here.parent / "examples")
                         if (d / "Acorulea_RootTip_Maturation.tif").exists()), None)
    if examples_dir is not None:
        gr.Examples(
            examples=[
                [str(examples_dir / "Acorulea_RootTip_Maturation.tif")],
                [str(examples_dir / "Spennellii_RootTip_EarlyMaturation.tif")],
            ],
            fn=on_upload,
            inputs=[tif_upload],
            outputs=[input_image, um_per_px, status, tif_state],
            run_on_click=True,
            # Only wires up the preview; the real work still happens on Run.
            cache_examples=False,
            label="Example images (click one, then press Run)",
        )

    tif_upload.change(on_upload, tif_upload,
                      [input_image, um_per_px, status, tif_state])
    run_btn.click(
        run_rootscope,
        [tif_state, tif_upload, um_per_px, model_pick, label_cells, max_rounds],
        [overlay, summary, stats_md, csv_btn, zip_btn, model_pick, results_state],
    )
    model_pick.change(switch_model, [results_state, model_pick],
                      [overlay, summary, stats_md, csv_btn])
    if SHOW_DIAGNOSTICS:
        diag_btn.click(run_diagnostics, None, diag_out)
        env_btn.click(env_report, None, diag_out)

if __name__ == "__main__":
    launch_kwargs = {"theme": _THEME} if _THEME_ON_LAUNCH else {}
    demo.queue(max_size=20).launch(allowed_paths=[str(TMP_ROOT)], **launch_kwargs)

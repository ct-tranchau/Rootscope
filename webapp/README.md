---
title: 'RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images'
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Classify root cell types across species from confocal TIFFs
---

# RootScope web app

A Gradio front end for [RootScope](https://github.com/ct-tranchau/Rootscope):
upload a confocal root-tip cross-section TIFF, get back a labeled overlay and a
per-cell CSV.

Built to mirror the [Cellpose-SAM Space](https://huggingface.co/spaces/mouseland/cellpose),
which is what most users of this kind of tool have already seen.

## How the GPU budget works

RootScope's pipeline is only partly GPU-bound. The app splits it so ZeroGPU is
billed for the GPU work alone:

| Stage | Where | Function |
|---|---|---|
| Read TIFF, max-project z-stack | CPU | `rootscope.predict.load_image` |
| Cellpose-SAM segmentation | **GPU** | `stage_segment` |
| Tissue mask, layer index, debris filter, handcrafted features | CPU | `stage_features` |
| Fine-tuned DINOv2 per-cell embeddings | **GPU** | `stage_embed` |
| Iterative RF/XGB/LGBM + ensemble, post-processing, overlays | CPU | `stage_classify` |

The two GPU stages are wrapped at three duration tiers (60 / 180 / 300 s),
chosen by image size and cell count, so a small image doesn't reserve a long
slot against the user's daily quota. All models — the ~460 MB of classifiers,
the Cellpose-SAM weights, and the DINOv2 backbone — load **once at startup**,
not per request.

## Measured runtimes

Wall-clock per stage, **CPU only** (no GPU was available on the machine these
were taken on), for the two images in `examples/`:

| Stage | 700×700, 215 cells | 910×910, 448 cells |
|---|---|---|
| Read TIFF | 0.1 s | 0.0 s |
| Cellpose-SAM segmentation *(GPU)* | 95.4 s | 147.9 s |
| Layers + handcrafted features | 7.8 s | 23.8 s |
| DINOv2 embeddings *(GPU)* | 42.5 s | 80.4 s |
| Iterative classify ×3 + ensemble + overlays | 6.0 s | 19.1 s |
| **Total** | **2 min 32 s** | **4 min 31 s** |

Startup costs ~12 s once (classifiers 2.2 s, Cellpose-SAM 6.1 s, DINOv2 3.4 s).

The GPU-eligible stages are 84–91% of the CPU work. On an actual GPU they
should shrink dramatically while the CPU stages stay roughly fixed (~14 s and
~43 s respectively) — which is the point of the split: those seconds are not
billed against the GPU reservation. **The GPU figures have not been measured**;
re-time on the target hardware before trusting the tier thresholds.

## Where to host

| Option | Cost | Good for |
|---|---|---|
| **HF Space + ZeroGPU** | free, **but see eligibility below** | the public demo — recommended |
| HF Space + T4 Small | $0.40/hr (~$290/mo always-on) | no duration caps, heavy or private use |
| Google Colab notebook | free, no eligibility gate | users who need batches beyond the daily quota |
| VT ARC / lab GPU node | free (already yours) | unpublished data that must not leave the institution |
| Modal / Replicate / RunPod | per-second | if ZeroGPU quota proves too tight |

> **Gradio Spaces are no longer free on `cpu-basic`.** Only *static* Spaces
> are. Hosting this app on Hugging Face at any tier requires either ZeroGPU
> eligibility or a paid plan — verified by two HTTP 402s when attempting to
> create the Space on 2026-08-17.

### ZeroGPU hosting eligibility

Free ZeroGPU hosting (up to 2 Spaces) needs an account that is **at least 30
days old**, with a verified email, in good standing. Otherwise the create call
fails with:

```
402 You must be subscribed to PRO to host Spaces with ZeroGPU.
If you recently created your account, please wait 30 days or request a
community grant.
```

Three ways past it:

1. **Wait.** The `ct-tranchau` account was created 2026-08-02, so it becomes
   eligible around **2026-09-01**. Costs nothing.
2. **Request a community grant.** Hugging Face grants free GPU to open-source
   and academic projects, which is exactly what this is. Ask from the Space's
   Settings once it exists, or via the ZeroGPU explorers discussion board.
3. **PRO, $9/month.** Immediate, plus 10 ZeroGPU Spaces and 8× daily quota.

Note the CLI OAuth token also expires **2026-09-01**, so re-run
`huggingface-cli login` before deploying.

Recommended pairing, which is what the Cellpose authors do: a **ZeroGPU Space**
for the click-and-try demo, plus a **Colab notebook** for anyone who wants to
run more images than the daily quota allows.

Two things to decide before publishing:

- **Ownership.** A free personal account can host 2 ZeroGPU Spaces, but
  org-owned ZeroGPU needs a paid Team plan. A demo tied to one person's
  personal account is a bus-factor risk for a tool cited in a paper.
- **Data sensitivity.** Uploads go to Hugging Face infrastructure. If
  collaborators' unpublished images cannot leave VT, host on the lab's own GPU
  node instead — `app.py` runs there unchanged, since the `spaces` import is
  optional.

## Known issue: XGBoost is incompatible with ZeroGPU

**Unpickling the XGBoost classifier at module level makes every subsequent
ZeroGPU task abort** — including a bare `torch.zeros(2048, 2048, device="cuda")`.

Bisected on 2026-08-18 with a 25-line Space that did nothing but load one model
and allocate a CUDA tensor:

| Loaded at import | ZeroGPU task |
|---|---|
| nothing | passes |
| RandomForest (350 MB) | passes |
| LightGBM | passes |
| **XGBoost** | **`GPU task aborted`** |
| `import xgboost` without loading a model | passes |

Ruled out along the way: the account (the public `mouseland/cellpose` ZeroGPU
Space runs fine from it), quota, private-vs-public visibility, the torch build
(2.11.0+cu130, CUDA available), total memory (a 3 GiB contiguous array is
fine), and every dependency import.

It is *creating the Booster* that does it, not importing the library. The saved
model is already CPU-configured (`device='cpu'`, `tree_method='auto'`,
`grow_quantile_histmaker`), so this is XGBoost probing for CUDA devices as it
builds the booster, which collides with the CUDA emulation ZeroGPU runs outside
`@spaces.GPU`.

### Options

1. **Dedicated GPU hardware** (T4 small, $0.40/hr) — no ZeroGPU, so no
   snapshotting and no conflict. `app.py` needs no changes: `@spaces.GPU` is
   documented as a no-op off ZeroGPU. Set a sleep timer so it only bills while
   in use.
2. **Modal** (see below) — same, and free on the Starter tier.
3. **Isolate XGBoost in a subprocess** so the main process never loads it.
   Keeps ZeroGPU and the exact published ensemble, but needs real work in
   `stage_classify`.
4. **Drop XGBoost from the ensemble.** Do not do this quietly: the ensemble
   averages three models with RandomForest double-weighted on xylem/phloem, so
   removing one changes the published predictions.

## Deploy to Modal (no waiting period)

Modal has no account-age gate, so this is the way to get the real thing — a
permanent URL, upload, download — live today. `webapp/modal_app.py` serves the
exact same `app.py`, so the interface is identical to the Space.

```bash
pip install modal
modal setup                              # one-time browser login

modal deploy webapp/modal_app.py         # prints your permanent URL
modal run webapp/modal_app.py::warm_cache   # pre-download weights (once)
```

Use `modal serve` instead of `modal deploy` for a temporary URL with live
reload while you are iterating.

Free Starter plan includes **$30/month of compute credits**, billed per second
on a T4, scaling to zero when idle. At roughly a minute of GPU-container time
per image that is on the order of a thousand images a month at no cost. The
`rootscope-cache` Volume keeps the ~550 MB of weights between containers so
cold starts stay quick.

Pin `ROOTSCOPE_SPEC` in that file to a tag or commit before sharing the URL.

## Deploy to Hugging Face Spaces

The Space and the model weights are separate repos and can share a name — the
`spaces/` prefix keeps them apart:

| | |
|---|---|
| Model weights | `huggingface.co/ct-tranchau/Rootscope` |
| Space | `huggingface.co/spaces/ct-tranchau/Rootscope` |
| Direct app URL | `ct-tranchau-rootscope.hf.space` |

```bash
# 1. Create the Space (Gradio SDK, ZeroGPU hardware)
huggingface-cli repo create Rootscope --type space --space_sdk gradio
git clone https://huggingface.co/spaces/ct-tranchau/Rootscope hf-rootscope
cd hf-rootscope

# 2. Copy the app in
cp ../Rootscope/webapp/app.py            .
cp ../Rootscope/webapp/requirements.txt  .
cp ../Rootscope/webapp/README.md         .
mkdir -p examples && cp ../Rootscope/examples/*.tif examples/

# 3. Push
git add -A && git commit -m "RootScope Gradio app" && git push
```

Then in the Space's **Settings**, set hardware to **ZeroGPU**. A free personal
account in good standing (verified email, older than 30 days) can host up to 2
ZeroGPU Spaces; PRO raises that to 10. A paid GPU also works and has no
duration cap.

### What ZeroGPU gives your users

The GPU is allocated on demand for the duration of each `@spaces.GPU` call and
released immediately after — visitors do not get a GPU for the whole session,
only for the two stages that need one. Backing hardware is half an NVIDIA RTX
Pro 6000 Blackwell (48 GB) by default.

Daily GPU quota is per visitor, not per Space:

| Visitor | Daily GPU quota |
|---|---|
| Not signed in | 2 minutes |
| Free account | 5 minutes |
| PRO / Team | 40 minutes |
| Enterprise | 60 minutes |

Only the segmentation and embedding stages draw on that budget, which is the
whole reason for the split above. How many images that buys depends on GPU
speed, which is still unmeasured.

### Expected GPU seconds per image (estimate)

The Cellpose-SAM Space reserves only `duration=10` for images up to 1000 px,
which anchors what its segmentation costs on ZeroGPU. Our examples are
700–910 px, so:

| Stage | Measured on CPU | Expected on ZeroGPU |
|---|---|---|
| Cellpose-SAM segmentation | 95–148 s | ~10 s |
| DINOv2 embeddings (215–448 cells) | 43–80 s | ~2–5 s |
| **Billed GPU time per image** | — | **~15 s** |

At ~15 GPU-seconds per image that works out to roughly 8 images/day for a
signed-out visitor, ~20 for a free account, and ~160 for PRO. The CPU stages
(14–43 s measured) are free. **This is an estimate, not a measurement** — no
GPU was available to time it on.

### Memory

Peak RSS measured end-to-end on the 910×910 / 448-cell example: **3.3 GiB**.

| | |
|---|---|
| torch + cellpose + rootscope imports | 0.58 GiB |
| 3 classifiers + scalers | 0.80 GiB |
| Cellpose-SAM model | 1.80 GiB |
| DINOv2 backbone | ~0 (shares torch allocations) |
| Processing one image | 0.09 GiB |

Comfortably within a Space's limits, and flat per request — the models are
resident, so a second image adds only ~0.1 GiB.

### Version constraints

ZeroGPU supports **torch 2.8.0+** and Python 3.10.13 / 3.12.12, and works only
with the Gradio SDK. RootScope's own `requirements.txt` pins `torch==2.4.0`,
which ZeroGPU will not run — see the note in `webapp/requirements.txt`. The
Space therefore runs an untested torch/cellpose combination; validate the two
example images against a local run before announcing it.

### Verified against

Gradio **5.49.1** (what `sdk_version` pins above, and what the live Space
reports). The app also runs on Gradio 6 — it detects which major version it is
on and passes `theme` to `Blocks()` or to `launch()` accordingly. Raising
`sdk_version` forces a rebuild on an untested Gradio major, so re-run the
example images afterwards.

### Before you announce it

- **Pin the RootScope version** in `requirements.txt` — swap `@main` for a tag
  or commit SHA so a later push can't silently change what users get.
- **Pre-cache the weights.** On first boot the Space downloads ~460 MB from
  `ct-tranchau/Rootscope`, ~85 MB for the DINOv2 backbone via `torch.hub`, and
  the Cellpose-SAM checkpoint. Either accept a slow first boot or add a
  build-time prefetch.
- `torch.hub.load('facebookresearch/dinov2', ...)` reaches out to GitHub. If the
  Space has no outbound GitHub access, vendor the backbone instead.

## Running it locally

The same `app.py` runs without Hugging Face — the `spaces` import is optional
and the `@spaces.GPU` decorator degrades to a no-op:

```bash
conda activate rootscope
pip install gradio
python webapp/app.py
```

On a CPU-only machine expect 15–30 minutes per image, so use a GPU node.

## Deliberate differences from the Cellpose-SAM Space

- **No resizing.** That Space downscales to 1000 px by default. RootScope
  cannot: shrinking the image invalidates every size-derived feature unless
  `um_per_px` is scaled to match. Images are processed at native resolution,
  with a 40 MP cap.
- **Pixel size is a first-class input.** It is read from OME metadata
  (`PhysicalSizeX`) on upload and prefilled, because the 1.0 default is wrong
  for essentially every real image.
- **One image at a time**, not multi-file. A RootScope run is minutes rather
  than seconds; batches belong in the CLI (`rootscope --tif-dir`).

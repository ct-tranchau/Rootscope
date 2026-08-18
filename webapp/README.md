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

Gradio front end for [RootScope](https://github.com/ct-tranchau/Rootscope):
upload a confocal root-tip cross-section TIFF, get back a labeled overlay and a
per-cell CSV. Runs on a Hugging Face Space with ZeroGPU.

## Do not load the classifiers in the main process

Unpickling XGBoost builds a Booster that probes for CUDA devices. On ZeroGPU
that registers state the snapshot/restore cycle cannot reproduce, and every
`@spaces.GPU` task then aborts, including a bare `torch.zeros` on cuda. It is
enough for the model to be resident; it does not have to be used.

`classify_worker.py` therefore runs the whole classification stage in a
subprocess that loads the models itself and exits. Do not "simplify" this back
into `app.py`. Bisected 2026-08-18: RandomForest and LightGBM are fine on their
own, XGBoost is not.

Only segmentation and the DINOv2 embeddings are wrapped in `@spaces.GPU`, so
the CPU stages do not bill against the visitor's daily GPU quota.

## Measured on the Space

| Image | Cells | Wall clock |
|---|---|---|
| 700x700 | 214 | 22 s |
| 910x910 | 441 | 44 s |

Labels match a local CPU run on 210/214 cells; the rest sit on layer
boundaries, where CPU and GPU are not bit-identical.

## Deploy

```bash
bash webapp/deploy.sh            # private Space
bash webapp/deploy.sh --public
```

The Space vendors its own copy of `rootscope/` rather than installing from
GitHub, so `requirements.txt` here does not list it. Set hardware to ZeroGPU in
the Space settings.

For Modal instead (T4, no ZeroGPU, so no XGBoost conflict):

```bash
modal deploy webapp/modal_app.py
modal run webapp/modal_app.py::warm_cache
```

## Run it locally

```bash
conda activate rootscope
pip install gradio
python webapp/app.py
```

`@spaces.GPU` is a no-op off Hugging Face. Expect 15-30 minutes per image on
CPU, so use a GPU node.

Set `ROOTSCOPE_DIAGNOSTICS=1` to expose a panel that tests the GPU step by
step. It is off by default and is how the XGBoost conflict above was found.

## Notes

- Images are processed at native resolution, capped at 40 MP. Downscaling would
  invalidate every size-derived feature unless `um_per_px` scaled with it.
- Pixel size is read from OME metadata on upload. The 1.0 default is wrong for
  essentially every real image.
- One image at a time. For batches use the CLI: `rootscope --tif-dir`.

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

Web front end for [RootScope](https://github.com/ct-tranchau/Rootscope).
Upload a confocal root cross-section TIFF and get back a labeled overlay and a
per-cell CSV. Every cell is segmented with Cellpose-SAM and assigned to one of
nine cell types: root cap, epidermis, exodermis, cortex, endodermis,
pericycle, stele, xylem, phloem.

## How to use

1. Upload a raw confocal TIFF (not a mask). Pixel size is read from the OME
   metadata.
2. Optionally type the species name. It is not required; if given, it is
   checked against the species table shipped with the package.
3. Press Run. Segmentation and the DINOv2 embeddings run on the GPU; a
   700x700 image with about 200 cells takes about 20 to 45 seconds.
4. Download the overlay image and the CSV of per-cell predictions.

One image at a time. For batches use the command line: `rootscope --tif-dir`.

## Run it locally

```bash
conda activate rootscope
pip install gradio
python webapp/app.py
```

Model weights download from [ct-tranchau/Rootscope](https://huggingface.co/ct-tranchau/Rootscope)
on first use.

## Author

Tran Chau (tnchau@vt.edu)

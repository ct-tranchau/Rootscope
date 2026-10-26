---
license: mit
tags:
  - biology
  - plant-biology
  - microscopy
  - image-segmentation
  - cell-type-classification
  - cellpose
  - dinov2
  - lightgbm
library_name: rootscope
pipeline_tag: image-classification
---

# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

Trained weights for [RootScope](https://github.com/ct-tranchau/Rootscope).
Give RootScope a confocal cross-section TIFF of a root. It segments every cell
with Cellpose-SAM and labels each one as one of nine cell types: root cap,
epidermis, exodermis, cortex, endodermis, pericycle, stele, xylem, phloem.
The package downloads these weights on first use.

## Files

| File | Size | What it is |
|---|---|---|
| `v4/backbone.pt` | 346 MB | DINOv2 ViT-B/14, fine-tuned on cell crops (`meta.json` next to it records the architecture; keep them together) |
| `v4/lgbm_s42.joblib` | 170 MB | LightGBM, seed 42: one model per refinement round, scaler, feature names, class order |
| `v4/lgbm_s1.joblib`, `v4/lgbm_s7.joblib` | 170 MB each | the other two seeds; seed 42 alone works, all three reproduce the published result |
| `v4/lgbm_geom_s42.joblib`, `v4/lgbm_geom_s1.joblib`, `v4/lgbm_geom_s7.joblib` | 180 MB each | the geometry-only models (shape, position, neighbourhood; no intensity, no embeddings) used for the outer layers; trained on the same sections with the same script |

## Performance

| | accuracy | macro-F1 |
|---|---|---|
| round 1 (no neighbor context) | 0.845 | |
| final, full-feature model alone | 0.881 (95% CI 0.858 to 0.905) | 0.852 |
| final, with the geometry model on the outer layers | 0.886 (95% CI 0.864 to 0.909) | 0.858 |

Per class F1 (with the geometry model): root cap 97.8, epidermis 89.2,
exodermis 90.5, cortex 93.6, endodermis 89.8, pericycle 87.0, stele 84.2,
xylem 71.4, phloem 69.0. The geometry model changes only the outer layers: it
keeps exodermis recall (0.876) and raises its precision from 0.84 to 0.93,
and cuts the number of wrongly labelled outer-layer cells by 6 percent. The
gain is measured on the held-out sections; on the training folds the two
models score the same. Because the outer layers are then decided from shape,
position and neighbourhood, they should hold up better than the vascular
core when the stain or the microscope changes.

Before training, 80 of the 51,763 labelled cells (27 sections) whose painted
class lay in the wrong layer were moved to the layer their position gives.
These came from reading the annotation bitmaps: a semi-transparent orange
painted over a bright cell shifts in hue and was read as the neighbouring
class, which put a few endodermis labels into the outer layers and a few
exodermis labels against the stele. Without the correction the model
reproduced those labels, for example endodermis cells in the exodermis layer
of Solanum pennellii. The correction changes the held-out score by 0.1 point.
Stele, xylem and phloem are the hard classes; most of their confusion is
among themselves.

After classification, a smoothing step along the epidermis, exodermis and
cortex layers replaces a single cell whose neighbours on both sides along the
layer agree on another type. A tie between the two sides is left alone. It
removes the isolated wrong cells inside a layer; the numbers above include
it.

Some species have no exodermis (in the training data pea, Vigna mungo and
Echinochloa crus-galli), and no species has one in the meristem. The package
judges from the image whether the layer is present, and from the species
table if a species name is given; when absent, the class is removed. On the
126 annotated sections, leaving one species out at a time, 16 of the 20
sections without exodermis are called absent, 2 uncertain, 2 present; of the
106 with exodermis, 104 present, 1 uncertain, 1 absent.

Every run writes a report with the share of cells in each confidence tier and
the accuracy that tier reached on the held-out sections (above 0.99: 0.93;
0.9 to 0.99: 0.68; below 0.9: 0.49), the per-class precision and recall, the
exodermis decision and its reason, and warnings.

## Author

Tran Chau (tnchau@vt.edu)

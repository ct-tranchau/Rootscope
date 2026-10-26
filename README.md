# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

<img width="468" height="28" alt="image" src="https://github.com/user-attachments/assets/ad8bf92f-e877-4534-8d9a-aac8bd606b66" />


Give RootScope a confocal cross-section TIFF of a root; it segments every cell
(Cellpose-SAM) and labels each one as one of nine cell types. This is the
model described in the manuscript.

## The model

- **Classifier**: LightGBM, one model per refinement round, three training
  seeds averaged.
- **Features per cell**: 915 = 96 morpho-topological + 768-d fine-tuned
  DINOv2 ViT-B/14 embedding + 51 context columns rebuilt every round.
- **Neighbor context**: the four neighbors' predicted probabilities (36 soft
  columns) and 6 agreement flags.
- **Position context**: distance to each cell-type layer the model itself
  predicted in the previous round (9 columns).
- **Refinement**: 6 rounds (the setting the model was evaluated with; 10
  changes accuracy by < 0.1 pp).
- **Outer layers by geometry**: a second LightGBM, trained on shape, position
  and neighbourhood only (no intensity and no embedding columns), runs through
  the same rounds. On every cell the full model places in the outer layers
  (cortex, epidermis, exodermis, root cap) the two probability vectors are
  averaged, so those layers are decided mostly by geometry and do not depend
  on the stain. The vascular core keeps the image features.
- **After the last round**: a radial-order prior (s = 0.3, λ = 1.0; 2.0 when
  the geometry model is not used) fitted on training out-of-fold predictions.
- **Held-out accuracy**: 0.886, macro-F1 0.858 (0.881 / 0.852 for the
  full-feature model alone).
- **Training labels**: 80 cells in 27 sections whose painted class sat in the
  wrong layer (a colour-threshold artefact of reading the annotation bitmaps:
  semi-transparent orange over a bright cell reads as the neighbouring hue)
  were moved to the layer their position gives before training. The list is
  `label_changes.csv` next to the feature table.
- **Pixel size** is read from the TIFF's OME metadata; the run stops if it is
  missing. Every segmented cell is kept (no debris filter), as in the
  manuscript's evaluation.

Accuracy is on the clean 26-section held-out split (11,043 labelled cells,
17 species, all present in training). On species the model has never seen,
expect roughly 0.70. For the full-feature model alone, the gain from refinement is concentrated
in the thin, positionally-defined layers: F1 +8.5 (exodermis), +8.1
(endodermis), +4.2 (pericycle) from round 1 to the final prediction, against
+0.6 for xylem. Adding the geometry model on the outer layers raises exodermis
precision from 0.84 to 0.93 at the same recall.

## Install

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
conda env create -f environment.yml      # or environment-cpu.yml
conda activate rootscope
pip install .
```

Weights (~850 MB: the DINOv2 backbone and three LightGBM bundles) download
from the Hugging Face model repo on first use and are cached. To work offline,
put them in `models/` next to the package or pass `--model-dir`.

## Run

```bash
rootscope --tif my_image.tif --out results/
rootscope --tif-dir my_tifs/ --out results/          # batch
```

`results/` gets, per image, a per-cell CSV (`<stem>_LightGBM_predictions.csv`)
and a labeled overlay PNG. The CSV carries the label, its confidence, the nine
class probabilities, and the round-1 label so the effect of refinement is
visible per cell.

Next to the CSV, `<stem>_LightGBM_report.txt` says how far to trust the
result: the share of cells in each confidence tier with the accuracy that tier
reached on the held-out sections, an expected accuracy for the image, the
per-class precision and recall from the held-out sections, whether the root
has an exodermis and why, and warnings (species not in the training set, an
unusual share of low-confidence cells). The CSV carries the same tier per cell
in `reliability`. The classifier's probability alone is optimistic: on the
held-out sections, cells above 0.99 are right 93% of the time, cells between
0.9 and 0.99 only 68%.

Two rules act on the outer layers after classification. Both are automatic.

Exodermis. Some species never form this layer (in the training data pea,
Vigna mungo and Echinochloa crus-galli), and no species has it in the
meristem. The classifier tends to draw a partial layer anyway. The package
decides from the image (how many exodermis cells, how much of the
circumference they cover, how confident the classifier is on them; the rule
was fitted on all 126 annotated sections and checked leaving one species out
at a time) and, if you give a species name, from the species table shipped
with the package. When the layer is judged absent, the class is removed and
each cell takes its next most likely type. When the image is ambiguous the
labels are left alone and the report says so.

```bash
rootscope --tif my_image.tif --out results/ --species "Pisum sativum"
rootscope --tif my_image.tif --out results/ --no-exodermis     # your own call
rootscope --tif my_image.tif --out results/ --has-exodermis    # your own call
```

Layer smoothing. A single cell whose neighbours on both sides along the layer
agree on another type takes that type. It acts only on epidermis, exodermis
and cortex, removes the isolated wrong cells inside a layer, and leaves
accuracy on the held-out sections unchanged. `--no-layer-smoothing` turns it
off. The CSV keeps the label before both rules in `predicted_raw` and marks
changed cells in `layer_smoothed`.

From Python:

```python
from rootscope import predict_tif
df = predict_tif("my_image.tif", out_dir="results/")     # GPU if available
df = predict_tif("pea.tif", out_dir="results/", species="Pisum sativum")
```

## Pixel size

Three of the hand-crafted features are in micrometres. V4 reads
`PhysicalSizeX` from the OME metadata of the TIFF and refuses to guess. If your
file has none, pass `--um-per-px <value>`; the training data spans 0.17 to
1.01 µm/px, so images far outside that range are outside what the model has
seen.

## Speed

Cellpose-SAM and the DINOv2 embedding are the cost; classification is seconds.
On one GPU an image is 1-3 minutes; on CPU expect 15 minutes or more. The web
app loads both networks once per process, so a second image is faster than the
first.

## Cell types

`root_cap` · `epidermis` · `exodermis` · `cortex` · `endodermis` · `pericycle` ·
`stele` · `xylem` · `phloem`

## Layout

```
rootscope/
  api.py, cli.py         entry points
  predict.py             segment -> features -> embed -> classify stages
  classify_v4.py         the refinement loop, seed averaging, radial prior
  context.py             soft-neighbor + layer-distance features
                         (copied verbatim from the research code, never retyped)
  extract_features.py    Cellpose-SAM + 96 morpho-topological features
  cnn_embeddings.py      fine-tuned DINOv2 ViT-B/14 embeddings
  radial_order.py        radial-order prior
  weights.py             finds / downloads the weights
webapp/                  Gradio app for the Hugging Face Space
pipeline/                feature-extraction and training scripts (the released
                         model was trained in the research repo; its evaluation
                         grid is described in the manuscript)
```

## Citation

Manuscript in preparation. Contact tnchau@vt.edu.

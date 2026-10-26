# RootScope: Cross-species Root Cell-Type Classification from Confocal Microscopy Images

Give RootScope a confocal cross-section TIFF of a root; it segments and classifies every cell. 

## Install

```bash
git clone https://github.com/ct-tranchau/Rootscope.git
cd Rootscope
conda env create -f environment.yml      # or environment-cpu.yml
conda activate rootscope
pip install .
```

Weights download from the Hugging Face model repo on first use and are cached. To work offline, put them in `models/` next to the package or pass `--model-dir`.

## Run

```bash
rootscope --tif my_image.tif --out results/
rootscope --tif-dir my_tifs/ --out results/          # batch
```

`results/` gets, per image, a per-cell CSV (`<stem>_LightGBM_predictions.csv`),
a labeled overlay PNG, and a short report (`<stem>_LightGBM_report.txt`, also as
JSON) that says how far to trust the result: the share of cells in each
confidence tier, the exodermis decision, and per-class counts. The CSV carries
the label, its confidence, the nine class probabilities, and the round-1 label
so the effect of refinement is visible per cell.

Exodermis. Some species never form this layer. The package decides from the image and, if you give a species name, from the species table shipped with the package. When the layer is judged absent, the class is removed and each cell takes its next most likely type. --species, --no-exodermis, --has-exodermis (optional) 

```bash
rootscope --tif my_image.tif --out results/ --species "Pisum sativum"
rootscope --tif my_image.tif --out results/ --no-exodermis     # your own call
rootscope --tif my_image.tif --out results/ --has-exodermis    # your own call
```

From Python:

```python
from rootscope import predict_tif
df = predict_tif("my_image.tif", out_dir="results/")     # GPU if available
df = predict_tif("pea.tif", out_dir="results/", species="Pisum sativum")
```

## Speed

Cellpose-SAM and the DINOv2 embedding are the cost; classification is seconds.
On one GPU an image is 1-3 minutes; on a multi-core CPU a few minutes. The web
app loads both networks once per process, so a second image is faster than the
first.

## Contact

Tran Chau at tnchau@vt.edu.

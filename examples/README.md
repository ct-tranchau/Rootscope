# Example images

Two raw confocal root-tip cross-sections for trying RootScope out and for
checking that an install works. Both are Aivia-exported 2-channel `uint8` TIFFs,
the same format the model was trained on.

| File | Size | Species | Stage | µm/px |
|------|------|---------|-------|-------|
| `Acorulea_RootTip_Maturation.tif` | 2.3 MB | *Aquilegia coerulea* | Maturation | 0.6478 |
| `Spennellii_RootTip_EarlyMaturation.tif` | 1.2 MB | *Solanum pennellii* | EarlyMaturation | 0.4546 |

`Spennellii` is one of the species the model was trained on. `Acorulea` is
**not** — it is included as a check on how the model behaves on a species it has
never seen.

## Run them

```bash
rootscope --tif examples/Acorulea_RootTip_Maturation.tif \
          --um-per-px 0.6478 \
          --out results/
```

```bash
rootscope --tif examples/Spennellii_RootTip_EarlyMaturation.tif \
          --um-per-px 0.4546 \
          --out results/
```

Or both at once:

```bash
rootscope --tif-dir examples/ --out results/
```

> Passing `--um-per-px` matters. It is not read from the file, and the default
> of 1.0 distorts every size-derived feature. The values above come from each
> image's OME metadata (`PhysicalSizeX`). In batch mode a single `--um-per-px`
> applies to every image, so run these two separately if you want both correct.

## What you should get

`results/` gets a per-cell CSV and an overlay PNG for each of the three models
plus the ensemble. On a single GPU each image takes roughly 1.5–3 minutes; on
CPU, expect 15–30 minutes.

For reference, on an NVIDIA L40S:

| | Cells found | Mean confidence | Cell types predicted |
|---|---|---|---|
| Acorulea | 440 | 0.90 | 8 of 9 |
| Spennellii | 215 | 0.96 | 8 of 9 |

Both produce a complete anatomy — epidermis and exodermis around the outside,
cortex, then endodermis, pericycle and a vascular cylinder containing stele,
xylem and phloem. If your run predicts almost everything as `cortex` and finds
no xylem or phloem, something is wrong with the input (see below).

## A note on masks

These are **raw images** — pixel values are brightness. RootScope does its own
segmentation with Cellpose-SAM, so do not feed it a labelled mask (an image
where each cell is filled with an integer ID). Most of the 480 features are
intensity- or embedding-derived, so a mask produces confident-looking but
meaningless predictions rather than an error.

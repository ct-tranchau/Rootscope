# Example images

Two raw confocal root-tip cross-sections for trying RootScope out: `Acorulea_RootTip_Maturation.tif` and `Spennellii_RootTip_EarlyMaturation.tif` 

## Run them

```bash
rootscope --tif examples/Acorulea_RootTip_Maturation.tif \
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

## What a correct run produces

Reference output from the `Ensemble` model (the published result), measured on
CPU with the pixel scales above:

### `Acorulea_RootTip_Maturation.tif` (910×910, `--um-per-px 0.6478`)

**443 cells**

| cell type | cells |
|---|---|
| cortex | 157 |
| epidermis | 83 |
| stele | 73 |
| exodermis | 53 |
| xylem | 27 |
| phloem | 25 |
| pericycle | 14 |
| endodermis | 11 |

### `Spennellii_RootTip_EarlyMaturation.tif` (700×700, `--um-per-px 0.4546`)

**214 cells**

| cell type | cells |
|---|---|
| epidermis | 59 |
| cortex | 55 |
| stele | 38 |
| exodermis | 16 |
| pericycle | 16 |
| endodermis | 13 |
| phloem | 10 |
| xylem | 7 |

Treat these as a sanity check, not an exact target. Cellpose-SAM is not
bit-identical between CPU and GPU, so a run on different hardware can land a
cell or two either way and shift the counts slightly. What should match is the
overall picture: the right number of cells to within about 1%, concentric
layers in the right order (epidermis outside, stele in the middle), and no cell
type collapsing to zero or swallowing the section.

> Passing `--um-per-px` matters. It is not read from the file, and the default
> of 1.0 distorts every size-derived feature. The values above come from each
> image's OME metadata (`PhysicalSizeX`). In batch mode a single `--um-per-px`
> applies to every image, so run these two separately if you want both correct.

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

> Passing `--um-per-px` matters. It is not read from the file, and the default
> of 1.0 distorts every size-derived feature. The values above come from each
> image's OME metadata (`PhysicalSizeX`). In batch mode a single `--um-per-px`
> applies to every image, so run these two separately if you want both correct.

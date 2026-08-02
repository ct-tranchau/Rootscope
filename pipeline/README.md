# Full pipeline (reproducibility)

These numbered scripts reproduce the complete workflow used to build the
published Rootscope model. **Most users do not need this** — to predict on your
own images just use the `rootscope` command (see the top-level README).

Run the steps in order. Each script is a thin wrapper around a module in the
`rootscope` package, so install the package first (`pip install -e .`).

| Step | Script | What it does | Key output |
|------|--------|--------------|-----------|
| 10 | `10_extract_features.py` | Segment TIF/BMP pairs (Cellpose-SAM), compute layer index, extract features + read labels from BMP overlays | `feature_outputs_finetuned/all_cell_features.csv` |
| 20 | `20_finetune_dinov2.py` | Fine-tune the DINOv2 backbone on labeled cell crops | `dinov2_finetuned/backbone.pt` |
| 30 | `30_train_iterative_cnn.py` | Train the iterative RF/XGBoost/LightGBM ensemble (with DINOv2 embeddings) | `trained_model_iterative_cnn_finetuned/` |
| 40 | `40_predict_cell_types.py` | Predict cell types on new images (explicit, all-flags form) | `predictions_.../` CSVs + overlays |

### Inputs you need for training

- **TIF images** (`tif/`) — raw confocal root-tip cross-sections.
- **BMP overlays** (`bmp/`) — color-coded ground-truth cell-type masks, one per
  TIF. Only needed for steps 10 (to read labels) — **not** needed to predict.
- **Metadata CSV** (`metadata_with_tif_sizes3.csv`) — lists the TIF/BMP pairs,
  species, and stage.

These raw data are **not** included in this repository (size / licensing); see
the paper's data-availability statement.

### Notes

- Steps 10, 20, and 40 use a GPU (`--gpu`). CPU works but is slow.
- Step 20 must run before step 30, and step 30's model must be paired with the
  **same** `backbone.pt` from step 20 at prediction time.
- Original SLURM submission scripts (`*.sbatch`) from the research environment
  are not included; adapt the commands above to your scheduler.

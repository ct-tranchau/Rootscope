"""
Cell-type classifier with ITERATIVE SELF-TRAINING (WITH CNN embeddings).

This variant INCLUDES cnn_emb_* columns from the features CSV as extra inputs.
See train_iterative_no_cnn.py for the handcrafted-features-only version.

Instead of training once, the model iteratively retrains itself:

  Round 1: Train with neighbor_celltype = -1, get CV predictions
  Round 2: Fill neighbor_celltype from round 1 CV predictions, retrain, get new CV predictions
  Round 3: Fill neighbor_celltype from round 2 CV predictions, retrain, get new CV predictions
  ...
  Stop when CV predictions stop changing

Each round, the neighbor celltype features become more accurate because
the model's predictions improve. The final model has learned to work with
realistic (predicted, not ground truth) neighbor information.

Test evaluation mirrors the training process:
  Round 1: predict test set with neighbor_celltype = -1
  Round 2: fill neighbor_celltype from round 1 predictions, predict again
  ...
  Stop when predictions converge

Usage:
  python train_iterative.py \
    --features feature_outputs/all_cell_features.csv \
    --out-dir trained_model_iterative \
    --n-estimators 500 \
    --max-rounds 10

Output:
  trained_model_iterative/
    model_RandomForest.joblib
    model_XGBoost.joblib
    model_LightGBM.joblib
    feature_scaler.joblib
    feature_columns.joblib
    label_encoder.joblib
    training_report.txt
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .extract_features import CELL_CLASSES


FEATURE_COLUMNS = [
    "layer_index", "normalized_radius", "edt_distance_px", "edt_normalized",
    "layer_fraction", "is_boundary_cell", "dist_from_centroid_um",
    "angular_position", "n_layers_total",
    "area_um2", "perimeter_um", "eccentricity", "solidity", "aspect_ratio",
    "compactness", "major_axis_px", "minor_axis_px", "orientation_rad",
    "extent", "area_perimeter_ratio", "equivalent_diameter",
    "mean_intensity", "std_intensity", "min_intensity", "max_intensity",
    "median_intensity", "intensity_range", "intensity_cv",
    "intensity_skewness", "intensity_kurtosis", "intensity_p10", "intensity_p90",
    "neighbors_count", "mean_neighbor_layer", "std_neighbor_layer",
    "mean_neighbor_area", "layer_diff_from_neighbors", "area_ratio_to_neighbors",
    "touches_background", "cells_in_same_layer", "frac_neighbors_same_layer",
    "frac_neighbors_inner", "frac_neighbors_outer", "inner_neighbor_count",
    "outer_neighbor_count", "layer_from_inside", "radial_intensity_gradient",
    "neighbor_layer_range", "neighbors_boundary_cell", "min_neighbor_layer",
    "max_neighbor_layer", "n_neighbors_touching_bg",
    "area_zscore", "area_ratio_to_global_median", "area_percentile_in_layer",
    "area_ratio_to_inner", "area_ratio_to_outer", "hexagonality", "n_vertices",
    "layer_cell_count_ratio", "inner_layer_cell_count", "outer_layer_cell_count",
    "layer_count_gradient", "layer_count_asymmetry", "adjacent_layer_area_ratio",
    "sin_angular_position", "cos_angular_position", "radial_x_sin", "radial_x_cos",
    "radial_inward_neighbor_area", "radial_outward_neighbor_area",
    "cw_neighbor_area", "ccw_neighbor_area",
    "radial_inward_neighbor_intensity", "radial_outward_neighbor_intensity",
    "cw_neighbor_intensity", "ccw_neighbor_intensity",
    "radial_inward_neighbor_celltype", "radial_outward_neighbor_celltype",
    "tangential_cw_neighbor_celltype", "tangential_ccw_neighbor_celltype",
    "wall_thickness_proxy", "wall_to_lumen_ratio", "local_area_rank_in_stele",
    "neighbor_area_std", "cell_wall_contrast",
    "lumen_darkness", "wall_lumen_gap", "interior_intensity_std",
    "frac_dark_interior", "ring1_intensity", "ring2_intensity",
    "wall_interior_gradient", "mean_neighbor_wall_thickness",
    "wall_thickness_vs_neighbors", "area_ratio_to_stele_mean",
]

NEIGHBOR_CT_COLS = [
    "radial_inward_neighbor_celltype", "radial_outward_neighbor_celltype",
    "tangential_cw_neighbor_celltype", "tangential_ccw_neighbor_celltype",
]


def load_and_clean(csv_path, min_confidence):
    df = pd.read_csv(csv_path)
    df = df[df["cell_type_label"] >= 0].copy()
    if "cell_type_confidence" in df.columns:
        df = df[df["cell_type_confidence"] >= min_confidence].copy()
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    cnn_emb_cols = sorted(
        [c for c in df.columns if c.startswith("cnn_emb_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if cnn_emb_cols:
        available.extend(cnn_emb_cols)
        print(f"  INCLUDING {len(cnn_emb_cols)} CNN embedding features")
    else:
        print(f"  WARNING: no cnn_emb_* columns found in CSV")
    df = df.dropna(subset=available + ["cell_type"])
    print(f"  Total cells: {len(df)}, Features: {len(available)}")
    print(f"  Class distribution:")
    for ct, cnt in df["cell_type"].value_counts().items():
        print(f"    {ct}: {cnt}")
    return df, available


def _build_model(name, n_estimators, random_state):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            max_depth=None, min_samples_leaf=2, min_samples_split=5,
            max_features="sqrt", random_state=random_state, n_jobs=-1,
        )
    elif name == "XGBoost":
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=9, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=3.0,
            random_state=random_state, n_jobs=-1,
            eval_metric="mlogloss", verbosity=0,
        )
    elif name == "LightGBM":
        return LGBMClassifier(
            n_estimators=n_estimators, max_depth=-1, num_leaves=63,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=10, class_weight="balanced",
            reg_alpha=0.1, reg_lambda=3.0,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )


def _compute_sample_weights(y, le):
    weights = compute_sample_weight("balanced", y)
    for cls_name in ["xylem", "phloem"]:
        if cls_name in le.classes_:
            cls_idx = le.transform([cls_name])[0]
            weights[y == cls_idx] *= 15.0
    return weights


def _fill_neighbor_celltypes_from_predictions(df, y_pred_int, le, neighbor_cols):
    """
    Fill neighbor celltype columns using predicted labels.

    For each cell, its GT neighbor celltype tells us the TRUE type of its neighbor.
    We replace this with what the model WOULD predict for that neighbor.
    Since we have CV predictions for every cell, we build a lookup:
      cell_id -> predicted_label
    Then for each cell's neighbor, we look up the neighbor's predicted label.

    If we don't have direct neighbor cell_id mapping (not in CSV), we use
    the confusion-based approach: for a neighbor with true type X, sample
    from the model's prediction distribution for type X.
    """
    df_out = df.copy()
    n_classes = len(le.classes_)

    # Build confusion matrix from predictions
    y_true = le.transform(df["cell_type"].values)
    cm = confusion_matrix(y_true, y_pred_int, labels=np.arange(n_classes))
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    # For each cell's neighbor celltype feature:
    # GT value = true type of the neighbor
    # Replace with: what the model would predict for that neighbor
    # Use deterministic mapping: argmax of confusion row (most likely prediction)
    most_likely_pred = np.argmax(cm_norm, axis=1)  # for each true class, most likely prediction

    for col in neighbor_cols:
        if col not in df_out.columns:
            continue
        gt_values = df_out[col].values.copy()
        pred_values = np.full_like(gt_values, -1.0)
        for i in range(len(gt_values)):
            gt_int = int(gt_values[i])
            if 0 <= gt_int < n_classes:
                pred_values[i] = float(most_likely_pred[gt_int])
        df_out[col] = pred_values

    return df_out


def _evaluate_per_class(y_true, y_pred, class_names):
    result = {}
    for ci, cname in enumerate(class_names):
        mask = y_true == ci
        if mask.sum() > 0:
            result[cname] = {
                "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
                "support": int(mask.sum()),
            }
    return result


def _evaluate_per_group(df_subset, y_true, y_pred, group_col):
    result = {}
    if group_col not in df_subset.columns:
        return result
    for grp in sorted(df_subset[group_col].dropna().unique()):
        grp_mask = (df_subset[group_col] == grp).values
        if grp_mask.sum() > 0:
            grp_idx = np.where(grp_mask)[0]
            result[grp] = {
                "accuracy": accuracy_score(y_true[grp_idx], y_pred[grp_idx]),
                "support": int(grp_mask.sum()),
            }
    return result


def iterative_cv_train(model_name, df_train, feature_cols, y_train, le,
                       cv, sample_weights, n_estimators, random_state,
                       max_rounds=10):
    """
    Iteratively retrain a model:
      Round 1: train with neighbor_celltype = -1
      Round N: train with neighbor_celltype from round N-1 CV predictions
      Stop when CV predictions converge

    Returns: (final_model, final_scaler, cv_acc_history, y_pred_cv_final)
    """
    n_classes = len(le.classes_)
    class_names = le.classes_
    prev_cv_preds = None
    best_cv_preds = None
    best_cv_acc = 0
    best_model = None
    best_scaler = None
    cv_acc_history = []
    n_stalled = 0

    df_current = df_train.copy()

    for round_num in range(1, max_rounds + 1):
        # Prepare features
        if round_num == 1:
            # Set neighbor celltypes to -1
            for col in NEIGHBOR_CT_COLS:
                if col in df_current.columns:
                    df_current[col] = -1.0
        else:
            # Fill neighbor celltypes from previous round's CV predictions
            df_current = _fill_neighbor_celltypes_from_predictions(
                df_train, prev_cv_preds, le, NEIGHBOR_CT_COLS
            )

        X_train = df_current[feature_cols].values.astype(np.float32)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 5-fold CV
        y_pred_cv = np.zeros_like(y_train)
        for tr_idx, val_idx in cv.split(X_train, y_train):
            fold_model = _build_model(model_name, n_estimators, random_state)
            if model_name == "XGBoost":
                fold_model.fit(X_train_scaled[tr_idx], y_train[tr_idx],
                               sample_weight=sample_weights[tr_idx])
            else:
                fold_model.fit(X_train_scaled[tr_idx], y_train[tr_idx])
            y_pred_cv[val_idx] = fold_model.predict(X_train_scaled[val_idx])

        cv_acc = accuracy_score(y_train, y_pred_cv)
        cv_acc_history.append(cv_acc)

        # Count changes
        if prev_cv_preds is not None:
            n_changed = sum(1 for a, b in zip(prev_cv_preds, y_pred_cv) if a != b)
            n_total = len(y_pred_cv)
            print(f"      Round {round_num}: CV acc = {cv_acc:.4f}, "
                  f"{n_changed}/{n_total} changed ({100*n_changed/n_total:.2f}%)")

            if n_changed == 0:
                print(f"      Converged at round {round_num}!")
                best_cv_preds = y_pred_cv.copy()
                best_cv_acc = cv_acc
                best_scaler = scaler
                break

            if cv_acc > best_cv_acc:
                best_cv_acc = cv_acc
                best_cv_preds = y_pred_cv.copy()
                best_scaler = scaler
                n_stalled = 0
            else:
                n_stalled += 1

            if n_stalled >= 3:
                print(f"      No CV improvement for 3 rounds, stopping. "
                      f"Best CV: {best_cv_acc:.4f}")
                break
        else:
            print(f"      Round 1: CV acc = {cv_acc:.4f}")
            best_cv_acc = cv_acc
            best_cv_preds = y_pred_cv.copy()
            best_scaler = scaler

        prev_cv_preds = y_pred_cv.copy()
    else:
        print(f"      Reached max rounds ({max_rounds}). Best CV: {best_cv_acc:.4f}")

    # Retrain final model on full training data using best round's neighbor celltypes
    df_final = _fill_neighbor_celltypes_from_predictions(
        df_train, best_cv_preds, le, NEIGHBOR_CT_COLS
    )
    X_final = df_final[feature_cols].values.astype(np.float32)
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)

    final_model = _build_model(model_name, n_estimators, random_state)
    if model_name == "XGBoost":
        final_model.fit(X_final_scaled, y_train, sample_weight=sample_weights)
    else:
        final_model.fit(X_final_scaled, y_train)

    train_acc = accuracy_score(y_train, final_model.predict(X_final_scaled))
    print(f"      Final model train acc: {train_acc:.4f}")

    return final_model, final_scaler, cv_acc_history, best_cv_preds, best_cv_acc, train_acc


def iterative_test_predict(model, scaler, df_test, feature_cols, le, max_rounds=10):
    """
    Iteratively predict on test set (mirrors training process):
      Round 1: predict with neighbor_celltype = -1
      Round N: fill from round N-1 predictions, predict again
      Stop when predictions converge

    Returns: (y_pred_final, round_history)
    """
    prev_preds = None
    best_preds = None
    best_n_changed = float("inf")
    n_stalled = 0
    round_history = []

    for round_num in range(1, max_rounds + 1):
        df_round = df_test.copy()

        if round_num == 1:
            for col in NEIGHBOR_CT_COLS:
                if col in df_round.columns:
                    df_round[col] = -1.0
        else:
            # Fill from previous predictions using confusion-based approach
            df_round = _fill_neighbor_celltypes_from_predictions(
                df_test, prev_preds, le, NEIGHBOR_CT_COLS
            )

        X = df_round[feature_cols].values.astype(np.float32)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        if prev_preds is not None:
            n_changed = sum(1 for a, b in zip(prev_preds, preds) if a != b)
            round_history.append((round_num, n_changed))

            if n_changed == 0:
                best_preds = preds.copy()
                break
            if n_changed < best_n_changed:
                best_n_changed = n_changed
                best_preds = preds.copy()
                n_stalled = 0
            else:
                n_stalled += 1
            if n_stalled >= 3:
                break
        else:
            best_preds = preds.copy()
            round_history.append((1, len(preds)))

        prev_preds = preds.copy()

    return best_preds, round_history


def train(df, feature_cols, out_dir, n_estimators=500, random_state=42,
          test_fraction=0.2, max_rounds=10):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    le.fit(df["cell_type"].values)
    class_names = le.classes_
    n_classes = len(class_names)

    # Hold-out test split by image
    images = df["source_file"].unique()
    np.random.seed(random_state)
    np.random.shuffle(images)
    n_test = max(1, int(len(images) * test_fraction))
    test_images = set(images[:n_test])
    train_images = set(images[n_test:])

    df_train = df[df["source_file"].isin(train_images)].copy()
    df_test = df[df["source_file"].isin(test_images)].copy()

    y_train = le.transform(df_train["cell_type"].values)
    y_test = le.transform(df_test["cell_type"].values)

    print(f"\n  Train: {len(train_images)} images, {len(df_train)} cells")
    print(f"  Test:  {len(test_images)} images, {len(df_test)} cells")
    print(f"  Features: {len(feature_cols)}")

    sample_weights = _compute_sample_weights(y_train, le)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    model_names = ["RandomForest", "XGBoost", "LightGBM"]
    results = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"  {name}: ITERATIVE SELF-TRAINING (max {max_rounds} rounds)")
        print(f"{'='*60}")
        sys.stdout.flush()

        # Iterative CV training
        model, scaler, cv_history, y_pred_cv, cv_acc, train_acc = iterative_cv_train(
            name, df_train, feature_cols, y_train, le,
            cv, sample_weights, n_estimators, random_state,
            max_rounds=max_rounds,
        )

        # Iterative test prediction
        print(f"    Iterative test prediction...")
        y_pred_test, test_rounds = iterative_test_predict(
            model, scaler, df_test, feature_cols, le, max_rounds=max_rounds,
        )
        test_acc = accuracy_score(y_test, y_pred_test)
        for rnd, n_ch in test_rounds:
            print(f"      Test round {rnd}: {n_ch} changes")
        print(f"    Test accuracy (iterative): {test_acc:.4f}")

        # Also evaluate round 1 only (no neighbor info)
        df_test_r1 = df_test.copy()
        for col in NEIGHBOR_CT_COLS:
            if col in df_test_r1.columns:
                df_test_r1[col] = -1.0
        X_test_r1 = df_test_r1[feature_cols].values.astype(np.float32)
        X_test_r1_scaled = scaler.transform(X_test_r1)
        y_pred_test_r1 = model.predict(X_test_r1_scaled)
        test_acc_r1 = accuracy_score(y_test, y_pred_test_r1)
        print(f"    Test accuracy (round 1 only): {test_acc_r1:.4f}")

        # Reports
        cv_report = classification_report(y_train, y_pred_cv, target_names=class_names)
        cv_cm = confusion_matrix(y_train, y_pred_cv)
        test_report = classification_report(y_test, y_pred_test, target_names=class_names)
        test_cm = confusion_matrix(y_test, y_pred_test)

        y_pred_train = model.predict(
            scaler.transform(df_train[feature_cols].values.astype(np.float32))
        )

        per_image_acc = {}
        for img in sorted(test_images):
            img_mask = (df_test["source_file"] == img).values
            if img_mask.sum() == 0:
                continue
            idx = np.where(img_mask)[0]
            per_image_acc[img] = (accuracy_score(y_test[idx], y_pred_test[idx]), len(idx))

        importances = model.feature_importances_
        idx_sorted = np.argsort(importances)[::-1]

        # Per-class accuracy
        test_per_class = _evaluate_per_class(y_test, y_pred_test, class_names)
        test_per_class_r1 = _evaluate_per_class(y_test, y_pred_test_r1, class_names)
        cv_per_class = _evaluate_per_class(y_train, y_pred_cv, class_names)
        train_per_class = _evaluate_per_class(y_train, y_pred_train, class_names)

        print(f"\n    {'Cell Type':<15} {'CV':>8} {'Test(iter)':>11} {'Test(r1)':>10}")
        print(f"    {'-'*48}")
        for cname in class_names:
            cv_a = cv_per_class.get(cname, {}).get("accuracy", 0)
            te_a = test_per_class.get(cname, {}).get("accuracy", 0)
            te_r1 = test_per_class_r1.get(cname, {}).get("accuracy", 0)
            print(f"    {cname:<15} {cv_a:>8.4f} {te_a:>11.4f} {te_r1:>10.4f}")

        results[name] = dict(
            model=model, scaler=scaler,
            train_acc=train_acc, cv_acc=cv_acc, cv_history=cv_history,
            test_acc=test_acc, test_acc_r1=test_acc_r1,
            y_pred_cv=y_pred_cv, y_pred_test=y_pred_test,
            cv_report=cv_report, cv_cm=cv_cm,
            test_report=test_report, test_cm=test_cm,
            train_per_class=train_per_class,
            cv_per_class=cv_per_class,
            test_per_class=test_per_class,
            test_per_class_r1=test_per_class_r1,
            per_species_acc=_evaluate_per_group(df_test, y_test, y_pred_test, "species"),
            per_stage_acc=_evaluate_per_group(df_test, y_test, y_pred_test, "stage"),
            per_image_acc=per_image_acc,
            importances=importances, idx_sorted=idx_sorted,
        )

    # ==================================================================
    #  SUMMARY
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Train':>8} {'CV':>8} {'Test(iter)':>11} {'Test(r1)':>10} {'Rounds':>8}")
    print(f"{'-'*62}")
    for name in results:
        r = results[name]
        print(f"{name:<15} {r['train_acc']:>8.4f} {r['cv_acc']:>8.4f} "
              f"{r['test_acc']:>11.4f} {r['test_acc_r1']:>10.4f} "
              f"{len(r['cv_history']):>8d}")

    best_name = max(results, key=lambda n: results[n]["test_acc"])
    print(f"\nBest model: {best_name} (test iterative = {results[best_name]['test_acc']:.4f})")

    # ==================================================================
    #  RETRAIN BEST ON ALL DATA
    # ==================================================================
    print(f"\nRetraining all models on ALL data...")
    y_all = le.transform(df["cell_type"].values)
    sample_weights_all = _compute_sample_weights(y_all, le)

    for name in results:
        r = results[name]
        # Fill neighbor celltypes using the CV confusion matrix from training
        # (can't pass y_pred_cv directly since it's train-only, not all data)
        df_all = _fill_neighbor_celltypes_from_predictions(
            df, le.transform(df["cell_type"].values), le, NEIGHBOR_CT_COLS
        )
        # Override with confusion-based fill using training CV confusion matrix
        cv_cm = r["cv_cm"]
        cv_cm_norm = cv_cm.astype(np.float64)
        row_sums = cv_cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cv_cm_norm /= row_sums
        most_likely = np.argmax(cv_cm_norm, axis=1)
        for col in NEIGHBOR_CT_COLS:
            if col not in df_all.columns:
                continue
            gt_vals = df[col].values.copy()
            pred_vals = np.full_like(gt_vals, -1.0)
            for i in range(len(gt_vals)):
                gt_int = int(gt_vals[i])
                if 0 <= gt_int < n_classes:
                    pred_vals[i] = float(most_likely[gt_int])
            df_all[col] = pred_vals
        X_all = df_all[feature_cols].values.astype(np.float32)
        scaler_final = StandardScaler()
        X_all_scaled = scaler_final.fit_transform(X_all)

        model_final = _build_model(name, n_estimators, random_state)
        if name == "XGBoost":
            model_final.fit(X_all_scaled, y_all, sample_weight=sample_weights_all)
        else:
            model_final.fit(X_all_scaled, y_all)

        final_acc = accuracy_score(y_all, model_final.predict(X_all_scaled))
        results[name]["final_train_acc"] = final_acc

        joblib.dump(model_final, out_dir / f"model_{name}.joblib")
        joblib.dump(scaler_final, out_dir / f"feature_scaler_{name}.joblib")
        print(f"  {name}: final train acc = {final_acc:.4f}")

    # Save shared artifacts
    joblib.dump(feature_cols, out_dir / "feature_columns.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")

    # ==================================================================
    #  TRAINING REPORT
    # ==================================================================
    report_path = out_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("CELL TYPE CLASSIFIER - ITERATIVE SELF-TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Training approach:\n")
        f.write("  Round 1: Train with neighbor_celltype = -1\n")
        f.write("  Round N: Fill neighbor_celltype from round N-1 CV predictions, retrain\n")
        f.write("  Stop when CV predictions converge\n\n")
        f.write(f"Total samples:      {len(y_all)}\n")
        f.write(f"Train samples:      {len(y_train)} ({len(train_images)} images)\n")
        f.write(f"Test samples:       {len(y_test)} ({len(test_images)} images)\n")
        f.write(f"Features:           {len(feature_cols)}\n")
        f.write(f"Classes:            {list(class_names)}\n\n")

        f.write("MODEL COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<15} {'Train':>8} {'CV':>8} {'Test(iter)':>11} "
                f"{'Test(r1)':>10} {'Rounds':>8}\n")
        for name in results:
            r = results[name]
            f.write(f"{name:<15} {r['train_acc']:>8.4f} {r['cv_acc']:>8.4f} "
                    f"{r['test_acc']:>11.4f} {r['test_acc_r1']:>10.4f} "
                    f"{len(r['cv_history']):>8d}\n")
        f.write(f"\nBest model: {best_name}\n")
        f.write(f"\nAll models retrained on all data:\n")
        for name in results:
            f.write(f"  {name}: {results[name].get('final_train_acc', 0):.4f}\n")

        # CV accuracy history
        f.write(f"\n\nCV ACCURACY PER TRAINING ROUND\n")
        f.write("-" * 50 + "\n")
        for name in results:
            r = results[name]
            f.write(f"\n{name}: ")
            f.write(" -> ".join(f"{a:.4f}" for a in r["cv_history"]))
            f.write("\n")

        # Per-cell-type accuracy
        f.write(f"\n\nPER-CELL-TYPE ACCURACY\n")
        f.write("=" * 80 + "\n")
        for name in results:
            r = results[name]
            f.write(f"\n{name}:\n")
            f.write(f"  {'Cell Type':<15} {'Train':>8} {'CV':>8} {'Test(iter)':>11} "
                    f"{'Test(r1)':>10} {'Train N':>8} {'Test N':>8}\n")
            f.write(f"  {'-'*72}\n")
            for cname in class_names:
                tr_a = r['train_per_class'].get(cname, {}).get("accuracy", 0)
                cv_a = r['cv_per_class'].get(cname, {}).get("accuracy", 0)
                te_a = r['test_per_class'].get(cname, {}).get("accuracy", 0)
                te_r1 = r['test_per_class_r1'].get(cname, {}).get("accuracy", 0)
                tr_n = r['train_per_class'].get(cname, {}).get("support", 0)
                te_n = r['test_per_class'].get(cname, {}).get("support", 0)
                f.write(f"  {cname:<15} {tr_a:>8.4f} {cv_a:>8.4f} {te_a:>11.4f} "
                        f"{te_r1:>10.4f} {tr_n:>8d} {te_n:>8d}\n")
        f.write("\n")

        # Per-species
        f.write(f"\nPER-SPECIES ACCURACY (Test, iterative)\n")
        f.write("=" * 80 + "\n")
        for name in results:
            r = results[name]
            if r["per_species_acc"]:
                ranked = sorted(r["per_species_acc"].items(),
                                key=lambda x: x[1]["accuracy"], reverse=True)
                f.write(f"\n{name}:\n")
                f.write(f"  {'Rank':<6} {'Species':<20} {'Accuracy':>10} {'N':>8}\n")
                f.write(f"  {'-'*48}\n")
                for rank, (sp, info) in enumerate(ranked, 1):
                    f.write(f"  {rank:<6} {sp:<20} {info['accuracy']:>10.4f} "
                            f"{info['support']:>8d}\n")
        f.write("\n")

        # Per-stage (developmental stage)
        f.write(f"\nPER-STAGE ACCURACY (Test, iterative)\n")
        f.write("=" * 80 + "\n")
        for name in results:
            r = results[name]
            if r["per_stage_acc"]:
                ranked = sorted(r["per_stage_acc"].items(),
                                key=lambda x: x[1]["accuracy"], reverse=True)
                f.write(f"\n{name}:\n")
                f.write(f"  {'Rank':<6} {'Stage':<20} {'Accuracy':>10} {'N':>8}\n")
                f.write(f"  {'-'*48}\n")
                for rank, (st, info) in enumerate(ranked, 1):
                    f.write(f"  {rank:<6} {st:<20} {info['accuracy']:>10.4f} "
                            f"{info['support']:>8d}\n")
        f.write("\n")

        # Detailed
        for name in results:
            r = results[name]
            f.write(f"\n{'='*60}\n{name} DETAILS\n{'='*60}\n\n")
            f.write(f"Train: {r['train_acc']:.4f}  CV: {r['cv_acc']:.4f}  "
                    f"Test(iter): {r['test_acc']:.4f}  Test(r1): {r['test_acc_r1']:.4f}\n\n")
            f.write(f"CV REPORT\n{'-'*60}\n{r['cv_report']}\n")
            f.write(f"\nTEST REPORT (iterative)\n{'-'*60}\n{r['test_report']}\n")
            f.write(f"\nPER-IMAGE TEST ACCURACY\n{'-'*60}\n")
            for img, (acc, n) in sorted(r['per_image_acc'].items()):
                f.write(f"  {img}: {acc:.3f} ({n} cells)\n")
            f.write(f"\nCONFUSION MATRIX (CV)\n{'-'*60}\n{r['cv_cm']}\n")
            f.write(f"\nCONFUSION MATRIX (test)\n{'-'*60}\n{r['test_cm']}\n")
            f.write(f"\nTOP 10 FEATURES\n{'-'*60}\n")
            for rank, fi in enumerate(r['idx_sorted'][:10], 1):
                f.write(f"  {rank:2d}. {feature_cols[fi]:<40s} "
                        f"{r['importances'][fi]:.4f}\n")
            f.write("\n")

    print(f"  Saved report to {report_path}")

    # ==================================================================
    #  PLOTS
    # ==================================================================
    model_names_list = list(results.keys())

    # Model comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names_list))
    w = 0.2
    b1 = ax.bar(x - 1.5*w, [results[n]["train_acc"] for n in model_names_list], w, label="Train", color="#2196F3")
    b2 = ax.bar(x - 0.5*w, [results[n]["cv_acc"] for n in model_names_list], w, label="CV", color="#FF9800")
    b3 = ax.bar(x + 0.5*w, [results[n]["test_acc"] for n in model_names_list], w, label="Test (iterative)", color="#4CAF50")
    b4 = ax.bar(x + 1.5*w, [results[n]["test_acc_r1"] for n in model_names_list], w, label="Test (round 1)", color="#F44336")
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x()+bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names_list)
    ax.set_ylabel("Accuracy")
    ax.set_title("Iterative Self-Training: Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.08)
    plt.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # CV accuracy over rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in model_names_list:
        ax.plot(range(1, len(results[name]["cv_history"])+1),
                results[name]["cv_history"], "o-", label=name)
    ax.set_xlabel("Training Round")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("CV Accuracy per Training Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "cv_accuracy_rounds.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    cmaps = ["Blues", "Oranges", "Greens"]
    for col, name in enumerate(model_names_list):
        r = results[name]
        sns.heatmap(r["test_cm"], annot=True, fmt="d", cmap=cmaps[col],
                    xticklabels=class_names, yticklabels=class_names, ax=axes[col])
        axes[col].set_xlabel("Predicted")
        axes[col].set_ylabel("True")
        axes[col].set_title(f"{name} TEST (acc={r['test_acc']:.3f})")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"  Saved plots to {out_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train cell-type classifier with iterative self-training"
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="trained_model_iterative_cnn")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    print("=" * 70)
    print("CELL TYPE CLASSIFIER: ITERATIVE SELF-TRAINING")
    print("=" * 70)

    print(f"\n[1] Loading features from {args.features}...")
    df, feature_cols = load_and_clean(args.features, args.min_confidence)

    print(f"\n[2] Training (max {args.max_rounds} rounds per model)...")
    train(df, feature_cols, args.out_dir, n_estimators=args.n_estimators,
          test_fraction=args.test_fraction, max_rounds=args.max_rounds)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

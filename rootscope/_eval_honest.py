"""
BUOC 5 -- danh gia trung thuc, khop dung cach tool chay tren anh moi.

Ba khac biet so voi train_iterative_cnn.py:

  1. DINO backbone chi hoc 100 section train -> cnn_emb_* cua 26 section test sach.

  2. Hang xom cua tap TEST khong lay tu BMP. Vong 1 = -1, vong sau lay tu chinh
     PREDICTION, tra qua adjacency that (dung cell_id), y het
     predict_cell_types_iterative.py. Khong dung
     _fill_neighbor_celltypes_from_predictions() vi ham do doc GT tu CSV.

  3. Hinh hoc hang xom tinh tren TAT CA te bao segment duoc (df_geom), khong chi
     te bao co nhan -- vi tren anh moi khong he co khai niem "khong nhan" hay
     "do tin cay thap"; moi te bao deu co mat trong mo va deu duoc du doan.
     Chi tap con co nhan + du tin cay (df_model) moi dung de train va cham diem.

Dong trung (source_file, cell_id) bi bo. Nhan BMP cua 26 section chi dung de cham diem.
"""
import argparse, json, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent   # khong dung khi suy dien; giu cho tuong thich

# Copied verbatim from the research training module (train_iterative_cnn.py) so
# that inference does not import it: that module pulls in xgboost, matplotlib and
# seaborn at import time, none of which v4 inference uses. Only NEIGHBOR_CT_COLS
# is read on the inference path (in _blank); FEATURE_COLUMNS is kept for
# load_tables(), and the two training helpers are stubs that refuse to run.
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


def _build_model(name, n_est, seed):  # training-only; not shipped with the package
    raise RuntimeError("_build_model is a training helper and is not part of the "
                       "RootScope inference package.")


def _compute_sample_weights(*a, **k):  # training-only; not shipped with the package
    raise RuntimeError("_compute_sample_weights is a training helper and is not part "
                       "of the RootScope inference package.")
from .extract_features import (  # noqa: E402
    build_cell_adjacency, compute_neighbor_celltypes, CELL_CLASSES,
)

OUT = ROOT / "Figure_groundtruth_predicted"

_BASE_BUILD = _build_model          # giu ban goc truoc khi thay
HP = {}                             # hyperparameter de de len tren, tu --hp


def _build_model_hp(name, n_est, seed):
    """Nhu _build_model nhung (a) nhan them 'Ensemble' = soft-vote ca 3 model,
    (b) cho phep de hyperparameter tu --hp len tren mac dinh hardcode."""
    if name == "Ensemble":
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier(
            [(n, _build_model_hp(n, n_est, seed))
             for n in ("RandomForest", "XGBoost", "LightGBM")],
            voting="soft")
    m = _BASE_BUILD(name, n_est, seed)
    if HP:
        ok = {k: v for k, v in HP.items() if k in m.get_params()}
        if ok:
            m.set_params(**ok)
    return m


def dedup_best(df):
    """
    Bo dong trung (source_file, cell_id).

    11 section co nhieu hon 1 file BMP (139 BMP / 126 section). Hai ban nhan
    khong phai luc nao cung giong: do duoc 8.2% cap mau thuan. Nen giu ban co
    cell_type_confidence CAO NHAT thay vi lay bua dong dau tien.
    """
    n0 = len(df)
    if "cell_type_confidence" in df.columns:
        df = df.sort_values("cell_type_confidence", ascending=False,
                            na_position="last", kind="mergesort")
    df = df.drop_duplicates(["source_file", "cell_id"], keep="first")
    df = df.sort_values(["source_file", "cell_id"], kind="mergesort").reset_index(drop=True)
    if len(df) != n0:
        print(f"  bo {n0-len(df):,} dong trung (giu ban confidence cao nhat)")
    return df


def load_tables(csv_path, min_confidence):
    df = pd.read_csv(csv_path, low_memory=False)
    df = dedup_best(df)

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if MT_KEEP is not None:
        available = [c for c in available if c in MT_KEEP]
        print(f"  chi giu {len(available)} feature hinh thai")
    if USE_NB_AGG:
        for c in NB_AGG_COLS:
            df[c] = -1.0
        available += NB_AGG_COLS
        print(f"  THEM {len(NB_AGG_COLS)} feature thanh phan hang xom")
    cnn = sorted([c for c in df.columns if c.startswith("cnn_emb_")],
                 key=lambda x: int(x.split("_")[-1]))
    sec = sorted([c for c in df.columns if c.startswith("sec_")])
    if sec:
        available += sec
        print(f"  INCLUDING {len(sec)} feature muc section: {sec}")
    if cnn:
        available += cnn
        print(f"  INCLUDING {len(cnn)} CNN embedding features")
    else:
        print("  WARNING: khong thay cot cnn_emb_*")

    geom = df.dropna(subset=available).reset_index(drop=True)
    m = (geom["cell_type_label"] >= 0) & geom["cell_type"].notna()
    if "cell_type_confidence" in geom.columns:
        m &= geom["cell_type_confidence"] >= min_confidence
    model = geom[m].reset_index(drop=True)
    print(f"  segment duoc: {len(geom):,} te bao;  dung train/cham diem: {len(model):,}")
    return geom, model, available


_ADJ = {}


def _adjacency_for(source_file, mask_dir):
    if source_file not in _ADJ:
        p = Path(mask_dir) / f"{Path(source_file).stem}_masks.npy"
        if not p.exists():
            raise FileNotFoundError(f"thieu mask: {p}")
        if len(_ADJ) > 200:   # du chua ca 126 section, khoi tinh lai moi vong
            _ADJ.clear()
        _ADJ[source_file] = build_cell_adjacency(np.load(p))
    return _ADJ[source_file]


def fill_neighbors(df_geom, pred_names, mask_dir):
    """Dien 4 cot hang xom tu NHAN DU DOAN cua MOI te bao, qua adjacency that."""
    pred_names = np.asarray(pred_names)
    parts = []
    for sf, idx in df_geom.groupby("source_file").groups.items():
        sub = df_geom.loc[idx].copy()
        pos = df_geom.index.get_indexer(idx)
        labels = {int(c): CELL_CLASSES.get(p, -1)
                  for c, p in zip(sub["cell_id"].values, pred_names[pos])}
        adj = _adjacency_for(sf, mask_dir)
        part = compute_neighbor_celltypes(sub, adj, cell_type_labels=labels)
        if USE_NB_AGG:
            names_by_cid = {int(c): pn for c, pn in zip(sub["cell_id"].values,
                                                        pred_names[pos])}
            for k, v in _nb_aggregates(sub, adj, names_by_cid).items():
                part[k] = v
        parts.append(part)
    return pd.concat(parts).loc[df_geom.index]


def _sample_weights(y, le, wp_boost):
    """Nhu _compute_sample_weights goc, nhung he so cho xylem/phloem chinh duoc.

    "balanced" da lo phan MAT CAN BANG (nghich dao tan suat). Moi he so nhan
    them sau do la UU TIEN, khong phai chinh mat can bang. Ban goc hardcode
    x15 cho xylem va phloem -- day la de do lai xem no con dang gia khong."""
    w = compute_sample_weight("balanced", y)
    for c in ("xylem", "phloem"):
        if c in le.classes_:
            w[y == le.transform([c])[0]] *= wp_boost
    return w


VASCULAR = ("stele", "xylem", "phloem")
OUTER_TISSUE = ("epidermis", "exodermis", "root_cap")
NB_AGG_COLS = ["frac_nb_vascular", "n_nb_vascular", "frac_nb_cortex",
               "frac_nb_same", "frac_nb_outer_tissue"]
ALL_CLASSES = ["root_cap", "epidermis", "exodermis", "cortex", "endodermis",
               "pericycle", "stele", "xylem", "phloem"]
NB_PER_CLASS = [f"frac_nb_{c}" for c in ALL_CLASSES]
USE_NB_AGG = False
MT_KEEP = None


def _nb_aggregates(sub_df, adj, names_by_cid):
    """Thanh phan lop cua TAT CA hang xom quanh moi cell.

    4 cot NEIGHBOR_CT_COLS chi lay DUNG MOT hang xom gan nhat moi huong, nen
    cell pericycle nam o lop trong bi pericycle khac che se mat han tin hieu
    stele (do duoc: 393/843 cell roi vao truong hop nay). Cac cot nay tong hop
    tren MOI hang xom nen khong bi che.

    Tinh du 5 dai luong roi chi tra ve nhung cot dang duoc yeu cau.
    """
    allv = {c: [] for c in ("frac_nb_vascular", "n_nb_vascular", "frac_nb_cortex",
                            "frac_nb_same", "frac_nb_outer_tissue")}
    for cid in sub_df["cell_id"].values:
        nb = [names_by_cid.get(b) for b in adj.get(int(cid), ())]
        nb = [x for x in nb if x is not None and x != "__none__"]
        n = len(nb)
        me = names_by_cid.get(int(cid))
        if n == 0:
            for c in allv:
                allv[c].append(-1.0)
            continue
        nv = sum(x in VASCULAR for x in nb)
        allv["frac_nb_vascular"].append(nv / n)
        allv["n_nb_vascular"].append(float(nv))
        allv["frac_nb_cortex"].append(sum(x == "cortex" for x in nb) / n)
        allv["frac_nb_same"].append(sum(x == me for x in nb) / n if me else -1.0)
        allv["frac_nb_outer_tissue"].append(sum(x in OUTER_TISSUE for x in nb) / n)
    for c in ALL_CLASSES:
        allv[f"frac_nb_{c}"] = []
    for cid in sub_df["cell_id"].values:
        nb = [names_by_cid.get(b) for b in adj.get(int(cid), ())]
        nb = [x for x in nb if x is not None and x != "__none__"]
        n = len(nb)
        for c in ALL_CLASSES:
            allv[f"frac_nb_{c}"].append(sum(x == c for x in nb) / n if n else -1.0)
    return {c: allv[c] for c in NB_AGG_COLS}


def _blank(df):
    d = df.copy()
    for c in list(NEIGHBOR_CT_COLS) + list(NB_AGG_COLS):
        if c in d.columns:
            d[c] = -1.0
    return d


def _positions(df_geom, df_model):
    """Vi tri cua tung dong df_model trong df_geom."""
    g = pd.MultiIndex.from_frame(df_geom[["source_file", "cell_id"]])
    m = pd.MultiIndex.from_frame(df_model[["source_file", "cell_id"]])
    pos = g.get_indexer(m)
    assert (pos >= 0).all(), "co dong model khong tim thay trong geom"
    return pos


def iterative_fit(name, geom, model_df, pos, fcols, y, le, cv, sw,
                  mask_dir, n_est, seed, max_rounds, preset=False, groups=None):
    df_cur, prev, hist = (geom if preset else _blank(geom)), None, []
    mdl = scaler = None
    for rnd in range(1, max_rounds + 1):
        if rnd > 1:
            df_cur = fill_neighbors(geom, prev, mask_dir)
        Xg = df_cur[fcols].values.astype(np.float32)
        Xm = Xg[pos]
        scaler = StandardScaler().fit(Xm)
        Xms = scaler.transform(Xm)

        oof = np.zeros(len(y), dtype=int)
        for tr, va in cv.split(Xms, y, groups):
            f = _build_model(name, n_est, seed)
            f.fit(Xms[tr], y[tr], sample_weight=sw[tr])
            oof[va] = f.predict(Xms[va])
        acc = accuracy_score(y, oof)
        hist.append(round(float(acc), 4))

        mdl = _build_model(name, n_est, seed)
        mdl.fit(Xms, y, sample_weight=sw)

        # nhan cho MOI te bao geom: co nhan -> oof (ngoai mau); con lai -> model vong nay
        names = le.inverse_transform(mdl.predict(scaler.transform(Xg)))
        names[pos] = le.inverse_transform(oof)

        chg = -1 if prev is None else int((names != prev).sum())
        print(f"      Round {rnd}: CV acc = {acc:.4f}" +
              ("" if chg < 0 else f", {chg}/{len(names)} doi ({100*chg/len(names):.2f}%)"),
              flush=True)
        if prev is not None and chg == 0:
            print("      Hoi tu!", flush=True)
            break
        prev = names
    return mdl, scaler, hist, oof


def iterative_predict(mdl, scaler, geom, fcols, le, mask_dir, max_rounds, preset=False):
    df_cur, prev, hist = (geom if preset else _blank(geom)), None, []
    names = None
    for rnd in range(1, max_rounds + 1):
        if rnd > 1:
            df_cur = fill_neighbors(geom, prev, mask_dir)
        names = le.inverse_transform(
            mdl.predict(scaler.transform(df_cur[fcols].values.astype(np.float32))))
        chg = -1 if prev is None else int((names != prev).sum())
        hist.append(chg)
        print(f"      Test round {rnd}" + ("" if chg < 0 else f": {chg}/{len(names)} doi"),
              flush=True)
        if prev is not None and chg == 0:
            break
        prev = names
    return names, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--tag", default="honest")
    ap.add_argument("--model", default="LightGBM")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--drop-neighbors", action="store_true")
    ap.add_argument("--boost-outer", type=float, default=0.0,
                    help="Nhan trong so mau len N lan cho cac class VONG NGOAI "
                         "(epidermis, exodermis, endodermis, root_cap). Muc tieu: model "
                         "uu tien dung o may vong don lop thay vi toi da accuracy tong.")
    ap.add_argument("--boost-classes", default="epidermis,exodermis,endodermis,root_cap",
                    help="danh sach class duoc boost, ngan cach bang dau phay")
    ap.add_argument("--boost-spec", default="",
                    help="He so RIENG cho tung class, vd "
                         "'endodermis:20,exodermis:15,epidermis:10'. Neu dat thi "
                         "no de len tren --boost-outer/--boost-classes.")
    ap.add_argument("--nb-agg", action="store_true",
                    help="Them feature tong hop tren TOAN BO hang xom.")
    ap.add_argument("--nb-per-class", action="store_true",
                    help="Mot feature cho MOI class: ty le hang xom thuoc tung "
                         "loai trong 9 loai. Day du, khong gop nhom tuy tien.")
    ap.add_argument("--nb-lean", action="store_true",
                    help="Nhu --nb-agg nhung chi giu 2 cot thuc su moi.")
    ap.add_argument("--mt-list", default="",
                    help="File liet ke feature hinh thai duoc giu (1 ten/dong).")
    ap.add_argument("--hp", default="",
                    help="JSON hyperparameter de len tren mac dinh, vd "
                         "'{\"num_leaves\": 127, \"learning_rate\": 0.03}'")
    ap.add_argument("--wp-boost", type=float, default=15.0,
                    help="He so nhan trong so cho xylem va phloem. Ban goc "
                         "hardcode 15.0; dat 1.0 de bo han, chi giu 'balanced'.")
    ap.add_argument("--group-cv", action="store_true",
                    help="Chia CV theo SECTION (StratifiedGroupKFold) thay vi theo te bao. "
                         "Quan trong: nhan hang xom bom vao vong lap se nhieu dung bang "
                         "luc test (~0.81) thay vi qua sach (~0.96), nen train va test "
                         "khop phan phoi.")
    ap.add_argument("--neighbors-from-gt", action="store_true",
                    help="CO CHU Y RO RI: dien hang xom tu NHAN THAT (kieu pipeline cu). "
                         "Chi de do xem DINO dong gop bao nhieu phan ro ri, "
                         "KHONG duoc dung lam so cong bo.")
    a = ap.parse_args()
    globals()["USE_NB_AGG"] = a.nb_agg or a.nb_lean
    if a.nb_lean:
        globals()["NB_AGG_COLS"] = ["frac_nb_vascular", "frac_nb_cortex"]
    if a.nb_per_class:
        # 9 ty le theo class + frac_nb_same. Cai thu 10 KHONG suy ra duoc tu 9
        # cai kia: model khong nhan class cua chinh no lam dau vao, nen no
        # khong biet cot nao la "cua minh".
        globals()["NB_AGG_COLS"] = list(NB_PER_CLASS) + ["frac_nb_same"]
        globals()["USE_NB_AGG"] = True
    if a.mt_list:
        globals()["MT_KEEP"] = {l.strip() for l in open(a.mt_list) if l.strip()}
    if a.hp:
        globals()["HP"] = json.loads(a.hp)
        print(f"  hyperparameter de len tren: {HP}")
    globals()["_build_model"] = _build_model_hp

    geom, model_df, fcols = load_tables(a.features, a.min_confidence)
    if a.drop_neighbors:
        fcols = [c for c in fcols if c not in NEIGHBOR_CT_COLS]
        a.max_rounds = 1
        print(f"  bo 4 cot hang xom -> {len(fcols)} features (1 vong, khong lap)")

    tr_s = {l.strip() for l in open(ROOT / "split_train.txt") if l.strip()}
    te_s = {l.strip() for l in open(ROOT / "split_test.txt") if l.strip()}
    assert not (tr_s & te_s), "split train/test chong nhau!"

    g_tr = geom[geom.source_file.isin(tr_s)].reset_index(drop=True)
    g_te = geom[geom.source_file.isin(te_s)].reset_index(drop=True)
    m_tr = model_df[model_df.source_file.isin(tr_s)].reset_index(drop=True)
    m_te = model_df[model_df.source_file.isin(te_s)].reset_index(drop=True)
    p_tr, p_te = _positions(g_tr, m_tr), _positions(g_te, m_te)
    print(f"  train {m_tr.source_file.nunique()} section: {len(g_tr):,} te bao "
          f"({len(m_tr):,} co nhan)")
    print(f"  test  {m_te.source_file.nunique()} section: {len(g_te):,} te bao "
          f"({len(m_te):,} co nhan)", flush=True)

    le = LabelEncoder().fit(sorted(model_df.cell_type.unique()))
    y_tr = le.transform(m_tr.cell_type)
    if a.group_cv:
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=a.seed)
        groups = m_tr.source_file.values
        print(f"  CV chia theo SECTION: {m_tr.source_file.nunique()} section -> 5 fold")
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
        groups = None
        print("  CV chia theo TE BAO (mac dinh)")
    sw = _sample_weights(y_tr, le, a.wp_boost)
    print(f"  trong so: balanced, xylem/phloem x{a.wp_boost}")
    if a.boost_spec:
        idx = {c: i for i, c in enumerate(le.classes_)}
        sw = sw.copy()
        for part in a.boost_spec.split(","):
            cls, _, mult = part.partition(":")
            cls, mult = cls.strip(), float(mult)
            assert cls in idx, f"class khong co: {cls}"
            m = y_tr == idx[cls]
            sw[m] *= mult
            print(f"  boost x{mult:g} cho {cls} ({int(m.sum()):,} te bao)")
    elif a.boost_outer > 0:
        OUTER = [c.strip() for c in a.boost_classes.split(",") if c.strip()]
        idx = {c: i for i, c in enumerate(le.classes_)}
        m = np.isin(y_tr, [idx[c] for c in OUTER if c in idx])
        sw = sw.copy(); sw[m] *= a.boost_outer
        print(f"  boost x{a.boost_outer} cho {OUTER} ({int(m.sum()):,} te bao)")

    if a.neighbors_from_gt:
        print("\n  *** CHE DO RO RI CO CHU Y: hang xom lay tu NHAN THAT ***", flush=True)
        g_tr = fill_neighbors(g_tr, g_tr.cell_type.fillna("__none__").values, a.mask_dir)
        g_te = fill_neighbors(g_te, g_te.cell_type.fillna("__none__").values, a.mask_dir)
        a.max_rounds = 1

    print("\n  --- train (hang xom = prediction, qua adjacency that) ---", flush=True)
    mdl, scaler, hist, oof = iterative_fit(a.model, g_tr, m_tr, p_tr, fcols, y_tr, le,
                                      cv, sw, a.mask_dir, a.n_estimators, a.seed,
                                      a.max_rounds, preset=a.neighbors_from_gt,
                                      groups=groups)

    oof_df = m_tr[["source_file", "cell_id", "species", "stage", "cell_type"]].copy()
    oof_df["predicted"] = le.inverse_transform(oof)
    oof_df["correct"] = oof_df.cell_type.values == oof_df.predicted.values
    oof_df.to_csv(OUT / f"oof_{a.tag}.csv", index=False)
    print(f"  OOF train da ghi -> oof_{a.tag}.csv  (CV acc {oof_df.correct.mean():.4f})")

    print("\n  --- cham diem 26 section (KHONG dung BMP lam feature) ---", flush=True)
    names_all, thist = iterative_predict(mdl, scaler, g_te, fcols, le,
                                         a.mask_dir, a.max_rounds,
                                         preset=a.neighbors_from_gt)
    pred = names_all[p_te]
    acc = accuracy_score(m_te.cell_type.values, pred)

    print(f"\n  >>> {a.tag}: TEST = {acc:.4f}", flush=True)
    print(f"  >>> vong TRAIN den hoi tu: {len(hist)}  (CV moi vong: {hist})", flush=True)
    print(f"  >>> vong TEST  den hoi tu: {len(thist)} (so cell doi: {thist})", flush=True)

    joblib.dump(mdl, OUT / f"model_{a.tag}.joblib")
    joblib.dump(scaler, OUT / f"scaler_{a.tag}.joblib")
    joblib.dump(le, OUT / f"labelencoder_{a.tag}.joblib")
    o = m_te[["source_file", "cell_id", "species", "stage", "cell_type"]].copy()
    o["predicted"], o["correct"] = pred, m_te.cell_type.values == pred
    o.to_csv(OUT / f"test_predictions_{a.tag}.csv", index=False)
    (o.groupby("source_file").agg(n_cells=("correct", "size"), accuracy=("correct", "mean"))
       .round(4).to_csv(OUT / f"per_image_accuracy_{a.tag}.csv"))
    pc = o.groupby("cell_type")["correct"].agg(["size", "mean"]).round(4)
    print("\n  Accuracy tung loai:"); print(pc.to_string())
    pc.to_csv(OUT / f"per_class_{a.tag}.csv")


if __name__ == "__main__":
    main()

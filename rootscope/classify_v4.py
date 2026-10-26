"""RootScope v4 -- tang phan loai.

Khac v2 o ba diem, va ca ba deu nam o day chu khong o phan trich dac trung:

  1. Hang xom dang MEM: thay vi chot nhan cua hang xom roi dua vao model, moi huong
     dong gop ca vector xac suat 9 lop (36 cot) + 6 co dong thuan.
  2. Dac trung LOP: khoang cach tu moi te bao toi cac lop mo (epidermis, exodermis,
     endodermis, pericycle) ma chinh model da du doan o vong truoc, do theo ban kinh
     cuc bo va tinh bang duong kinh te bao.
  3. Radial-order prior: uu tien thu tu xuyen tam dung giai phau, ap sau vong cuoi.

Sau do hai buoc hau xu ly cho cac lop ngoai (layer_rules.py): lam min doc theo
lop (mot te bao ma hai ben theo vong deu cung mot lop khac thi lay lop do), va
quyet dinh section co exodermis hay khong (bang chung tu anh + bang loai +
co `no_exodermis` neu nguoi dung dat); neu khong co thi bo lop exodermis.

Cac ham tinh dac trung nam o context.py va duoc CAT NGUYEN VAN tu code nghien cuu.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import context as ctx
from . import _eval_honest as _eh
from .extract_features import build_cell_adjacency
from .radial_order import correct
from .layer_rules import (drop_exodermis, smooth_along_layer, exodermis_evidence,
                          decide_exodermis)
from .species import lookup as species_lookup

CLASSES = ["cortex", "endodermis", "epidermis", "exodermis", "pericycle",
           "phloem", "root_cap", "stele", "xylem"]
PRIOR = (0.3, 2.0)        # (s, lambda) cua model san xuat (full-feature model alone)
PRIOR_HYBRID = (0.3, 1.0) # prior at half strength once the geometry model is averaged in:
                          # tuned on the 100 out-of-fold training sections (neutral there),
                          # +0.48 pt on the 26 held-out sections (paired bootstrap P 0.006)
OUTER = ("cortex", "epidermis", "exodermis", "root_cap")


def load_models(model_dir, seeds=None, prefix="lgbm_s"):
    """Nap cac goi LightGBM v4. Moi goi chua round_models = model cua tung vong.
    prefix "lgbm_s" = the full-feature bundles; "lgbm_geom_s" = the geometry-only
    bundles (no intensity, no embedding columns) used for the outer layers."""
    model_dir = Path(model_dir)
    out = []
    for p in sorted(model_dir.glob(prefix + "*.joblib")):
        s = int(p.stem.split("_s")[-1])
        if seeds is None or s in seeds:
            b = joblib.load(p)
            if "round_models" not in b:
                raise ValueError(
                    f"{p.name} khong co 'round_models'. Goi nay chi luu model vong cuoi "
                    f"nen khong chay lai duoc vong lap. Dung goi san xuat.")
            out.append((s, b))
    if not out:
        raise FileNotFoundError(f"khong co {prefix}*.joblib trong {model_dir}")
    f0 = list(out[0][1]["fcols"])
    for s, b in out:
        if list(b["fcols"]) != f0:
            raise ValueError(f"seed {s} co bo feature khac")
        if list(b["classes"]) != CLASSES:
            raise ValueError(f"seed {s} co thu tu lop khac")
    return out


def _run_rounds(bundles, fcols, geom, mats, max_rounds, verbose, tag):
    """The per-round loop for one set of bundles. Returns (P_round1, P_final), each the
    mean over the bundles' seeds. Every bundle rebuilds its own neighbour / layer context
    from its own previous-round predictions, exactly as it was trained."""
    P1s, PFs = [], []
    for s, b in bundles:
        prev = prevP = None
        for rnd in range(1, max_rounds + 1):
            cur = ctx._blank(geom) if rnd == 1 else ctx.fill_neighbors(geom, prev, ctx.MASK)
            cur = pd.concat([cur.reset_index(drop=True),
                             ctx.soft_feats(prevP, mats, CLASSES),
                             ctx.ring_feats(geom, prev, "q", prevP)], axis=1)
            m_r, s_r = b["round_models"][min(rnd, len(b["round_models"])) - 1]
            P = m_r.predict_proba(s_r.transform(cur[fcols].values.astype(np.float32)))
            if rnd == 1:
                P1s.append(P.astype(np.float64))
            names = np.array(CLASSES)[P.argmax(1)]
            chg = -1 if prev is None else int((names != prev).sum())
            if verbose:
                print(f"    {tag}seed {s} round {rnd}: {chg} labels changed", flush=True)
            prev, prevP = names, P
            if chg == 0:
                break
        PFs.append(P.astype(np.float64))
    return np.mean(P1s, 0), np.mean(PFs, 0)


def classify(df_base, masks, source_file="image.tif", model_dir=None, seeds=None,
             max_rounds=6, prior=None, verbose=True, no_exodermis=None,
             smooth_layers=True, species=None, geometry=True):
    """Tra (nhan, xac suat, nhan vong 1, nhan truoc hau xu ly, info) cho tung
    dong cua df_base.

    species      : optional species name; looked up in data/species_exodermis.csv.
    no_exodermis : None = decide from the image and the species table (default);
                   True/False = the user's own call, overrides everything.
                   When the section is judged to have no exodermis, that class
                   is removed and every cell takes its next best class.
    smooth_layers: along-the-layer smoothing of epidermis/exodermis/cortex
                   (see layer_rules.smooth_along_layer).
    geometry     : also run the geometry-only bundles (lgbm_geom_s*.joblib) and
                   average their probabilities with the full model's on every
                   cell the full model places in the outer layers (cortex,
                   epidermis, exodermis, root cap). The outer layers are then
                   decided mostly by shape, position and neighbourhood, which
                   does not depend on the stain; the vascular core keeps the
                   image features. False = the v4.0 behaviour, bit for bit.
    prior        : (s, lambda) of the radial-order prior; None = PRIOR_HYBRID
                   when the geometry bundles run, PRIOR otherwise.
    info         : dict(status, reason, evidence, species_row, species_query).
    """
    bundles = load_models(model_dir, seeds)
    fcols = list(bundles[0][1]["fcols"])

    geom = df_base.copy()
    geom["source_file"] = source_file
    geom = geom.reset_index(drop=True)

    # Nap thang adjacency vao cache cua _eval_honest: khoi phai ghi mask ra dia
    # roi doc lai nhu duong danh gia trong du an.
    _eh._ADJ[source_file] = build_cell_adjacency(masks)
    ctx.MASK = "<in-memory>"
    ctx.CL = CLASSES
    # ring_feats caches the static surface features under the fixed key "q"
    # (the research code keyed train/test tables). One process may classify
    # several images in a row, so drop the previous image's entry here;
    # otherwise the second image fails with a shape mismatch.
    ctx._STATIC.clear()

    # The 4 nearest-neighbour cell-type columns are model inputs that exist only
    # as placeholders before round 1 (-1 = unknown); stage_features does not
    # produce them, so create them here. fill_neighbors overwrites them from round 2.
    for c in _eh.NEIGHBOR_CT_COLS:
        if c not in geom.columns:
            geom[c] = -1.0

    missing = [c for c in fcols if c not in geom.columns
               and c not in ctx.RINGCOLS and not c.startswith(("nbp_", "agree_"))]
    if missing:
        raise ValueError(f"thieu {len(missing)} cot dac trung, vd {missing[:6]}")

    mats = ctx.dir_mats(geom)
    P1, P = _run_rounds(bundles, fcols, geom, mats, max_rounds, verbose, "")

    # Geometry-only bundles for the outer layers (see the docstring).
    used_geometry = False
    if geometry:
        try:
            gbundles = load_models(model_dir, seeds, prefix="lgbm_geom_s")
        except FileNotFoundError:
            gbundles = []
        if gbundles:
            gfcols = list(gbundles[0][1]["fcols"])
            gmissing = [c for c in gfcols if c not in geom.columns
                        and c not in ctx.RINGCOLS and not c.startswith(("nbp_", "agree_"))]
            if gmissing:
                raise ValueError(f"geometry bundle: thieu {len(gmissing)} cot, vd {gmissing[:6]}")
            _, Pg = _run_rounds(gbundles, gfcols, geom, mats, max_rounds, verbose, "geom ")
            outer = np.isin(np.array(CLASSES)[P.argmax(1)], OUTER)
            P = P.copy()
            P[outer] = 0.5 * P[outer] + 0.5 * Pg[outer]
            used_geometry = True
            if verbose:
                print(f"    geometry model averaged in on {int(outer.sum())} outer-layer cells", flush=True)
        elif verbose:
            print("    (no lgbm_geom_s*.joblib in the model folder; outer layers use the full model only)", flush=True)
    if prior is None:
        prior = PRIOR_HYBRID if used_geometry else PRIOR

    cy = geom.centroid_y.values.astype(float)
    cx = geom.centroid_x.values.astype(float)
    cid = geom.cell_id.values
    if prior and prior[1]:
        look = {int(i): (y, x) for i, y, x in zip(cid, cy, cx)}
        P = correct(P, np.full(len(geom), source_file), cid, CLASSES,
                    geom={source_file: (cy, cx, look)}, s=prior[0], lam=prior[1])
    raw = np.array(CLASSES)[P.argmax(1)]

    def _smooth(n):
        if not smooth_layers:
            return n
        out, changed = smooth_along_layer(n, cy, cx, cid, _eh._ADJ[source_file])
        if verbose:
            print(f"    layer smoothing: {int(changed.sum())} labels changed", flush=True)
        return out

    names = _smooth(raw)
    # exodermis present in this section?  image evidence + species table + user flag
    species_row = species_lookup(species) if species else None
    evidence = exodermis_evidence(names, P, cy, cx, CLASSES)
    status, drop, reason = decide_exodermis(evidence, species_row, no_exodermis)
    if drop and (names == "exodermis").any():
        P = drop_exodermis(P, CLASSES)
        names = _smooth(np.array(CLASSES)[P.argmax(1)])
    if verbose:
        print(f"    exodermis: {status} ({reason})", flush=True)
    info = dict(status=status, reason=reason, evidence=evidence,
                species_row=species_row, species_query=species,
                geometry=used_geometry, prior=tuple(prior) if prior else None)
    return names, P, np.array(CLASSES)[P1.argmax(1)], raw, info

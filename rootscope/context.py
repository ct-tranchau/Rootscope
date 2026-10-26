"""Dac trung ngu canh cua RootScope v4:
  - hang xom bon huong, dang MEM (xac suat) thay vi nhan cung
  - khoang cach toi cac LOP mo ma chinh model da du doan o vong truoc

CAT NGUYEN VAN tu Working/_figgrid/eval_grid.py (2026-09-21). Khong go lai bang tay:
mot sai lech nho o day cho ket qua sai ma khong bao loi nao.
"""
import os
import numpy as np, pandas as pd, scipy.sparse as sp
from ._eval_honest import _adjacency_for, fill_neighbors, _blank   # noqa: F401

MASK = None        # thu muc chua <stem>_masks.npy; predict.py dat truoc khi goi
CL = None          # danh sach ten lop; predict.py dat truoc khi goi

DIRS = ["inner", "outer", "cw", "ccw"]


def dir_mats(g):
    """4 ma tran thua (n x n), chuan hoa theo hang: hang xom theo tung huong cuc."""
    r, c = {k: [] for k in DIRS}, {k: [] for k in DIRS}
    for sf, sub in g.groupby("source_file"):
        adj = _adjacency_for(sf, MASK)
        row = {int(i): p for i, p in zip(sub.cell_id.values, sub.index.values)}
        cy0, cx0 = sub.centroid_y.mean(), sub.centroid_x.mean()
        cen = {int(i): (y, x) for i, y, x in zip(sub.cell_id, sub.centroid_y, sub.centroid_x)}
        for cid, (cy, cx) in cen.items():
            md = np.hypot(cy - cy0, cx - cx0); ma = np.arctan2(cy - cy0, cx - cx0)
            for n in adj.get(cid, ()):
                if n not in cen:
                    continue
                ny, nx = cen[n]
                dr = np.hypot(ny - cy0, nx - cx0) - md
                da = (np.arctan2(ny - cy0, nx - cx0) - ma + np.pi) % (2 * np.pi) - np.pi
                arc = abs(da) * md if md > 1.0 else abs(da)
                k = ("inner" if dr < 0 else "outer") if abs(dr) >= arc else ("cw" if da < 0 else "ccw")
                r[k].append(row[cid]); c[k].append(row[n])
    n = len(g); mats = {}
    for k in DIRS:
        A = sp.csr_matrix((np.ones(len(r[k])), (r[k], c[k])), shape=(n, n))
        deg = np.asarray(A.sum(1)).ravel()
        mats[k] = (sp.diags(1 / np.maximum(deg, 1)) @ A, deg > 0)
    return mats


def soft_feats(P, mats, classes):
    n = len(mats["cw"][1]); cols = {}
    if P is None:
        names = [f"nbp_{k}_{c}" for k in DIRS for c in classes] + [f"agree_{k}" for k in DIRS] + ["agree_cw_ccw", "agree_in_out"]
        return pd.DataFrame(-1.0, index=range(n), columns=names)
    M = {}
    for k in DIRS:
        A, has = mats[k]; M[k] = A @ P
        for j, c in enumerate(classes):
            cols[f"nbp_{k}_{c}"] = np.where(has, M[k][:, j], -1.0)
    for k in DIRS:
        cols[f"agree_{k}"] = np.where(mats[k][1], (P * M[k]).sum(1), -1.0)
    cols["agree_cw_ccw"] = np.where(mats["cw"][1] & mats["ccw"][1], (M["cw"] * M["ccw"]).sum(1), -1.0)
    cols["agree_in_out"] = np.where(mats["inner"][1] & mats["outer"][1], (M["inner"] * M["outer"]).sum(1), -1.0)
    return pd.DataFrame(cols)



LAND = ["epidermis", "exodermis", "endodermis", "pericycle"]
RINGCOLS = ["depth_surf", "frac_r_surf"] + [f"d_{c}" for c in LAND] + ["d_vasc_own", "d_endo_own", "ratio_endo"]
RING = int(os.environ.get("RINGFEAT", "1"))
PERROUND = int(os.environ.get("PERROUND", "1"))   # inference uses the model trained FOR each round
SOFTRING = int(os.environ.get("SOFTRING", "0"))   # landmarks weighted by probability instead of argmax
DAMP = float(os.environ.get("DAMP", "0"))          # P_used = DAMP*P_prev + (1-DAMP)*P_new
_STATIC = {}
def _adiff(a, b): return np.abs((a[:, None] - b[None, :] + np.pi) % (2 * np.pi) - np.pi)
def ring_feats(g, names, key, Pm=None):
    """Where is each cell relative to landmarks of ITS OWN section?  Static: the tissue surface.  Dynamic (from the
    previous round's predictions; out-of-fold during training): the predicted epidermis / exodermis / endodermis /
    pericycle rings and the outer envelope of the vascular cylinder.  Distances are in cell diameters, measured along
    the local radius, so a non-circular root or an off-centre stele does not matter."""
    out = np.full((len(g), len(RINGCOLS)), np.nan, dtype=np.float32)
    if not RING: return pd.DataFrame(out, columns=RINGCOLS)
    if key not in _STATIC:
        st = np.full((len(g), 2), np.nan, dtype=np.float32)
        for sf, sub in g.groupby("source_file"):
            idx = sub.index.values; cy, cx = sub.centroid_y.mean(), sub.centroid_x.mean()
            r = np.hypot(sub.centroid_y - cy, sub.centroid_x - cx).values; th = np.arctan2(sub.centroid_y - cy, sub.centroid_x - cx).values
            near = _adiff(th, th) < np.deg2rad(15); rs = np.where(near, r[None, :], -1).max(1); dm = np.sqrt(np.median(sub.area_px))
            st[idx, 0] = (rs - r) / dm; st[idx, 1] = r / np.maximum(rs, 1)
        _STATIC[key] = st
    out[:, :2] = _STATIC[key]
    if names is None: return pd.DataFrame(out, columns=RINGCOLS)
    names = np.asarray(names)
    for sf, sub in g.groupby("source_file"):
        idx = sub.index.values; nm = names[idx]; own = np.sqrt(sub.area_px.values)
        core = np.isin(nm, ["endodermis", "pericycle", "stele", "xylem", "phloem"])
        cy, cx = (sub.centroid_y[core].mean(), sub.centroid_x[core].mean()) if core.sum() >= 5 else (sub.centroid_y.mean(), sub.centroid_x.mean())
        r = np.hypot(sub.centroid_y - cy, sub.centroid_x - cx).values; th = np.arctan2(sub.centroid_y - cy, sub.centroid_x - cx).values
        def ring(sel, how="median", win=40):
            m = _adiff(th, th[sel]) < np.deg2rad(win); A = np.where(m, r[sel][None, :], np.nan)
            v = np.nanmedian(A, 1) if how == "median" else np.nanmax(A, 1)
            fb = np.median(r[sel]) if how == "median" else np.percentile(r[sel], 90)
            return np.where(m.sum(1) >= 2, v, fb)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, c in enumerate(LAND):
                if SOFTRING and Pm is not None:
                    w = Pm[idx, CL.index(c)]; sel = w > 0.2
                    if w[sel].sum() < 4: continue
                    ww = w[sel] ** 2; mwin = (_adiff(th, th[sel]) < np.deg2rad(40)) * ww[None, :]
                    tot = mwin.sum(1); glob = np.average(r[sel], weights=ww)
                    rc = np.where(tot > ww.max() * 1.5, (mwin * r[sel][None, :]).sum(1) / np.maximum(tot, 1e-9), glob)
                    dm = np.sqrt(np.average(sub.area_px.values[sel], weights=ww))
                else:
                    sel = nm == c
                    if sel.sum() < 6: continue
                    rc = ring(sel); dm = np.sqrt(np.median(sub.area_px.values[sel]))
                out[idx, 2 + j] = (r - rc) / dm
                if c == "endodermis": out[idx, 7] = (r - rc) / own; out[idx, 8] = r / np.maximum(rc, 1)
            sel = np.isin(nm, ["stele", "xylem", "phloem"])
            if sel.sum() >= 5: out[idx, 6] = (r - ring(sel, "max", 30)) / own
    return pd.DataFrame(out, columns=RINGCOLS)


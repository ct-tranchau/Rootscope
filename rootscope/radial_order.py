"""Soft radial-order constraint  (fix for error family B: one-layer boundary shifts).

POST-HOC.  Takes the model's probability matrix and the cell centroids; it never looks at
the 768 DINOv2 columns, so it can be re-applied unchanged after the re-fine-tune.

Method
  1. Per section, a monotone radial coordinate r_rel in [0,1]:  r_rel = r / r_surface(theta),
     centre = mean centroid of the cells PREDICTED vascular-core, r_surface = furthest cell
     within +-15 deg of the cell's own angle (all cells of the section, labels not needed).
  2. Tissues get a radial RANK: epidermis/root_cap 0 < exodermis 1 < cortex 2 < endodermis 3
     < pericycle 4 < {stele,xylem,phloem} 5.  The trio shares ONE rank - there is no reliable
     radial order inside it, so the constraint cannot touch stele/xylem/phloem against each other.
  3. Per section, an ordered-logit model of rank against -r_rel is fitted to the model's OWN
     posterior (soft targets, no labels, nothing from the ground truth).  Two passes: the second
     fit uses the once-corrected posterior.
  4. P_final  ~  P_model * P_order ** lambda,  renormalised.

Hyperparameters (s = 0.3, lambda = 2.0, 2 passes) were chosen on the 100 out-of-fold training
sections only; the OOF optimum is a broad plateau (s 0.3-1.2, 2-4 passes, lambda 2).

Usage
  from apply_radial_order import correct
  P_final = correct(P, source_file, cell_id, classes)         # P: (n_cells, n_classes)
"""
import numpy as np
from scipy.optimize import minimize

CLASSES_DEFAULT = ['cortex','endodermis','epidermis','exodermis','pericycle','phloem','root_cap','stele','xylem']
RANK = {'epidermis':0,'root_cap':0,'exodermis':1,'cortex':2,'endodermis':3,'pericycle':4,
        'stele':5,'xylem':5,'phloem':5}
NRANK = 6
CORE = ('endodermis','pericycle','stele','xylem','phloem')
S_SCALE, LAMBDA, PASSES, FLOOR = 0.3, 2.0, 2, 1e-3


def radial_coordinate(cy, cx, cy_all, cx_all, pred_names, win_deg=15.0):
    """r_rel for the cells (cy,cx); the surface envelope uses every cell of the section
    (cy_all,cx_all), which may be a superset (unlabelled cells included)."""
    core = np.isin(pred_names, CORE)
    if core.sum() >= 5:
        y0, x0 = cy[core].mean(), cx[core].mean()
    else:
        y0, x0 = cy_all.mean(), cx_all.mean()
    r_all = np.hypot(cy_all - y0, cx_all - x0); th_all = np.arctan2(cy_all - y0, cx_all - x0)
    r = np.hypot(cy - y0, cx - x0); th = np.arctan2(cy - y0, cx - x0)
    da = np.abs((th[:, None] - th_all[None, :] + np.pi) % (2 * np.pi) - np.pi)
    rs = np.maximum(np.where(da < np.deg2rad(win_deg), r_all[None, :], -1.0).max(1), np.maximum(r, 1.0))
    return r / rs


def _nll(par, v, Q, s):
    t = np.concatenate([[par[0]], par[0] + np.cumsum(np.logaddexp(0.0, par[1:]))])
    z = np.clip((t[None, :] - v[:, None]) / s, -50, 50)
    cum = np.concatenate([np.zeros((len(v), 1)), 1 / (1 + np.exp(-z)), np.ones((len(v), 1))], 1)
    return -(Q * np.log(np.clip(np.diff(cum, axis=1), 1e-9, 1.0))).sum()


def _order_prior(v, Q, s):
    cw = np.clip(np.cumsum(Q.sum(0) + 1e-9)[:-1] / (Q.sum() + 1e-9), 0.01, 0.99)
    t0 = np.maximum.accumulate(np.quantile(v, cw))
    par0 = np.concatenate([[t0[0]], np.log(np.expm1(np.maximum(np.diff(t0), 1e-3)))])
    par = minimize(_nll, par0, args=(v, Q, s), method='L-BFGS-B', options=dict(maxiter=500)).x
    t = np.concatenate([[par[0]], par[0] + np.cumsum(np.logaddexp(0.0, par[1:]))])
    z = np.clip((t[None, :] - v[:, None]) / s, -50, 50)
    cum = np.concatenate([np.zeros((len(v), 1)), 1 / (1 + np.exp(-z)), np.ones((len(v), 1))], 1)
    return np.clip(np.diff(cum, axis=1), 1e-9, 1.0), t


def correct_section(P, cy, cx, classes, cy_all=None, cx_all=None,
                    s=S_SCALE, lam=LAMBDA, passes=PASSES):
    cy_all = cy if cy_all is None else cy_all; cx_all = cx if cx_all is None else cx_all
    pred = np.asarray(classes)[P.argmax(1)]
    v = -radial_coordinate(cy, cx, cy_all, cx_all, pred)
    Pc = P
    for k in range(passes):
        Q = np.zeros((len(P), NRANK))
        for j, c in enumerate(classes):
            Q[:, RANK[c]] += Pc[:, j]
        Q /= np.maximum(Q.sum(1, keepdims=True), 1e-12)
        Po, _ = _order_prior(v, Q, s)
        Po = Po + FLOOR; Po /= Po.sum(1, keepdims=True)
        pri = np.stack([Po[:, RANK[c]] for c in classes], 1)
        F = P * pri ** lam; Pc = F / F.sum(1, keepdims=True)
    return Pc


def correct(P, source_file, cell_id, classes=None, geom=None, **kw):
    """P (n,k) posterior; source_file (n,) section of each row.  geom: optional dict
    section -> (cy_all, cx_all, cy, cx) if the centroids are not in geom.npz."""
    classes = classes or CLASSES_DEFAULT
    if geom is None:
        raise ValueError(
            "correct() needs cell centroids. Pass geom={source_file: (cy_all, cx_all, "
            "{cell_id: (cy, cx)})}. The research build read these from a cached "
            "geom.npz that is not part of this package.")
        d = None  # unreachable; kept so the original block below still parses
        gsf = d['source_file'].astype(str); gid = d['cell_id'].astype(np.int64)
        geom = {}
        for s_ in np.unique(source_file):
            m = gsf == s_
            geom[s_] = (d['cy'][m], d['cx'][m], dict(zip(gid[m], zip(d['cy'][m], d['cx'][m]))))
    out = P.copy()
    for s_ in np.unique(source_file):
        m = source_file == s_
        cy_all, cx_all, look = geom[s_]
        yx = np.array([look[int(c)] for c in np.asarray(cell_id)[m]])
        out[m] = correct_section(P[m], yx[:, 0], yx[:, 1], classes, cy_all, cx_all, **kw)
    return out

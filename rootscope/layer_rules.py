"""Post-processing on the outer tissue layers (epidermis, exodermis, cortex).

Two rules, both applied after the classifier and the radial-order prior:

  1. ``drop_exodermis``: for species that have no exodermis the classifier still
     draws a partial ring, because it has learned "second layer from the surface
     is exodermis". With the flag set, the exodermis probability is removed and
     every cell takes its next most likely class. Applied BEFORE the radial prior
     so that the prior's ordinal fit never sees exodermis mass.

  2. ``smooth_along_layer``: a cell whose neighbours on both sides ALONG the layer
     (clockwise and counter-clockwise around the section centre) agree on
     another outer-layer class takes that class. This removes single cells of
     cortex sitting inside the exodermis layer and single exodermis cells inside
     the cortex, which are visually obvious errors. Measured on the 26 held-out
     sections of the manuscript: every single-cell mix removed, accuracy
     unchanged (0.8811 -> 0.8806). It never touches the vascular classes or the
     root cap, and it does not fill longer gaps in a layer -- those come from
     the classifier being confidently wrong, and no geometric rule fixes them
     without breaking more correct cells than it repairs.
"""
import numpy as np

CORE = ("endodermis", "pericycle", "stele", "xylem", "phloem")
OUTER = ("epidermis", "exodermis", "cortex")


def drop_exodermis(P, classes):
    """Zero the exodermis column and renormalise each row."""
    P = np.array(P, dtype=np.float64, copy=True)
    j = list(classes).index("exodermis")
    P[:, j] = 0.0
    s = P.sum(1, keepdims=True)
    # a row that was all exodermis has nothing left: fall back to uniform over
    # the other classes rather than dividing by zero
    empty = s[:, 0] <= 0
    if empty.any():
        P[empty] = 1.0
        P[empty, j] = 0.0
        s = P.sum(1, keepdims=True)
    return P / s


def section_centre(cy, cx, names):
    """Same rule as the radial prior: mean centroid of the cells predicted to be
    in the vascular core; all cells if fewer than five."""
    core = np.isin(names, CORE)
    if core.sum() >= 5:
        return float(cy[core].mean()), float(cx[core].mean())
    return float(cy.mean()), float(cx.mean())


def _tangential_sides(i, cy, cx, nbrs, c0, cos_max=0.5):
    """Indices of the neighbours of cell i that lie along the layer (displacement
    within 60 degrees of the tangent), split by side."""
    dy0, dx0 = cy[i] - c0[0], cx[i] - c0[1]
    r0 = np.hypot(dy0, dx0)
    if r0 == 0:
        return [], []
    uy, ux = dy0 / r0, dx0 / r0
    cw, ccw = [], []
    for k in nbrs:
        dy, dx = cy[k] - cy[i], cx[k] - cx[i]
        d = np.hypot(dy, dx)
        if d == 0:
            continue
        if abs((dy * uy + dx * ux) / d) < cos_max:
            (cw if uy * dx - ux * dy > 0 else ccw).append(k)
    return cw, ccw


def _majority(labels, min_frac):
    if not labels:
        return None
    vals, cnt = np.unique(labels, return_counts=True)
    k = cnt.argmax()
    if (cnt == cnt[k]).sum() > 1:
        # A tie (e.g. one cortex and one exodermis neighbour) is no evidence
        # for either class. np.unique orders alphabetically, so without this
        # guard "cortex" won every tie and correct exodermis cells were lost.
        return None
    return vals[k] if cnt[k] / len(labels) >= min_frac else None


def smooth_along_layer(names, cy, cx, cell_ids, adjacency, classes=OUTER,
                       passes=3, min_frac=0.5, centre=None):
    """Return (new_names, changed_mask). ``adjacency`` maps cell_id -> iterable of
    neighbouring cell_ids (as built by ``build_cell_adjacency``)."""
    names = np.asarray(names, dtype=object).copy()
    cy = np.asarray(cy, dtype=float)
    cx = np.asarray(cx, dtype=float)
    pos = {int(c): i for i, c in enumerate(cell_ids)}
    nbr = [[pos[int(b)] for b in adjacency.get(int(c), ()) if int(b) in pos]
           for c in cell_ids]
    c0 = section_centre(cy, cx, names) if centre is None else centre
    sides = [_tangential_sides(i, cy, cx, nbr[i], c0) for i in range(len(names))]
    changed = np.zeros(len(names), dtype=bool)
    classes = set(classes)
    for _ in range(passes):
        flips = []
        for i in range(len(names)):
            if names[i] not in classes:
                continue
            cw, ccw = sides[i]
            a = _majority([names[k] for k in cw], min_frac)
            b = _majority([names[k] for k in ccw], min_frac)
            if a is not None and a == b and a != names[i] and a in classes:
                flips.append((i, a))
        if not flips:
            break
        for i, lab in flips:          # simultaneous update within a pass
            names[i] = lab
            changed[i] = True
    return names, changed


# ---------------------------------------------------------------------------
# Does this section have an exodermis at all?
#
# Some species never form the layer (in the training data: Pisum sativum,
# Vigna mungo, Echinochloa crus-galli) and every species lacks it in the
# meristem. The classifier still tends to draw a partial layer. Three
# properties of the PREDICTED exodermis separate the two cases: how many
# exodermis cells there are relative to epidermis cells, how much of the
# circumference they cover, and how confident the classifier is on them.
# A logistic model on these three, fitted on all 126 annotated sections (20
# without exodermis) and evaluated leaving one species out at a time: AUC
# 0.95; with the thresholds below 16 of the 20 sections without exodermis are
# called absent, 2 uncertain, 2 present; of 106 sections with exodermis 104
# are called present, 1 uncertain, 1 absent.
_EXO_COEF = {"ratio_ex_ep": 1.5188, "coverage": 2.6856, "p_ex_mean": 1.9851}
_EXO_INTERCEPT = -3.4758
EXO_ABSENT_BELOW = 0.15
EXO_PRESENT_FROM = 0.60


def exodermis_evidence(names, P, cy, cx, classes):
    """Section-level evidence that an exodermis is present. Returns a dict with
    the three features and ``score`` in [0, 1] (1 = present)."""
    names = np.asarray(names, dtype=object)
    ex, ep = names == "exodermis", names == "epidermis"
    n_ex, n_ep = int(ex.sum()), int(ep.sum())
    ratio = n_ex / max(n_ep, 1)
    if n_ex:
        c0 = section_centre(np.asarray(cy, float), np.asarray(cx, float), names)
        th = np.arctan2(np.asarray(cy, float) - c0[0], np.asarray(cx, float) - c0[1])
        nb = max(8, min(36, n_ep // 2)) if n_ep else 12
        cov = len(set(((th[ex] + np.pi) / (2 * np.pi) * nb).astype(int) % nb)) / nb
        p_mean = float(P[ex, list(classes).index("exodermis")].mean())
    else:
        cov, p_mean = 0.0, 0.0
    z = (_EXO_INTERCEPT + _EXO_COEF["ratio_ex_ep"] * ratio
         + _EXO_COEF["coverage"] * cov + _EXO_COEF["p_ex_mean"] * p_mean)
    return dict(n_exodermis=n_ex, n_epidermis=n_ep, ratio_ex_ep=round(ratio, 3),
                coverage=round(cov, 3), p_ex_mean=round(p_mean, 3),
                score=round(float(1.0 / (1.0 + np.exp(-z))), 3))


def decide_exodermis(evidence, species_row=None, user_flag=None):
    """Combine the image evidence, the species table and an explicit user flag.

    Returns (status, drop, reason): status in {"present", "absent",
    "uncertain"}; drop=True means predicted exodermis is converted to each
    cell's next best class. Precedence: user flag > training annotations of
    the named species > image evidence. Literature-only rows and genus-level
    matches never change labels; they are reported."""
    if user_flag is True:
        return "absent", True, "set by the user"
    if user_flag is False:
        return "present", False, "set by the user"
    if (species_row and species_row.get("in_training") == "yes"
            and species_row.get("match") == "species"):
        if species_row["exodermis_in_training"] == "no":
            return ("absent", True,
                    f"{species_row['binomial']}: no exodermis in the training annotations")
        return ("present", False,
                f"{species_row['binomial']}: exodermis annotated in the training sections")
    s = evidence["score"]
    if s < EXO_ABSENT_BELOW:
        return "absent", True, (f"image evidence (score {s:.2f}): "
                                f"{evidence['n_exodermis']} exodermis cells predicted")
    if s >= EXO_PRESENT_FROM:
        return "present", False, f"image evidence (score {s:.2f})"
    return "uncertain", False, (
        f"image evidence is ambiguous (score {s:.2f}): the predicted exodermis covers "
        f"{100 * evidence['coverage']:.0f}% of the circumference with mean confidence "
        f"{evidence['p_ex_mean']:.2f}. If you know the species has no exodermis, "
        f"give the species name or set no_exodermis.")

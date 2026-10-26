"""Per-image report: how far to trust the predictions.

Numbers come from the 26 held-out sections of the manuscript (11,043
labelled cells, accuracy 0.886 with the v4.1 geometry hybrid; 0.881 for the
v4.0 full-feature model alone). The classifier's probability is optimistic,
so it is mapped to three reliability tiers with the accuracy actually
observed in each tier rather than reported as is.
"""
import json

# confidence tier -> (lower bound, observed accuracy on the held-out sections,
# share of cells in the tier)
TIERS = [("high", 0.99, 0.932, 0.77), ("medium", 0.90, 0.787, 0.13), ("low", 0.0, 0.653, 0.10)]

# per class on the held-out sections: precision, recall, F1
CLASS_PERF = {
    "root_cap":   (0.962, 0.995, 0.978), "epidermis": (0.834, 0.959, 0.892),
    "exodermis":  (0.930, 0.881, 0.905), "cortex":    (0.975, 0.900, 0.936),
    "endodermis": (0.903, 0.893, 0.898), "pericycle": (0.905, 0.837, 0.870),
    "stele":      (0.789, 0.904, 0.842), "xylem":     (0.823, 0.630, 0.714),
    "phloem":     (0.700, 0.681, 0.690),
}
# accuracy of a low-confidence (< 0.9) call, per predicted class
LOW_CONF_ACC = {"cortex": 0.89, "endodermis": 0.73, "epidermis": 0.44, "exodermis": 0.70,
                "pericycle": 0.65, "phloem": 0.42, "root_cap": 0.69, "stele": 0.55, "xylem": 0.43}


def reliability(conf):
    for name, lo, _, _ in TIERS:
        if conf >= lo:
            return name
    return "low"


def build(df, exo, species_row=None, species_query=None, n_smoothed=0, um_per_px=None):
    """Return the report as a dict; ``exo`` is dict(status, reason, evidence)."""
    n = len(df)
    tiers = df["reliability"].value_counts().to_dict()
    tier_rows = [dict(tier=t, cells=int(tiers.get(t, 0)),
                      share=round(tiers.get(t, 0) / n, 3) if n else 0.0,
                      accuracy_on_held_out=acc)
                 for t, _, acc, _ in TIERS]
    expected = (sum(tiers.get(t, 0) * acc for t, _, acc, _ in TIERS) / n) if n else None
    warnings = []
    if exo["status"] == "uncertain":
        warnings.append("Exodermis status is uncertain; see exodermis.reason.")
    if species_query and species_row is None:
        warnings.append(f"Species '{species_query}' is not in the species table; "
                        f"the exodermis decision used the image evidence only.")
    if species_row is not None and species_row.get("in_training") != "yes":
        warnings.append("Species is not in the training set; accuracy on it is unknown. "
                        "Held-out accuracy on unseen species was about 0.70 in the manuscript.")
    low_share = tiers.get("low", 0) / n if n else 0
    if low_share > 0.12:
        warnings.append(f"{100 * low_share:.0f}% of cells are low confidence (held-out share 10%); "
                        f"the image may be unlike the training data (staining, scale, section quality).")
    counts = df["predicted_cell_type"].value_counts()
    classes = [dict(cell_type=c, cells=int(counts[c]),
                    low_confidence=int(((df.predicted_cell_type == c) & (df.reliability == "low")).sum()),
                    held_out_precision=CLASS_PERF[c][0], held_out_recall=CLASS_PERF[c][1])
               for c in counts.index if c in CLASS_PERF]
    rep = dict(
        n_cells=n, um_per_px=um_per_px,
        expected_accuracy=round(expected, 3) if expected is not None else None,
        reliability=tier_rows, classes=classes,
        exodermis=dict(status=exo["status"], reason=exo["reason"], **exo["evidence"]),
        species=(None if species_row is None else
                 {k: species_row.get(k) for k in ("binomial", "family", "match", "in_training",
                                                  "exodermis_in_training", "exodermis_literature",
                                                  "note", "source")}),
        layer_smoothing_changed=int(n_smoothed),
        warnings=warnings,
        basis="26 held-out sections, 11,043 labelled cells, accuracy 0.886 (95% CI 0.864 to 0.909)",
    )
    return rep


def to_text(rep):
    L = [f"Cells: {rep['n_cells']}",
         f"Expected accuracy: {rep['expected_accuracy']} "
         f"(from the confidence tiers; held-out accuracy 0.886)",
         "Reliability tiers (share of cells -> accuracy observed on held-out sections):"]
    for t in rep["reliability"]:
        L.append(f"  {t['tier']:6s} {t['cells']:5d} cells ({100 * t['share']:.0f}%) -> {t['accuracy_on_held_out']:.2f}")
    e = rep["exodermis"]
    L.append(f"Exodermis: {e['status']} ({e['reason']})")
    if rep["species"]:
        s = rep["species"]
        lit = f"; literature: {s['exodermis_literature']}" if s.get("exodermis_literature") else ""
        L.append(f"Species: {s['binomial']} ({s['family']}, {s['match']} match; "
                 f"in training: {s['in_training']}{lit})")
        if s.get("note"):
            L.append(f"  {s['note']}")
    L.append(f"Cells changed by layer smoothing: {rep['layer_smoothing_changed']}")
    L.append("Per class (cells, low-confidence cells, held-out precision / recall):")
    for c in rep["classes"]:
        L.append(f"  {c['cell_type']:11s} {c['cells']:5d}  low {c['low_confidence']:4d}  "
                 f"{c['held_out_precision']:.2f} / {c['held_out_recall']:.2f}")
    for w in rep["warnings"]:
        L.append(f"WARNING: {w}")
    return "\n".join(L)


def write(rep, path_json):
    with open(path_json, "w") as f:
        json.dump(rep, f, indent=2)
    txt = str(path_json).rsplit(".", 1)[0] + ".txt"
    with open(txt, "w") as f:
        f.write(to_text(rep) + "\n")
    return txt

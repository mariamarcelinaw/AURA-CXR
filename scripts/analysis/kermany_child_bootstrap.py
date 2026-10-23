#!/usr/bin/env python3
"""Post-hoc check: child-level (cluster) bootstrap for the frozen Kermany evaluation.

The official Kermany test partition contains several images of some children. The file names
identify the child: PNEUMONIA/personN_<bacteria|virus>_K.jpeg (child "personN") and
NORMAL/IM-NNNN-KKKK.jpeg or NORMAL/NORMAL2-IM-NNNN-KKKK.jpeg (child "IM-NNNN" or
"NORMAL2-IM-NNNN"; the two prefixes are kept as distinct identifiers). Every child carries one
label. The image-level intervals of the paper (Section 4.1) resample images; this script resamples
children instead, stratified by class (2,000 resamples, seed 42), using the released per-case
Kermany predictions with the frozen models and thresholds unchanged (no retraining, no refitting):

  1. ROC-AUC with child-level 95% CIs for A1-A4, A7 and A8 (Table 10), and
  2. child-level 95% CIs for the paired ROC-AUC differences A8 - EVA-X-S and A8 - A4 (Section 5.6).

Usage (repository root):
    python scripts/analysis/kermany_child_bootstrap.py           # writes results/analysis/kermany_child_bootstrap/
    python scripts/analysis/kermany_child_bootstrap.py --check   # recomputes and compares with the released files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "results/predictions/kermany.csv"
OUT = ROOT / "results/analysis/kermany_child_bootstrap"
N_BOOT, SEED = 2000, 42
MODELS = {  # configuration -> probability (Table 10)
    "A1_DenseNet121-XRV": lambda d: d.xrv_probability,
    "A2_EVA-X-S": lambda d: d.eva_x_probability,
    "A3_LightGBM": lambda d: d.radiomics_probability,
    "A4_mean": lambda d: (d.xrv_probability + d.eva_x_probability) / 2,
    "A7_soft_voting": lambda d: d.soft_probability,
    "A8_frozen_stack": lambda d: d.a8_probability,
}
PAIRS = (("A8_frozen_stack", "A2_EVA-X-S"), ("A8_frozen_stack", "A4_mean"))
IMAGE_LEVEL_CI_A8 = (0.9103, 0.9498)  # reported image-level interval (Section 5.6; result_registry.csv)


def child_id(path: pd.Series) -> pd.Series:
    name = path.str.split("/").str[-1]
    child = name.str.extract(r"^(person\d+)_")[0]
    child = child.fillna(name.str.extract(r"^((?:NORMAL2-)?IM-\d+)-")[0])
    if child.isna().any():
        raise ValueError(f"unparsed Kermany file names: {path[child.isna()].tolist()[:5]}")
    return child


def resamples(y, child):
    """Class-stratified child-level bootstrap: children of each class are drawn with replacement."""
    by = pd.Series(np.arange(len(y))).groupby(child.to_numpy()).indices
    label = {k: int(y[v[0]]) for k, v in by.items()}
    assert all(len(set(y[v])) == 1 for v in by.values()), "a child carries two labels"
    pos = np.array(sorted(k for k in by if label[k] == 1))
    neg = np.array(sorted(k for k in by if label[k] == 0))
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(N_BOOT):
        keys = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(np.concatenate([by[k] for k in keys]))
    return out, len(pos), len(neg)


def compute():
    d = pd.read_csv(PRED)
    d = d.sort_values("relative_path").reset_index(drop=True)
    y = d.true_label.to_numpy(int)
    child = child_id(d.relative_path)
    idx, n_pos_children, n_neg_children = resamples(y, child)
    prob = {m: f(d).to_numpy(float) for m, f in MODELS.items()}
    boot = {m: np.array([roc_auc_score(y[i], p[i]) for i in idx]) for m, p in prob.items()}
    rows = [{"model": m, "n_images": len(y), "n_children": int(child.nunique()),
             "roc_auc": roc_auc_score(y, p),
             "roc_auc_child_ci_low": float(np.percentile(boot[m], 2.5)),
             "roc_auc_child_ci_high": float(np.percentile(boot[m], 97.5))} for m, p in prob.items()]
    pairs = []
    for a, b in PAIRS:
        diff = boot[a] - boot[b]
        pairs.append({"model_new": a, "model_ref": b,
                      "delta_roc_auc": roc_auc_score(y, prob[a]) - roc_auc_score(y, prob[b]),
                      "child_ci_low": float(np.percentile(diff, 2.5)),
                      "child_ci_high": float(np.percentile(diff, 97.5))})
    summary = {
        "cohort": "Kermany official test partition (frozen external evaluation)",
        "images": int(len(y)), "pneumonia_images": int(y.sum()), "normal_images": int((y == 0).sum()),
        "children": int(child.nunique()), "pneumonia_children": n_pos_children, "normal_children": n_neg_children,
        "max_images_per_child": int(child.value_counts().max()),
        "bootstrap": f"class-stratified child-level, {N_BOOT} resamples, seed {SEED}",
        "a8_image_level_ci_reported": list(IMAGE_LEVEL_CI_A8),
        "analysis_origin": "post_hoc; released per-case predictions; frozen models and thresholds; no retraining",
    }
    return pd.DataFrame(rows), pd.DataFrame(pairs), summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="recompute and compare with the released results")
    args = ap.parse_args()
    res, pairs, summary = compute()
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pairs.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    if args.check:
        ok = True
        for got, fname, keys in ((res, "kermany_child_bootstrap_metrics.csv", ["model"]),
                                 (pairs, "kermany_child_bootstrap_paired.csv", ["model_new", "model_ref"])):
            ref = pd.read_csv(OUT / fname)
            num = [c for c in ref.columns if c not in keys]
            ok = ok and len(got) == len(ref) and (got[keys].values == ref[keys].values).all() and np.allclose(
                got[num].to_numpy(float), ref[num].to_numpy(float), atol=1e-12)
        ok = ok and json.loads((OUT / "summary.json").read_text()) == summary
        print("KERMANY CHILD BOOTSTRAP =", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "kermany_child_bootstrap_metrics.csv", index=False)
    pairs.to_csv(OUT / "kermany_child_bootstrap_paired.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute the A1-A10 ablation on the locked RSNA test (paper Table 5) from released files.

Inputs (no images, no GPU):
  results/predictions/development_oof.csv       development OOF probabilities (thresholds, A9/A10 stacks)
  results/predictions/locked_rsna_a1_a10.csv    locked-test probabilities: the three frozen components, the
                                                post-lock five-fold ensembles of the four deep candidates
                                                (used only by the exploratory A9/A10 stacks), and A1-A10

The A1-A8 columns are the deployed outputs (as in locked_rsna.csv and test_predictions_full.csv).
With --check the script
  1. rebuilds every configuration from its members (A4-A6 means, A7 development-weighted soft vote,
     A8 frozen stack, A9/A10 Logistic Regression stacks fitted on the development OOF predictions with
     the A8 meta-learner settings) and checks that it matches the released column,
  2. re-derives every threshold from development OOF predictions with the rule of Section 3.4
     (A9/A10 from five-fold cross-fitted OOF stacks), and
  3. recomputes ROC-AUC, AUPRC, sensitivity, specificity, balanced accuracy, MCC and F1 and compares them
     with results/metrics/ablation_selected_dl_pair.csv (Table 5).

Usage (repository root):
    python scripts/analysis/ablation_a1_a10.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
sys.path.insert(0, str(ROOT / "src"))
from recompute_selection import crossfit, meta_learner, threshold_rule  # noqa: E402
from aura_cxr.fusion import a8_probability, weighted_soft_vote  # noqa: E402

PER_CASE = ROOT / "results/predictions/locked_rsna_a1_a10.csv"
REFERENCE = ROOT / "results/metrics/ablation_selected_dl_pair.csv"
CONFIGS = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10")
FOUR_DEEP = ("efficientnetv2", "resnet50", "xrv", "eva_x")


def development_stacks():
    """A9/A10 meta-learners fitted on all development OOF predictions, and every OOF-derived threshold."""
    d = pd.read_csv(ROOT / "results/predictions/development_oof.csv")
    y, fold = d.label.to_numpy(int), d.oof_fold.to_numpy(int)
    x9 = d[[f"p_{m}" for m in FOUR_DEEP]].to_numpy(float)
    x10 = np.column_stack([x9, d.p_radiomics_lgbm.to_numpy(float)])
    x, e, r = d.p_xrv.to_numpy(float), d.p_eva_x.to_numpy(float), d.p_radiomics_lgbm.to_numpy(float)
    thr = {"a1": threshold_rule(y, x), "a2": threshold_rule(y, e), "a3": threshold_rule(y, r),
           "a4": threshold_rule(y, (x + e) / 2), "a5": threshold_rule(y, (x + r) / 2),
           "a6": threshold_rule(y, (e + r) / 2), "a7": threshold_rule(y, d.p_a7_soft_oof.to_numpy(float)),
           "a8": threshold_rule(y, d.p_a8_crossfit_oof.to_numpy(float)),
           "a9": threshold_rule(y, crossfit(x9, y, fold)), "a10": threshold_rule(y, crossfit(x10, y, fold))}
    return meta_learner().fit(x9, y), meta_learner().fit(x10, y), thr


def rebuild(d, m9, m10):
    x, e, r = (d[f"{m}_probability"].to_numpy(float) for m in ("xrv", "eva_x", "radiomics"))
    cv = [d[f"{m}_cv_probability"].to_numpy(float) for m in FOUR_DEEP]
    return {"a1": x, "a2": e, "a3": r, "a4": (x + e) / 2, "a5": (x + r) / 2, "a6": (e + r) / 2,
            "a7": weighted_soft_vote(x, e, r), "a8": a8_probability(x, e, r),
            "a9": m9.predict_proba(np.column_stack(cv))[:, 1],
            "a10": m10.predict_proba(np.column_stack(cv + [r]))[:, 1]}


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    se, sp = tp / (tp + fn), tn / (tn + fp)
    return {"AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p), "sensitivity": se,
            "specificity": sp, "accuracy": (tp + tn) / len(y), "balanced_acc": (se + sp) / 2,
            "MCC": matthews_corrcoef(y, pred), "F1_pneumonia": 2 * tp / (2 * tp + fp + fn)}


def run():
    d = pd.read_csv(PER_CASE)
    lr = pd.read_csv(ROOT / "results/predictions/locked_rsna.csv")
    assert len(d) == 5338 and int(d.true_label.sum()) == 1203
    assert (d.case_id.values == lr.case_id.values).all() and (d.true_label.values == lr.true_label.values).all()
    m9, m10, thr = development_stacks()
    built = rebuild(d, m9, m10)
    y = d.true_label.to_numpy(int)
    rows, max_diff = [], {}
    for k in CONFIGS:
        p = d[f"{k}_probability"].to_numpy(float)
        max_diff[k] = float(np.abs(built[k] - p).max())
        rows.append({"config": k.upper(), "threshold": thr[k], **metrics(y, p, thr[k])})
    return pd.DataFrame(rows), max_diff


def check(res, max_diff):
    ref = pd.read_csv(REFERENCE)
    ref["config"] = ref.scenario.str.split("_").str[0]
    ref = ref.set_index("config")
    bad = []
    for _, r in res.iterrows():
        for col in ("threshold", "AUC", "AUPRC", "sensitivity", "specificity", "balanced_acc", "MCC", "F1_pneumonia"):
            if abs(r[col] - ref.loc[r.config, col]) > 1e-6:   # float32 storage of the members
                bad.append(f"{r.config} {col}: recomputed {r[col]!r}, released {ref.loc[r.config, col]!r}")
    for k, v in max_diff.items():
        if v > 1e-6:
            bad.append(f"{k.upper()} rebuilt from its members differs from the released column by {v:.2e}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the released Table 5 values")
    args = ap.parse_args()
    res, max_diff = run()
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))
    print("max |rebuilt - released| probability:", ", ".join(f"{k.upper()} {v:.1e}" for k, v in max_diff.items()))
    if args.check:
        bad = check(res, max_diff)
        for b in bad:
            print("MISMATCH", b)
        print("ABLATION A1-A10 =", "FAIL" if bad else "PASS")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

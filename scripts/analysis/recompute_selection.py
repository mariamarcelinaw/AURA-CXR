#!/usr/bin/env python3
"""Recompute every development-only selection step from the released OOF predictions.

Input: results/predictions/development_oof.csv (21,346 development cases; five-fold
out-of-fold probabilities of the four deep candidates and the two refined radiomics
learners, plus the cross-fitted A8 stack and the A7 soft vote). No images, no GPU and
no locked-test or external data are read.

Recomputed (paper Sections 3.2-3.4 and 5.1, Tables 3 and 4):
  1. OOF ROC-AUC, AUPRC, Brier score and fold-AUC mean/SD of each candidate;
  2. the cross-fitted macro-F1 of the two refined radiomics learners (threshold of each
     outer fold derived from the other four folds) and the LightGBM selection;
  3. the cross-fitted Logistic Regression stack of every deep pair with the radiomics
     probability, the pair ranking and the selected pair (DenseNet121-XRV + EVA-X-S);
  4. the full-development A8 coefficients and the development-derived thresholds of
     A8, A4, A7 and the single components.

With --check the results are compared with the released values in
configs/specs/deployment_manifest.json, configs/frozen_config.yaml and
results/metrics/result_registry.csv.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEEP = ("efficientnetv2", "resnet50", "xrv", "eva_x")
TOL_AUC = 0.002                      # selection tolerance (Sections 3.2 and 3.4)
GRID = np.round(np.arange(0.01, 0.99 + 1e-9, 0.005), 3)
MIN_SENS = 0.85


def threshold_rule(y, p):
    """Youden's J on a 0.01-0.99 grid (step 0.005) with sensitivity >= 0.85 (Section 3.4)."""
    best_j, best_t = -np.inf, None
    pos, neg = y == 1, y == 0
    for t in GRID:
        pred = p >= t
        se = pred[pos].mean()
        sp = (~pred[neg]).mean()
        if se >= MIN_SENS and se + sp - 1 > best_j + 1e-12:
            best_j, best_t = se + sp - 1, float(t)
    return best_t


def fold_auc(y, p, fold):
    a = [roc_auc_score(y[fold == k], p[fold == k]) for k in sorted(np.unique(fold))]
    return float(np.mean(a)), float(np.std(a, ddof=1))


def summary(y, p, fold):
    m, s = fold_auc(y, p, fold)
    return {"ROC_AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "BRIER": brier_score_loss(y, p), "FOLD_AUC_MEAN": m, "FOLD_AUC_SD": s}


def meta_learner():
    return LogisticRegression(class_weight="balanced", solver="lbfgs", C=1.0, max_iter=5000, random_state=42)


def crossfit(X, y, fold):
    out = np.zeros(len(y))
    for k in sorted(np.unique(fold)):
        tr = fold != k
        out[fold == k] = meta_learner().fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    return out


def crossfitted_macro_f1(y, p, fold):
    pred = np.zeros(len(y), dtype=int)
    for k in sorted(np.unique(fold)):
        held = fold == k
        pred[held] = (p[held] >= threshold_rule(y[~held], p[~held])).astype(int)
    return f1_score(y, pred, average="macro")


def run():
    d = pd.read_csv(ROOT / "results/predictions/development_oof.csv")
    assert len(d) == 21346 and d.patientId.is_unique
    folds = pd.read_csv(ROOT / "data/splits/development_oof_folds.csv")
    assert (d.patientId.values == folds.patientId.values).all() and (d.oof_fold.values == folds.oof_fold.values).all()
    test_ids = set(pd.read_csv(ROOT / "data/splits/test.csv").patientId)
    assert not test_ids & set(d.patientId), "locked-test identifiers in development OOF"
    y, fold = d.label.to_numpy(int), d.oof_fold.to_numpy(int)
    P = {m: d[f"p_{m}"].to_numpy(float) for m in DEEP}
    rad = {"lgbm": d.p_radiomics_lgbm.to_numpy(float), "histgb": d.p_radiomics_histgb.to_numpy(float)}

    res = {"individual": {}, "radiomics": {}, "pairs": {}}
    for m in DEEP:
        res["individual"][m] = {**summary(y, P[m], fold), "threshold": threshold_rule(y, P[m])}
    for m, p in rad.items():
        res["radiomics"][m] = {**summary(y, p, fold), "MACRO_F1": crossfitted_macro_f1(y, p, fold),
                               "threshold": threshold_rule(y, p)}
    best_f1 = max(v["MACRO_F1"] for v in res["radiomics"].values())
    within = [m for m, v in res["radiomics"].items() if v["MACRO_F1"] >= best_f1 - 0.002]
    res["radiomics_selected"] = max(within, key=lambda m: res["radiomics"][m]["ROC_AUC"])

    p_rad = rad[res["radiomics_selected"]]
    for a, b in itertools.combinations(DEEP, 2):
        oof = crossfit(np.column_stack([P[a], P[b], p_rad]), y, fold)
        res["pairs"][f"{a}+{b}"] = summary(y, oof, fold)
    ranked = sorted(res["pairs"], key=lambda k: -res["pairs"][k]["FOLD_AUC_MEAN"])
    res["pair_ranking"] = ranked
    res["pair_margin"] = res["pairs"][ranked[0]]["FOLD_AUC_MEAN"] - res["pairs"][ranked[1]]["FOLD_AUC_MEAN"]
    res["selected_pair"] = ranked[0]

    X = np.column_stack([P["xrv"], P["eva_x"], p_rad])
    a8_oof = crossfit(X, y, fold)
    full = meta_learner().fit(X, y)
    res["a8"] = {"intercept": float(full.intercept_[0]),
                 "coefficients": dict(zip(("p_xrv", "p_eva_x", "p_radiomics"), map(float, full.coef_[0]))),
                 "threshold": threshold_rule(y, a8_oof),
                 "max_abs_diff_vs_released_crossfit_oof": float(np.abs(a8_oof - d.p_a8_crossfit_oof).max())}
    res["a4_threshold"] = threshold_rule(y, (P["xrv"] + P["eva_x"]) / 2)
    w = {"xrv": res["individual"]["xrv"]["ROC_AUC"] - 0.5, "eva_x": res["individual"]["eva_x"]["ROC_AUC"] - 0.5,
         "radiomics": res["radiomics"][res["radiomics_selected"]]["ROC_AUC"] - 0.5}
    w = {k: v / sum(w.values()) for k, v in w.items()}
    soft = w["xrv"] * P["xrv"] + w["eva_x"] * P["eva_x"] + w["radiomics"] * p_rad
    res["a7"] = {"weights": w, "threshold": threshold_rule(y, soft),
                 "max_abs_diff_vs_released_soft_oof": float(np.abs(soft - d.p_a7_soft_oof).max())}
    return res


def check(res):
    reg = pd.read_csv(ROOT / "results/metrics/result_registry.csv").set_index("result_id").full_precision_value
    dep = json.loads((ROOT / "configs/specs/deployment_manifest.json").read_text())
    fz = yaml.safe_load((ROOT / "configs/frozen_config.yaml").read_text())["fusion"]
    bad = []

    def cmp(name, got, want, tol=1e-6):
        if want is None or abs(got - want) > tol:
            bad.append(f"{name}: recomputed {got!r}, released {want!r}")

    key = {"efficientnetv2": "EFFICIENTNETV2", "resnet50": "RESNET50", "xrv": "XRV", "eva_x": "EVA_X"}
    for m, v in res["individual"].items():
        for k in ("ROC_AUC", "AUPRC", "BRIER", "FOLD_AUC_MEAN", "FOLD_AUC_SD"):
            cmp(f"OOF_{key[m]}_{k}", v[k], reg.get(f"OOF_{key[m]}_{k}"))
    for m, v in res["radiomics"].items():
        for k in ("ROC_AUC", "AUPRC", "BRIER", "MACRO_F1", "FOLD_AUC_MEAN", "FOLD_AUC_SD"):
            cmp(f"RAD_OOF_{m.upper()}_{k}", v[k], reg.get(f"RAD_OOF_{m.upper()}_{k}"))
    for pair, v in res["pairs"].items():
        a, b = pair.split("+")
        for k in ("ROC_AUC", "AUPRC", "BRIER", "FOLD_AUC_MEAN", "FOLD_AUC_SD"):
            cmp(f"OOF_PAIR_{key[a]}_{key[b]}_{k}", v[k], reg.get(f"OOF_PAIR_{key[a]}_{key[b]}_{k}"))
    if res["radiomics_selected"] != dep["radiomics_learner"].lower().replace("lightgbm", "lgbm"):
        bad.append(f"radiomics learner {res['radiomics_selected']} != {dep['radiomics_learner']}")
    if res["selected_pair"] != "+".join(dep["selected_models"]) or res["pair_margin"] <= TOL_AUC:
        bad.append(f"pair selection {res['selected_pair']} (margin {res['pair_margin']:.4f})")
    cmp("A8 intercept", res["a8"]["intercept"], dep["LR_intercept"])
    for k, v in res["a8"]["coefficients"].items():
        cmp(f"A8 {k}", v, dep["LR_coefficients"][k])
    cmp("A8 threshold", res["a8"]["threshold"], dep["decision_threshold"])
    cmp("A8 cross-fitted OOF", res["a8"]["max_abs_diff_vs_released_crossfit_oof"], 0.0, 1e-6)
    cmp("A4 threshold", res["a4_threshold"], fz["a4_threshold"])
    cmp("A7 threshold", res["a7"]["threshold"], fz["soft_voting_threshold"])
    for k, v in res["a7"]["weights"].items():
        cmp(f"A7 weight {k}", v, fz["soft_voting_weights"][k])
    cmp("A7 soft OOF", res["a7"]["max_abs_diff_vs_released_soft_oof"], 0.0, 1e-6)
    for m, k in (("xrv", "xrv"), ("eva_x", "eva_x")):
        cmp(f"{m} threshold", res["individual"][m]["threshold"], fz["individual_thresholds"][k])
    cmp("radiomics threshold", res["radiomics"]["lgbm"]["threshold"], fz["individual_thresholds"]["radiomics"])
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs/recompute_selection"))
    ap.add_argument("--check", action="store_true", help="compare with the released values")
    args = ap.parse_args()
    res = run()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "recomputed_selection.json").write_text(json.dumps(res, indent=2))
    print("Candidate OOF (ROC-AUC / fold mean +- SD / threshold):")
    for m, v in {**res["individual"], **{f"radiomics_{k}": v for k, v in res["radiomics"].items()}}.items():
        print(f"  {m:18s} {v['ROC_AUC']:.4f}  {v['FOLD_AUC_MEAN']:.4f} +- {v['FOLD_AUC_SD']:.4f}  {v['threshold']:.3f}")
    print(f"Radiomics learner: {res['radiomics_selected']}  (cross-fitted macro-F1 LightGBM "
          f"{res['radiomics']['lgbm']['MACRO_F1']:.4f}, HistGB {res['radiomics']['histgb']['MACRO_F1']:.4f})")
    print("Pair + radiomics stacks, ranked by fold-mean ROC-AUC:")
    for k in res["pair_ranking"]:
        v = res["pairs"][k]
        print(f"  {k:26s} {v['FOLD_AUC_MEAN']:.4f} +- {v['FOLD_AUC_SD']:.4f}  pooled {v['ROC_AUC']:.4f}")
    a8 = res["a8"]
    print(f"Selected pair: {res['selected_pair']} (margin {res['pair_margin']:.4f} > tolerance {TOL_AUC})")
    print(f"A8: intercept {a8['intercept']:.4f}, coefficients "
          + ", ".join(f"{k} {v:.4f}" for k, v in a8["coefficients"].items()) + f", threshold {a8['threshold']:.3f}")
    print(f"A4 threshold {res['a4_threshold']:.3f}; A7 threshold {res['a7']['threshold']:.3f}")
    if args.check:
        bad = check(res)
        for b in bad:
            print("MISMATCH", b)
        print("RECOMPUTE SELECTION =", "FAIL" if bad else "PASS")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

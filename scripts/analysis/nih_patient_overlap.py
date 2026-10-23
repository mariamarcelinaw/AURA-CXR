#!/usr/bin/env python3
"""Post-hoc check: do development and locked-test images share NIH patients?

RSNA challenge identifiers ("patientId") index images, not persons: the 30,000 challenge
images came from 12,274 NIH patients (Shih et al., Radiology: AI 2019). The splits of this
study are disjoint at the level of these identifiers. This script maps every identifier to
its NIH ChestX-ray14 image (NNNNNNNN_FFF.png; the first eight digits are the NIH patient ID)
with the official RSNA mapping file, then

  1. counts NIH patients and images shared between the development cohort and the locked test,
  2. re-scores the released locked-test predictions of all eleven evaluated configurations
     (A1-A10 and the same-split ConvNeXt V2 Tiny comparator; thresholds unchanged, no retraining)
     on all test images and on the test images whose NIH patient has no development image, with
     class-stratified and patient-clustered bootstrap CIs,
  3. repeats the paired DeLong comparisons of the paper within each subset (the Holm-corrected
     family of Section 5.2, A8 versus A4 and A8 versus ConvNeXt V2 Tiny), and
  4. among test images of shared patients whose development images all carry one label,
     compares images with the same label and with a different label, and
  5. checks the development-only selection (Sections 3.2-3.4) on the development images of NIH
     patients whose images all lie in one OOF fold: candidate OOF ROC-AUC and the ranking of the
     six cross-fitted pair + radiomics stacks, recomputed from the released OOF predictions.

Usage (repository root):
    python scripts/analysis/nih_patient_overlap.py --mapping pneumonia-challenge-dataset-mappings_2018.json
    python scripts/analysis/nih_patient_overlap.py --check   # re-scores from the released per-case file;
                                                             # needs no mapping file

The mapping file is "A mapping of the RSNA image dataset to the original NIH dataset (JSON)" on
https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from paired_comparisons import paired_delong  # noqa: E402
from recompute_selection import DEEP, crossfit  # noqa: E402
OUT = ROOT / "results/analysis/nih_patient_overlap"
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
NIH = re.compile(r"(\d{8})_(\d{3})\.png$")
# development-derived thresholds (Table 5; ConvNeXt V2 Tiny: its own development OOF threshold)
THRESHOLDS = {"a8": 0.470, "a4": 0.490, "xrv": 0.495, "eva_x": 0.470, "radiomics": 0.225, "a5": 0.380,
              "a6": 0.375, "a7": 0.435, "a9": 0.455, "a10": 0.470, "convnextv2": 0.455}
SOURCE = {"xrv": "a1", "eva_x": "a2", "radiomics": "a3"}
HOLM_FAMILY = ("xrv", "eva_x", "radiomics", "a7", "a8")          # Section 5.2, ten comparisons
EXTRA_PAIRS = (("a8", "a4"), ("a8", "convnextv2"))


def collect_pairs(obj, out):
    """Records holding exactly one RSNA identifier and one NIH image name, whatever the JSON layout."""
    if isinstance(obj, dict):
        vals = [v for v in obj.values() if isinstance(v, str)]
        uu = [v for v in vals if UUID.match(v)]
        nih = [v for v in vals if NIH.search(v)]
        if len(uu) == 1 and len(nih) == 1:
            out.append((uu[0], NIH.search(nih[0]).group(0)))
        for v in obj.values():
            collect_pairs(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_pairs(v, out)
    return out


def auc(y, p):
    r = rankdata(p)
    n1 = int(y.sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def resamples(y, groups, n_boot=2000, seed=42):
    """Index sets for the class-stratified and the patient-clustered bootstrap."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    strat = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n_boot)]
    rng = np.random.default_rng(seed)
    by = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).indices
    keys = np.array(list(by))
    clus = []
    while len(clus) < n_boot:
        idx = np.concatenate([by[k] for k in rng.choice(keys, len(keys))])
        if 0 < y[idx].sum() < len(idx):
            clus.append(idx)
    return strat, clus


def metrics(y, p, thr, strat, clus):
    b = [auc(y[i], p[i]) for i in strat]
    c = [auc(y[i], p[i]) for i in clus]
    pred = p >= thr
    return {"n": int(len(y)), "positive": int(y.sum()), "prevalence": float(y.mean()),
            "roc_auc": auc(y, p), "roc_auc_ci_low": float(np.percentile(b, 2.5)),
            "roc_auc_ci_high": float(np.percentile(b, 97.5)),
            "roc_auc_patient_ci_low": float(np.percentile(c, 2.5)),
            "roc_auc_patient_ci_high": float(np.percentile(c, 97.5)), "auprc": float(average_precision_score(y, p)),
            "sensitivity": float(pred[y == 1].mean()), "specificity": float((~pred[y == 0]).mean()),
            "accuracy": float((pred == (y == 1)).mean())}


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for i, k in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(run, 1.0)
    return adj


def load_predictions(cases):
    abl = pd.read_csv(ROOT / "results/predictions/locked_rsna_a1_a10.csv")
    cnx = pd.read_csv(ROOT / "results/analysis/contemporary_baseline/06_BASELINE_LOCKED_PREDICTIONS.csv")
    assert (cnx.patient_id.values == abl.case_id.values).all()
    d = abl.assign(convnextv2_probability=cnx.probability.to_numpy(float))
    for k, v in SOURCE.items():
        d[f"{k}_probability"] = d[f"{v}_probability"]
    d = d.merge(cases, left_on="case_id", right_on="patientId", validate="one_to_one")
    assert len(d) == 5338
    return d


def subsets(d):
    uniform = d[d.shared & d.dev_label_uniform]
    return [("all locked test", d),
            ("no NIH patient in development", d[~d.shared]),
            ("NIH patient also in development", d[d.shared]),
            ("shared, same label as patient's development images", uniform[uniform.true_label == uniform.dev_label]),
            ("shared, label differs from patient's development images", uniform[uniform.true_label != uniform.dev_label])]


def score(cases):
    d = load_predictions(cases)
    rows, tests = [], []
    for name, s in subsets(d):
        y = s.true_label.to_numpy(int)
        if y.min() == y.max():
            continue
        strat, clus = resamples(y, s.nih_patient.to_numpy())
        for model, thr in THRESHOLDS.items():
            rows.append({"subset": name, "model": model, "threshold": thr,
                         **metrics(y, s[f"{model}_probability"].to_numpy(float), thr, strat, clus)})
        if name not in ("all locked test", "no NIH patient in development", "NIH patient also in development"):
            continue
        fam = [(b, a) for a, b in itertools.combinations(HOLM_FAMILY, 2)]
        res = [paired_delong(y, s[f"{a}_probability"].to_numpy(float), s[f"{b}_probability"].to_numpy(float))
               for a, b in fam + list(EXTRA_PAIRS)]
        p_holm = list(holm([r["p_value"] for r in res[:len(fam)]])) + [np.nan] * len(EXTRA_PAIRS)
        for (a, b), r, ph, fm in zip(fam + list(EXTRA_PAIRS), res, p_holm,
                                     ["Section 5.2 Holm family (10)"] * len(fam) + ["single comparison"] * 2):
            tests.append({"subset": name, "model_new": a, "model_ref": b, "delta_auc": r["delta"],
                          "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_value": r["p_value"],
                          "p_holm": ph, "family": fm})
    return pd.DataFrame(rows), pd.DataFrame(tests)


def development_selection(dev_cases):
    """Candidate OOF ROC-AUC and pair-stack ranking on all development images and on those of patients in one fold."""
    d = pd.read_csv(ROOT / "results/predictions/development_oof.csv").merge(
        dev_cases[["patientId", "single_fold"]], on="patientId", validate="one_to_one")
    assert len(d) == 21346
    y, fold = d.label.to_numpy(int), d.oof_fold.to_numpy(int)
    p = {m: d[f"p_{m}"].to_numpy(float) for m in DEEP}
    p.update(radiomics_lgbm=d.p_radiomics_lgbm.to_numpy(float), radiomics_histgb=d.p_radiomics_histgb.to_numpy(float))
    stacks = {f"{a}+{b}+radiomics_lgbm": crossfit(np.column_stack([p[a], p[b], p["radiomics_lgbm"]]), y, fold)
              for a, b in itertools.combinations(DEEP, 2)}
    rows = []
    for name, mask in (("all development", np.ones(len(d), bool)),
                       ("NIH patient within one OOF fold", d.single_fold.to_numpy(bool))):
        for kind, preds in (("candidate", {k: v for k, v in p.items()}), ("pair stack", stacks)):
            for model, q in preds.items():
                fold_auc = [auc(y[mask & (fold == k)], q[mask & (fold == k)]) for k in range(5)]
                rows.append({"subset": name, "kind": kind, "model": model, "n": int(mask.sum()),
                             "positive": int(y[mask].sum()), "oof_roc_auc": auc(y[mask], q[mask]),
                             "fold_auc_mean": float(np.mean(fold_auc)), "fold_auc_sd": float(np.std(fold_auc, ddof=1))})
    res = pd.DataFrame(rows)
    res["rank_within_kind"] = (res.groupby(["subset", "kind"]).fold_auc_mean.rank(ascending=False, method="first")
                               .astype(int))
    return res


def build(mapping_path):
    m = pd.DataFrame(collect_pairs(json.loads(Path(mapping_path).read_text()), []),
                     columns=["patientId", "nih_image"]).drop_duplicates()
    if m.patientId.duplicated().any():
        raise SystemExit("mapping assigns more than one NIH image to an RSNA identifier")
    m["nih_patient"] = m.nih_image.str[:8]
    dev = pd.read_csv(ROOT / "data/splits/development.csv")[["patientId", "label"]]
    test = pd.read_csv(ROOT / "data/splits/test.csv")[["patientId"]]
    dev, test = dev.merge(m, on="patientId", how="left"), test.merge(m, on="patientId", how="left")
    missing = int(dev.nih_patient.isna().sum() + test.nih_patient.isna().sum())
    if missing:
        raise SystemExit(f"{missing} identifiers have no NIH mapping; use the RSNA 2018 challenge mapping file")
    folds = pd.read_csv(ROOT / "data/splits/development_oof_folds.csv")[["patientId", "oof_fold"]]
    dev = dev.merge(folds, on="patientId", validate="one_to_one")
    n_folds = dev.groupby("nih_patient").oof_fold.nunique()
    dev["single_fold"] = dev.nih_patient.map(n_folds).eq(1)
    train_pat = set(pd.read_csv(ROOT / "data/splits/train.csv").merge(m, on="patientId").nih_patient)
    val = pd.read_csv(ROOT / "data/splits/val.csv").merge(m, on="patientId")
    dev_lab = dev.groupby("nih_patient").label.agg(["min", "max"])
    cases = test.assign(shared=test.nih_patient.isin(dev_lab.index))
    cases["dev_label_uniform"] = cases.nih_patient.map(dev_lab["min"] == dev_lab["max"]).fillna(False).astype(bool)
    cases["dev_label"] = cases.nih_patient.map(dev_lab["max"]).astype("Int64")
    summary = {
        "mapping_file": "pneumonia-challenge-dataset-mappings_2018.json (RSNA)",
        "mapping_sha256": hashlib.sha256(Path(mapping_path).read_bytes()).hexdigest(),
        "rsna_identifiers": int(len(dev) + len(test)),
        "nih_patients_total": int(pd.concat([dev.nih_patient, test.nih_patient]).nunique()),
        "nih_patients_development": int(dev.nih_patient.nunique()),
        "nih_patients_locked_test": int(test.nih_patient.nunique()),
        "nih_patients_shared": int(len(set(dev.nih_patient) & set(test.nih_patient))),
        "locked_test_images": int(len(test)),
        "locked_test_images_shared_patient": int(cases.shared.sum()),
        "locked_test_images_unshared_patient": int((~cases.shared).sum()),
        "max_images_per_nih_patient": int(pd.concat([dev.nih_patient, test.nih_patient]).value_counts().max()),
        "development_images_patient_in_several_oof_folds": int((~dev.single_fold).sum()),
        "development_images_patient_within_one_oof_fold": int(dev.single_fold.sum()),
        "validation_images_patient_in_training_split": int(val.nih_patient.isin(train_pat).sum()),
        "analysis_origin": "post_hoc; uses released predictions and frozen thresholds; no retraining",
    }
    return (cases[["patientId", "nih_image", "nih_patient", "shared", "dev_label_uniform", "dev_label"]],
            dev[["patientId", "nih_image", "nih_patient", "oof_fold", "single_fold"]], summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mapping", help="RSNA-to-NIH mapping JSON (pneumonia-challenge-dataset-mappings_2018.json)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="re-score from the released per-case file and compare")
    args = ap.parse_args()
    out = Path(args.out)
    if args.check:
        cases = pd.read_csv(OUT / "locked_test_nih_patients.csv", dtype={"nih_patient": str})
        ok = True
        for got, fname, keys in zip(score(cases), ("nih_patient_overlap_metrics.csv", "nih_patient_overlap_delong.csv"),
                                    (["subset", "model"], ["subset", "model_new", "model_ref", "family"])):
            ref = pd.read_csv(OUT / fname)
            num = [c for c in ref.columns if c not in keys]
            ok = ok and (len(got) == len(ref) and (got[keys].values == ref[keys].values).all()
                         and np.allclose(got[num].to_numpy(float), ref[num].to_numpy(float), atol=1e-9,
                                         equal_nan=True))
        dev_cases = pd.read_csv(OUT / "development_nih_patients.csv", dtype={"nih_patient": str})
        got, ref = development_selection(dev_cases), pd.read_csv(OUT / "nih_patient_overlap_development_selection.csv")
        keys = ["subset", "kind", "model"]
        num = [c for c in ref.columns if c not in keys]
        ok = ok and (len(got) == len(ref) and (got[keys].values == ref[keys].values).all()
                     and np.allclose(got[num].to_numpy(float), ref[num].to_numpy(float), atol=1e-9))
        s = json.loads((OUT / "summary.json").read_text())
        ok = ok and s["locked_test_images_shared_patient"] == int(cases.shared.sum())
        ok = ok and s["development_images_patient_within_one_oof_fold"] == int(dev_cases.single_fold.sum())
        print("NIH PATIENT OVERLAP =", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if not args.mapping:
        ap.error("--mapping is required unless --check is given")
    cases, dev_cases, summary = build(args.mapping)
    res, tests = score(cases)
    dsel = development_selection(dev_cases)
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / "locked_test_nih_patients.csv", index=False)
    res.to_csv(out / "nih_patient_overlap_metrics.csv", index=False)
    tests.to_csv(out / "nih_patient_overlap_delong.csv", index=False)
    dev_cases.to_csv(out / "development_nih_patients.csv", index=False)
    dsel.to_csv(out / "nih_patient_overlap_development_selection.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(res[["subset", "model", "n", "prevalence", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "auprc",
               "sensitivity", "specificity"]].round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(dsel.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

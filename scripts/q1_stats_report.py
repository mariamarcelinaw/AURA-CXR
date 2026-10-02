#!/usr/bin/env python3
"""
q1_stats_report.py — Q1 statistics for development-OOF selected DL pair + radiomics.

Membaca artefak yang SUDAH ADA (tanpa latih ulang, tanpa GPU/DICOM):
  <results_dir>/probs/*.npy         (probabilitas val & test tiap model)
  <results_dir>/splits/{val,test}.csv  ATAU  --labels_csv untuk label
  <results_dir>/reports/threshold_tuning_*.json  (threshold operasi, opsional)

Menghasilkan (di <results_dir>/reports/):
  q1_test_metrics.csv     — AUC, AUPRC, F1, sensitivitas, spesifisitas, PPV, NPV,
                            balanced acc, MCC + 95% CI bootstrap tiap model
  q1_auc_delong.csv       — uji DeLong (p-value) untuk beda AUC antar model
  q1_calibration.csv      — Brier, ECE, MCE (+ setelah temperature scaling)
  q1_reliability_<m>.csv  — data reliability curve per model
  Q1_RESULTS_TABLE.md     — tabel siap tempel ke manuskrip

Contoh:
  python q1_stats_report.py --results_dir /path/hasil --n_boot 2000
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, balanced_accuracy_score,
                             matthews_corrcoef, brier_score_loss, confusion_matrix)

RNG = np.random.default_rng(42)

from aura_dl_pair_selection import (
    load_pair_manifest, load_oof_artifact, ensure_development_table,
    normalize_binary_probability, selected_pair_markdown, DISPLAY_NAMES, tune_threshold,
    load_deployment_manifest, resolve_probability_file,
)


# ────────────────────────── DeLong (Sun & Xu, 2014) ──────────────────────────
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Return (auc_a, auc_b, p_value) for H0: AUC_a == AUC_b (correlated ROC)."""
    order = (-np.stack([prob_a, prob_b])).argsort()[0]
    label_1_count = int(np.sum(y_true == 1))
    y = y_true[order]
    # reorder so positives first
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    new_order = np.concatenate([order[pos_idx], order[neg_idx]])
    preds = np.stack([prob_a[new_order], prob_b[new_order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    z_denom = np.sqrt(np.dot(np.dot(l, cov), l.T))
    if z_denom[0, 0] <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = np.abs(aucs[0] - aucs[1]) / z_denom[0, 0]
    p = 2 * (1 - stats.norm.cdf(z))
    return float(aucs[0]), float(aucs[1]), float(p)


# ────────────────────────── metrik + bootstrap CI ───────────────────────────
def point_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "AUC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "F1_pneumonia": f1_score(y, pred, zero_division=0),
        "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv,
        "balanced_acc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
    }


def bootstrap_ci(y, p, thr, n_boot=2000, alpha=0.05):
    keys = ["AUC", "AUPRC", "F1_pneumonia", "sensitivity", "specificity",
            "PPV", "NPV", "balanced_acc", "MCC"]
    acc = {k: [] for k in keys}
    n = len(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    for _ in range(n_boot):
        bi = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                             RNG.choice(neg, len(neg), replace=True)])
        yb, pb = y[bi], p[bi]
        if len(np.unique(yb)) < 2:
            continue
        m = point_metrics(yb, pb, thr)
        for k in keys:
            acc[k].append(m[k])
    ci = {}
    for k in keys:
        arr = np.array([v for v in acc[k] if np.isfinite(v)])
        if len(arr):
            ci[k] = (float(np.percentile(arr, 100 * alpha / 2)),
                     float(np.percentile(arr, 100 * (1 - alpha / 2))))
        else:
            ci[k] = (np.nan, np.nan)
    return ci


# ────────────────────────── kalibrasi ───────────────────────────────────────
def ece_mce(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    rows = []
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            rows.append((0.5 * (bins[i] + bins[i + 1]), np.nan, np.nan, 0))
            continue
        conf = p[m].mean(); acc = y[m].mean(); w = m.mean()
        gap = abs(acc - conf)
        ece += w * gap; mce = max(mce, gap)
        rows.append((0.5 * (bins[i] + bins[i + 1]), conf, acc, int(m.sum())))
    return float(ece), float(mce), rows


def fit_temperature(y_val, p_val):
    """Fit scalar T>0 memaksimalkan NLL kalibrasi pada validasi (grid + refine)."""
    eps = 1e-6
    p = np.clip(p_val, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    def nll(T):
        pt = 1 / (1 + np.exp(-logit / T))
        pt = np.clip(pt, eps, 1 - eps)
        return -np.mean(y_val * np.log(pt) + (1 - y_val) * np.log(1 - pt))
    Ts = np.linspace(0.5, 3.0, 51)
    T = min(Ts, key=nll)
    for _ in range(3):
        Ts = np.linspace(max(0.3, T - 0.1), T + 0.1, 41)
        T = min(Ts, key=nll)
    return float(T)


def apply_temperature(p, T):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    return 1 / (1 + np.exp(-logit / T))


# ────────────────────────── loader ──────────────────────────────────────────
def col1(a):
    a = np.asarray(a)
    return a[:, 1] if a.ndim == 2 else a



def load_probs(results_dir, split, manifest):
    """Load only exact artifacts registered in the locked deployment manifest."""
    root = Path(results_dir); selected = list(manifest["selected_dl_models"])
    display = dict(DISPLAY_NAMES)
    try:
        dep = load_deployment_manifest(root, strict=True)
        display["radiomics"] = dep.get("radiomics_selection", {}).get(
            "selected_display_name", display["radiomics"]
        )
    except Exception:
        pass
    out = {}
    for key in selected + ["radiomics", "soft", "stacked"]:
        try:
            path = resolve_probability_file(root, key, split, exact_deployment=True)
        except FileNotFoundError:
            continue
        label = display.get(key, "Selected Pair Soft Voting" if key == "soft" else "Selected Pair Stacked Ensemble")
        out[label] = col1(np.load(path))
    return out






def load_predictive_interval_widths(results_dir, selected, display, expected_n):
    """Load exact locked-test 95% predictive interval widths for all final models."""
    root = Path(results_dir); pdir = root / "probs"
    out = {}
    for key in selected:
        path = pdir / f"{key}_development_refit_test_predictive_interval_width_95.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictive uncertainty artifact for {key}: {path}")
        arr = col1(np.load(path)).astype(float)
        if len(arr) != expected_n:
            raise ValueError(f"Predictive uncertainty length mismatch for {key}: {len(arr)} != {expected_n}")
        out[display[key]] = arr
    out[display["radiomics"]] = np.zeros(expected_n, dtype=float)
    for key, label in [("soft", "Selected Pair Soft Voting"), ("stacked", "Selected Pair Stacked Ensemble")]:
        path = pdir / f"{key}_predictive_interval_width_95_locked_test.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing ensemble predictive uncertainty artifact: {path}")
        arr = col1(np.load(path)).astype(float)
        if len(arr) != expected_n:
            raise ValueError(f"Predictive uncertainty length mismatch for {key}: {len(arr)} != {expected_n}")
        out[label] = arr
    return out


def holm_bonferroni(p_values):
    """Return Holm-adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float); m=len(p)
    order=np.argsort(p); adjusted=np.empty(m,dtype=float); running=0.0
    for rank, idx in enumerate(order):
        val=(m-rank)*p[idx]; running=max(running,val); adjusted[idx]=min(1.0,running)
    return adjusted



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--sens_floor", type=float, default=0.85)
    args = ap.parse_args()

    root = Path(args.results_dir); rep = root / "reports"; rep.mkdir(parents=True, exist_ok=True)
    pair = load_pair_manifest(root, strict=True); deployment = load_deployment_manifest(root, strict=True)
    if not pair.get("final_test_evaluated_once") or deployment.get("deployment_status") not in {"LOCKED_TEST_INFERRED", "FINALIZED"}:
        raise RuntimeError("Final statistics require completed one-time locked-test fusion")
    yt = pd.read_csv(root / "splits" / "test.csv")["label"].to_numpy(dtype=int)
    dev = ensure_development_table(root); ydev = dev["label"].to_numpy(dtype=int)
    selected = list(pair["selected_dl_models"])
    display = dict(DISPLAY_NAMES)
    display["radiomics"] = deployment.get("radiomics_selection", {}).get(
        "selected_display_name", display["radiomics"]
    )

    test = load_probs(root, "test", pair)
    final_models = [display[selected[0]], display[selected[1]], display["radiomics"],
                    "Selected Pair Soft Voting", "Selected Pair Stacked Ensemble"]
    missing = [m for m in final_models if m not in test]
    if missing: raise FileNotFoundError(f"Missing exact locked deployment probabilities: {missing}")
    predictive_width = load_predictive_interval_widths(root, selected, display, len(yt))

    oof_map = {}
    for key in selected + ["radiomics"]:
        oof_map[key], _ = load_oof_artifact(root, key, dev)
    stacked_oof = col1(np.load(root / "probs" / "stacked_development_oof.npy"))
    soft_oof = col1(np.load(root / "probs" / "soft_development_oof.npy"))
    development_prob = {display[k]: v for k,v in oof_map.items()}
    development_prob["Selected Pair Stacked Ensemble"] = stacked_oof
    development_prob["Selected Pair Soft Voting"] = soft_oof

    thresholds = {display[k]: float(pair["individual_thresholds"][k]) for k in selected + ["radiomics"]}
    thresholds["Selected Pair Stacked Ensemble"] = float(pair["threshold"])
    thresholds["Selected Pair Soft Voting"] = float(pair["soft_voting_threshold"])

    print(f"[INFO] Locked selected pair: {pair['selected_pair_display']}")
    print(f"[INFO] Exact deployment lock: {deployment['deployment_lock_id']}")
    selected_pair_markdown(root)

    rows=[]
    for m in final_models:
        thr=thresholds[m]; pm=point_metrics(yt,test[m],thr); ci=bootstrap_ci(yt,test[m],thr,n_boot=args.n_boot)
        row={
            "model":m, "threshold":round(thr,4), "artifact_source":"locked_deployment_manifest",
            "predictive_interval_width_95_mean": round(float(np.mean(predictive_width[m])), 6),
            "predictive_interval_width_95_median": round(float(np.median(predictive_width[m])), 6),
            "predictive_uncertainty_definition": "two_sided_95_predictive_interval_width_equals_2_times_1.96_times_mc_probability_std",
        }
        for k in ["AUC","AUPRC","F1_pneumonia","sensitivity","specificity","PPV","NPV","balanced_acc","MCC"]:
            row[k]=round(pm[k],4); row[f"{k}_CI"]=f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        rows.append(row)
    mdf=pd.DataFrame(rows); mdf.to_csv(rep/"q1_test_metrics.csv",index=False)

    drows=[]
    for i in range(len(final_models)):
        for j in range(i+1,len(final_models)):
            a,b=final_models[i],final_models[j]; aa,bb,pv=delong_roc_test(yt,test[a],test[b])
            primary = bool("Selected Pair Stacked Ensemble" in {a,b})
            drows.append({"model_A":a,"model_B":b,"AUC_A":aa,"AUC_B":bb,"delta":aa-bb,
                          "p_value_raw":pv,"comparison_role":"primary" if primary else "secondary_exploratory"})
    ddf=pd.DataFrame(drows)
    ddf["p_value_holm"] = holm_bonferroni(ddf["p_value_raw"].to_numpy())
    ddf["significant_holm_0.05"] = ddf["p_value_holm"] < 0.05
    ddf.to_csv(rep/"q1_auc_delong.csv",index=False)

    crows=[]
    for m in final_models:
        pv,pt=development_prob[m],test[m]; T=fit_temperature(ydev,pv); ptc=apply_temperature(pt,T)
        e0,m0,rel0=ece_mce(yt,pt); e1,m1,_=ece_mce(yt,ptc)
        crows.append({"model":m,"temperature":T,"Brier":brier_score_loss(yt,pt),
                      "Brier_tempscaled":brier_score_loss(yt,ptc),"ECE":e0,"ECE_tempscaled":e1,
                      "MCE":m0,"MCE_tempscaled":m1,"calibration_source":"development_oof"})
        pd.DataFrame(rel0,columns=["bin_center","confidence","accuracy","count"]).to_csv(
            rep/f"q1_reliability_{m.lower().replace(' ','_').replace('-','_')}.csv",index=False)
    pd.DataFrame(crows).to_csv(rep/"q1_calibration.csv",index=False)

    def fmt(m,k):
        r=mdf[mdf.model==m].iloc[0]; return f"{r[k]:.3f} {r[k+'_CI']}"
    lines=["# Q1 Results — Locked Selected-Pair AURA-CXR","",
           f"Selected pair: **{pair['selected_pair_display']}**. Test n={len(yt)}; pneumonia={int(yt.sum())}.",
           "All model identities, thresholds, calibration sources, and feature order were locked from development OOF before test inference.","",
           "| Model | AUC | AUPRC | Sensitivity | Specificity | F1 (pneu) | Bal.Acc | MCC | Mean PI width 95% |",
           "|---|---|---|---|---|---|---|---|---:|"]
    for m in final_models:
        u = float(mdf.loc[mdf.model == m, "predictive_interval_width_95_mean"].iloc[0])
        lines.append(f"| {m} | {fmt(m,'AUC')} | {fmt(m,'AUPRC')} | {fmt(m,'sensitivity')} | {fmt(m,'specificity')} | {fmt(m,'F1_pneumonia')} | {fmt(m,'balanced_acc')} | {fmt(m,'MCC')} | {u:.4f} |")
    lines += ["", "## DeLong comparisons (Holm-corrected)", "",
              "| Model A | Model B | ΔAUC | raw p | Holm p | Role | Significant |",
              "|---|---|---:|---:|---:|---|---|"]
    for _,r in ddf.iterrows():
        lines.append(f"| {r.model_A} | {r.model_B} | {r.delta:+.4f} | {r.p_value_raw:.4g} | {r.p_value_holm:.4g} | {r.comparison_role} | {'Yes' if bool(r['significant_holm_0.05']) else 'No'} |")
    (rep/"Q1_RESULTS_TABLE.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    deployment["deployment_status"]="FINALIZED"; deployment["statistics_completed_at"]=pd.Timestamp.now().isoformat()
    from aura_dl_pair_selection import save_json
    save_json(deployment,rep/"q1_locked_deployment_manifest.json")
    print("[OK] Strict locked-deployment Q1 statistics completed.")



if __name__ == "__main__":
    main()

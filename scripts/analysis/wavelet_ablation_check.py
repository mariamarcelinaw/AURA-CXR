#!/usr/bin/env python3
"""Recompute the adaptive-versus-fixed wavelet ablation (Section 5.3, Table 6) from released files.

Reads the released out-of-fold radiomics predictions of the five wavelet policies
(results/analysis/wavelet_ablation/oof/{adaptive,fixed_haar,fixed_db4,fixed_sym4,fixed_coif3}.csv)
and recomputes, on a CPU and without images:

  * the pooled OOF ROC-AUC, AUPRC, Brier score, cross-fitted macro-F1 and fold-AUC mean/SD (Table 6);
  * the four adaptive-minus-fixed paired DeLong ROC-AUC comparisons with Holm adjustment (Section 5.3);
  * the per-image wavelet selection frequencies of the adaptive policy (Section 5.3).

It also checks that the adaptive policy equals the released radiomics LightGBM OOF predictions
(results/predictions/development_oof.csv, column p_radiomics_lgbm).

    python scripts/analysis/wavelet_ablation_check.py            # write to runs/wavelet_ablation_check/
    python scripts/analysis/wavelet_ablation_check.py --check    # also compare with the released results

Regenerating the OOF predictions themselves needs the images: scripts/analysis/wavelet_ablation_run.py,
then scripts/analysis/wavelet_ablation_statistics.py.
"""
from pathlib import Path
import argparse, importlib.util, sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / 'results/analysis/wavelet_ablation'
POLICIES = ['adaptive', 'fixed_haar', 'fixed_db4', 'fixed_sym4', 'fixed_coif3']
FIXED = POLICIES[1:]
MIN_SENS = 0.85

spec = importlib.util.spec_from_file_location('pc', ROOT / 'scripts/analysis/paired_comparisons.py')
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i])); adj[i] = running
    return adj


def tune_threshold(y, p, min_sensitivity=MIN_SENS):
    """Same rule as wavelet_ablation_statistics.py: Youden's J on a 0.01-0.99 grid with a sensitivity floor."""
    best = None
    for floor in (min_sensitivity, None):
        for th in np.linspace(0.01, 0.99, 197):
            pred = p >= th
            tp = np.sum((y == 1) & pred); fn = np.sum((y == 1) & ~pred)
            tn = np.sum((y == 0) & ~pred); fp = np.sum((y == 0) & pred)
            sens = tp / max(tp + fn, 1); spec_ = tn / max(tn + fp, 1)
            if floor is not None and sens < floor:
                continue
            cand = (sens + spec_ - 1, spec_, sens, -abs(th - 0.5), th)
            if best is None or cand > best:
                best = cand
        if best is not None:
            break
    return float(best[-1])


def crossfitted_macro_f1(y, p, folds):
    pred = np.zeros(len(y), dtype=int)
    for k in sorted(np.unique(folds)):
        hold = folds == k
        pred[hold] = (p[hold] >= tune_threshold(y[~hold], p[~hold])).astype(int)
    return float(f1_score(y, pred, average='macro', zero_division=0))


def load():
    data = {p: pd.read_csv(REL / 'oof' / f'{p}.csv') for p in POLICIES}
    base = data['adaptive']
    for p, df in data.items():
        assert df.patient_id.tolist() == base.patient_id.tolist(), f'{p}: row order differs'
        assert (df.fold.to_numpy() == base.fold.to_numpy()).all() and (df.true_label.to_numpy() == base.true_label.to_numpy()).all()
    return data


def compute():
    data = load(); base = data['adaptive']
    y = base.true_label.to_numpy(int); folds = base.fold.to_numpy(int)
    pred = {p: data[p].probability.to_numpy(float) for p in POLICIES}

    dev = pd.read_csv(ROOT / 'results/predictions/development_oof.csv').set_index('patientId')
    adaptive_vs_released = float(np.abs(dev.reindex(base.patient_id).p_radiomics_lgbm.to_numpy(float) - pred['adaptive']).max())

    pooled = []
    for p in POLICIES:
        fold_aucs = [roc_auc_score(y[folds == k], pred[p][folds == k]) for k in sorted(np.unique(folds))]
        pooled.append({'policy': p, 'ROC_AUC': roc_auc_score(y, pred[p]), 'AUPRC': average_precision_score(y, pred[p]),
                       'Brier': brier_score_loss(y, pred[p]), 'crossfitted_macro_F1': crossfitted_macro_f1(y, pred[p], folds),
                       'fold_auc_mean': float(np.mean(fold_aucs)), 'fold_auc_sd': float(np.std(fold_aucs, ddof=1))})
    pooled = pd.DataFrame(pooled)

    rows = []
    for f in FIXED:
        r = pc.paired_delong(y, pred['adaptive'], pred[f])
        rows.append({'fixed_policy': f, 'delta_adaptive_minus_fixed': r['delta'], 'CI_low': r['ci_low'],
                     'CI_high': r['ci_high'], 'raw_p': r['p_value']})
    delong = pd.DataFrame(rows); delong['adjusted_p'] = holm(delong.raw_p)

    sel = base.selected_wavelet.value_counts(normalize=True).mul(100).rename('percentage').reset_index()
    return pooled, delong, sel, adaptive_vs_released


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true', help='compare with the released results')
    args = ap.parse_args()
    pooled, delong, sel, adaptive_diff = compute()
    dst = ROOT / 'runs/wavelet_ablation_check'; dst.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(dst / 'pooled_metrics.csv', index=False); delong.to_csv(dst / 'paired_delong.csv', index=False)
    print('Policy        ROC-AUC  AUPRC   Brier   macro-F1  fold AUC mean +- SD')
    for r in pooled.itertuples():
        print(f'{r.policy:12s}  {r.ROC_AUC:.4f}  {r.AUPRC:.4f}  {r.Brier:.4f}  {r.crossfitted_macro_F1:.4f}    {r.fold_auc_mean:.4f} +- {r.fold_auc_sd:.4f}')
    print('Adaptive - fixed (paired DeLong, Holm across four contrasts):')
    for r in delong.itertuples():
        print(f'  {r.fixed_policy:12s} {r.delta_adaptive_minus_fixed:+.4f}  95% CI {r.CI_low:+.4f} to {r.CI_high:+.4f}  Holm p = {r.adjusted_p:.3g}')
    print('Adaptive selection (%): ' + ', '.join(f'{w} {v:.1f}' for w, v in zip(sel.selected_wavelet, sel.percentage)))
    print(f'Adaptive policy vs released radiomics OOF: max |difference| {adaptive_diff:.1e}')
    if not args.check:
        return 0
    ref_pooled = pd.read_csv(REL / 'pooled_metrics.csv').set_index('policy').loc[POLICIES]
    cols = ['ROC_AUC', 'AUPRC', 'Brier', 'crossfitted_macro_F1', 'fold_auc_mean', 'fold_auc_sd']
    d1 = float(np.abs(ref_pooled[cols].to_numpy() - pooled.set_index('policy')[cols].to_numpy()).max())
    ref_pc = pd.read_csv(REL / 'paired_comparisons.csv')
    ref_pc = ref_pc[ref_pc.metric == 'ROC_AUC'].set_index('fixed_policy').loc[FIXED]
    cols2 = ['delta_adaptive_minus_fixed', 'CI_low', 'CI_high', 'raw_p', 'adjusted_p']
    d2 = float(np.abs(ref_pc[cols2].to_numpy() - delong.set_index('fixed_policy')[cols2].to_numpy()).max())
    ref_sel = pd.read_csv(REL / 'selection_distribution.csv')
    ref_sel = ref_sel[ref_sel.stratum_type == 'overall'].set_index('wavelet').percentage
    d3 = float(np.abs(ref_sel.reindex(sel.selected_wavelet).to_numpy() - sel.percentage.to_numpy()).max())
    print(f'Largest difference from the released results: Table 6 {d1:.1e}, DeLong/Holm {d2:.1e}, selection {d3:.1e}')
    ok = d1 < 1e-9 and d2 < 1e-9 and d3 < 1e-9 and adaptive_diff < 1e-6
    print('WAVELET ABLATION = ' + ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

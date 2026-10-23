#!/usr/bin/env python3
"""Recompute the post-hoc paired comparisons from the released per-case predictions.

Comparisons (difference orientation in brackets):
  1. locked RSNA test, A8 versus A4                     [A8 - A4]
  2. Kermany pediatric test, A8 versus EVA-X-S alone     [A8 - EVA-X-S]
  3. locked RSNA test, A8 versus ConvNeXt V2 Tiny        [A8 - ConvNeXt V2 Tiny]
  4. locked RSNA test, DenseNet121-XRV versus ConvNeXt V2 Tiny  [XRV - ConvNeXt V2 Tiny]
  5. locked RSNA test, EVA-X-S versus ConvNeXt V2 Tiny          [EVA-X-S - ConvNeXt V2 Tiny]
  6. Kermany pediatric test, A8 versus A4               [A8 - A4]
  7. locked RSNA test, A8 versus the initial (mixed-precision) ConvNeXt V2 Tiny run
                                                        [A8 - initial ConvNeXt V2 Tiny]

Comparisons 4 and 5 are component-level sensitivity checks for the same-split comparison:
the upstream pretraining corpora of the two encoders include RSNA-derived images
(configs/specs/upstream_pretraining_data.csv), whereas the comparator was pretrained on ImageNet-1K.

ROC-AUC differences use the paired DeLong test; the other metrics use 2,000 paired
class-stratified percentile bootstrap resamples (numpy default_rng, seed 42, positive
class sampled first). Thresholds are the development-derived values (A8 0.470,
A4 0.490, DenseNet121-XRV 0.495, EVA-X-S 0.470, ConvNeXt V2 Tiny 0.455, initial ConvNeXt V2 Tiny
run 0.470); nothing is refitted or retuned.

    python scripts/analysis/paired_comparisons.py            # write to runs/paired_comparisons/
    python scripts/analysis/paired_comparisons.py --check    # also compare with the released results

Runs on a CPU from the CSV files in results/; no images or model weights are needed.
"""
from pathlib import Path
import argparse, math, sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, matthews_corrcoef, f1_score, brier_score_loss)

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / 'results/predictions'
ANALYSIS = ROOT / 'results/analysis'
METRICS = ['ROC_AUC', 'AUPRC', 'Sensitivity', 'Specificity', 'Accuracy', 'Balanced_Accuracy', 'F1', 'MCC', 'Brier']
N_BOOT, SEED = 2000, 42
BOOT = {}
TAU = {'A8': 0.47, 'A4': 0.49, 'XRV': 0.495, 'EVA_XS': 0.47, 'CONVNEXTV2': 0.455, 'CONVNEXTV2_INITIAL': 0.47}
INITIAL_RAW = 'contemporary_baseline/attempt1_amp_audit/initial_run_raw'
COMPONENT_JOBS = ('xrv_vs_convnextv2_tiny_locked_rsna', 'eva_x_s_vs_convnextv2_tiny_locked_rsna')
COMPONENT_FILE = 'contemporary_baseline/15_COMPONENT_VS_BASELINE.csv'


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'ROC_AUC': float(roc_auc_score(y, p)), 'AUPRC': float(average_precision_score(y, p)),
            'Sensitivity': float(tp / (tp + fn)), 'Specificity': float(tn / (tn + fp)),
            'Accuracy': float(accuracy_score(y, pred)), 'Balanced_Accuracy': float(balanced_accuracy_score(y, pred)),
            'F1': float(f1_score(y, pred, zero_division=0)), 'MCC': float(matthews_corrcoef(y, pred)),
            'Brier': float(brier_score_loss(y, p))}


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, float)
    out[order] = ranks
    return out


def paired_delong(y, p_new, p_ref):
    """Two-sided paired DeLong test for ROC-AUC(p_new) - ROC-AUC(p_ref)."""
    order = np.argsort(-y, kind='stable')
    m = int(y.sum())
    preds = np.vstack([p_new[order], p_ref[order]])
    n = preds.shape[1] - m
    tx = np.vstack([_midrank(r[:m]) for r in preds])
    ty = np.vstack([_midrank(r[m:]) for r in preds])
    tz = np.vstack([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    l = np.array([1.0, -1.0])
    d = float(l @ aucs)
    se = math.sqrt(max(float(l @ cov @ l), 0.0))
    p = float(2 * stats.norm.sf(abs(d / se))) if se else 1.0
    zc = float(stats.norm.ppf(0.975))
    return {'delta': d, 'ci_low': d - zc * se, 'ci_high': d + zc * se, 'p_value': p, 'se': se}


def paired_bootstrap(y, p_new, t_new, p_ref, t_ref, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rows = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        a, b = metrics(y[idx], p_new[idx], t_new), metrics(y[idx], p_ref[idx], t_ref)
        rows.append([a[k] - b[k] for k in METRICS])
    return pd.DataFrame(rows, columns=METRICS)


def compare(y, p_new, t_new, p_ref, t_ref):
    y = np.asarray(y, int)
    p_new, p_ref = np.asarray(p_new, float), np.asarray(p_ref, float)
    m_new, m_ref = metrics(y, p_new, t_new), metrics(y, p_ref, t_ref)
    dl = paired_delong(y, p_new, p_ref)
    boot = paired_bootstrap(y, p_new, t_new, p_ref, t_ref)
    rows = []
    for k in METRICS:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        auc = k == 'ROC_AUC'
        rows.append({'metric': k, 'reference_estimate': m_ref[k], 'new_estimate': m_new[k],
                     'delta': m_new[k] - m_ref[k],
                     'CI_low': dl['ci_low'] if auc else float(lo), 'CI_high': dl['ci_high'] if auc else float(hi),
                     'p_value': dl['p_value'] if auc else np.nan,
                     'method': 'paired_DeLong' if auc else 'paired_class_stratified_percentile_bootstrap'})
    return pd.DataFrame(rows), boot


def load_inputs():
    locked = pd.read_csv(PRED / 'locked_rsna.csv')
    kermany = pd.read_csv(PRED / 'kermany.csv')
    convnext = pd.read_csv(ANALYSIS / 'contemporary_baseline/06_BASELINE_LOCKED_PREDICTIONS.csv')
    merged = locked.merge(convnext[['patient_id', 'true_label', 'probability']], left_on='case_id',
                          right_on='patient_id', how='inner', suffixes=('', '_convnext'))
    assert len(merged) == len(locked) == 5338 and (merged.true_label == merged.true_label_convnext).all()
    assert len(kermany) == 624
    kermany = kermany.assign(a4_probability=(kermany.xrv_probability + kermany.eva_x_probability) / 2)
    initial = pd.read_csv(ANALYSIS / INITIAL_RAW / '06_BASELINE_LOCKED_PREDICTIONS.csv')
    merged_initial = locked.merge(initial[['patient_id', 'true_label', 'probability']], left_on='case_id',
                                  right_on='patient_id', how='inner', suffixes=('', '_convnext'))
    assert len(merged_initial) == 5338 and (merged_initial.true_label == merged_initial.true_label_convnext).all()
    return locked, kermany, merged, merged_initial


def run(out_dir):
    locked, kermany, merged, merged_initial = load_inputs()
    jobs = {
        'a8_vs_a4_locked_rsna': (locked.true_label, locked.a8_probability, TAU['A8'],
                                 locked.a4_probability, TAU['A4']),
        'a8_vs_eva_x_s_kermany': (kermany.true_label, kermany.a8_probability, TAU['A8'],
                                  kermany.eva_x_probability, TAU['EVA_XS']),
        'a8_vs_convnextv2_tiny_locked_rsna': (merged.true_label, merged.a8_probability, TAU['A8'],
                                              merged.probability, TAU['CONVNEXTV2']),
        'xrv_vs_convnextv2_tiny_locked_rsna': (merged.true_label, merged.xrv_probability, TAU['XRV'],
                                               merged.probability, TAU['CONVNEXTV2']),
        'eva_x_s_vs_convnextv2_tiny_locked_rsna': (merged.true_label, merged.eva_x_probability, TAU['EVA_XS'],
                                                   merged.probability, TAU['CONVNEXTV2']),
        'a8_vs_a4_kermany': (kermany.true_label, kermany.a8_probability, TAU['A8'],
                             kermany.a4_probability, TAU['A4']),
        'a8_vs_convnextv2_tiny_initial_amp_locked_rsna': (merged_initial.true_label, merged_initial.a8_probability,
                                                          TAU['A8'], merged_initial.probability,
                                                          TAU['CONVNEXTV2_INITIAL']),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, args in jobs.items():
        table, boot = compare(*args)
        BOOT[name] = boot
        table.to_csv(out_dir / f'{name}.csv', index=False)
        boot.to_csv(out_dir / f'{name}_bootstrap_differences.csv', index_label='bootstrap_index')
        results[name] = table
        print(f'\n{name}  (delta = new - reference)')
        for _, r in table.iterrows():
            p = '' if pd.isna(r.p_value) else f'  p = {r.p_value:.3f}'
            print(f'  {r.metric:<18}{r.delta:+.5f}  95% CI {r.CI_low:+.5f} to {r.CI_high:+.5f}{p}')
    pd.concat([results[n].assign(comparison=n) for n in COMPONENT_JOBS]).to_csv(
        out_dir / Path(COMPONENT_FILE).name, index=False)
    return results


def check(results):
    released = {
        'a8_vs_a4_locked_rsna': (ANALYSIS / 'a4_vs_a8/paired_results.csv', 'delta_A8_minus_A4'),
        'a8_vs_eva_x_s_kermany': (ANALYSIS / 'a4_vs_a8/external_secondary.csv', 'delta_A8_minus_EVA'),
        'a8_vs_convnextv2_tiny_locked_rsna': (ANALYSIS / 'contemporary_baseline/12_A8_BASELINE_PAIRED_COMPARISON.csv',
                                              'delta_A8_minus_baseline'),
        'a8_vs_a4_kermany': (ANALYSIS / 'a4_vs_a8/kermany_a8_vs_a4.csv', 'delta'),
        'a8_vs_convnextv2_tiny_initial_amp_locked_rsna': (ANALYSIS / INITIAL_RAW / '12_A8_BASELINE_PAIRED_COMPARISON.csv',
                                                          'delta_A8_minus_baseline'),
    }
    worst = 0.0
    for name, (path, delta_col) in released.items():
        ref = pd.read_csv(path).set_index('metric')
        new = results[name].set_index('metric')
        for k in METRICS:
            for a, b in ((new.at[k, 'delta'], ref.at[k, delta_col]), (new.at[k, 'CI_low'], ref.at[k, 'CI_low']),
                         (new.at[k, 'CI_high'], ref.at[k, 'CI_high'])):
                worst = max(worst, abs(float(a) - float(b)))
            if k == 'ROC_AUC':
                worst = max(worst, abs(float(new.at[k, 'p_value']) - float(ref.at[k, 'p_value'])))
        print(f'checked {name} against {path.relative_to(ROOT)}')
    comp = pd.read_csv(ANALYSIS / COMPONENT_FILE)
    for name in COMPONENT_JOBS:
        ref = comp[comp.comparison == name].set_index('metric')
        new = results[name].set_index('metric')
        for col in ('delta', 'CI_low', 'CI_high', 'new_estimate', 'reference_estimate'):
            worst = max(worst, float(np.abs(new[col] - ref.loc[new.index, col]).max()))
        worst = max(worst, abs(float(new.at['ROC_AUC', 'p_value']) - float(ref.at['ROC_AUC', 'p_value'])))
        print(f'checked {name} against results/analysis/{COMPONENT_FILE}')
    for name, raw in (('a8_vs_convnextv2_tiny_locked_rsna',
                       ANALYSIS / 'contemporary_baseline/fp32_run_raw/bootstrap_differences.csv'),
                      ('a8_vs_convnextv2_tiny_initial_amp_locked_rsna',
                       ANALYSIS / INITIAL_RAW / 'bootstrap_differences.csv')):
        if raw.is_file() and BOOT.get(name) is not None:
            ref = pd.read_csv(raw)
            new = BOOT[name]
            worst = max(worst, float(np.abs(ref[METRICS].to_numpy() - new[METRICS].to_numpy()).max()))
            print(f'checked the {name} bootstrap replicates against {raw.relative_to(ROOT)}')
    status = 'PASS' if worst < 1e-9 else 'FAIL'
    print(f'Largest difference from the released paired results: {worst:.1e}')
    print('PAIRED COMPARISONS =', status)
    return 0 if status == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=str(ROOT / 'runs/paired_comparisons'), help='output directory (untracked)')
    ap.add_argument('--check', action='store_true', help='compare with the released paired results')
    args = ap.parse_args()
    results = run(Path(args.out))
    return check(results) if args.check else 0


if __name__ == '__main__':
    sys.exit(main())

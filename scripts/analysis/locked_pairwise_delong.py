#!/usr/bin/env python3
"""Recompute the original locked-test pairwise ROC-AUC comparisons with Holm correction.

Family: all ten pairwise paired-DeLong comparisons among the five locked-test models
DenseNet121-XRV (A1), EVA-X-S (A2), LightGBM radiomics (A3), soft voting (A7) and the
frozen stack (A8), Holm-adjusted across the ten tests, as in the original analysis
(scripts/q1_stats_report.py). The manuscript (Section 5.2) reports the A8-versus-A1 and
A8-versus-A2 rows.

    python scripts/analysis/locked_pairwise_delong.py            # write to runs/locked_pairwise_delong/
    python scripts/analysis/locked_pairwise_delong.py --check    # also compare with the released result

Runs on a CPU from results/predictions/test_predictions_full.csv.
"""
from pathlib import Path
import argparse, importlib.util, itertools, sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASED = ROOT / 'results/analysis/locked_pairwise_delong/pairwise_delong_holm.csv'
MODELS = {'A1_DenseNet121-XRV': 'xrv_prob', 'A2_EVA-X-S': 'eva_x_prob', 'A3_LightGBM': 'radiomics_prob',
          'A7_soft_voting': 'soft_prob', 'A8_frozen_stack': 'stacked_prob'}

spec = importlib.util.spec_from_file_location('pc', ROOT / 'scripts/analysis/paired_comparisons.py')
pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i])); adj[i] = running
    return adj


def compute():
    d = pd.read_csv(ROOT / 'results/predictions/test_predictions_full.csv')
    y = d['label'].to_numpy(int)
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        r = pc.paired_delong(y, d[MODELS[b]].to_numpy(float), d[MODELS[a]].to_numpy(float))
        rows.append({'model_new': b, 'model_ref': a, 'delta_new_minus_ref': r['delta'],
                     'ci_low': r['ci_low'], 'ci_high': r['ci_high'], 'p_raw': r['p_value']})
    out = pd.DataFrame(rows)
    out['p_holm_10'] = holm(out['p_raw'])
    out['method'] = 'paired_DeLong; Holm across the 10 pairwise comparisons'
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); args = ap.parse_args()
    out = compute()
    dst = ROOT / 'runs/locked_pairwise_delong'; dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / 'pairwise_delong_holm.csv', index=False)
    for r in out.itertuples():
        print(f'{r.model_new:>20} - {r.model_ref:<20} {r.delta_new_minus_ref:+.5f}  p = {r.p_raw:.3g}  Holm p = {r.p_holm_10:.3g}')
    if args.check:
        ref = pd.read_csv(RELEASED)
        diff = np.abs(ref[['delta_new_minus_ref', 'p_raw', 'p_holm_10']].to_numpy() -
                      out[['delta_new_minus_ref', 'p_raw', 'p_holm_10']].to_numpy()).max()
        print(f'Largest difference from the released result: {diff:.1e}')
        if diff > 1e-9:
            print('LOCKED PAIRWISE DELONG = FAIL'); sys.exit(1)
        print('LOCKED PAIRWISE DELONG = PASS')


if __name__ == '__main__':
    main()

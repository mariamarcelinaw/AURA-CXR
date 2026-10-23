"""CPU smoke test of the frozen A8 inference path (no images or weights needed).

The frozen stack in src/aura_cxr/fusion.py is applied to the released component probabilities
of the locked RSNA test and the Kermany cohort; the resulting probabilities, decisions and
headline metrics must equal the released per-case outputs and reported metrics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from aura_cxr.fusion import A8_THRESHOLD, a4_probability, a8_probability, assert_frozen_a8, threshold

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_output_provenance_present():
    json.loads((ROOT / 'results/audit/golden_reference.json').read_text())
    c = json.loads((ROOT / 'configs/checkpoints.json').read_text())
    assert c['xrv']['sha256'] and c['eva_x']['sha256'] and c['radiomics']['sha256']
    assert (ROOT / 'results/predictions/locked_rsna.csv').exists()


def test_frozen_a8_inference_locked_rsna():
    assert_frozen_a8()
    d = pd.read_csv(ROOT / 'results/predictions/locked_rsna.csv')
    p = a8_probability(d.xrv_probability, d.eva_x_probability, d.radiomics_probability)
    assert np.abs(p - d.a8_probability).max() < 1e-6
    assert (threshold(p, A8_THRESHOLD) == d.a8_prediction).all()
    assert np.abs(a4_probability(d.xrv_probability, d.eva_x_probability) - d.a4_probability).max() < 1e-6
    rep = json.loads((ROOT / 'results/metrics/reported_metrics.json').read_text())
    assert abs(roc_auc_score(d.true_label, p) - rep['internal_A8']['AUC']) < 1e-6


def test_frozen_a8_inference_kermany():
    k = pd.read_csv(ROOT / 'results/predictions/kermany.csv')
    p = a8_probability(k.xrv_probability, k.eva_x_probability, k.radiomics_probability)
    assert np.abs(p - k.a8_probability).max() < 1e-6
    yhat = threshold(p, A8_THRESHOLD); y = k.true_label.to_numpy()
    rep = json.loads((ROOT / 'results/metrics/reported_metrics.json').read_text())['external_A8']
    assert [int(((yhat == a) & (y == b)).sum()) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))] == \
        [rep['tn'], rep['fp'], rep['fn'], rep['tp']]
    assert abs(roc_auc_score(y, p) - rep['roc_auc']) < 5e-5

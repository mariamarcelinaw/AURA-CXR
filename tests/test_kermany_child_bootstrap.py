import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/kermany_child_bootstrap'


def test_child_counts_and_single_label():
    d = pd.read_csv(ROOT / 'results/predictions/kermany.csv')
    name = d.relative_path.str.split('/').str[-1]
    child = name.str.extract(r'^(person\d+)_')[0].fillna(name.str.extract(r'^((?:NORMAL2-)?IM-\d+)-')[0])
    assert child.notna().all()
    assert (d.groupby(child).true_label.nunique() == 1).all()
    s = json.loads((OUT / 'summary.json').read_text())
    assert (s['images'], s['children'], s['pneumonia_children'], s['normal_children']) == (624, 427, 202, 225)


def test_point_estimates_match_released_predictions():
    d = pd.read_csv(ROOT / 'results/predictions/kermany.csv')
    res = pd.read_csv(OUT / 'kermany_child_bootstrap_metrics.csv').set_index('model')
    a8 = res.loc['A8_frozen_stack']
    assert abs(a8.roc_auc - roc_auc_score(d.true_label, d.a8_probability)) < 1e-12
    assert round(a8.roc_auc_child_ci_low, 4) == 0.9081 and round(a8.roc_auc_child_ci_high, 4) == 0.9514
    for _, r in res.iterrows():
        assert r.roc_auc_child_ci_low < r.roc_auc < r.roc_auc_child_ci_high

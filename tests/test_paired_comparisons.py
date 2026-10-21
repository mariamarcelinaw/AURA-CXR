import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location('paired_comparisons', ROOT / 'scripts/analysis/paired_comparisons.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_delong_matches_released_a8_vs_a4():
    mod = _load()
    d = pd.read_csv(ROOT / 'results/predictions/locked_rsna.csv')
    res = mod.paired_delong(d.true_label.to_numpy(int), d.a8_probability.to_numpy(float), d.a4_probability.to_numpy(float))
    ref = pd.read_csv(ROOT / 'results/analysis/a4_vs_a8/paired_results.csv').set_index('metric').loc['ROC_AUC']
    assert abs(res['delta'] - ref['delta_A8_minus_A4']) < 1e-12
    assert abs(res['p_value'] - ref['p_value']) < 1e-12
    assert abs(res['ci_low'] - ref['CI_low']) < 1e-9 and abs(res['ci_high'] - ref['CI_high']) < 1e-9


def test_component_vs_baseline_matches_released():
    mod = _load()
    locked, _, merged, _ = mod.load_inputs()
    comp = pd.read_csv(ROOT / 'results/analysis/contemporary_baseline/15_COMPONENT_VS_BASELINE.csv')
    for name, col in (('xrv_vs_convnextv2_tiny_locked_rsna', 'xrv_probability'),
                      ('eva_x_s_vs_convnextv2_tiny_locked_rsna', 'eva_x_probability')):
        res = mod.paired_delong(merged.true_label.to_numpy(int), merged[col].to_numpy(float),
                                merged.probability.to_numpy(float))
        ref = comp[(comp.comparison == name) & (comp.metric == 'ROC_AUC')].iloc[0]
        assert abs(res['delta'] - ref['delta']) < 1e-12 and abs(res['p_value'] - ref['p_value']) < 1e-12


def test_a8_vs_a4_kermany_matches_released():
    mod = _load()
    _, kermany, _, _ = mod.load_inputs()
    res = mod.paired_delong(kermany.true_label.to_numpy(int), kermany.a8_probability.to_numpy(float),
                            kermany.a4_probability.to_numpy(float))
    ref = pd.read_csv(ROOT / 'results/analysis/a4_vs_a8/kermany_a8_vs_a4.csv').set_index('metric').loc['ROC_AUC']
    assert abs(res['delta'] - ref['delta']) < 1e-12 and abs(res['p_value'] - ref['p_value']) < 1e-12
    assert res['ci_high'] < 0


def test_initial_comparator_run_matches_record():
    mod = _load()
    _, _, _, merged_initial = mod.load_inputs()
    res = mod.paired_delong(merged_initial.true_label.to_numpy(int), merged_initial.a8_probability.to_numpy(float),
                            merged_initial.probability.to_numpy(float))
    ref = pd.read_csv(ROOT / 'results/analysis/contemporary_baseline/attempt1_amp_audit/initial_run_raw/'
                      '12_A8_BASELINE_PAIRED_COMPARISON.csv').set_index('metric').loc['ROC_AUC']
    assert abs(res['delta'] - ref['delta_A8_minus_baseline']) < 1e-12
    assert abs(res['p_value'] - ref['p_value']) < 1e-9

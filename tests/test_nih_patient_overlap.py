import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/nih_patient_overlap'
MODELS = {'a8': 'a8', 'a4': 'a4', 'xrv': 'a1', 'eva_x': 'a2', 'radiomics': 'a3', 'a5': 'a5', 'a6': 'a6',
          'a7': 'a7', 'a9': 'a9', 'a10': 'a10'}


def _cases():
    return pd.read_csv(OUT / 'locked_test_nih_patients.csv', dtype={'nih_patient': str})


def test_overlap_counts():
    cases, summary = _cases(), json.loads((OUT / 'summary.json').read_text())
    test = pd.read_csv(ROOT / 'data/splits/test.csv')
    assert set(cases.patientId) == set(test.patientId) and len(cases) == 5338
    assert int(cases.shared.sum()) == summary['locked_test_images_shared_patient'] == 3738
    assert cases[~cases.shared].nih_patient.nunique() == summary['nih_patients_locked_test'] - summary['nih_patients_shared']


def test_auc_rescoring_all_configurations():
    pred = pd.read_csv(ROOT / 'results/predictions/locked_rsna_a1_a10.csv').merge(
        _cases(), left_on='case_id', right_on='patientId')
    cnx = pd.read_csv(ROOT / 'results/analysis/contemporary_baseline/06_BASELINE_LOCKED_PREDICTIONS.csv')
    pred = pred.merge(cnx[['patient_id', 'probability']], left_on='case_id', right_on='patient_id')
    ref = pd.read_csv(OUT / 'nih_patient_overlap_metrics.csv').set_index(['subset', 'model']).roc_auc
    for name, d in (('all locked test', pred), ('no NIH patient in development', pred[~pred.shared]),
                    ('NIH patient also in development', pred[pred.shared])):
        for model, col in MODELS.items():
            assert abs(roc_auc_score(d.true_label, d[f'{col}_probability']) - ref[(name, model)]) < 1e-12
        assert abs(roc_auc_score(d.true_label, d.probability) - ref[(name, 'convnextv2')]) < 1e-12


def test_full_set_delong_matches_section_5_2():
    got = pd.read_csv(OUT / 'nih_patient_overlap_delong.csv')
    got = got[(got.subset == 'all locked test') & (got.family.str.startswith('Section'))]
    ref = pd.read_csv(ROOT / 'results/analysis/locked_pairwise_delong/pairwise_delong_holm.csv')
    assert len(got) == len(ref) == 10
    assert np.allclose(np.sort(got.delta_auc), np.sort(ref.delta_new_minus_ref), atol=1e-12)
    assert np.allclose(np.sort(got.p_holm), np.sort(ref.p_holm_10), rtol=1e-9, atol=1e-300)


def test_development_selection_on_single_fold_patients():
    dev = pd.read_csv(OUT / 'development_nih_patients.csv', dtype={'nih_patient': str})
    summary = json.loads((OUT / 'summary.json').read_text())
    folds = pd.read_csv(ROOT / 'data/splits/development_oof_folds.csv')
    assert set(dev.patientId) == set(folds.patientId) and len(dev) == 21346
    assert (dev.groupby('nih_patient').oof_fold.nunique().eq(1)
            == dev.groupby('nih_patient').single_fold.all()).all()
    assert int(dev.single_fold.sum()) == summary['development_images_patient_within_one_oof_fold']
    sel = pd.read_csv(OUT / 'nih_patient_overlap_development_selection.csv')
    for subset in ('all development', 'NIH patient within one OOF fold'):
        pairs = sel[(sel.subset == subset) & (sel.kind == 'pair stack')].sort_values('rank_within_kind')
        assert pairs.model.iloc[0] == 'xrv+eva_x+radiomics_lgbm'
        assert pairs.fold_auc_mean.iloc[0] - pairs.fold_auc_mean.iloc[1] > 0.002
    full = sel[(sel.subset == 'all development') & (sel.kind == 'pair stack')]
    assert abs(full[full.model == 'xrv+eva_x+radiomics_lgbm'].fold_auc_mean.iloc[0] - 0.8916) < 5e-5

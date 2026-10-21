#!/usr/bin/env python3
"""AURA-CXR command line.

    python aura_cxr.py verify      check configs, splits, predictions, analyses and file integrity
    python aura_cxr.py reproduce   recompute the reported A4/A8 metrics from the per-case predictions
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, re, sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

SPLITS = ROOT / 'data/splits'
SPECS = ROOT / 'configs/specs'
PRED = ROOT / 'results/predictions'
METRICS = ROOT / 'results/metrics'
ANALYSIS = ROOT / 'results/analysis'
AUDIT = ROOT / 'results/audit'

REQUIRED = [
    'README.md', 'LICENSE', 'requirements.txt', 'SHA256SUMS.txt',
    'configs/frozen_config.yaml', 'configs/checkpoints.json', 'configs/seeds.json', 'configs/xai_frozen.json',
    'configs/specs/deployment_manifest.json', 'configs/specs/deep_training_specification.json',
    'configs/specs/radiomics_specification.json', 'configs/specs/feature_schema_746.csv',
    'configs/specs/mrfo_search_space.csv',
    'data/splits/train.csv', 'data/splits/val.csv', 'data/splits/test.csv', 'data/splits/development.csv',
    'data/splits/development_oof_folds.csv', 'data/splits/manifest.json',
    'results/metrics/reported_metrics.json', 'results/metrics/result_registry.csv',
    'results/predictions/locked_rsna.csv', 'results/predictions/kermany.csv',
    'results/predictions/test_predictions_full.csv',
    'results/predictions/external_validation_kermany_predictions.csv',
    'results/audit/golden_reference.json', 'results/audit/provenance.json',
]
# Files that are produced locally and never tracked.
UNTRACKED_DIRS = {'.git', '__pycache__', '.pytest_cache', 'runs', 'checkpoints'}
PROHIBITED_EXT = {'.dcm', '.dicom', '.pt', '.pth', '.ckpt', '.h5', '.keras', '.onnx', '.safetensors',
                  '.npy', '.npz', '.pkl', '.joblib'}
MIB = 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tracked_files():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or UNTRACKED_DIRS & set(rel.parts) or rel.parts[:2] == ('data', 'raw'):
            continue
        yield rel.as_posix(), p


def read_json(rel):
    return json.loads((ROOT / rel).read_text())


# ---------------------------------------------------------------- verification checks
# Each check returns a short detail string on success and raises AssertionError on failure.

def check_required():
    missing = [r for r in REQUIRED if not (ROOT / r).is_file()]
    assert not missing, 'missing ' + ', '.join(missing)
    return f'{len(REQUIRED)} files'


def check_integrity():
    recorded = {}
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        if line.strip():
            digest, rel = line.split('  ', 1)
            recorded[rel.strip()] = digest.strip()
    present = {rel for rel, _ in tracked_files()} - {'SHA256SUMS.txt'}
    missing = sorted(r for r in recorded if not (ROOT / r).is_file())
    unlisted = sorted(present - set(recorded))
    changed = sorted(r for r in recorded if (ROOT / r).is_file() and sha256(ROOT / r) != recorded[r])
    problems = [f'{len(x)} {k}' + (f' ({", ".join(x[:3])})' if x else '')
                for k, x in (('missing', missing), ('unlisted', unlisted), ('changed', changed)) if x]
    assert not problems, '; '.join(problems)
    return f'{len(recorded)} files'


def check_config():
    from aura_cxr.config import load_config
    from aura_cxr.fusion import assert_frozen_a8, A8_INTERCEPT, A8_COEFFICIENTS, A8_THRESHOLD, A8_ORDER
    cfg = load_config(ROOT / 'configs/frozen_config.yaml')
    assert (ROOT / cfg['radiomics']['mrfo']['search_space']).is_file(), 'MRFO search space path does not resolve'
    assert_frozen_a8()
    dep = read_json('configs/specs/deployment_manifest.json')
    gold = read_json('results/audit/golden_reference.json')['frozen_a8']
    assert dep['meta_feature_order'] == list(A8_ORDER) == gold['meta_feature_order']
    assert dep['LR_intercept'] == A8_INTERCEPT == gold['coefficients']['intercept']
    assert list(dep['LR_coefficients'].values()) == A8_COEFFICIENTS.tolist() == [
        gold['coefficients'][k] for k in ('beta_xrv', 'beta_eva_x', 'beta_radiomics')]
    assert dep['decision_threshold'] == A8_THRESHOLD == gold['threshold']
    assert dep['deployment_lock_id'] == gold['deployment_lock_id']
    fz = cfg['fusion']
    assert (fz['meta_feature_order'] == dep['meta_feature_order'] and fz['intercept'] == dep['LR_intercept']
            and fz['coefficients'] == dep['LR_coefficients'] and fz['threshold'] == dep['decision_threshold']), \
        'frozen_config.yaml fusion values differ from the deployment manifest'
    ck = read_json('configs/checkpoints.json')
    for k in ('xrv', 'eva_x', 'radiomics'):
        assert ck[k]['sha256'] == dep['checkpoint_hashes'][k] == gold['checkpoint_hashes'][k], f'{k} checkpoint hash'
    return 'frozen A8 stack, threshold and checkpoint hashes consistent'


def check_splits():
    import pandas as pd
    from aura_cxr.data import assert_split_integrity
    tr, va, te = (pd.read_csv(SPLITS / f'{n}.csv') for n in ('train', 'val', 'test'))
    folds = pd.read_csv(SPLITS / 'development_oof_folds.csv')
    assert_split_integrity(tr, va, te, folds)
    dev = pd.read_csv(SPLITS / 'development.csv')
    assert set(dev.patientId) == set(tr.patientId) | set(va.patientId), 'development != train + val'
    for f in read_json('data/splits/manifest.json')['files']:
        assert sha256(SPLITS / f['filename']) == f['sha256'], f'{f["filename"]} hash differs from manifest'
    return f'train {len(tr)}, val {len(va)}, test {len(te)}, OOF folds {sorted(int(f) for f in folds.oof_fold.unique())}'


def check_predictions():
    import numpy as np, pandas as pd
    from aura_cxr.fusion import a8_probability
    expected = {'test_predictions_full.csv': (5338, 1203), 'locked_rsna.csv': (5338, 1203),
                'external_validation_kermany_predictions.csv': (624, 390), 'kermany.csv': (624, 390)}
    for name, (n, pos) in expected.items():
        d = pd.read_csv(PRED / name)
        lab = 'label' if 'label' in d.columns else 'true_label'
        assert len(d) == n and int(d[lab].sum()) == pos, f'{name}: {len(d)} rows, {int(d[lab].sum())} positive'
        assert not d.filter(like='prob').isna().any().any(), f'{name}: missing probabilities'
    for name in ('locked_rsna.csv', 'kermany.csv'):
        d = pd.read_csv(PRED / name)
        diff = np.abs(a8_probability(d.xrv_probability, d.eva_x_probability, d.radiomics_probability) - d.a8_probability).max()
        assert diff < 1e-6, f'{name}: A8 column differs from the frozen stack by {diff:.2e}'
    return 'RSNA 5338/1203, Kermany 624/390; A8 = frozen stack of components'


def check_deep_config():
    sp = read_json('configs/specs/deep_training_specification.json')
    ck = read_json('configs/checkpoints.json')
    m = sp['models']
    assert set(m) == {'efficientnetv2s', 'resnet50', 'densenet121_xrv', 'eva_x_s'}
    assert all('input' in v for v in m.values())
    assert ck['xrv']['upstream_checkpoint_identifier'] == 'densenet121-res224-all'
    assert 'merged520k' in ck['eva_x']['upstream_checkpoint_identifier']
    assert m['densenet121_xrv']['loss']['gamma'] == 2.0
    assert len(m['efficientnetv2s']['augmentation']) >= 5 and m['densenet121_xrv']['augmentation'] == []
    assert m['densenet121_xrv']['optimizer']['name'] == 'AdamW'
    assert m['densenet121_xrv']['scheduler']['name'] == 'CosineAnnealingLR'
    assert len(m['efficientnetv2s']['training_phases']) == 2
    assert sp['inner_validation']['patient_disjoint'] is True and sp['inner_validation']['seeds'] == [42, 43, 44, 45, 46]
    assert m['densenet121_xrv']['early_stopping']['patience'] == 12
    assert m['densenet121_xrv']['final_refit']['derived_epoch_phase1'] == 2
    assert m['eva_x_s']['final_refit']['derived_epoch_phase1'] == 2
    assert sp['seeds']['global'] == 42
    folds = re.findall(r'"fold_manifest": "([^"]+)"', (SPECS / 'deep_training_specification.json').read_text())
    assert folds and all((ROOT / f).is_file() for f in folds), 'fold manifest path does not resolve'
    return '4 backbones, loss, optimizers, schedules, refit rules'


def check_radiomics():
    s = read_json('configs/specs/radiomics_specification.json')
    rows = list(csv.DictReader((SPECS / 'feature_schema_746.csv').open()))
    mrfo = list(csv.DictReader((SPECS / 'mrfo_search_space.csv').open()))
    assert s['preprocessing']['internal_rsna']['resize'] == [224, 224]
    assert s['wavelets']['candidates'] == ['haar', 'db4', 'sym4', 'coif3']
    assert s['entropy']['log_base'] == 2 and s['entropy']['epsilon'] == 1e-12
    assert s['entropy']['tie_break'].startswith('strict >')
    assert s['wavelets']['extension_mode']['effective'] == 'symmetric'
    assert s['decomposition']['subband_order'] == ['LH1', 'HL1', 'HH1', 'LH2', 'HL2', 'HH2', 'LH3', 'HL3', 'HH3', 'LL3']
    assert s['glcm']['levels'] == 256 and s['lbp']['points'] == 24 and s['lbp']['bins'] == 26
    assert len(rows) == 746, 'feature count'
    assert sha256(SPECS / 'feature_schema_746.csv') == s['feature_schema']['schema_sha256'], 'feature schema hash'
    assert s['lightgbm']['full_development_effective_params']['max_depth'] == -1
    assert len([x for x in mrfo if x['estimator'] == 'lgbm']) == 5 and s['mrfo']['final']['population'] == 15
    import numpy as np
    from aura_cxr.radiomics import select_best_wavelet, extract_radiomics
    x = np.zeros((224, 224), np.float32)
    assert select_best_wavelet(x) == 'haar' and extract_radiomics(x)[0].shape == (746,), 'feature extraction'
    return '746 features, schema hash, MRFO space, live extraction'


def check_xai():
    import pandas as pd
    from aura_cxr.xai import assert_xai_config
    assert_xai_config()
    spec = read_json('configs/xai_frozen.json')
    assert spec['checkpoints']['xrv_sha256'] and spec['checkpoints']['eva_sha256'] and spec['checkpoints']['unchanged_after_run']
    assert len(spec['xrv']['hooks']) == 2 and spec['xrv']['activation_shapes'] == ['[1, 128, 7, 7]'] * 2
    assert len(spec['eva']['hooks']) == 2 and spec['eva']['activation_shapes'] == ['[1, 197, 384]'] * 2
    t = spec['eva']['token_structure']
    assert t['num_patch_tokens'] == 196 and t['num_prefix_tokens'] == 1 and t['class_token'] is True and t['other_prefix_tokens'] == 0
    assert t['ordering'] == 'row-major raster' and t['grid'] == [14, 14] and t['patch_size'] == [16, 16]
    assert spec['target']['class_index'] == 1 and spec['target']['scalar'] == 'pre-softmax class-1 logit'
    assert spec['fusion']['layer_weights'] == [0.75, 0.25]
    assert spec['fusion']['xrv_weight'] == 0.25 and spec['fusion']['eva_weight'] == 0.75
    assert spec['postprocessing']['map_policy'] == 'raw' and abs(spec['postprocessing']['threshold'] - 0.25) < 1e-12
    f = spec['faithfulness']
    assert f['sample_n'] == 256 and f['top_fraction'] == 0.10 and f['explains_final_A8'] is False
    sc = spec['scope']
    assert sc['xai_scope_constant'] == 'HYBRID_CNN_VIT_BRANCH_ONLY'
    assert not sc['explains_radiomics'] and not sc['explains_meta_learner'] and not sc['explains_final_A8']
    cmp_ = pd.read_csv(ANALYSIS / 'transformer_xai/comparator_results.csv')
    case = pd.read_csv(ANALYSIS / 'transformer_xai/case_level_sensitivity.csv')
    assert len(case) == 1203 and case.eva_ig_valid.astype(bool).all()
    assert len(cmp_[cmp_.analysis_level == 'EVA_branch']) == 4
    return 'hooks, token grid, fusion weights, branch scope, 1203 comparator cases'


def check_analyses():
    import numpy as np, pandas as pd
    from aura_cxr.evaluation import binary_metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    # contemporary same-split baseline (ConvNeXt V2 Tiny)
    cb = ANALYSIS / 'contemporary_baseline'
    oof = pd.read_csv(cb / '08_BASELINE_OOF_PREDICTIONS.csv')
    locked = pd.read_csv(cb / '06_BASELINE_LOCKED_PREDICTIONS.csv')
    lock = json.loads((cb / '05_BASELINE_DEPLOYMENT_LOCK.json').read_text())
    res = pd.read_csv(cb / '11_BASELINE_LOCKED_RESULTS.csv').set_index('metric').point_estimate
    assert len(oof) == 21346 and oof.patient_id.nunique() == 21346 and oof.probability.between(0, 1).all()
    assert len(locked) == 5338 and locked.patient_id.nunique() == 5338 and int(locked.true_label.sum()) == 1203
    assert lock['threshold'] == 0.455 and np.allclose(locked.threshold, 0.455)
    assert lock['attempt_id'] == 'STAGE8_CORRECTED_FP32_V2'
    m = binary_metrics(locked.true_label, locked.probability, 0.455)
    assert abs(m['AUC'] - res['ROC_AUC']) < 1e-12 and abs(m['AUPRC'] - res['AUPRC']) < 1e-12, 'baseline AUC/AUPRC'
    # Kermany external validation
    ext = pd.read_csv(PRED / 'external_validation_kermany_predictions.csv')
    rep = pd.read_csv(ANALYSIS / 'external_validation/02_KERMANY_REPRODUCTION.csv').set_index('metric').canonical
    y, p = ext.label.to_numpy(int), ext.stacked_selected_pair_probability.to_numpy(float)
    assert ext.sample_id.nunique() == 624 and int((y == 0).sum()) == 234
    assert abs(roc_auc_score(y, p) - rep['ROC_AUC']) < 1e-12 and abs(average_precision_score(y, p) - rep['AUPRC']) < 1e-12
    k = binary_metrics(y, p, 0.47)
    assert (k['TN'], k['FP'], k['FN'], k['TP']) == (217, 17, 80, 310), 'Kermany confusion matrix'
    # XAI statistical families
    xs = ANALYSIS / 'xai_statistics'
    fa = pd.read_csv(xs / '02_FAMILY_A_LOCALIZATION_BASELINE_RESULTS.csv')
    fb = pd.read_csv(xs / '15_FAMILY_B_TP_FN_RESULTS.csv')
    fc = pd.read_csv(xs / '16_FAMILY_C_TRANSFORMER_SENSITIVITY_RESULTS.csv')
    claim = json.loads((xs / '13_XAI_CLAIM_LOCK.json').read_text())
    assert len(fa) == 16 and fa.comparison_id.nunique() == 16 and fa.p_Holm.notna().all()
    assert int(fa.Holm_significant.sum()) == claim['localization_baseline_family']['holm_significant']
    assert pd.read_csv(xs / '05_ORIGINAL_XAI_CONTRAST_REPRODUCTION.csv').reproduction_pass.all()
    assert len(fb) == claim['tp_fn_family']['size'] and fb.p_Holm.notna().all()
    assert fc.holm_verification_pass.all()
    assert claim['spatial_xai_scope'] == 'hybrid_cnn_vit_branch_only'
    return 'contemporary baseline, Kermany reproduction, XAI families A/B/C'


def check_initial_run():
    import math
    import pandas as pd
    from scipy import stats
    base = ANALYSIS / 'contemporary_baseline'
    rec = json.loads((base / 'attempt1_amp_audit/ATTEMPT_1_AUDIT_SUMMARY.json').read_text())
    t, pr = rec['locked_test'], rec['paired_A8_vs_baseline']
    assert (t['n'], t['positive'], t['negative']) == (5338, 1203, 4135), 'initial-run cohort'
    audit = pd.read_csv(base / 'ATTEMPT1_VS_CORRECTED_AUDIT.csv').set_index('item')
    assert abs(audit.loc['locked_test_ROC_AUC', 'attempt1_AMP'] - t['auc']) < 1e-12, 'initial-run ROC-AUC'
    a8_auc = read_json('results/metrics/reported_metrics.json')['internal_A8']['AUC']
    d = pr['delta_auc_A8_minus_baseline']
    assert abs((a8_auc - t['auc']) - d) < 1e-9, 'A8 minus initial-run ROC-AUC'
    se = (pr['ci_high'] - pr['ci_low']) / (2 * stats.norm.ppf(0.975))
    assert abs((pr['ci_high'] + pr['ci_low']) / 2 - d) < 1e-9, 'CI centred on difference'
    assert abs(2 * stats.norm.sf(abs(d / se)) - pr['p_value']) < 1e-9, 'p-value implied by CI'
    tp, tn = round(t['sensitivity'] * 1203), round(t['specificity'] * 4135)
    fn, fp = 1203 - tp, 4135 - tn
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert abs((tp + tn) / 5338 - t['accuracy']) < 1e-12 and abs(mcc - t['mcc']) < 1e-12, 'confusion-matrix metrics'
    assert abs(2 * tp / (2 * tp + fp + fn) - t['f1']) < 1e-12, 'F1'
    raw = base / 'attempt1_amp_audit/initial_run_raw'
    from sklearn.metrics import roc_auc_score
    pc = pd.read_csv(raw / '06_BASELINE_LOCKED_PREDICTIONS.csv')
    assert len(pc) == 5338 and int(pc.true_label.sum()) == 1203, 'initial-run per-case predictions'
    assert abs(roc_auc_score(pc.true_label, pc.probability) - t['auc']) < 1e-12, 'initial-run ROC-AUC from per-case predictions'
    lock = json.loads((raw / '05_BASELINE_DEPLOYMENT_LOCK.json').read_text())
    assert sha256(raw / '08_BASELINE_OOF_PREDICTIONS.csv') == lock['oof_sha256'], 'initial-run OOF hash'
    return (f'initial AMP run ROC-AUC {t["auc"]:.4f} (recomputed from per-case predictions); '
            f'A8 - initial {d:.4f}, p = {pr["p_value"]:.3f} (record consistent)')


def check_provenance():
    prov = read_json('results/audit/provenance.json')
    refs = set()
    for r in prov['relationships']:
        for k in ('input', 'output'):
            v = r[k]
            if '/' in v and '*' not in v:
                refs.add(v)
    dangling = sorted(r for r in refs if not (ROOT / r).is_file())
    assert not dangling, 'unresolved ' + ', '.join(dangling)
    bad = [f['path'] for f in prov['source_equivalence']['files'] if sha256(ROOT / f['path']) != f['released_sha256']]
    assert not bad, 'source hash differs: ' + ', '.join(bad)
    return f'{len(refs)} paths resolve; {len(prov["source_equivalence"]["files"])} source hashes match'


def check_credentials():
    cred = re.compile(r'(?i)(api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*["\']?[A-Za-z0-9_\-]{16,}')
    priv = re.compile(r'(/drive0-storage/|/mnt/data/|/home/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\)')
    hits = []
    for rel, p in tracked_files():
        if rel == 'aura_cxr.py' or p.suffix.lower() not in {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.csv'}:
            continue  # this file holds the patterns themselves
        t = p.read_text(encoding='utf-8', errors='ignore')
        if cred.search(t) or priv.search(t):
            hits.append(rel)
    assert not hits, ', '.join(hits[:5])
    return 'no credentials or machine-specific paths'


def check_prohibited():
    bad, large = [], []
    for rel, p in tracked_files():
        if p.suffix.lower() in PROHIBITED_EXT:
            bad.append(rel)
        size = p.stat().st_size
        if size > 100 * MIB:
            bad.append(f'{rel} (>100 MiB)')
        elif size > 50 * MIB:
            large.append(rel)
    assert not bad, ', '.join(bad[:5])
    return 'no raw images or weight files' + (f'; large: {", ".join(large)}' if large else '')


def check_weights():
    """Released trained checkpoints (GitHub release v1.0) placed in checkpoints/ are optional;
    when present, each must match the SHA-256 recorded in configs/checkpoints.json."""
    ck = read_json('configs/checkpoints.json')
    listed = dict(line.split()[::-1] for line in (ROOT / 'configs/release_weights.sha256').read_text().splitlines() if line.strip())
    found, wrong = [], []
    for k in ('xrv', 'eva_x', 'radiomics'):
        name = ck[k]['local_filename']
        assert listed.get(name) == ck[k]['sha256'], f'{name}: release_weights.sha256 differs from checkpoints.json'
        path = ROOT / 'checkpoints' / name
        if path.is_file():
            found.append(name)
            if sha256(path) != ck[k]['sha256']:
                wrong.append(name)
    assert not wrong, 'SHA-256 mismatch: ' + ', '.join(wrong)
    return (f'{len(found)}/3 trained checkpoints in checkpoints/ match the recorded SHA-256' if found
            else 'not downloaded (optional; GitHub release v1.0, hashes in configs/release_weights.sha256)')


CHECKS = [
    ('REQUIRED FILES', check_required),
    ('INTEGRITY', check_integrity),
    ('FROZEN CONFIG', check_config),
    ('SPLITS', check_splits),
    ('PREDICTIONS', check_predictions),
    ('DEEP CONFIG', check_deep_config),
    ('RADIOMICS', check_radiomics),
    ('XAI', check_xai),
    ('ANALYSES', check_analyses),
    ('INITIAL RUN', check_initial_run),
    ('PROVENANCE', check_provenance),
    ('CREDENTIALS', check_credentials),
    ('PROHIBITED FILES', check_prohibited),
    ('WEIGHTS', check_weights),
]


def verify():
    failed = 0
    for name, fn in CHECKS:
        try:
            print(f'{name:<17} PASS  {fn()}')
        except Exception as e:  # report every check, then fail once
            failed += 1
            print(f'{name:<17} FAIL  {type(e).__name__}: {e}')
    print('Raw images are not part of the repository. Trained weights are GitHub release v1.0 assets; their SHA-256 values are in configs/checkpoints.json.')
    print('VERIFY =', 'FAIL' if failed else 'PASS')
    return 1 if failed else 0


# ---------------------------------------------------------------- reproduction

KEYS = ['AUC', 'AUPRC', 'sensitivity', 'specificity', 'accuracy', 'balanced_acc', 'MCC']


def reproduce():
    import pandas as pd
    from aura_cxr.evaluation import reproduce_locked, reproduce_external
    AUDIT.mkdir(parents=True, exist_ok=True)
    internal = reproduce_locked(PRED / 'locked_rsna.csv')
    external = reproduce_external(PRED / 'kermany.csv')
    pd.DataFrame(internal).T.to_csv(AUDIT / 'internal_metrics.csv')
    pd.DataFrame(external).T.to_csv(AUDIT / 'external_metrics.csv')
    fs = json.loads((METRICS / 'xai_faithfulness_summary.json').read_text())
    pd.DataFrame([{'deletion_drop': fs['deletion_confidence_drop']['mean'],
                   'retention_ratio': fs['retention_probability_ratio']['mean'],
                   'n': fs['deletion_confidence_drop']['n']}]).to_csv(AUDIT / 'faithfulness_summary.csv', index=False)

    gold = json.loads((AUDIT / 'golden_reference.json').read_text())['key_metric_fingerprints']
    rep = json.loads((METRICS / 'reported_metrics.json').read_text())
    diffs = {}
    for c in ('A4', 'A8'):
        diffs[f'internal_{c}_vs_golden_reference'] = {k: abs(internal[c][k] - gold[c][k]) for k in KEYS}
        diffs[f'internal_{c}_vs_reported_metrics'] = {k: abs(internal[c][k] - rep[f'internal_{c}'][k]) for k in KEYS}
    e, r = external['A8'], rep['external_A8']
    # external values are reported to four decimals
    pairs = {'roc_auc': 'AUC', 'auprc': 'AUPRC', 'sensitivity': 'sensitivity', 'specificity': 'specificity',
             'balanced_accuracy': 'balanced_acc', 'mcc': 'MCC'}
    diffs['external_A8_vs_reported_metrics'] = {k: abs(round(e[v], 4) - r[k]) for k, v in pairs.items()}
    diffs['external_A8_vs_reported_metrics'].update({k: abs(e[k.upper()] - r[k]) for k in ('tn', 'fp', 'fn', 'tp')})
    diffs['faithfulness_vs_reported_metrics'] = {
        'deletion_drop': abs(fs['deletion_confidence_drop']['mean'] - rep['faithfulness']['deletion_drop']),
        'retention_ratio': abs(fs['retention_probability_ratio']['mean'] - rep['faithfulness']['retention_ratio'])}
    worst = max(v for d in diffs.values() for v in d.values())
    status = 'PASS' if worst < 1e-12 else 'FAIL'
    report = {'inputs': ['results/predictions/locked_rsna.csv', 'results/predictions/kermany.csv',
                         'results/metrics/xai_faithfulness_summary.json'],
              'absolute_differences': diffs, 'max_absolute_difference': worst, 'status': status}
    (AUDIT / 'reproduce_report.json').write_text(json.dumps(report, indent=2) + '\n')

    print(f'{"":<22}' + ''.join(f'{k:>13}' for k in KEYS))
    for label, m in (('RSNA locked  A4', internal['A4']), ('RSNA locked  A8', internal['A8']), ('Kermany      A8', external['A8'])):
        print(f'{label:<22}' + ''.join(f'{m[k]:>13.4f}' for k in KEYS))
    print(f'Largest difference from the reported values: {worst:.1e}')
    print('Wrote results/audit/{internal_metrics,external_metrics,faithfulness_summary}.csv and reproduce_report.json')
    print('REPRODUCE =', status)
    return 0 if status == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser(prog='aura_cxr.py', description='AURA-CXR verification and metric reproduction.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('verify', help='check configs, splits, predictions, analyses and file integrity')
    sub.add_parser('reproduce', help='recompute the reported A4/A8 metrics from the per-case predictions')
    args = ap.parse_args()
    return verify() if args.cmd == 'verify' else reproduce()


if __name__ == '__main__':
    raise SystemExit(main())

import subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'data/splits'


def _synthetic_labels(path):
    ids = pd.concat([pd.read_csv(SPLITS / f'{s}.csv') for s in ('train', 'val', 'test')])
    rows = ids[['patientId', 'label']].rename(columns={'label': 'Target'})
    rows['x'] = np.where(rows.Target == 1, 100.0, np.nan)
    rows['y'] = np.where(rows.Target == 1, 200.0, np.nan)
    rows['width'] = np.where(rows.Target == 1, 50.0, np.nan)
    rows['height'] = np.where(rows.Target == 1, 80.0, np.nan)
    rows[['patientId', 'x', 'y', 'width', 'height', 'Target']].to_csv(path, index=False)


def test_prepare_splits_reproduces_released_split(tmp_path):
    labels = tmp_path / 'stage_2_train_labels.csv'
    _synthetic_labels(labels)
    run = tmp_path / 'run'
    subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_splits.py'), '--labels_csv', str(labels),
                    '--dicom_dir', str(tmp_path / 'dicom'), '--results_dir', str(run), '--skip_dicom_check'],
                   check=True, capture_output=True)
    for s, n in (('train', 18678), ('val', 2668), ('test', 5338)):
        out = pd.read_csv(run / 'splits' / f'{s}.csv')
        rel = pd.read_csv(SPLITS / f'{s}.csv')
        assert len(out) == n
        assert out.patientId.tolist() == rel.patientId.tolist()
        assert (out.label.values == rel.label.values).all() and (out.sample_id.values == rel.sample_id.values).all()
        assert {'image_path', 'bbox_area_max', 'bbox_cx_mean', 'bbox_cy_mean'} <= set(out.columns)
    # the fold assignment that the pipeline regenerates (seed 42) equals the released one
    tr, va = pd.read_csv(run / 'splits/train.csv'), pd.read_csv(run / 'splits/val.csv')
    y = pd.concat([tr.label, va.label]).to_numpy()
    fold = np.full(len(y), -1)
    for k, (_, hold) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
        fold[hold] = k
    assert (fold == pd.read_csv(run / 'splits/development_oof_folds.csv').oof_fold.values).all()


def test_prepare_splits_requires_dicoms(tmp_path):
    labels = tmp_path / 'labels.csv'
    _synthetic_labels(labels)
    r = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_splits.py'), '--labels_csv', str(labels),
                        '--dicom_dir', str(tmp_path / 'missing'), '--results_dir', str(tmp_path / 'run')],
                       capture_output=True, text=True)
    assert r.returncode != 0 and 'DICOM files are missing' in r.stderr

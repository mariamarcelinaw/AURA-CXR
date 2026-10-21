#!/usr/bin/env python3
"""Shared Q1-safe utilities for validation-only DL pair selection in AURA-CXR.

Protocol
--------
1. Build one patient-level development table = train + validation.
2. Generate one out-of-fold (OOF) pneumonia probability per development patient
   for each candidate model and radiomics.
3. Evaluate all 2-of-4 DL pairs with radiomics using a cross-fitted Logistic
   Regression meta-learner. No test or external labels are read here.
4. Select one pair using a pre-specified hierarchical rule and lock it in a
   manifest consumed by fusion, XAI, statistics, ablation, and external validation.

The locked test and any external dataset are never used by this module for model
selection, calibration, threshold selection, or tie-breaking.
"""
from __future__ import annotations

import itertools
import json
import math
import hashlib
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


FINAL_LOCK_FILENAME = "q1_final_experiment_lock.json"
DEPLOYMENT_MANIFEST_FILENAME = "q1_locked_deployment_manifest.json"


def sha256_file(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_hash(obj: Mapping) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PAIR_LOCK_MUTABLE_FIELDS = {
    "created_at", "final_test_evaluated_once", "locked_test_probabilities_generated_at",
    "locked_test_labels_read_during_fusion", "selection_lock_id",
}
DEPLOYMENT_LOCK_MUTABLE_FIELDS = {
    "created_at", "probability_artifacts", "deployment_status",
    "locked_test_inferred_at", "statistics_completed_at", "deployment_lock_id",
}
PREDICTIVE_INTERVAL_Z95 = 1.96
PREDICTIVE_UNCERTAINTY_DEFINITION = "two_sided_95_predictive_interval_width_equals_2_times_1.96_times_mc_probability_std"


def predictive_interval_width_95(std_probability) -> np.ndarray:
    return (2.0 * PREDICTIVE_INTERVAL_Z95 * np.asarray(std_probability, dtype=np.float64)).astype(np.float64)


def _pair_lock_hash(manifest: Mapping) -> str:
    return _canonical_json_hash({k: v for k, v in manifest.items() if k not in PAIR_LOCK_MUTABLE_FIELDS})


def _deployment_lock_hash(manifest: Mapping) -> str:
    return _canonical_json_hash({k: v for k, v in manifest.items() if k not in DEPLOYMENT_LOCK_MUTABLE_FIELDS})


def verify_pair_manifest_lock(manifest: Mapping) -> None:
    actual = str(manifest.get("selection_lock_id", ""))
    expected = _pair_lock_hash(manifest)
    if not actual or actual != expected:
        raise RuntimeError(f"Pair manifest tampering detected: stored={actual}, expected={expected}")


def verify_deployment_manifest_lock(manifest: Mapping) -> None:
    actual = str(manifest.get("deployment_lock_id", ""))
    expected = _deployment_lock_hash(manifest)
    if not actual or actual != expected:
        raise RuntimeError(f"Deployment manifest tampering detected: stored={actual}, expected={expected}")


def ensemble_predictive_uncertainty(meta_model, stacked_mean: np.ndarray,
                                    member_names: Sequence[str],
                                    member_std: Mapping[str, np.ndarray],
                                    soft_weights: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Propagate member MC deviations using a documented independence approximation."""
    stacked_mean = np.asarray(stacked_mean, dtype=np.float64).reshape(-1)
    n = len(stacked_mean)
    std_matrix = np.column_stack([
        np.asarray(member_std.get(name, np.zeros(n)), dtype=np.float64).reshape(-1)
        for name in member_names
    ])
    if std_matrix.shape != (n, len(member_names)):
        raise ValueError(f"Member uncertainty shape mismatch: {std_matrix.shape}")
    weights = np.asarray([float(soft_weights[name]) for name in member_names], dtype=np.float64)
    soft_std = np.sqrt(np.sum((std_matrix * weights[None, :]) ** 2, axis=1))
    coef = np.asarray(getattr(meta_model, "coef_", None), dtype=np.float64)
    if coef.ndim != 2 or coef.shape[0] != 1 or coef.shape[1] != len(member_names):
        raise RuntimeError(f"Unexpected logistic meta-learner coefficient shape: {coef.shape}")
    linear_std = np.sqrt(np.sum((std_matrix * coef[0][None, :]) ** 2, axis=1))
    stacked_std = np.clip(stacked_mean * (1.0 - stacked_mean) * linear_std, 0.0, 0.5)
    return {
        "soft_std": soft_std.astype(np.float32),
        "soft_predictive_interval_width_95": predictive_interval_width_95(soft_std).astype(np.float32),
        "stacked_std": stacked_std.astype(np.float32),
        "stacked_predictive_interval_width_95": predictive_interval_width_95(stacked_std).astype(np.float32),
    }


def _reports_dir(results_dir: Path) -> Path:
    p = Path(results_dir) / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_experiment_lock(results_dir: Path) -> Dict:
    path = _reports_dir(results_dir) / FINAL_LOCK_FILENAME
    if not path.exists():
        return {"schema": SCHEMA, "test_opened": False, "selection_immutable": False}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_selection_mutable(results_dir: Path, action: str = "modify selection") -> None:
    """Block any selection/fold/threshold mutation after locked-test inference."""
    lock = load_experiment_lock(results_dir)
    manifest_path = _reports_dir(results_dir) / "dl_pair_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if bool(lock.get("test_opened")) or bool(lock.get("selection_immutable")) or bool(manifest.get("final_test_evaluated_once")):
        raise RuntimeError(
            f"Cannot {action}: pair selection is permanently locked because the locked test has been opened. "
            "Create a new results directory for a new experiment."
        )


def assert_no_test_prediction_artifacts_before_selection(results_dir: Path) -> None:
    """Refuse a strict selection run in an output directory that already contains test predictions.

    A fresh results directory is required for the strict protocol. Split CSVs are
    allowed; model probabilities, test feature caches, and final-test audits are not.
    """
    root = Path(results_dir)
    suspicious = []
    for pat in ["probs/*test*.npy", "probs/*test*.npz", "cache/*test*.joblib", "reports/*locked_test*.json"]:
        suspicious.extend(root.glob(pat))
    suspicious = sorted({p.resolve() for p in suspicious if p.is_file()})
    if suspicious:
        preview = "\n".join(f"  - {p}" for p in suspicious[:20])
        raise RuntimeError(
            "Strict development-OOF selection requires a fresh output directory with no prior test predictions/cache. "
            f"Found:\n{preview}\nArchive these artifacts or use a new results directory."
        )


def mark_selection_locked(results_dir: Path, manifest: Mapping) -> Dict:
    lock = load_experiment_lock(results_dir)
    if bool(lock.get("test_opened")):
        raise RuntimeError("Cannot relock selection after the test has been opened.")
    lock.update({
        "schema": SCHEMA,
        "selection_immutable": False,
        "test_opened": False,
        "selected_pair_id": manifest.get("selected_pair_id"),
        "selection_manifest_hash": _canonical_json_hash(manifest),
        "selection_locked_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_json(lock, _reports_dir(results_dir) / FINAL_LOCK_FILENAME)
    return lock


def _model_manifest_parameter_count(results_dir: Path, model_name: str) -> Optional[int]:
    reports = _reports_dir(results_dir)
    candidates = [
        reports / f"oof_{model_name}_manifest.json",
        reports / f"{model_name}_oof_protocol.json",
        reports / f"{model_name}_development_refit_manifest.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("parameter_count", "trainable_parameter_count", "actual_parameter_count"):
                if data.get(key) is not None:
                    return int(data[key])
        except Exception:
            continue
    return None


def actual_parameter_counts(results_dir: Path, candidates: Sequence[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    values, sources = {}, {}
    for name in candidates:
        n = _model_manifest_parameter_count(results_dir, name)
        if n is None:
            n = int(DEFAULT_PARAMETER_COUNTS.get(name, 10**12))
            sources[name] = "fallback_prespecified_estimate"
        else:
            sources[name] = "actual_checkpoint_model_count"
        values[name] = int(n)
    return values, sources


SCHEMA = "aura_cxr_dl_pair_selection_q1_v17"
DEFAULT_CANDIDATES = ("efficientnetv2", "resnet50", "xrv", "eva_x")
DISPLAY_NAMES = {
    "efficientnetv2": "EfficientNetV2S",
    "resnet50": "ResNet50",
    "xrv": "DenseNet121-XRV",
    "eva_x": "EVA-X-S",
    "radiomics": "MRFO-optimized radiomics branch (best learner)",
}
DEFAULT_PARAMETER_COUNTS = {
    "efficientnetv2": 21_500_000,
    "resnet50": 25_600_000,
    "xrv": 7_000_000,
    "eva_x": 22_000_000,
}



def save_json(obj: Mapping, path: Path) -> None:
    """Atomically save JSON so a stopped VM cannot leave a half-written lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, default=_json_default)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)



def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def normalize_binary_probability(arr: np.ndarray, expected_n: Optional[int] = None, name: str = "probability") -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 2:
        a = a[:, 1]
    elif a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    elif a.ndim != 1:
        raise ValueError(f"{name} must be (N,), (N,1), or (N,2); got {a.shape}")
    if expected_n is not None and len(a) != int(expected_n):
        raise ValueError(f"{name} row mismatch: got {len(a)}, expected {expected_n}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN/Inf")
    if float(np.min(a)) < -1e-6 or float(np.max(a)) > 1 + 1e-6:
        raise ValueError(f"{name} outside [0,1]: min={a.min()}, max={a.max()}")
    return np.clip(a, 0.0, 1.0).astype(np.float64)


def two_col(p: np.ndarray) -> np.ndarray:
    p = normalize_binary_probability(p)
    return np.column_stack([1.0 - p, p]).astype(np.float32)


def ensure_development_table(results_dir: Path, train_df: Optional[pd.DataFrame] = None,
                             val_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    results_dir = Path(results_dir)
    split_dir = results_dir / "splits"
    if train_df is None:
        train_df = pd.read_csv(split_dir / "train.csv")
    if val_df is None:
        val_df = pd.read_csv(split_dir / "val.csv")
    train_df = train_df.copy(); val_df = val_df.copy()
    train_df["original_split"] = "train"
    val_df["original_split"] = "validation"
    dev = pd.concat([train_df, val_df], ignore_index=True)
    if "patientId" not in dev or "label" not in dev:
        raise ValueError("Development table requires patientId and label columns")
    if dev["patientId"].astype(str).duplicated().any():
        dup = dev.loc[dev["patientId"].astype(str).duplicated(), "patientId"].head().tolist()
        raise AssertionError(f"Patient leakage/duplicates in development table: {dup}")
    dev["development_index"] = np.arange(len(dev), dtype=np.int64)
    out = split_dir / "development.csv"
    dev.to_csv(out, index=False)
    return dev



def make_patient_oof_folds(dev_df: pd.DataFrame, n_splits: int = 5, seed: int = 42,
                           force: bool = False, results_dir: Optional[Path] = None) -> np.ndarray:
    if results_dir is not None and force:
        assert_selection_mutable(results_dir, "regenerate development OOF folds")
    y = dev_df["label"].to_numpy(dtype=int)
    fold_id = np.full(len(dev_df), -1, dtype=np.int16)
    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    for fold, (_, hold_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[hold_idx] = fold
    if np.any(fold_id < 0):
        raise AssertionError("Not all development rows received an OOF fold")
    if results_dir is not None:
        df = dev_df[["development_index", "patientId", "label", "original_split"]].copy()
        df["oof_fold"] = fold_id
        path = Path(results_dir) / "splits" / "development_oof_folds.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            old = pd.read_csv(path)
            cols = ["development_index", "patientId", "label", "original_split", "oof_fold"]
            if len(old) == len(df) and old[cols].astype(str).equals(df[cols].astype(str)):
                return old["oof_fold"].to_numpy(dtype=np.int16)
            raise RuntimeError(f"Existing fold manifest differs: {path}. A new experiment directory is required.")
        df.to_csv(path, index=False)
        save_json({
            "schema": SCHEMA,
            "selection_data": "development_oof",
            "patient_disjoint_folds": True,
            "n_splits": int(n_splits),
            "seed": int(seed),
            "test_set_used_for_selection": False,
            "external_set_used_for_selection": False,
            "fold_csv_sha256": sha256_file(path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, Path(results_dir) / "reports" / "development_oof_fold_manifest.json")
    return fold_id



def save_oof_artifact(results_dir: Path, model_name: str, proba: np.ndarray,
                      dev_df: pd.DataFrame, fold_id: np.ndarray,
                      source: str, extra: Optional[Mapping] = None) -> Path:
    results_dir = Path(results_dir)
    proba = normalize_binary_probability(proba, len(dev_df), f"{model_name} OOF")
    if len(fold_id) != len(dev_df):
        raise ValueError("fold_id length mismatch")
    path = results_dir / "probs" / f"oof_{model_name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        proba=proba.astype(np.float32),
        fold_id=np.asarray(fold_id, dtype=np.int16),
        patient_id=np.asarray(dev_df["patientId"].astype(str).tolist(), dtype="U"),
        label=dev_df["label"].to_numpy(dtype=np.int8),
        development_index=dev_df["development_index"].to_numpy(dtype=np.int64),
        original_split=np.asarray(dev_df["original_split"].astype(str).tolist(), dtype="U"),
        schema=np.asarray(SCHEMA),
        source=np.asarray(str(source)),
    )
    meta = {
        "schema": SCHEMA,
        "model": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "n": int(len(dev_df)),
        "n_folds": int(len(np.unique(fold_id))),
        "source": source,
        "patient_disjoint": True,
        "test_set_used": False,
        "external_set_used": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(dict(extra))
    save_json(meta, results_dir / "reports" / f"oof_{model_name}_manifest.json")
    return path


def load_oof_artifact(results_dir: Path, model_name: str, dev_df: pd.DataFrame,
                      require_fold_match: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    results_dir = Path(results_dir)
    npz = results_dir / "probs" / f"oof_{model_name}.npz"
    npy = results_dir / "probs" / f"oof_{model_name}.npy"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        p = normalize_binary_probability(z["proba"], len(dev_df), f"oof_{model_name}")
        fold = np.asarray(z["fold_id"], dtype=int)
        pid = np.asarray(z["patient_id"]).astype(str)
        if not np.array_equal(pid, dev_df["patientId"].astype(str).to_numpy()):
            raise ValueError(f"OOF patient order mismatch for {model_name}")
        if "label" in z and not np.array_equal(np.asarray(z["label"], dtype=int), dev_df["label"].to_numpy(dtype=int)):
            raise ValueError(f"OOF label order mismatch for {model_name}")
        return p, fold
    if npy.exists():
        p = normalize_binary_probability(np.load(npy), len(dev_df), f"oof_{model_name}")
        fold_path = results_dir / "splits" / "development_oof_folds.csv"
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing fold manifest for legacy OOF array: {fold_path}")
        fold = pd.read_csv(fold_path)["oof_fold"].to_numpy(dtype=int)
        return p, fold
    raise FileNotFoundError(f"Missing OOF artifact for {model_name}: {npz}")


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=int); p = normalize_binary_probability(p, len(y))
    edges = np.linspace(0, 1, int(n_bins) + 1)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def _fold_metric_sd(y: np.ndarray, p: np.ndarray, fold_id: np.ndarray) -> Tuple[float, float]:
    vals = []
    for f in sorted(np.unique(fold_id)):
        m = fold_id == f
        if len(np.unique(y[m])) < 2:
            continue
        vals.append(roc_auc_score(y[m], p[m]))
    if not vals:
        return math.nan, math.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def individual_metrics(y: np.ndarray, p: np.ndarray, fold_id: np.ndarray) -> Dict[str, float]:
    fm, fsd = _fold_metric_sd(y, p, fold_id)
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
        "fold_auc_mean": fm,
        "fold_auc_sd": fsd,
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    out = np.zeros(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and z[j] == z[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    restored = np.empty(len(x), dtype=float)
    restored[order] = out
    return restored


def delong_correlated_auc_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> Tuple[float, float, float]:
    """Two-sided DeLong test for correlated OOF ROC AUCs."""
    y = np.asarray(y, dtype=int)
    p_a = normalize_binary_probability(p_a, len(y), "DeLong A")
    p_b = normalize_binary_probability(p_b, len(y), "DeLong B")
    order = np.argsort(-y, kind="mergesort")
    m = int(np.sum(y == 1)); n = len(y) - m
    if m < 2 or n < 2:
        return math.nan, math.nan, math.nan
    preds = np.vstack([p_a[order], p_b[order]])
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m+n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m*n) - (m+1.0)/(2.0*n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    contrast = np.array([[1.0, -1.0]])
    var = float((contrast @ cov @ contrast.T)[0,0])
    if not np.isfinite(var) or var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    from scipy.stats import norm
    z = abs(float(aucs[0] - aucs[1])) / math.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2.0 * (1.0 - norm.cdf(z)))


def diversity_metrics(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    p1 = normalize_binary_probability(p1, len(y)); p2 = normalize_binary_probability(p2, len(y))
    pred1 = (p1 >= threshold).astype(int); pred2 = (p2 >= threshold).astype(int)
    err1 = pred1 != y; err2 = pred2 != y
    n11 = int(np.sum(~err1 & ~err2)); n00 = int(np.sum(err1 & err2))
    n10 = int(np.sum(~err1 & err2)); n01 = int(np.sum(err1 & ~err2))
    denom = n11 * n00 + n10 * n01
    q = ((n11 * n00 - n10 * n01) / denom) if denom else math.nan
    return {
        "pearson_probability": float(np.corrcoef(p1, p2)[0, 1]) if np.std(p1) > 0 and np.std(p2) > 0 else math.nan,
        "spearman_probability": float(pd.Series(p1).corr(pd.Series(p2), method="spearman")),
        "disagreement_rate": float(np.mean(pred1 != pred2)),
        "error_overlap_rate": float(np.mean(err1 & err2)),
        "double_fault_rate": float(n00 / len(y)),
        "q_statistic": float(q),
    }


def crossfit_meta_predictions(X: np.ndarray, y: np.ndarray, fold_id: np.ndarray,
                              seed: int = 42) -> Tuple[np.ndarray, LogisticRegression]:
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int); fold_id = np.asarray(fold_id, dtype=int)
    oof = np.full(len(y), np.nan, dtype=float)
    for f in sorted(np.unique(fold_id)):
        tr = fold_id != f; va = fold_id == f
        if not np.any(va):
            continue
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=int(seed))
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("Meta OOF predictions contain missing values")
    final = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=int(seed))
    final.fit(X, y)
    return oof, final


def tune_threshold(y: np.ndarray, p: np.ndarray, min_sensitivity: float = 0.85) -> float:
    y = np.asarray(y, dtype=int); p = normalize_binary_probability(p, len(y))
    best = None
    for th in np.linspace(0.01, 0.99, 197):
        pred = p >= th
        tp = np.sum((y == 1) & pred); fn = np.sum((y == 1) & ~pred)
        tn = np.sum((y == 0) & ~pred); fp = np.sum((y == 0) & pred)
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        if sens < min_sensitivity:
            continue
        score = sens + spec - 1
        candidate = (score, spec, sens, -abs(th - 0.5), th)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        # deterministic fallback: maximize Youden without floor
        for th in np.linspace(0.01, 0.99, 197):
            pred = p >= th
            tp = np.sum((y == 1) & pred); fn = np.sum((y == 1) & ~pred)
            tn = np.sum((y == 0) & ~pred); fp = np.sum((y == 0) & pred)
            sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
            candidate = (sens + spec - 1, spec, sens, -abs(th - 0.5), th)
            if best is None or candidate > best:
                best = candidate
    return float(best[-1])


def _hierarchical_select(pair_df: pd.DataFrame, auc_tolerance: float) -> pd.Series:
    """Pre-specified hierarchical pair selection.

    Primary criterion is the mean patient-level five-fold OOF AUC, not the
    pooled OOF AUC. Pooled OOF AUC is still reported as a secondary audit.
    """
    max_auc = float(pair_df["fold_auc_mean"].max())
    contenders = pair_df[pair_df["fold_auc_mean"] >= max_auc - float(auc_tolerance)].copy()
    contenders = contenders.sort_values(
        ["oof_auprc", "brier", "ece", "fold_auc_sd", "total_parameters", "pair_id"],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    )
    return contenders.iloc[0]



def select_dl_pair_and_train_meta(results_dir: Path,
                                  candidates: Sequence[str] = DEFAULT_CANDIDATES,
                                  n_splits: int = 5,
                                  seed: int = 42,
                                  auc_tolerance: float = 0.002,
                                  min_sensitivity: float = 0.85,
                                  parameter_counts: Optional[Mapping[str, int]] = None,
                                  force: bool = False) -> Dict:
    """Select 2 DL models using development OOF only and train a locked meta learner."""
    results_dir = Path(results_dir)
    reports = results_dir / "reports"; models_dir = results_dir / "models"; probs_dir = results_dir / "probs"
    reports.mkdir(parents=True, exist_ok=True); models_dir.mkdir(parents=True, exist_ok=True); probs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports / "dl_pair_selection_manifest.json"
    if manifest_path.exists() and not force:
        return load_pair_manifest(results_dir, strict=True)
    if force:
        assert_selection_mutable(results_dir, "force pair reselection")
    # A strict V17 experiment must not have touched the test before selection.
    assert_no_test_prediction_artifacts_before_selection(results_dir)

    dev = ensure_development_table(results_dir)
    y = dev["label"].to_numpy(dtype=int)
    canonical_fold = make_patient_oof_folds(dev, n_splits=n_splits, seed=seed, results_dir=results_dir, force=force)
    oof: Dict[str, np.ndarray] = {}
    oof_hashes = {}
    for name in list(candidates) + ["radiomics"]:
        p_arr, fold = load_oof_artifact(results_dir, name, dev)
        if not np.array_equal(fold, canonical_fold):
            raise ValueError(f"OOF fold assignment mismatch for {name}")
        oof[name] = p_arr
        oof_path = results_dir / "probs" / f"oof_{name}.npz"
        oof_hashes[name] = sha256_file(oof_path)

    radiomics_selection_path = reports / "radiomics_learner_selection_manifest.json"
    if not radiomics_selection_path.exists():
        raise FileNotFoundError(f"Missing radiomics learner selection manifest: {radiomics_selection_path}")
    radiomics_selection = json.loads(radiomics_selection_path.read_text(encoding="utf-8"))
    if radiomics_selection.get("test_set_used_for_selection") is not False:
        raise RuntimeError("Unsafe radiomics selection manifest")
    if radiomics_selection.get("selection_scope") != "development_outer_oof_fixed_best_radiomics":
        raise RuntimeError("Radiomics learner selection must use fixed-best development outer OOF")
    if radiomics_selection.get("oof_deployment_family_identical") is not True:
        raise RuntimeError("Radiomics OOF learner family must be identical to the deployment learner family")
    if radiomics_selection.get("outer_holdout_used_for_hyperparameter_tuning") is not False:
        raise RuntimeError("Unsafe radiomics selection: outer holdout influenced MRFO tuning")
    selected_estimator = str(radiomics_selection.get("selected_estimator", ""))
    candidate_artifacts = radiomics_selection.get("candidate_oof_artifacts", {})
    selected_candidate = candidate_artifacts.get(selected_estimator, {})
    selected_candidate_path = Path(selected_candidate.get("path", ""))
    if not selected_candidate_path.exists():
        raise FileNotFoundError(f"Missing selected fixed radiomics candidate OOF: {selected_candidate_path}")
    if sha256_file(selected_candidate_path) != selected_candidate.get("sha256"):
        raise RuntimeError("Selected fixed radiomics candidate OOF hash mismatch")
    candidate_p, candidate_fold = load_oof_artifact(results_dir, f"radiomics_{selected_estimator}", dev)
    if not np.array_equal(candidate_fold, canonical_fold):
        raise RuntimeError("Selected fixed radiomics candidate fold assignment mismatch")
    if not np.allclose(candidate_p, oof["radiomics"], rtol=0.0, atol=1e-12):
        raise RuntimeError("Generic radiomics OOF is not identical to the selected fixed learner OOF")

    individual_rows = []
    for name in candidates:
        row = {"model": name, "display_name": DISPLAY_NAMES.get(name, name)}
        row.update(individual_metrics(y, oof[name], canonical_fold))
        individual_rows.append(row)
    individual_df = pd.DataFrame(individual_rows)
    individual_df.to_csv(reports / "dl_candidate_oof_metrics.csv", index=False)
    best_individual_row = individual_df.sort_values(["fold_auc_mean", "auc", "model"], ascending=[False, False, True], kind="mergesort").iloc[0]
    best_individual_model = str(best_individual_row["model"])
    best_individual_oof = oof[best_individual_model]

    params, param_sources = actual_parameter_counts(results_dir, candidates)
    if parameter_counts:
        for k, v in parameter_counts.items():
            params[str(k)] = int(v); param_sources[str(k)] = "explicit_cli_override"

    pair_rows: List[Dict] = []
    pair_models: Dict[str, LogisticRegression] = {}
    pair_oof: Dict[str, np.ndarray] = {}
    for a, b in itertools.combinations(candidates, 2):
        pair_id = f"{a}+{b}"
        X = np.column_stack([oof[a], oof[b], oof["radiomics"]])
        meta_oof, meta = crossfit_meta_predictions(X, y, canonical_fold, seed=seed)
        m = individual_metrics(y, meta_oof, canonical_fold)
        div = diversity_metrics(y, oof[a], oof[b])
        pair_auc_d, best_auc_d, delong_p = delong_correlated_auc_test(y, meta_oof, best_individual_oof)
        row = {
            "pair_id": pair_id, "model_1": a, "model_2": b,
            "display_pair": f"{DISPLAY_NAMES.get(a,a)} + {DISPLAY_NAMES.get(b,b)}",
            "oof_auc": m["auc"], "oof_auprc": m["auprc"], "brier": m["brier"],
            "ece": m["ece"], "log_loss": m["log_loss"], "fold_auc_mean": m["fold_auc_mean"],
            "fold_auc_sd": m["fold_auc_sd"], "total_parameters": int(params[a] + params[b]),
            "parameter_count_source": f"{a}:{param_sources[a]};{b}:{param_sources[b]}",
            "best_individual_model": best_individual_model,
            "delta_auc_vs_best_individual": float(pair_auc_d - best_auc_d),
            "delong_p_vs_best_individual": float(delong_p), **div,
        }
        pair_rows.append(row); pair_models[pair_id] = meta; pair_oof[pair_id] = meta_oof

    pair_df = pd.DataFrame(pair_rows)
    selected = _hierarchical_select(pair_df, auc_tolerance)
    pair_df["selected"] = pair_df["pair_id"].eq(selected["pair_id"])
    pair_df.to_csv(reports / "dl_pair_selection_validation.csv", index=False)

    selected_id = str(selected["pair_id"])
    selected_models = [str(selected["model_1"]), str(selected["model_2"])]
    meta_model = pair_models[selected_id]
    meta_oof = pair_oof[selected_id]
    threshold = tune_threshold(y, meta_oof, min_sensitivity=min_sensitivity)
    meta_path = models_dir / "selected_pair_meta_learner.pkl"
    joblib.dump(meta_model, meta_path)
    np.save(probs_dir / "stacked_development_oof.npy", two_col(meta_oof))

    member_names = selected_models + ["radiomics"]
    member_auc = {m: float(roc_auc_score(y, oof[m])) for m in member_names}
    raw_w = np.asarray([max(member_auc[m] - 0.5, 1e-6) for m in member_names], dtype=float)
    soft_weights = raw_w / raw_w.sum()
    soft_oof = sum(float(w) * oof[m] for w, m in zip(soft_weights, member_names))
    np.save(probs_dir / "soft_development_oof.npy", two_col(soft_oof))
    individual_thresholds = {m: float(tune_threshold(y, oof[m], min_sensitivity=min_sensitivity)) for m in member_names}
    soft_threshold = float(tune_threshold(y, soft_oof, min_sensitivity=min_sensitivity))

    original_val_mask = dev["original_split"].astype(str).eq("validation").to_numpy()
    np.save(probs_dir / "stacked_val.npy", two_col(meta_oof[original_val_mask]))
    np.save(probs_dir / "soft_val.npy", two_col(soft_oof[original_val_mask]))

    fold_manifest_path = reports / "development_oof_fold_manifest.json"
    manifest = {
        "schema": SCHEMA, "selection_status": "LOCKED", "selection_data": "development_oof",
        "patient_disjoint_folds": True, "n_development": int(len(dev)), "n_splits": int(n_splits), "seed": int(seed),
        "candidate_models": list(candidates), "candidate_display_names": {m: DISPLAY_NAMES.get(m, m) for m in candidates},
        "candidate_parameter_counts": params, "candidate_parameter_count_sources": param_sources,
        "best_individual_oof_model": best_individual_model,
        "best_individual_oof_display": DISPLAY_NAMES.get(best_individual_model, best_individual_model),
        "radiomics_member": "radiomics",
        "radiomics_selected_estimator": radiomics_selection.get("selected_estimator"),
        "radiomics_selected_display_name": radiomics_selection.get("selected_display_name"),
        "radiomics_hyperparameter_policy": "screen four families with two-seed stability, evaluate the same global top two on all outer folds, select one fixed best learner from aggregate development OOF, and refit that same family",
        "radiomics_selection_manifest": str(radiomics_selection_path),
        "radiomics_selection_manifest_sha256": sha256_file(radiomics_selection_path),
        "radiomics_refit_per_outer_fold": True,
        "radiomics_oof_deployment_family_identical": True,
        "selected_dl_models": selected_models, "selected_pair_id": selected_id,
        "selected_pair_display": str(selected["display_pair"]),
        "meta_features": [f"p_{selected_models[0]}", f"p_{selected_models[1]}", "p_radiomics"],
        "meta_feature_count": 3, "meta_learner": "LogisticRegression(class_weight=balanced)",
        "meta_learner_path": str(meta_path), "meta_learner_sha256": sha256_file(meta_path),
        "threshold": float(threshold), "threshold_source": "development_oof",
        "individual_thresholds": individual_thresholds, "soft_voting_threshold": soft_threshold,
        "soft_voting_members": member_names,
        "soft_voting_weights": {m: float(w) for m, w in zip(member_names, soft_weights)},
        "selection_rule": {
            "primary": "mean_patient_level_5fold_oof_auc", "auc_tolerance": float(auc_tolerance),
            "tie_breakers": ["oof_auprc_desc", "brier_asc", "ece_asc", "fold_auc_sd_asc", "actual_total_parameters_asc"],
            "pair_diversity_reported_not_primary": True,
        },
        "candidate_training_budget_policy": {
            "policy": "prespecified_architecture_specific_training_with_identical_OOF_folds_and_MC_passes",
            "fairness_audit_required": True,
            "test_or_external_feedback_used": False,
        },
        "selected_pair_metrics": {k: _json_default(selected[k]) for k in pair_df.columns if k in selected.index},
        "selection_primary_value": float(selected["fold_auc_mean"]),
        "pooled_oof_auc_reported_secondary": float(selected["oof_auc"]),
        "oof_artifact_sha256": oof_hashes,
        "fold_manifest_sha256": sha256_file(fold_manifest_path) if fold_manifest_path.exists() else None,
        "test_set_used_for_selection": False, "test_images_accessed_before_selection": False,
        "external_set_used_for_selection": False, "kermany_used_for_selection": False,
        "chexpert_used_for_selection": False, "pair_selection_rule_prespecified": True,
        "final_test_evaluated_once": False, "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest["selection_lock_id"] = _pair_lock_hash(manifest)
    save_json(manifest, manifest_path)
    mark_selection_locked(results_dir, manifest)
    save_json({
        "feature_order": manifest["meta_features"], "n_features": 3,
        "training_probabilities": "patient-level development out-of-fold",
        "test_set_used_for_meta_training": False, "external_set_used_for_meta_training": False,
        "selection_lock_id": manifest["selection_lock_id"],
    }, reports / "meta_learner_features.json")
    save_json({
        "stacked_ensemble": float(threshold), "soft_voting": soft_threshold,
        "individual_models": individual_thresholds, "selection_split": "development_oof",
        "test_set_used_for_selection": False,
    }, reports / "fusion_thresholds_validation_tuned.json")
    return manifest




def validate_pair_manifest(manifest: Mapping, require_locked: bool = True) -> None:
    required = ["schema", "selection_data", "selected_dl_models", "threshold",
                "test_set_used_for_selection", "external_set_used_for_selection", "selection_lock_id"]
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"Pair-selection manifest missing fields: {missing}")
    if str(manifest.get("schema")) != SCHEMA:
        raise ValueError(f"Pair-selection schema mismatch: {manifest.get('schema')} != {SCHEMA}")
    if manifest.get("test_set_used_for_selection") is not False:
        raise ValueError("Unsafe manifest: test set used for selection")
    if manifest.get("external_set_used_for_selection") is not False:
        raise ValueError("Unsafe manifest: external set used for selection")
    if manifest.get("selection_data") != "development_oof":
        raise ValueError("Q1 strict mode requires selection_data=development_oof")
    selected = list(manifest.get("selected_dl_models", []))
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError(f"Exactly two distinct DL models must be selected, got {selected}")
    if require_locked and manifest.get("selection_status") != "LOCKED":
        raise ValueError("Pair-selection manifest is not locked")



def load_pair_manifest(results_dir: Path, strict: bool = True) -> Dict:
    path = Path(results_dir) / "reports" / "dl_pair_selection_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-pair manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_pair_manifest(manifest, require_locked=strict)
    verify_pair_manifest_lock(manifest)
    return manifest




def _default_refit_model_path(results_dir: Path, model_name: str) -> Path:
    root = Path(results_dir)
    if model_name in {"efficientnetv2", "resnet50"}:
        return root / "models" / f"{model_name}_development_refit.keras"
    if model_name in {"xrv", "eva_x"}:
        return root / "models" / f"{model_name}_development_refit.pt"
    if model_name == "radiomics":
        return root / "models" / "radiomics_mrfo_development_refit.pkl"
    raise KeyError(model_name)


def create_or_refresh_deployment_manifest(results_dir: Path, require_complete: bool = True) -> Dict:
    """Create a single source-of-truth deployment manifest from development-refit models."""
    results_dir = Path(results_dir)
    pair = load_pair_manifest(results_dir, strict=True)
    if pair.get("final_test_evaluated_once"):
        raise RuntimeError("Deployment manifest cannot be changed after locked-test evaluation.")
    members = list(pair["selected_dl_models"]) + ["radiomics"]
    model_artifacts, missing = {}, []
    for name in members:
        path = _default_refit_model_path(results_dir, name)
        if not path.exists():
            missing.append(str(path)); continue
        model_artifacts[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "role": "development_refit"}
    if missing and require_complete:
        raise FileNotFoundError("Missing selected development-refit model artifacts:\n" + "\n".join(missing))
    rad_sel_path = _reports_dir(results_dir) / "radiomics_learner_selection_manifest.json"
    if not rad_sel_path.exists():
        raise FileNotFoundError(f"Missing radiomics learner selection manifest: {rad_sel_path}")
    rad_sel = json.loads(rad_sel_path.read_text(encoding="utf-8"))
    if rad_sel.get("test_set_used_for_selection") is not False:
        raise RuntimeError("Unsafe radiomics selection manifest")
    if rad_sel.get("selection_scope") != "development_outer_oof_fixed_best_radiomics":
        raise RuntimeError("Deployment requires fixed-best outer-OOF radiomics selection")
    if rad_sel.get("oof_deployment_family_identical") is not True:
        raise RuntimeError("Deployment requires the same fixed radiomics family used for OOF")
    manifest = {
        "schema": SCHEMA,
        "deployment_status": "READY_FOR_LOCKED_TEST" if not missing else "WAITING_FOR_SELECTED_REFITS",
        "selection_lock_id": pair["selection_lock_id"],
        "selected_dl_models": list(pair["selected_dl_models"]),
        "members": members,
        "base_model_artifacts": model_artifacts,
        "radiomics_artifact": model_artifacts.get("radiomics"),
        "radiomics_selection": {
            "selected_estimator": rad_sel.get("selected_estimator"),
            "selected_display_name": rad_sel.get("selected_display_name"),
            "selection_metric": rad_sel.get("selection_metric"),
            "selection_scope": rad_sel.get("selection_scope"),
            "oof_deployment_family_identical": bool(rad_sel.get("oof_deployment_family_identical")),
            "feature_config": dict(rad_sel.get("feature_config", {})),
            "screening": dict(rad_sel.get("screening", {})),
            "final_refinement": dict(rad_sel.get("final_refinement", {})),
            "manifest_path": str(rad_sel_path.resolve()),
            "manifest_sha256": sha256_file(rad_sel_path),
        },
        "meta_learner_artifact": {
            "path": str(Path(pair["meta_learner_path"]).resolve()),
            "sha256": sha256_file(Path(pair["meta_learner_path"])),
        },
        "meta_feature_order": list(pair["meta_features"]),
        "threshold": float(pair["threshold"]),
        "threshold_source": "development_oof",
        "individual_thresholds": dict(pair.get("individual_thresholds", {})),
        "soft_voting_threshold": float(pair.get("soft_voting_threshold", pair["threshold"])),
        "soft_voting_weights": dict(pair["soft_voting_weights"]),
        "probability_artifacts": {},
        "test_set_used_for_selection": False,
        "external_set_used_for_selection": False,
        "missing_refit_artifacts": missing,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _reports_dir(results_dir) / DEPLOYMENT_MANIFEST_FILENAME
    if path.exists():
        old = load_deployment_manifest(results_dir, strict=False)
        if old.get("deployment_status") in {"LOCKED_TEST_INFERRED", "FINALIZED"}:
            raise RuntimeError("Deployment manifest is immutable after locked-test inference.")
        manifest["probability_artifacts"] = old.get("probability_artifacts", {})
        manifest["created_at"] = old.get("created_at", manifest["created_at"])
    manifest["deployment_lock_id"] = _deployment_lock_hash(manifest)
    save_json(manifest, path)
    return manifest


def load_deployment_manifest(results_dir: Path, strict: bool = True) -> Dict:
    path = _reports_dir(results_dir) / DEPLOYMENT_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing locked deployment manifest: {path}. Run selected development refit first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_deployment_manifest_lock(manifest)
    pair = load_pair_manifest(results_dir, strict=True)
    if manifest.get("schema") != SCHEMA or manifest.get("selection_lock_id") != pair.get("selection_lock_id"):
        raise RuntimeError("Deployment manifest does not match the locked pair-selection manifest.")
    if strict and manifest.get("deployment_status") not in {"READY_FOR_LOCKED_TEST", "LOCKED_TEST_INFERRED", "FINALIZED"}:
        raise RuntimeError(f"Deployment is incomplete: {manifest.get('deployment_status')}")
    for name, rec in manifest.get("base_model_artifacts", {}).items():
        path_m = Path(rec["path"])
        if not path_m.exists() or sha256_file(path_m) != rec.get("sha256"):
            raise RuntimeError(f"Locked model artifact missing/changed for {name}: {path_m}")
    rad_sel = manifest.get("radiomics_selection", {})
    rad_path = Path(rad_sel.get("manifest_path", ""))
    if not rad_path.exists() or sha256_file(rad_path) != rad_sel.get("manifest_sha256"):
        raise RuntimeError(f"Radiomics learner-selection manifest missing/changed: {rad_path}")
    rad_data = json.loads(rad_path.read_text(encoding="utf-8"))
    if rad_data.get("selected_estimator") != rad_sel.get("selected_estimator"):
        raise RuntimeError("Radiomics selected-estimator mismatch in deployment manifest")
    if rad_data.get("oof_deployment_family_identical") is not True:
        raise RuntimeError("Locked deployment radiomics family is not identical to its OOF family")
    return manifest


def resolve_locked_model_artifact(results_dir: Path, model_name: str) -> Path:
    deployment = load_deployment_manifest(results_dir, strict=True)
    rec = deployment.get("base_model_artifacts", {}).get(model_name)
    if not rec:
        raise FileNotFoundError(f"Model {model_name} is not part of the locked deployment")
    path = Path(rec["path"])
    if sha256_file(path) != rec.get("sha256"):
        raise RuntimeError(f"Locked model hash mismatch: {path}")
    return path


def register_probability_artifact(results_dir: Path, model_name: str, split: str, path: Path,
                                  source_model_path: Optional[Path] = None) -> Dict:
    """Register an exact inference artifact before fusion; no fallback is allowed."""
    results_dir = Path(results_dir); path = Path(path)
    deployment = load_deployment_manifest(results_dir, strict=True)
    if deployment.get("deployment_status") in {"LOCKED_TEST_INFERRED", "FINALIZED"}:
        raise RuntimeError("Cannot register/replace probabilities after locked-test fusion.")
    if not path.exists():
        raise FileNotFoundError(path)
    if source_model_path is not None:
        source_model_path = Path(source_model_path)
        locked = resolve_locked_model_artifact(results_dir, model_name)
        if source_model_path.resolve() != locked.resolve() or sha256_file(source_model_path) != sha256_file(locked):
            raise RuntimeError(f"Inference source is not the locked development-refit model for {model_name}")
    key = f"{model_name}:{split}"
    deployment.setdefault("probability_artifacts", {})[key] = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "source_model": str(source_model_path.resolve()) if source_model_path else None,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(deployment, _reports_dir(results_dir) / DEPLOYMENT_MANIFEST_FILENAME)
    return deployment


def resolve_probability_file(results_dir: Path, model_name: str, split: str,
                             exact_deployment: bool = True) -> Path:
    """Resolve only the exact artifact recorded in the locked deployment manifest.

    Legacy/cross-validation fallbacks are intentionally forbidden in V17 final mode.
    """
    deployment = load_deployment_manifest(results_dir, strict=True)
    key = f"{model_name}:{split}"
    registry = deployment.get("probability_artifacts", {})
    if key not in registry:
        raise FileNotFoundError(f"Locked deployment has no registered probability artifact for {key}")
    path = Path(registry[key]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    expected = registry[key].get("sha256")
    if expected and actual != expected:
        raise RuntimeError(f"Probability artifact hash mismatch for {key}: {path}")
    return path




def finalize_locked_test_fusion(results_dir: Path, exact_deployment: bool = True,
                                allow_deterministic_recompute: bool = False) -> Dict:
    """Apply the OOF-trained meta learner once to exact development-refit test probabilities."""
    results_dir = Path(results_dir)
    pair = load_pair_manifest(results_dir, strict=True)
    deployment = load_deployment_manifest(results_dir, strict=True)
    if pair.get("final_test_evaluated_once") and not allow_deterministic_recompute:
        raise RuntimeError(
            "Locked test fusion has already been generated. Re-running or changing it is disabled. "
            "Use a new results directory for a new experiment."
        )
    selected = list(pair["selected_dl_models"]); members = selected + ["radiomics"]
    probs, expected_n, paths = {}, None, {}
    for m in members:
        path = resolve_probability_file(results_dir, m, "test", exact_deployment=True)
        p_arr = normalize_binary_probability(np.load(path), name=f"{m} locked-test")
        if expected_n is None: expected_n = len(p_arr)
        elif len(p_arr) != expected_n: raise ValueError(f"Locked-test probability length mismatch for {m}")
        probs[m] = p_arr; paths[m] = str(path)
    X = np.column_stack([probs[selected[0]], probs[selected[1]], probs["radiomics"]])
    meta_path = Path(deployment["meta_learner_artifact"]["path"])
    if sha256_file(meta_path) != deployment["meta_learner_artifact"]["sha256"]:
        raise RuntimeError("Meta-learner hash mismatch")
    meta = joblib.load(meta_path)
    stacked = meta.predict_proba(X)[:, 1]
    soft_weights = deployment["soft_voting_weights"]
    soft = sum(float(soft_weights[m]) * probs[m] for m in members)
    member_std = {"radiomics": np.zeros(expected_n, dtype=np.float32)}
    for m in selected:
        std_path = results_dir / "probs" / f"{m}_development_refit_test_std.npy"
        if not std_path.exists():
            raise FileNotFoundError(f"Missing selected-member MC standard deviation: {std_path}")
        std_arr = np.asarray(np.load(std_path), dtype=np.float64)
        member_std[m] = std_arr[:, 1] if std_arr.ndim == 2 else std_arr.reshape(-1)
        if len(member_std[m]) != expected_n:
            raise ValueError(f"Uncertainty length mismatch for {m}")
    ensemble_uncertainty = ensemble_predictive_uncertainty(
        meta, stacked, members, member_std, soft_weights
    )
    pdir = results_dir / "probs"; pdir.mkdir(parents=True, exist_ok=True)
    stacked_path = pdir / "stacked_locked_test.npy"; soft_path = pdir / "soft_locked_test.npy"
    np.save(stacked_path, two_col(stacked)); np.save(soft_path, two_col(soft))
    uncertainty_paths = {}
    for key, values in ensemble_uncertainty.items():
        path_u = pdir / f"{key}_locked_test.npy"
        np.save(path_u, np.asarray(values, dtype=np.float32))
        uncertainty_paths[key] = {"path": str(path_u.resolve()), "sha256": sha256_file(path_u)}
    # Compatibility aliases are exact byte-content derivatives, never independently resolved.
    np.save(pdir / "stacked_test.npy", two_col(stacked)); np.save(pdir / "stacking_test.npy", two_col(stacked))
    np.save(pdir / "soft_test.npy", two_col(soft)); np.save(pdir / "soft_voting_test.npy", two_col(soft))
    members_path = pdir / "selected_pair_locked_test_member_probs.npz"
    np.savez_compressed(members_path, **{m: probs[m].astype(np.float32) for m in members})
    audit = {
        "schema": SCHEMA, "selection_lock_id": pair["selection_lock_id"],
        "deployment_lock_id": deployment["deployment_lock_id"],
        "selected_dl_models": selected, "members": members, "probability_sources": paths,
        "meta_feature_order": pair["meta_features"], "threshold": float(pair["threshold"]),
        "threshold_source": "development_oof", "test_labels_read": False,
        "test_set_used_for_selection": False, "external_set_used_for_selection": False,
        "development_refit_artifacts_required": True, "n_test": int(expected_n),
        "stacked_probability_sha256": sha256_file(stacked_path),
        "soft_probability_sha256": sha256_file(soft_path),
        "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        "ensemble_uncertainty_method": "soft_linear_independence; stacked_logistic_delta_method_independence",
        "uncertainty_artifacts": uncertainty_paths,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(audit, _reports_dir(results_dir) / "selected_pair_locked_test_fusion_audit.json")
    deployment["probability_artifacts"]["stacked:test"] = {"path": str(stacked_path.resolve()), "sha256": sha256_file(stacked_path)}
    deployment["probability_artifacts"]["soft:test"] = {"path": str(soft_path.resolve()), "sha256": sha256_file(soft_path)}
    for key, rec in uncertainty_paths.items():
        deployment["probability_artifacts"][f"uncertainty:{key}:test"] = rec
    deployment["deployment_status"] = "LOCKED_TEST_INFERRED"
    deployment["locked_test_inferred_at"] = audit["created_at"]
    save_json(deployment, _reports_dir(results_dir) / DEPLOYMENT_MANIFEST_FILENAME)
    pair["final_test_evaluated_once"] = True
    pair["locked_test_probabilities_generated_at"] = audit["created_at"]
    pair["locked_test_labels_read_during_fusion"] = False
    save_json(pair, _reports_dir(results_dir) / "dl_pair_selection_manifest.json")
    lock = load_experiment_lock(results_dir)
    lock.update({"test_opened": True, "selection_immutable": True,
                 "test_opened_at": audit["created_at"], "selection_lock_id": pair["selection_lock_id"]})
    save_json(lock, _reports_dir(results_dir) / FINAL_LOCK_FILENAME)
    return {"manifest": pair, "deployment": deployment, "stacked_test": two_col(stacked),
            "soft_test": two_col(soft), "member_probs": probs}



def selected_pair_markdown(results_dir: Path) -> str:
    results_dir = Path(results_dir)
    manifest = load_pair_manifest(results_dir)
    table_path = results_dir / "reports" / "dl_pair_selection_validation.csv"
    df = pd.read_csv(table_path)
    lines = [
        "# Validation-only DL Pair Selection",
        "",
        f"Selected pair: **{manifest['selected_pair_display']}**.",
        "Selection source: patient-level five-fold development OOF probabilities. Locked test and external datasets were not used.",
        "",
        "| Pair | Mean fold AUC | Pooled OOF AUC | OOF AP | Brier | ECE | Fold AUC SD | ΔAUC vs best single | DeLong p | Corr. | Disagreement | Error overlap | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['display_pair']} | {r['fold_auc_mean']:.4f} | {r['oof_auc']:.4f} | {r['oof_auprc']:.4f} | {r['brier']:.4f} | "
            f"{r['ece']:.4f} | {r['fold_auc_sd']:.4f} | {r['delta_auc_vs_best_individual']:+.4f} | "
            f"{r['delong_p_vs_best_individual']:.4f} | {r['pearson_probability']:.4f} | "
            f"{r['disagreement_rate']:.4f} | {r['error_overlap_rate']:.4f} | {'Yes' if bool(r['selected']) else 'No'} |"
        )
    text = "\n".join(lines) + "\n"
    (results_dir / "reports" / "Q1_DL_PAIR_SELECTION_TABLE.md").write_text(text, encoding="utf-8")
    return text

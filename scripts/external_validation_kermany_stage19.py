#!/usr/bin/env python3
"""Locked Kermany external validation for AURA-CXR (evidence-complete run).

The file name is retained from the original development tree so that the recorded
provenance hashes stay resolvable.

This module reuses the immutable development deployment and development-OOF
thresholds.  It never selects a model, tunes a threshold, or recalibrates on
Kermany.  In addition to inference, it writes the full evidence set: per-sample
probabilities, uncertainty, confusion matrices, calibration diagnostics, plots,
dataset provenance, and artifact hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import external_validation_kermany as base
from aura_dl_pair_selection import (
    DISPLAY_NAMES,
    PREDICTIVE_UNCERTAINTY_DEFINITION,
    ensemble_predictive_uncertainty,
    load_deployment_manifest,
    load_pair_manifest,
    normalize_binary_probability,
    resolve_locked_model_artifact,
    sha256_file,
)

SCHEMA = "aura_cxr_external_kermany_stage19_v1"
MODEL_ORDER_SUFFIX = ("radiomics", "stacked_selected_pair", "soft_selected_pair")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_sample_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.replace("\\", "/").encode("utf-8")).hexdigest()[:20]


def resolve_test_root(image_dir: str | Path) -> Path:
    root = Path(image_dir).expanduser().resolve()
    direct = root / "NORMAL"
    return root if direct.is_dir() else root / "test"


def expected_output_paths(results: Path) -> Sequence[Path]:
    reports = results / "reports"
    plots = results / "plots"
    return (
        reports / "external_validation_kermany_protocol.json",
        reports / "external_validation_kermany_locked_selected_pair.csv",
        reports / "external_validation_kermany_predictions.csv",
        plots / "external_validation_kermany_confusion_matrices.png",
    )


def guard_or_clear_previous_outputs(results: Path, overwrite: bool) -> None:
    existing = [p for p in expected_output_paths(results) if p.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Locked validation output already exists. Refusing to overwrite evidence. "
            "Set OVERWRITE_STAGE19=1 only for an intentional deterministic rerun. Existing: "
            + ", ".join(str(p) for p in existing)
        )
    if not overwrite:
        return
    exact_patterns = (
        "reports/external_validation_kermany_*",
        "plots/external_validation_kermany_*",
    )
    for pattern in exact_patterns:
        for path in results.glob(pattern):
            if path.name == "external_validation_kermany_preflight.json":
                continue
            if path.is_file():
                path.unlink()
    probability_dir = results / "probs" / "external_kermany"
    if probability_dir.is_dir():
        shutil.rmtree(probability_dir)


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    try:
        import tensorflow as tf

        # The thesis-final locked pair is PyTorch-only (XRV + EVA-X). TensorFlow
        # is imported solely because train_aura owns the radiomics extractor;
        # keep it off the GPU so it cannot reserve memory needed by PyTorch.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError(
                "TensorFlow GPU visibility was initialized before this script could isolate it"
            ) from exc
        tf.random.set_seed(seed)
    except ImportError:
        pass


def ece_uniform(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    total = len(y)
    value = 0.0
    for i in range(n_bins):
        mask = bin_id == i
        if mask.any():
            value += float(mask.sum()) / total * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def metric_values(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "F1_pneumonia": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "MCC": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "PPV": float(tp / max(tp + fp, 1)),
        "NPV": float(tn / max(tn + fn, 1)),
        "Brier": float(brier_score_loss(y, p)),
        "NLL": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ECE_10_uniform": ece_uniform(y, p, 10),
    }


def stratified_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> Mapping[str, Sequence[float]]:
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both Kermany classes are required for stratified bootstrap")
    keys = tuple(metric_values(y, p, threshold))
    values = {key: [] for key in keys}
    for _ in range(int(n_boot)):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        current = metric_values(y[idx], p[idx], threshold)
        for key in keys:
            values[key].append(current[key])
    return {
        key: (float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5)))
        for key, value in values.items()
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, model_key: str, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = bin_id == i
        if not mask.any():
            continue
        rows.append(
            {
                "model_key": model_key,
                "bin": i,
                "lower": edges[i],
                "upper": edges[i + 1],
                "n": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_fraction": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def infer_locked_models(args, results: Path, df: pd.DataFrame, pair: dict, deployment: dict):
    selected = list(pair["selected_dl_models"])
    train_module = base.import_module(args.train_script, "aura_train_stage19")
    probabilities: Dict[str, np.ndarray] = {}
    standard_deviations: Dict[str, np.ndarray] = {}
    interval_widths: Dict[str, np.ndarray] = {}

    for name in selected:
        model_path = resolve_locked_model_artifact(results, name)
        print(f"[INFERENCE] {name}: {model_path}", flush=True)
        if name in {"efficientnetv2", "resnet50"}:
            p, std, width = base.keras_infer(name, model_path, df, args, train_module)
        elif name == "xrv":
            p, std, width = base.xrv_infer(model_path, df, args)
        elif name == "eva_x":
            p, std, width = base.eva_infer(model_path, df, args)
        else:
            raise ValueError(f"Unsupported locked model: {name}")
        probabilities[name] = normalize_binary_probability(p, len(df), name)
        standard_deviations[name] = np.asarray(std, dtype=np.float32).reshape(-1)
        interval_widths[name] = np.asarray(width, dtype=np.float32).reshape(-1)

    radiomics_path = resolve_locked_model_artifact(results, "radiomics")
    print(f"[INFERENCE] radiomics: {radiomics_path}", flush=True)
    probabilities["radiomics"] = base.radiomics_infer(
        radiomics_path, df, train_module, deployment
    )
    standard_deviations["radiomics"] = np.zeros(len(df), dtype=np.float32)
    interval_widths["radiomics"] = np.zeros(len(df), dtype=np.float32)

    meta_path = Path(deployment["meta_learner_artifact"]["path"])
    if not meta_path.exists() or sha256_file(meta_path) != deployment["meta_learner_artifact"]["sha256"]:
        raise RuntimeError(f"Meta-learner missing or hash mismatch: {meta_path}")
    feature_order = list(deployment.get("meta_feature_order", []))
    expected_order = [f"p_{selected[0]}", f"p_{selected[1]}", "p_radiomics"]
    if feature_order != expected_order:
        raise RuntimeError(f"Locked meta-feature order mismatch: {feature_order} != {expected_order}")
    X = np.column_stack(
        [probabilities[selected[0]], probabilities[selected[1]], probabilities["radiomics"]]
    )
    meta = joblib.load(meta_path)
    probabilities["stacked_selected_pair"] = normalize_binary_probability(
        meta.predict_proba(X), len(df), "stacked_selected_pair"
    )
    weights = deployment["soft_voting_weights"]
    members = selected + ["radiomics"]
    probabilities["soft_selected_pair"] = np.asarray(
        sum(float(weights[name]) * probabilities[name] for name in members), dtype=np.float64
    )
    propagated = ensemble_predictive_uncertainty(
        meta, probabilities["stacked_selected_pair"], members, standard_deviations, weights
    )
    standard_deviations["stacked_selected_pair"] = propagated["stacked_std"]
    standard_deviations["soft_selected_pair"] = propagated["soft_std"]
    interval_widths["stacked_selected_pair"] = propagated[
        "stacked_predictive_interval_width_95"
    ]
    interval_widths["soft_selected_pair"] = propagated[
        "soft_predictive_interval_width_95"
    ]
    return selected, probabilities, standard_deviations, interval_widths


def model_thresholds(selected: Sequence[str], deployment: dict) -> Dict[str, float]:
    values = {k: float(v) for k, v in deployment.get("individual_thresholds", {}).items()}
    values["stacked_selected_pair"] = float(deployment["threshold"])
    values["soft_selected_pair"] = float(deployment["soft_voting_threshold"])
    required = list(selected) + list(MODEL_ORDER_SUFFIX)
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Development-OOF threshold is missing for: {missing}")
    return values


def display_name_map(deployment: dict) -> Dict[str, str]:
    values = dict(DISPLAY_NAMES)
    values["radiomics"] = deployment.get("radiomics_selection", {}).get(
        "selected_display_name", values.get("radiomics", "Radiomics")
    )
    values["stacked_selected_pair"] = "A8 stacked selected-pair ensemble"
    values["soft_selected_pair"] = "Soft-voting selected-pair ensemble"
    return values


def write_dataset_manifest(df: pd.DataFrame, test_root: Path, output_path: Path) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        path = Path(row.image_path).resolve()
        rel = path.relative_to(test_root.resolve()).as_posix()
        rows.append(
            {
                "sample_id": stable_sample_id(rel),
                "relative_path": rel,
                "class_name": "PNEUMONIA" if int(row.label) == 1 else "NORMAL",
                "label": int(row.label),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    if manifest["sample_id"].duplicated().any() or manifest["relative_path"].duplicated().any():
        raise RuntimeError("Duplicate sample identity detected in Kermany test manifest")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def plot_confusion_matrices(
    y: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    display_names: Mapping[str, str],
    order: Sequence[str],
    output_path: Path,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=180)
    for ax, name in zip(axes.flat, order):
        pred = (probabilities[name] >= thresholds[name]).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "model_key": name,
                "model": display_names[name],
                "threshold_internal": thresholds[name],
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "n": int(cm.sum()),
            }
        )
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=11)
        ax.set_title(display_names[name], fontsize=9)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
    for ax in axes.flat[len(order) :]:
        ax.axis("off")
    fig.suptitle("Confusion Matrix Validasi Eksternal Kermany (Ambang Development OOF)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_calibration_and_discrimination(
    y: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    display_names: Mapping[str, str],
    order: Sequence[str],
    calibration_path: Path,
    roc_pr_path: Path,
) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration_rows = []
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=180)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Kalibrasi ideal")
    for name in order:
        prob_true, prob_pred = calibration_curve(y, probabilities[name], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.4, markersize=4, label=display_names[name])
        calibration_rows.append(calibration_bins(y, probabilities[name], name, 10))
    ax.set(xlabel="Rerata probabilitas prediksi", ylabel="Proporsi pneumonia teramati", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Kermany Reliability Diagram (Descriptive Audit; No Recalibration)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(calibration_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=180)
    for name in order:
        fpr, tpr, _ = roc_curve(y, probabilities[name])
        precision, recall, _ = precision_recall_curve(y, probabilities[name])
        axes[0].plot(fpr, tpr, linewidth=1.5, label=f"{display_names[name]} ({roc_auc_score(y, probabilities[name]):.3f})")
        axes[1].plot(recall, precision, linewidth=1.5, label=f"{display_names[name]} ({average_precision_score(y, probabilities[name]):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC eksternal Kermany", xlim=(0, 1), ylim=(0, 1))
    axes[1].axhline(float(y.mean()), color="k", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall eksternal Kermany", xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(roc_pr_path, bbox_inches="tight")
    plt.close(fig)
    return pd.concat(calibration_rows, ignore_index=True)


def write_probability_artifacts(
    results: Path,
    dataset_manifest: pd.DataFrame,
    y: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    standard_deviations: Mapping[str, np.ndarray],
    interval_widths: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    order: Sequence[str],
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    probability_dir = results / "probs" / "external_kermany"
    probability_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = np.asarray(dataset_manifest["sample_id"].astype(str).tolist(), dtype="U20")
    np.save(probability_dir / "labels.npy", y.astype(np.int8))
    np.save(probability_dir / "sample_ids.npy", sample_ids)
    combined = {"labels": y.astype(np.int8), "sample_ids": sample_ids}
    predictions = dataset_manifest[["sample_id", "relative_path", "label"]].copy()
    uncertainty = dataset_manifest[["sample_id", "relative_path", "label"]].copy()
    for name in order:
        p = np.asarray(probabilities[name], dtype=np.float32)
        std = np.asarray(standard_deviations[name], dtype=np.float32)
        width = np.asarray(interval_widths[name], dtype=np.float32)
        np.save(probability_dir / f"{name}_mean_probability.npy", p)
        np.save(probability_dir / f"{name}_mc_std.npy", std)
        np.save(probability_dir / f"{name}_predictive_interval_width_95.npy", width)
        combined[f"{name}_mean_probability"] = p
        combined[f"{name}_mc_std"] = std
        combined[f"{name}_predictive_interval_width_95"] = width
        predictions[f"{name}_probability"] = p
        predictions[f"{name}_prediction"] = (p >= thresholds[name]).astype(np.int8)
        uncertainty[f"{name}_mean_probability"] = p
        uncertainty[f"{name}_mc_std"] = std
        uncertainty[f"{name}_predictive_interval_width_95"] = width
    combined_path = probability_dir / "external_validation_kermany_probabilities.npz"
    np.savez_compressed(combined_path, **combined)
    return combined_path, predictions, uncertainty


def runtime_information(args) -> dict:
    data = {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_requested": bool(args.gpu),
        "batch_size": int(args.batch_size),
        "mc_passes": int(args.mc_passes),
        "num_workers": int(args.num_workers),
        "bootstrap_repetitions": int(args.n_boot),
        "seed": int(args.seed),
        "tensorflow_gpu_disabled_for_pytorch_locked_pair": True,
    }
    try:
        import torch

        data["torch_version"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        data["torch_error"] = str(exc)
    try:
        import tensorflow as tf

        data["tensorflow_version"] = tf.__version__
    except Exception:
        data["tensorflow_version"] = None
    return data


def hashes_for(paths: Iterable[Path], base_dir: Path) -> dict:
    records = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        records[path.relative_to(base_dir.resolve()).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--train_script", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--mc_passes", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--xrv_script", required=True)
    parser.add_argument("--eva_x_script", required=True)
    parser.add_argument("--eva_x_repo", required=True)
    parser.add_argument("--eva_x_checkpoint", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size != 224:
        raise ValueError("The locked AURA-CXR external protocol requires image_size=224")
    if args.mc_passes != 30:
        raise ValueError("The locked AURA-CXR external protocol requires exactly 30 MC-Dropout passes")
    if args.n_boot < 1000:
        raise ValueError("The final locked validation requires at least 1000 bootstrap repetitions")

    results = Path(args.results_dir).expanduser().resolve()
    reports = results / "reports"
    plots = results / "plots"
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    guard_or_clear_previous_outputs(results, args.overwrite)
    set_reproducibility(args.seed)

    pair = load_pair_manifest(results, strict=True)
    deployment = load_deployment_manifest(results, strict=True)
    if not pair.get("final_test_evaluated_once"):
        raise RuntimeError("External validation is allowed only after the internal locked test is permanently frozen")
    experiment_lock_path = reports / "q1_final_experiment_lock.json"
    if not experiment_lock_path.exists():
        raise FileNotFoundError(experiment_lock_path)
    experiment_lock = json.loads(experiment_lock_path.read_text(encoding="utf-8"))
    if experiment_lock.get("test_opened") is not True or experiment_lock.get("selection_immutable") is not True:
        raise RuntimeError("Final experiment lock is not immutable/opened as required")
    if experiment_lock.get("selection_lock_id") != pair.get("selection_lock_id"):
        raise RuntimeError("Final experiment lock does not match the selected pair")

    df = base.build_kermany_df(args.image_dir, max_samples=0)
    test_root = resolve_test_root(args.image_dir)
    y = df["label"].to_numpy(dtype=int)
    print(
        f"[KERMANY] official test-only scope: n={len(y)}, NORMAL={(y == 0).sum()}, "
        f"PNEUMONIA={(y == 1).sum()}, prevalence={(y == 1).mean():.4f}",
        flush=True,
    )
    print("[POLICY] thresholds/calibration/model selection are frozen from development OOF", flush=True)

    dataset_manifest_path = reports / "external_validation_kermany_dataset_manifest.csv"
    dataset_manifest = write_dataset_manifest(df, test_root, dataset_manifest_path)
    # Align inference order with the stable, sorted manifest order.
    path_lookup = {Path(p).resolve().relative_to(test_root).as_posix(): (str(Path(p).resolve()), int(label)) for p, label in zip(df["image_path"], df["label"])}
    aligned = [path_lookup[rel] for rel in dataset_manifest["relative_path"]]
    df = pd.DataFrame({"image_path": [x[0] for x in aligned], "label": [x[1] for x in aligned]})
    y = df["label"].to_numpy(dtype=int)

    selected, probabilities, standard_deviations, interval_widths = infer_locked_models(
        args, results, df, pair, deployment
    )
    order = list(selected) + list(MODEL_ORDER_SUFFIX)
    thresholds = model_thresholds(selected, deployment)
    names = display_name_map(deployment)

    metric_rows = []
    calibration_summary_rows = []
    for index, name in enumerate(order):
        p = np.asarray(probabilities[name], dtype=float)
        current = metric_values(y, p, thresholds[name])
        ci = stratified_bootstrap_ci(y, p, thresholds[name], args.n_boot, args.seed + index)
        row = {
            "model_key": name,
            "model": names[name],
            "threshold_internal": thresholds[name],
            "threshold_source": "development_oof",
            "n": len(y),
            "scope": "official_kermany_test",
            "artifact_role": "exact_locked_development_refit",
            "predictive_interval_width_95_mean": float(np.mean(interval_widths[name])),
            "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        }
        for key, value in current.items():
            row[key] = value
            row[f"{key}_CI_low"] = ci[key][0]
            row[f"{key}_CI_high"] = ci[key][1]
        metric_rows.append(row)
        calibration_summary_rows.append(
            {
                "model_key": name,
                "model": names[name],
                "n": len(y),
                "Brier": current["Brier"],
                "NLL": current["NLL"],
                "ECE_10_uniform": current["ECE_10_uniform"],
                "assessment_only": True,
                "external_recalibration_performed": False,
            }
        )
    metrics_path = reports / "external_validation_kermany_locked_selected_pair.csv"
    calibration_summary_path = reports / "external_validation_kermany_calibration.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(calibration_summary_rows).to_csv(calibration_summary_path, index=False)

    cm_plot = plots / "external_validation_kermany_confusion_matrices.png"
    cm_csv = reports / "external_validation_kermany_confusion_matrices.csv"
    plot_confusion_matrices(y, probabilities, thresholds, names, order, cm_plot).to_csv(cm_csv, index=False)

    calibration_plot = plots / "external_validation_kermany_calibration.png"
    roc_pr_plot = plots / "external_validation_kermany_roc_pr.png"
    calibration_bins_path = reports / "external_validation_kermany_calibration_bins.csv"
    plot_calibration_and_discrimination(
        y, probabilities, names, order, calibration_plot, roc_pr_plot
    ).to_csv(calibration_bins_path, index=False)

    combined_probability_path, predictions, uncertainty = write_probability_artifacts(
        results, dataset_manifest, y, probabilities, standard_deviations, interval_widths,
        thresholds, order
    )
    predictions_path = reports / "external_validation_kermany_predictions.csv"
    uncertainty_path = reports / "external_validation_kermany_predictive_uncertainty.csv"
    predictions.to_csv(predictions_path, index=False)
    uncertainty.to_csv(uncertainty_path, index=False)

    runtime_path = reports / "external_validation_kermany_runtime.json"
    runtime_path.write_text(json.dumps(runtime_information(args), indent=2), encoding="utf-8")

    output_paths = [
        dataset_manifest_path,
        metrics_path,
        calibration_summary_path,
        calibration_bins_path,
        cm_csv,
        predictions_path,
        uncertainty_path,
        runtime_path,
        combined_probability_path,
        cm_plot,
        calibration_plot,
        roc_pr_plot,
    ]
    output_paths.extend((results / "probs" / "external_kermany").glob("*.npy"))
    source_hashes = {
        name: {
            "path": rec["path"],
            "sha256": rec["sha256"],
            "role": rec.get("role"),
        }
        for name, rec in deployment["base_model_artifacts"].items()
    }
    source_hashes["meta_learner"] = dict(deployment["meta_learner_artifact"])
    protocol = {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "dataset": "Kermany pediatric chest X-ray",
        "scope": "official_test_partition_only",
        "n": int(len(y)),
        "normal": int((y == 0).sum()),
        "pneumonia": int((y == 1).sum()),
        "prevalence": float((y == 1).mean()),
        "dataset_manifest": {
            "path": dataset_manifest_path.relative_to(results).as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "selection_lock_id": pair["selection_lock_id"],
        "deployment_lock_id": deployment["deployment_lock_id"],
        "selected_dl_models": selected,
        "selected_radiomics_estimator": deployment.get("radiomics_selection", {}).get("selected_estimator"),
        "meta_feature_order": deployment.get("meta_feature_order"),
        "thresholds": {name: thresholds[name] for name in order},
        "threshold_source": "development_oof",
        "external_threshold_tuned": False,
        "external_recalibration_performed": False,
        "external_set_used_for_selection": False,
        "external_set_used_for_tuning": False,
        "exact_internal_deployment_reused": True,
        "image_size": 224,
        "mc_passes": int(args.mc_passes),
        "bootstrap": {
            "method": "stratified_percentile",
            "repetitions": int(args.n_boot),
            "confidence_level": 0.95,
            "seed": int(args.seed),
        },
        "calibration_audit": {
            "descriptive_only": True,
            "ece_bins": 10,
            "ece_strategy": "uniform",
            "no_parameters_applied_back_to_predictions": True,
        },
        "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        "ensemble_uncertainty_method": "soft_linear_independence; stacked_logistic_delta_method_independence",
        "source_artifacts": source_hashes,
        "generated_artifacts": hashes_for(output_paths, results),
    }
    protocol_path = reports / "external_validation_kermany_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print(pd.DataFrame(metric_rows)[["model", "AUC", "AUPRC", "sensitivity", "specificity", "balanced_acc", "Brier", "ECE_10_uniform"]].to_string(index=False), flush=True)
    print(f"[DONE] Stage 19 evidence written under: {results}", flush=True)


if __name__ == "__main__":
    main()

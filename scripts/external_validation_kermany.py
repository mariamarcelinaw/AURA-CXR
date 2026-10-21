#!/usr/bin/env python3
"""Exact locked Kermany-only external validation for AURA-CXR.

The exact two selected DL development-refit models, the selected best radiomics
learner, meta-learner, feature order, calibration state, and development-OOF
threshold are reused without tuning on Kermany.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from aura_dl_pair_selection import (
    DISPLAY_NAMES,
    load_deployment_manifest,
    load_pair_manifest,
    normalize_binary_probability,
    resolve_locked_model_artifact,
    sha256_file,
    ensemble_predictive_uncertainty,
    predictive_interval_width_95,
    PREDICTIVE_UNCERTAINTY_DEFINITION,
)

RNG = np.random.default_rng(42)


def import_module(path, name):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "F1_pneumonia": f1_score(y, pred, zero_division=0),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
    }


def boot_ci(y, p, thr, n_boot):
    keys = list(metrics(y, p, thr))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    acc = {k: [] for k in keys}
    for _ in range(int(n_boot)):
        idx = np.concatenate([
            RNG.choice(pos, len(pos), replace=True),
            RNG.choice(neg, len(neg), replace=True),
        ])
        m = metrics(y[idx], p[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in acc.items()
    }


def build_kermany_df(image_dir: str, max_samples: int) -> pd.DataFrame:
    """Load only the official Kermany test partition.

    ``image_dir`` may point either to the dataset root containing ``test/`` or
    directly to the ``test`` directory. Train/val/all scopes are intentionally
    unsupported in the locked external protocol.
    """
    root = Path(image_dir)
    base = root if (root / "NORMAL").is_dir() and (root / "PNEUMONIA").is_dir() else root / "test"
    rows = []
    for label, name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        class_dir = base / name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Kermany test class directory not found: {class_dir}. "
                "Expected <image_dir>/test/NORMAL and PNEUMONIA, or image_dir itself to be the test folder."
            )
        files = []
        for pat in ("*.jpeg", "*.jpg", "*.png"):
            files.extend(class_dir.rglob(pat))
        rows.extend({"image_path": str(x), "label": label} for x in sorted(set(files)))
    df = pd.DataFrame(rows)
    if df.empty or len(df["label"].unique()) < 2:
        raise ValueError("Kermany test set is empty or has only one class")
    if max_samples and len(df) > int(max_samples):
        parts = []
        for _, g in df.groupby("label"):
            n = max(1, round(int(max_samples) * len(g) / len(df)))
            parts.append(g.sample(min(n, len(g)), random_state=42))
        df = pd.concat(parts, ignore_index=True)
    return df.reset_index(drop=True)

def load_external_gray(path, size=224):
    """Read Kermany JPEG/PNG safely; DICOM support is retained for unit tests."""
    from PIL import Image
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        import pydicom
        ds = pydicom.dcmread(str(path))
        a = ds.pixel_array.astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            a = a.max() - a
    else:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    a = np.zeros_like(a, dtype=np.float32) if not np.isfinite(lo + hi) or hi <= lo else (a - lo) / (hi - lo)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size)), dtype=np.float32) / 255.0


def load_gray(path, size, train_module):
    return load_external_gray(path, size)


def keras_infer(name, model_path, df, args, train_module):
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    model = train_module.prepare_mc_dropout_inference(model)
    bn_before = train_module.batchnorm_state_snapshot(model)
    preprocess = train_module.get_preprocess_for_backbone(name)
    mean_batches, std_batches = [], []
    for i in range(0, len(df), args.batch_size):
        paths = df.image_path.iloc[i:i + args.batch_size]
        rgb = np.stack([
            np.repeat(load_gray(p, 224, train_module)[..., None], 3, axis=-1)
            for p in paths
        ]).astype(np.float32)
        x = preprocess(tf.constant(rgb))
        draws = np.stack(
            [np.asarray(model(x, training=True), dtype=np.float32) for _ in range(args.mc_passes)],
            axis=0,
        )
        mean_batches.append(draws.mean(axis=0)[:, 1])
        std_batches.append(draws.std(axis=0)[:, 1])
    train_module.assert_batchnorm_state_unchanged(bn_before, model)
    del model
    tf.keras.backend.clear_session()
    mean = np.concatenate(mean_batches); std = np.concatenate(std_batches)
    return mean, std, predictive_interval_width_95(std)


class _ExternalXrvDataset:
    def __init__(self, df, xrv):
        self.paths = df["image_path"].astype(str).tolist()
        self.labels = df["label"].to_numpy(dtype=np.int64)
        self.xrv = xrv
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        import torch
        g = load_external_gray(self.paths[i], 224)
        x = self.xrv.datasets.normalize((g * 255.0).astype(np.float32), 255)
        return torch.from_numpy(x[None, ...].astype(np.float32)), int(self.labels[i])


def _external_xrv_loader(df, xrv, args):
    from torch.utils.data import DataLoader
    return DataLoader(
        _ExternalXrvDataset(df, xrv), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=bool(args.gpu),
        persistent_workers=args.num_workers > 0,
    )


def load_xrv_binary_state_dict_strict(model, checkpoint):
    """Load the AURA-CXR binary XRV checkpoint with an audited XRV exception.

    ``torchxrayvision.models.DenseNet(weights=None)`` does not register the
    pretrained multi-label operating-threshold buffer in some XRV releases,
    while a checkpoint created from the pretrained model contains
    ``backbone.op_threshs``.  AURA-CXR never uses that buffer: its forward path
    extracts ``backbone.features`` and applies the separately trained binary
    head.  Remove only this known, non-parameter buffer and retain strict
    loading for every trainable parameter and all other persistent buffers.
    """
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise TypeError("XRV checkpoint state_dict must be a mapping")

    state = dict(raw_state)
    ignored = []
    key = "backbone.op_threshs"
    if key in state and key not in model.state_dict():
        value = state.pop(key)
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        ignored.append({"key": key, "shape": shape})

    # Deliberately strict after the single audited compatibility removal.
    # Any architecture, head, parameter, or unknown-buffer mismatch still
    # aborts external validation.
    model.load_state_dict(state, strict=True)
    if ignored:
        print(
            "[XRV][CHECKPOINT_COMPAT] ignored non-inference buffer(s): "
            + ", ".join(f"{item['key']} shape={item['shape']}" for item in ignored),
            flush=True,
        )
    return ignored


def xrv_infer(model_path, df, args):
    import torch
    import torchxrayvision as xrv
    mod = import_module(args.xrv_script, "aura_xrv")
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    # Development-refit checkpoints are plain state dictionaries. Explicit
    # weights_only avoids arbitrary pickle object loading on recent PyTorch.
    try:
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older torch versions.
        ck = torch.load(model_path, map_location="cpu")
    dropout = float(ck.get("dropout", 0.3))
    model = mod.build_model(xrv, torch, dropout=dropout, weights=None).to(device)
    load_xrv_binary_state_dict_strict(model, ck)
    dl = _external_xrv_loader(df, xrv, args)
    mean, std, width = mod.mc_predict(model, dl, torch, args.mc_passes, device)
    p = mean[:, 1]; s = std[:, 1]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, s, width


class _ExternalEvaDataset:
    def __init__(self, df, mean, std):
        self.df = df.reset_index(drop=True)
        self.mean = np.asarray(mean, dtype=np.float32)[:, None, None]
        self.std = np.asarray(std, dtype=np.float32)[:, None, None]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        import torch
        row = self.df.iloc[i]
        g = load_external_gray(str(row.image_path), 224)
        rgb = np.repeat(g[None, ...], 3, axis=0)
        x = (rgb - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), int(row.label)


def _external_eva_loader(df, args):
    from torch.utils.data import DataLoader
    ds = _ExternalEvaDataset(df, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=bool(args.gpu), persistent_workers=args.num_workers > 0,
    )


def eva_infer(model_path, df, args):
    import torch
    mod = import_module(args.eva_x_script, "aura_eva")
    try:
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older torch versions.
        ck = torch.load(model_path, map_location="cpu")
    repo = args.eva_x_repo
    if not repo:
        mf = Path(args.results_dir) / "reports" / "eva_x_development_refit_manifest.json"
        if mf.exists():
            repo = json.loads(mf.read_text(encoding="utf-8")).get("official_repo", "")
    if not repo:
        raise ValueError("Selected EVA-X requires --eva_x_repo or its refit manifest")
    checkpoint = args.eva_x_checkpoint or ck.get("checkpoint", "")
    if not checkpoint:
        raise ValueError("EVA-X initialization checkpoint is unavailable")
    official = mod.import_eva_x(Path(repo))
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    args.pretrained_checkpoint = checkpoint
    args.image_size = 224
    args.mean = (0.5, 0.5, 0.5)
    args.std = (0.5, 0.5, 0.5)
    args.dropout = float(ck.get("dropout", 0.2))
    model = mod.build_model(args, official, torch).to(device)
    model.load_state_dict(ck.get("state_dict", ck))
    mean, std, width = mod.predict(model, _external_eva_loader(df, args), torch, device, args.mc_passes)
    p = mean[:, 1]; s = std[:, 1]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return p, s, width


def _radiomics_feature_config(deployment: dict) -> dict:
    cfg = dict(deployment.get("radiomics_selection", {}).get("feature_config", {}))
    required = ["wavelet_levels", "glcm_distances", "glcm_angles_deg", "lbp_radius", "lbp_n_points"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"Locked deployment lacks radiomics feature configuration: {missing}")
    if int(cfg.get("image_size", 224)) != 224:
        raise RuntimeError("Locked radiomics feature configuration is not 224x224")
    return cfg


def radiomics_infer(model_path, df, train_module, deployment):
    model = joblib.load(model_path)
    selected_estimator = deployment.get("radiomics_selection", {}).get("selected_estimator")
    if not selected_estimator:
        raise RuntimeError("Locked deployment does not identify the selected radiomics estimator")
    train_module.assert_radiomics_estimator_family(model, selected_estimator)
    cfg = _radiomics_feature_config(deployment)
    features = []
    for path in df.image_path:
        g = load_gray(path, 224, train_module)
        feat, _ = train_module.extract_adaptive_wavelet_glcm_lbp(
            g,
            distances=[int(x) for x in cfg["glcm_distances"]],
            angles_deg=[float(x) for x in cfg["glcm_angles_deg"]],
            wavelet_levels=int(cfg["wavelet_levels"]),
            lbp_radius=int(cfg["lbp_radius"]),
            lbp_n_points=int(cfg["lbp_n_points"]),
        )
        features.append(feat)
    X = np.stack(features).astype(np.float32)
    expected_dim = int(cfg.get("feature_dimension", X.shape[1]))
    if X.shape[1] != expected_dim:
        raise RuntimeError(f"Kermany radiomics feature dimension mismatch: {X.shape[1]} != {expected_dim}")
    return normalize_binary_probability(model.predict_proba(X), len(df), "Kermany radiomics")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--train_script", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mc_passes", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--xrv_script", default="train_xrv_backbone.py")
    ap.add_argument("--eva_x_script", default="train_eva_x_backbone.py")
    ap.add_argument("--eva_x_repo", default="")
    ap.add_argument("--eva_x_checkpoint", default="")
    args = ap.parse_args()

    if args.image_size != 224:
        raise ValueError("Locked Kermany external validation is fixed at 224x224")
    results = Path(args.results_dir)
    pair = load_pair_manifest(results, strict=True)
    deployment = load_deployment_manifest(results, strict=True)
    if not pair.get("final_test_evaluated_once"):
        raise RuntimeError(
            "Kermany validation is permitted only after the internal locked test permanently freezes selection"
        )
    selected = list(pair["selected_dl_models"])
    train_module = import_module(args.train_script, "aura_train")
    df = build_kermany_df(args.image_dir, args.max_samples)
    y = df.label.to_numpy(dtype=int)
    print(f"[KERMANY] scope=test n={len(df)} pneumonia={(y==1).sum()} non={(y==0).sum()} prevalence={(y==1).mean():.2%}")
    print("[KERMANY] threshold policy: development OOF only; no external tuning")

    probs, stds, predictive_widths = {}, {}, {}
    for name in selected:
        model_path = resolve_locked_model_artifact(results, name)
        if name in {"efficientnetv2", "resnet50"}:
            p, s, width = keras_infer(name, model_path, df, args, train_module)
        elif name == "xrv":
            p, s, width = xrv_infer(model_path, df, args)
        elif name == "eva_x":
            p, s, width = eva_infer(model_path, df, args)
        else:
            raise ValueError(name)
        probs[name] = normalize_binary_probability(p, len(df), name)
        stds[name] = np.asarray(s, dtype=np.float32).reshape(-1)
        predictive_widths[name] = np.asarray(width, dtype=np.float32).reshape(-1)

    rad_path = resolve_locked_model_artifact(results, "radiomics")
    probs["radiomics"] = radiomics_infer(rad_path, df, train_module, deployment)
    stds["radiomics"] = np.zeros(len(df), dtype=np.float32)
    predictive_widths["radiomics"] = np.zeros(len(df), dtype=np.float32)
    meta_path = Path(deployment["meta_learner_artifact"]["path"])
    if sha256_file(meta_path) != deployment["meta_learner_artifact"]["sha256"]:
        raise RuntimeError("Meta-learner hash mismatch")
    X = np.column_stack([
        probs[selected[0]],
        probs[selected[1]],
        probs["radiomics"],
    ])
    meta = joblib.load(meta_path)
    probs["stacked_selected_pair"] = meta.predict_proba(X)[:, 1]
    soft_weights = deployment["soft_voting_weights"]
    member_names = selected + ["radiomics"]
    probs["soft_selected_pair"] = sum(float(soft_weights[m]) * probs[m] for m in member_names)
    ens_unc = ensemble_predictive_uncertainty(
        meta, probs["stacked_selected_pair"], member_names, stds, soft_weights
    )
    stds["stacked_selected_pair"] = ens_unc["stacked_std"]
    stds["soft_selected_pair"] = ens_unc["soft_std"]
    predictive_widths["stacked_selected_pair"] = ens_unc["stacked_predictive_interval_width_95"]
    predictive_widths["soft_selected_pair"] = ens_unc["soft_predictive_interval_width_95"]

    thresholds = {
        **deployment.get("individual_thresholds", {}),
        "stacked_selected_pair": float(deployment["threshold"]),
        "soft_selected_pair": float(deployment["soft_voting_threshold"]),
    }
    display_names = dict(DISPLAY_NAMES)
    display_names["stacked_selected_pair"] = "Stacked selected-pair ensemble"
    display_names["soft_selected_pair"] = "Soft-voting selected-pair ensemble"
    display_names["radiomics"] = deployment.get("radiomics_selection", {}).get(
        "selected_display_name", display_names["radiomics"]
    )
    rows = []
    for name, p in probs.items():
        if name not in thresholds:
            raise KeyError(f"Missing development-OOF threshold for {name}")
        thr = float(thresholds[name])
        m = metrics(y, p, thr)
        ci = boot_ci(y, p, thr, args.n_boot)
        row = {
            "model": display_names.get(name, name),
            "threshold_internal": thr,
            "n": len(y),
            "scope": "test",
            "artifact_role": "exact_locked_development_refit",
            "predictive_interval_width_95_mean": float(np.mean(predictive_widths[name])),
            "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        }
        for k, v in m.items():
            row[k] = v
            row[k + "_CI"] = f"[{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
        rows.append(row)

    reports = results / "reports"
    reports.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(reports / "external_validation_kermany_locked_selected_pair.csv", index=False)
    uncertainty_df = pd.DataFrame({"image_path": df["image_path"], "label": y})
    for name in probs:
        uncertainty_df[f"{name}_mean_probability"] = probs[name]
        uncertainty_df[f"{name}_mc_std"] = stds[name]
        uncertainty_df[f"{name}_predictive_interval_width_95"] = predictive_widths[name]
    uncertainty_df.to_csv(reports / "external_validation_kermany_predictive_uncertainty.csv", index=False)
    audit = {
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "dataset": "kermany",
        "scope": "test",
        "selection_lock_id": pair["selection_lock_id"],
        "deployment_lock_id": deployment["deployment_lock_id"],
        "selected_dl_models": selected,
        "selected_radiomics_estimator": deployment.get("radiomics_selection", {}).get("selected_estimator"),
        "threshold": deployment["threshold"],
        "threshold_source": "development_oof",
        "external_threshold_tuned": False,
        "external_set_used_for_selection": False,
        "exact_internal_deployment_reused": True,
        "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        "ensemble_uncertainty_method": "soft_linear_independence; stacked_logistic_delta_method_independence",
        "n": len(y),
        "prevalence": float((y == 1).mean()),
    }
    (reports / "external_validation_kermany_protocol.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(out[["model", "AUC", "AUPRC", "sensitivity", "specificity", "balanced_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resumable post-hoc Stage 4 adaptive-vs-fixed wavelet ablation.

Scientific logic is intentionally identical to the Stage-4 script supplied in
AURA_CXR_REPRODUCIBILITY_RC1_STAGE4_READY.zip.  The only changes are checkpoint,
resume, progress logging, and atomic state management.

Origin: post_hoc_analysis; selection_eligible=False.
No locked-test or external data are accepted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

try:
    import pydicom
    import pywt
    from PIL import Image
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from lightgbm import LGBMClassifier
except Exception as e:
    raise SystemExit(f"Missing Stage 4 runtime dependency: {e}")

WAVELETS = ["haar", "db4", "sym4", "coif3"]
PROPS = ["contrast", "energy", "homogeneity", "dissimilarity", "correlation", "ASM"]
FEATURE_DIM = 746
EXPECTED_N = 21346
CACHE_VERSION = 1


def sha256_file(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def norm(x):
    x = np.asarray(x, np.float32)
    lo = float(x.min())
    hi = float(x.max())
    return np.zeros_like(x) if np.isclose(lo, hi) else (x - lo) / (hi - lo)


def u8(x):
    return np.clip(norm(x) * 255, 0, 255).astype(np.uint8)


def load_dicom(path, size=224):
    ds = pydicom.dcmread(str(path))
    x = ds.pixel_array.astype(np.float32)
    x = x * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        x = x.max() - x
    x = norm(x)
    im = Image.fromarray((x * 255).astype(np.uint8), mode="L").resize(
        (size, size), resample=Image.Resampling.BILINEAR
    )
    return np.asarray(im, np.float32) / 255.0


def entropy(c):
    c = np.abs(np.asarray(c).ravel())
    c = c[c > 0]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def entropy_all(gray):
    out = {}
    for w in WAVELETS:
        _, (lh, hl, hh) = pywt.dwt2(gray, w)
        out[w] = entropy(np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()]))
    return out


def choose(gray):
    e = entropy_all(gray)
    # Exact tie behavior preserved from original Stage-4 script.
    return max(WAVELETS, key=lambda w: (e[w], -WAVELETS.index(w))), e


def subbands(gray, w):
    d = []
    cur = gray
    for _ in range(3):
        ll, (lh, hl, hh) = pywt.dwt2(cur, w)
        d.extend([lh, hl, hh])
        cur = ll
    return d + [cur]


def features(gray, w):
    vals = []
    for s in subbands(gray, w):
        g = graycomatrix(
            u8(s),
            distances=[1, 2, 3],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )
        for prop in PROPS:
            vals.extend(graycoprops(g, prop).flatten().tolist())
    l = local_binary_pattern(u8(gray), 24, 3, method="uniform")
    h, _ = np.histogram(l.ravel(), bins=26, range=(0, 26), density=True)
    vals.extend(h.tolist())
    a = np.asarray(vals, np.float32)
    if a.shape != (FEATURE_DIM,) or not np.all(np.isfinite(a)):
        raise RuntimeError(f"Invalid feature shape/values: {a.shape}")
    return a


def build_lgbm(params, n_train):
    depth = int(params["max_depth"])
    return LGBMClassifier(
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["max_leaf_nodes"]),
        max_depth=(depth if depth < 12 else -1),
        reg_lambda=float(params["l2_regularization"]),
        min_child_samples=max(5, int(float(params["min_samples_leaf_frac"]) * n_train)),
        n_estimators=400,
        subsample=.9,
        subsample_freq=1,
        colsample_bytree=.9,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def open_memmap(path: Path, dtype, shape, create: bool):
    if create:
        return np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    expected_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not path.exists() or path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Cache file missing/size mismatch: {path} (expected {expected_bytes} bytes)"
        )
    return np.memmap(path, mode="r+", dtype=dtype, shape=shape)


def init_or_resume_cache(cache: Path, split: pd.DataFrame, policies: list[str], split_hash: str, config_hash: str, force_reset: bool):
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / "cache_manifest.json"

    expected_meta = {
        "cache_version": CACHE_VERSION,
        "n": int(len(split)),
        "feature_dim": FEATURE_DIM,
        "policies": policies,
        "split_sha256": split_hash,
        "config_sha256": config_hash,
        "patient_order_sha256": hashlib.sha256("\n".join(split.patientId.astype(str)).encode()).hexdigest(),
    }

    if force_reset and cache.exists():
        for p in cache.iterdir():
            if p.is_file():
                p.unlink()

    create = not meta_path.exists()
    if create:
        atomic_json(meta_path, expected_meta)
    else:
        with open(meta_path, encoding="utf-8") as f:
            actual = json.load(f)
        if actual != expected_meta:
            raise RuntimeError(
                "Existing resume cache does not match current split/config. "
                "Use a different --cache-dir or --reset-cache after verifying inputs."
            )

    n = len(split)
    featmaps = {
        pol: open_memmap(cache / f"features_{pol}.f32", np.float32, (n, FEATURE_DIM), create)
        for pol in policies
    }
    ent = open_memmap(cache / "entropy.f64", np.float64, (n, len(WAVELETS)), create)
    winner_idx = open_memmap(cache / "winner.u1", np.uint8, (n,), create)
    complete = open_memmap(cache / "complete.u1", np.uint8, (n,), create)

    if create:
        complete[:] = 0
        winner_idx[:] = 255
        ent[:] = np.nan
        for mm in featmaps.values():
            mm[:] = np.nan
        for mm in [*featmaps.values(), ent, winner_idx, complete]:
            mm.flush()

    return featmaps, ent, winner_idx, complete, meta_path


def write_progress(out: Path, complete, started: float, last_patient: str | None, phase: str, policy: str | None = None, fold: int | None = None):
    done = int(np.count_nonzero(np.asarray(complete)))
    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (len(complete) - done) / rate if rate > 0 else None
    obj = {
        "phase": phase,
        "policy": policy,
        "fold": fold,
        "completed_cases": done,
        "total_cases": int(len(complete)),
        "percent": 100.0 * done / len(complete),
        "elapsed_seconds_this_run": elapsed,
        "estimated_remaining_seconds": remaining,
        "last_patient_id": last_patient,
        "timestamp_unix": time.time(),
    }
    atomic_json(out / "progress.json", obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--dicom-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="Resume cache; default: <out>/resume_cache")
    ap.add_argument("--checkpoint-every", type=int, default=25, help="Flush checkpoint every N newly completed cases")
    ap.add_argument("--progress-every", type=int, default=10, help="Print progress every N newly completed cases")
    ap.add_argument("--reset-cache", action="store_true", help="Discard compatible/incompatible resume cache and start from zero")
    args = ap.parse_args()

    if args.checkpoint_every < 1 or args.progress_every < 1:
        raise SystemExit("checkpoint/progress intervals must be >= 1")

    split_path = Path(args.split)
    config_path = Path(args.config)
    split = pd.read_csv(split_path)
    root = Path(args.dicom_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir) if args.cache_dir else out / "resume_cache"

    required_cols = {"patientId", "label", "oof_fold"}
    if not required_cols.issubset(split.columns):
        raise SystemExit(f"Split missing columns: {sorted(required_cols - set(split.columns))}")
    if len(split) != EXPECTED_N or split.patientId.nunique() != EXPECTED_N:
        raise SystemExit("Canonical development split mismatch")
    if set(split.oof_fold.astype(int).unique()) != {0, 1, 2, 3, 4}:
        raise SystemExit("Canonical OOF fold set mismatch")

    paths = {pid: root / f"{pid}.dcm" for pid in split.patientId.astype(str)}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {len(missing)}/{EXPECTED_N} canonical RSNA DICOM files; first={missing[0]}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    fold_params = {int(k): v for k, v in cfg["fold_lightgbm_params"].items()}
    policies = list(cfg["policies"])
    expected_policies = ["adaptive", "fixed_haar", "fixed_db4", "fixed_sym4", "fixed_coif3"]
    if policies != expected_policies:
        raise SystemExit(f"Unexpected policy order: {policies}")
    if sorted(fold_params) != [0, 1, 2, 3, 4]:
        raise SystemExit("Missing fold-specific frozen LightGBM parameters")

    split_hash = sha256_file(split_path)
    config_hash = sha256_file(config_path)
    featmaps, ent_mm, winner_mm, complete_mm, cache_manifest = init_or_resume_cache(
        cache, split, policies, split_hash, config_hash, args.reset_cache
    )

    started = time.time()
    already = int(np.count_nonzero(np.asarray(complete_mm)))
    print(f"[RESUME] cache={cache}", flush=True)
    print(f"[RESUME] completed={already}/{len(split)} ({100*already/len(split):.2f}%)", flush=True)
    print(f"[LOCK] split_sha256={split_hash}", flush=True)
    print(f"[LOCK] config_sha256={config_hash}", flush=True)

    newly_done = 0
    last_pid = None
    for i, row in split.iterrows():
        if int(complete_mm[i]) == 1:
            continue

        pid = str(row.patientId)
        last_pid = pid
        gray = load_dicom(paths[pid])
        winner, e = choose(gray)
        winner_index = WAVELETS.index(winner)

        # Compute all five policy representations before marking this row complete.
        local_features = {}
        for pol in policies:
            w = winner if pol == "adaptive" else pol.replace("fixed_", "")
            local_features[pol] = features(gray, w)

        # Commit row data. `complete` is changed only after all policy features exist.
        for pol in policies:
            featmaps[pol][i, :] = local_features[pol]
        ent_mm[i, :] = [e[w] for w in WAVELETS]
        winner_mm[i] = winner_index

        # Flush scientific payload before progress bit, so a crash cannot mark a partial row complete.
        for mm in featmaps.values():
            mm.flush()
        ent_mm.flush()
        winner_mm.flush()
        complete_mm[i] = 1
        complete_mm.flush()

        newly_done += 1
        if newly_done % args.progress_every == 0:
            p = write_progress(out, complete_mm, started, pid, "feature_extraction")
            eta = p["estimated_remaining_seconds"]
            eta_txt = "unknown" if eta is None else f"{eta/3600:.2f} h"
            print(
                f"[FEATURES] {p['completed_cases']}/{p['total_cases']} "
                f"({p['percent']:.2f}%) last={pid} ETA(this-run-rate)={eta_txt}",
                flush=True,
            )
        elif newly_done % args.checkpoint_every == 0:
            # Data are already flushed per row; this writes an explicit durable progress manifest.
            write_progress(out, complete_mm, started, pid, "feature_extraction")

    # Validate full feature extraction before OOF modeling.
    if int(np.count_nonzero(np.asarray(complete_mm))) != len(split):
        raise RuntimeError("Feature cache incomplete after extraction loop")
    for pol, mm in featmaps.items():
        if not np.all(np.isfinite(np.asarray(mm))):
            raise RuntimeError(f"Non-finite cached features for {pol}")

    write_progress(out, complete_mm, started, last_pid, "feature_extraction_complete")
    print("[FEATURES] COMPLETE. Building adaptive wavelet choices and OOF predictions.", flush=True)

    # Materialize adaptive-wavelet metadata deterministically from cache.
    ent = np.asarray(ent_mm)
    winners = np.asarray(winner_mm, dtype=np.int64)
    se = np.sort(ent, axis=1)
    entrows = pd.DataFrame({
        "patient_id": split.patientId,
        "fold": split.oof_fold.astype(int),
        "label": split.label.astype(int),
        "selected_wavelet": [WAVELETS[int(x)] for x in winners],
        **{f"entropy_{w}": ent[:, j] for j, w in enumerate(WAVELETS)},
        "winner_entropy": np.max(ent, axis=1),
        "second_best_entropy": se[:, -2],
        "margin": se[:, -1] - se[:, -2],
    })
    entrows.to_csv(out / "adaptive_wavelet_choices.csv", index=False)

    y = split.label.to_numpy(int)
    folds = split.oof_fold.to_numpy(int)

    # OOF policy outputs are individually resumable at policy-file granularity.
    for pol in policies:
        policy_out = out / f"{pol}.csv"
        if policy_out.exists():
            old = pd.read_csv(policy_out)
            if (
                len(old) == len(split)
                and old.patient_id.astype(str).tolist() == split.patientId.astype(str).tolist()
                and np.all(np.isfinite(old.probability.to_numpy(float)))
            ):
                print(f"[OOF] {pol}: existing complete output verified; skipping.", flush=True)
                continue
            raise RuntimeError(f"Existing OOF file is incomplete/incompatible: {policy_out}")

        pred = np.full(len(split), np.nan)
        for k in range(5):
            tr = folds != k
            va = folds == k
            print(f"[OOF] policy={pol} fold={k} train={int(tr.sum())} holdout={int(va.sum())}", flush=True)
            model = build_lgbm(fold_params[k], int(tr.sum()))
            model.fit(np.asarray(featmaps[pol][tr]), y[tr])
            pred[va] = model.predict_proba(np.asarray(featmaps[pol][va]))[:, 1]
            atomic_json(out / "oof_progress.json", {
                "phase": "oof_modeling",
                "policy": pol,
                "completed_fold": k,
                "timestamp_unix": time.time(),
            })

        if not np.all(np.isfinite(pred)):
            raise RuntimeError(f"Incomplete OOF {pol}")
        selected = entrows.selected_wavelet if pol == "adaptive" else pol.replace("fixed_", "")
        df_out = pd.DataFrame({
            "patient_id": split.patientId,
            "fold": folds,
            "true_label": y,
            "wavelet_policy": pol,
            "selected_wavelet": selected,
            "probability": pred,
        })
        tmp = policy_out.with_suffix(".csv.tmp")
        df_out.to_csv(tmp, index=False)
        os.replace(tmp, policy_out)
        print(f"[OOF] {pol}: COMPLETE -> {policy_out}", flush=True)

    # Lightweight final QC summary; no inferential analysis is performed here.
    summary = []
    for pol in policies:
        d = pd.read_csv(out / f"{pol}.csv")
        yy = d.true_label.to_numpy(int)
        pp = d.probability.to_numpy(float)
        summary.append({
            "policy": pol,
            "n": len(d),
            "roc_auc": float(roc_auc_score(yy, pp)),
            "auprc": float(average_precision_score(yy, pp)),
            "brier": float(brier_score_loss(yy, pp)),
        })
    pd.DataFrame(summary).to_csv(out / "stage4_policy_point_metrics.csv", index=False)

    atomic_json(out / "run_complete.json", {
        "status": "COMPLETE",
        "split_sha256": split_hash,
        "config_sha256": config_hash,
        "cache_manifest": str(cache_manifest),
        "policies": policies,
        "n": len(split),
        "timestamp_unix": time.time(),
    })
    print("Five policy OOF files written. Run the paired-statistics phase in the repository analysis utility.", flush=True)


if __name__ == "__main__":
    main()

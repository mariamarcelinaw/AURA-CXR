#!/usr/bin/env python3
"""
============================================================================
AURA-CXR — Proposal-Aligned Pneumonia Detection Pipeline
============================================================================
Pipeline deteksi pneumonia biner pada DICOM chest X-ray RSNA stage 2:

1) Cabang radiomik: adaptive wavelet selection berbasis Shannon entropy, GLCM, dan LBP.
2) Empat learner radiomik (KNN, HistGB, LightGBM, RBF-SVM) dioptimasi dengan
   MRFO budget sama; learner terbaik dipilih dari nested patient-level development OOF.
3) Empat kandidat deep learning: EfficientNetV2S, ResNet50, DenseNet121-XRV, dan EVA-X-S; dua dipilih dari patient-level development OOF.
4) MC Dropout untuk estimasi ketidakpastian prediksi (mean ± CI 95%).
5) Fusi final: dua DL terpilih + MRFO-optimized radiomics branch (best learner), Logistic Regression meta-learner dilatih dari OOF development.
6) XAI: Grad-CAM dan Contrastive Grad-CAM; evaluasi kuantitatif dijalankan
   di eval_only_xai.py terhadap bounding box radiolog.

Catatan proposal:
- Split selalu patient-level 70/10/20 untuk mencegah kebocoran data.
- Skor risiko klinis dihapus karena tidak ada di proposal final.
- Bounding box hanya disimpan sebagai ground-truth XAI, bukan fitur prediksi.
- Dimensi radiomik mengikuti Proposal Rumus 3.12: detail subband per level
  + LL terakhir, sehingga L=3 menghasilkan 3L+1 = 10 subband.
============================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import hashlib
from datetime import datetime
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pydicom
import pywt
import scipy.stats as stats
import tensorflow as tf
from PIL import Image

# Python 3.13 / Pillow 12 compatibility: Image.BILINEAR may be deprecated/removed.
PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

from aura_dl_pair_selection import (
    ensure_development_table, make_patient_oof_folds, save_oof_artifact, load_oof_artifact,
    select_dl_pair_and_train_meta, finalize_locked_test_fusion, load_pair_manifest,
    selected_pair_markdown, normalize_binary_probability, two_col,
    assert_selection_mutable, create_or_refresh_deployment_manifest,
    load_deployment_manifest, resolve_locked_model_artifact,
    register_probability_artifact, sha256_file, save_json as save_protocol_json, tune_threshold,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.segmentation import slic, mark_boundaries
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & IEEE Style
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES   = {0: "Non-Pneumonia", 1: "Pneumonia"}
RFC_UNITS     = [512, 256, 128, 64, 32]
RFC_DROPOUT   = 0.40
DEFAULT_L2    = 5e-5
LABEL_SMOOTH  = 0.05

IEEE_DPI = 300


def make_cnn_loss(use_focal: bool = True, gamma: float = 2.0,
                  label_smoothing: float = LABEL_SMOOTH):
    """Loss untuk CNN. Focal loss menekankan contoh sulit / kelas positif (pneumonia).
    Fallback aman ke CategoricalCrossentropy bila API focal tidak tersedia.
    Kompatibel dengan target soft (mixup) karena keduanya per-kelas."""
    if use_focal:
        focal_cls = getattr(tf.keras.losses, "CategoricalFocalCrossentropy", None)
        if focal_cls is not None:
            try:
                return focal_cls(gamma=float(gamma), label_smoothing=float(label_smoothing))
            except TypeError:
                return focal_cls(gamma=float(gamma))
    return tf.keras.losses.CategoricalCrossentropy(label_smoothing=float(label_smoothing))
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11,   "axes.titlesize": 12,
    "legend.fontsize": 10,  "figure.dpi": IEEE_DPI,
    "savefig.dpi": IEEE_DPI, "savefig.bbox": "tight",
    "axes.grid": True,      "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
})
CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "jet_alpha", ["navy", "cyan", "lime", "yellow", "red"]
)

# Novelty 1: candidate wavelet families
WAVELET_CANDIDATES = ["haar", "db4", "sym4", "coif3"]

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    labels_csv: str = ""
    dicom_dir: str = ""
    output_dir: str = ""
    image_size: int = 224
    seed: int = 42
    stratify_splits: bool = True
    # Proposal Tabel 3.1 / BAB 3: patient-level 70% train, 10% val, 20% test.
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    batch_size: int = 32
    epochs: int = 40  # phase-1 maximum; total TF budget = 40 + 20 = 60
    warmup_epochs: int = 5
    learning_rate: float = 1e-4
    dropout: float = RFC_DROPOUT
    l2_strength: float = DEFAULT_L2
    unfreeze_last_n: int = 120
    glcm_distances: str = "1,2,3"
    glcm_angles: str = "0,45,90,135"
    wavelet_levels: int = 3
    lbp_radius: int = 3
    lbp_n_points: int = 24
    use_mixup: bool = True
    mixup_alpha: float = 0.20
    use_stacking: bool = True
    # Fix pack: sensitivity-first pipeline.
    use_focal_loss: bool = True          # focal loss lebih fokus ke kelas positif (pneumonia)
    focal_gamma: float = 2.0
    soft_vote_auc_floor: float = 0.78    # anggota di bawah AUC ini dikeluarkan dari soft voting
    finetune_epochs: int = 20            # phase-2 fine-tuning; total TF budget 40 + 20 = 60
    save_tsne: bool = True
    tsne_max_samples: int = 3000
    num_xai_samples: int = 16
    subset_rows: Optional[int] = None
    cache_dirname: str = "cache"
    use_efficientnet: bool = True
    mc_dropout_n: int = 30
    uncertainty_threshold: Optional[float] = None
    uncertainty_review_rate: float = 0.30
    decision_threshold: float = 0.40
    train_efficientnetv2: bool = True
    train_resnet50: bool = True
    # MRFO sesuai proposal: k ganjil 3–101, weights/metric dieksplorasi.
    # Successive-screening radiomics search. The expensive final nested search
    # is limited to the two learners promoted inside each outer-training fold.
    mrfo_pop_size: int = 15
    mrfo_max_iter: int = 15
    mrfo_cv_folds: int = 3
    radiomics_screen_rows: int = 4000
    radiomics_screen_folds: int = 3
    radiomics_screen_inner_folds: int = 2
    radiomics_screen_pop: int = 8
    radiomics_screen_iter: int = 8
    radiomics_screen_seeds: str = "42,123"
    radiomics_screen_top_k: int = 2
    radiomics_final_inner_folds: int = 3
    radiomics_final_pop: int = 15
    radiomics_final_iter: int = 15
    mrfo_restart_patience: int = 6
    mrfo_restart_fraction: float = 0.35
    mrfo_epsilon: float = 1e-5
    # Learner cabang optimasi-metaheuristik. 'histgb' (gradient boosting sklearn,
    # tanpa dependensi ekstra) jauh lebih kuat dari kNN pada fitur radiomik.
    # Pilihan: histgb | lgbm | svm | knn. Fitness: f1_macro (balanced-F1) | roc_auc.
    # "auto" benchmarks KNN, HistGB, LightGBM, and RBF-SVM with the same
    # MRFO budget and full training rows, then promotes the best CV learner.
    mrfo_estimator: str = "auto"
    radiomics_candidates: str = "knn,histgb,lgbm,svm"
    radiomics_selection_tolerance: float = 0.002
    mrfo_fitness: str = "f1_macro"
    mrfo_subsample: int = 0   # subsampel CV utk estimator mahal (svm); 0 = pakai semua
    # Resumable stages (Proposal Tabel staging S0–S7).
    stage: str = "all"
    force: bool = False
    chunk_id: Optional[int] = None
    n_chunks: Optional[int] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    merge: bool = False
    finalize: bool = False
    run_ablation: bool = False
    ablation_only: bool = False
    ablation_scenarios: str = ""
    auto_resume_cnn: bool = True
    save_epoch_checkpoints: bool = False
    checkpoint_monitor: str = "val_auc"
    early_stopping_patience: int = 12
    reduce_lr_patience: int = 5
    tune_threshold: bool = True
    threshold_metric: str = "youden"
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 181
    threshold_min_sensitivity: float = 0.85
    threshold_min_specificity: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    if math.isclose(lo, hi):
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(normalize_minmax(image) * 255, 0, 255).astype(np.uint8)


def load_dicom_grayscale(path, image_size: int) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = img.max() - img
    img = normalize_minmax(img)
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image_size, image_size), resample=PIL_BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def load_dicom_rgb(path, image_size: int) -> np.ndarray:
    gray = load_dicom_grayscale(path, image_size)
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


def compute_model_metrics(y_true, y_pred, y_proba) -> Dict:
    report = classification_report(
        y_true, y_pred,
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy":     float(accuracy_score(y_true, y_pred)),
        "f1_weighted":  float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro":     float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "auc":          float(roc_auc_score(y_true, y_proba[:, 1]) if len(np.unique(y_true)) > 1 else 0.0),
        "report":       report,
    }


@tf.keras.utils.register_keras_serializable(package="AURA")
class OneHotBinaryF1(tf.keras.metrics.Metric):
    """Streaming binary F1 for one-hot softmax outputs; class index 1 = Pneumonia."""
    def __init__(self, name="f1", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_pos = y_true[..., 1] if y_true.shape.rank is not None and y_true.shape.rank > 1 else y_true
        y_pred_pos = y_pred[..., 1] if y_pred.shape.rank is not None and y_pred.shape.rank > 1 else y_pred
        yt = tf.cast(y_true_pos >= 0.5, tf.float32)
        yp = tf.cast(y_pred_pos >= self.threshold, tf.float32)
        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            yt = yt * sw
            yp = yp * sw
        self.tp.assign_add(tf.reduce_sum(yp * yt))
        self.fp.assign_add(tf.reduce_sum(yp * (1.0 - yt)))
        self.fn.assign_add(tf.reduce_sum((1.0 - yp) * yt))

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        return 2.0 * precision * recall / (precision + recall + 1e-8)

    def reset_state(self):
        self.tp.assign(0.0); self.fp.assign(0.0); self.fn.assign(0.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg


def cnn_compile_metrics() -> List[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        OneHotBinaryF1(name="f1"),
    ]


def checkpoint_mode_for_monitor(monitor: str) -> str:
    monitor = str(monitor).lower()
    return "min" if monitor.endswith("loss") else "max"


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def build_samples_dataframe(labels_csv, dicom_dir, subset_rows=None) -> pd.DataFrame:
    """Build one sample per patientId.

    Proposal BAB 3 requires patient-level splitting. The earlier per-row mode
    (one bbox row = one sample) is intentionally removed to avoid leakage.
    Bounding-box aggregates are retained only as radiologist GT metadata for
    quantitative XAI, never as predictive features.
    """
    labels = pd.read_csv(labels_csv)
    if "patientId" not in labels.columns or "Target" not in labels.columns:
        raise ValueError("labels_csv must contain RSNA columns: patientId and Target")

    # One patientId = one DICOM = one sample; positive if any bbox row is positive.
    df = labels.groupby("patientId", as_index=False)["Target"].max()
    df["label"] = df["Target"].astype(int)
    df["sample_id"] = np.arange(len(df))

    dicom_dir = Path(dicom_dir)
    df["image_path"] = df["patientId"].astype(str).map(lambda x: str(dicom_dir / f"{x}.dcm"))

    # Radiologist bbox metadata for XAI metrics only (not used by classifiers).
    if {"x", "y", "width", "height"}.issubset(labels.columns):
        bb_cols = labels[["patientId", "x", "y", "width", "height"]].copy()
        bb_cols["bbox_area"] = bb_cols["width"].fillna(0) * bb_cols["height"].fillna(0)
        bb_cols["bbox_cx"] = bb_cols["x"].fillna(0) + bb_cols["width"].fillna(0) / 2
        bb_cols["bbox_cy"] = bb_cols["y"].fillna(0) + bb_cols["height"].fillna(0) / 2
        bb_agg = bb_cols.groupby("patientId").agg(
            bbox_area_max=("bbox_area", "max"),
            bbox_cx_mean=("bbox_cx", "mean"),
            bbox_cy_mean=("bbox_cy", "mean"),
        ).reset_index()
        df = df.merge(bb_agg, on="patientId", how="left")
        df["bbox_area_max"] = df["bbox_area_max"].fillna(0.0)
        df["bbox_cx_mean"] = df["bbox_cx_mean"].fillna(512.0)
        df["bbox_cy_mean"] = df["bbox_cy_mean"].fillna(512.0)
    else:
        df["bbox_area_max"] = 0.0
        df["bbox_cx_mean"] = 512.0
        df["bbox_cy_mean"] = 512.0

    # Drop missing DICOMs before split, so all split CSVs are executable.
    df["exists"] = df["image_path"].map(lambda x: Path(x).exists())
    n_miss = int((~df["exists"]).sum())
    if n_miss > 0:
        print(f"[WARN] Missing {n_miss} DICOM files → dropped before patient-level split.")
        df = df[df["exists"]].copy()

    if subset_rows:
        # Kept as a quick debug option; after aggregation it means number of patients.
        df = df.iloc[:subset_rows].copy()
        df["sample_id"] = np.arange(len(df))

    return df.reset_index(drop=True)


def _assert_disjoint_patient_splits(train_df, val_df, test_df) -> None:
    train_ids = set(train_df["patientId"].astype(str))
    val_ids = set(val_df["patientId"].astype(str))
    test_ids = set(test_df["patientId"].astype(str))
    leaks = {
        "train_val": train_ids & val_ids,
        "train_test": train_ids & test_ids,
        "val_test": val_ids & test_ids,
    }
    if any(leaks.values()):
        raise AssertionError({k: sorted(list(v))[:10] for k, v in leaks.items() if v})
    print("[SPLIT] Anti-leak assertion passed: patientId sets are disjoint.")


def _log_split_distribution(name: str, sdf: pd.DataFrame, total_n: int) -> None:
    pos = int(sdf["label"].sum())
    neg = int((sdf["label"] == 0).sum())
    print(f"[SPLIT] {name:5s}: n={len(sdf):6d} ({len(sdf)/max(total_n,1):6.2%}) | "
          f"Pneumonia={pos:5d} | Non={neg:5d}")


def split_dataframe(df, val_frac, test_frac, seed, stratify):
    """Stratified patient-level split.

    Because build_samples_dataframe already aggregates one row per patientId,
    train_test_split with stratify is patient-disjoint by construction.
    """
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1:
        raise ValueError("Require 0 < val_fraction, test_fraction and val+test < 1")
    stratify_col = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), random_state=seed,
        shuffle=True, stratify=stratify_col,
    )
    rel_test = test_frac / (val_frac + test_frac)
    strat_temp = temp_df["label"] if stratify else None
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=seed,
        shuffle=True, stratify=strat_temp,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    _assert_disjoint_patient_splits(train_df, val_df, test_df)
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _log_split_distribution(name, sdf, len(df))
    return train_df, val_df, test_df


def save_split_manifest(out: Path, train_df, val_df, test_df, config: RunConfig) -> None:
    split_dir = ensure_dir(out / "splits")
    for split, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        required_cols = ["patientId", "label", "bbox_area_max", "bbox_cx_mean", "bbox_cy_mean", "image_path"]
        missing = [c for c in required_cols if c not in sdf.columns]
        if missing:
            raise ValueError(f"Split {split} missing required columns: {missing}")
        sdf.to_csv(split_dir / f"{split}.csv", index=False)
    save_json({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_level": "patientId",
        "train_fraction_observed": len(train_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
        "val_fraction_observed": len(val_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
        "test_fraction_observed": len(test_df) / max(len(train_df) + len(val_df) + len(test_df), 1),
        "requested_val_fraction": config.val_fraction,
        "requested_test_fraction": config.test_fraction,
        "seed": config.seed,
    }, split_dir / "split_manifest.json")


def load_or_create_splits(out: Path, config: RunConfig, force: bool = False):
    split_dir = ensure_dir(out / "splits")
    paths = {s: split_dir / f"{s}.csv" for s in ["train", "val", "test"]}
    if (not force) and all(p.exists() for p in paths.values()):
        train_df = pd.read_csv(paths["train"])
        val_df = pd.read_csv(paths["val"])
        test_df = pd.read_csv(paths["test"])
        _assert_disjoint_patient_splits(train_df, val_df, test_df)
        print("[SPLIT] Loaded existing patient-level split CSVs.")
        for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            _log_split_distribution(name, sdf, len(train_df) + len(val_df) + len(test_df))
        return train_df, val_df, test_df
    if not config.labels_csv or not config.dicom_dir:
        raise FileNotFoundError("Stage splits requires --labels_csv and --dicom_dir, or existing splits/*.csv")
    df = build_samples_dataframe(config.labels_csv, config.dicom_dir, config.subset_rows)
    print(f"[INFO] Dataset patient-level: {len(df)} patients | Pneumonia={int(df['label'].sum())} | Non={int((df['label']==0).sum())}")
    train_df, val_df, test_df = split_dataframe(df, config.val_fraction, config.test_fraction, config.seed, config.stratify_splits)
    save_split_manifest(out, train_df, val_df, test_df, config)
    return train_df, val_df, test_df

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(classes)
    return {int(c): float(total / (n_cls * cnt)) for c, cnt in zip(classes, counts)}


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 1 — Adaptive Wavelet Bank Selection
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(coeffs: np.ndarray) -> float:
    """Shannon entropy of DWT coefficient magnitudes (normalized)."""
    c = np.abs(coeffs.ravel())
    c = c[c > 0]
    if len(c) == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def select_best_wavelet(gray_image: np.ndarray,
                        candidates: List[str] = WAVELET_CANDIDATES) -> str:
    """
    Novelty 1 core: Pilih wavelet dengan entropy Shannon maksimum
    dari koefisien detail level-1 DWT.
    Entropy tinggi → koefisien tersebar luas → wavelet menangkap tekstur lebih kaya.
    """
    best_wavelet = candidates[0]
    best_entropy = -np.inf
    for w in candidates:
        try:
            _, (lh, hl, hh) = pywt.dwt2(gray_image, w)
            detail_all = np.concatenate([lh.ravel(), hl.ravel(), hh.ravel()])
            ent = _shannon_entropy(detail_all)
            if ent > best_entropy:
                best_entropy = ent
                best_wavelet = w
        except Exception:
            continue
    return best_wavelet


def _wavelet_subbands_proposal(gray_image: np.ndarray, wavelet: str, wavelet_levels: int) -> List[np.ndarray]:
    """Proposal Rumus 3.12: {LH_l, HL_l, HH_l} for each level + final LL_L.

    For L=3 this yields 3L+1 = 10 subbands, not 4L. With 6 GLCM props,
    3 distances and 4 angles, the full GLCM dimension is 10*6*3*4 = 720.
    Adding one uniform LBP histogram (P+2=26) gives 746 full features.
    """
    details: List[np.ndarray] = []
    current = gray_image
    for _ in range(wavelet_levels):
        ll, (lh, hl, hh) = pywt.dwt2(current, wavelet)
        details.extend([lh, hl, hh])
        current = ll
    return details + [current]


def radiomic_feature_tag(use_wavelet: bool = True, adaptive: bool = True,
                         fixed_wavelet: str = "db4", use_glcm: bool = True,
                         use_lbp: bool = True, wavelet_levels: int = 3) -> str:
    wtag = "nowavelet" if not use_wavelet else ("adaptive" if adaptive else f"fixed_{fixed_wavelet}")
    gtag = "glcm" if use_glcm else "noglcm"
    ltag = "lbp" if use_lbp else "nolbp"
    # v2 invalidates old cache that used 4L subbands.
    return f"radiomic_v2_{wtag}_L{wavelet_levels}_{gtag}_{ltag}"


def extract_adaptive_wavelet_glcm_lbp(
    gray_image: np.ndarray,
    distances: List[int] = [1, 2, 3],
    angles_deg: List[int] = [0, 45, 90, 135],
    wavelet_levels: int = 3,
    lbp_radius: int = 3,
    lbp_n_points: int = 24,
    use_wavelet: bool = True,
    adaptive: bool = True,
    fixed_wavelet: str = "db4",
    use_glcm: bool = True,
    use_lbp: bool = True,
) -> Tuple[np.ndarray, str]:
    """Radiomic extraction for main and ablation scenarios.

    Flags implement Proposal Tabel 3.12 and feature-ablation scenarios.
    Returns (feature_vector, selected_wavelet_label).
    """
    if not use_glcm and not use_lbp:
        raise ValueError("At least one of use_glcm/use_lbp must be True")

    angles_rad = [np.deg2rad(a) for a in angles_deg]
    features: List[float] = []

    if use_wavelet:
        chosen_wavelet = select_best_wavelet(gray_image) if adaptive else fixed_wavelet
        coeffs_all = _wavelet_subbands_proposal(gray_image, chosen_wavelet, wavelet_levels)
    else:
        chosen_wavelet = "none"
        coeffs_all = [gray_image]

    # GLCM per selected image/subband.
    if use_glcm:
        for coeff in coeffs_all:
            coeff_u8 = to_uint8(coeff)
            glcm = graycomatrix(coeff_u8, distances=distances, angles=angles_rad,
                                levels=256, symmetric=True, normed=True)
            for prop in ["contrast", "energy", "homogeneity", "dissimilarity", "correlation", "ASM"]:
                vals = graycoprops(glcm, prop)
                features.extend(vals.flatten().tolist())

    # LBP remains one histogram from the preprocessed image, matching proposal dimension.
    if use_lbp:
        gray_u8 = to_uint8(gray_image)
        lbp = local_binary_pattern(gray_u8, lbp_n_points, lbp_radius, method="uniform")
        n_bins = lbp_n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.extend(lbp_hist.tolist())

    return np.asarray(features, dtype=np.float32), chosen_wavelet


def compute_feature_cache(df, image_size, distances, angles_deg, wavelet_levels,
                           lbp_radius, lbp_n_points, cache_path: Path,
                           use_wavelet: bool = True, adaptive: bool = True,
                           fixed_wavelet: str = "db4", use_glcm: bool = True,
                           use_lbp: bool = True, force: bool = False,
                           chunk_start: Optional[int] = None, chunk_end: Optional[int] = None):
    """Compute/cache radiomic features and wavelet info.

    cache_path must include the scenario tag; v2 cache prevents loading old
    4L-subband features from previous code.
    """
    wavelet_info_path = cache_path.with_suffix(".wavelet_info.json")
    if (not force) and cache_path.exists() and wavelet_info_path.exists():
        print(f"[INFO] Loading cached features: {cache_path}")
        return joblib.load(cache_path), json.loads(wavelet_info_path.read_text())

    unique_paths = sorted(df["image_path"].unique())
    if chunk_start is not None or chunk_end is not None:
        unique_paths = unique_paths[chunk_start or 0: chunk_end]

    cache: Dict[str, np.ndarray] = {}
    wavelet_info: Dict[str, str] = {}
    wavelet_counts = {w: 0 for w in WAVELET_CANDIDATES}
    wavelet_counts["none"] = 0
    desc = radiomic_feature_tag(use_wavelet, adaptive, fixed_wavelet, use_glcm, use_lbp, wavelet_levels)

    for path in tqdm(unique_paths, desc=f"[Radiomics] {desc}"):
        gray = load_dicom_grayscale(path, image_size)
        feat, chosen = extract_adaptive_wavelet_glcm_lbp(
            gray, distances=distances, angles_deg=angles_deg,
            wavelet_levels=wavelet_levels, lbp_radius=lbp_radius,
            lbp_n_points=lbp_n_points, use_wavelet=use_wavelet,
            adaptive=adaptive, fixed_wavelet=fixed_wavelet,
            use_glcm=use_glcm, use_lbp=use_lbp,
        )
        cache[path] = feat
        wavelet_info[path] = chosen
        wavelet_counts[chosen] = wavelet_counts.get(chosen, 0) + 1

    if cache:
        first_dim = len(next(iter(cache.values())))
        print(f"[Radiomics] Feature dimension ({desc}) = {first_dim}")
        if desc.startswith("radiomic_v2_adaptive_L3_glcm_lbp") and first_dim != 746:
            raise AssertionError(f"Expected 746 full radiomic features for L=3, got {first_dim}")
    print("[Radiomics] Wavelet selection distribution:", wavelet_counts)
    joblib.dump(cache, cache_path)
    wavelet_info_path.write_text(json.dumps(wavelet_info, indent=2))
    return cache, wavelet_info

def features_from_dataframe(df, cache) -> np.ndarray:
    return np.stack([cache[p] for p in df["image_path"]], axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 5 — Manta Ray Foraging Optimization (MRFO) for radiomics learner tuning
# ─────────────────────────────────────────────────────────────────────────────
# Paper: Zhao, W., Zhang, Z., & Wang, L. (2020). Manta ray foraging
#        optimization: An effective bio-inspired optimizer for engineering
#        applications. Engineering Applications of Artificial Intelligence,
#        87, 103300. https://doi.org/10.1016/j.engappai.2019.103300
# ─────────────────────────────────────────────────────────────────────────────

class MRFOptimizer:
    """
    Novelty 5: Manta Ray Foraging Optimization (MRFO).
    Fix v2:
      - default population is larger (50 recommended)
      - fitness cache avoids re-evaluating identical discrete KNN settings
      - diversity injection / random restart if the best score is stagnant
      - wider K range improves exploration and reduces early lock-in
    """
    K_RANGE      = list(range(3, 102, 2))
    WEIGHT_MAP   = ["uniform", "distance"]
    METRIC_MAP   = ["euclidean", "manhattan", "minkowski", "chebyshev"]

    def __init__(self, pop_size: int = 50, max_iter: int = 50,
                 x_train=None, y_train=None, cv_folds: int = 5, seed: int = 42,
                 restart_patience: int = 10, restart_fraction: float = 0.35,
                 epsilon: float = 1e-5, estimator: str = "histgb",
                 fitness: str = "f1_macro", subsample: int = 6000):
        self.pop_size = int(pop_size)
        self.max_iter = int(max_iter)
        self.x_train  = x_train
        self.y_train  = y_train
        self.cv_folds = int(cv_folds)
        self.rng      = np.random.default_rng(seed)
        self.restart_patience = int(restart_patience)
        self.restart_fraction = float(np.clip(restart_fraction, 0.05, 0.90))
        self.epsilon = float(epsilon)
        self.estimator = str(estimator).lower()
        self.fitness_metric = str(fitness).lower()
        self.subsample = int(subsample)
        self._fitness_cache: Dict[Tuple, float] = {}

        # Ruang pencarian hyperparameter per-learner (posisi kontinu di [lb, ub]).
        if self.estimator == "knn":
            self.lb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.ub = np.array([len(self.K_RANGE) - 1.0,
                                len(self.WEIGHT_MAP) - 1.0,
                                len(self.METRIC_MAP) - 1.0], dtype=np.float32)
        elif self.estimator == "svm":
            # [log10(C) in [-2,3], log10(gamma) in [-4,0]]
            self.lb = np.array([-2.0, -4.0], dtype=np.float32)
            self.ub = np.array([3.0, 0.0], dtype=np.float32)
        else:  # histgb / lgbm : [lr, leaves, depth, l2, min_leaf_frac]
            # log10(lr)∈[-2.3,-0.4]; leaves∈[15,255]; depth∈[2,12];
            # log10(l2)∈[-4,1]; min_samples_leaf frac∈[0.001,0.05]
            self.lb = np.array([-2.3, 15.0, 2.0, -4.0, 0.001], dtype=np.float32)
            self.ub = np.array([-0.4, 255.0, 12.0, 1.0, 0.05], dtype=np.float32)
        self.dim = int(len(self.lb))
        # Subsampel indeks tetap untuk CV pada estimator mahal (SVM).
        self._cv_idx = None
        if self.subsample and self.x_train is not None and len(self.y_train) > self.subsample and self.estimator == "svm":
            self._cv_idx = self._stratified_subsample(self.subsample)

    def _stratified_subsample(self, n):
        y = np.asarray(self.y_train)
        idx = []
        for cls in np.unique(y):
            ci = np.where(y == cls)[0]
            take = max(1, int(round(n * len(ci) / len(y))))
            idx.append(self.rng.choice(ci, size=min(take, len(ci)), replace=False))
        return np.sort(np.concatenate(idx))

    def _decode(self, pos: np.ndarray) -> Tuple:
        """Kembalikan tuple hyperparameter yang hashable (untuk cache & build)."""
        if self.estimator == "knn":
            k_idx = int(np.clip(round(float(pos[0])), 0, len(self.K_RANGE) - 1))
            w_idx = int(np.clip(round(float(pos[1])), 0, len(self.WEIGHT_MAP) - 1))
            m_idx = int(np.clip(round(float(pos[2])), 0, len(self.METRIC_MAP) - 1))
            return (self.K_RANGE[k_idx], self.WEIGHT_MAP[w_idx], self.METRIC_MAP[m_idx])
        if self.estimator == "svm":
            C = round(10.0 ** float(np.clip(pos[0], self.lb[0], self.ub[0])), 5)
            g = round(10.0 ** float(np.clip(pos[1], self.lb[1], self.ub[1])), 6)
            return (C, g)
        lr = round(10.0 ** float(np.clip(pos[0], self.lb[0], self.ub[0])), 5)
        leaves = int(np.clip(round(float(pos[1])), 15, 255))
        depth = int(np.clip(round(float(pos[2])), 2, 12))
        l2 = round(10.0 ** float(np.clip(pos[3], self.lb[3], self.ub[3])), 5)
        min_leaf = round(float(np.clip(pos[4], self.lb[4], self.ub[4])), 4)
        return (lr, leaves, depth, l2, min_leaf)

    def _build_estimator(self, decoded: Tuple):
        """Bangun estimator sklearn dari tuple hyperparameter terdekode."""
        if self.estimator == "knn":
            k, weight, metric = decoded
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k, weights=weight,
                                             metric=metric, algorithm="auto", p=2)),
            ])
        if self.estimator == "svm":
            from sklearn.svm import SVC
            C, g = decoded
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=C, gamma=g, kernel="rbf", probability=True,
                            class_weight="balanced", random_state=42)),
            ])
        if self.estimator == "lgbm":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:
                raise ImportError(
                    "Estimator lgbm dipilih tetapi package lightgbm belum terpasang. "
                    "Install dengan: pip install lightgbm"
                ) from exc
            lr, leaves, depth, l2, min_leaf = decoded
            n = len(self.y_train)
            return Pipeline([("clf", LGBMClassifier(
                learning_rate=lr, num_leaves=leaves,
                max_depth=(depth if depth < 12 else -1),
                reg_lambda=l2, min_child_samples=max(5, int(min_leaf * n)),
                n_estimators=400, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.9, class_weight="balanced",
                random_state=42, n_jobs=-1, verbose=-1))])
        # histgb (default, tanpa dependensi tambahan)
        from sklearn.ensemble import HistGradientBoostingClassifier
        lr, leaves, depth, l2, min_leaf = decoded
        n = len(self.y_train)
        return Pipeline([("clf", HistGradientBoostingClassifier(
            learning_rate=lr, max_leaf_nodes=leaves, max_depth=depth,
            l2_regularization=l2, min_samples_leaf=max(20, int(min_leaf * n)),
            max_iter=400, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, class_weight="balanced", random_state=42))])

    def _score_fold(self, pipe, xtr, ytr, xva, yva) -> float:
        pipe.fit(xtr, ytr)
        if self.fitness_metric == "roc_auc":
            try:
                p = pipe.predict_proba(xva)[:, 1]
                return float(roc_auc_score(yva, p)) if len(np.unique(yva)) > 1 else 0.0
            except Exception:
                pass
        y_pred = pipe.predict(xva)
        # macro-F1 = balanced-F1: bobot setara ke kelas minoritas (pneumonia).
        return float(f1_score(yva, y_pred, average="macro", zero_division=0))

    def _fitness(self, pos: np.ndarray) -> float:
        """CV score (dinegasikan untuk minimisasi). Learner & metrik configurable."""
        key = (self.estimator, self.fitness_metric) + self._decode(pos)
        if key in self._fitness_cache:
            return self._fitness_cache[key]
        decoded = self._decode(pos)
        X = self.x_train; y = self.y_train
        if self._cv_idx is not None:  # subsampel utk estimator mahal
            X = X[self._cv_idx]; y = y[self._cv_idx]
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            pipe = self._build_estimator(decoded)
            scores.append(self._score_fold(
                pipe, X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
        val = -float(np.mean(scores))
        self._fitness_cache[key] = val
        return val

    def _random_positions(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.lb, self.ub, size=(int(n), self.dim)).astype(np.float32)

    def _inject_diversity(self, X: np.ndarray, fitness: np.ndarray, X_best: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_restart = max(1, int(round(self.pop_size * self.restart_fraction)))
        worst_idx = np.argsort(fitness)[-n_restart:]
        # Mix true random restarts and local jitter around current best.
        n_global = max(1, n_restart // 2)
        fresh = self._random_positions(n_restart)
        jitter_scale = 0.20 * (self.ub - self.lb + 1e-8)
        fresh[n_global:] = X_best + self.rng.normal(0.0, jitter_scale, size=(n_restart - n_global, self.dim))
        fresh = np.clip(fresh, self.lb, self.ub)
        fresh_fitness = np.array([self._fitness(p) for p in fresh])
        X[worst_idx] = fresh
        fitness[worst_idx] = fresh_fitness
        return X, fitness

    def optimize(self) -> Tuple[Pipeline, Dict]:
        X = self._random_positions(self.pop_size)
        fitness = np.array([self._fitness(X[i]) for i in range(self.pop_size)])
        best_idx = int(np.argmin(fitness))
        X_best = X[best_idx].copy()
        f_best = float(fitness[best_idx])
        history = [-f_best]
        restarts: List[int] = []
        stagnant = 0

        print(f"[Novelty5-MRFO] Init best {self.fitness_metric}={-f_best:.4f} | params={self._decode(X_best)}")

        for t in range(1, self.max_iter + 1):
            prev_best = f_best
            r1 = self.rng.random((self.pop_size, self.dim))
            r2 = self.rng.random()

            if r2 < 0.5:
                ref = X_best if r2 < 0.25 else self._random_positions(1)[0]
                new_X = X.copy()
                new_X[0] = X[0] + r1[0] * (ref - X[0])
                for i in range(1, self.pop_size):
                    new_X[i] = X[i] + r1[i] * (X[i-1] - X[i]) + r1[i] * (ref - X[i])
            else:
                beta = 2.0 * np.exp(self.rng.random() * (self.max_iter - t) / max(self.max_iter, 1)) * np.sin(2 * np.pi * self.rng.random())
                ref = self._random_positions(1)[0] if (t / max(self.max_iter, 1)) < self.rng.random() else X_best
                new_X = X.copy()
                new_X[0] = ref + r1[0] * (ref - X[0]) + beta * (ref - X[0])
                for i in range(1, self.pop_size):
                    new_X[i] = ref + r1[i] * (X[i-1] - X[i]) + beta * (ref - X[i])

            # Somersault foraging plus small decayed Gaussian mutation.
            S = 2.0
            r3 = self.rng.random((self.pop_size, self.dim))
            r4 = self.rng.random((self.pop_size, self.dim))
            new_X = new_X + S * (r3 * X_best - r4 * new_X)
            mutation_scale = (0.10 * (1.0 - t / max(self.max_iter, 1)) + 0.02) * (self.ub - self.lb + 1e-8)
            new_X = new_X + self.rng.normal(0.0, mutation_scale, size=new_X.shape)
            new_X = np.clip(new_X, self.lb, self.ub)

            new_fitness = np.array([self._fitness(new_X[i]) for i in range(self.pop_size)])
            improved = new_fitness < fitness
            X[improved] = new_X[improved]
            fitness[improved] = new_fitness[improved]

            best_idx = int(np.argmin(fitness))
            if float(fitness[best_idx]) < f_best - self.epsilon:
                f_best = float(fitness[best_idx])
                X_best = X[best_idx].copy()

            if prev_best - f_best > self.epsilon:
                stagnant = 0
            else:
                stagnant += 1

            if stagnant >= self.restart_patience and t < self.max_iter:
                X, fitness = self._inject_diversity(X, fitness, X_best)
                best_idx = int(np.argmin(fitness))
                if float(fitness[best_idx]) < f_best - self.epsilon:
                    f_best = float(fitness[best_idx])
                    X_best = X[best_idx].copy()
                restarts.append(t)
                stagnant = 0
                print(f"[Novelty5-MRFO] Restart/diversity injection at iter {t}; searched={len(self._fitness_cache)} configs")

            history.append(-f_best)
            if t % 10 == 0 or t == self.max_iter:
                print(f"[Novelty5-MRFO] Iter {t:3d}/{self.max_iter} | Best {self.fitness_metric}={-f_best:.4f} | params={self._decode(X_best)} | unique={len(set(round(v, 6) for v in history))}")

        decoded = self._decode(X_best)
        # Nama field hyperparameter per-learner untuk pelaporan.
        if self.estimator == "knn":
            params = {"k": decoded[0], "weights": decoded[1], "metric": decoded[2]}
        elif self.estimator == "svm":
            params = {"C": decoded[0], "gamma": decoded[1]}
        else:
            params = {"learning_rate": decoded[0], "max_leaf_nodes": decoded[1],
                      "max_depth": decoded[2], "l2_regularization": decoded[3],
                      "min_samples_leaf_frac": decoded[4]}
        best_info = {
            "estimator": self.estimator,
            "fitness_metric": self.fitness_metric,
            "best_params": params,
            **params,  # backward-compat (kNN: k/weights/metric tetap ada)
            "best_cv_score": float(-f_best),
            "best_cv_f1": float(-f_best),  # nama lama dipertahankan
            "history": history,
            "unique_history_values": int(len(set(round(v, 8) for v in history))),
            "restart_iterations": restarts,
            "evaluated_unique_configs": int(len(self._fitness_cache)),
            "pop_size": int(self.pop_size),
            "max_iter": int(self.max_iter),
        }
        print(f"[Novelty5-MRFO] Optimal ({self.estimator}, {self.fitness_metric}): "
              f"{params} | score={-f_best:.4f}")

        final_pipe = self._build_estimator(decoded)
        return final_pipe, best_info


def train_radiomics_mrfo(x_train, y_train, config: RunConfig) -> Tuple[Pipeline, Dict]:
    """Novelty 5: train the configured radiomics learner using MRFO.

    Function name is retained for backward compatibility with older scripts.
    """
    mrfo = MRFOptimizer(
        pop_size=config.mrfo_pop_size,
        max_iter=config.mrfo_max_iter,
        x_train=x_train,
        y_train=y_train,
        cv_folds=int(getattr(config, "mrfo_cv_folds", 3)),
        seed=config.seed,
        restart_patience=config.mrfo_restart_patience,
        restart_fraction=config.mrfo_restart_fraction,
        epsilon=config.mrfo_epsilon,
        estimator=("histgb" if str(getattr(config, "mrfo_estimator", "histgb")).lower() == "auto"
                   else str(getattr(config, "mrfo_estimator", "histgb")).lower()),
        fitness=getattr(config, "mrfo_fitness", "f1_macro"),
        subsample=getattr(config, "mrfo_subsample", 0),
    )
    pipe, best_info = mrfo.optimize()
    pipe.fit(x_train, y_train)
    return pipe, best_info


# ─────────────────────────────────────────────────────────────────────────────
# CNN Architecture (exact EfficientNetV2-S / ResNet50; no silent fallback)
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_model(image_size, learning_rate, dropout, l2_strength,
                    backbone_name: str = "efficientnetv2",
                    use_focal_loss: bool = True, focal_gamma: float = 2.0) -> tf.keras.Model:
    """
    Build a CNN head on top of EfficientNetV2-S (default) or ResNet50.
    Fix v2: Dropout is no longer hard-wired with training=True. It behaves
    normally during training, and MC Dropout explicitly enables it at inference.
    """
    backbone_name = str(backbone_name).lower()
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")

    if backbone_name in {"efficientnet", "efficientnetv2", "efficientnetv2s", "effnet"}:
        app = getattr(tf.keras.applications, "EfficientNetV2S", None)
        if app is None:
            raise RuntimeError(
                "EfficientNetV2S is unavailable in this TensorFlow build. "
                "Silent substitution with EfficientNetB3 or another architecture is forbidden."
            )
        base = app(
            include_top=False, weights="imagenet",
            input_shape=(image_size, image_size, 3), pooling="avg",
        )
        model_name = "cnn_efficientnetv2s"
    elif backbone_name in {"resnet", "resnet50"}:
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            input_shape=(image_size, image_size, 3), pooling="avg",
        )
        model_name = "cnn_resnet50"
    else:
        raise ValueError(f"Unsupported backbone_name={backbone_name!r}")

    # Keep the application model as an explicit nested backbone. The fixed
    # training=False call keeps all internal BatchNorm layers in inference mode
    # while gradients still flow through non-BN layers opened during phase 2.
    base.trainable = False
    x = base(inputs, training=False)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)

    for idx, units in enumerate(RFC_UNITS, 1):
        x = tf.keras.layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(l2_strength),
            name=f"dense_{idx}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{idx}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{idx}")(x)
        x = tf.keras.layers.Dropout(dropout, name=f"dropout_{idx}")(x)

    # Keep probabilities in float32 even when mixed_float16 is enabled.
    outputs = tf.keras.layers.Dense(2, activation="softmax", dtype="float32", name="output_softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=make_cnn_loss(use_focal=use_focal_loss, gamma=focal_gamma),
        metrics=cnn_compile_metrics(),
    )
    return model


def preprocess_for_efficientnet(batch_rgb: tf.Tensor) -> tf.Tensor:
    """Exact EfficientNetV2 preprocessing; a v1 EfficientNet fallback is forbidden."""
    arr = tf.cast(batch_rgb, tf.float32) * 255.0
    efficientnet_v2 = getattr(tf.keras.applications, "efficientnet_v2", None)
    if efficientnet_v2 is None or not hasattr(efficientnet_v2, "preprocess_input"):
        raise RuntimeError("TensorFlow EfficientNetV2 preprocessing API is unavailable")
    return efficientnet_v2.preprocess_input(arr)


def _find_nested_cnn_backbone(model: tf.keras.Model, backbone_name: str) -> tf.keras.Model:
    """Resolve the exact nested application backbone; never guess a substitute."""
    key = str(backbone_name).lower()
    expected = "efficientnet" if key in {"efficientnet", "efficientnetv2", "efficientnetv2s", "effnet"} else "resnet"
    candidates = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and expected in layer.name.lower()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one nested {expected} backbone, found {[x.name for x in candidates]}"
        )
    return candidates[0]


def _recursive_layers(layer):
    """Yield a layer and all nested descendants once."""
    seen = set()
    stack = [layer]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend(reversed(list(getattr(current, "layers", []) or [])))


def configure_backbone_finetuning(model: tf.keras.Model, backbone_name: str,
                                  unfreeze_last_n: int) -> Dict:
    """Open exactly the last N non-BN internal backbone layers.

    All BatchNormalization layers in the complete model—including the classifier
    head and nested application backbone—remain frozen. The returned audit is
    persisted and fail-fast validated.
    """
    backbone = _find_nested_cnn_backbone(model, backbone_name)
    requested = max(0, int(unfreeze_last_n))
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    candidates = [
        layer for layer in backbone.layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    selected = candidates[-min(requested, len(candidates)):] if requested else []
    for layer in selected:
        layer.trainable = True
    all_layers = list(_recursive_layers(model))
    bn_layers = [x for x in all_layers if isinstance(x, tf.keras.layers.BatchNormalization)]
    for layer in bn_layers:
        layer.trainable = False
    actual = sum(bool(layer.trainable) for layer in selected)
    bn_trainable = sum(bool(layer.trainable) for layer in bn_layers)
    expected_actual = min(requested, len(candidates))
    if actual != expected_actual or bn_trainable != 0:
        raise RuntimeError(
            "Fine-tuning configuration mismatch: "
            f"requested={requested}, expected={expected_actual}, actual={actual}, "
            f"trainable_batchnorm={bn_trainable}"
        )
    return {
        "backbone_layer_name": backbone.name,
        "requested_unfreeze_last_n": requested,
        "available_backbone_non_bn_layers": int(len(candidates)),
        "actual_backbone_layers_unfrozen": int(actual),
        "batchnorm_layers_total": int(len(bn_layers)),
        "batchnorm_layers_trainable": int(bn_trainable),
        "batchnorm_policy": "all_model_batchnorm_frozen_inference",
    }


def preprocess_for_resnet(batch_rgb: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.resnet50.preprocess_input(
        tf.cast(batch_rgb, tf.float32) * 255.0
    )


def _tf_augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 1.0)


def _tf_load_image(image_path: tf.Tensor, label: tf.Tensor, image_size: int):
    def _py_loader(path_bytes: bytes) -> np.ndarray:
        return load_dicom_rgb(path_bytes.decode("utf-8"), image_size).astype(np.float32)
    image = tf.numpy_function(_py_loader, [image_path], Tout=tf.float32)
    image.set_shape((image_size, image_size, 3))
    label = tf.one_hot(tf.cast(label, tf.int32), depth=2)
    return image, label


def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.cast(np.random.beta(alpha, alpha) if alpha > 0 else 0.5, tf.float32)
    indices = tf.random.shuffle(tf.range(batch_size))
    return (lam * images + (1 - lam) * tf.gather(images, indices),
            lam * labels + (1 - lam) * tf.gather(labels, indices))


def make_tf_dataset(df, image_size, batch_size, training, seed,
                     use_efficientnet=True, use_mixup=False, mixup_alpha=0.2):
    image_paths = df["image_path"].astype(str).values
    labels      = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _tf_load_image(p, y, image_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (_tf_augment(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    preprocess_fn = preprocess_for_efficientnet if use_efficientnet else preprocess_for_resnet
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training and use_mixup:
        ds = ds.map(lambda x, y: mixup_batch(x, y, mixup_alpha),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 2 — Bayesian Uncertainty-Aware Fusion (MC Dropout)
# ─────────────────────────────────────────────────────────────────────────────


def _walk_keras_layers(root):
    """Yield nested Keras layers once, including application submodels."""
    seen = set()
    stack = list(getattr(root, "layers", []))
    while stack:
        layer = stack.pop()
        if id(layer) in seen:
            continue
        seen.add(id(layer))
        yield layer
        stack.extend(list(getattr(layer, "layers", [])))


def prepare_mc_dropout_inference(model: tf.keras.Model) -> tf.keras.Model:
    """Enable stochastic Dropout safely while forcing every BatchNorm to inference.

    Keras respects ``BatchNormalization.trainable=False`` even when the outer
    model is called with ``training=True``. This lets Dropout remain stochastic
    without batch-composition-dependent BatchNorm statistics.
    """
    for layer in _walk_keras_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def batchnorm_state_snapshot(model: tf.keras.Model):
    state = []
    for layer in _walk_keras_layers(model):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            state.append((layer.name, layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy()))
    return state


def assert_batchnorm_state_unchanged(before, model: tf.keras.Model) -> None:
    after = {name: (mean, var) for name, mean, var in batchnorm_state_snapshot(model)}
    for name, mean0, var0 in before:
        mean1, var1 = after[name]
        if not np.array_equal(mean0, mean1) or not np.array_equal(var0, var1):
            raise RuntimeError(f"BatchNorm state changed during MC Dropout inference: {name}")


PREDICTIVE_INTERVAL_Z95 = 1.96
PREDICTIVE_UNCERTAINTY_DEFINITION = "two_sided_95_predictive_interval_width_equals_2_times_1.96_times_mc_probability_std"


def predictive_interval_width_95(std_probability) -> np.ndarray:
    """Two-sided MC-Dropout predictive interval width, not a CI of the mean."""
    return (2.0 * PREDICTIVE_INTERVAL_Z95 * np.asarray(std_probability, dtype=np.float32)).astype(np.float32)


def mc_dropout_predict(model: tf.keras.Model,
                        X_tensor,
                        n_samples: int = 30,
                        verify_batchnorm: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Novelty 2: Monte Carlo Dropout Inference.
    Jalankan forward pass N kali dengan Dropout aktif.
    Returns:
      mean_proba  : shape (N_samples, 2)
      std_proba   : shape (N_samples, 2)
      interval_width_95 : shape (N_samples,) — lebar interval ketidakpastian prediktif 95% kelas Pneumonia
    """
    model = prepare_mc_dropout_inference(model)
    bn_before = batchnorm_state_snapshot(model) if verify_batchnorm else None
    preds = []
    for _ in range(n_samples):
        # Only Dropout is stochastic; all BatchNorm layers remain in inference mode.
        p = model(X_tensor, training=True).numpy()
        preds.append(p)
    if bn_before is not None:
        assert_batchnorm_state_unchanged(bn_before, model)
    preds = np.stack(preds, axis=0)   # (n_samples, N, 2)

    mean_proba = preds.mean(axis=0)   # (N, 2)
    std_proba  = preds.std(axis=0)    # (N, 2)

    # Two-sided 95% MC predictive interval width for Pneumonia (index 1).
    ci_width = predictive_interval_width_95(std_proba[:, 1])

    return mean_proba, std_proba, ci_width


def flag_uncertain_cases(ci_width: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """Tandai kasus dengan interval ketidakpastian prediktif lebar untuk review radiologis."""
    return ci_width > threshold



MC_CACHE_SCHEMA = "aura_cxr_mc_dropout_cache_q1_v17"


def _keras_model_state_sha256(model: tf.keras.Model) -> str:
    """Hash the exact in-memory Keras weights used for prediction."""
    h = hashlib.sha256()
    for weight in model.weights:
        arr = np.asarray(weight.numpy())
        h.update(str(weight.name).encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _mc_cache_expected(model: tf.keras.Model, df: pd.DataFrame, image_size: int,
                       batch_size: int, n_mc: int, preprocess_fn,
                       cache_prefix: str, model_artifact_sha256: str = "",
                       preprocessing_id: str = "") -> Dict:
    patient_values = (
        df["patientId"].astype(str).tolist()
        if "patientId" in df.columns
        else [str(x) for x in df.index.tolist()]
    )
    image_values = df["image_path"].astype(str).tolist()
    label_hash = (
        _stable_sha256_values(df["label"].astype(int).tolist())
        if "label" in df.columns else "not_present"
    )
    return {
        "schema": MC_CACHE_SCHEMA,
        "cache_prefix": str(cache_prefix),
        "model_state_sha256": _keras_model_state_sha256(model),
        "model_artifact_sha256": str(model_artifact_sha256 or "in_memory_model_state_only"),
        "ordered_patient_sha256": _stable_sha256_values(patient_values),
        "ordered_image_path_sha256": _stable_sha256_values(image_values),
        "ordered_label_sha256": label_hash,
        "row_count": int(len(df)),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "mc_passes": int(n_mc),
        "preprocessing_id": str(preprocessing_id or getattr(preprocess_fn, "__name__", "unknown_preprocess")),
        "batchnorm_policy": "all_nested_batchnorm_frozen_inference_dropout_stochastic",
        "uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        "tensorflow_version": str(tf.__version__),
    }


def _validate_mc_cache_manifest(payload: Dict, expected: Dict, output_paths: Optional[Dict] = None) -> None:
    mismatches = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
    if output_paths:
        for key, path in output_paths.items():
            path = Path(path)
            if not path.exists():
                mismatches[f"{key}_path"] = ("missing", str(path))
                continue
            expected_hash = payload.get("output_sha256", {}).get(key)
            actual_hash = sha256_file(path)
            if expected_hash != actual_hash:
                mismatches[f"{key}_sha256"] = (expected_hash, actual_hash)
    if mismatches:
        raise RuntimeError(
            "Stale or incompatible MC-Dropout cache detected. "
            f"Mismatches: {mismatches}. Use a new OUT directory or rerun with --force before test lock."
        )


def cnn_mc_predict_batch(model: tf.keras.Model,
                          df: pd.DataFrame,
                          image_size: int,
                          batch_size: int,
                          preprocess_fn,
                          n_mc: int = 30,
                          output_dir: Optional[Path] = None,
                          cache_prefix: str = "",
                          resume: bool = True,
                          force: bool = False,
                          model_artifact_sha256: str = "",
                          preprocessing_id: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout prediction with hash-scoped final and batch-resume caches.

    Cache reuse requires an exact match for model state, ordered patients,
    ordered image paths, labels, MC passes, image size, batch size,
    preprocessing identity, TensorFlow version, and BatchNorm policy.
    """
    model = prepare_mc_dropout_inference(model)
    bn_before_all = batchnorm_state_snapshot(model)
    all_mean, all_std, all_ci = [], [], []
    image_paths = df["image_path"].astype(str).values
    n = len(image_paths)
    expected = _mc_cache_expected(
        model, df, image_size, batch_size, n_mc, preprocess_fn, cache_prefix,
        model_artifact_sha256=model_artifact_sha256,
        preprocessing_id=preprocessing_id,
    )

    final_paths = None
    final_manifest_path = None
    chunk_dir = None
    chunk_manifest_path = None
    if output_dir is not None and cache_prefix:
        output_dir = Path(output_dir)
        probs_dir = ensure_dir(output_dir / "probs")
        final_paths = {
            "mean": probs_dir / f"{cache_prefix}_mean.npy",
            "std": probs_dir / f"{cache_prefix}_std.npy",
            "ci": probs_dir / f"{cache_prefix}_ci.npy",
        }
        final_manifest_path = probs_dir / f"{cache_prefix}_mc_cache_manifest.json"
        final_exists = any(p.exists() for p in final_paths.values()) or final_manifest_path.exists()
        if resume and (not force) and final_exists:
            if not (all(p.exists() for p in final_paths.values()) and final_manifest_path.exists()):
                raise RuntimeError(f"Incomplete MC cache for {cache_prefix}; use a new OUT directory or --force")
            payload = json.loads(final_manifest_path.read_text(encoding="utf-8"))
            _validate_mc_cache_manifest(payload, expected, final_paths)
            mean = np.load(final_paths["mean"])
            std = np.load(final_paths["std"])
            ci = np.load(final_paths["ci"])
            if mean.shape != (n, 2) or std.shape != (n, 2) or ci.shape != (n,):
                raise RuntimeError(
                    f"MC cache shape mismatch for {cache_prefix}: mean={mean.shape}, std={std.shape}, ci={ci.shape}"
                )
            print(f"[RESUME-MC] Loaded provenance-validated MC outputs for {cache_prefix}: {probs_dir}")
            return mean, std, ci
        chunk_dir = ensure_dir(output_dir / "resume" / "mc_dropout" / cache_prefix)
        chunk_manifest_path = chunk_dir / "cache_manifest.json"
        existing_chunks = list(chunk_dir.glob("batch_*.npz"))
        if resume and (not force) and (chunk_manifest_path.exists() or existing_chunks):
            if not chunk_manifest_path.exists():
                raise RuntimeError(f"MC chunk cache lacks provenance manifest: {chunk_dir}")
            payload = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
            _validate_mc_cache_manifest(payload, expected)
        else:
            if force:
                for old in chunk_dir.glob("batch_*.npz"):
                    old.unlink()
            save_json({**expected, "cache_level": "batch_resume"}, chunk_manifest_path)

    for start in tqdm(range(0, n, batch_size), desc="[Novelty2] MC Dropout inference"):
        end = min(start + batch_size, n)
        chunk_path = chunk_dir / f"batch_{start:06d}_{end:06d}.npz" if chunk_dir is not None else None
        if resume and (not force) and chunk_path is not None and chunk_path.exists():
            try:
                z = np.load(chunk_path)
                mean_p, std_p, ci_w = z["mean"], z["std"], z["ci"]
                if mean_p.shape == (end - start, 2) and std_p.shape == (end - start, 2) and ci_w.shape == (end - start,):
                    all_mean.append(mean_p)
                    all_std.append(std_p)
                    all_ci.append(ci_w)
                    continue
                raise ValueError(f"shape mismatch mean={mean_p.shape} std={std_p.shape} ci={ci_w.shape}")
            except Exception as exc:
                raise RuntimeError(f"Corrupt MC chunk {chunk_path}: {exc}") from exc
        batch_imgs = np.stack([
            np.repeat(load_dicom_grayscale(p, image_size)[..., None], 3, axis=-1)
            for p in image_paths[start:end]
        ], axis=0).astype(np.float32)
        X_t = preprocess_fn(tf.constant(batch_imgs))
        mean_p, std_p, ci_w = mc_dropout_predict(model, X_t, n_samples=n_mc, verify_batchnorm=False)
        if chunk_path is not None:
            tmp = chunk_path.with_suffix(".npz.tmp")
            np.savez_compressed(tmp, mean=mean_p, std=std_p, ci=ci_w)
            actual_tmp = Path(str(tmp) if str(tmp).endswith(".npz") else str(tmp) + ".npz")
            actual_tmp.replace(chunk_path)
        all_mean.append(mean_p)
        all_std.append(std_p)
        all_ci.append(ci_w)

    mean = np.concatenate(all_mean, axis=0) if all_mean else np.zeros((0, 2), dtype=np.float32)
    std = np.concatenate(all_std, axis=0) if all_std else np.zeros((0, 2), dtype=np.float32)
    ci = np.concatenate(all_ci, axis=0) if all_ci else np.zeros((0,), dtype=np.float32)
    assert_batchnorm_state_unchanged(bn_before_all, model)
    if mean.shape != (n, 2) or std.shape != (n, 2) or ci.shape != (n,):
        raise RuntimeError(f"Unexpected MC output shapes: mean={mean.shape}, std={std.shape}, ci={ci.shape}")
    if final_paths is not None:
        np.save(final_paths["mean"], mean)
        np.save(final_paths["std"], std)
        np.save(final_paths["ci"], ci)
        payload = {
            **expected,
            "cache_level": "final",
            "output_sha256": {key: sha256_file(path) for key, path in final_paths.items()},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(payload, final_manifest_path)
        print(f"[OK] Provenance-locked MC cache saved for {cache_prefix}: {final_paths['mean'].parent}")
    return mean, std, ci


# Risk-score module removed: not part of proposal final. BBox metadata is XAI ground truth only.

# ─────────────────────────────────────────────────────────────────────────────
# NOVELTY 4 — Contrastive GradCAM (Why-A-not-B XAI)
# ─────────────────────────────────────────────────────────────────────────────

ACTIVATION_THRESHOLD = 1e-5


def _iter_conv_layers(model: tf.keras.Model) -> List[tf.keras.layers.Layer]:
    """Return connected Conv2D/DepthwiseConv2D layers, prioritizing deeper blocks."""
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    layers: List[tf.keras.layers.Layer] = []

    def visit(layer):
        if isinstance(layer, conv_types):
            layers.append(layer)
        elif isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                visit(sub)

    for layer in model.layers:
        visit(layer)

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for layer in layers:
        if id(layer) not in seen:
            unique.append(layer)
            seen.add(id(layer))

    def priority(layer):
        name = layer.name.lower()
        keys = ("top", "block7", "block6", "conv5", "stage4")
        return 0 if any(k in name for k in keys) else 1

    return sorted(list(reversed(unique)), key=priority)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    candidates = _iter_conv_layers(model)
    if not candidates:
        raise ValueError("No Conv2D/DepthwiseConv2D layer found.")
    return candidates[0].name


def _gradcam_for_layer(model: tf.keras.Model, image_array, layer: tf.keras.layers.Layer,
                       class_index: int = 1, use_plusplus: bool = False) -> Optional[np.ndarray]:
    if image_array.ndim == 3:
        image_array = image_array[np.newaxis, ...]
    try:
        grad_model = tf.keras.Model(inputs=model.inputs, outputs=[layer.output, model.output])
    except Exception:
        return None

    with tf.GradientTape() as tape:
        inputs = tf.cast(image_array, tf.float32)
        conv_out, preds = grad_model(inputs, training=False)
        # Log probability reduces softmax saturation while staying compatible with saved softmax models.
        score = tf.math.log(tf.clip_by_value(tf.cast(preds[:, class_index], tf.float32), 1e-7, 1.0))
    grads = tape.gradient(score, conv_out)
    if grads is None:
        return None

    conv_out = tf.cast(conv_out, tf.float32)
    grads = tf.cast(grads, tf.float32)
    if tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grads), tf.math.is_inf(grads))):
        return None

    if use_plusplus:
        grads2 = tf.square(grads)
        grads3 = grads2 * grads
        sum_activ = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
        alpha = grads2 / (2.0 * grads2 + sum_activ * grads3 + 1e-8)
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))[0]
    else:
        weights = tf.reduce_mean(grads[0], axis=(0, 1))

    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)
    hmax = float(np.nanmax(heatmap)) if heatmap.size else 0.0
    if not np.isfinite(hmax) or hmax <= 1e-12:
        return None
    return (heatmap / hmax).astype(np.float32)


def compute_gradcam_robust(model, image_array, candidate_layers=None,
                           class_index: int = 1, debug: bool = False) -> Tuple[np.ndarray, str, str]:
    if candidate_layers is None or isinstance(candidate_layers, str):
        if isinstance(candidate_layers, str):
            by_name = {l.name: l for l in _iter_conv_layers(model)}
            candidate_layers = [by_name[candidate_layers]] if candidate_layers in by_name else _iter_conv_layers(model)
        else:
            candidate_layers = _iter_conv_layers(model)

    best = None
    best_mean = -1.0
    for layer in candidate_layers:
        for use_pp, method in [(False, "GradCAM"), (True, "GradCAM++")]:
            hm = _gradcam_for_layer(model, image_array, layer, class_index=class_index, use_plusplus=use_pp)
            if hm is None:
                continue
            mu = float(np.mean(hm))
            if debug:
                print(f"    [{method}] class={class_index} layer={layer.name} mu={mu:.6f}")
            if mu > best_mean:
                best = (hm, layer.name, method)
                best_mean = mu
            if mu >= ACTIVATION_THRESHOLD:
                return hm, layer.name, method

    if best is not None:
        hm, lname, method = best
        return hm, lname, method + " (weak)"
    return np.zeros((7, 7), dtype=np.float32), "none", "FAILED"


def compute_contrastive_gradcam(model, image_array, last_conv_name=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = _iter_conv_layers(model) if last_conv_name is None else last_conv_name
    hm_pneu, _, _ = compute_gradcam_robust(model, image_array, candidates, class_index=1)
    hm_non_pneu, _, _ = compute_gradcam_robust(model, image_array, candidates, class_index=0)
    hm_pneu_n = hm_pneu / (float(hm_pneu.max()) + 1e-8)
    hm_npneu_n = hm_non_pneu / (float(hm_non_pneu.max()) + 1e-8)
    contrastive = np.clip(hm_pneu_n - hm_npneu_n, 0, 1).astype(np.float32)
    return hm_pneu_n.astype(np.float32), hm_npneu_n.astype(np.float32), contrastive


def overlay_heatmap(raw_image: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    h, w = raw_image.shape[:2]
    hm_r = zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1]))
    hm_r = np.clip(hm_r, 0, 1)
    hm_c = (CMAP_HEATMAP(hm_r)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * hm_c + (1 - alpha) * raw_image).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble & Calibration
# ─────────────────────────────────────────────────────────────────────────────

def train_stacking_meta_learner(knn_val_proba, cnn_val_proba, y_val) -> LogisticRegression:
    meta_x = np.concatenate([knn_val_proba, cnn_val_proba], axis=1)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(meta_x, y_val)
    return clf


def train_stacking_meta_learner_from_list(proba_list: List[np.ndarray], y_val) -> LogisticRegression:
    meta_x = np.concatenate(proba_list, axis=1)
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(meta_x, y_val)
    return clf


def stacking_oof_probabilities(proba_list: List[np.ndarray], y_val: np.ndarray,
                               n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Generate out-of-fold validation probabilities for the stacking meta-learner.

    The final meta-learner is still fitted on all validation rows for locked test
    inference, but threshold tuning and learner selection use these OOF scores so
    they are not evaluated on the same rows used to fit each fold's meta-model.
    """
    meta_x = np.concatenate(proba_list, axis=1)
    y = np.asarray(y_val).astype(int)
    counts = np.bincount(y, minlength=2)
    feasible = int(min(int(n_splits), counts.min()))
    if feasible < 2:
        print("[WARN] Too few validation samples per class for stacking OOF; using in-sample probabilities.")
        return train_stacking_meta_learner_from_list(proba_list, y).predict_proba(meta_x)
    cv = StratifiedKFold(n_splits=feasible, shuffle=True, random_state=int(seed))
    oof = np.zeros((len(y), 2), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(meta_x, y), 1):
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=int(seed) + fold)
        clf.fit(meta_x[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(meta_x[va_idx]).astype(np.float32)
    return oof


def predict_from_proba(proba: np.ndarray, threshold: float = 0.40) -> np.ndarray:
    return (np.asarray(proba)[:, 1] >= float(threshold)).astype(np.int32)


def tune_threshold_from_validation(y_val: np.ndarray, val_proba: np.ndarray, config: RunConfig,
                                   model_tag: str, reports_dir: Path, plots_dir: Path) -> float:
    """Tune operating threshold on validation set only, then lock it for test."""
    if not bool(getattr(config, "tune_threshold", True)):
        return float(config.decision_threshold)
    y_val = np.asarray(y_val).astype(int)
    p = np.asarray(val_proba, dtype=np.float32)[:, 1]
    thresholds = np.linspace(float(config.threshold_min), float(config.threshold_max), int(config.threshold_steps))
    rows = []
    for th in thresholds:
        yp = (p >= th).astype(int)
        sens, spec = sens_spec(y_val, yp)
        rows.append({
            "threshold": float(th),
            "f1_weighted": float(f1_score(y_val, yp, average="weighted", zero_division=0)),
            "f1_pneumonia": float(f1_score(y_val, yp, average="binary", zero_division=0)),
            "accuracy": float(accuracy_score(y_val, yp)),
            "sensitivity": sens,
            "specificity": spec,
            "balanced_accuracy": float(0.5 * (sens + spec)),
            "youden_j": float(sens + spec - 1.0),
        })
    df = pd.DataFrame(rows)
    min_sens = float(getattr(config, "threshold_min_sensitivity", 0.0))
    min_spec = float(getattr(config, "threshold_min_specificity", 0.0))
    feasible = df[(df["sensitivity"] >= min_sens) & (df["specificity"] >= min_spec)].copy()
    note = "Selected from thresholds satisfying validation constraints."
    if feasible.empty:
        feasible = df.copy()
        note = "No threshold satisfied min_sensitivity/min_specificity; selected from full validation grid."
    metric = str(getattr(config, "threshold_metric", "f1")).lower()
    metric_col = {
        "f1": "f1_weighted",
        "f1_weighted": "f1_weighted",
        "f1_pneumonia": "f1_pneumonia",
        "sensitivity": "sensitivity",
        "balanced_accuracy": "balanced_accuracy",
        "youden": "youden_j",
        "youden_j": "youden_j",
    }.get(metric, "f1_weighted")
    feasible = feasible.sort_values([metric_col, "balanced_accuracy", "specificity"], ascending=[False, False, False])
    best = feasible.iloc[0].to_dict()
    reports_dir = ensure_dir(reports_dir)
    plots_dir = ensure_dir(plots_dir)
    safe_tag = str(model_tag).replace(" ", "_").replace("/", "_").lower()
    csv_path = reports_dir / f"threshold_tuning_{safe_tag}.csv"
    json_path = reports_dir / f"threshold_tuning_{safe_tag}.json"
    df.to_csv(csv_path, index=False)
    save_json({
        "model": model_tag,
        "selection_split": "validation",
        "test_set_used_for_selection": False,
        "metric_optimized": metric_col,
        "selected_threshold": float(best["threshold"]),
        "validation_metrics_at_selected_threshold": {k: float(best[k]) for k in best if k != "threshold"},
        "min_sensitivity_constraint": min_sens,
        "min_specificity_constraint": min_spec,
        "note": note,
        "csv": str(csv_path),
    }, json_path)
    try:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(df["threshold"], df["f1_weighted"], label="F1-weighted")
        ax.plot(df["threshold"], df["f1_pneumonia"], label="F1 Pneumonia")
        ax.plot(df["threshold"], df["sensitivity"], label="Sensitivity")
        ax.plot(df["threshold"], df["specificity"], label="Specificity")
        ax.axvline(float(best["threshold"]), linestyle="--", label=f"selected={float(best['threshold']):.3f}")
        ax.set_xlabel("Decision threshold on validation set")
        ax.set_ylabel("Metric")
        ax.set_title(f"Validation Threshold Tuning — {model_tag}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"threshold_tuning_{safe_tag}.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] Could not save threshold curve for {model_tag}: {exc}")
    print(f"[Q1-THRESHOLD] {model_tag}: selected threshold={float(best['threshold']):.3f} using validation {metric_col}={float(best[metric_col]):.4f}")
    return float(best["threshold"])


def load_saved_threshold(output_dir: Path, model_tag: str, default: float) -> float:
    safe_tag = str(model_tag).replace(" ", "_").replace("/", "_").lower()
    path = Path(output_dir) / "reports" / f"threshold_tuning_{safe_tag}.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text()).get("selected_threshold", default))
        except Exception:
            return float(default)
    return float(default)


def sens_spec(y_true, y_pred) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return float(sensitivity), float(specificity)


def choose_uncertainty_threshold(val_ci: np.ndarray, config: RunConfig) -> float:
    if config.uncertainty_threshold is not None:
        return float(config.uncertainty_threshold)
    review_rate = float(np.clip(config.uncertainty_review_rate, 0.01, 0.80))
    thr = float(np.quantile(val_ci, 1.0 - review_rate))
    # Guard against the old failure mode where almost every case is flagged.
    return max(0.15, thr)


def get_preprocess_for_backbone(backbone_name: str):
    return preprocess_for_resnet if str(backbone_name).lower() in {"resnet", "resnet50"} else preprocess_for_efficientnet


def use_efficientnet_for_backbone(backbone_name: str) -> bool:
    return str(backbone_name).lower() not in {"resnet", "resnet50"}



def _safe_backup_restore_callback(backup_dir: Path, enabled: bool = True):
    """Create a Keras BackupAndRestore callback with TF/Keras-version fallbacks."""
    if not enabled:
        return []
    backup_dir = ensure_dir(backup_dir)
    try:
        return [tf.keras.callbacks.BackupAndRestore(
            backup_dir=str(backup_dir), save_freq="epoch", delete_checkpoint=False,
        )]
    except TypeError:
        try:
            return [tf.keras.callbacks.BackupAndRestore(
                backup_dir=str(backup_dir), save_freq="epoch",
            )]
        except TypeError:
            return [tf.keras.callbacks.BackupAndRestore(str(backup_dir))]


def _write_resume_state(path: Path, payload: Dict) -> None:
    payload = dict(payload or {})
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(payload, path)


def _compile_cnn_model_for_phase(model: tf.keras.Model, learning_rate: float,
                                 use_focal_loss: bool = True, focal_gamma: float = 2.0) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=make_cnn_loss(use_focal=use_focal_loss, gamma=focal_gamma),
        metrics=cnn_compile_metrics(),
    )
    return model


def _cnn_callbacks(backbone_name: str, phase: str, ckpt_root: Path, models_dir: Path,
                   monitor: str = "val_f1", mode: Optional[str] = None,
                   auto_resume: bool = True, save_epoch_checkpoints: bool = False,
                   early_stopping_patience: int = 12, reduce_lr_patience: int = 5) -> List[tf.keras.callbacks.Callback]:
    phase_dir = ensure_dir(ckpt_root / phase)
    mode = mode or checkpoint_mode_for_monitor(monitor)
    callbacks: List[tf.keras.callbacks.Callback] = []
    callbacks.extend(_safe_backup_restore_callback(phase_dir / "backup", enabled=auto_resume))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        str(models_dir / f"{backbone_name}_{phase}_best_{monitor}.keras"),
        monitor=monitor, mode=mode, save_best_only=True, verbose=1,
    ))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        str(models_dir / f"{backbone_name}_{phase}_best.keras"),
        monitor=monitor, mode=mode, save_best_only=True, verbose=0,
    ))
    if phase == "phase1_initial":
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(models_dir / f"{backbone_name}_best.keras"),
            monitor=monitor, mode=mode, save_best_only=True, verbose=0,
        ))
    if save_epoch_checkpoints:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            str(phase_dir / "epoch_{epoch:03d}.keras"),
            save_best_only=False, save_freq="epoch", verbose=0,
        ))
    callbacks.extend([
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=int(early_stopping_patience), mode=mode,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=int(reduce_lr_patience), verbose=1, min_lr=1e-7,
        ),
        tf.keras.callbacks.CSVLogger(str(phase_dir / "training_log.csv"), append=True),
    ])
    print(f"[Q1-CHECKPOINT] {backbone_name}/{phase}: monitor={monitor}, mode={mode}, early_stopping_patience={early_stopping_patience}")
    return callbacks




def _read_phase_history(ckpt_root: Path) -> pd.DataFrame:
    frames = []
    offset = 0
    for phase in ["phase1_initial", "phase2_finetune"]:
        csv_path = Path(ckpt_root) / phase / "training_log.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        df["phase"] = phase
        df["global_epoch"] = np.arange(offset + 1, offset + len(df) + 1)
        offset += len(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_training_curves_and_overfit_report(backbone_name: str, ckpt_root: Path, reports_dir: Path, plots_dir: Path) -> None:
    hist = _read_phase_history(ckpt_root)
    if hist.empty:
        print(f"[WARN] No training history found for {backbone_name}; train-val curve not generated.")
        return
    reports_dir = ensure_dir(reports_dir)
    plots_dir = ensure_dir(plots_dir)
    hist_path = reports_dir / f"training_history_{backbone_name}.csv"
    hist.to_csv(hist_path, index=False)
    metrics = [("loss", "val_loss"), ("accuracy", "val_accuracy"), ("auc", "val_auc"), ("f1", "val_f1")]
    for tr_col, va_col in metrics:
        if tr_col not in hist.columns or va_col not in hist.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(hist["global_epoch"], hist[tr_col], label=f"train_{tr_col}")
        ax.plot(hist["global_epoch"], hist[va_col], label=va_col)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(tr_col.upper() if tr_col != "loss" else "Loss")
        ax.set_title(f"Train–Validation {tr_col.upper()} Curve — {backbone_name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"training_curve_{backbone_name}_{tr_col}.png", dpi=300)
        plt.close(fig)
    last = hist.iloc[-1].to_dict()
    def _f(key):
        try:
            return float(last[key]) if key in last and np.isfinite(float(last[key])) else None
        except Exception:
            return None
    final_train_f1, final_val_f1 = _f("f1"), _f("val_f1")
    final_train_auc, final_val_auc = _f("auc"), _f("val_auc")
    f1_gap = None if final_train_f1 is None or final_val_f1 is None else final_train_f1 - final_val_f1
    auc_gap = None if final_train_auc is None or final_val_auc is None else final_train_auc - final_val_auc
    warnings_overfit = []
    if f1_gap is not None and f1_gap > 0.08:
        warnings_overfit.append("Potential overfitting: final train F1 exceeds validation F1 by > 0.08")
    if auc_gap is not None and auc_gap > 0.05:
        warnings_overfit.append("Potential overfitting: final train AUC exceeds validation AUC by > 0.05")
    report = {
        "backbone": backbone_name,
        "history_csv": str(hist_path),
        "best_val_f1": float(hist["val_f1"].max()) if "val_f1" in hist else None,
        "best_val_auc": float(hist["val_auc"].max()) if "val_auc" in hist else None,
        "final_train_f1": final_train_f1,
        "final_val_f1": final_val_f1,
        "final_f1_gap_train_minus_val": f1_gap,
        "final_train_auc": final_train_auc,
        "final_val_auc": final_val_auc,
        "final_auc_gap_train_minus_val": auc_gap,
        "overfitting_warnings": warnings_overfit,
        "interpretation": "Overfitting cannot be guaranteed away, but this report audits train-validation gaps. Test set is reserved for final evaluation only.",
    }
    save_json(report, reports_dir / f"overfitting_report_{backbone_name}.json")
    print(f"[Q1-CURVE] Saved train-val curves/history for {backbone_name}")
    if warnings_overfit:
        print(f"[WARN] {backbone_name} overfitting diagnostics: {'; '.join(warnings_overfit)}")

def train_cnn_backbone(backbone_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       y_train: np.ndarray, config: RunConfig, models_dir: Path) -> tf.keras.Model:
    """Train one CNN backbone with interruption-safe auto-resume.

    Auto-resume uses tf.keras.callbacks.BackupAndRestore for in-epoch run
    state plus explicit phase-complete model files. If the VM/SSH session stops
    during phase 1 or fine-tuning, running the same --stage again resumes from
    the last completed epoch whenever the backup directory is intact. If a phase
    has already completed, it is skipped and the next phase starts from the saved
    phase model.
    """
    print(f"\n[INFO] Building CNN backbone: {backbone_name}")
    models_dir = ensure_dir(models_dir)
    ckpt_root = ensure_dir(models_dir / "checkpoints" / backbone_name)
    phase1_model_path = ckpt_root / "phase1_initial_complete.keras"
    phase1_state_path = ckpt_root / "phase1_initial_complete.json"
    phase2_state_path = ckpt_root / "phase2_finetune_complete.json"
    final_path = models_dir / f"{backbone_name}_final.keras"

    train_ds = make_tf_dataset(
        train_df, config.image_size, config.batch_size,
        training=True, seed=config.seed,
        use_efficientnet=use_efficientnet_for_backbone(backbone_name),
        use_mixup=config.use_mixup, mixup_alpha=config.mixup_alpha,
    )
    val_ds = make_tf_dataset(
        val_df, config.image_size, config.batch_size,
        training=False, seed=config.seed,
        use_efficientnet=use_efficientnet_for_backbone(backbone_name),
    )
    cw = compute_class_weights(y_train)

    auto_resume = bool(getattr(config, "auto_resume_cnn", True)) and not bool(config.force)
    save_epoch_ckpt = bool(getattr(config, "save_epoch_checkpoints", False))

    if auto_resume and final_path.exists():
        print(f"[RESUME-CNN] Final model already exists for {backbone_name}: {final_path}")
        return tf.keras.models.load_model(str(final_path), compile=False)

    if auto_resume and phase1_model_path.exists() and phase1_state_path.exists():
        print(f"[RESUME-CNN] Phase 1 already completed for {backbone_name}; loading {phase1_model_path}")
        model = tf.keras.models.load_model(str(phase1_model_path), compile=False)
    else:
        model = build_cnn_model(
            config.image_size, config.learning_rate,
            config.dropout, config.l2_strength, backbone_name=backbone_name,
            use_focal_loss=bool(getattr(config, "use_focal_loss", True)),
            focal_gamma=float(getattr(config, "focal_gamma", 2.0)),
        )
        model.summary(print_fn=lambda line: print(line) if "Total" in line else None)
        print(f"[RESUME-CNN] Phase 1 backup dir: {ckpt_root / 'phase1_initial' / 'backup'}")
        print(f"[INFO] Training {backbone_name} initial phase for up to {config.epochs} epochs...")
        callbacks = _cnn_callbacks(
            backbone_name, "phase1_initial", ckpt_root, models_dir,
            monitor=config.checkpoint_monitor,
            auto_resume=auto_resume, save_epoch_checkpoints=save_epoch_ckpt,
            early_stopping_patience=config.early_stopping_patience,
            reduce_lr_patience=config.reduce_lr_patience,
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=config.epochs,
            class_weight=cw, callbacks=callbacks, verbose=1,
        )
        model.save(str(phase1_model_path))
        _write_resume_state(phase1_state_path, {
            "backbone": backbone_name,
            "phase": "phase1_initial",
            "complete": True,
            "epochs_requested": int(config.epochs),
            "phase_model": str(phase1_model_path),
        })

    if auto_resume and final_path.exists() and phase2_state_path.exists():
        print(f"[RESUME-CNN] Fine-tuning already completed for {backbone_name}; loading final model.")
        return tf.keras.models.load_model(str(final_path), compile=False)

    print(f"[INFO] Fine-tuning {backbone_name}: opening exactly the last internal non-BN layers...")
    finetune_audit = configure_backbone_finetuning(
        model, backbone_name, config.unfreeze_last_n
    )
    print(f"[FINE-TUNE-AUDIT] {finetune_audit}")
    model = _compile_cnn_model_for_phase(
        model, config.learning_rate * 0.1,
        use_focal_loss=bool(getattr(config, "use_focal_loss", True)),
        focal_gamma=float(getattr(config, "focal_gamma", 2.0)),
    )

    print(f"[RESUME-CNN] Fine-tune backup dir: {ckpt_root / 'phase2_finetune' / 'backup'}")
    callbacks = _cnn_callbacks(
        backbone_name, "phase2_finetune", ckpt_root, models_dir,
        monitor=config.checkpoint_monitor,
        auto_resume=auto_resume, save_epoch_checkpoints=save_epoch_ckpt,
        early_stopping_patience=config.early_stopping_patience,
        reduce_lr_patience=config.reduce_lr_patience,
    )
    _ft_epochs = int(getattr(config, "finetune_epochs", 20))
    model.fit(
        train_ds, validation_data=val_ds, epochs=_ft_epochs,
        class_weight=cw, callbacks=callbacks, verbose=1,
    )

    # Keras 3 native format is safer on Python 3.13 / TensorFlow 2.20+.
    model.save(str(final_path))
    _write_resume_state(phase2_state_path, {
        "backbone": backbone_name,
        "phase": "phase2_finetune",
        "complete": True,
        "epochs_requested": int(getattr(config, "finetune_epochs", 20)),
        "final_model": str(final_path),
        "checkpoint_monitor": config.checkpoint_monitor,
        "fine_tuning_audit": finetune_audit,
    })
    save_training_curves_and_overfit_report(
        backbone_name, ckpt_root, models_dir.parent / "reports", models_dir.parent / "plots"
    )

    # Optional legacy H5 copy for old eval scripts. If H5 serialization fails,
    # the .keras model above remains the source of truth.
    try:
        model.save(str(models_dir / f"{backbone_name}_final.h5"))
        if backbone_name == "efficientnetv2":
            model.save(str(models_dir / "cnn_final.h5"))
    except Exception as exc:
        print(f"[WARN] Legacy .h5 save skipped for {backbone_name}: {exc}")

    if backbone_name == "efficientnetv2":
        model.save(str(models_dir / "cnn_final.keras"))
    return model

def select_xai_samples(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int) -> List[int]:
    cat_need = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
    per_cat = max(1, n_samples // 4)
    selected: List[int] = []
    for _, (tt, pp) in cat_need.items():
        idxs = np.where((y_true == tt) & (y_pred == pp))[0].tolist()
        selected.extend(idxs[:per_cat])
    if len(selected) < n_samples:
        for idx in range(len(y_true)):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_samples:
                break
    return selected[:n_samples]


def temperature_scaling(proba: np.ndarray, temperature=1.5) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-8, 1 - 1e-8))
    scaled = logits / temperature
    exp_s  = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization & Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, normalize, subtitle in zip(axes, [None, "true"], ["Counts", "Normalized"]):
        disp = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
            normalize=normalize, ax=ax, colorbar=False,
            cmap="Blues",
        )
        ax.set_title(f"{title}\n({subtitle})", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)


def save_roc_pr_curves(roc_curves_dict, y_trues, y_probas, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(roc_curves_dict)))

    for ax, mode in zip(axes, ["ROC", "PR"]):
        for (name, _), c in zip(roc_curves_dict.items(), colors):
            y_true = y_trues[name]
            y_prob  = y_probas[name]
            if mode == "ROC":
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=c, label=f"{name} (AUC={auc_val:.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves")
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
                ap = float(trapz_fn(prec[::-1], rec[::-1]))
                ax.plot(rec, prec, color=c, label=f"{name} (AP={ap:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curves")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)


def save_uncertainty_plot(ci_widths: np.ndarray, y_true: np.ndarray,
                           y_pred: np.ndarray, threshold: float, path: Path) -> None:
    """Novelty 2: visualisasi lebar interval ketidakpastian prediktif MC Dropout."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram predictive interval width
    ax = axes[0]
    correct   = ci_widths[y_true == y_pred]
    incorrect = ci_widths[y_true != y_pred]
    ax.hist(correct,   bins=30, alpha=0.6, color="steelblue", label="Correct")
    ax.hist(incorrect, bins=30, alpha=0.6, color="tomato",    label="Incorrect")
    ax.axvline(threshold, color="black", ls="--", lw=1.5, label=f"Threshold={threshold}")
    ax.set_xlabel("95% Predictive Interval Width")
    ax.set_ylabel("Count")
    ax.set_title("MC Dropout Uncertainty Distribution\n(Novelty 2)")
    ax.legend()

    # Scatter: probability vs uncertainty
    ax = axes[1]
    colors_s = ["steelblue" if p == t else "tomato" for p, t in zip(y_pred, y_true)]
    ax.scatter(range(len(ci_widths)), ci_widths, c=colors_s, alpha=0.5, s=15)
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"Flag threshold={threshold}")
    n_flagged = int((ci_widths > threshold).sum())
    ax.set_title(f"Per-Sample Uncertainty\n({n_flagged} cases flagged for review)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("95% Predictive Interval Width")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)



def save_contrastive_gradcam_grid(model, sample_images_raw,
                                   sample_preds, sample_trues,
                                   last_conv_name, path: Path,
                                   preprocess_fn, image_size: int) -> None:
    """Novelty 4: Contrastive GradCAM grid with robust gradient handling."""
    n = len(sample_images_raw)
    if n == 0:
        print("[Novelty4] No XAI samples available.")
        return
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "GradCAM (Pneumonia)", "GradCAM (Non-Pneumonia)", "Contrastive Delta"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight="bold")

    candidates = _iter_conv_layers(model)
    print(f"[Novelty4] {len(candidates)} candidate conv layers. Top candidates: {[l.name for l in candidates[:3]]}")

    for row, (raw_img, pred, true) in enumerate(zip(sample_images_raw, sample_preds, sample_trues)):
        raw_rgb = np.repeat(raw_img[..., None], 3, axis=-1) if raw_img.ndim == 2 else raw_img
        raw_u8 = to_uint8(raw_rgb)
        preprocessed = preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32))).numpy()

        hm_pneu, layer_p, method_p = compute_gradcam_robust(model, preprocessed, candidates, class_index=1, debug=(row == 0))
        hm_npneu, layer_n, method_n = compute_gradcam_robust(model, preprocessed, candidates, class_index=0, debug=False)
        hm_pneu_n = hm_pneu / (float(hm_pneu.max()) + 1e-8)
        hm_npneu_n = hm_npneu / (float(hm_npneu.max()) + 1e-8)
        hm_contrast = np.clip(hm_pneu_n - hm_npneu_n, 0, 1).astype(np.float32)

        overlay_pneu = overlay_heatmap(raw_u8, hm_pneu_n)
        overlay_npneu = overlay_heatmap(raw_u8, hm_npneu_n)
        overlay_contrast = overlay_heatmap(raw_u8, hm_contrast)

        row_label = f"True:{CLASS_NAMES[true]} | Pred:{CLASS_NAMES[pred]}"
        for col, img in enumerate([raw_u8, overlay_pneu, overlay_npneu, overlay_contrast]):
            axes[row, col].imshow(img if img.ndim == 3 else img, cmap="gray" if img.ndim == 2 else None)
            axes[row, col].axis("off")
            if col == 1:
                axes[row, col].text(0.03, 0.05, f"mu={hm_pneu_n.mean():.4f}\n{method_p}\n{layer_p}", transform=axes[row, col].transAxes,
                                  fontsize=6, color="white", bbox=dict(fc="black", alpha=0.6))
            elif col == 2:
                axes[row, col].text(0.03, 0.05, f"mu={hm_npneu_n.mean():.4f}\n{method_n}\n{layer_n}", transform=axes[row, col].transAxes,
                                  fontsize=6, color="white", bbox=dict(fc="black", alpha=0.6))
            elif col == 3:
                axes[row, col].text(0.03, 0.05, f"mu={hm_contrast.mean():.4f}", transform=axes[row, col].transAxes,
                                  fontsize=6, color="white", bbox=dict(fc="black", alpha=0.6))
        axes[row, 0].set_ylabel(row_label, fontsize=8, rotation=90, labelpad=5)

    fig.suptitle("Contrastive GradCAM: Why Pneumonia, Not Non-Pneumonia? (Novelty 4)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Novelty4] Contrastive GradCAM saved: {path}")


def save_mrfo_convergence_plot(history: List[float], path: Path) -> None:
    """Novelty 5: Plot konvergensi MRFO."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(history, color="steelblue", lw=2, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cross-Validated F1 Score")
    ax.set_title("MRFO Convergence Curve\n(KNN Hyperparameter Optimization — Novelty 5)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)
    print(f"[Novelty5] MRFO convergence plot saved: {path}")


def save_tsne_plot(features, labels, path: Path, max_samples=3000) -> None:
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features, labels = features[idx], labels[idx]
    print("[INFO] Running t-SNE...")
    embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for cls, name in CLASS_NAMES.items():
        mask = labels == cls
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   label=name, alpha=0.5, s=15)
    ax.legend(); ax.set_title("t-SNE of Adaptive Wavelet Features (Novelty 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)


def save_wavelet_selection_pie(wavelet_info: Dict[str, str], path: Path) -> None:
    """Novelty 1: Pie chart distribusi wavelet yang dipilih."""
    counts = {w: 0 for w in WAVELET_CANDIDATES}
    for v in wavelet_info.values():
        counts[v] = counts.get(v, 0) + 1
    labels  = [k for k, v in counts.items() if v > 0]
    values  = [counts[k] for k in labels]
    colors  = plt.cm.Set2(np.linspace(0, 0.8, len(labels)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Adaptive Wavelet Bank Selection Distribution\n(Novelty 1 — Per-Image Optimal Wavelet)")
    fig.tight_layout()
    fig.savefig(path, dpi=IEEE_DPI)
    plt.close(fig)


def save_performance_table(val_metrics, test_metrics, out_base: Path) -> None:
    rows = []
    for model_name in val_metrics:
        vm = val_metrics[model_name]
        tm = test_metrics.get(model_name, {})
        rows.append({
            "Model": model_name,
            "Val Acc": f"{vm.get('accuracy',0):.4f}",
            "Val F1w": f"{vm.get('f1_weighted',0):.4f}",
            "Val AUC": f"{vm.get('auc',0):.4f}",
            "Test Acc": f"{tm.get('accuracy',0):.4f}",
            "Test F1w": f"{tm.get('f1_weighted',0):.4f}",
            "Test AUC": f"{tm.get('auc',0):.4f}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(str(out_base) + ".csv", index=False)
    # LaTeX
    with open(str(out_base) + ".tex", "w") as f:
        f.write(df.to_latex(index=False, caption="Model Performance Summary", label="tab:results"))
    print(f"[INFO] Performance table saved: {out_base}.csv / .tex")



def compute_binary_metrics_row(scenario_id: str, scenario_name: str, components: str,
                               y_true, y_proba, threshold: float,
                               ci_width: Optional[np.ndarray] = None) -> Dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=np.float32)
    y_pred = predict_from_proba(y_proba, threshold)
    sens, spec = sens_spec(y_true, y_pred)
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = np.nan
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
        pr = auc(rec, prec)
    except Exception:
        pr = np.nan
    return {
        "scenario_id": str(scenario_id),
        "scenario_name": scenario_name,
        "components": components,
        "decision_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc) if np.isfinite(roc_auc) else np.nan,
        "pr_auc": float(pr) if np.isfinite(pr) else np.nan,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "predictive_interval_width_95_mean": float(np.nanmean(ci_width)) if ci_width is not None else np.nan,
        "ci_width_mean": float(np.nanmean(ci_width)) if ci_width is not None else np.nan,  # legacy alias
    }


def train_default_knn(x_train, y_train) -> Pipeline:
    # Proposal Tabel 3.12 skenario 2: adaptive KNN tanpa MRFO.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski")),
    ])
    pipe.fit(x_train, y_train)
    return pipe


def cnn_predict_deterministic_batch(model: tf.keras.Model, df: pd.DataFrame,
                                    image_size: int, batch_size: int, preprocess_fn) -> np.ndarray:
    """Single forward pass with training=False for Proposal Tabel 3.12 scenario 7."""
    out = []
    image_paths = df["image_path"].values
    for start in tqdm(range(0, len(image_paths), batch_size), desc="[CNN] deterministic inference"):
        end = min(start + batch_size, len(image_paths))
        batch_imgs = np.stack([
            np.repeat(load_dicom_grayscale(p, image_size)[..., None], 3, axis=-1)
            for p in image_paths[start:end]
        ], axis=0).astype(np.float32)
        x_t = preprocess_fn(tf.constant(batch_imgs))
        p = model(x_t, training=False).numpy().astype(np.float32)
        out.append(p)
    return np.concatenate(out, axis=0)


def save_ablation_outputs(rows: List[Dict], reports: Path, plots: Path) -> pd.DataFrame:
    ensure_dir(reports); ensure_dir(plots)
    df = pd.DataFrame(rows)
    # Append/update by scenario_id for resumable per-scenario runs.
    csv_path = reports / "ablation_results.csv"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["scenario_id"], keep="last")
    df = df.sort_values("scenario_id", key=lambda s: s.astype(str)).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    (reports / "ablation_results.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")

    # IEEE-friendly LaTeX table.
    keep_cols = ["scenario_id", "scenario_name", "accuracy", "weighted_f1", "roc_auc", "sensitivity", "specificity", "ci_width_mean"]
    with open(reports / "ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(df[keep_cols].to_latex(index=False, float_format="%.4f", caption="Ablation Study Results", label="tab:ablation"))

    # Table PNG.
    fig_h = max(3.0, 0.35 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table_df = df[keep_cols].copy()
    for c in ["accuracy", "weighted_f1", "roc_auc", "sensitivity", "specificity", "ci_width_mean"]:
        table_df[c] = table_df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
    fig.tight_layout()
    fig.savefig(reports / "ablation_table.png", dpi=IEEE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Bar chart.
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df["accuracy"].astype(float), width, label="Accuracy")
    ax.bar(x, df["weighted_f1"].astype(float), width, label="Weighted F1")
    ax.bar(x + width, df["roc_auc"].astype(float), width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario_id"].astype(str), rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "ablation_barchart.png", dpi=IEEE_DPI)
    plt.close(fig)

    print("\n[ABLATION] Ranking by weighted-F1:")
    for _, r in df.sort_values("weighted_f1", ascending=False).iterrows():
        print(f"  {r['scenario_id']:>8s} | F1w={r['weighted_f1']:.4f} | {r['scenario_name']}")
    return df


def normalize_ci_width_vector(ci_width, expected_rows: Optional[int] = None, source: str = "MC-CI") -> np.ndarray:
    """Normalize (N,), (N,1), or (N,2) CI artifacts to pneumonia-class (N,)."""
    arr = np.asarray(ci_width, dtype=np.float32); shape = tuple(arr.shape)
    if arr.ndim == 1: vec = arr
    elif arr.ndim == 2 and arr.shape[1] == 1: vec = arr[:, 0]
    elif arr.ndim == 2 and arr.shape[1] >= 2: vec = arr[:, 1]
    else: raise ValueError(f"{source} unsupported CI shape {shape}")
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if expected_rows is not None and len(vec) != int(expected_rows):
        raise ValueError(f"{source} row mismatch: {len(vec)} != {expected_rows}; original={shape}")
    if not np.all(np.isfinite(vec)) or np.any(vec < -1e-8):
        raise ValueError(f"{source} contains invalid CI widths")
    return np.clip(vec, 0, None)


def combine_soft_vote_ci_width(cnn_outputs: Dict[str, Dict], reports: Path, expected_rows: int):
    """Shape-safe CI aggregation using saved soft-voting weights when available."""
    info_path = Path(reports) / "soft_voting_members.json"
    weights = {}; kept = []
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8")); weights = {str(k): float(v) for k,v in info.get("weights",{}).items()}; kept=list(info.get("kept_members",[]))
        except Exception as exc: print(f"[WARN] soft-voting CI manifest unreadable: {exc}")
    display = {"efficientnetv2":"EfficientNetV2 (MC-Dropout)","resnet50":"ResNet50 (MC-Dropout)","xrv":"TorchXRayVision-DenseNet121 (CXR-pretrained)","eva_x":"EVA-X-S"}
    used=[]; fused=np.zeros(int(expected_rows),dtype=np.float64)
    for name,out_i in cnn_outputs.items():
        if "test_ci" not in out_i: continue
        member=display.get(str(name),str(name)); w=float(weights.get(member,0.0))
        if not weights: w=1.0
        if w<=0: continue
        try: vec=normalize_ci_width_vector(out_i["test_ci"],expected_rows,f"ablation.{name}.test_ci")
        except Exception as exc: print(f"[WARN] CI excluded for {name}: {exc}"); continue
        fused += w*vec; used.append((member,w))
    if not used: return None,"no compatible stochastic member CI"
    if not weights: fused /= len(used)
    print("[ABLATION-CI] " + ", ".join(f"{n}(w={w:.3f})" for n,w in used))
    return fused.astype(np.float32), " + ".join(kept) if kept else "shape-safe stochastic members"


def run_ablation_study(train_df, val_df, test_df, y_train, y_val, y_test,
                       config: RunConfig, cache_dir: Path, reports: Path, plots: Path,
                       models: Path, existing: Dict) -> pd.DataFrame:
    """Proposal Tabel 3.12 + radiomic feature ablation on the same test set."""
    requested = {s.strip() for s in (config.ablation_scenarios or "").split(",") if s.strip()}
    rows: List[Dict] = []
    distances = [int(x) for x in config.glcm_distances.split(",")]
    angles = [int(x) for x in config.glcm_angles.split(",")]
    threshold = config.decision_threshold

    def want(sid: str) -> bool:
        return not requested or sid in requested

    def get_features(use_wavelet=True, adaptive=True, fixed_wavelet="db4", use_glcm=True, use_lbp=True):
        tag = radiomic_feature_tag(use_wavelet, adaptive, fixed_wavelet, use_glcm, use_lbp, config.wavelet_levels)
        cache, _ = compute_feature_cache(
            pd.concat([train_df, val_df, test_df], ignore_index=True),
            config.image_size, distances, angles, config.wavelet_levels,
            config.lbp_radius, config.lbp_n_points, cache_dir / f"{tag}.joblib",
            use_wavelet=use_wavelet, adaptive=adaptive, fixed_wavelet=fixed_wavelet,
            use_glcm=use_glcm, use_lbp=use_lbp, force=config.force,
        )
        return features_from_dataframe(train_df, cache), features_from_dataframe(test_df, cache)

    # 5 radiomic representation ablations, default KNN for a fair lightweight comparison.
    feature_scenarios = [
        ("F1", "Radiomic: no wavelet", "GLCM+LBP direct image + default KNN", dict(use_wavelet=False, adaptive=False, fixed_wavelet="none", use_glcm=True, use_lbp=True)),
        ("F2", "Radiomic: fixed db4", "Fixed db4 wavelet + GLCM+LBP + default KNN", dict(use_wavelet=True, adaptive=False, fixed_wavelet="db4", use_glcm=True, use_lbp=True)),
        ("F3", "Radiomic: adaptive GLCM only", "Adaptive wavelet + GLCM + default KNN", dict(use_wavelet=True, adaptive=True, fixed_wavelet="db4", use_glcm=True, use_lbp=False)),
        ("F4", "Radiomic: adaptive LBP only", "Adaptive wavelet + LBP + default KNN", dict(use_wavelet=True, adaptive=True, fixed_wavelet="db4", use_glcm=False, use_lbp=True)),
        ("F5", "Radiomic: adaptive GLCM+LBP", "Adaptive wavelet + GLCM+LBP + default KNN", dict(use_wavelet=True, adaptive=True, fixed_wavelet="db4", use_glcm=True, use_lbp=True)),
    ]
    for sid, name, comp, kwargs in feature_scenarios:
        if want(sid):
            xtr, xte = get_features(**kwargs)
            pipe = train_default_knn(xtr, y_train)
            rows.append(compute_binary_metrics_row(sid, name, comp, y_test, pipe.predict_proba(xte), threshold))

    # Proposal Tabel 3.12 pipeline scenarios.
    if want("1"):
        xtr, xte = get_features(use_wavelet=True, adaptive=False, fixed_wavelet="db4", use_glcm=True, use_lbp=True)
        pipe = train_default_knn(xtr, y_train)
        rows.append(compute_binary_metrics_row("1", "Fixed-wavelet db4 KNN baseline", "fixed db4 + default KNN", y_test, pipe.predict_proba(xte), threshold))
    if want("2"):
        xtr, xte = get_features(use_wavelet=True, adaptive=True, fixed_wavelet="db4", use_glcm=True, use_lbp=True)
        pipe = train_default_knn(xtr, y_train)
        rows.append(compute_binary_metrics_row("2", "Adaptive KNN without MRFO", "adaptive wavelet + default KNN", y_test, pipe.predict_proba(xte), threshold))
    if want("3") and "wg_knn_test_proba" in existing:
        rows.append(compute_binary_metrics_row("3", "WG-KNN with MRFO", "adaptive wavelet + MRFO-KNN", y_test, existing["wg_knn_test_proba"], threshold))
    if want("4") and "efficientnetv2" in existing.get("cnn_outputs", {}):
        out = existing["cnn_outputs"]["efficientnetv2"]
        rows.append(compute_binary_metrics_row("4", "EfficientNetV2 + MC Dropout", "EfficientNetV2S + MC Dropout", y_test, out["test_mean"], threshold, out["test_ci"]))
    if want("5") and "resnet50" in existing.get("cnn_outputs", {}):
        out = existing["cnn_outputs"]["resnet50"]
        rows.append(compute_binary_metrics_row("5", "ResNet50 + MC Dropout", "ResNet50 + MC Dropout", y_test, out["test_mean"], threshold, out["test_ci"]))
    if want("6") and "soft_test_proba" in existing:
        ci_mean, ci_desc = combine_soft_vote_ci_width(existing.get("cnn_outputs", {}), reports, len(y_test))
        rows.append(compute_binary_metrics_row("6", "Soft voting full", ci_desc, y_test, existing["soft_test_proba"], threshold, ci_mean))
    if want("7") and existing.get("cnn_outputs") and "wg_knn_test_proba" in existing:
        det_list = [existing["wg_knn_test_proba"]]
        for name, out in existing["cnn_outputs"].items():
            det_key = f"{name}_test_det"
            if det_key in existing:
                det_list.append(existing[det_key])
        if len(det_list) >= 2:
            soft_no_mc = np.mean(np.stack(det_list, axis=0), axis=0)
            rows.append(compute_binary_metrics_row("7", "Soft voting without MC Dropout", "WG-KNN + deterministic CNN 1x forward pass", y_test, soft_no_mc, threshold))
    if want("8") and "stacked_test_proba" in existing:
        rows.append(compute_binary_metrics_row("8", "Stacked ensemble full", "Logistic Regression meta-learner", y_test, existing["stacked_test_proba"], threshold))

    return save_ablation_outputs(rows, reports, plots)


def update_run_manifest(out: Path, stage_name: str, config: RunConfig, artifacts: Optional[Dict] = None) -> None:
    path = out / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"stages": {}, "config": {}}
    cfg = asdict(config)
    # Keep only stable scalar config values.
    manifest["config"].update({k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, type(None)))})
    manifest["stages"][stage_name] = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": artifacts or {},
    }
    save_json(manifest, path)



# ─────────────────────────────────────────────────────────────────────────────
# Stage artifact helpers — resumable/idempotent execution
# ─────────────────────────────────────────────────────────────────────────────

def artifact_exists(path: Path) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def require_artifact(path: Path, description: str) -> Path:
    path = Path(path)
    if not artifact_exists(path):
        raise FileNotFoundError(
            f"[ERROR] Missing {description}: {path}. Run the prerequisite stage first."
        )
    return path


def require_artifacts(paths: Dict[str, Path]) -> Dict[str, Path]:
    return {name: require_artifact(path, name) for name, path in paths.items()}


def _stage_outputs_ready(outputs: List[Path]) -> bool:
    return bool(outputs) and all(artifact_exists(Path(p)) for p in outputs)


def maybe_skip_stage(stage_name: str, outputs: List[Path], force: bool) -> bool:
    if (not force) and _stage_outputs_ready(outputs):
        print(f"[SKIP] stage {stage_name} already completed. Use --force to rerun.")
        return True
    return False


def proposal_radiomic_cache_path(output_dir: Path, config: RunConfig) -> Path:
    tag = radiomic_feature_tag(True, True, "db4", True, True, config.wavelet_levels)
    return Path(output_dir) / config.cache_dirname / f"{tag}.joblib"


def expected_radiomic_dim(config: RunConfig) -> Optional[int]:
    # Proposal-aligned full feature set: (3L+1) subbands × 6 props × 3 distances × 4 angles + 26 LBP bins.
    distances = [x for x in str(config.glcm_distances).split(",") if x]
    angles = [x for x in str(config.glcm_angles).split(",") if x]
    if config.wavelet_levels == 3 and len(distances) == 3 and len(angles) == 4 and config.lbp_n_points == 24:
        return 746
    return (3 * int(config.wavelet_levels) + 1) * 6 * len(distances) * len(angles) + (int(config.lbp_n_points) + 2)


def validate_radiomic_cache(cache: Dict[str, np.ndarray], config: RunConfig, source: Path) -> None:
    if not cache:
        raise ValueError(f"[ERROR] Radiomic cache is empty: {source}")
    first = next(iter(cache.values()))
    got = int(np.asarray(first).shape[0])
    exp = expected_radiomic_dim(config)
    if exp is not None and got != exp:
        raise ValueError(
            f"Radiomic feature dimension mismatch: expected {exp}, got {got}. "
            "Do not use legacy adaptive_wavelet_features.pkl for proposal-aligned training/evaluation."
        )
    print(f"[INFO] Loaded proposal-aligned radiomic cache: {source}")
    print(f"[INFO] Radiomic feature dimension: {got}")


def load_radiomic_cache(output_dir: Path, config: RunConfig) -> Dict[str, np.ndarray]:
    path = proposal_radiomic_cache_path(output_dir, config)
    require_artifact(path, "proposal-aligned radiomic_v2 feature cache")
    cache = joblib.load(path)
    validate_radiomic_cache(cache, config, path)
    return cache


def load_radiomic_arrays(output_dir: Path, config: RunConfig, train_df, val_df, test_df):
    cache = load_radiomic_cache(output_dir, config)
    return (
        features_from_dataframe(train_df, cache),
        features_from_dataframe(val_df, cache),
        features_from_dataframe(test_df, cache),
        cache,
    )


def load_knn_outputs(output_dir: Path) -> Dict[str, np.ndarray]:
    """Load the selected radiomics-MRFO learner outputs.

    Generic filenames are preferred. Legacy knn_* filenames are retained as a
    compatibility fallback because older fusion/statistics scripts expect them.
    """
    probs = Path(output_dir) / "probs"
    generic_val = probs / "radiomics_val.npy"
    generic_test = probs / "radiomics_test.npy"
    legacy_val = probs / "knn_val.npy"
    legacy_test = probs / "knn_test.npy"
    val_path = generic_val if artifact_exists(generic_val) else legacy_val
    test_path = generic_test if artifact_exists(generic_test) else legacy_test
    require_artifacts({
        "Best-radiomics validation probabilities": val_path,
        "Best-radiomics test probabilities": test_path,
    })
    return {
        "val": np.load(val_path),
        "test": np.load(test_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }


def selected_radiomics_label(output_dir: Path, fallback_estimator: str = "histgb") -> str:
    """Return a scientifically accurate display name for the promoted learner."""
    reports = Path(output_dir) / "reports"
    candidates = [
        reports / "radiomics_learner_selection_manifest.json",
        reports / "radiomics_best_learner.json",
        reports / "radiomics_mrfo_results.json",
        reports / "n5_mrfo_results.json",
    ]
    estimator = str(fallback_estimator).lower()
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            estimator = str(data.get("selected_estimator", data.get("estimator", estimator))).lower()
            break
        except Exception:
            continue
    pretty = {"histgb": "HistGradientBoosting", "lgbm": "LightGBM",
              "svm": "SVM", "knn": "Weighted-kNN"}.get(estimator, estimator.upper())
    return f"Radiomics-MRFO ({pretty})"


def _cnn_prob_paths(output_dir: Path, backbone_name: str) -> Dict[str, Path]:
    probs = Path(output_dir) / "probs"
    return {
        "val_mean": probs / f"{backbone_name}_val_mean.npy",
        "val_std": probs / f"{backbone_name}_val_std.npy",
        "val_ci": probs / f"{backbone_name}_val_ci.npy",
        "test_mean": probs / f"{backbone_name}_test_mean.npy",
        "test_std": probs / f"{backbone_name}_test_std.npy",
        "test_ci": probs / f"{backbone_name}_test_ci.npy",
    }


def load_cnn_outputs(output_dir: Path, y_val: np.ndarray, y_test: np.ndarray,
                     config: RunConfig, backbones: Optional[List[str]] = None) -> Dict[str, Dict]:
    names = backbones or ["efficientnetv2", "resnet50", "xrv"]
    outputs: Dict[str, Dict] = {}
    for name in names:
        paths = _cnn_prob_paths(output_dir, name)
        if not all(artifact_exists(p) for p in paths.values()):
            continue
        arr = {k: np.load(v) for k, v in paths.items()}
        thr = load_saved_threshold(output_dir, name, config.decision_threshold)
        outputs[name] = {
            "model": None,
            "preprocess_fn": get_preprocess_for_backbone(name),
            "val_mean": arr["val_mean"],
            "val_std": arr["val_std"],
            "val_ci": arr["val_ci"],
            "val_pred": predict_from_proba(arr["val_mean"], thr),
            "val_flagged": flag_uncertain_cases(arr["val_ci"], choose_uncertainty_threshold(arr["val_ci"], config)),
            "test_mean": arr["test_mean"],
            "test_std": arr["test_std"],
            "test_ci": arr["test_ci"],
            "test_pred": predict_from_proba(arr["test_mean"], thr),
            "test_flagged": flag_uncertain_cases(arr["test_ci"], choose_uncertainty_threshold(arr["val_ci"], config)),
            "uncertainty_threshold": choose_uncertainty_threshold(arr["val_ci"], config),
            "decision_threshold": thr,
        }
    if not outputs:
        raise FileNotFoundError(
            "[ERROR] Missing CNN probability artifacts. Run --stage effnet and/or --stage resnet first."
        )
    return outputs


def load_fusion_inputs(output_dir: Path, y_val: np.ndarray, y_test: np.ndarray, config: RunConfig) -> Dict:
    knn = load_knn_outputs(output_dir)
    cnn_outputs = load_cnn_outputs(output_dir, y_val, y_test, config)
    return {
        "wg_knn_val_proba": knn["val"],
        "wg_knn_test_proba": knn["test"],
        "cnn_outputs": cnn_outputs,
    }


def run_single_cnn_stage(backbone_name: str, train_df, val_df, test_df, y_train, y_val, y_test,
                         config: RunConfig, out: Path, models: Path, reports: Path, novelty: Path) -> None:
    probs_dir = ensure_dir(out / "probs")
    expected_outputs = [models / f"{backbone_name}_final.keras"] + list(_cnn_prob_paths(out, backbone_name).values())
    if maybe_skip_stage("effnet" if backbone_name == "efficientnetv2" else "resnet", expected_outputs, config.force):
        return
    model_i = train_cnn_backbone(backbone_name, train_df, val_df, y_train, config, models)
    preprocess_i = get_preprocess_for_backbone(backbone_name)
    print(f"\n[NOVELTY 2] MC Dropout Inference — {backbone_name}...")
    model_artifact_hash = sha256_file(models / f"{backbone_name}_final.keras")
    val_mean, val_std, val_ci = cnn_mc_predict_batch(
        model_i, val_df, config.image_size, config.batch_size, preprocess_i,
        n_mc=config.mc_dropout_n, output_dir=out, cache_prefix=f"{backbone_name}_val",
        resume=getattr(config, "auto_resume_cnn", True), force=config.force,
        model_artifact_sha256=model_artifact_hash, preprocessing_id=backbone_name,
    )
    test_mean, test_std, test_ci = cnn_mc_predict_batch(
        model_i, test_df, config.image_size, config.batch_size, preprocess_i,
        n_mc=config.mc_dropout_n, output_dir=out, cache_prefix=f"{backbone_name}_test",
        resume=getattr(config, "auto_resume_cnn", True), force=config.force,
        model_artifact_sha256=model_artifact_hash, preprocessing_id=backbone_name,
    )
    unc_thr = choose_uncertainty_threshold(val_ci, config)
    decision_thr = tune_threshold_from_validation(y_val, val_mean, config, backbone_name, reports, out / "plots")
    test_pred = predict_from_proba(test_mean, decision_thr)
    test_flagged = flag_uncertain_cases(test_ci, unc_thr)
    s, sp = sens_spec(y_test, test_pred)
    print(f"[Novelty2] {backbone_name}: uncertainty threshold={unc_thr:.4f}; flagged={test_flagged.sum()} / {len(test_flagged)} ({100*test_flagged.mean():.1f}%).")
    print(f"[Decision] {backbone_name}: validation-tuned threshold={decision_thr:.3f}; final-test sensitivity={s:.3f}; specificity={sp:.3f}")
    save_uncertainty_plot(test_ci, y_test, test_pred, unc_thr, novelty / f"n2_uncertainty_{backbone_name}.png")
    unc_df = test_df[["sample_id", "patientId", "label"]].copy()
    unc_df[f"{backbone_name}_mean_proba"] = test_mean[:, 1]
    unc_df[f"{backbone_name}_std"] = test_std[:, 1]
    unc_df[f"{backbone_name}_predictive_interval_width_95"] = test_ci
    unc_df[f"{backbone_name}_ci_width_95"] = test_ci  # legacy alias
    unc_df[f"{backbone_name}_decision_threshold"] = decision_thr
    unc_df[f"{backbone_name}_needs_review"] = test_flagged
    unc_df.to_csv(reports / f"n2_uncertainty_report_{backbone_name}.csv", index=False)
    np.save(probs_dir / f"{backbone_name}_val_mean.npy", val_mean)
    np.save(probs_dir / f"{backbone_name}_val_std.npy", val_std)
    np.save(probs_dir / f"{backbone_name}_val_ci.npy", val_ci)
    np.save(probs_dir / f"{backbone_name}_test_mean.npy", test_mean)
    np.save(probs_dir / f"{backbone_name}_test_std.npy", test_std)
    np.save(probs_dir / f"{backbone_name}_test_ci.npy", test_ci)
    update_run_manifest(out, "effnet" if backbone_name == "efficientnetv2" else "resnet", config, {
        "model": str(models / f"{backbone_name}_final.keras"),
        "probs": {k: str(v) for k, v in _cnn_prob_paths(out, backbone_name).items()},
    })


def run_fusion_from_saved(train_df, val_df, test_df, y_train, y_val, y_test,
                          config: RunConfig, out: Path, models: Path, reports: Path,
                          plots: Optional[Path] = None) -> Dict:
    pair_manifest_path = Path(reports) / "dl_pair_selection_manifest.json"
    if pair_manifest_path.exists():
        fused = finalize_locked_test_fusion(out, exact_deployment=True, allow_deterministic_recompute=False)
        return {
            "selected_pair_manifest": fused["manifest"],
            "soft_test_proba": fused["soft_test"],
            "stacked_test_proba": fused["stacked_test"],
        }
    probs_dir = ensure_dir(out / "probs")
    fusion_outputs = [
        probs_dir / "soft_val.npy", probs_dir / "soft_test.npy",
        probs_dir / "soft_voting_test.npy", probs_dir / "stacked_val.npy",
        probs_dir / "stacked_test.npy", probs_dir / "stacking_test.npy",
    ]
    if maybe_skip_stage("fusion", fusion_outputs, config.force):
        knn = load_knn_outputs(out)
        return {
            "wg_knn_val_proba": knn["val"],
            "wg_knn_test_proba": knn["test"],
            "cnn_outputs": load_cnn_outputs(out, y_val, y_test, config),
            "soft_test_proba": np.load(probs_dir / "soft_test.npy"),
            "stacked_test_proba": np.load(probs_dir / "stacked_test.npy"),
        }
    inputs = load_fusion_inputs(out, y_val, y_test, config)
    wg_knn_val_proba = inputs["wg_knn_val_proba"]
    wg_knn_test_proba = inputs["wg_knn_test_proba"]
    cnn_outputs = inputs["cnn_outputs"]
    cnn_names = [n for n in ["efficientnetv2", "resnet50", "xrv"] if n in cnn_outputs]
    val_proba_list = [wg_knn_val_proba] + [cnn_outputs[n]["val_mean"] for n in cnn_names]
    test_proba_list = [wg_knn_test_proba] + [cnn_outputs[n]["test_mean"] for n in cnn_names]
    _disp = {"efficientnetv2": "EfficientNetV2 (MC-Dropout)",
             "resnet50": "ResNet50 (MC-Dropout)",
             "xrv": "TorchXRayVision-DenseNet121 (CXR-pretrained)"}
    radiomics_display = selected_radiomics_label(out, config.mrfo_estimator)
    ensemble_names = [radiomics_display] + [_disp.get(n, n) for n in cnn_names]

    # ── Soft voting yang diperbaiki ────────────────────────────────────────────
    # Masalah lama: soft voting = rata-rata polos [kNN, eff, res]. kNN yang lemah
    # (AUC ~0.71, sensitivitas ~0) menyeret soft voting jauh ke bawah. Perbaikan:
    #   1) hitung AUC validasi tiap anggota,
    #   2) buang anggota di bawah `soft_vote_auc_floor` (kNN otomatis gugur),
    #   3) rata-rata tertimbang sisanya dengan bobot ~ (AUC - 0.5).
    # kNN TETAP dipakai sebagai fitur di stacked meta-learner (LR bisa
    # mempelajari bobotnya sendiri), jadi "novelty" WG-KNN tidak dibuang.
    y_val_arr = np.asarray(y_val).astype(int)
    auc_floor = float(getattr(config, "soft_vote_auc_floor", 0.78))
    member_aucs = []
    for name, vp in zip(ensemble_names, val_proba_list):
        try:
            a = float(roc_auc_score(y_val_arr, np.asarray(vp)[:, 1]))
        except Exception:
            a = 0.0
        member_aucs.append(a)
    keep_idx = [i for i, a in enumerate(member_aucs) if a >= auc_floor]
    if not keep_idx:  # jangan sampai kosong; pakai anggota terbaik
        keep_idx = [int(np.argmax(member_aucs))]
    weights = np.array([max(member_aucs[i] - 0.5, 1e-6) for i in keep_idx], dtype=np.float64)
    weights = weights / weights.sum()
    print("[FUSION] Soft-voting members (val AUC): " +
          ", ".join(f"{ensemble_names[i]}={member_aucs[i]:.3f}" for i in range(len(ensemble_names))))
    print("[FUSION] Kept in soft vote: " +
          ", ".join(f"{ensemble_names[i]}(w={weights[j]:.2f})" for j, i in enumerate(keep_idx)) +
          f" | AUC floor={auc_floor}")
    save_json({
        "auc_floor": auc_floor,
        "member_val_auc": {ensemble_names[i]: member_aucs[i] for i in range(len(ensemble_names))},
        "kept_members": [ensemble_names[i] for i in keep_idx],
        "weights": {ensemble_names[keep_idx[j]]: float(weights[j]) for j in range(len(keep_idx))},
        "note": "Weak members (below AUC floor, e.g. WG-KNN) excluded from soft voting; still used by stacked meta-learner.",
    }, reports / "soft_voting_members.json")
    val_stack = np.stack([val_proba_list[i] for i in keep_idx], axis=0)
    test_stack = np.stack([test_proba_list[i] for i in keep_idx], axis=0)
    w = weights.reshape(-1, 1, 1)
    soft_val_proba = np.sum(val_stack * w, axis=0)
    soft_test_proba = np.sum(test_stack * w, axis=0)
    if config.use_stacking:
        stacked_val_proba = stacking_oof_probabilities(
            val_proba_list, y_val, n_splits=5, seed=config.seed)
        meta_clf = train_stacking_meta_learner_from_list(val_proba_list, y_val)
        joblib.dump(meta_clf, models / "meta_learner.pkl")
        joblib.dump(meta_clf, models / "stacking_lr.joblib")
        save_json({
            "feature_order": ensemble_names,
            "n_features": int(2 * len(val_proba_list)),
            "validation_probabilities": "5-fold out-of-fold",
            "final_meta_fit": "all validation rows",
        }, reports / "meta_learner_features.json")
        stacked_test_proba = meta_clf.predict_proba(np.concatenate(test_proba_list, axis=1))
    else:
        stacked_val_proba = soft_val_proba
        stacked_test_proba = soft_test_proba
    plots_dir = plots or ensure_dir(Path(out) / "plots")
    soft_thr = tune_threshold_from_validation(y_val, soft_val_proba, config, "soft_voting", reports, plots_dir)
    stacked_thr = tune_threshold_from_validation(y_val, stacked_val_proba, config, "stacked_ensemble", reports, plots_dir)
    save_json({
        "soft_voting": float(soft_thr),
        "stacked_ensemble": float(stacked_thr),
        "selection_split": "validation",
        "test_set_used_for_selection": False,
    }, reports / "fusion_thresholds_validation_tuned.json")
    np.save(probs_dir / "soft_val.npy", soft_val_proba)
    np.save(probs_dir / "soft_test.npy", soft_test_proba)
    np.save(probs_dir / "soft_voting_test.npy", soft_test_proba)
    np.save(probs_dir / "stacked_val.npy", stacked_val_proba)
    np.save(probs_dir / "stacked_test.npy", stacked_test_proba)
    np.save(probs_dir / "stacking_test.npy", stacked_test_proba)
    update_run_manifest(out, "fusion", config, {
        "probs": {
            "soft_val": str(probs_dir / "soft_val.npy"),
            "soft_test": str(probs_dir / "soft_test.npy"),
            "soft_voting_test": str(probs_dir / "soft_voting_test.npy"),
            "stacked_val": str(probs_dir / "stacked_val.npy"),
            "stacked_test": str(probs_dir / "stacked_test.npy"),
            "stacking_test": str(probs_dir / "stacking_test.npy"),
        },
        "meta_learner": str(models / "meta_learner.pkl"),
    })
    print("[STAGE] fusion complete from saved probability artifacts; no KNN/CNN retraining was performed.")
    return {
        **inputs,
        "soft_val_proba": soft_val_proba,
        "soft_test_proba": soft_test_proba,
        "stacked_val_proba": stacked_val_proba,
        "stacked_test_proba": stacked_test_proba,
        "soft_threshold": soft_thr,
        "stacked_threshold": stacked_thr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q1 validation-only four-candidate DL selection
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_radiomic_cache_for_oof(out: Path, config: RunConfig) -> Path:
    """Resolve the development-only radiomics cache; never accept an all-data legacy cache."""
    cache_dir = Path(out) / config.cache_dirname
    candidates = sorted(cache_dir.glob("radiomic_v3_development_adaptive_L*_glcm_lbp.joblib"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"Development-only radiomics cache was not found in {cache_dir}. "
        "Run --stage development_features first. Legacy caches containing test images are rejected."
    )




def run_development_feature_stage(train_df, val_df, config: RunConfig, out: Path, novelty: Path) -> Path:
    """Extract radiomics only for train+validation before pair selection."""
    if int(config.image_size) != 224:
        raise ValueError("Strict Q1 selected-pair protocol requires image_size=224")
    dev = ensure_development_table(out, train_df, val_df)
    cache_dir = ensure_dir(Path(out) / config.cache_dirname)
    tag = f"radiomic_v3_development_adaptive_L{config.wavelet_levels}_glcm_lbp"
    cache_path = cache_dir / f"{tag}.joblib"
    info_path = cache_path.with_suffix(".wavelet_info.json")
    distances = [int(x) for x in config.glcm_distances.split(",")]
    angles = [int(x) for x in config.glcm_angles.split(",")]
    cache, wavelet_info = compute_feature_cache(
        dev, config.image_size, distances, angles, config.wavelet_levels,
        config.lbp_radius, config.lbp_n_points, cache_path,
        use_wavelet=True, adaptive=True, fixed_wavelet="db4",
        use_glcm=True, use_lbp=True, force=config.force,
    )
    validate_radiomic_cache(cache, config, cache_path)
    save_json(wavelet_info, info_path)
    if novelty is not None:
        save_wavelet_selection_pie(wavelet_info, Path(novelty) / "n1_wavelet_selection_development_only.png")
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "scope": "development_train_plus_validation_only",
        "n_samples": int(len(dev)), "image_size": int(config.image_size),
        "test_images_accessed": False, "test_features_generated": False,
        "cache_path": str(cache_path), "cache_sha256": sha256_file(cache_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(out) / "reports" / "development_radiomics_feature_manifest.json")
    return cache_path


def _build_locked_test_feature_cache(test_df, config: RunConfig, out: Path) -> Path:
    """Create test radiomics features only after pair and deployment are locked."""
    load_deployment_manifest(out, strict=True)
    cache_dir = ensure_dir(Path(out) / config.cache_dirname)
    tag = f"radiomic_v3_locked_test_adaptive_L{config.wavelet_levels}_glcm_lbp"
    cache_path = cache_dir / f"{tag}.joblib"
    distances = [int(x) for x in config.glcm_distances.split(",")]
    angles = [int(x) for x in config.glcm_angles.split(",")]
    cache, wavelet_info = compute_feature_cache(
        test_df, config.image_size, distances, angles, config.wavelet_levels,
        config.lbp_radius, config.lbp_n_points, cache_path,
        use_wavelet=True, adaptive=True, fixed_wavelet="db4",
        use_glcm=True, use_lbp=True, force=False,
    )
    validate_radiomic_cache(cache, config, cache_path)
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17", "scope": "locked_test_only",
        "n_samples": int(len(test_df)), "pair_locked_before_extraction": True,
        "test_labels_used": False, "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(out) / "reports" / "locked_test_radiomics_feature_manifest.json")
    return cache_path


def load_radiomics_selection_manifest(out: Path, strict: bool = True) -> Dict:
    path = Path(out) / "reports" / "radiomics_learner_selection_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing radiomics learner-selection manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"knn", "histgb", "lgbm", "svm"}
    got = set(data.get("candidate_estimators", []))
    if strict and got != required:
        raise RuntimeError(f"Radiomics candidates must be exactly {sorted(required)}, got {sorted(got)}")
    if data.get("test_set_used_for_selection") is not False:
        raise RuntimeError("Unsafe radiomics manifest: test set was used for selection")
    if strict and data.get("selection_scope") != "development_outer_oof_fixed_best_radiomics":
        raise RuntimeError(f"Radiomics selection is not fixed-best outer OOF: {data.get('selection_scope')}")
    if strict and data.get("oof_deployment_family_identical") is not True:
        raise RuntimeError("Radiomics OOF family and deployment family must be identical")
    if strict and data.get("outer_holdout_used_for_hyperparameter_tuning") is not False:
        raise RuntimeError("Unsafe radiomics manifest: outer holdout influenced MRFO tuning")
    selected = str(data.get("selected_estimator", "")).lower()
    if selected not in required:
        raise RuntimeError(f"Invalid selected radiomics estimator: {selected}")
    return data


def _radiomics_estimator_identity(model) -> Dict:
    clf = getattr(model, "named_steps", {}).get("clf", model)
    info = {"class_name": clf.__class__.__name__}
    if hasattr(clf, "kernel"):
        info["kernel"] = str(getattr(clf, "kernel"))
    return info


def assert_radiomics_estimator_family(model, selected_estimator: str) -> Dict:
    """Fail fast when a cached/refit model is not the locked learner family."""
    clf = getattr(model, "named_steps", {}).get("clf", model)
    selected = str(selected_estimator).lower()
    expected_class_names = {
        "knn": {"KNeighborsClassifier"},
        "histgb": {"HistGradientBoostingClassifier"},
        "lgbm": {"LGBMClassifier"},
        "svm": {"SVC"},
    }
    actual = clf.__class__.__name__
    if selected not in expected_class_names or actual not in expected_class_names[selected]:
        raise RuntimeError(
            f"Radiomics deployment family mismatch: selected={selected}, actual={actual}"
        )
    if selected == "svm" and str(getattr(clf, "kernel", "")).lower() != "rbf":
        raise RuntimeError(
            f"Locked SVM radiomics learner must use kernel='rbf', got {getattr(clf, 'kernel', None)!r}"
        )
    return _radiomics_estimator_identity(model)


def _stable_sha256_values(values) -> str:
    payload = "\n".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _radiomics_outer_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else 0.5,
        "auprc": float(average_precision_score(y_true, p)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def _radiomics_cache_expected(dev: pd.DataFrame, train_mask: np.ndarray, config: RunConfig,
                               cache_path: Path, X_dev: np.ndarray, estimator: str,
                               outer_fold: int, fold_seed: int,
                               feature_cache_sha256: Optional[str] = None) -> Dict:
    tr = dev.loc[train_mask].reset_index(drop=True)
    return {
        "schema": "aura_cxr_radiomics_nested_candidate_q1_v17",
        "estimator": str(estimator),
        "outer_fold": int(outer_fold),
        "fitness_metric": str(config.mrfo_fitness),
        "pop_size": int(config.radiomics_final_pop),
        "max_iter": int(config.radiomics_final_iter),
        "inner_cv_folds": int(config.radiomics_final_inner_folds),
        "seed": int(fold_seed),
        "mrfo_subsample": int(config.mrfo_subsample),
        "training_rows": int(train_mask.sum()),
        "training_patient_sha256": _stable_sha256_values(tr["patientId"].astype(str).tolist()),
        "training_label_sha256": _stable_sha256_values(tr["label"].astype(int).tolist()),
        "feature_cache_path": str(Path(cache_path).resolve()),
        "feature_cache_sha256": feature_cache_sha256 or sha256_file(cache_path),
        "feature_dimension": int(X_dev.shape[1]),
        "image_size": int(config.image_size),
        "wavelet_levels": int(config.wavelet_levels),
        "glcm_distances": str(config.glcm_distances),
        "glcm_angles": str(config.glcm_angles),
        "lbp_radius": int(config.lbp_radius),
        "lbp_n_points": int(config.lbp_n_points),
        "test_set_used": False,
        "external_kermany_used": False,
    }


def _validate_nested_radiomics_cache(payload: Dict, expected: Dict, model_path: Path) -> None:
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if not model_path.exists():
        mismatches["model_path"] = ("missing", str(model_path))
    elif payload.get("model_sha256") != sha256_file(model_path):
        mismatches["model_sha256"] = (payload.get("model_sha256"), sha256_file(model_path))
    if mismatches:
        raise RuntimeError(
            "Stale or incompatible nested-radiomics cache detected. "
            f"Mismatches: {mismatches}. Use a new OUT directory or --force before the locked test is opened."
        )



def _parse_screen_seeds(value: str) -> List[int]:
    seeds = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(seeds) < 1:
        raise ValueError("radiomics_screen_seeds must contain at least one integer")
    return seeds


def _stratified_screen_sample(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    n = len(y)
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.arange(n)
    keep, _ = train_test_split(
        idx, train_size=int(max_rows), stratify=y, random_state=int(seed)
    )
    return np.sort(np.asarray(keep, dtype=np.int64))


def _screen_radiomics_inside_outer_train(X_train: np.ndarray, y_train: np.ndarray,
                                          patient_ids: Sequence[str], config: RunConfig,
                                          outer_fold: int, reports: Path,
                                          feature_cache_sha256: str) -> Tuple[List[str], pd.DataFrame, List[Dict]]:
    """Cheap, two-seed screening strictly inside one outer-training fold.

    Promotion is stability-aware: every learner receives a rank for each seed;
    top-2 frequency and mean rank are evaluated before pooled performance. The
    outer holdout is never read during screening.
    """
    candidates = ["knn", "histgb", "lgbm", "svm"]
    seeds = _parse_screen_seeds(config.radiomics_screen_seeds)
    manifest_path = Path(reports) / f"radiomics_screening_outer_fold_{int(outer_fold)}_manifest.json"
    runs_path = Path(reports) / f"radiomics_screening_outer_fold_{int(outer_fold)}_runs.csv"
    summary_path = Path(reports) / f"radiomics_screening_outer_fold_{int(outer_fold)}.csv"
    seed_rank_path = Path(reports) / f"radiomics_screening_outer_fold_{int(outer_fold)}_seed_ranks.csv"
    expected = {
        "schema": "aura_cxr_radiomics_screening_cache_q1_v17",
        "outer_fold": int(outer_fold),
        "patient_sha256": _stable_sha256_values(patient_ids),
        "label_sha256": _stable_sha256_values(np.asarray(y_train, dtype=int).tolist()),
        "feature_cache_sha256": str(feature_cache_sha256),
        "feature_dimension": int(X_train.shape[1]),
        "screen_rows": int(config.radiomics_screen_rows),
        "screen_folds": int(config.radiomics_screen_folds),
        "screen_inner_folds": int(config.radiomics_screen_inner_folds),
        "screen_population": int(config.radiomics_screen_pop),
        "screen_iterations": int(config.radiomics_screen_iter),
        "screen_seeds": seeds,
        "top_k": int(config.radiomics_screen_top_k),
        "fitness": str(config.mrfo_fitness),
        "image_size": int(config.image_size),
        "wavelet_levels": int(config.wavelet_levels),
        "glcm_distances": str(config.glcm_distances),
        "glcm_angles": str(config.glcm_angles),
        "lbp_radius": int(config.lbp_radius),
        "lbp_n_points": int(config.lbp_n_points),
        "promotion_policy": "per_seed_rank_then_top2_frequency_then_mean_rank_then_pooled_metrics",
    }
    needed = [manifest_path, runs_path, summary_path, seed_rank_path]
    if all(x.exists() for x in needed) and not (config.force or config.pair_selection_force):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
        hashes = {
            "runs_sha256": sha256_file(runs_path),
            "summary_sha256": sha256_file(summary_path),
            "seed_rank_sha256": sha256_file(seed_rank_path),
        }
        for key, value in hashes.items():
            if payload.get(key) != value:
                mismatches[key] = (payload.get(key), value)
        if mismatches:
            raise RuntimeError(f"Stale radiomics screening cache: {mismatches}")
        records_df = pd.read_csv(runs_path)
        summary = pd.read_csv(summary_path)
        promoted_col = summary["promoted_to_global_refinement"]
        promoted_mask = promoted_col if promoted_col.dtype == bool else promoted_col.astype(str).str.lower().eq("true")
        promoted = summary.loc[promoted_mask, "estimator"].astype(str).tolist()
        print(f"[RADIOMICS-SCREEN] reuse outer={outer_fold} local_top2={promoted}")
        return promoted, summary, records_df.to_dict(orient="records")

    complexity = {"knn": 0, "histgb": 1, "lgbm": 2, "svm": 3}
    records = []
    for estimator in candidates:
        for seed in seeds:
            sample_seed = int(seed) + 100003 * int(outer_fold)
            sample_idx = _stratified_screen_sample(y_train, int(config.radiomics_screen_rows), sample_seed)
            Xs, ys = X_train[sample_idx], y_train[sample_idx]
            skf = StratifiedKFold(
                n_splits=int(config.radiomics_screen_folds), shuffle=True, random_state=sample_seed
            )
            for screen_fold, (tr, va) in enumerate(skf.split(np.zeros(len(ys)), ys)):
                cfg = copy.deepcopy(config)
                cfg.mrfo_estimator = estimator
                cfg.mrfo_pop_size = int(config.radiomics_screen_pop)
                cfg.mrfo_max_iter = int(config.radiomics_screen_iter)
                cfg.mrfo_cv_folds = int(config.radiomics_screen_inner_folds)
                cfg.mrfo_subsample = 0
                cfg.seed = sample_seed + 1009 * int(screen_fold)
                cfg.mrfo_restart_patience = min(int(config.mrfo_restart_patience), 5)
                model, info = train_radiomics_mrfo(Xs[tr], ys[tr], cfg)
                pv = normalize_binary_probability(
                    model.predict_proba(Xs[va]), len(va),
                    f"screen {estimator} outer={outer_fold} seed={seed} fold={screen_fold}",
                )
                m = _radiomics_outer_metrics(ys[va], pv)
                records.append({
                    "outer_fold": int(outer_fold), "estimator": estimator,
                    "screen_seed": int(seed), "screen_fold": int(screen_fold),
                    "sample_rows": int(len(sample_idx)),
                    "screen_train_rows": int(len(tr)), "screen_holdout_rows": int(len(va)),
                    "mrfo_population": int(config.radiomics_screen_pop),
                    "mrfo_iterations": int(config.radiomics_screen_iter),
                    "inner_cv_folds": int(config.radiomics_screen_inner_folds),
                    "best_inner_cv_score": float(info["best_cv_score"]),
                    **m,
                })

    df = pd.DataFrame(records)
    seed_summary = df.groupby(["screen_seed", "estimator"], as_index=False).agg(
        seed_f1_macro_mean=("f1_macro", "mean"),
        seed_f1_macro_sd=("f1_macro", "std"),
        seed_auc_mean=("auc", "mean"),
        seed_auprc_mean=("auprc", "mean"),
        seed_brier_mean=("brier", "mean"),
        seed_inner_cv_mean=("best_inner_cv_score", "mean"),
    ).fillna(0.0)
    seed_summary["complexity_rank"] = seed_summary["estimator"].map(complexity).astype(int)
    primary_seed = "seed_f1_macro_mean" if str(config.mrfo_fitness) == "f1_macro" else "seed_auc_mean"
    ranked_parts = []
    for seed, part in seed_summary.groupby("screen_seed", sort=True):
        part = part.sort_values(
            [primary_seed, "seed_auprc_mean", "seed_brier_mean", "seed_f1_macro_sd", "complexity_rank"],
            ascending=[False, False, True, True, True], kind="mergesort",
        ).reset_index(drop=True)
        part["seed_rank"] = np.arange(1, len(part) + 1, dtype=int)
        part["seed_top2"] = part["seed_rank"] <= int(config.radiomics_screen_top_k)
        ranked_parts.append(part)
    seed_ranks = pd.concat(ranked_parts, ignore_index=True)

    pooled = df.groupby("estimator", as_index=False).agg(
        screen_f1_macro_mean=("f1_macro", "mean"),
        screen_f1_macro_sd=("f1_macro", "std"),
        screen_auc_mean=("auc", "mean"),
        screen_auprc_mean=("auprc", "mean"),
        screen_brier_mean=("brier", "mean"),
        screen_inner_cv_mean=("best_inner_cv_score", "mean"),
    ).fillna(0.0)
    stability = seed_ranks.groupby("estimator", as_index=False).agg(
        top2_seed_count=("seed_top2", "sum"),
        mean_seed_rank=("seed_rank", "mean"),
        worst_seed_rank=("seed_rank", "max"),
        seed_rank_sd=("seed_rank", "std"),
    ).fillna(0.0)
    summary = pooled.merge(stability, on="estimator", how="left")
    summary["top2_seed_frequency"] = summary["top2_seed_count"] / max(len(seeds), 1)
    summary["complexity_rank"] = summary["estimator"].map(complexity).astype(int)
    primary = "screen_f1_macro_mean" if str(config.mrfo_fitness) == "f1_macro" else "screen_auc_mean"
    summary = summary.sort_values(
        ["top2_seed_count", "mean_seed_rank", primary, "screen_auprc_mean", "screen_brier_mean", "screen_f1_macro_sd", "complexity_rank"],
        ascending=[False, True, False, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    top_k = max(1, min(int(config.radiomics_screen_top_k), len(summary)))
    promoted = summary.head(top_k)["estimator"].astype(str).tolist()
    summary["promoted_to_global_refinement"] = summary["estimator"].isin(promoted)

    df.to_csv(runs_path, index=False)
    seed_ranks.to_csv(seed_rank_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_json({
        **expected,
        "promoted_estimators": promoted,
        "runs_sha256": sha256_file(runs_path),
        "summary_sha256": sha256_file(summary_path),
        "seed_rank_sha256": sha256_file(seed_rank_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, manifest_path)
    return promoted, summary, records


def _fixed_candidate_oof_metrics(y: np.ndarray, p: np.ndarray, folds: np.ndarray,
                                 min_sensitivity: float) -> Dict:
    """Aggregate outer-OOF metrics with a cross-fitted operating threshold."""
    y = np.asarray(y, dtype=int)
    p = normalize_binary_probability(p, len(y), "fixed radiomics candidate OOF")
    folds = np.asarray(folds, dtype=int)
    pred = np.zeros(len(y), dtype=int)
    thresholds = []
    fold_aucs = []
    for fold in sorted(np.unique(folds)):
        hold = folds == fold
        train = ~hold
        threshold = tune_threshold(y[train], p[train], min_sensitivity=float(min_sensitivity))
        pred[hold] = (p[hold] >= threshold).astype(int)
        thresholds.append({"outer_fold": int(fold), "threshold_from_other_outer_folds": float(threshold)})
        if len(np.unique(y[hold])) > 1:
            fold_aucs.append(float(roc_auc_score(y[hold], p[hold])))
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "crossfitted_f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "fold_auc_mean": float(np.mean(fold_aucs)) if fold_aucs else 0.5,
        "fold_auc_sd": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "crossfitted_thresholds": thresholds,
    }


def run_radiomics_nested_development_oof(train_df, val_df, config: RunConfig,
                                           out: Path, models: Path, reports: Path) -> str:
    """Select one fixed best radiomics learner using development outer OOF.

    Phase 1 performs cheap stability-aware screening strictly inside each outer
    training fold. The screening summaries are pooled to nominate two global
    learner families. Phase 2 evaluates both nominated families on every outer
    fold with full outer-training rows and nested MRFO tuning. A single learner
    is selected from aggregate outer-OOF results; its complete OOF vector is the
    only radiomics feature used by DL-pair selection and the meta-learner. The
    same learner family is later refit on the full development set.
    """
    assert_selection_mutable(out, "run fixed-best radiomics selection")
    if int(config.image_size) != 224:
        raise ValueError("Strict Q1 radiomics selection requires image_size=224")
    if int(config.oof_folds) != 5:
        raise ValueError("Final radiomics outer OOF must use five folds")
    requested = [x.strip().lower() for x in str(config.radiomics_candidates).split(",") if x.strip()]
    if sorted(requested) != sorted(["knn", "histgb", "lgbm", "svm"]):
        raise ValueError("Strict screening requires exactly knn,histgb,lgbm,svm")
    if int(config.radiomics_screen_top_k) != 2:
        raise ValueError("Final protocol requires radiomics_screen_top_k=2")
    if int(config.mrfo_subsample) != 0:
        raise ValueError("Final refinement requires all outer-training rows (--mrfo_subsample 0)")

    dev = ensure_development_table(out, train_df, val_df)
    folds = make_patient_oof_folds(dev, int(config.oof_folds), int(config.seed),
                                   bool(config.pair_selection_force), out)
    cache_path = _resolve_radiomic_cache_for_oof(out, config)
    cache = joblib.load(cache_path)
    X_dev = features_from_dataframe(dev, cache).astype(np.float32, copy=False)
    y_dev = dev["label"].to_numpy(dtype=int)
    feature_cache_sha = sha256_file(cache_path)
    if X_dev.ndim != 2 or len(X_dev) != len(dev):
        raise RuntimeError(f"Invalid development radiomics matrix: {X_dev.shape}")

    candidates = ["knn", "histgb", "lgbm", "svm"]
    display = {"knn": "Weighted-kNN", "histgb": "HistGradientBoosting",
               "lgbm": "LightGBM", "svm": "RBF-SVM"}
    complexity = {"knn": 0, "histgb": 1, "lgbm": 2, "svm": 3}
    nested_root = ensure_dir(Path(models) / "fixed_best_radiomics")

    # Phase 1: screen in every outer-training fold, then derive one global top-2
    # list using explicit per-seed rank stability. No outer-holdout predictions
    # or labels enter this phase.
    screening_records, screening_summaries = [], []
    for fold in sorted(np.unique(folds)):
        tr_mask = folds != fold
        promoted_local, summary, records = _screen_radiomics_inside_outer_train(
            X_dev[tr_mask], y_dev[tr_mask],
            dev.loc[tr_mask, "patientId"].astype(str).tolist(),
            config, int(fold), reports, feature_cache_sha,
        )
        screening_records.extend(records)
        tmp = summary.copy()
        tmp["outer_fold"] = int(fold)
        tmp["locally_promoted"] = tmp["estimator"].isin(promoted_local)
        screening_summaries.append(tmp)

    screen_all = pd.concat(screening_summaries, ignore_index=True)
    global_screen = screen_all.groupby("estimator", as_index=False).agg(
        top2_seed_count=("top2_seed_count", "sum"),
        outer_folds_promoted=("locally_promoted", "sum"),
        mean_seed_rank=("mean_seed_rank", "mean"),
        worst_seed_rank=("worst_seed_rank", "max"),
        seed_rank_sd=("seed_rank_sd", "mean"),
        screen_f1_macro_mean=("screen_f1_macro_mean", "mean"),
        screen_f1_macro_sd=("screen_f1_macro_sd", "mean"),
        screen_auc_mean=("screen_auc_mean", "mean"),
        screen_auprc_mean=("screen_auprc_mean", "mean"),
        screen_brier_mean=("screen_brier_mean", "mean"),
        screen_inner_cv_mean=("screen_inner_cv_mean", "mean"),
    ).fillna(0.0)
    total_seed_opportunities = int(config.oof_folds) * len(_parse_screen_seeds(config.radiomics_screen_seeds))
    global_screen["top2_seed_frequency"] = global_screen["top2_seed_count"] / max(total_seed_opportunities, 1)
    global_screen["complexity_rank"] = global_screen["estimator"].map(complexity).astype(int)
    primary_screen = "screen_f1_macro_mean" if str(config.mrfo_fitness) == "f1_macro" else "screen_auc_mean"
    global_screen = global_screen.sort_values(
        ["top2_seed_count", "outer_folds_promoted", "mean_seed_rank", primary_screen,
         "screen_auprc_mean", "screen_brier_mean", "screen_f1_macro_sd", "complexity_rank"],
        ascending=[False, False, True, False, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    global_top2 = global_screen.head(2)["estimator"].astype(str).tolist()
    global_screen["promoted_to_fixed_candidate_oof"] = global_screen["estimator"].isin(global_top2)
    global_screen_path = Path(reports) / "radiomics_global_screening_stability.csv"
    global_screen.to_csv(global_screen_path, index=False)
    global_top2_hash = hashlib.sha256(
        json.dumps({"global_top2": global_top2, "summary_sha256": sha256_file(global_screen_path)}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    print(f"[RADIOMICS-SCREEN] global top-2={global_top2}")

    # Phase 2: BOTH global candidates are refined on ALL five outer folds.
    candidate_oof = {est: np.full(len(dev), np.nan, dtype=np.float64) for est in global_top2}
    refinement_records = []
    for fold in sorted(np.unique(folds)):
        tr_mask, va_mask = folds != fold, folds == fold
        Xtr, ytr = X_dev[tr_mask], y_dev[tr_mask]
        for estimator in global_top2:
            est_dir = ensure_dir(nested_root / estimator / f"outer_fold_{int(fold)}")
            model_path = est_dir / "model.pkl"
            info_path = est_dir / "refinement_manifest.json"
            cfg = copy.deepcopy(config)
            cfg.mrfo_estimator = estimator
            cfg.mrfo_pop_size = int(config.radiomics_final_pop)
            cfg.mrfo_max_iter = int(config.radiomics_final_iter)
            cfg.mrfo_cv_folds = int(config.radiomics_final_inner_folds)
            cfg.mrfo_subsample = 0
            cfg.seed = int(config.seed) + 10007 * int(fold) + 101 * complexity[estimator]
            expected = {
                "schema": "aura_cxr_radiomics_fixed_best_refinement_q1_v17",
                "outer_fold": int(fold), "estimator": estimator,
                "global_top2": list(global_top2), "global_top2_hash": global_top2_hash,
                "outer_train_patient_sha256": _stable_sha256_values(dev.loc[tr_mask, "patientId"].astype(str).tolist()),
                "outer_train_label_sha256": _stable_sha256_values(ytr.tolist()),
                "feature_cache_sha256": feature_cache_sha,
                "feature_dimension": int(Xtr.shape[1]), "image_size": 224,
                "training_rows": int(tr_mask.sum()),
                "fitness_metric": str(config.mrfo_fitness),
                "mrfo_population": int(config.radiomics_final_pop),
                "mrfo_iterations": int(config.radiomics_final_iter),
                "inner_cv_folds": int(config.radiomics_final_inner_folds),
                "mrfo_subsample": 0, "restart_patience": int(config.mrfo_restart_patience),
                "wavelet_levels": int(config.wavelet_levels),
                "glcm_distances": str(config.glcm_distances),
                "glcm_angles": str(config.glcm_angles),
                "lbp_radius": int(config.lbp_radius),
                "lbp_n_points": int(config.lbp_n_points),
                "seed": int(cfg.seed),
                "outer_holdout_used_for_tuning": False,
            }
            if model_path.exists() and info_path.exists() and not (config.force or config.pair_selection_force):
                payload = json.loads(info_path.read_text(encoding="utf-8"))
                mismatches = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
                if payload.get("model_sha256") != sha256_file(model_path):
                    mismatches["model_sha256"] = (payload.get("model_sha256"), sha256_file(model_path))
                if mismatches:
                    raise RuntimeError(f"Stale fixed-best radiomics refinement cache: {mismatches}")
                model = joblib.load(model_path)
            else:
                model, info = train_radiomics_mrfo(Xtr, ytr, cfg)
                joblib.dump(model, model_path)
                payload = {
                    **expected, **info,
                    "best_cv_score": float(info["best_cv_score"]),
                    "best_params": info.get("best_params", {}),
                    "model_identity": _radiomics_estimator_identity(model),
                    "model_path": str(model_path.resolve()),
                    "model_sha256": sha256_file(model_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                save_json(payload, info_path)
            p_hold = normalize_binary_probability(
                model.predict_proba(X_dev[va_mask]), int(va_mask.sum()),
                f"fixed best radiomics {estimator} outer fold {fold}",
            )
            candidate_oof[estimator][va_mask] = p_hold
            refinement_records.append({
                "outer_fold": int(fold), "estimator": estimator,
                "best_inner_cv_score": float(payload["best_cv_score"]),
                "best_params": payload.get("best_params", {}),
                "model_path": str(model_path.resolve()), "model_sha256": sha256_file(model_path),
                "outer_train_rows": int(tr_mask.sum()), "outer_holdout_rows": int(va_mask.sum()),
                "outer_holdout_used_for_selection": False,
                **_radiomics_outer_metrics(y_dev[va_mask], p_hold),
            })
            print(f"[RADIOMICS-FIXED] outer={fold} estimator={estimator} inner={float(payload['best_cv_score']):.6f}")

    comparison_rows = []
    candidate_paths = {}
    for estimator in global_top2:
        p_oof = candidate_oof[estimator]
        if not np.all(np.isfinite(p_oof)):
            raise RuntimeError(f"Incomplete fixed candidate OOF for {estimator}")
        metrics = _fixed_candidate_oof_metrics(
            y_dev, p_oof, folds, float(config.threshold_min_sensitivity)
        )
        save_oof_artifact(
            out, f"radiomics_{estimator}", p_oof, dev, folds,
            source=f"fixed global radiomics candidate {estimator}; nested MRFO in every outer fold",
            extra={
                "fixed_single_learner_family": estimator,
                "selection_scope": "development_outer_oof_fixed_best_radiomics",
                "outer_holdout_used_for_inner_tuning": False,
                "global_screen_top2": global_top2,
            },
        )
        oof_path = Path(out) / "probs" / f"oof_radiomics_{estimator}.npz"
        candidate_paths[estimator] = {"path": str(oof_path.resolve()), "sha256": sha256_file(oof_path)}
        comparison_rows.append({
            "estimator": estimator, "display_name": display[estimator],
            "complexity_rank": complexity[estimator], **metrics,
        })

    comparison = pd.DataFrame(comparison_rows)
    primary_final = "crossfitted_f1_macro" if str(config.mrfo_fitness) == "f1_macro" else "fold_auc_mean"
    tolerance = max(0.0, float(config.radiomics_selection_tolerance))
    best_primary = float(comparison[primary_final].max())
    comparison["selection_tolerance"] = tolerance
    comparison["best_primary_value"] = best_primary
    comparison["within_selection_tolerance"] = comparison[primary_final] >= (best_primary - tolerance)
    eligible = comparison.loc[comparison["within_selection_tolerance"]].copy()
    eligible = eligible.sort_values(
        ["auc", "auprc", "brier", "fold_auc_sd", "complexity_rank", "estimator"],
        ascending=[False, False, True, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    if eligible.empty:
        raise RuntimeError("No radiomics learner remained inside the pre-specified selection tolerance")
    selected_estimator = str(eligible.iloc[0]["estimator"])
    comparison["selected_fixed_best"] = comparison["estimator"].eq(selected_estimator)
    comparison = comparison.sort_values(
        ["selected_fixed_best", "within_selection_tolerance", primary_final, "auc", "auprc", "brier", "fold_auc_sd", "complexity_rank"],
        ascending=[False, False, False, False, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    selected_oof = candidate_oof[selected_estimator]
    save_oof_artifact(
        out, "radiomics", selected_oof, dev, folds,
        source=f"fixed best radiomics learner selected from complete candidate OOF: {selected_estimator}",
        extra={
            "fixed_single_learner_family": selected_estimator,
            "selection_scope": "development_outer_oof_fixed_best_radiomics",
            "oof_deployment_family_identical": True,
            "outer_holdout_used_for_inner_tuning": False,
            "global_screen_top2": global_top2,
        },
    )
    selected_metrics = _fixed_candidate_oof_metrics(
        y_dev, selected_oof, folds, float(config.threshold_min_sensitivity)
    )

    screening_runs_path = Path(reports) / "radiomics_screening_all_runs.csv"
    screening_summary_path = Path(reports) / "radiomics_screening_summary_by_outer_fold.csv"
    refinement_path = Path(reports) / "radiomics_fixed_candidate_outer_folds.json"
    comparison_path = Path(reports) / "radiomics_fixed_candidate_oof_comparison.csv"
    pd.DataFrame(screening_records).to_csv(screening_runs_path, index=False)
    screen_all.to_csv(screening_summary_path, index=False)
    Path(refinement_path).write_text(json.dumps(refinement_records, indent=2, default=str), encoding="utf-8")
    comparison.to_csv(comparison_path, index=False)

    feature_config = {
        "wavelet_levels": int(config.wavelet_levels),
        "glcm_distances": [int(x) for x in str(config.glcm_distances).split(",")],
        "glcm_angles_deg": [float(x) for x in str(config.glcm_angles).split(",")],
        "lbp_radius": int(config.lbp_radius),
        "lbp_n_points": int(config.lbp_n_points),
        "feature_dimension": int(X_dev.shape[1]), "image_size": 224,
    }
    selected_generic_path = Path(out) / "probs" / "oof_radiomics.npz"
    selection_manifest = {
        "schema": "aura_cxr_radiomics_fixed_best_q1_v17",
        "selection_scope": "development_outer_oof_fixed_best_radiomics",
        "selection_metric": str(config.mrfo_fitness),
        "selection_tolerance": float(config.radiomics_selection_tolerance),
        "selection_tolerance_policy": "retain learners within tolerance of best primary metric, then AUC/AUPRC/Brier/fold-stability/complexity tie-break",
        "procedure": "screen four learners with per-seed stability; evaluate the same global top two on all five outer folds; select one fixed learner from aggregate outer OOF",
        "candidate_estimators": candidates,
        "screening": {
            "rows": int(config.radiomics_screen_rows), "folds": int(config.radiomics_screen_folds),
            "inner_folds": int(config.radiomics_screen_inner_folds),
            "population": int(config.radiomics_screen_pop), "iterations": int(config.radiomics_screen_iter),
            "seeds": _parse_screen_seeds(config.radiomics_screen_seeds), "top_k": int(config.radiomics_screen_top_k),
            "promotion_policy": "top2_seed_frequency_desc_mean_rank_asc_pooled_metric_tiebreak",
            "global_top2": global_top2,
            "global_stability_summary_path": str(global_screen_path.resolve()),
            "global_stability_summary_sha256": sha256_file(global_screen_path),
            "purpose": "candidate_reduction_only", "outer_holdout_accessed": False,
        },
        "final_refinement": {
            "fixed_candidate_families": global_top2,
            "each_candidate_evaluated_on_all_outer_folds": True,
            "outer_folds": int(config.oof_folds), "inner_folds": int(config.radiomics_final_inner_folds),
            "population": int(config.radiomics_final_pop), "iterations": int(config.radiomics_final_iter),
            "subsample_rows": 0, "outer_holdout_used_for_tuning": False,
        },
        "candidate_oof_results": comparison.to_dict(orient="records"),
        "candidate_oof_artifacts": candidate_paths,
        "selected_estimator": selected_estimator,
        "selected_display_name": display[selected_estimator],
        "oof_model_family_policy": "single_global_fixed_best_learner",
        "deployment_family_policy": "same_fixed_best_learner_refit_on_full_development",
        "oof_deployment_family_identical": True,
        "selected_branch_oof_metrics": selected_metrics,
        "feature_config": feature_config,
        "feature_cache": str(Path(cache_path).resolve()), "feature_cache_sha256": feature_cache_sha,
        "development_patient_sha256": _stable_sha256_values(dev["patientId"].astype(str).tolist()),
        "development_label_sha256": _stable_sha256_values(dev["label"].astype(int).tolist()),
        "selected_oof_path": str(selected_generic_path.resolve()),
        "selected_oof_sha256": sha256_file(selected_generic_path),
        "selected_candidate_oof_path": candidate_paths[selected_estimator]["path"],
        "selected_candidate_oof_sha256": candidate_paths[selected_estimator]["sha256"],
        "outer_holdout_used_for_hyperparameter_tuning": False,
        "test_predictions_generated": False, "test_images_accessed": False,
        "external_kermany_used_for_selection": False, "test_set_used_for_selection": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(selection_manifest, Path(reports) / "radiomics_learner_selection_manifest.json")
    save_json({
        "selected_estimator": selected_estimator,
        "selected_display_name": display[selected_estimator],
        "selection_source": "development_outer_oof_fixed_best_radiomics",
        "oof_deployment_family_identical": True,
    }, Path(reports) / "radiomics_best_learner.json")
    print(
        f"[RADIOMICS-FIXED] selected={display[selected_estimator]} "
        f"OOF_AUC={selected_metrics['auc']:.6f} OOF_F1macro={selected_metrics['crossfitted_f1_macro']:.6f}"
    )
    return selected_estimator

def run_radiomics_train_only(train_df, val_df, config: RunConfig, out: Path,
                              models: Path, reports: Path) -> str:
    """Backward-compatible stage alias for strict nested radiomics OOF selection."""
    return run_radiomics_nested_development_oof(train_df, val_df, config, out, models, reports)

def _oof_best_epoch_count(out: Path, backbone_name: str, phase: str, monitor: str, mode: str) -> int:
    vals = []
    for path in sorted((Path(out) / "models" / "oof" / backbone_name).glob(f"fold_*/checkpoints/{backbone_name}/{phase}/training_log.csv")):
        try:
            df = pd.read_csv(path)
            if df.empty or monitor not in df.columns:
                continue
            s = pd.to_numeric(df[monitor], errors="coerce")
            idx = s.idxmin() if mode == "min" else s.idxmax()
            vals.append(int(idx) + 1)
        except Exception:
            continue
    if not vals:
        raise FileNotFoundError(f"No OOF training histories found for {backbone_name}/{phase}/{monitor}")
    return max(1, int(round(float(np.median(vals)))))


def train_tf_development_refit(backbone_name: str, dev_df: pd.DataFrame, config: RunConfig,
                               out: Path, models: Path, reports: Path) -> Path:
    """Refit a selected Keras backbone on all development rows for OOF-derived fixed epochs."""
    final_path = Path(models) / f"{backbone_name}_development_refit.keras"
    if final_path.exists():
        return final_path
    mode = checkpoint_mode_for_monitor(config.checkpoint_monitor)
    p1 = _oof_best_epoch_count(out, backbone_name, "phase1_initial", config.checkpoint_monitor, mode)
    p2 = _oof_best_epoch_count(out, backbone_name, "phase2_finetune", config.checkpoint_monitor, mode)
    train_ds = make_tf_dataset(
        dev_df, config.image_size, config.batch_size, training=True, seed=config.seed,
        use_efficientnet=use_efficientnet_for_backbone(backbone_name),
        use_mixup=config.use_mixup, mixup_alpha=config.mixup_alpha,
    )
    y = dev_df["label"].to_numpy(dtype=int); cw = compute_class_weights(y)
    model = build_cnn_model(
        config.image_size, config.learning_rate, config.dropout, config.l2_strength,
        backbone_name=backbone_name, use_focal_loss=config.use_focal_loss,
        focal_gamma=config.focal_gamma,
    )
    log_dir = ensure_dir(Path(out) / "reports" / "development_refit_logs")
    model.fit(train_ds, epochs=p1, class_weight=cw,
              callbacks=[tf.keras.callbacks.CSVLogger(str(log_dir / f"{backbone_name}_phase1.csv"))], verbose=1)
    finetune_audit = configure_backbone_finetuning(
        model, backbone_name, config.unfreeze_last_n
    )
    model = _compile_cnn_model_for_phase(model, config.learning_rate * 0.1,
                                         use_focal_loss=config.use_focal_loss,
                                         focal_gamma=config.focal_gamma)
    model.fit(train_ds, epochs=p2, class_weight=cw,
              callbacks=[tf.keras.callbacks.CSVLogger(str(log_dir / f"{backbone_name}_phase2.csv"))], verbose=1)
    model.save(str(final_path))
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17", "model": backbone_name,
        "training_scope": "full_development_train_plus_validation",
        "epoch_policy": "median_best_epoch_from_outer_OOF_inner_validation",
        "phase1_epochs": int(p1), "phase2_epochs": int(p2),
        "fine_tuning_audit": finetune_audit,
        "parameter_count": int(model.count_params()), "image_size": int(config.image_size),
        "test_images_accessed": False, "test_labels_used": False,
        "model_path": str(final_path), "model_sha256": sha256_file(final_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(reports) / f"{backbone_name}_development_refit_manifest.json")
    tf.keras.backend.clear_session()
    return final_path


def run_selected_development_refit(train_df, val_df, config: RunConfig, out: Path,
                                    models: Path, reports: Path) -> Dict:
    """Refit the locked radiomics family and selected DL pair on all development rows.

    The radiomics learner family is already locked by nested outer OOF. Its final
    hyperparameters are optimized again using the configured inner CV on the complete
    development set, which is valid because the locked test remains unopened.
    """
    manifest = load_pair_manifest(out, strict=True)
    if manifest.get("final_test_evaluated_once"):
        raise RuntimeError("Cannot refit after the locked test has been opened")
    dev = ensure_development_table(out, train_df, val_df)
    cache_path = _resolve_radiomic_cache_for_oof(out, config)
    cache = joblib.load(cache_path)
    X_dev = features_from_dataframe(dev, cache)
    y_dev = dev["label"].to_numpy(dtype=int)
    selection = load_radiomics_selection_manifest(out, strict=True)
    selected_estimator = str(selection["selected_estimator"])
    rad_path = Path(models) / "radiomics_mrfo_development_refit.pkl"
    rad_info_path = Path(reports) / "radiomics_development_refit_mrfo.json"
    selection_path = Path(out) / "reports" / "radiomics_learner_selection_manifest.json"
    expected = {
        "schema": "aura_cxr_radiomics_development_refit_q1_v17",
        "selected_estimator": selected_estimator,
        "selection_manifest_sha256": sha256_file(selection_path),
        "development_patient_sha256": _stable_sha256_values(dev["patientId"].astype(str).tolist()),
        "development_label_sha256": _stable_sha256_values(dev["label"].astype(int).tolist()),
        "feature_cache_sha256": sha256_file(cache_path),
        "feature_dimension": int(X_dev.shape[1]),
        "image_size": int(config.image_size),
        "fitness_metric": str(config.mrfo_fitness),
        "pop_size": int(config.radiomics_final_pop),
        "max_iter": int(config.radiomics_final_iter),
        "inner_cv_folds": int(config.radiomics_final_inner_folds),
        "mrfo_subsample": 0,
        "wavelet_levels": int(config.wavelet_levels),
        "glcm_distances": str(config.glcm_distances),
        "glcm_angles": str(config.glcm_angles),
        "lbp_radius": int(config.lbp_radius),
        "lbp_n_points": int(config.lbp_n_points),
        "oof_deployment_family_identical": True,
        "seed": int(config.seed),
        "test_set_used": False,
        "external_kermany_used": False,
    }
    if rad_path.exists() and rad_info_path.exists() and not config.force:
        payload = json.loads(rad_info_path.read_text(encoding="utf-8"))
        mismatches = {k: (payload.get(k), v) for k, v in expected.items() if payload.get(k) != v}
        if payload.get("model_sha256") != sha256_file(rad_path):
            mismatches["model_sha256"] = (payload.get("model_sha256"), sha256_file(rad_path))
        if mismatches:
            raise RuntimeError(
                f"Stale radiomics development-refit cache: {mismatches}. "
                "Use a new OUT directory or --force before locked-test inference."
            )
        rad = joblib.load(rad_path)
        assert_radiomics_estimator_family(rad, selected_estimator)
        mrfo_info = payload
        print(f"[RADIOMICS-REFIT] Reusing development refit: {selected_estimator}")
    else:
        cfg = copy.deepcopy(config)
        cfg.mrfo_estimator = selected_estimator
        cfg.mrfo_pop_size = int(config.radiomics_final_pop)
        cfg.mrfo_max_iter = int(config.radiomics_final_iter)
        cfg.mrfo_cv_folds = int(config.radiomics_final_inner_folds)
        cfg.mrfo_subsample = 0
        cfg.seed = int(config.seed)
        rad, mrfo_info = train_radiomics_mrfo(X_dev, y_dev, cfg)
        assert_radiomics_estimator_family(rad, selected_estimator)
        joblib.dump(rad, rad_path)
        mrfo_info = {
            **mrfo_info,
            **expected,
            "best_cv_score": float(mrfo_info["best_cv_score"]),
            "best_params": mrfo_info.get("best_params", {}),
            "model_identity": _radiomics_estimator_identity(rad),
            "model_path": str(rad_path.resolve()),
            "model_sha256": sha256_file(rad_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(mrfo_info, rad_info_path)
    identity = assert_radiomics_estimator_family(rad, selected_estimator)
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "model": "radiomics",
        "selected_estimator": selected_estimator,
        "selected_display_name": selection["selected_display_name"],
        "selected_model_identity": identity,
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256_file(selection_path),
        "learner_family_selection_scope": "development_outer_oof_fixed_best_radiomics",
        "oof_deployment_family_identical": True,
        "final_hyperparameter_scope": f"full_development_inner_{int(config.radiomics_final_inner_folds)}fold_cv_after_family_lock",
        "feature_config": selection.get("feature_config", {}),
        "best_params": mrfo_info.get("best_params", {}),
        "best_cv_score": float(mrfo_info["best_cv_score"]),
        "training_scope": "full_development_train_plus_validation",
        "parameter_count": None,
        "test_images_accessed": False,
        "test_labels_used": False,
        "model_path": str(rad_path.resolve()),
        "model_sha256": sha256_file(rad_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(reports) / "radiomics_development_refit_manifest.json")

    generated, pending = ["radiomics"], []
    for name in manifest["selected_dl_models"]:
        if name in {"efficientnetv2", "resnet50"}:
            train_tf_development_refit(name, dev, config, out, models, reports)
            generated.append(name)
        else:
            pending.append(name)
    deployment = create_or_refresh_deployment_manifest(out, require_complete=False)
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "selected_dl_models": manifest["selected_dl_models"],
        "selected_radiomics_estimator": selected_estimator,
        "generated_here": generated,
        "pending_pytorch_refits": pending,
        "test_images_accessed": False,
        "deployment_status": deployment["deployment_status"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(reports) / "selected_development_refit_audit.json")
    return deployment


def run_radiomics_development_oof(train_df, val_df, config: RunConfig,
                                   out: Path, models: Path) -> None:
    """Compatibility guard: fixed-best radiomics OOF must already exist."""
    selection = load_radiomics_selection_manifest(out, strict=True)
    dev = ensure_development_table(out, train_df, val_df)
    generic_path = Path(out) / "probs" / "oof_radiomics.npz"
    selected_estimator = str(selection["selected_estimator"])
    selected_path = Path(out) / "probs" / f"oof_radiomics_{selected_estimator}.npz"
    if not generic_path.exists() or sha256_file(generic_path) != selection.get("selected_oof_sha256"):
        raise FileNotFoundError(
            "Fixed-best radiomics OOF is missing or changed. Run --stage radiomics_nested_oof first."
        )
    if not selected_path.exists() or sha256_file(selected_path) != selection.get("selected_candidate_oof_sha256"):
        raise FileNotFoundError(f"Selected candidate OOF is missing or changed: {selected_path}")
    generic, generic_fold = load_oof_artifact(out, "radiomics", dev)
    selected, selected_fold = load_oof_artifact(out, f"radiomics_{selected_estimator}", dev)
    if not np.array_equal(generic_fold, selected_fold) or not np.allclose(generic, selected, rtol=0.0, atol=1e-12):
        raise RuntimeError("Generic radiomics OOF is not identical to the fixed selected learner OOF")
    print(f"[OOF][radiomics] verified fixed learner={selected_estimator} n={len(dev)}")

def run_tf_backbone_development_oof(backbone_name: str, train_df, val_df,
                                    config: RunConfig, out: Path) -> None:
    """Train fold-specific EfficientNetV2/ResNet50 and save aligned OOF probabilities.

    Locked-test images and labels are not passed to this function. Only development
    rows are available during OOF generation, so the test set cannot affect selection.
    """
    import copy
    dev = ensure_development_table(out, train_df, val_df)
    folds = make_patient_oof_folds(
        dev, n_splits=int(getattr(config, "oof_folds", 5)), seed=config.seed,
        force=bool(getattr(config, "pair_selection_force", False)), results_dir=out,
    )
    y = dev["label"].to_numpy(dtype=int)
    oof = np.full(len(dev), np.nan, dtype=np.float64)
    base_dir = ensure_dir(Path(out) / "models" / "oof" / backbone_name)
    inner_fraction = float(getattr(config, "oof_inner_val_fraction", 0.10))
    if not (0.02 <= inner_fraction <= 0.30):
        raise ValueError("--oof_inner_val_fraction must be in [0.02,0.30].")
    for fold in sorted(np.unique(folds)):
        outer_train_idx = np.where(folds != fold)[0]
        hold_idx = np.where(folds == fold)[0]
        outer_train = dev.iloc[outer_train_idx].reset_index(drop=True)
        hold_df = dev.iloc[hold_idx].reset_index(drop=True)

        # Strict nested protocol: the outer OOF holdout is NEVER used for early
        # stopping, checkpoint choice, LR scheduling, or any hyperparameter.
        inner_train_idx, inner_val_idx = train_test_split(
            np.arange(len(outer_train)), test_size=inner_fraction,
            stratify=outer_train["label"].to_numpy(dtype=int),
            random_state=int(config.seed) + int(fold),
        )
        inner_train = outer_train.iloc[inner_train_idx].reset_index(drop=True)
        inner_val = outer_train.iloc[inner_val_idx].reset_index(drop=True)
        fold_models = ensure_dir(base_dir / f"fold_{int(fold)}")
        cfg = copy.deepcopy(config)
        cfg.force = bool(getattr(config, "pair_selection_force", False))
        cfg.auto_resume_cnn = True
        model = train_cnn_backbone(
            backbone_name, inner_train, inner_val,
            inner_train["label"].to_numpy(dtype=int), cfg, fold_models
        )
        pp = get_preprocess_for_backbone(backbone_name)
        fold_model_path = fold_models / f"{backbone_name}_final.keras"
        held_mean, _, _ = cnn_mc_predict_batch(
            model, hold_df, cfg.image_size, cfg.batch_size, pp,
            n_mc=cfg.mc_dropout_n, output_dir=fold_models,
            cache_prefix=f"{backbone_name}_fold{fold}_outer_hold", resume=True, force=cfg.force,
            model_artifact_sha256=sha256_file(fold_model_path), preprocessing_id=backbone_name,
        )
        oof[hold_idx] = normalize_binary_probability(held_mean, len(hold_idx), f"{backbone_name} fold {fold}")
        print(
            f"[OOF][{backbone_name}] fold={fold} outer_hold={len(hold_idx)} "
            f"inner_train={len(inner_train)} inner_val={len(inner_val)} complete"
        )
        tf.keras.backend.clear_session()
    if not np.all(np.isfinite(oof)):
        raise RuntimeError(f"{backbone_name} OOF has missing rows")
    save_oof_artifact(out, backbone_name, oof, dev, folds,
                      source=f"five patient-level fold models ({backbone_name})")
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "model": backbone_name,
        "development_oof": True,
        "locked_test_prediction": "deferred_until_pair_lock_and_development_refit",
        "locked_test_images_accessed": False,
        "test_labels_read": False,
        "outer_holdout_used_for_early_stopping": False,
        "inner_validation_fraction": float(inner_fraction),
        "image_size": int(config.image_size),
        "parameter_count": int(model.count_params()) if 'model' in locals() else None,
        "candidate_training_budget": {"max_phase1_epochs": int(config.epochs), "max_phase2_epochs": int(config.finetune_epochs), "early_stopping_patience": int(config.early_stopping_patience), "mc_passes": int(config.mc_dropout_n)},
    }, Path(out) / "reports" / f"{backbone_name}_oof_protocol.json")


def run_development_oof_stages(train_df, val_df, config: RunConfig,
                               out: Path, models: Path) -> None:
    if int(config.image_size) != 224:
        raise ValueError("Q1 four-candidate OOF protocol requires --image_size 224 consistently.")
    names = [x.strip() for x in str(getattr(config, "oof_backbones", "efficientnetv2,resnet50")).split(",") if x.strip()]
    unknown = sorted(set(names) - {"efficientnetv2", "resnet50"})
    if unknown:
        raise ValueError(f"train_aura handles TF OOF only for efficientnetv2,resnet50; unsupported: {unknown}")
    # Radiomics nested OOF is generated and selected in the dedicated
    # radiomics_nested_oof stage. This stage trains only the TF candidates.
    run_radiomics_development_oof(train_df, val_df, config, out, models)
    for name in names:
        run_tf_backbone_development_oof(name, train_df, val_df, config, out)





def _tf_exploratory_fold_ensemble_test(name: str, test_infer: pd.DataFrame,
                                       config: RunConfig, out: Path) -> Path:
    """Generate post-lock five-fold TF probabilities for exploratory A9/A10."""
    fold_paths = sorted((Path(out) / "models" / "oof" / name).glob(f"fold_*/{name}_final.keras"))
    if not fold_paths:
        raise FileNotFoundError(f"Missing {name} OOF fold models for exploratory A9/A10")
    pp = get_preprocess_for_backbone(name)
    draws = []
    for fold_idx, model_path in enumerate(fold_paths):
        model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        mean, _, _ = cnn_mc_predict_batch(
            model, test_infer, config.image_size, config.batch_size, pp,
            n_mc=config.mc_dropout_n,
            output_dir=Path(out) / "exploratory_test_inference" / name / f"fold_{fold_idx}",
            cache_prefix=f"{name}_exploratory_fold{fold_idx}", resume=True, force=False,
            model_artifact_sha256=sha256_file(model_path), preprocessing_id=name,
        )
        draws.append(normalize_binary_probability(mean, len(test_infer), f"{name} exploratory fold {fold_idx}"))
        del model
        tf.keras.backend.clear_session()
    arr = np.stack(draws, axis=0)
    path = Path(out) / "probs" / f"{name}_exploratory_cvensemble_test.npy"
    np.save(path, two_col(arr.mean(axis=0)))
    save_json({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "model": name,
        "role": "exploratory_nonselected_models",
        "method": "five_outer_fold_checkpoint_ensemble",
        "generated_after_pair_lock": True,
        "test_labels_used": False,
        "test_set_used_for_selection": False,
        "fold_model_paths": [str(x) for x in fold_paths],
        "probability_path": str(path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, Path(out) / "reports" / f"{name}_exploratory_test_manifest.json")
    return path


def run_selected_locked_test_inference(train_df, val_df, test_df, config: RunConfig,
                                       out: Path, models: Path) -> Dict:
    """Infer exact selected refits and optional post-lock all-four exploratory models."""
    pair = load_pair_manifest(out, strict=True)
    load_deployment_manifest(out, strict=True)
    if pair.get("final_test_evaluated_once"):
        raise RuntimeError("Locked test has already been evaluated; use a new results directory")
    selected = list(pair["selected_dl_models"])
    test_infer = test_df.copy(); test_infer["label"] = 0
    probs_dir = ensure_dir(Path(out) / "probs"); sources = {}

    # Test radiomics features are created here for the first time, after pair/deployment lock.
    test_cache_path = _build_locked_test_feature_cache(test_infer, config, out)
    test_cache = joblib.load(test_cache_path); X_test = features_from_dataframe(test_infer, test_cache)
    rad_path = resolve_locked_model_artifact(out, "radiomics"); rad = joblib.load(rad_path)
    locked_rad_estimator = str(load_radiomics_selection_manifest(out, strict=True)["selected_estimator"])
    assert_radiomics_estimator_family(rad, locked_rad_estimator)
    rad_prob = normalize_binary_probability(rad.predict_proba(X_test), len(test_df), "radiomics development refit test")
    rad_prob_path = probs_dir / "radiomics_development_refit_test.npy"; np.save(rad_prob_path, two_col(rad_prob))
    register_probability_artifact(out, "radiomics", "test", rad_prob_path, rad_path)
    sources["radiomics"] = str(rad_path)

    # Exact deployment probabilities for selected Keras members.
    for name in [m for m in selected if m in {"efficientnetv2", "resnet50"}]:
        model_path = resolve_locked_model_artifact(out, name)
        model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
        pp = get_preprocess_for_backbone(name)
        mean, std, predictive_width = cnn_mc_predict_batch(
            model, test_infer, config.image_size, config.batch_size, pp,
            n_mc=config.mc_dropout_n, output_dir=Path(out) / "locked_test_inference" / name,
            cache_prefix=f"{name}_development_refit_locked_test", resume=True, force=False,
            model_artifact_sha256=sha256_file(model_path), preprocessing_id=name,
        )
        prob_path = probs_dir / f"{name}_development_refit_test_mean.npy"
        std_path = probs_dir / f"{name}_development_refit_test_std.npy"
        width_path = probs_dir / f"{name}_development_refit_test_predictive_interval_width_95.npy"
        np.save(prob_path, mean.astype(np.float32)); np.save(std_path, std.astype(np.float32))
        np.save(width_path, predictive_width.astype(np.float32))
        register_probability_artifact(out, name, "test", prob_path, model_path)
        sources[name] = str(model_path)
        del model; tf.keras.backend.clear_session()

    exploratory_paths = {}
    if bool(getattr(config, "exploratory_all4_test", False)):
        for name in ("efficientnetv2", "resnet50"):
            exploratory_paths[name] = str(_tf_exploratory_fold_ensemble_test(name, test_infer, config, out))

    pending = [m for m in selected if m in {"xrv", "eva_x"}]
    audit = {
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "selection_lock_id": pair["selection_lock_id"],
        "selected_dl_models": selected,
        "generated_here": sorted(sources),
        "pending_pytorch_selected_models": pending,
        "exploratory_all_four_test": bool(getattr(config, "exploratory_all4_test", False)),
        "exploratory_tf_artifacts": exploratory_paths,
        "test_inference_started_after_pair_lock": True,
        "development_refit_models_only_for_final_method": True,
        "test_labels_used_for_inference": False,
        "test_set_used_for_selection": False,
        "probability_sources": sources,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(audit, Path(out) / "reports" / "selected_pair_locked_test_inference_audit.json")
    print(f"[LOCKED-TEST] Generated exact refit probabilities. Pending PyTorch: {pending}")
    return audit

def run_q1_pair_selection(out: Path, config: RunConfig) -> Dict:
    manifest = select_dl_pair_and_train_meta(
        out,
        candidates=["efficientnetv2", "resnet50", "xrv", "eva_x"],
        n_splits=int(getattr(config, "oof_folds", 5)),
        seed=config.seed,
        auc_tolerance=float(getattr(config, "pair_auc_tolerance", 0.002)),
        min_sensitivity=float(config.threshold_min_sensitivity),
        force=bool(getattr(config, "pair_selection_force", False)),
    )
    selected_pair_markdown(out)
    print(f"[PAIR-SELECT] Locked: {manifest['selected_pair_display']}")
    return manifest

# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AURA-CXR proposal-aligned pipeline")
    parser.add_argument("--labels_csv",      required=False, default="")
    parser.add_argument("--dicom_dir",       required=False, default="")
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--image_size",      type=int,   default=224,
                        help="Global Q1 image size. Must remain 224 for CNN, XRV, EVA-X, XAI, ablation, and external validation.")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--epochs",          type=int,   default=40,
                        help="Phase-1 maximum epochs. With --finetune_epochs 20, total maximum is 60.")
    parser.add_argument("--subset_rows",     type=int,   default=None)
    parser.add_argument("--test_fraction",   type=float, default=0.20)
    parser.add_argument("--val_fraction",    type=float, default=0.10)
    parser.add_argument("--stage", default="all",
                        help="Strict stages: splits,development_features,radiomics_nested_oof,development_oof,pair_select,selected_refit,selected_test_infer,fusion,ablation. radiomics_train_only is a backward alias.")
    parser.add_argument("--force", action="store_true", help="Recompute existing stage artifacts")
    parser.add_argument("--merge", action="store_true", help="Merge chunk outputs where supported")
    parser.add_argument("--finalize", action="store_true", help="Finalize ablation tables/plots")
    parser.add_argument("--chunk_id", type=int, default=None)
    parser.add_argument("--n_chunks", type=int, default=None)
    parser.add_argument("--chunk_start", type=int, default=None)
    parser.add_argument("--chunk_end", type=int, default=None)
    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--ablation_only", action="store_true")
    parser.add_argument("--ablation_scenarios", default="", help="Comma-separated: 1,2,3,7,F1,F2,...")
    parser.add_argument("--mrfo_pop",        type=int,   default=15)
    parser.add_argument("--mrfo_iter",       type=int,   default=15)
    parser.add_argument("--mc_n",            type=int,   default=30)
    parser.add_argument("--decision_threshold", type=float, default=0.40,
                        help="Fallback threshold if --no_threshold_tuning is used; final paper mode tunes threshold on validation only.")
    parser.add_argument("--checkpoint_monitor", choices=["val_f1", "val_auc", "val_loss"], default="val_auc",
                        help="Validation metric for EarlyStopping and best CNN checkpoint. Default val_auc (lebih stabil terhadap threshold).")
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--reduce_lr_patience", type=int, default=5)
    parser.add_argument("--no_threshold_tuning", action="store_true",
                        help="Disable validation-set threshold tuning and use --decision_threshold directly.")
    parser.add_argument("--threshold_metric", choices=["f1", "f1_weighted", "f1_pneumonia", "balanced_accuracy", "sensitivity", "youden"], default="youden",
                        help="Metrik pemilihan threshold. Default 'youden' (bukan f1_weighted yang menekan sensitivitas pada data imbalanced).")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_steps", type=int, default=181)
    parser.add_argument("--threshold_min_sensitivity", type=float, default=0.85,
                        help="Batas bawah sensitivitas validasi saat memilih threshold (default 0.85 untuk skrining pneumonia). Set 0.0 untuk menonaktifkan.")
    parser.add_argument("--unfreeze_last_n", type=int, default=120,
                        help="Jumlah layer teratas backbone yang dibuka saat fine-tune (naik dari 30).")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Epoch fine-tune phase-2 (naik dari 20).")
    parser.add_argument("--no_focal_loss", action="store_true",
                        help="Nonaktifkan focal loss dan kembali ke CategoricalCrossentropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma untuk focal loss (default 2.0).")
    parser.add_argument("--soft_vote_auc_floor", type=float, default=0.78,
                        help="Legacy fixed-member option only; not used by the strict selected-pair pipeline.")
    parser.add_argument("--mrfo_estimator", choices=["auto", "best", "histgb", "lgbm", "svm", "knn"], default="auto",
                        help="Use auto/best for final Q1: equal-budget benchmark of KNN, HistGB, LightGBM, and RBF-SVM.")
    parser.add_argument("--radiomics_candidates", default="knn,histgb,lgbm,svm",
                        help="Must contain exactly knn,histgb,lgbm,svm in strict final mode.")
    parser.add_argument("--radiomics_selection_tolerance", type=float, default=0.002,
                        help="Inner-score tolerance used only for deterministic tie handling.")
    parser.add_argument("--radiomics_screen_rows", type=int, default=4000)
    parser.add_argument("--radiomics_screen_folds", type=int, default=3)
    parser.add_argument("--radiomics_screen_inner_folds", type=int, default=2)
    parser.add_argument("--radiomics_screen_pop", type=int, default=8)
    parser.add_argument("--radiomics_screen_iter", type=int, default=8)
    parser.add_argument("--radiomics_screen_seeds", default="42,123")
    parser.add_argument("--radiomics_screen_top_k", type=int, default=2)
    parser.add_argument("--radiomics_final_inner_folds", type=int, default=3)
    parser.add_argument("--radiomics_final_pop", type=int, default=15)
    parser.add_argument("--radiomics_final_iter", type=int, default=15)
    parser.add_argument("--mrfo_fitness", choices=["f1_macro", "roc_auc"], default="f1_macro",
                        help="Metrik fitness MRFO. f1_macro=balanced-F1 (default), atau roc_auc.")
    parser.add_argument("--mrfo_subsample", type=int, default=0,
                        help="Subsampel stratified untuk CV pada estimator mahal (svm). 0 = pakai semua.")
    parser.add_argument("--threshold_min_specificity", type=float, default=0.0,
                        help="Optional validation specificity constraint for threshold selection.")
    parser.add_argument("--uncertainty_threshold", type=float, default=None,
                        help="MC Dropout predictive-interval-width threshold. Default learns the review-rate quantile from development validation.")
    parser.add_argument("--uncertainty_review_rate", type=float, default=0.30,
                        help="If threshold is None, flag the top fraction of most uncertain validation cases.")
    parser.add_argument("--no_resnet50", action="store_true",
                        help="Legacy fixed-member option only. The strict pipeline evaluates four DL candidates and locks two by development OOF.")
    parser.add_argument("--no_efficientnet", action="store_true")
    parser.add_argument("--no_auto_resume_cnn", action="store_true",
                        help="Disable CNN BackupAndRestore auto-resume. Default: enabled.")
    parser.add_argument("--save_epoch_checkpoints", action="store_true",
                        help="Also save full .keras checkpoint every epoch. Uses more disk space.")
    # ── Q1 four-candidate development-OOF selection ───────────────────────
    parser.add_argument("--oof_folds", type=int, default=5,
                        help="Patient-level development OOF folds for four-candidate pair selection.")
    parser.add_argument("--oof_backbones", default="efficientnetv2,resnet50",
                        help="TF backbones generated by train_aura during development_oof. XRV/EVA-X use their own scripts.")
    parser.add_argument("--oof_inner_val_fraction", type=float, default=0.10,
                        help="Inner validation fraction inside each outer OOF training fold. Outer holdout is never used for early stopping.")
    parser.add_argument("--pair_auc_tolerance", type=float, default=0.002,
                        help="Pairs within this OOF AUC delta use AP, Brier, ECE, fold stability, then complexity tie-breakers.")
    parser.add_argument("--pair_selection_force", action="store_true",
                        help="Rebuild OOF folds/pair manifest. Do not use after locked-test results have been inspected.")
    parser.add_argument("--exploratory_all4_test", action="store_true",
                        help="After pair lock only: also generate non-selected DL locked-test probabilities for exploratory A9/A10 ablation.")
    parser.add_argument("--strict_selected_pair_fusion", action="store_true",
                        help="Require OOF-selected pair and fold-ensemble/refit test probabilities for fusion.")
    parser.add_argument("--allow_legacy_all", action="store_true", help="Explicitly allow the old fixed-member --stage all path. Not Q1 pair-selection protocol.")
    # ── GPU flags ──────────────────────────────────────────────────────────
    parser.add_argument("--gpu40",           action="store_true",
                        help="Aktifkan optimasi GPU 40GB: mixed precision + XLA + memory growth")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Aktifkan mixed float16 precision (subset dari --gpu40)")
    parser.add_argument("--gpu_id",          type=int, default=0,
                        help="GPU device index yang digunakan (default: 0)")
    args = parser.parse_args()

    print(f"[ENV] Python/TF/NumPy/Pandas: {os.sys.version.split()[0]} / {tf.__version__} / {np.__version__} / {pd.__version__}")
    if tuple(map(int, os.sys.version.split()[0].split('.')[:2])) >= (3, 13):
        print("[ENV] Python 3.13 mode: gunakan tensorflow>=2.20 dan NumPy>=2.x; jangan pin tensorflow==2.15.")

    # ── GPU Setup (HARUS sebelum TF model build) ──────────────────────────
    if args.gpu40 or args.mixed_precision:
        # Pilih GPU tertentu jika diminta
        if args.gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            print(f"[GPU] CUDA_VISIBLE_DEVICES = {args.gpu_id}")

        # Mixed Precision (fp16) — 2-3x lebih cepat di A100/A6000
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("[GPU] Mixed Precision: float16 aktif")

        # Memory growth — hindari TF monopoli seluruh VRAM
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"[GPU] GPU terdeteksi: {len(gpus)} device(s)")

        # XLA JIT compilation
        tf.config.optimizer.set_jit(True)
        print("[GPU] XLA JIT: aktif")

        if args.gpu40 and args.batch_size == 32:
            print("[GPU] TIP: Gunakan --batch_size 128 untuk GPU 40GB agar lebih cepat")

    else:
        # CPU/default mode — tetap set memory growth untuk keamanan
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            print(f"[INFO] GPU terdeteksi ({len(gpus)} device). "
                  f"Tambahkan --gpu40 untuk optimasi penuh GPU 40GB.")

    config = RunConfig(
        labels_csv=args.labels_csv,
        dicom_dir=args.dicom_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        subset_rows=args.subset_rows,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        mrfo_pop_size=args.mrfo_pop,
        mrfo_max_iter=args.mrfo_iter,
        mc_dropout_n=args.mc_n,
        uncertainty_threshold=args.uncertainty_threshold,
        uncertainty_review_rate=args.uncertainty_review_rate,
        decision_threshold=args.decision_threshold,
        train_efficientnetv2=not args.no_efficientnet,
        train_resnet50=not args.no_resnet50,
        use_efficientnet=not args.no_efficientnet,
        auto_resume_cnn=not args.no_auto_resume_cnn,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        checkpoint_monitor=args.checkpoint_monitor,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        tune_threshold=not args.no_threshold_tuning,
        threshold_metric=args.threshold_metric,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
        threshold_min_sensitivity=args.threshold_min_sensitivity,
        threshold_min_specificity=args.threshold_min_specificity,
        unfreeze_last_n=args.unfreeze_last_n,
        finetune_epochs=args.finetune_epochs,
        use_focal_loss=not args.no_focal_loss,
        focal_gamma=args.focal_gamma,
        soft_vote_auc_floor=args.soft_vote_auc_floor,
        mrfo_estimator=args.mrfo_estimator,
        radiomics_candidates=args.radiomics_candidates,
        radiomics_selection_tolerance=args.radiomics_selection_tolerance,
        radiomics_screen_rows=args.radiomics_screen_rows,
        radiomics_screen_folds=args.radiomics_screen_folds,
        radiomics_screen_inner_folds=args.radiomics_screen_inner_folds,
        radiomics_screen_pop=args.radiomics_screen_pop,
        radiomics_screen_iter=args.radiomics_screen_iter,
        radiomics_screen_seeds=args.radiomics_screen_seeds,
        radiomics_screen_top_k=args.radiomics_screen_top_k,
        radiomics_final_inner_folds=args.radiomics_final_inner_folds,
        radiomics_final_pop=args.radiomics_final_pop,
        radiomics_final_iter=args.radiomics_final_iter,
        mrfo_cv_folds=args.radiomics_final_inner_folds,
        mrfo_fitness=args.mrfo_fitness,
        mrfo_subsample=args.mrfo_subsample,
        stage=args.stage,
        force=args.force,
        chunk_id=args.chunk_id,
        n_chunks=args.n_chunks,
        chunk_start=args.chunk_start,
        chunk_end=args.chunk_end,
        merge=args.merge,
        finalize=args.finalize,
        run_ablation=args.run_ablation,
        ablation_only=args.ablation_only,
        ablation_scenarios=args.ablation_scenarios,
    )

    config.oof_folds = int(args.oof_folds)
    config.oof_backbones = str(args.oof_backbones)
    config.oof_inner_val_fraction = float(args.oof_inner_val_fraction)
    config.pair_auc_tolerance = float(args.pair_auc_tolerance)
    config.pair_selection_force = bool(args.pair_selection_force)
    config.strict_selected_pair_fusion = bool(args.strict_selected_pair_fusion)
    config.exploratory_all4_test = bool(args.exploratory_all4_test)
    if int(config.image_size) != 224:
        print("[WARN] Global Q1 protocol is fixed at 224x224. Non-224 runs are exploratory and cannot share OOF/pair artifacts.")

    seed_everything(config.seed)

    # ── Output dirs ──────────────────────────────────────────────────────────
    out      = ensure_dir(config.output_dir)
    plots    = ensure_dir(out / "plots")
    xai      = ensure_dir(out / "xai")
    models   = ensure_dir(out / "models")
    reports  = ensure_dir(out / "reports")
    splits   = ensure_dir(out / "splits")
    novelty  = ensure_dir(out / "novelty_outputs")

    print("=" * 70)
    print("  AURA-CXR — Proposal-Aligned Pipeline")
    print("=" * 70)

    # ── Data Loading / Stage S0: patient-level splits ───────────────────────
    requested_stages = {s.strip().lower() for s in config.stage.split(",") if s.strip()}
    # Stage name is now generic because the MRFO learner can be HistGB, LightGBM,
    # SVM, or kNN. Keep legacy aliases so old commands remain reproducible.
    legacy_radiomics_aliases = {"knn", "mrfo", "wg_knn"}
    if requested_stages & legacy_radiomics_aliases:
        used = sorted(requested_stages & legacy_radiomics_aliases)
        print(f"[WARN] Legacy stage alias {used} mapped to --stage radiomics.")
        requested_stages = (requested_stages - legacy_radiomics_aliases) | {"radiomics"}
    if "all" in requested_stages:
        requested_stages = {"all"}
        if not args.allow_legacy_all:
            raise ValueError(
                "The final pipeline intentionally blocks legacy --stage all because XRV/EVA-X OOF are separate PyTorch stages. "
                "Use run_aura_cxr.sh, or pass --allow_legacy_all only for archived fixed-member experiments."
            )
    # For single-backbone stages, train only the requested CNN to avoid occupying the one A100 twice.
    if requested_stages in [{"effnet"}, {"efficientnet"}, {"efficientnetv2"}]:
        config.train_efficientnetv2 = True
        config.train_resnet50 = False
    if requested_stages in [{"resnet"}, {"resnet50"}]:
        config.train_efficientnetv2 = False
        config.train_resnet50 = True
    print("[INFO] Loading/creating patient-level splits...")
    # Before the pair is locked, strict stages read ONLY train.csv and val.csv.
    # test.csv (including its labels and image paths) is deliberately not opened.
    prelock_stages = {
        "development_features", "features_dev", "radiomics_nested_oof", "nested_radiomics",
        "radiomics_train_only", "radiomics", "development_oof", "dl_oof", "oof",
        "pair_select", "dl_pair_selection",
        "selected_refit", "development_refit",
    }
    strict_prelock = bool(requested_stages and requested_stages.issubset(prelock_stages))
    label_free_test_infer = requested_stages in [{"selected_test_infer"}, {"locked_test_infer"}]
    label_free_fusion = requested_stages == {"fusion"} and (
        bool(getattr(config, "strict_selected_pair_fusion", False)) or (reports / "dl_pair_selection_manifest.json").exists()
    )
    force_splits = bool(config.force and ("all" in requested_stages or "splits" in requested_stages))
    if strict_prelock or label_free_test_infer or label_free_fusion:
        train_path, val_path = splits / "train.csv", splits / "val.csv"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError("Run --stage splits first; strict stages require existing train.csv and val.csv")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        overlap = set(train_df["patientId"].astype(str)) & set(val_df["patientId"].astype(str))
        if overlap:
            raise AssertionError(f"Patient leakage between train and validation: {sorted(overlap)[:10]}")
        if label_free_test_infer:
            test_path = splits / "test.csv"
            if not test_path.exists(): raise FileNotFoundError(test_path)
            test_columns = [c for c in pd.read_csv(test_path, nrows=0).columns if c != "label"]
            test_df = pd.read_csv(test_path, usecols=test_columns)
            test_df["label"] = 0  # inference-only placeholder; locked-test labels were not loaded
            print(f"[SPLIT] Locked-test inference load: n={len(test_df)}; label column NOT READ")
        else:
            test_df = pd.DataFrame(columns=train_df.columns)
            mode = "pre-lock" if strict_prelock else "label-free fusion"
            print(f"[SPLIT] Strict {mode} load: train={len(train_df)}, val={len(val_df)}; test.csv NOT READ")
    else:
        train_df, val_df, test_df = load_or_create_splits(out, config, force=force_splits)
    df = pd.concat([train_df, val_df] + ([] if test_df.empty else [test_df]), ignore_index=True)
    y_train = train_df["label"].values
    y_val   = val_df["label"].values
    y_test  = test_df["label"].values if (not test_df.empty and not label_free_test_infer) else np.asarray([], dtype=int)
    update_run_manifest(out, "splits", config, {
        "splits_dir": str(splits),
        "strict_prelock": strict_prelock,
        "label_free_test_infer": label_free_test_infer,
        "label_free_fusion": label_free_fusion,
        "test_label_column_read": not (strict_prelock or label_free_test_infer or label_free_fusion),
    })
    if requested_stages == {"splits"}:
        print("[STAGE] splits complete. Stop because --stage splits was requested.")
        return

    if requested_stages in [{"development_features"}, {"features_dev"}]:
        run_development_feature_stage(train_df, val_df, config, out, novelty)
        print("[STAGE] development_features complete; locked-test images were not accessed.")
        return

    if requested_stages in [{"radiomics_nested_oof"}, {"nested_radiomics"}, {"radiomics_train_only"}, {"radiomics"}]:
        run_radiomics_nested_development_oof(train_df, val_df, config, out, models, reports)
        print("[STAGE] radiomics_nested_oof complete; all learner/hyperparameter selection used nested development OOF only.")
        return

    if requested_stages in [{"development_oof"}, {"dl_oof"}, {"oof"}]:
        run_development_oof_stages(train_df, val_df, config, out, models)
        print("[STAGE] development_oof complete for requested TF backbones; nested radiomics OOF was verified. Run XRV and EVA-X OOF scripts next.")
        return

    if requested_stages in [{"pair_select"}, {"dl_pair_selection"}]:
        run_q1_pair_selection(out, config)
        print("[STAGE] pair_select complete. The locked test and external sets were not read.")
        return

    if requested_stages in [{"selected_refit"}, {"development_refit"}]:
        run_selected_development_refit(train_df, val_df, config, out, models, reports)
        print("[STAGE] selected_refit complete/pending PyTorch selected refits; test was not accessed.")
        return

    if requested_stages in [{"selected_test_infer"}, {"locked_test_infer"}]:
        run_selected_locked_test_inference(train_df, val_df, test_df, config, out, models)
        print("[STAGE] selected_test_infer complete. Pair and threshold remained locked.")
        return

    # Stage-specific resumable branches. These branches intentionally avoid
    # running unrelated heavy prerequisites. For example, --stage fusion only
    # loads saved probabilities and never retrains KNN/CNN models.
    if requested_stages in [{"effnet"}, {"efficientnet"}, {"efficientnetv2"}]:
        run_single_cnn_stage("efficientnetv2", train_df, val_df, test_df, y_train, y_val, y_test,
                             config, out, models, reports, novelty)
        print("[STAGE] effnet complete. Stop because --stage effnet was requested.")
        return

    if requested_stages in [{"resnet"}, {"resnet50"}]:
        run_single_cnn_stage("resnet50", train_df, val_df, test_df, y_train, y_val, y_test,
                             config, out, models, reports, novelty)
        print("[STAGE] resnet complete. Stop because --stage resnet was requested.")
        return

    if requested_stages == {"fusion"}:
        pair_manifest = reports / "dl_pair_selection_manifest.json"
        if bool(getattr(config, "strict_selected_pair_fusion", False)) or pair_manifest.exists():
            fused = finalize_locked_test_fusion(out, exact_deployment=True, allow_deterministic_recompute=False)
            print(f"[FUSION] Selected DL pair: {fused['manifest']['selected_pair_display']} + Radiomics")
            print("[FUSION] Threshold source: development OOF; locked-test labels were not used.")
        else:
            run_fusion_from_saved(train_df, val_df, test_df, y_train, y_val, y_test,
                                  config, out, models, reports, plots)
        print("[STAGE] fusion complete. Stop because --stage fusion was requested.")
        return

    if requested_stages == {"ablation"} or (requested_stages == {"all"} and config.ablation_only):
        cache_dir = ensure_dir(out / config.cache_dirname)
        existing_inputs = load_fusion_inputs(out, y_val, y_test, config)
        soft_path = out / "probs" / "soft_test.npy"
        stacked_path = out / "probs" / "stacked_test.npy"
        if artifact_exists(soft_path):
            existing_inputs["soft_test_proba"] = np.load(soft_path)
        if artifact_exists(stacked_path):
            existing_inputs["stacked_test_proba"] = np.load(stacked_path)
        det_dir = out / "probs"
        for name in list(existing_inputs["cnn_outputs"].keys()):
            det_path = det_dir / f"{name}_test_deterministic.npy"
            if artifact_exists(det_path):
                existing_inputs[f"{name}_test_det"] = np.load(det_path)
        existing_for_ablation = {
            "wg_knn_test_proba": existing_inputs["wg_knn_test_proba"],
            "cnn_outputs": existing_inputs["cnn_outputs"],
            **{k: v for k, v in existing_inputs.items() if k.endswith("_test_det")},
        }
        if "soft_test_proba" in existing_inputs:
            existing_for_ablation["soft_test_proba"] = existing_inputs["soft_test_proba"]
        if "stacked_test_proba" in existing_inputs:
            existing_for_ablation["stacked_test_proba"] = existing_inputs["stacked_test_proba"]
        run_ablation_study(train_df, val_df, test_df, y_train, y_val, y_test,
                           config, cache_dir, reports, plots, models, existing_for_ablation)
        update_run_manifest(out, "ablation", config, {"results": str(reports / "ablation_results.csv")})
        print("[STAGE] ablation complete. Stop because --stage ablation/--ablation_only was requested.")
        return

    # ── NOVELTY 1: Adaptive Wavelet Feature Extraction ───────────────────────
    print("\n[NOVELTY 1] Adaptive Wavelet Bank Selection...")
    cache_dir = ensure_dir(out / config.cache_dirname)
    distances = [int(x) for x in config.glcm_distances.split(",")]
    angles    = [int(x) for x in config.glcm_angles.split(",")]

    main_feat_tag = radiomic_feature_tag(True, True, "db4", True, True, config.wavelet_levels)
    main_cache_path = cache_dir / f"{main_feat_tag}.joblib"
    main_wavelet_info_path = main_cache_path.with_suffix(".wavelet_info.json")
    if requested_stages == {"features"} and maybe_skip_stage("features", [main_cache_path, main_wavelet_info_path], config.force):
        update_run_manifest(out, "features", config, {"feature_cache": str(main_cache_path)})
        return
    cache, wavelet_info = compute_feature_cache(
        df, config.image_size, distances, angles,
        config.wavelet_levels, config.lbp_radius, config.lbp_n_points,
        main_cache_path,
        use_wavelet=True, adaptive=True, fixed_wavelet="db4",
        use_glcm=True, use_lbp=True, force=config.force,
    )
    validate_radiomic_cache(cache, config, main_cache_path)
    save_wavelet_selection_pie(wavelet_info, novelty / "n1_wavelet_selection.png")

    X_train = features_from_dataframe(train_df, cache)
    X_val   = features_from_dataframe(val_df,   cache)
    X_test  = features_from_dataframe(test_df,  cache)
    update_run_manifest(out, "features", config, {"feature_cache": str(main_cache_path)})
    if requested_stages == {"features"}:
        print("[STAGE] features complete. Stop because --stage features was requested.")
        return

    if config.save_tsne and ("all" in requested_stages or "features" in requested_stages):
        save_tsne_plot(X_train, y_train,
                       novelty / "n1_tsne_adaptive_wavelet.png",
                       config.tsne_max_samples)

    # ── NOVELTY 5: MRFO-optimized radiomics learner ──────────────────────────
    probs_dir = ensure_dir(out / "probs")
    generic_model = models / "radiomics_mrfo_best.pkl"
    legacy_model = models / "wg_knn_mrfo.pkl"
    generic_val = probs_dir / "radiomics_val.npy"
    generic_test = probs_dir / "radiomics_test.npy"
    legacy_val = probs_dir / "knn_val.npy"
    legacy_test = probs_dir / "knn_test.npy"

    model_to_load = generic_model if artifact_exists(generic_model) else legacy_model
    val_to_load = generic_val if artifact_exists(generic_val) else legacy_val
    test_to_load = generic_test if artifact_exists(generic_test) else legacy_test
    radiomics_outputs = [model_to_load, val_to_load, test_to_load]

    if maybe_skip_stage("radiomics", radiomics_outputs, config.force):
        wg_knn_pipe = joblib.load(model_to_load)
        wg_knn_val_proba = np.load(val_to_load)
        wg_knn_test_proba = np.load(test_to_load)
        wg_knn_thr = tune_threshold_from_validation(y_val, wg_knn_val_proba, config, "radiomics_mrfo", reports, plots)
        wg_knn_val_pred = predict_from_proba(wg_knn_val_proba, wg_knn_thr)
        wg_knn_test_pred = predict_from_proba(wg_knn_test_proba, wg_knn_thr)
    else:
        print(f"\n[NOVELTY 5] MRFO optimization for radiomics learner: {config.mrfo_estimator}...")
        wg_knn_pipe, mrfo_info = train_radiomics_mrfo(X_train, y_train, config)
        save_json(mrfo_info, reports / "radiomics_mrfo_results.json")
        save_json(mrfo_info, reports / "n5_mrfo_results.json")  # legacy alias
        save_mrfo_convergence_plot(mrfo_info["history"],
                                    novelty / "n5_mrfo_convergence.png")
        joblib.dump(wg_knn_pipe, generic_model)
        joblib.dump(wg_knn_pipe, legacy_model)          # compatibility
        joblib.dump(wg_knn_pipe, models / "knn_mrfo.joblib")
        wg_knn_val_proba  = wg_knn_pipe.predict_proba(X_val)
        wg_knn_test_proba = wg_knn_pipe.predict_proba(X_test)
        wg_knn_val_pred   = wg_knn_pipe.predict(X_val)
        wg_knn_test_pred  = wg_knn_pipe.predict(X_test)
        np.save(generic_val, wg_knn_val_proba)
        np.save(generic_test, wg_knn_test_proba)
        np.save(legacy_val, wg_knn_val_proba)            # compatibility
        np.save(legacy_test, wg_knn_test_proba)
        wg_knn_thr = tune_threshold_from_validation(y_val, wg_knn_val_proba, config, "radiomics_mrfo", reports, plots)
        wg_knn_val_pred = predict_from_proba(wg_knn_val_proba, wg_knn_thr)
        wg_knn_test_pred = predict_from_proba(wg_knn_test_proba, wg_knn_thr)
    update_run_manifest(out, "radiomics", config, {"model": str(generic_model), "probs": str(probs_dir)})
    if requested_stages == {"radiomics"}:
        print("[STAGE] radiomics complete. Stop because --stage radiomics was requested.")
        return

    # ── CNN Training: EfficientNetV2 + ResNet50 ──────────────────────────────
    print("\n[INFO] Training CNN backbones for triple ensemble...")
    cnn_specs: List[str] = []
    if config.train_efficientnetv2:
        cnn_specs.append("efficientnetv2")
    if config.train_resnet50:
        cnn_specs.append("resnet50")
    if not cnn_specs:
        raise ValueError("At least one CNN backbone must be enabled.")

    cnn_outputs: Dict[str, Dict] = {}
    for backbone_name in cnn_specs:
        preprocess_i = get_preprocess_for_backbone(backbone_name)
        model_i = None
        cnn_stage_name = "effnet" if backbone_name == "efficientnetv2" else "resnet"
        expected_cnn_outputs = [models / f"{backbone_name}_final.keras"] + list(_cnn_prob_paths(out, backbone_name).values())
        if maybe_skip_stage(cnn_stage_name, expected_cnn_outputs, config.force):
            paths = _cnn_prob_paths(out, backbone_name)
            val_mean = np.load(paths["val_mean"])
            val_std = np.load(paths["val_std"])
            val_ci = np.load(paths["val_ci"])
            test_mean = np.load(paths["test_mean"])
            test_std = np.load(paths["test_std"])
            test_ci = np.load(paths["test_ci"])
            try:
                model_i = tf.keras.models.load_model(str(models / f"{backbone_name}_final.keras"), compile=False)
            except Exception as exc:
                print(f"[WARN] Could not reload {backbone_name} model for optional XAI/deterministic ablation: {exc}")
                model_i = None
        else:
            model_i = train_cnn_backbone(backbone_name, train_df, val_df, y_train, config, models)

            print(f"\n[NOVELTY 2] MC Dropout Inference — {backbone_name}...")
            model_artifact_hash = sha256_file(models / f"{backbone_name}_final.keras")
            val_mean, val_std, val_ci = cnn_mc_predict_batch(
                model_i, val_df, config.image_size, config.batch_size,
                preprocess_i, n_mc=config.mc_dropout_n, output_dir=out,
                cache_prefix=f"{backbone_name}_val",
                resume=getattr(config, "auto_resume_cnn", True), force=config.force,
                model_artifact_sha256=model_artifact_hash, preprocessing_id=backbone_name,
            )
            test_mean, test_std, test_ci = cnn_mc_predict_batch(
                model_i, test_df, config.image_size, config.batch_size,
                preprocess_i, n_mc=config.mc_dropout_n, output_dir=out,
                cache_prefix=f"{backbone_name}_test",
                resume=getattr(config, "auto_resume_cnn", True), force=config.force,
                model_artifact_sha256=model_artifact_hash, preprocessing_id=backbone_name,
            )
        unc_thr = choose_uncertainty_threshold(val_ci, config)
        decision_thr = tune_threshold_from_validation(y_val, val_mean, config, backbone_name, reports, plots)
        val_pred = predict_from_proba(val_mean, decision_thr)
        test_pred = predict_from_proba(test_mean, decision_thr)
        val_flagged = flag_uncertain_cases(val_ci, unc_thr)
        test_flagged = flag_uncertain_cases(test_ci, unc_thr)
        s, sp = sens_spec(y_test, test_pred)
        print(f"[Novelty2] {backbone_name}: uncertainty threshold={unc_thr:.4f}; "
              f"flagged={test_flagged.sum()} / {len(test_flagged)} ({100*test_flagged.mean():.1f}%).")
        print(f"[Decision] {backbone_name}: validation-tuned threshold={decision_thr:.3f}; "
              f"final-test sensitivity={s:.3f}; specificity={sp:.3f}")

        suffix = backbone_name
        save_uncertainty_plot(test_ci, y_test, test_pred, unc_thr,
                              novelty / f"n2_uncertainty_{suffix}.png")
        unc_df = test_df[["sample_id", "patientId", "label"]].copy()
        unc_df[f"{suffix}_mean_proba"] = test_mean[:, 1]
        unc_df[f"{suffix}_std"] = test_std[:, 1]
        unc_df[f"{suffix}_predictive_interval_width_95"] = test_ci
        unc_df[f"{suffix}_ci_width_95"] = test_ci  # legacy alias
        unc_df[f"{suffix}_decision_threshold"] = decision_thr
        unc_df[f"{suffix}_needs_review"] = test_flagged
        unc_df.to_csv(reports / f"n2_uncertainty_report_{suffix}.csv", index=False)
        if backbone_name == "efficientnetv2":
            save_uncertainty_plot(test_ci, y_test, test_pred, unc_thr,
                                  novelty / "n2_uncertainty.png")
            unc_df.rename(columns={
                f"{suffix}_mean_proba": "cnn_mean_proba",
                f"{suffix}_std": "cnn_std",
                f"{suffix}_ci_width_95": "ci_width_95",
                f"{suffix}_needs_review": "needs_review",
            }).to_csv(reports / "n2_uncertainty_report.csv", index=False)

        # Save probabilities for stage-resumable fusion/ablation.
        np.save(probs_dir / f"{backbone_name}_val_mean.npy", val_mean)
        np.save(probs_dir / f"{backbone_name}_val_std.npy", val_std)
        np.save(probs_dir / f"{backbone_name}_val_ci.npy", val_ci)
        np.save(probs_dir / f"{backbone_name}_test_mean.npy", test_mean)
        np.save(probs_dir / f"{backbone_name}_test_std.npy", test_std)
        np.save(probs_dir / f"{backbone_name}_test_ci.npy", test_ci)

        cnn_outputs[backbone_name] = {
            "model": model_i,
            "preprocess_fn": preprocess_i,
            "val_mean": val_mean,
            "val_std": val_std,
            "val_ci": val_ci,
            "val_pred": val_pred,
            "val_flagged": val_flagged,
            "test_mean": test_mean,
            "test_std": test_std,
            "test_ci": test_ci,
            "test_pred": test_pred,
            "test_flagged": test_flagged,
            "uncertainty_threshold": unc_thr,
            "decision_threshold": decision_thr,
        }
        update_run_manifest(out, backbone_name, config, {"model": str(models / f"{backbone_name}.keras"), "probs": str(probs_dir)})

    if requested_stages in [{"effnet"}, {"efficientnet"}, {"efficientnetv2"}, {"resnet"}, {"resnet50"}]:
        print(f"[STAGE] {','.join(sorted(requested_stages))} complete. Stop because that stage was requested.")
        return

    # Backward-compatible aliases use EfficientNetV2 if present, otherwise first CNN.
    primary_backbone = "efficientnetv2" if "efficientnetv2" in cnn_outputs else cnn_specs[0]
    primary = cnn_outputs[primary_backbone]
    model = primary["model"]
    preprocess_fn = primary["preprocess_fn"]
    cnn_val_mean = primary["val_mean"]
    cnn_test_mean = primary["test_mean"]
    cnn_val_std = primary["val_std"]
    cnn_test_std = primary["test_std"]
    cnn_val_ci = primary["val_ci"]
    cnn_test_ci = primary["test_ci"]
    cnn_val_pred = primary["val_pred"]
    cnn_test_pred = primary["test_pred"]
    test_flagged = primary["test_flagged"]

    # Risk-score module removed: proposal uses bbox only for XAI ground truth, not scoring.

    # Proposal Tabel 3.12 skenario 7: deterministic CNN probabilities (no MC Dropout).
    deterministic_test_proba: Dict[str, np.ndarray] = {}
    if config.run_ablation or config.ablation_only or ("ablation" in requested_stages):
        for name in cnn_specs:
            det_path = probs_dir / f"{name}_test_deterministic.npy"
            if (not config.force) and artifact_exists(det_path):
                deterministic_test_proba[name] = np.load(det_path)
                continue
            if cnn_outputs[name].get("model") is None:
                print(f"[WARN] Deterministic ablation for {name} skipped because the model was not loaded. Rerun that stage or use --force.")
                continue
            deterministic_test_proba[name] = cnn_predict_deterministic_batch(
                cnn_outputs[name]["model"], test_df, config.image_size, config.batch_size,
                cnn_outputs[name]["preprocess_fn"],
            )
            np.save(det_path, deterministic_test_proba[name])

    # ── Ensemble: selected radiomics learner + CNN members ───────────────────
    radiomics_display = selected_radiomics_label(out, config.mrfo_estimator)
    print(f"\n[INFO] Building ensemble with {radiomics_display}...")
    wg_knn_thr = tune_threshold_from_validation(y_val, wg_knn_val_proba, config, "radiomics_mrfo", reports, plots)
    wg_knn_val_pred = predict_from_proba(wg_knn_val_proba, wg_knn_thr)
    wg_knn_test_pred = predict_from_proba(wg_knn_test_proba, wg_knn_thr)

    val_proba_list = [wg_knn_val_proba] + [cnn_outputs[name]["val_mean"] for name in cnn_specs]
    test_proba_list = [wg_knn_test_proba] + [cnn_outputs[name]["test_mean"] for name in cnn_specs]
    ensemble_names = [radiomics_display] + ["EfficientNetV2 (MC-Dropout)" if n == "efficientnetv2" else "ResNet50 (MC-Dropout)" for n in cnn_specs]

    soft_val_proba = np.mean(np.stack(val_proba_list, axis=0), axis=0)
    soft_test_proba = np.mean(np.stack(test_proba_list, axis=0), axis=0)
    soft_thr = tune_threshold_from_validation(y_val, soft_val_proba, config, "soft_voting", reports, plots)
    soft_val_pred = predict_from_proba(soft_val_proba, soft_thr)
    soft_test_pred = predict_from_proba(soft_test_proba, soft_thr)

    if config.use_stacking:
        stacked_val_proba = stacking_oof_probabilities(
            val_proba_list, y_val, n_splits=5, seed=config.seed)
        meta_clf = train_stacking_meta_learner_from_list(val_proba_list, y_val)
        joblib.dump(meta_clf, models / "meta_learner.pkl")
        joblib.dump(meta_clf, models / "stacking_lr.joblib")
        meta_info = {
            "feature_order": ensemble_names,
            "n_features": int(2 * len(val_proba_list)),
            "validation_probabilities": "5-fold out-of-fold",
            "final_meta_fit": "all validation rows",
        }
        save_json(meta_info, reports / "meta_learner_features.json")
        stacked_test_proba = meta_clf.predict_proba(np.concatenate(test_proba_list, axis=1))
    else:
        stacked_val_proba = soft_val_proba
        stacked_test_proba = soft_test_proba

    stacked_thr = tune_threshold_from_validation(y_val, stacked_val_proba, config, "stacked_ensemble", reports, plots)
    stacked_val_pred = predict_from_proba(stacked_val_proba, stacked_thr)
    stacked_test_pred = predict_from_proba(stacked_test_proba, stacked_thr)
    np.save(probs_dir / "soft_val.npy", soft_val_proba)
    np.save(probs_dir / "soft_test.npy", soft_test_proba)
    np.save(probs_dir / "soft_voting_test.npy", soft_test_proba)
    np.save(probs_dir / "stacked_val.npy", stacked_val_proba)
    np.save(probs_dir / "stacked_test.npy", stacked_test_proba)
    np.save(probs_dir / "stacking_test.npy", stacked_test_proba)
    update_run_manifest(out, "fusion", config, {"probs": str(probs_dir), "meta_learner": str(models / "meta_learner.pkl")})
    if requested_stages == {"fusion"}:
        print("[STAGE] fusion complete. Stop because --stage fusion was requested.")
        return

    # ── Metrics ──────────────────────────────────────────────────────────────
    val_metrics = {
        radiomics_display: compute_model_metrics(y_val, wg_knn_val_pred, wg_knn_val_proba),
        "Soft Voting": compute_model_metrics(y_val, soft_val_pred, soft_val_proba),
        "Stacked Ensemble": compute_model_metrics(y_val, stacked_val_pred, stacked_val_proba),
    }
    test_metrics = {
        radiomics_display: compute_model_metrics(y_test, wg_knn_test_pred, wg_knn_test_proba),
        "Soft Voting": compute_model_metrics(y_test, soft_test_pred, soft_test_proba),
        "Stacked Ensemble": compute_model_metrics(y_test, stacked_test_pred, stacked_test_proba),
    }
    for name, display in zip(cnn_specs, ensemble_names[1:]):
        val_metrics[display] = compute_model_metrics(y_val, cnn_outputs[name]["val_pred"], cnn_outputs[name]["val_mean"])
        test_metrics[display] = compute_model_metrics(y_test, cnn_outputs[name]["test_pred"], cnn_outputs[name]["test_mean"])

    save_json(val_metrics, reports / "metrics_val.json")
    save_json(test_metrics, reports / "metrics_test.json")
    save_performance_table(val_metrics, test_metrics, reports / "performance_table")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for split_name, mdict, y_ref in [("Validation", val_metrics, y_val), ("Test", test_metrics, y_test)]:
        print(f"\n{split_name}:")
        for mname, m in mdict.items():
            print(f"  {mname:32s} | Acc={m['accuracy']:.4f} | F1w={m['f1_weighted']:.4f} | AUC={m['auc']:.4f}")
    threshold_summary = {
        radiomics_display: wg_knn_thr,
        "Soft Voting": soft_thr,
        "Stacked Ensemble": stacked_thr,
        **{name: cnn_outputs[name].get("decision_threshold", config.decision_threshold) for name in cnn_specs},
    }
    save_json({k: float(v) for k, v in threshold_summary.items()}, reports / "final_operating_thresholds_validation_tuned.json")
    print("\n[Operating point] validation-tuned thresholds locked before final test evaluation:")
    for k, v in threshold_summary.items():
        print(f"  {k:32s}: threshold={float(v):.3f}")

    # ── Confusion Matrices ────────────────────────────────────────────────────
    cm_items = [("wg_knn", wg_knn_test_pred), ("soft_voting", soft_test_pred), ("stacked", stacked_test_pred)]
    for name in cnn_specs:
        cm_items.append((name, cnn_outputs[name]["test_pred"]))
    for tag, y_pred in cm_items:
        save_confusion_matrix(y_test, y_pred, plots / f"cm_{tag}_test.png",
                              f"Confusion Matrix — {tag.replace('_', ' ').title()}")

    # ── ROC + PR Curves ───────────────────────────────────────────────────────
    roc_d, y_t_d, y_p_d = {}, {}, {}
    curve_items = [(radiomics_display, wg_knn_test_proba), ("Soft Voting", soft_test_proba), ("Stacked Ensemble", stacked_test_proba)]
    for name, display in zip(cnn_specs, ensemble_names[1:]):
        curve_items.append((display, cnn_outputs[name]["test_mean"]))
    for mname, proba in curve_items:
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
        roc_d[mname] = (fpr, tpr, auc(fpr, tpr))
        y_t_d[mname] = y_test
        y_p_d[mname] = proba[:, 1]
    save_roc_pr_curves(roc_d, y_t_d, y_p_d,
                        plots / "roc_pr_curves.png",
                        "ROC & Precision-Recall — Triple Ensemble Models")

    if config.run_ablation or config.ablation_only or ("ablation" in requested_stages):
        existing = {
            "wg_knn_test_proba": wg_knn_test_proba,
            "cnn_outputs": cnn_outputs,
            "soft_test_proba": soft_test_proba,
            "stacked_test_proba": stacked_test_proba,
        }
        for name, arr in deterministic_test_proba.items():
            existing[f"{name}_test_det"] = arr
        run_ablation_study(train_df, val_df, test_df, y_train, y_val, y_test,
                           config, cache_dir, reports, plots, models, existing)
        update_run_manifest(out, "ablation", config, {"results": str(reports / "ablation_results.csv")})
        if config.ablation_only or requested_stages == {"ablation"}:
            print("[STAGE] ablation complete. Stop because ablation-only/stage was requested.")
            return

    # ── NOVELTY 4: Contrastive GradCAM ───────────────────────────────────────
    print("\n[NOVELTY 4] Contrastive GradCAM (Why-A-not-B XAI)...")
    try:
        n_xai_samples = min(config.num_xai_samples, len(test_df))
        xai_indices = select_xai_samples(y_test, stacked_test_pred, n_xai_samples)
        xai_imgs, xai_preds, xai_trues = [], [], []
        for idx in xai_indices:
            row = test_df.iloc[idx]
            raw = load_dicom_grayscale(row["image_path"], config.image_size)
            xai_imgs.append(raw)
            xai_preds.append(int(stacked_test_pred[idx]))
            xai_trues.append(int(y_test[idx]))

        # Save GradCAM for primary model and per-CNN backbones.
        save_contrastive_gradcam_grid(
            model, xai_imgs, xai_preds, xai_trues,
            None, xai / "n4_contrastive_gradcam.png",
            preprocess_fn, config.image_size,
        )
        for name in cnn_specs:
            if name == primary_backbone:
                continue
            save_contrastive_gradcam_grid(
                cnn_outputs[name]["model"], xai_imgs, xai_preds, xai_trues,
                None, xai / f"n4_contrastive_gradcam_{name}.png",
                cnn_outputs[name]["preprocess_fn"], config.image_size,
            )
    except Exception as e:
        print(f"[WARN] Contrastive GradCAM failed: {e}")
        import traceback; traceback.print_exc()

    # ── Save Predictions ──────────────────────────────────────────────────────
    pred_df = test_df[["sample_id", "patientId", "label"]].copy()
    pred_df["stacked_decision_threshold"] = stacked_thr
    pred_df["soft_voting_decision_threshold"] = soft_thr
    pred_df["radiomics_decision_threshold"] = wg_knn_thr
    pred_df["radiomics_prob"] = wg_knn_test_proba[:, 1]
    pred_df["radiomics_pred"] = wg_knn_test_pred
    # Legacy aliases retained for downstream scripts created before stage rename.
    pred_df["wg_knn_decision_threshold"] = wg_knn_thr
    pred_df["wg_knn_prob"] = wg_knn_test_proba[:, 1]
    pred_df["wg_knn_pred"] = wg_knn_test_pred
    for name in cnn_specs:
        prefix = "effnetv2" if name == "efficientnetv2" else name
        out_i = cnn_outputs[name]
        pred_df[f"{prefix}_prob"] = out_i["test_mean"][:, 1]
        pred_df[f"{prefix}_std"] = out_i["test_std"][:, 1]
        pred_df[f"{prefix}_predictive_interval_width_95"] = out_i["test_ci"]
        pred_df[f"{prefix}_ci95"] = out_i["test_ci"]  # legacy alias
        pred_df[f"{prefix}_uncertainty_threshold"] = out_i["uncertainty_threshold"]
        pred_df[f"{prefix}_decision_threshold"] = out_i.get("decision_threshold", config.decision_threshold)
        pred_df[f"{prefix}_needs_review"] = out_i["test_flagged"]
        pred_df[f"{prefix}_pred"] = out_i["test_pred"]
    # Compatibility columns expected by older eval/report scripts.
    pred_df["cnn_mean_prob"] = cnn_test_mean[:, 1]
    pred_df["cnn_std"] = cnn_test_std[:, 1]
    pred_df["cnn_predictive_interval_width_95"] = cnn_test_ci
    pred_df["cnn_ci95"] = cnn_test_ci  # legacy alias
    pred_df["needs_review"] = test_flagged
    pred_df["cnn_pred"] = cnn_test_pred
    pred_df["soft_prob"] = soft_test_proba[:, 1]
    pred_df["soft_pred"] = soft_test_pred
    pred_df["stacked_prob"] = stacked_test_proba[:, 1]
    pred_df["stacked_pred"] = stacked_test_pred
    pred_df.to_csv(reports / "test_predictions_full.csv", index=False)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n[INFO] ✓ All outputs saved to:", out)
    print("[INFO] Novel contributions:")
    print("  Novelty 1 — Adaptive Wavelet Bank Selection (pie chart + t-SNE)")
    print("  Novelty 2 — Bayesian MC Dropout Uncertainty (flagged cases)")
    print("  Contribution 3 — Bayesian MC Dropout Uncertainty (mean ± CI95, review flag)")
    print("  Contribution 4 — Ablation Study + Contrastive GradCAM / quantitative XAI")
    print("  MRFO Optimization for WG-KNN (convergence plot)")
    print("\n[INFO] Output structure:")
    for sub in ["plots", "xai", "models", "reports", "novelty_outputs", "splits"]:
        files = list((out / sub).glob("*"))
        print(f"  {sub}/  ({len(files)} files)")


if __name__ == "__main__":
    main()

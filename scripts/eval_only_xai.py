#!/usr/bin/env python3
"""
eval_only_xai_Q1_STRONG_BASELINE_CV_V8.py — ROBUST VALIDATION / STRONG-BASELINE Q1 XAI

Proposal-aligned evaluation and quantitative XAI script for AURA-CXR.
The final classification source and explainable model scope are read from a locked
patient-level development-OOF selected-pair manifest; non-selected CNNs are never
silently substituted.

Major additions compared with eval_only_v4_PY313.py:
  1. Radiologist ground-truth box support from --labels_csv.
  2. Quantitative localization metrics:
       - Pointing Game accuracy
       - IoU@threshold against radiologist boxes
       - Localization score = max IoU over multiple thresholds
       - Energy-inside-GT and peak distance diagnostics
  3. Ground-truth overlay in qualitative GradCAM figures.
  4. Failure analysis panel for GradCAM cases that focus outside the GT lesion,
     false positives, and false negatives.
  5. Aggregate heatmaps averaged by GT class and by prediction category.
  6. Paper-friendly resolution and larger labels.
  7. Optional LIME and SHAP visualizations.
  8. Fixed-layer/logit Grad-CAM and signed Contrastive Grad-CAM diagnostics.
  9. Primary quantitative map = Pneumonia-class Grad-CAM (not contrastive subtraction).
 10. Strict post-processed lung gating with provenance and outside-lung attribution audit.
 11. Resolution-aware deterministic CAM layer selection and float32/XLA-off XAI defaults.
 12. Bootstrap 95% CIs, zero-map rejection, chance baselines, and Q1 QC fail-fast.
 13. Balanced localization audit, repeated LIME stability, and signed SHAP with train background.
 14. Validation-only fixed CAM-layer selection with a reusable audit manifest.
 15. Paired random/center/lung-prior baselines and raw-vs-gated shortcut-bias QC.
 16. Aggregate raw-CAM/lung-prior/residual audit and median/IQR reporting.
 17. Hard QUICK-vs-FINAL artifact isolation and minimum full-test coverage enforcement.
 18. Final baseline-superiority, validation-layer-lock, and Q1 readiness report guardrails.
 19. Automatic Markdown/CSV/JSON final XAI results table with explicit readiness verdict.
 20. Validation-only joint selection of CAM method, single/multi-layer configuration,
     CNN fusion weights, lung-constraint policy, and heatmap threshold.
 21. Grad-CAM, Grad-CAM++, LayerCAM, and HiResCAM candidates using pre-softmax logits.
 22. Strong baselines: repeated random maps, Gaussian image-center, lung prior,
     and training-derived lesion-prevalence prior.
 23. Non-fatal scientific QC: technical completion is separated from validated
     localization-claim readiness.
 24. Faithfulness deletion/retention audit, TP-vs-FN statistics, and optional
     classifier-head randomization sanity check.
 25. Pilot compatibility screening rejects systematically invalid layer-method
     pairs before full validation evaluation.
 26. GradCAM++ and DepthwiseConv2D are excluded from default candidates because
     they produced repeated zero maps; both remain explicit opt-in options.
 27. Candidate eligibility requires a configurable full-validation valid-map
     rate, and failed maps are excluded rather than scored as localization zero.
 28. Repeated CAM-failure warnings are deduplicated per model/layer/method/class.
 29. Deterministic TTA lung-mask recovery reprocesses heuristic/ellipse cache entries.
 30. Primary CAM selection excludes hard anatomical gating by default and rewards
     superiority over center, lung, and training lesion-prevalence priors.
 31. Robust validation uses fold stability, bootstrap lower bounds, and an all-positive
     confirmation audit before the test configuration is locked.
 32. Ellipse-fallback cases are retained only as an explicitly separated sensitivity cohort.

Example:
  python eval_only_xai.py \
    --results_dir /path/to/aura_results \
    --dicom_dir /path/to/rsna/stage_2_train_images \
    --labels_csv /path/to/rsna/stage_2_train_labels.csv \
    --gpu40 --n_xai 8 --cam_pdf_n 100 --cam_pdf_per_page 4 --xai_eval_max 0 --n_lime 4 --n_shap 4 --paper_dpi 600

Notes:
  - xai_eval_max=0 evaluates all test samples. Use a positive number for a
    faster stratified subset.
  - LIME requires: pip install lime scikit-image
  - SHAP requires: pip install shap
"""

import argparse
import copy
import importlib.util
import itertools
import sys
import textwrap
from types import SimpleNamespace
import gc
import math
import os
import json
import shutil
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
from scipy.stats import mannwhitneyu, spearmanr
from scipy.ndimage import (
    label as scipy_label,
    zoom,
    gaussian_filter,
    binary_closing,
    binary_opening,
    binary_fill_holes,
    binary_dilation,
    distance_transform_edt,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from aura_dl_pair_selection import (
    load_pair_manifest, DISPLAY_NAMES, load_deployment_manifest,
    resolve_locked_model_artifact, resolve_probability_file,
    load_oof_artifact, ensure_development_table, sha256_file, normalize_binary_probability,
)

PIL_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
PIL_NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

DEFAULT_DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

CMAP_HEATMAP = LinearSegmentedColormap.from_list(
    "jet_hot", ["navy", "blue", "cyan", "lime", "yellow", "red"]
)
CLASS_NAMES = {0: "Non-Pneumonia", 1: "Pneumonia"}
CATEGORY_COLORS = {
    "TP": "#00C853",
    "TN": "#2196F3",
    "FP": "#FF5722",
    "FN": "#D50000",
}
CATEGORY_LABELS = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
}
GT_COLOR = "#00E5FF"
PRED_COLOR = "#FF1744"
ACTIVATION_THRESHOLD = 1e-5
XAI_SCHEMA_VERSION = "q1_fixed_best_radiomics_locked_pair_v18_dual_cnn_cam"
_CAM_LAYER_LOCK = {}
_CAM_LAYER_SELECTION_AUDIT = {}
_CAM_CONFIG_LOCK = {}
# Runtime cache remembers a valid deterministic layer/method per model and class.
# This avoids repeating an expensive failed GradCAM++ attempt on every image.
_GRADCAM_RUNTIME_CACHE = {}
# Deduplicate systematic CAM failures so one bad candidate cannot flood logs.
_CAM_FAILURE_WARNING_COUNTS = {}
_XAI_DEPLOYMENT_LOCK_ID = ""
_XAI_MODEL_HASHES = {}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def normalize_minmax(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(lo - hi) < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def to_uint8(img):
    return np.clip(normalize_minmax(img) * 255.0, 0, 255).astype(np.uint8)


def load_dicom_grayscale(path, image_size=224):
    """Load a grayscale image normalized to [0,1].

    Supports both RSNA DICOM files and external JPEG/PNG images such as the
    Kermany cohort. The historic function name is retained for compatibility.
    """
    path = Path(path)
    if path.suffix.lower() == ".dcm":
        ds = pydicom.dcmread(str(path))
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
            img = img.max() - img
    else:
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    img = normalize_minmax(img)
    pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
    pil = pil.resize((image_size, image_size), resample=PIL_BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def get_dicom_hw(path):
    """Return original DICOM height and width without reading full pixel data when possible."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        h = int(getattr(ds, "Rows"))
        w = int(getattr(ds, "Columns"))
        return h, w
    except Exception:
        ds = pydicom.dcmread(str(path))
        return int(ds.pixel_array.shape[0]), int(ds.pixel_array.shape[1])


def preprocess_efficientnet(batch_rgb):
    arr = tf.cast(batch_rgb, tf.float32) * 255.0
    efficientnet_v2 = getattr(tf.keras.applications, "efficientnet_v2", None)
    if efficientnet_v2 is not None and hasattr(efficientnet_v2, "preprocess_input"):
        return efficientnet_v2.preprocess_input(arr)
    return tf.keras.applications.efficientnet.preprocess_input(arr)


def preprocess_resnet(batch_rgb):
    arr = tf.cast(batch_rgb, tf.float32) * 255.0
    return tf.keras.applications.resnet50.preprocess_input(arr)


def predict_from_proba(proba, threshold=0.40):
    return (np.asarray(proba)[:, 1] >= float(threshold)).astype(np.int32)


def sens_spec(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def resize_heatmap(heatmap, target_h, target_w):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or min(heatmap.shape) <= 0:
        return np.zeros((target_h, target_w), dtype=np.float32)
    factor = (target_h / heatmap.shape[0], target_w / heatmap.shape[1])
    return np.clip(zoom(heatmap, factor, order=1), 0, 1).astype(np.float32)


def overlay_heatmap_rgba(raw_image, heatmap, alpha=0.50):
    h, w = raw_image.shape[:2]
    hm_r = resize_heatmap(heatmap, h, w)
    hm_c = (CMAP_HEATMAP(hm_r)[:, :, :3] * 255).astype(np.uint8)
    raw_3ch = raw_image if raw_image.ndim == 3 else np.repeat(raw_image[..., None], 3, axis=-1)
    if raw_3ch.dtype != np.uint8:
        raw_3ch = to_uint8(raw_3ch)
    blended = (alpha * hm_c + (1.0 - alpha) * raw_3ch).astype(np.uint8)
    return blended, hm_r



def overlay_signed_heatmap(raw_image, signed_heatmap, alpha=0.48):
    """Overlay a signed attribution map using a symmetric diverging scale.

    Positive values support Pneumonia and negative values support
    Non-Pneumonia. This is used only as a supplementary contrastive view; the
    quantitative localization metrics use the primary Pneumonia-class CAM.
    """
    h, w = raw_image.shape[:2]
    hm = np.asarray(signed_heatmap, dtype=np.float32)
    if hm.ndim != 2:
        hm = np.zeros((h, w), dtype=np.float32)
    else:
        factor = (h / hm.shape[0], w / hm.shape[1])
        hm = zoom(hm, factor, order=1).astype(np.float32)
        hm = hm[:h, :w]
        if hm.shape != (h, w):
            fixed = np.zeros((h, w), dtype=np.float32)
            fixed[:hm.shape[0], :hm.shape[1]] = hm
            hm = fixed
    vmax = float(np.nanmax(np.abs(hm))) if hm.size else 0.0
    hm = hm / vmax if np.isfinite(vmax) and vmax > 1e-8 else np.zeros_like(hm)
    cmap = plt.get_cmap("coolwarm")
    colored = (cmap((hm + 1.0) / 2.0)[..., :3] * 255).astype(np.uint8)
    raw_3ch = raw_image if raw_image.ndim == 3 else np.repeat(raw_image[..., None], 3, axis=-1)
    if raw_3ch.dtype != np.uint8:
        raw_3ch = to_uint8(raw_3ch)
    blended = (alpha * colored + (1.0 - alpha) * raw_3ch).astype(np.uint8)
    return blended, hm


def resize_binary_mask(mask, target_h, target_w):
    """Resize a binary mask with nearest-neighbour interpolation only."""
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2:
        return np.zeros((target_h, target_w), dtype=bool)
    pil = Image.fromarray(arr * 255, mode="L")
    pil = pil.resize((int(target_w), int(target_h)), resample=PIL_NEAREST)
    return np.asarray(pil, dtype=np.uint8) >= 128


def postprocess_lung_mask(mask, target_shape, dilation_frac=0.012, min_area=0.08, max_area=0.85):
    """Clean and validate a lung mask for strict heatmap gating.

    The two largest connected components are retained, holes are filled, and a
    small deterministic dilation protects peripheral opacities near the pleura.
    Invalid masks are rejected rather than silently used.
    """
    h, w = int(target_shape[0]), int(target_shape[1])
    m = resize_binary_mask(mask, h, w)
    if not m.any():
        return None
    m = binary_opening(m, iterations=1)
    m = binary_closing(m, iterations=2)
    m = binary_fill_holes(m)
    lbl, n_comp = scipy_label(m.astype(np.uint8))
    if n_comp > 0:
        comps = []
        for cid in range(1, n_comp + 1):
            area = int((lbl == cid).sum())
            if area > 0:
                comps.append((area, cid))
        comps.sort(reverse=True)
        keep = {cid for _, cid in comps[:2]}
        m = np.isin(lbl, list(keep))
    iters = max(1, int(round(min(h, w) * float(dilation_frac))))
    m = binary_dilation(m, iterations=iters)
    area_ratio = float(m.mean())
    if not (float(min_area) <= area_ratio <= float(max_area)):
        return None
    return m.astype(bool)


def compute_heatmap_audit(heatmap, lung_mask=None):
    """Return numerical quality-control fields for one attribution map."""
    hm = np.asarray(heatmap, dtype=np.float32)
    finite = np.isfinite(hm)
    if hm.ndim != 2 or hm.size == 0 or not finite.any():
        return {
            "heatmap_valid": False, "heatmap_max": np.nan,
            "heatmap_mean": np.nan, "heatmap_std": np.nan,
            "heatmap_nonzero_ratio": 0.0, "heatmap_entropy": np.nan,
            "inside_lung_energy": np.nan, "outside_lung_energy": np.nan,
            "outside_lung_ratio": np.nan, "lung_mask_area_ratio": np.nan,
        }
    clean = np.where(finite, np.clip(hm, 0, None), 0.0).astype(np.float32)
    mx = float(clean.max())
    mean = float(clean.mean())
    std = float(clean.std())
    nz = float((clean > max(ACTIVATION_THRESHOLD, mx * 1e-4)).mean()) if mx > 0 else 0.0
    total = float(clean.sum())
    valid = bool(mx > ACTIVATION_THRESHOLD and std > 1e-7 and nz > 1e-4 and total > 1e-8)
    entropy = np.nan
    if total > 1e-8:
        p = clean.ravel() / total
        p = p[p > 0]
        entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(clean.size, 2)))
    inside = outside = ratio = mask_ratio = np.nan
    if lung_mask is not None:
        lm = np.asarray(lung_mask, dtype=bool)
        if lm.shape != clean.shape:
            lm = resize_binary_mask(lm, clean.shape[0], clean.shape[1])
        mask_ratio = float(lm.mean())
        inside = float(clean[lm].sum())
        outside = float(clean[~lm].sum())
        ratio = float(outside / total) if total > 1e-8 else np.nan
    return {
        "heatmap_valid": valid,
        "heatmap_max": mx,
        "heatmap_mean": mean,
        "heatmap_std": std,
        "heatmap_nonzero_ratio": nz,
        "heatmap_entropy": entropy,
        "inside_lung_energy": inside,
        "outside_lung_energy": outside,
        "outside_lung_ratio": ratio,
        "lung_mask_area_ratio": mask_ratio,
    }


def _bootstrap_mean_ci(values, n_boot=2000, seed=42):
    vals = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1 or int(n_boot) <= 0:
        return float(vals.mean()), float(vals.mean())
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        means[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _bootstrap_stat_ci(values, stat="mean", n_boot=2000, seed=42):
    """Percentile bootstrap CI for mean or median, with NaN-safe input."""
    vals = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    fn = np.nanmedian if str(stat).lower() == "median" else np.nanmean
    if len(vals) == 1 or int(n_boot) <= 0:
        v = float(fn(vals))
        return v, v
    rng = np.random.default_rng(int(seed))
    stats_arr = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        stats_arr[i] = fn(rng.choice(vals, size=len(vals), replace=True))
    return float(np.percentile(stats_arr, 2.5)), float(np.percentile(stats_arr, 97.5))


def _paired_bootstrap_diff_ci(model_values, baseline_values, n_boot=2000, seed=42):
    """Paired bootstrap CI for mean(model - baseline)."""
    a = pd.to_numeric(pd.Series(model_values), errors="coerce")
    b = pd.to_numeric(pd.Series(baseline_values), errors="coerce")
    valid = a.notna() & b.notna()
    diff = (a[valid] - b[valid]).to_numpy(dtype=float)
    if len(diff) == 0:
        return np.nan, np.nan, np.nan, 0
    point = float(diff.mean())
    if len(diff) == 1 or int(n_boot) <= 0:
        return point, point, point, int(len(diff))
    rng = np.random.default_rng(int(seed))
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        boots[i] = rng.choice(diff, size=len(diff), replace=True).mean()
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), int(len(diff))

def _get_category(true_l, pred_l):
    mapping = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}
    return mapping.get((int(true_l), int(pred_l)), "??")


def _add_border(ax, color, lw=3.0):
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)
        sp.set_visible(True)


def format_metric(x, nd=3):
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(xf):
        return "NA"
    return f"{xf:.{nd}f}"



def _compact_xai_text(value, max_chars=58):
    """Shorten long provenance strings for figure footers without changing data."""
    if value is None:
        return "NA"
    s = str(value).strip()
    replacements = (
        ("validation_weighted_cnn_", "weighted "),
        ("validation_selected_", "selected "),
        ("pretrained_adaptive_", "pretrained adaptive "),
        ("pretrained_mask", "pretrained mask"),
        ("ellipse_runtime_fallback", "ellipse fallback"),
        ("ellipse_fallback", "ellipse fallback"),
        ("_", " "),
    )
    for old, new in replacements:
        s = s.replace(old, new)
    s = " ".join(s.split())
    return textwrap.shorten(s, width=max(int(max_chars), 8), placeholder="…")


def _add_metric_footer(ax, text, fontsize=7.2, y=-0.115, width=48):
    """Place technical metadata below an image instead of covering the image."""
    wrapped = "\n".join(
        textwrap.fill(str(line), width=max(int(width), 18), break_long_words=False)
        for line in str(text).splitlines()
    )
    ax.text(
        0.5, y, wrapped,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=fontsize, color="black",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white", edgecolor="#B0BEC5",
            linewidth=0.8, alpha=0.97,
        ),
        clip_on=False, zorder=30,
    )


def _style_image_axis(ax, title=None, border_color=None, title_size=10.5):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=5)
    if border_color is not None:
        _add_border(ax, border_color, lw=1.8)


def _save_xai_figure(fig, path, paper_dpi=600, save_pdf=False, pad_inches=0.08):
    path = Path(path)
    fig.savefig(
        path, dpi=paper_dpi, bbox_inches="tight",
        pad_inches=pad_inches, facecolor="white",
    )
    if save_pdf:
        fig.savefig(
            path.with_suffix(".pdf"), bbox_inches="tight",
            pad_inches=pad_inches, facecolor="white",
        )


def load_probability_array(path, expected_rows=None, name=None, require_two_columns=True):
    """Load and validate a cached probability array.

    This prevents silent evaluation with stale/misaligned fusion artifacts.
    Accepted input is an N x 2 array [P(negative), P(positive)]. A one-column
    positive-class vector is converted to two columns only when
    ``require_two_columns`` is False.
    """
    path = Path(path)
    label = name or path.name
    if not path.exists():
        raise FileNotFoundError(f"Probability file not found: {path}")

    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1 and not require_two_columns:
        arr = np.column_stack([1.0 - arr, arr]).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{label} must have shape (N, 2), got {arr.shape}")
    if expected_rows is not None and len(arr) != int(expected_rows):
        raise ValueError(
            f"{label} row mismatch: got {len(arr)}, expected {int(expected_rows)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN/Inf values")
    if float(arr.min()) < -1e-5 or float(arr.max()) > 1.0 + 1e-5:
        raise ValueError(
            f"{label} contains values outside [0,1]: min={arr.min():.6g}, max={arr.max():.6g}"
        )

    # Normalize tiny numerical drift, but reject clearly invalid rows.
    row_sum = arr.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 1e-8):
        raise ValueError(f"{label} contains zero-sum probability rows")
    max_deviation = float(np.max(np.abs(row_sum - 1.0)))
    if max_deviation > 5e-2:
        raise ValueError(
            f"{label} rows do not sum to one (max deviation={max_deviation:.6f})"
        )
    if max_deviation > 1e-5:
        arr = arr / row_sum
    return arr.astype(np.float32)


# -----------------------------------------------------------------------------
# Radiologist annotation / ground truth boxes
# -----------------------------------------------------------------------------

def _find_col(df, aliases):
    lower_to_real = {str(c).lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_to_real:
            return lower_to_real[alias.lower()]
    return None


def load_radiologist_bboxes(labels_csv):
    """
    Load bbox annotations from RSNA-style labels CSV.

    Supported formats:
      - patientId, x, y, width, height, Target
      - patientId, xmin/x_min, ymin/y_min, xmax/x_max, ymax/y_max, Target/label

    Returns a dict: patient_id -> list of original-coordinate boxes (x, y, w, h).
    """
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        print(f"[WARN] labels_csv not found: {labels_csv}. Ground-truth overlay disabled.")
        return {}

    df = pd.read_csv(labels_csv)
    pid_col = _find_col(df, ["patientId", "patient_id", "id", "sample_id"])
    if pid_col is None:
        print("[WARN] labels_csv has no patientId/patient_id column. Ground-truth overlay disabled.")
        return {}

    target_col = _find_col(df, ["Target", "target", "label", "Label"])
    x_col = _find_col(df, ["x", "xmin", "x_min", "left"])
    y_col = _find_col(df, ["y", "ymin", "y_min", "top"])
    w_col = _find_col(df, ["width", "w"])
    h_col = _find_col(df, ["height", "h"])
    x2_col = _find_col(df, ["xmax", "x_max", "right"])
    y2_col = _find_col(df, ["ymax", "y_max", "bottom"])

    if x_col is None or y_col is None or (w_col is None and x2_col is None) or (h_col is None and y2_col is None):
        print("[WARN] labels_csv does not contain recognizable bbox columns. Ground-truth boxes disabled.")
        return {}

    out = {}
    for _, row in df.iterrows():
        if target_col is not None:
            try:
                if int(float(row[target_col])) != 1:
                    continue
            except Exception:
                continue
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if w_col is not None and h_col is not None:
                w = float(row[w_col])
                h = float(row[h_col])
            else:
                w = float(row[x2_col]) - x
                h = float(row[y2_col]) - y
            if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
                continue
        except Exception:
            continue
        pid = str(row[pid_col])
        out.setdefault(pid, []).append((x, y, w, h))

    n_boxes = sum(len(v) for v in out.values())
    print(f"[INFO] Radiologist annotations loaded: {len(out)} positive patients, {n_boxes} boxes")
    return out


def scale_boxes_to_image(patient_id, image_path, gt_box_map, image_size):
    """Scale original DICOM-coordinate boxes to the resized model image."""
    boxes = gt_box_map.get(str(patient_id), [])
    if not boxes:
        return []
    try:
        h0, w0 = get_dicom_hw(image_path)
    except Exception:
        h0, w0 = image_size, image_size
    sx = float(image_size) / max(float(w0), 1.0)
    sy = float(image_size) / max(float(h0), 1.0)
    scaled = []
    for x, y, w, h in boxes:
        x0 = max(0.0, min(float(image_size), x * sx))
        y0 = max(0.0, min(float(image_size), y * sy))
        x1 = max(0.0, min(float(image_size), (x + w) * sx))
        y1 = max(0.0, min(float(image_size), (y + h) * sy))
        if x1 > x0 and y1 > y0:
            scaled.append((x0, y0, x1 - x0, y1 - y0))
    return scaled


def boxes_to_mask(boxes, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for x, y, w, h in boxes or []:
        x0 = int(max(0, math.floor(x)))
        y0 = int(max(0, math.floor(y)))
        x1 = int(min(width, math.ceil(x + w)))
        y1 = int(min(height, math.ceil(y + h)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def draw_boxes(ax, boxes, color=GT_COLOR, label="GT", lw=2.8, linestyle="--", fontsize=10):
    n = 0
    for i, (x, y, w, h) in enumerate(boxes or []):
        rect = mpatches.Rectangle(
            (x, y), w, h,
            linewidth=lw,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)
        txt = label if i == 0 else f"{label} {i+1}"
        ax.text(
            x + w / 2.0,
            max(2, y - 5),
            txt,
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec=color, lw=0.8, alpha=0.72),
        )
        n += 1
    return n



class _TorchLayerProxy:
    def __init__(self, name, grid=14):
        self.name = str(name)
        self.output = SimpleNamespace(shape=(None, int(grid), int(grid), 1))


class TorchCamAdapter:
    """Minimal PyTorch model adapter for exact selected-pair CAM and probability calls."""
    _is_torch_adapter = True
    def __init__(self, name, model, device, transform_batch, target_modules, kind="conv"):
        self.name = name; self._aura_cam_name = name; self.model = model; self.device = device
        self.transform_batch = transform_batch; self.target_modules = dict(target_modules); self.kind = kind
        self.layers = [_TorchLayerProxy(k, 14 if kind == "conv" else 14) for k in self.target_modules]

    def get_layer(self, name):
        return next((x for x in self.layers if x.name == str(name)), None)

    def _tensor(self, images):
        import torch
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim == 3: arr = arr[None]
        return self.transform_batch(arr).to(self.device)

    def predict_numpy(self, images, mc_passes=1):
        import torch
        import torch.nn.functional as F
        x = self._tensor(images); draws=[]
        with torch.no_grad():
            for _ in range(max(1,int(mc_passes))):
                self.model.eval()
                if int(mc_passes)>1:
                    for mod in self.model.modules():
                        if isinstance(mod, torch.nn.Dropout): mod.train()
                draws.append(F.softmax(self.model(x),dim=1).detach().cpu().numpy())
        return np.mean(np.stack(draws),axis=0).astype(np.float32)

    def cam_from_rgb(self, raw_batch, layer_name, class_index=1, method="GradCAM"):
        import torch
        module = self.target_modules[str(layer_name)]
        captured = {}
        def hook(_m,_i,o):
            if isinstance(o,(tuple,list)): o=o[0]
            captured["act"] = o
            if hasattr(o,"retain_grad"): o.retain_grad()
        handle=module.register_forward_hook(hook)
        try:
            self.model.zero_grad(set_to_none=True); self.model.eval(); x=self._tensor(raw_batch)
            logits=self.model(x); score=logits[:,int(class_index)].sum(); score.backward()
            act=captured.get("act")
            if act is None or act.grad is None: return None
            grad=act.grad
            if act.ndim==4:
                m=str(method).lower().replace(" ","")
                if m=="layercam": cam=(torch.relu(grad)*act).sum(dim=1)
                elif m=="hirescam": cam=(grad*act).sum(dim=1)
                else:
                    w=grad.mean(dim=(2,3),keepdim=True); cam=(w*act).sum(dim=1)
                cam=torch.relu(cam)[0]
            elif act.ndim==3:
                # ViT LayerCAM-style patch-token adaptation (NOT a CNN feature map):
                # exclude CLS if present, score patch activations using patch-wise
                # gradients from the pneumonia logit, reshape to square patch grid.
                # The downstream CAM pipeline upsamples this grid to CXR pixels.
                a=act[0]; g=grad[0]; n=a.shape[0]
                if int(round((n-1)**0.5))**2==n-1: a=a[1:]; g=g[1:]; n=n-1
                side=int(round(n**0.5))
                if side*side!=n: return None
                m=str(method).lower().replace(" ","")
                if m=="layercam": token=(torch.relu(g)*a).sum(dim=1)
                elif m=="hirescam": token=(g*a).sum(dim=1)
                else: token=(g.mean(dim=0,keepdim=True)*a).sum(dim=1)
                cam=torch.relu(token.reshape(side,side))
            else: return None
            arr=cam.detach().cpu().numpy().astype(np.float32)
            mx=float(np.nanmax(arr)) if arr.size else 0.0
            return arr/mx if np.isfinite(mx) and mx>1e-8 else None
        finally:
            handle.remove()


def _import_module_from_path(path, name):
    path=Path(path).resolve(); spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


def load_torch_cam_adapter(model_name, model_path, args):
    import torch
    device="cuda" if bool(args.gpu40) and torch.cuda.is_available() else "cpu"
    if model_name=="xrv":
        import torchxrayvision as xrv
        mod=_import_module_from_path(args.xrv_script,"aura_xrv_xai")
        ck=torch.load(model_path,map_location="cpu"); dropout=float(ck.get("dropout",0.3))
        model=mod.build_model(xrv,torch,dropout=dropout).to(device); model.load_state_dict(ck.get("state_dict",ck)); model.eval()
        named=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Conv2d)]
        chosen=dict(named[-5:])
        def transform(arr):
            gray=arr.mean(axis=-1)
            outs=[]
            for g in gray:
                g=g-g.min(); g=g/g.max() if g.max()>0 else np.zeros_like(g)
                z=xrv.datasets.normalize((g*255).astype(np.float32),255)
                outs.append(z[None])
            return torch.from_numpy(np.stack(outs).astype(np.float32))
        return TorchCamAdapter("xrv",model,device,transform,chosen,"conv")
    if model_name=="eva_x":
        mod=_import_module_from_path(args.eva_x_script,"aura_eva_xai")
        ck=torch.load(model_path,map_location="cpu")
        repo=args.eva_x_repo
        if not repo:
            mf=Path(args.results_dir)/"reports"/"eva_x_development_refit_manifest.json"
            if mf.exists(): repo=json.loads(mf.read_text()).get("official_repo","")
        if not repo: raise ValueError("Selected EVA-X XAI requires --eva_x_repo or refit manifest official_repo")
        checkpoint=args.eva_x_checkpoint or ck.get("checkpoint","")
        if not checkpoint: raise ValueError("EVA-X initialization checkpoint unavailable")
        official=mod.import_eva_x(Path(repo)); args.pretrained_checkpoint=checkpoint; args.dropout=float(ck.get("dropout",0.2))
        model=mod.build_model(args,official,torch).to(device); model.load_state_dict(ck.get("state_dict",ck)); model.eval()
        blocks=[]
        for n,m in model.named_modules():
            if "block" in n.lower() and len(list(m.children()))>0: blocks.append((n,m))
        if not blocks: blocks=[(n,m) for n,m in model.named_modules() if n and len(list(m.children()))>0]
        chosen=dict(blocks[-5:])
        mean=np.asarray(args.eva_mean,dtype=np.float32)[None,:,None,None]; std=np.asarray(args.eva_std,dtype=np.float32)[None,:,None,None]
        def transform(arr):
            nchw=np.transpose(arr,(0,3,1,2)).astype(np.float32); return torch.from_numpy((nchw-mean)/std)
        return TorchCamAdapter("eva_x",model,device,transform,chosen,"tokens")
    raise ValueError(model_name)

# -----------------------------------------------------------------------------
# GradCAM core — fixed layer, same-layer contrastive, logit target
# -----------------------------------------------------------------------------

def _iter_conv_layers(model):
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    layers = []

    def visit(layer):
        if isinstance(layer, conv_types):
            layers.append(layer)
        elif isinstance(layer, tf.keras.Model):
            for sub in layer.layers:
                visit(sub)

    for layer in model.layers:
        visit(layer)
    seen, out = set(), []
    for layer in layers:
        if id(layer) not in seen:
            out.append(layer)
            seen.add(id(layer))
    return out



def _get_candidate_conv_layers(model):
    if getattr(model,"_is_torch_adapter",False): return list(model.layers)
    layers = _iter_conv_layers(model); spatial_layers=[]
    for layer in layers:
        try:
            shape=layer.output.shape
            if len(shape)==4 and shape[1] is not None and shape[2] is not None and int(shape[1])>=4 and int(shape[2])>=4:
                spatial_layers.append(layer)
        except Exception: continue
    return list(reversed(spatial_layers or layers))




def _find_layer_recursive(model, names):
    if getattr(model,"_is_torch_adapter",False):
        names={str(n).lower() for n in names}
        return next((l for l in model.layers if l.name.lower() in names),None)
    names={n.lower() for n in names}; found=[]
    def visit(layer):
        if layer.name.lower() in names: found.append(layer)
        if isinstance(layer,tf.keras.Model):
            for sub in layer.layers: visit(sub)
    visit(model); return found[0] if found else None



def _get_fixed_gradcam_layer(model, candidate_layers=None):
    """Use a stable last spatial layer, not the previous mean-activation heuristic.

    EfficientNetV2S: top_conv; ResNet50: conv5_block3_out. If the exact layer is
    unavailable, use the deepest candidate conv layer as a deterministic fallback.
    """
    layer = _find_layer_recursive(model, ["top_conv", "conv5_block3_out"])
    if layer is not None:
        return layer
    layers = candidate_layers or _iter_conv_layers(model)
    if not layers:
        raise ValueError("No Conv2D/DepthwiseConv2D layer found for Grad-CAM")
    return layers[-1]


def _logit_tensor_from_softmax_model(model):
    """Return a logits tensor when the final layer is Dense(softmax).

    Proposal Rumus 3.27 uses pre-softmax class score y^c. The training model
    keeps softmax for probabilities, so for Grad-CAM we reconstruct logits from
    the final Dense kernel and its input when possible.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            try:
                activation_name = getattr(layer.activation, "__name__", "")
                if activation_name == "softmax":
                    linear = tf.keras.layers.Dense(layer.units, activation=None, dtype="float32", name=layer.name + "_logits_for_cam")
                    logits = linear(layer.input)
                    linear.set_weights(layer.get_weights())
                    return logits
            except Exception:
                continue
    return None


def _grad_model_for_layer(model, conv_layer):
    logits = _logit_tensor_from_softmax_model(model)
    if logits is not None:
        return tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, logits]), "logit"
    # Fallback: model output may already be logits or a probability. If it is a
    # probability, _gradcam_single applies a stable log transform.
    return tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output]), "model_output"



def _cam_single(model, image_array_f32, conv_layer, class_index, method="GradCAM"):
    """Compute one CAM variant from a fixed layer and a pre-softmax target.

    Supported variants are GradCAM, GradCAM++, LayerCAM, and HiResCAM. All
    variants use the same pre-softmax pneumonia logit whenever the trained head
    permits exact logit reconstruction. This avoids probability saturation.
    """
    if image_array_f32.ndim == 3:
        image_array_f32 = image_array_f32[np.newaxis]

    try:
        grad_model, score_source = _grad_model_for_layer(model, conv_layer)
    except Exception:
        return None

    with tf.GradientTape() as tape:
        inp = tf.cast(image_array_f32, tf.float32)
        conv_out_raw, scores = grad_model(inp, training=False)
        scores = tf.cast(scores, tf.float32)
        if scores.shape.rank == 2 and scores.shape[-1] > class_index:
            class_score = scores[:, class_index]
        else:
            class_score = scores[:, 0]
        if score_source != "logit":
            class_score = tf.math.log(tf.clip_by_value(class_score, 1e-7, 1.0))
        loss = tf.reduce_sum(class_score)

    grads = tape.gradient(loss, conv_out_raw)
    if grads is None:
        return None

    conv_out = tf.cast(conv_out_raw, tf.float32)
    grads = tf.cast(grads, tf.float32)
    bad = tf.reduce_any(tf.math.logical_or(tf.math.is_nan(grads), tf.math.is_inf(grads)))
    if bool(bad.numpy()):
        return None

    method_key = str(method).lower().replace("_", "").replace("-", "")
    activ = conv_out[0]
    grad = grads[0]
    if method_key in {"gradcam++", "gradcampp", "++"}:
        grads2 = tf.square(grads)
        grads3 = grads2 * grads
        sum_activ = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
        alpha = grads2 / (2.0 * grads2 + sum_activ * grads3 + 1e-8)
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))[0]
        heatmap = tf.reduce_sum(activ * weights, axis=-1)
    elif method_key == "layercam":
        heatmap = tf.reduce_sum(activ * tf.nn.relu(grad), axis=-1)
    elif method_key == "hirescam":
        heatmap = tf.reduce_sum(activ * grad, axis=-1)
    else:
        weights = tf.reduce_mean(grad, axis=(0, 1))
        heatmap = tf.reduce_sum(activ * weights, axis=-1)

    heatmap = tf.nn.relu(heatmap).numpy().astype(np.float32)
    hmax = float(np.nanmax(heatmap)) if heatmap.size else 0.0
    if not np.isfinite(hmax) or hmax <= 1e-12:
        return None
    return (heatmap / hmax).astype(np.float32)


def _gradcam_single(model, image_array_f32, conv_layer, class_index, use_plusplus=False):
    """Backward-compatible wrapper retained for archived calls."""
    method = "GradCAM++" if use_plusplus else "GradCAM"
    return _cam_single(model, image_array_f32, conv_layer, class_index, method=method)


def compute_gradcam_robust(
    model,
    image_array_f32,
    candidate_layers=None,
    class_index=1,
    debug=False,
    fixed_layer=None,
    force_method=None,
    allow_method_fallback=True,
    allow_layer_fallback=True,
    max_fallback_layers=10,
    warn_on_failure=True,
):
    """Deterministic, failure-tolerant Grad-CAM.

    Preferred behavior remains proposal-aligned: use the fixed deep spatial
    layer (EfficientNetV2S ``top_conv`` or ResNet50 ``conv5_block3_out``).
    If the preferred method returns a zero/invalid map, the function tries the
    alternate CAM formulation and then a small deterministic list of deeper
    spatial layers. It never selects a layer by mean activation, avoiding
    sample-dependent cherry-picking.

    ``force_method`` is treated as the preferred method. Set
    ``allow_method_fallback=False`` when the exact same method is required for
    contrastive subtraction.
    """
    if getattr(model, "_is_torch_adapter", False):
        preferred_layer = fixed_layer or (_get_candidate_conv_layers(model)[0])
        layers = [preferred_layer]
        if allow_layer_fallback and fixed_layer is None:
            layers += [l for l in _get_candidate_conv_layers(model) if l.name != preferred_layer.name][:max(int(max_fallback_layers)-1,0)]
        requested = str(force_method or "GradCAM")
        methods = [requested] if requested.lower() not in {"auto","validation_selected","none"} else ["GradCAM","LayerCAM","HiResCAM"]
        if allow_method_fallback:
            methods += [m for m in ["GradCAM","LayerCAM","HiResCAM"] if m not in methods]
        for layer in layers:
            for method in methods:
                try:
                    hm=model.cam_from_rgb(np.asarray(image_array_f32,dtype=np.float32),layer.name,class_index,method)
                except Exception as exc:
                    hm=None
                    if debug: print(f"[TORCH-CAM][WARN] {model.name}/{layer.name}/{method}: {exc}")
                if hm is not None and compute_heatmap_audit(hm)["heatmap_valid"]:
                    return hm.astype(np.float32),layer.name,method
        return np.zeros((14,14),dtype=np.float32),preferred_layer.name,"FAILED"

    cache_key = (id(model), int(class_index), str(force_method or "auto"))
    cached = _GRADCAM_RUNTIME_CACHE.get(cache_key, {}) if fixed_layer is None else {}

    preferred_layer = fixed_layer or _get_fixed_gradcam_layer(model, candidate_layers)
    cached_layer_name = cached.get("layer_name")
    if fixed_layer is None and cached_layer_name:
        cached_layer = _find_layer_recursive(model, [cached_layer_name])
        if cached_layer is not None:
            preferred_layer = cached_layer

    layer_candidates = [preferred_layer]
    if allow_layer_fallback and fixed_layer is None:
        fallback = candidate_layers or _get_candidate_conv_layers(model)
        for layer in fallback:
            if id(layer) != id(preferred_layer) and layer not in layer_candidates:
                layer_candidates.append(layer)
            if len(layer_candidates) >= max(int(max_fallback_layers), 1):
                break

    supported_methods = ["GradCAM", "GradCAM++", "LayerCAM", "HiResCAM"]
    stable_fallback_methods = ["GradCAM", "LayerCAM", "HiResCAM"]
    requested = str(force_method or "auto")
    if requested.lower() in {"auto", "validation_selected", "none"}:
        # GradCAM++ is intentionally not an automatic fallback. In this project it
        # generated systematic empty maps for several EfficientNetV2 layers.
        method_candidates = stable_fallback_methods.copy()
    else:
        canonical = next((m for m in supported_methods if m.lower() == requested.lower()), requested)
        method_candidates = [canonical]
        if allow_method_fallback:
            method_candidates.extend([m for m in stable_fallback_methods if m != canonical])

    cached_method = cached.get("method")
    if cached_method:
        method_candidates = sorted(
            method_candidates, key=lambda item: 0 if item == cached_method else 1
        )

    failures = []
    for layer in layer_candidates:
        for method in method_candidates:
            try:
                hm = _cam_single(
                    model, image_array_f32, layer, class_index, method=method
                )
            except Exception as exc:
                hm = None
                failures.append(f"{layer.name}/{method}: {type(exc).__name__}: {exc}")
            if hm is None:
                failures.append(f"{layer.name}/{method}: empty_or_invalid")
                continue
            peak = float(np.nanmax(hm)) if hm.size else 0.0
            if not np.isfinite(peak) or peak <= ACTIVATION_THRESHOLD:
                failures.append(f"{layer.name}/{method}: peak={peak}")
                continue
            if debug:
                peakiness = float(peak / (np.nanmean(hm) + 1e-8))
                fallback_tag = "" if id(layer) == id(preferred_layer) else " fallback_layer"
                print(
                    f"    [{method}] class={class_index} layer={layer.name}{fallback_tag} "
                    f"peakiness={peakiness:.3f} mean={hm.mean():.6f}"
                )
            if fixed_layer is None:
                _GRADCAM_RUNTIME_CACHE[cache_key] = {
                    "layer_name": layer.name,
                    "method": method,
                }
            return hm.astype(np.float32), layer.name, method

    detail = failures[-4:] if failures else ["unknown failure"]
    if warn_on_failure:
        warning_key = (
            id(model), str(preferred_layer.name), int(class_index),
            str(force_method or "auto"),
        )
        seen = int(_CAM_FAILURE_WARNING_COUNTS.get(warning_key, 0))
        if seen == 0:
            print(
                f"  [WARN] CAM failed after deterministic fallbacks; "
                f"preferred_layer={preferred_layer.name}, class={class_index}, "
                f"requested_method={force_method or 'auto'}, last_attempts={detail}. "
                "Returning an invalid zero placeholder; localization metrics will exclude it."
            )
        elif seen == 1:
            print(
                f"  [WARN] Further CAM-failure messages suppressed for "
                f"layer={preferred_layer.name}, class={class_index}, "
                f"method={force_method or 'auto'}."
            )
        _CAM_FAILURE_WARNING_COUNTS[warning_key] = seen + 1
    return np.zeros((7, 7), dtype=np.float32), preferred_layer.name, "FAILED"


def compute_contrastive_gradcam_robust(model, image_array_f32, candidate_layers=None, debug=False):
    """Contrastive Grad-CAM using the same deterministic layer and method.

    The pneumonia map may use a deterministic layer/method fallback. Once a
    valid combination is found, the non-pneumonia map is forced to use that
    exact combination so the subtraction remains methodologically valid.
    """
    hm_pneu, layer_name, method = compute_gradcam_robust(
        model,
        image_array_f32,
        candidate_layers,
        class_index=1,
        debug=debug,
        fixed_layer=None,
        allow_method_fallback=True,
        allow_layer_fallback=True,
    )

    used_layer = _find_layer_recursive(model, [layer_name])
    if used_layer is None:
        used_layer = _get_fixed_gradcam_layer(model, candidate_layers)

    contrastive_valid = method != "FAILED"
    if method == "FAILED":
        hm_npneu = np.zeros_like(hm_pneu, dtype=np.float32)
    else:
        hm_npneu, _, method_np = compute_gradcam_robust(
            model,
            image_array_f32,
            candidate_layers,
            class_index=0,
            debug=debug,
            fixed_layer=used_layer,
            force_method=method,
            allow_method_fallback=False,
            allow_layer_fallback=False,
        )
        if method_np == "FAILED":
            # A contrastive map is not meaningful when only one class CAM is
            # available. Preserve the pneumonia CAM for qualitative review and
            # return a zero contrast map instead of subtracting invalid data.
            hm_npneu = np.zeros_like(hm_pneu, dtype=np.float32)
            contrastive_valid = False

    hm_pneu_n = (hm_pneu / (hm_pneu.max() + 1e-8)).astype(np.float32)
    hm_npneu_n = (hm_npneu / (hm_npneu.max() + 1e-8)).astype(np.float32)
    if contrastive_valid:
        contrastive = np.clip(hm_pneu_n - hm_npneu_n, 0, 1).astype(np.float32)
    else:
        contrastive = np.zeros_like(hm_pneu_n, dtype=np.float32)
    return hm_pneu_n, hm_npneu_n, contrastive, layer_name, method



def find_radiomic_cache(results_dir, wavelet_levels=3, allow_legacy=False):
    """Find proposal-aligned radiomic cache created by train_aura.py.

    Default behavior deliberately refuses legacy adaptive_wavelet_features.pkl
    to prevent silently evaluating with old 4L-subband feature dimensions.
    """
    cache_dir = Path(results_dir) / "cache"
    expected = cache_dir / f"radiomic_v2_adaptive_L{int(wavelet_levels)}_glcm_lbp.joblib"
    candidates = []
    if expected.exists():
        candidates.append(expected)
    candidates.extend(sorted(cache_dir.glob("radiomic_v2_*.joblib")))
    candidates.extend(sorted(cache_dir.glob("radiomic_v2_*.pkl")))
    # Deduplicate while preserving priority.
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    if candidates:
        return candidates[0]
    legacy = cache_dir / "adaptive_wavelet_features.pkl"
    if allow_legacy and legacy.exists():
        print(f"[WARN] Legacy radiomic cache explicitly allowed: {legacy}")
        return legacy
    raise FileNotFoundError(
        f"[ERROR] Proposal-aligned radiomic_v2 cache not found in {cache_dir}. "
        f"Expected {expected.name}. Run `python train_aura.py --stage features` first. "
        "Legacy adaptive_wavelet_features.pkl is blocked by default; pass "
        "--allow_legacy_radiomic_cache only for archived/non-final experiments."
    )


def expected_radiomic_dim(wavelet_levels=3):
    # Proposal-aligned full feature set: (3L+1) subbands × 6 GLCM props × 3 distances × 4 angles + 26 LBP bins.
    return (3 * int(wavelet_levels) + 1) * 6 * 3 * 4 + 26


def load_proposal_radiomic_cache(results_dir, wavelet_levels=3, allow_legacy=False):
    cache_path = find_radiomic_cache(results_dir, wavelet_levels=wavelet_levels, allow_legacy=allow_legacy)
    cache = joblib.load(cache_path)
    if not isinstance(cache, dict) or not cache:
        raise ValueError(f"[ERROR] Radiomic cache is empty or unsupported: {cache_path}")
    first = next(iter(cache.values()))
    got = int(np.asarray(first).shape[0])
    exp = expected_radiomic_dim(wavelet_levels)
    if int(wavelet_levels) == 3 and got != 746:
        raise ValueError(
            f"Radiomic feature dimension mismatch: expected 746, got {got}. "
            "Do not use legacy adaptive_wavelet_features.pkl for proposal-aligned evaluation."
        )
    if got != exp:
        raise ValueError(f"Radiomic feature dimension mismatch: expected {exp}, got {got} for L={wavelet_levels}.")
    print(f"[INFO] Loaded proposal-aligned radiomic cache: {cache_path}")
    print(f"[INFO] Radiomic feature dimension: {got}")
    return cache, cache_path


def apply_lung_mask_to_heatmap(heatmap, patient_id, lung_mask_dir=None,
                               use_lung_mask=False, allow_ellipse=False):
    if not use_lung_mask:
        return np.asarray(heatmap, dtype=np.float32), False
    mask, _mode = load_lung_mask_for_patient(
        patient_id, lung_mask_dir, np.asarray(heatmap).shape[0],
        allow_ellipse=allow_ellipse,
    )
    if mask is None:
        return np.asarray(heatmap, dtype=np.float32), False
    hm = np.asarray(heatmap, dtype=np.float32)
    gated = hm * mask.astype(np.float32)
    mx = float(np.nanmax(gated)) if gated.size else 0.0
    if not np.isfinite(mx) or mx <= 1e-8:
        return np.zeros_like(hm, dtype=np.float32), True
    return (gated / mx).astype(np.float32), True


def _ellipse_fallback_lung_mask(image_size=224):
    """Deterministic anatomical prior used only when explicitly permitted."""
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    left = (((xx - image_size * 0.38) / (image_size * 0.20)) ** 2 + ((yy - image_size * 0.50) / (image_size * 0.34)) ** 2) <= 1
    right = (((xx - image_size * 0.62) / (image_size * 0.20)) ** 2 + ((yy - image_size * 0.50) / (image_size * 0.34)) ** 2) <= 1
    return (left | right).astype(bool)



def load_lung_mask_for_patient(patient_id, lung_mask_dir, image_size,
                               allow_ellipse=False):
    """Load, post-process, validate, and identify one lung mask."""
    mask = None
    mode = "missing"
    if lung_mask_dir is not None:
        path = Path(lung_mask_dir) / f"{patient_id}.npy"
        meta_path = Path(lung_mask_dir) / f"{patient_id}.json"
        if path.exists():
            try:
                mask = postprocess_lung_mask(
                    np.load(path), (int(image_size), int(image_size))
                )
                source = "pretrained_mask"
                if meta_path.exists():
                    source_raw = str(json.loads(meta_path.read_text()).get("source", "pretrained"))
                    source = {
                        "pretrained": "pretrained_mask",
                        "ellipse": "ellipse_prior",
                        "ellipse_fallback": "ellipse_fallback",
                    }.get(source_raw, source_raw)
                mode = source if mask is not None else f"invalid_{source}"
            except Exception:
                mask = None
                mode = "invalid_cached_mask"
    if mask is None and allow_ellipse:
        mask = postprocess_lung_mask(
            _ellipse_fallback_lung_mask(int(image_size)),
            (int(image_size), int(image_size)),
        )
        mode = "ellipse_runtime_fallback" if mask is not None else "invalid_ellipse_prior"
    return mask, mode

def gate_heatmap_to_lungs(heatmap, patient_id, lung_mask_dir, image_size,
                          allow_ellipse=False):
    """Strictly zero attribution outside the validated lung field.

    The function never falls back to the ungated heatmap after a valid mask has
    removed all activation. Such a case is returned as a zero map and later
    marked invalid by quality control.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    mask, mode = load_lung_mask_for_patient(
        patient_id, lung_mask_dir, int(image_size), allow_ellipse=allow_ellipse
    )
    if mask is None:
        return hm, "none", None
    if mask.shape != hm.shape:
        mask = resize_binary_mask(mask, hm.shape[0], hm.shape[1])
    gated = hm * mask.astype(np.float32)
    mx = float(np.nanmax(gated)) if gated.size else 0.0
    if not np.isfinite(mx) or mx <= 1e-8:
        return np.zeros_like(hm, dtype=np.float32), f"{mode}_zero_after_gate", mask
    return (gated / mx).astype(np.float32), mode, mask



def _resolution_aware_gradcam_layers(model, min_grid=16, max_layers=5, allow_depthwise=False):
    if getattr(model,"_is_torch_adapter",False): return list(reversed(model.layers))[:max(1,int(max_layers))]
    selected=[]; all_candidates=_get_candidate_conv_layers(model)
    for layer in all_candidates:
        if isinstance(layer,tf.keras.layers.DepthwiseConv2D) and not allow_depthwise: continue
        try: h,w=int(layer.output.shape[1]),int(layer.output.shape[2])
        except Exception: continue
        if h>=int(min_grid) and w>=int(min_grid): selected.append(layer)
        if len(selected)>=int(max_layers): break
    if not selected:
        regular=[l for l in all_candidates if not isinstance(l,tf.keras.layers.DepthwiseConv2D)]
        selected=[regular[0] if regular else _get_fixed_gradcam_layer(model,all_candidates)]
    return selected



def _model_cam_key(model):
    return str(getattr(model, "_aura_cam_name", getattr(model, "name", "cnn")))


def _locked_cam_layer(model, candidate_layers=None, min_grid=16):
    """Resolve the validation-locked layer; architecture-only fallback if absent."""
    key = _model_cam_key(model)
    locked_name = _CAM_LAYER_LOCK.get(key)
    if locked_name:
        layer = _find_layer_recursive(model, [locked_name])
        if layer is not None:
            return layer, "validation_manifest"
    layers = _resolution_aware_gradcam_layers(model, min_grid=min_grid, max_layers=1)
    return layers[0], "architecture_fallback"



def _parse_csv_names(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        vals = [str(v).strip() for v in value if str(v).strip()]
    else:
        vals = [v.strip() for v in str(value).split(",") if v.strip()]
    return vals or list(default)


def _parse_csv_floats(value, default):
    vals = []
    for token in _parse_csv_names(value, [str(v) for v in default]):
        try:
            vals.append(float(token))
        except Exception:
            continue
    vals = sorted(set(v for v in vals if np.isfinite(v)))
    return vals or list(default)


def _apply_lung_constraint_policy(heatmap, lung_mask, policy="raw", outside_weight=0.20):
    hm = normalize_minmax(np.asarray(heatmap, dtype=np.float32))
    if lung_mask is None or str(policy).lower() == "raw":
        return hm
    lm = np.asarray(lung_mask, dtype=bool)
    if lm.shape != hm.shape:
        lm = resize_binary_mask(lm, hm.shape[0], hm.shape[1])
    policy_l = str(policy).lower()
    if policy_l == "soft":
        multiplier = np.where(lm, 1.0, float(np.clip(outside_weight, 0.0, 1.0))).astype(np.float32)
        constrained = hm * multiplier
    else:
        constrained = hm * lm.astype(np.float32)
    mx = float(np.nanmax(constrained)) if constrained.size else 0.0
    return (constrained / mx).astype(np.float32) if np.isfinite(mx) and mx > 1e-8 else np.zeros_like(hm)


def _cosine_map_similarity(a, b):
    if a is None or b is None:
        return np.nan
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    x, y = x[ok], y[ok]
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / den) if den > 1e-12 else np.nan


def _localization_quality(met):
    """Composite localization quality used only for validation selection."""
    def v(name, default=0.0):
        x = met.get(name, default) if isinstance(met, dict) else default
        return float(x) if np.isfinite(x) else float(default)
    return float(
        0.30 * v("pointing_hit")
        + 0.20 * v("energy_inside_gt")
        + 0.25 * v("localization_score")
        + 0.15 * v("iou_at_thr")
        + 0.10 * v("gt_coverage_at_thr")
    )


def _validation_objective(summary):
    """Robust validation objective rewarding lesion specificity over strong priors."""
    def v(name, default=0.0):
        x = summary.get(name, default)
        return float(x) if np.isfinite(x) else float(default)
    delta = v("strong_baseline_quality_delta_mean")
    delta_ci = v("strong_baseline_quality_delta_ci95_low", -1.0)
    fold_min = v("strong_baseline_quality_delta_fold_min")
    fold_std = v("strong_baseline_quality_delta_fold_std")
    return float(
        0.18 * v("pointing_hit_mean")
        + 0.13 * v("energy_inside_gt_mean")
        + 0.14 * v("localization_score_mean")
        + 0.10 * v("iou_at_thr_mean")
        + 0.05 * v("gt_coverage_at_thr_mean")
        + 0.35 * max(delta, 0.0)
        + 0.20 * max(delta_ci, -0.25)
        - 0.50 * max(-delta, 0.0)
        - 0.15 * max(-fold_min, 0.0)
        - 0.08 * fold_std
        - 0.07 * v("outside_lung_ratio_raw_mean")
        - 0.05 * v("image_center_similarity_mean")
        - 0.07 * v("lung_prior_similarity_mean")
        - 0.10 * v("lesion_prior_similarity_mean")
        - 0.20 * v("invalid_rate")
    )


def _evaluate_validation_maps(map_records, policies, thresholds, outside_weight,
                              metric_size=96, n_boot=500, seed=42):
    """Evaluate fixed maps against GT and all prespecified strong priors.

    This routine never accesses test labels.  It records per-case superiority over
    image-center, lung-anatomical, and training lesion-prevalence priors; a paired
    bootstrap lower bound and fold stability are used in the selection objective.
    """
    rows = []
    for policy in policies:
        for threshold in thresholds:
            values = {
                "pointing_hit": [], "energy_inside_gt": [], "localization_score": [],
                "iou_at_thr": [], "gt_coverage_at_thr": [], "baseline_delta": [],
                "strong_baseline_quality_delta": [], "outside_lung_ratio_raw": [],
                "image_center_similarity": [], "lung_prior_similarity": [],
                "lesion_prior_similarity": [],
            }
            fold_deltas = {}
            invalid = 0
            for rec_i, rec in enumerate(map_records):
                raw_map = rec.get("map")
                if raw_map is None:
                    invalid += 1
                    continue
                original_size = int(np.asarray(raw_map).shape[0])
                selected = _apply_lung_constraint_policy(
                    raw_map, rec.get("lung_mask"), policy=policy,
                    outside_weight=outside_weight,
                )
                lung_mask_small = rec.get("lung_mask")
                if int(metric_size) > 0 and original_size != int(metric_size):
                    selected = resize_heatmap(selected, int(metric_size), int(metric_size))
                    lung_mask_small = resize_binary_mask(lung_mask_small, int(metric_size), int(metric_size)) if lung_mask_small is not None else None
                    scale = float(metric_size) / float(original_size)
                    boxes_eval = [(x*scale, y*scale, w*scale, h*scale) for x,y,w,h in rec["gt_boxes"]]
                else:
                    boxes_eval = rec["gt_boxes"]
                audit = compute_heatmap_audit(selected, lung_mask_small)
                if not audit["heatmap_valid"]:
                    invalid += 1
                    continue
                met = compute_localization_metrics(
                    selected, boxes_eval, threshold_pct=float(threshold),
                    heatmap_valid=True,
                )

                baseline_metrics = []
                baseline_maps = {
                    "image_center": rec.get("center_map"),
                    "lung_prior": rec.get("lung_prior"),
                    "lesion_prior": rec.get("lesion_prior"),
                }
                processed_baselines = {}
                for base_name, base_map in baseline_maps.items():
                    if base_map is None:
                        continue
                    base_selected = _apply_lung_constraint_policy(
                        base_map, rec.get("lung_mask"), policy=policy,
                        outside_weight=outside_weight,
                    )
                    if int(metric_size) > 0 and base_selected.shape[0] != int(metric_size):
                        base_selected = resize_heatmap(base_selected, int(metric_size), int(metric_size))
                    base_met = compute_localization_metrics(
                        base_selected, boxes_eval, threshold_pct=float(threshold),
                        heatmap_valid=True,
                    )
                    processed_baselines[base_name] = base_selected
                    baseline_metrics.append(base_met)

                for key in ["pointing_hit", "energy_inside_gt", "localization_score", "iou_at_thr", "gt_coverage_at_thr"]:
                    values[key].append(met.get(key, np.nan))
                best_prior_loc = max(
                    [float(b.get("localization_score", -np.inf)) for b in baseline_metrics]
                    or [np.nan]
                )
                values["baseline_delta"].append(met.get("localization_score", np.nan) - best_prior_loc)
                model_quality = _localization_quality(met)
                best_baseline_quality = max([_localization_quality(b) for b in baseline_metrics] or [0.0])
                quality_delta = float(model_quality - best_baseline_quality)
                values["strong_baseline_quality_delta"].append(quality_delta)
                fold_id = int(rec.get("fold_id", rec_i % 5))
                fold_deltas.setdefault(fold_id, []).append(quality_delta)

                raw_audit = compute_heatmap_audit(raw_map, rec.get("lung_mask"))
                values["outside_lung_ratio_raw"].append(raw_audit.get("outside_lung_ratio", np.nan))
                values["image_center_similarity"].append(
                    _cosine_map_similarity(selected, processed_baselines.get("image_center"))
                )
                values["lung_prior_similarity"].append(
                    _cosine_map_similarity(selected, processed_baselines.get("lung_prior"))
                )
                values["lesion_prior_similarity"].append(
                    _cosine_map_similarity(selected, processed_baselines.get("lesion_prior"))
                )

            summary = {
                "map_policy": str(policy),
                "heatmap_threshold_pct": float(threshold),
                "n_attempted": int(len(map_records)),
                "n_valid": int(len(map_records) - invalid),
                "invalid_rate": float(invalid / max(len(map_records), 1)),
            }
            for key, vals in values.items():
                s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
                summary[f"{key}_mean"] = float(s.mean()) if len(s) else np.nan
            deltas = pd.to_numeric(pd.Series(values["strong_baseline_quality_delta"]), errors="coerce").dropna()
            lo, hi = _bootstrap_stat_ci(
                deltas, stat="mean", n_boot=max(100, int(n_boot)), seed=int(seed)
            )
            fold_means = [float(np.mean(v)) for v in fold_deltas.values() if len(v)]
            summary.update({
                "strong_baseline_quality_delta_ci95_low": lo,
                "strong_baseline_quality_delta_ci95_high": hi,
                "strong_baseline_quality_delta_fold_mean": float(np.mean(fold_means)) if fold_means else np.nan,
                "strong_baseline_quality_delta_fold_min": float(np.min(fold_means)) if fold_means else np.nan,
                "strong_baseline_quality_delta_fold_std": float(np.std(fold_means)) if fold_means else np.nan,
                "strong_baseline_mean_superiority": bool(len(deltas) and float(deltas.mean()) > 0),
                "strong_baseline_ci_superiority": bool(np.isfinite(lo) and lo > 0),
            })
            summary["selection_objective"] = _validation_objective(summary)
            rows.append(summary)
    return rows


def _compute_model_cam_from_config(model, preprocess_fn, candidate_layers, raw_rgb,
                                   image_size, model_config, allow_fallback=True,
                                   warn_on_failure=True):
    if getattr(model,"_is_torch_adapter",False):
        preproc=np.asarray(raw_rgb[np.newaxis],dtype=np.float32)
    else:
        preproc=preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32))).numpy()
    layers=list(model_config.get("layers") or [model_config.get("layer")]); methods=list(model_config.get("methods") or [model_config.get("method","GradCAM")])
    weights=np.asarray(model_config.get("layer_weights") or [1.0]*len(layers),dtype=float)
    if len(methods)==1 and len(layers)>1: methods=methods*len(layers)
    if len(weights)!=len(layers) or weights.sum()<=0: weights=np.ones(len(layers),dtype=float)
    weights=weights/weights.sum(); maps=[];details=[];used_weights=[]
    for layer_name,method,weight in zip(layers,methods,weights):
        layer=_find_layer_recursive(model,[str(layer_name)])
        if layer is None: details.append({"status":"layer_missing","layer":layer_name,"method":method});continue
        hm,used_layer,used_method=compute_gradcam_robust(model,preproc,candidate_layers,class_index=1,fixed_layer=layer,
            force_method=method,allow_method_fallback=bool(allow_fallback),allow_layer_fallback=False,warn_on_failure=bool(warn_on_failure))
        audit=compute_heatmap_audit(hm)
        if used_method=="FAILED" or not audit["heatmap_valid"]: details.append({"status":"cam_failed","layer":layer_name,"method":method});continue
        hm=normalize_minmax(resize_heatmap(hm,image_size,image_size));maps.append(hm);used_weights.append(float(weight))
        details.append({"status":"ok","layer":used_layer,"method":used_method,"configured_method":method,
                        "configured_weight":float(weight),"fallback_used":bool(used_method!=method)})
    if not maps:return None,details
    w=np.asarray(used_weights,dtype=float);w=w/w.sum() if w.sum()>0 else np.ones(len(maps))/len(maps)
    return normalize_minmax(np.tensordot(w,np.stack(maps),axes=(0,0)).astype(np.float32)),details



def configure_cam_layer_lock(
    models_pp_layers,
    val_df,
    y_val,
    gt_box_map,
    image_size,
    reports_dir,
    lung_mask_dir=None,
    use_lung_mask=True,
    allow_ellipse=False,
    policy="validation_fixed",
    val_samples=64,
    candidate_count=5,
    force_method="validation_selected",
    method_candidates="GradCAM,LayerCAM,HiResCAM",
    map_policy_candidates="raw,soft",
    candidate_pilot_samples=12,
    candidate_pilot_min_valid_rate=0.80,
    candidate_min_valid_rate=0.95,
    allow_depthwise_candidates=False,
    threshold_candidates="0.25,0.35,0.45,0.55,0.65,0.75",
    model_weight_grid="0.25,0.50,0.75",
    cam_min_model_weight=0.25,
    require_all_selected_cnn_contributions=True,
    soft_lung_outside_weight=0.20,
    selection_metric_size=96,
    lesion_prevalence_map=None,
    validation_folds=5,
    selection_bootstrap=500,
    validation_confirm_all=True,
    allow_ellipse_in_validation=False,
    min_grid=16,
    heatmap_threshold_pct=0.60,
    force_reselect=False,
    seed=42,
):
    """Joint validation-only selection of the complete primary CAM configuration.

    The test set is never used to choose method, layer, layer fusion, CNN fusion,
    lung constraint, or heatmap threshold. The selected manifest is immutable for
    test evaluation and invalidates old resume caches through XAI_SCHEMA_VERSION.
    """
    global _CAM_LAYER_LOCK, _CAM_LAYER_SELECTION_AUDIT, _CAM_CONFIG_LOCK
    reports_dir = ensure_dir(reports_dir)
    manifest_path = reports_dir / "xai_cam_configuration_validation.json"
    legacy_path = reports_dir / "xai_cam_layer_selection_validation.json"
    methods = _parse_csv_names(method_candidates, ["GradCAM", "LayerCAM", "HiResCAM"])
    if str(force_method).lower() not in {"validation_selected", "auto", "none"}:
        methods = [str(force_method)]
    policies = _parse_csv_names(map_policy_candidates, ["raw", "soft", "hard"])
    thresholds = _parse_csv_floats(threshold_candidates, [heatmap_threshold_pct])
    min_model_weight = float(cam_min_model_weight)
    if not (0.0 <= min_model_weight < 0.5):
        raise ValueError(
            f"cam_min_model_weight must satisfy 0 <= value < 0.5; got {min_model_weight}."
        )
    raw_weight_grid = _parse_csv_floats(model_weight_grid, [0.25, 0.50, 0.75])
    weight_grid = sorted({
        float(w) for w in raw_weight_grid
        if min_model_weight <= float(w) <= (1.0 - min_model_weight)
    })
    config = {
        "schema": XAI_SCHEMA_VERSION,
        "policy": str(policy),
        "image_size": int(image_size),
        "val_samples_requested": int(val_samples),
        "candidate_count": int(candidate_count),
        "cam_method_candidates": methods,
        "map_policy_candidates": policies,
        "candidate_pilot_samples": int(candidate_pilot_samples),
        "candidate_pilot_min_valid_rate": float(candidate_pilot_min_valid_rate),
        "candidate_min_valid_rate": float(candidate_min_valid_rate),
        "allow_depthwise_candidates": bool(allow_depthwise_candidates),
        "threshold_candidates": thresholds,
        "model_weight_grid": weight_grid,
        "cam_min_model_weight": min_model_weight,
        "require_all_selected_cnn_contributions": bool(require_all_selected_cnn_contributions),
        "fusion_constraint": (
            "all_selected_cnn_positive_contribution"
            if bool(require_all_selected_cnn_contributions)
            else "partial_fusion_allowed"
        ),
        "soft_lung_outside_weight": float(soft_lung_outside_weight),
        "selection_metric_size": int(selection_metric_size),
        "validation_folds": int(validation_folds),
        "selection_bootstrap": int(selection_bootstrap),
        "validation_confirm_all": bool(validation_confirm_all),
        "allow_ellipse_in_validation": bool(allow_ellipse_in_validation),
        "lesion_prevalence_prior_enabled": bool(lesion_prevalence_map is not None),
        "cam_min_grid": int(min_grid),
        "selection_split": "validation",
        "test_set_used_for_selection": False,
        "deployment_lock_id": str(_XAI_DEPLOYMENT_LOCK_ID),
        "model_checkpoint_sha256": dict(_XAI_MODEL_HASHES),
        "seed": int(seed),
    }
    if manifest_path.exists() and not force_reselect:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            comparable = all(manifest.get(k) == v for k, v in config.items())
            if comparable and manifest.get("selected_models"):
                _CAM_CONFIG_LOCK = manifest
                _CAM_LAYER_LOCK = dict(manifest.get("selected_layers", {}))
                _CAM_LAYER_SELECTION_AUDIT = manifest
                print(f"[CAM-CONFIG] Reused validation manifest: {manifest_path}")
                return manifest
        except Exception as exc:
            print(f"[CAM-CONFIG] Existing manifest incompatible ({exc}); rebuilding.")

    positive_indices = []
    for idx in range(len(val_df)):
        if int(y_val[idx]) != 1:
            continue
        row = val_df.iloc[idx]
        boxes = scale_boxes_to_image(str(row["patientId"]), row["image_path"], gt_box_map, image_size)
        if boxes:
            positive_indices.append(idx)
    if not positive_indices:
        raise RuntimeError("No GT-positive validation cases with boxes for CAM configuration selection.")
    rng = np.random.default_rng(int(seed))
    all_positive_indices = list(rng.permutation(positive_indices))
    if int(val_samples) <= 0 or int(val_samples) >= len(all_positive_indices):
        selection_indices = list(all_positive_indices)
    else:
        selection_indices = list(all_positive_indices[:int(val_samples)])
    center_map = _center_bias_heatmap(image_size)
    cases = []
    for case_order, idx in enumerate(selection_indices):
        row = val_df.iloc[int(idx)]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(row["image_path"], image_size)
        mask, mask_mode = load_lung_mask_for_patient(
            pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
        ) if use_lung_mask else (None, "off")
        if (not allow_ellipse_in_validation) and "ellipse" in str(mask_mode).lower():
            continue
        cases.append({
            "idx": int(idx), "pid": pid, "raw": raw,
            "raw_rgb": np.repeat(raw[..., None], 3, axis=-1),
            "gt_boxes": scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size),
            "lung_mask": mask, "lung_mask_mode": mask_mode,
            "center_map": center_map,
            "lung_prior": _lung_prior_heatmap(mask),
            "lesion_prior": lesion_prevalence_map,
            "fold_id": int(len(cases) % max(int(validation_folds), 1)),
        })
    print(
        f"[CAM-CONFIG] Joint validation selection: n={len(cases)}/{len(all_positive_indices)}, "
        f"methods={methods}, policies={policies}, thresholds={thresholds}"
    )

    selected_models, selected_layers, model_audits = {}, {}, {}
    validation_model_maps = {}
    for model, preprocess_fn, cand in models_pp_layers:
        model_name = _model_cam_key(model)
        preprocessed = {}
        for case in cases:
            preprocessed[case["idx"]] = preprocess_fn(
                tf.constant(case["raw_rgb"][np.newaxis].astype(np.float32))
            ).numpy()
        candidate_layers = _resolution_aware_gradcam_layers(
            model, min_grid=min_grid, max_layers=max(1, int(candidate_count)),
            allow_depthwise=bool(allow_depthwise_candidates),
        )
        candidate_rows = []
        compatibility_rows = []
        pilot_n = min(max(1, int(candidate_pilot_samples)), len(cases))
        pilot_cases = cases[:pilot_n]
        for layer_i, layer in enumerate(candidate_layers):
            for method in methods:
                # GradCAM++ on depthwise layers is a known unstable combination and
                # is rejected even when both are explicitly enabled.
                if (
                    isinstance(layer, tf.keras.layers.DepthwiseConv2D)
                    and str(method).lower().replace(" ", "") in {"gradcam++", "gradcampp"}
                ):
                    compatibility_rows.append({
                        "model": model_name, "layer": layer.name, "method": method,
                        "status": "rejected_known_incompatible",
                        "pilot_n": int(pilot_n), "pilot_valid": 0,
                        "pilot_valid_rate": 0.0,
                    })
                    continue

                pilot_records = []
                pilot_valid = 0
                for case in pilot_cases:
                    hm, _ln, used_method = compute_gradcam_robust(
                        model, preprocessed[case["idx"]], cand, class_index=1,
                        fixed_layer=layer, force_method=method,
                        allow_method_fallback=False, allow_layer_fallback=False,
                        warn_on_failure=False,
                    )
                    hm_r = None
                    if used_method != "FAILED" and compute_heatmap_audit(hm)["heatmap_valid"]:
                        hm_r = normalize_minmax(resize_heatmap(hm, image_size, image_size))
                        pilot_valid += 1
                    pilot_records.append({**case, "map": hm_r})

                pilot_rate = float(pilot_valid / max(pilot_n, 1))
                if pilot_rate < float(candidate_pilot_min_valid_rate):
                    compatibility_rows.append({
                        "model": model_name, "layer": layer.name, "method": method,
                        "status": "rejected_pilot_valid_rate",
                        "pilot_n": int(pilot_n), "pilot_valid": int(pilot_valid),
                        "pilot_valid_rate": pilot_rate,
                    })
                    print(
                        f"[CAM-SCREEN] Reject {model_name}/{layer.name}/{method}: "
                        f"pilot valid={pilot_valid}/{pilot_n} ({pilot_rate:.1%}) < "
                        f"{float(candidate_pilot_min_valid_rate):.1%}"
                    )
                    continue

                map_records = list(pilot_records)
                for case in cases[pilot_n:]:
                    hm, _ln, used_method = compute_gradcam_robust(
                        model, preprocessed[case["idx"]], cand, class_index=1,
                        fixed_layer=layer, force_method=method,
                        allow_method_fallback=False, allow_layer_fallback=False,
                        warn_on_failure=False,
                    )
                    hm_r = None
                    if used_method != "FAILED" and compute_heatmap_audit(hm)["heatmap_valid"]:
                        hm_r = normalize_minmax(resize_heatmap(hm, image_size, image_size))
                    map_records.append({**case, "map": hm_r})

                full_valid = sum(rec.get("map") is not None for rec in map_records)
                full_rate = float(full_valid / max(len(map_records), 1))
                status = (
                    "eligible" if full_rate >= float(candidate_min_valid_rate)
                    else "rejected_full_valid_rate"
                )
                compatibility_rows.append({
                    "model": model_name, "layer": layer.name, "method": method,
                    "status": status, "pilot_n": int(pilot_n),
                    "pilot_valid": int(pilot_valid), "pilot_valid_rate": pilot_rate,
                    "full_n": int(len(map_records)), "full_valid": int(full_valid),
                    "full_valid_rate": full_rate,
                })
                if status != "eligible":
                    print(
                        f"[CAM-SCREEN] Reject {model_name}/{layer.name}/{method}: "
                        f"full valid={full_valid}/{len(map_records)} ({full_rate:.1%}) < "
                        f"{float(candidate_min_valid_rate):.1%}"
                    )
                    continue

                rows = _evaluate_validation_maps(
                    map_records, policies, thresholds,
                    outside_weight=soft_lung_outside_weight, metric_size=selection_metric_size,
                    n_boot=selection_bootstrap, seed=seed,
                )
                for row_eval in rows:
                    row_eval.update({
                        "layers": [layer.name], "methods": [method],
                        "layer_weights": [1.0], "configuration_type": "single_layer",
                        "candidate_rank_architecture": int(layer_i + 1),
                        "candidate_status": "eligible",
                        "pilot_valid_rate": pilot_rate,
                        "full_valid_rate": full_rate,
                    })
                    candidate_rows.append(row_eval)
        usable = [
            r for r in candidate_rows
            if r.get("candidate_status") == "eligible"
            and np.isfinite(r.get("selection_objective", np.nan))
            and r.get("n_valid", 0) > 0
            and (1.0 - float(r.get("invalid_rate", 1.0))) >= float(candidate_min_valid_rate)
        ]
        if not usable:
            audit_path = reports_dir / "xai_cam_candidate_compatibility_validation.csv"
            pd.DataFrame(compatibility_rows).to_csv(audit_path, index=False)
            raise RuntimeError(
                f"No CAM candidate reached the required validation valid-map rate "
                f"({float(candidate_min_valid_rate):.1%}) for {model_name}. "
                f"See {audit_path}. Try regular Conv2D GradCAM/LayerCAM candidates; "
                "do not lower the threshold merely to force a Q1-ready result."
            )
        else:
            best = max(usable, key=lambda r: (r["selection_objective"], -r["invalid_rate"]))
            # Multi-layer candidate: fuse the two strongest distinct layers.
            top_distinct = []
            for row_c in sorted(usable, key=lambda r: r["selection_objective"], reverse=True):
                layer_name = row_c["layers"][0]
                if layer_name not in [r["layers"][0] for r in top_distinct]:
                    top_distinct.append(row_c)
                if len(top_distinct) == 2:
                    break
            if len(top_distinct) == 2:
                pair_records_by_alpha = {}
                for alpha in [0.25, 0.50, 0.75]:
                    pair_records = []
                    cfgs = top_distinct
                    for case in cases:
                        pair_maps = []
                        for cfg in cfgs:
                            layer = _find_layer_recursive(model, [cfg["layers"][0]])
                            hm, _ln, used_method = compute_gradcam_robust(
                                model, preprocessed[case["idx"]], cand, class_index=1,
                                fixed_layer=layer, force_method=cfg["methods"][0],
                                allow_method_fallback=False, allow_layer_fallback=False,
                                warn_on_failure=False,
                            )
                            if used_method == "FAILED" or not compute_heatmap_audit(hm)["heatmap_valid"]:
                                pair_maps = []
                                break
                            pair_maps.append(normalize_minmax(resize_heatmap(hm, image_size, image_size)))
                        fused = None
                        if len(pair_maps) == 2:
                            fused = normalize_minmax(alpha * pair_maps[0] + (1.0 - alpha) * pair_maps[1])
                        pair_records.append({**case, "map": fused})
                    pair_rows = _evaluate_validation_maps(
                        pair_records, policies, thresholds,
                        outside_weight=soft_lung_outside_weight, metric_size=selection_metric_size,
                        n_boot=selection_bootstrap, seed=seed,
                    )
                    for pair_row in pair_rows:
                        pair_row.update({
                            "layers": [cfgs[0]["layers"][0], cfgs[1]["layers"][0]],
                            "methods": [cfgs[0]["methods"][0], cfgs[1]["methods"][0]],
                            "layer_weights": [float(alpha), float(1.0 - alpha)],
                            "configuration_type": "multi_layer",
                        })
                        pair_row["candidate_status"] = (
                            "eligible" if (1.0 - float(pair_row.get("invalid_rate", 1.0)))
                            >= float(candidate_min_valid_rate) else "rejected_full_valid_rate"
                        )
                        candidate_rows.append(pair_row)
                        if (
                            pair_row["candidate_status"] == "eligible"
                            and pair_row["selection_objective"] > best["selection_objective"]
                        ):
                            best = pair_row
        selected_models[model_name] = {
            "layers": list(best["layers"]), "methods": list(best["methods"]),
            "layer_weights": [float(x) for x in best["layer_weights"]],
            "configuration_type": best.get("configuration_type"),
            "validation_objective": float(best.get("selection_objective", np.nan)),
        }
        selected_layers[model_name] = best["layers"][0]
        model_audits[model_name] = {
            "selected": selected_models[model_name],
            "candidate_compatibility": compatibility_rows,
            "candidate_summary": sorted(candidate_rows, key=lambda r: r.get("selection_objective", -np.inf), reverse=True)[:50],
        }
        model_maps = []
        for case in cases:
            hm, _details = _compute_model_cam_from_config(
                model, preprocess_fn, cand, case["raw_rgb"], image_size,
                selected_models[model_name], allow_fallback=False,
                warn_on_failure=False,
            )
            model_maps.append(hm)
        validation_model_maps[model_name] = model_maps
        print(f"[CAM-CONFIG] {model_name}: {selected_models[model_name]}")

    model_names = list(selected_models)
    fusion_candidates = []
    if len(model_names) == 1:
        weight_sets = [{model_names[0]: 1.0}]
    elif len(model_names) == 2:
        if not weight_grid:
            raise RuntimeError(
                "No valid two-CNN CAM fusion weights remain after applying "
                f"cam_min_model_weight={min_model_weight}. "
                "Provide --cam_model_weight_grid values inside the allowed interval."
            )
        weight_sets = [
            {model_names[0]: float(w), model_names[1]: float(1.0 - w)}
            for w in weight_grid
        ]
        if bool(require_all_selected_cnn_contributions):
            bad = [
                weights for weights in weight_sets
                if any(float(weights.get(name, 0.0)) < min_model_weight for name in model_names)
            ]
            if bad:
                raise RuntimeError(
                    "Hybrid CNN-ViT CAM fusion constraint violated: every selected CNN must have "
                    f"weight >= {min_model_weight}. Invalid candidates: {bad}"
                )
    else:
        if bool(require_all_selected_cnn_contributions):
            raise RuntimeError(
                "Strict positive-contribution CAM fusion currently supports the locked one- or "
                "two-CNN AURA-CXR configuration only."
            )
        equal = {name: 1.0 / len(model_names) for name in model_names}
        weight_sets = [equal] + [{n: 1.0 if n == name else 0.0 for n in model_names} for name in model_names]
    for weights in weight_sets:
        records = []
        for case_i, case in enumerate(cases):
            maps, ws = [], []
            missing_required = []
            for name in model_names:
                weight = float(weights.get(name, 0.0))
                hm = validation_model_maps[name][case_i]
                if weight <= 0:
                    continue
                if hm is None:
                    missing_required.append(name)
                    if bool(require_all_selected_cnn_contributions):
                        continue
                else:
                    maps.append(hm); ws.append(weight)
            fused = None
            if bool(require_all_selected_cnn_contributions) and missing_required:
                fused = None
            elif maps:
                w = np.asarray(ws, dtype=float); w = w / w.sum()
                fused = normalize_minmax(np.tensordot(w, np.stack(maps), axes=(0, 0)))
            records.append({**case, "map": fused})
        rows = _evaluate_validation_maps(records, policies, thresholds, soft_lung_outside_weight, metric_size=selection_metric_size, n_boot=selection_bootstrap, seed=seed)
        for row_eval in rows:
            row_eval["model_weights"] = weights
            fusion_candidates.append(row_eval)
    eligible_fusions = [
        r for r in fusion_candidates
        if np.isfinite(r.get("selection_objective", np.nan))
        and (1.0 - float(r.get("invalid_rate", 1.0))) >= float(candidate_min_valid_rate)
    ]
    if not eligible_fusions:
        raise RuntimeError(
            "No multi-CNN fusion candidate met the validation valid-map-rate requirement. "
            "Inspect the per-model candidate compatibility audit."
        )
    best_fusion = max(
        eligible_fusions,
        key=lambda r: (r["selection_objective"], -r["invalid_rate"]),
    )
    # Confirm the locked configuration on every validation-positive case with a box.
    confirmation = None
    if bool(validation_confirm_all):
        confirm_cases = []
        for case_order, idx in enumerate(all_positive_indices):
            row = val_df.iloc[int(idx)]
            pid = str(row["patientId"])
            raw = load_dicom_grayscale(row["image_path"], image_size)
            mask, mask_mode = load_lung_mask_for_patient(
                pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
            ) if use_lung_mask else (None, "off")
            if (not allow_ellipse_in_validation) and "ellipse" in str(mask_mode).lower():
                continue
            confirm_cases.append({
                "idx": int(idx), "pid": pid, "raw": raw,
                "raw_rgb": np.repeat(raw[..., None], 3, axis=-1),
                "gt_boxes": scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size),
                "lung_mask": mask, "lung_mask_mode": mask_mode,
                "center_map": center_map, "lung_prior": _lung_prior_heatmap(mask),
                "lesion_prior": lesion_prevalence_map,
                "fold_id": int(len(confirm_cases) % max(int(validation_folds), 1)),
            })
        confirm_model_maps = {}
        for model, preprocess_fn, cand in models_pp_layers:
            name = _model_cam_key(model)
            cfg = selected_models[name]
            maps = []
            for case in confirm_cases:
                hm, _details = _compute_model_cam_from_config(
                    model, preprocess_fn, cand, case["raw_rgb"], image_size,
                    cfg, allow_fallback=False, warn_on_failure=False,
                )
                maps.append(hm)
            confirm_model_maps[name] = maps
        confirm_records = []
        for case_i, case in enumerate(confirm_cases):
            maps, ws = [], []
            missing_required = []
            for name in model_names:
                hm = confirm_model_maps[name][case_i]
                weight = float(best_fusion["model_weights"].get(name, 0.0))
                if weight <= 0:
                    continue
                if hm is None:
                    missing_required.append(name)
                    if bool(require_all_selected_cnn_contributions):
                        continue
                else:
                    maps.append(hm); ws.append(weight)
            fused = None
            if bool(require_all_selected_cnn_contributions) and missing_required:
                fused = None
            elif maps:
                w = np.asarray(ws, dtype=float); w = w / w.sum()
                fused = normalize_minmax(np.tensordot(w, np.stack(maps), axes=(0, 0)))
            confirm_records.append({**case, "map": fused})
        confirm_rows = _evaluate_validation_maps(
            confirm_records, [best_fusion["map_policy"]],
            [best_fusion["heatmap_threshold_pct"]], soft_lung_outside_weight,
            metric_size=selection_metric_size, n_boot=max(1000, int(selection_bootstrap)), seed=seed + 101,
        )
        confirmation = confirm_rows[0] if confirm_rows else None
        if confirmation is not None:
            confirmation["n_all_validation_positive_with_boxes"] = int(len(confirm_cases))
            confirmation["selection_subset_n"] = int(len(cases))
            confirmation["validation_baseline_superiority_pass"] = bool(
                confirmation.get("strong_baseline_mean_superiority", False)
                and confirmation.get("strong_baseline_ci_superiority", False)
            )
            print(
                "[CAM-CONFIG] All-positive validation confirmation: "
                f"delta={confirmation.get('strong_baseline_quality_delta_mean', np.nan):.4f}, "
                f"CI-low={confirmation.get('strong_baseline_quality_delta_ci95_low', np.nan):.4f}, "
                f"pass={confirmation['validation_baseline_superiority_pass']}"
            )

    selected_weights = {k: float(v) for k, v in best_fusion["model_weights"].items()}
    if len(model_names) == 2 and bool(require_all_selected_cnn_contributions):
        too_small = {k: v for k, v in selected_weights.items() if v < min_model_weight}
        if too_small or set(selected_weights) != set(model_names):
            raise RuntimeError(
                "Selected CAM fusion violates the hybrid CNN-ViT positive-contribution lock: "
                f"weights={selected_weights}, min={min_model_weight}."
            )

    manifest = {
        **config,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_validation_cases": int(len(cases)),
        "n_validation_positive_available": int(len(all_positive_indices)),
        "validation_confirmation": confirmation,
        "validation_baseline_superiority_pass": bool(
            confirmation and confirmation.get("validation_baseline_superiority_pass", False)
        ),
        "selected_layers": selected_layers,
        "selected_models": selected_models,
        "selected_model_weights": selected_weights,
        "selected_map_policy": str(best_fusion["map_policy"]),
        "selected_threshold": float(best_fusion["heatmap_threshold_pct"]),
        "selected_validation_objective": float(best_fusion["selection_objective"]),
        "models": model_audits,
        "fusion_candidate_summary": sorted(fusion_candidates, key=lambda r: r.get("selection_objective", -np.inf), reverse=True)[:50],
    }
    compatibility_flat = []
    for model_name, audit in model_audits.items():
        compatibility_flat.extend(audit.get("candidate_compatibility", []))
    pd.DataFrame(compatibility_flat).to_csv(
        reports_dir / "xai_cam_candidate_compatibility_validation.csv", index=False
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8")
    legacy_path.write_text(json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8")
    _CAM_CONFIG_LOCK = manifest
    _CAM_LAYER_LOCK = dict(selected_layers)
    _CAM_LAYER_SELECTION_AUDIT = manifest
    print(f"[CAM-CONFIG] Validation-only manifest saved: {manifest_path}")
    print(
        f"[CAM-CONFIG] Selected fusion weights={manifest['selected_model_weights']} | "
        f"policy={manifest['selected_map_policy']} | threshold={manifest['selected_threshold']}"
    )
    return manifest


def compute_class_localization_map(models_pp_layers, raw_rgb, image_size,
                                   class_index=1, smooth_sigma_frac=0.005,
                                   force_method="validation_selected", min_grid=16):
    """Validation-selected multi-layer and multi-CNN pneumonia-class CAM fusion."""
    del class_index, min_grid
    manifest = _CAM_CONFIG_LOCK or _CAM_LAYER_SELECTION_AUDIT
    selected_models = manifest.get("selected_models", {}) if manifest else {}
    model_weights = manifest.get("selected_model_weights", {}) if manifest else {}
    maps, details, used_weights = [], [], []
    require_all = bool(manifest.get("require_all_selected_cnn_contributions", False)) if manifest else False
    min_model_weight = float(manifest.get("cam_min_model_weight", 0.0)) if manifest else 0.0
    expected_models = [
        name for name, weight in model_weights.items()
        if float(weight) > 0
    ]
    missing_required = []
    for model, preprocess_fn, cand in models_pp_layers:
        model_name = _model_cam_key(model)
        cfg = selected_models.get(model_name)
        if cfg is None:
            layer, source = _locked_cam_layer(model, cand, min_grid=min_grid)
            cfg = {"layers": [layer.name], "methods": ["GradCAM"], "layer_weights": [1.0], "configuration_type": source}
        if str(force_method).lower() not in {"validation_selected", "auto", "none"}:
            cfg = dict(cfg)
            cfg["methods"] = [str(force_method)] * len(cfg.get("layers", [1]))
        hm, model_details = _compute_model_cam_from_config(
            model, preprocess_fn, cand, raw_rgb, image_size, cfg, allow_fallback=True,
        )
        for d in model_details:
            d["model"] = model_name
        details.extend(model_details)
        weight = float(model_weights.get(model_name, 1.0))
        if weight <= 0:
            continue
        if require_all and weight < min_model_weight:
            missing_required.append(model_name)
            continue
        if hm is None:
            if require_all:
                missing_required.append(model_name)
            continue
        maps.append(hm); used_weights.append(weight)
    if require_all:
        present_models = {
            _model_cam_key(model) for model, _, _ in models_pp_layers
            if float(model_weights.get(_model_cam_key(model), 0.0)) > 0
        }
        absent_from_runtime = sorted(set(expected_models) - present_models)
        missing_required.extend(absent_from_runtime)
        if missing_required or len(maps) != len(expected_models):
            details.append({
                "status": "failed_required_dual_cnn_contribution",
                "missing_models": sorted(set(missing_required)),
                "expected_models": expected_models,
                "selected_model_weights": {k: float(v) for k, v in model_weights.items()},
            })
            return np.zeros((image_size, image_size), dtype=np.float32), "FAILED", details
    if not maps:
        return np.zeros((image_size, image_size), dtype=np.float32), "FAILED", details
    w = np.asarray(used_weights, dtype=float); w = w / w.sum()
    fused = np.tensordot(w, np.stack(maps), axes=(0, 0)).astype(np.float32)
    if smooth_sigma_frac and float(smooth_sigma_frac) > 0:
        sigma = max(0.5, float(smooth_sigma_frac) * int(image_size))
        fused = gaussian_filter(fused, sigma=sigma).astype(np.float32)
    fused = normalize_minmax(fused)
    method_tag = "+".join(sorted({d.get("method", "") for d in details if d.get("status") == "ok"}))
    return fused, f"validation_weighted_cnn_{method_tag or 'CAM'}", details


def compute_dual_cnn_component_maps(models_pp_layers, raw_rgb, image_size,
                                    smooth_sigma_frac=0.005,
                                    force_method="validation_selected",
                                    min_grid=16):
    """Return per-CNN CAM maps plus the existing validation-locked fused CAM.

    This helper is visualization-only. It reuses the exact validation-selected
    layer/method configuration for each selected CNN and delegates the final
    fusion to ``compute_class_localization_map`` so the fused panel is identical
    to the map used by the quantitative XAI pipeline.
    """
    min_grid = max(1, int(min_grid))
    manifest = _CAM_CONFIG_LOCK or _CAM_LAYER_SELECTION_AUDIT
    selected_models = manifest.get("selected_models", {}) if manifest else {}
    component_maps = {}
    component_details = {}

    for model, preprocess_fn, cand in models_pp_layers:
        model_name = _model_cam_key(model)
        cfg = selected_models.get(model_name)
        if cfg is None:
            layer, source = _locked_cam_layer(model, cand, min_grid=16)
            cfg = {
                "layers": [layer.name],
                "methods": ["GradCAM"],
                "layer_weights": [1.0],
                "configuration_type": source,
            }
        if str(force_method).lower() not in {"validation_selected", "auto", "none"}:
            cfg = dict(cfg)
            cfg["methods"] = [str(force_method)] * len(cfg.get("layers", [1]))

        hm, details = _compute_model_cam_from_config(
            model, preprocess_fn, cand, raw_rgb, image_size, cfg, allow_fallback=True,
        )
        for d in details:
            d["model"] = model_name
        component_details[model_name] = details
        if hm is None:
            continue
        hm = np.asarray(hm, dtype=np.float32)
        if smooth_sigma_frac and float(smooth_sigma_frac) > 0:
            sigma = max(0.5, float(smooth_sigma_frac) * int(image_size))
            hm = gaussian_filter(hm, sigma=sigma).astype(np.float32)
        component_maps[model_name] = normalize_minmax(hm)

    fused_map, fused_method, fused_details = compute_class_localization_map(
        models_pp_layers,
        raw_rgb,
        image_size,
        class_index=1,
        smooth_sigma_frac=smooth_sigma_frac,
        force_method=force_method,
        min_grid=min_grid,
    )
    return component_maps, fused_map, fused_method, component_details, fused_details


def _resolve_dual_cnn_visual_branches(component_maps):
    """Resolve XRV and EVA-X keys deterministically for four-stage figures."""
    keys = list(component_maps.keys())
    xrv_key = next((k for k in keys if "xrv" in str(k).lower() or "densenet" in str(k).lower()), None)
    eva_key = next((k for k in keys if "eva" in str(k).lower()), None)
    remaining = [k for k in keys if k not in {xrv_key, eva_key}]
    if xrv_key is None and remaining:
        xrv_key = remaining.pop(0)
    if eva_key is None:
        remaining = [k for k in keys if k != xrv_key]
        if remaining:
            eva_key = remaining[0]
    return xrv_key, eva_key


class PretrainedLungSegmenter:
    """Pretrained CXR lung segmentation via torchxrayvision (chestx_det PSPNet).

    Produces a per-patient anatomical lung mask (union of Left/Right Lung),
    resized to `image_size`. CPU is sufficient; the model is loaded lazily once.
    Masks are cached to disk by the caller. On any failure the segmenter reports
    unavailability so the caller can decide (raise vs. explicit ellipse fallback).
    """

    def __init__(self, seg_input_size=512):
        self.seg_input_size = int(seg_input_size)
        self._model = None
        self._xrv = None
        self._torch = None
        self._load_failed = False
        self._lung_idx = None
        self.last_source = "pretrained_unset"
        self.last_threshold = None

    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            import torchxrayvision as xrv
            self._torch = torch
            self._xrv = xrv
            self._model = xrv.baseline_models.chestx_det.PSPNet()
            self._model.eval()
            targets = [str(t).lower() for t in getattr(self._model, "targets", [])]
            self._lung_idx = [i for i, t in enumerate(targets) if "lung" in t]
            if not self._lung_idx:
                raise RuntimeError("PSPNet targets do not expose lung channels.")
            print(f"[LUNGMASK] Pretrained torchxrayvision PSPNet loaded; lung channels={self._lung_idx}.")
            return True
        except Exception as exc:
            print(f"[LUNGMASK][ERROR] Could not load pretrained lung segmentation: {exc}")
            self._load_failed = True
            return False

    def segment(self, raw_gray_01: np.ndarray, image_size: int):
        """Return a validated lung mask with deterministic anatomy-only TTA rescue.

        The standard image is attempted first.  Only when it cannot yield a
        plausible mask are fixed contrast/gamma/flip variants evaluated.  No
        diagnosis, bounding box, validation metric, or test label is consulted.
        """
        self.last_source = "pretrained_failed"
        self.last_threshold = None
        if not self.available():
            return None
        try:
            xrv, torch = self._xrv, self._torch
            raw = np.clip(np.asarray(raw_gray_01, dtype=np.float32), 0, 1)

            def predict_lung(img01, flip=False):
                img8 = (np.clip(img01, 0, 1) * 255.0).astype(np.float32)
                img_norm = xrv.datasets.normalize(img8, 255)
                t = torch.from_numpy(img_norm)[None, None, ...].float()
                t = torch.nn.functional.interpolate(
                    t, size=(self.seg_input_size, self.seg_input_size),
                    mode="bilinear", align_corners=False,
                )
                with torch.no_grad():
                    out = self._model(t)
                prob = torch.sigmoid(out)[0].cpu().numpy()
                lung_prob = np.clip(sum(prob[i] for i in self._lung_idx), 0, 1)
                return np.fliplr(lung_prob) if flip else lung_prob

            def contrast01(img):
                lo, hi = np.percentile(img, [1.0, 99.0])
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
                    return img
                return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)

            variants = [("original", raw, False)]
            rescue_variants = [
                ("contrast", contrast01(raw), False),
                ("gamma080", np.power(raw, 0.80).astype(np.float32), False),
                ("gamma120", np.power(raw, 1.20).astype(np.float32), False),
                ("hflip", np.fliplr(raw).copy(), True),
            ]
            thresholds = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.60, 0.70, 0.80)

            def candidates_for(variant_list):
                found = []
                for variant_rank, (name, img, was_flipped) in enumerate(variant_list):
                    lung = predict_lung(img, flip=was_flipped)
                    for threshold in thresholds:
                        lung_bin = lung >= float(threshold)
                        fy = image_size / lung_bin.shape[0]
                        fx = image_size / lung_bin.shape[1]
                        resized = zoom(lung_bin.astype(np.float32), (fy, fx), order=0) >= 0.5
                        cleaned = postprocess_lung_mask(
                            resized, (int(image_size), int(image_size)),
                            min_area=0.07 if name != "original" else 0.08,
                            max_area=0.88 if name != "original" else 0.85,
                        )
                        if cleaned is None:
                            continue
                        area = float(cleaned.mean())
                        yy, xx = np.nonzero(cleaned)
                        cx = float(xx.mean() / max(image_size - 1, 1))
                        cy = float(yy.mean() / max(image_size - 1, 1))
                        left = float(cleaned[:, :image_size // 2].mean())
                        right = float(cleaned[:, image_size // 2:].mean())
                        bilateral_penalty = abs(left - right)
                        border_penalty = float(cleaned[0].mean() + cleaned[-1].mean())
                        score = (
                            abs(area - 0.42)
                            + 0.35 * abs(cx - 0.50)
                            + 0.20 * abs(cy - 0.52)
                            + 0.40 * bilateral_penalty
                            + 0.20 * border_penalty
                            + 0.015 * abs(float(threshold) - 0.50)
                            + 0.01 * variant_rank
                        )
                        found.append((score, variant_rank, abs(float(threshold)-0.50), name, threshold, cleaned))
                return found

            candidates = candidates_for(variants)
            if not candidates:
                candidates = candidates_for(rescue_variants)
            if not candidates:
                return None
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            _, _, _, variant, threshold, mask = candidates[0]
            self.last_threshold = float(threshold)
            if variant == "original" and abs(float(threshold) - 0.50) < 1e-9:
                self.last_source = "pretrained_mask"
            elif variant == "original":
                self.last_source = f"pretrained_adaptive_t{float(threshold):.2f}"
            else:
                self.last_source = f"pretrained_tta_{variant}_t{float(threshold):.2f}"
            return np.asarray(mask, dtype=bool)
        except Exception as exc:
            print(f"[LUNGMASK][WARN] Segmentation failed on a sample: {exc}")
            self.last_source = "pretrained_exception"
            return None



def build_lung_mask_cache(test_df, out_dir, image_size=224, chunk_id=None,
                          n_chunks=None, method="pretrained",
                          allow_ellipse_fallback=False, seg_input_size=512,
                          repair_heuristic_masks=False):
    """Build a resumable lung-mask cache with per-patient provenance metadata."""
    out_dir = ensure_dir(out_dir)
    rows = test_df.reset_index(drop=True)
    if chunk_id is not None and n_chunks:
        parts = np.array_split(np.arange(len(rows)), int(n_chunks))
        idx = parts[int(chunk_id)] if 0 <= int(chunk_id) < len(parts) else []
        rows = rows.iloc[idx]
    use_ellipse = str(method).lower() == "ellipse"
    segmenter = None
    if not use_ellipse:
        segmenter = PretrainedLungSegmenter(seg_input_size=seg_input_size)
        if not segmenter.available():
            if allow_ellipse_fallback:
                print("[LUNGMASK][WARN] Pretrained model unavailable; explicit ellipse fallback active.")
                use_ellipse = True
            else:
                raise RuntimeError(
                    "Pretrained lung segmentation unavailable. Install torch and "
                    "torchxrayvision, or explicitly use --allow_ellipse_fallback."
                )

    records=[]; n_ok=0; n_fail=0; n_fallback=0
    for _,row in tqdm(rows.iterrows(),total=len(rows),desc="[LUNGMASK] cache"):
        pid=str(row["patientId"])
        out=out_dir/f"{pid}.npy"; meta=out_dir/f"{pid}.json"
        # Validate existing artifact.  V8 can explicitly reprocess heuristic
        # masks with the pretrained TTA rescue while preserving the old mask if
        # recovery still fails.
        preserved_existing = None
        preserved_source = None
        if out.exists():
            try:
                existing=postprocess_lung_mask(np.load(out),(image_size,image_size))
                meta_payload = json.loads(meta.read_text()) if meta.exists() else {"source":"unknown_existing"}
                existing_source = str(meta_payload.get("source", "unknown_existing"))
                is_heuristic = any(token in existing_source.lower() for token in ("ellipse", "unknown", "invalid"))
                if existing is not None and not (repair_heuristic_masks and is_heuristic and not use_ellipse):
                    np.save(out,existing)
                    if not meta.exists():
                        meta.write_text(json.dumps({"source":existing_source,"image_size":int(image_size)}),encoding="utf-8")
                    n_ok+=1
                    records.append({"patientId":pid,"status":"ok_existing","source":existing_source})
                    continue
                if existing is not None:
                    preserved_existing = existing
                    preserved_source = existing_source
            except Exception:
                preserved_existing = None
                preserved_source = None

        source="ellipse" if use_ellipse else "pretrained"
        if use_ellipse:
            mask=_ellipse_fallback_lung_mask(image_size)
            if str(method).lower()!="ellipse":
                source="ellipse_fallback"; n_fallback+=1
        else:
            raw=load_dicom_grayscale(row["image_path"],image_size)
            mask=segmenter.segment(raw,image_size)
            source = str(getattr(segmenter, "last_source", "pretrained_mask"))
            if mask is None and preserved_existing is not None:
                mask = preserved_existing
                source = f"{preserved_source}_retained_after_tta_failure"
                if "ellipse" in str(preserved_source).lower():
                    n_fallback += 1
            elif mask is None and allow_ellipse_fallback:
                mask=_ellipse_fallback_lung_mask(image_size)
                source="ellipse_fallback"; n_fallback+=1
        mask=postprocess_lung_mask(mask,(image_size,image_size)) if mask is not None else None
        if mask is None:
            n_fail+=1; records.append({"patientId":pid,"status":"failed","source":source}); continue
        np.save(out,mask)
        meta.write_text(json.dumps({
            "patientId":pid,"source":source,"image_size":int(image_size),
            "area_ratio":float(mask.mean()),"created_at":datetime.now().isoformat(timespec="seconds")
        },indent=2),encoding="utf-8")
        n_ok+=1; records.append({"patientId":pid,"status":"ok","source":source,"area_ratio":float(mask.mean())})
    manifest=out_dir/"lung_mask_manifest.csv"
    new_df=pd.DataFrame(records)
    if manifest.exists():
        try:
            old=pd.read_csv(manifest); new_df=pd.concat([old,new_df],ignore_index=True).drop_duplicates("patientId",keep="last")
        except Exception:
            pass
    new_df.to_csv(manifest,index=False)
    print(f"[LUNGMASK] done: ok={n_ok} | failed={n_fail} | ellipse_fallback={n_fallback} | manifest={manifest}")
    return out_dir

# -----------------------------------------------------------------------------
# Heatmap localization metrics
# -----------------------------------------------------------------------------

def heatmap_to_binary_mask(heatmap_resized, threshold_pct=0.60):
    hm = np.asarray(heatmap_resized, dtype=np.float32)
    hmax = float(np.nanmax(hm)) if hm.size else 0.0
    if not np.isfinite(hmax) or hmax <= 1e-8:
        return np.zeros_like(hm, dtype=bool)
    return hm >= (hmax * float(threshold_pct))


def mask_iou(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return np.nan
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return float(inter / union)



def peak_point(heatmap_resized):
    hm = np.asarray(heatmap_resized, dtype=np.float32)
    if hm.ndim != 2 or hm.size == 0 or not np.isfinite(hm).any():
        return np.nan, np.nan
    mx = float(np.nanmax(hm))
    if not np.isfinite(mx) or mx <= ACTIVATION_THRESHOLD:
        return np.nan, np.nan
    y, x = np.unravel_index(int(np.nanargmax(hm)), hm.shape)
    return int(x), int(y)

def distance_to_nearest_box(px, py, boxes, height, width):
    if not boxes or not np.isfinite(px) or not np.isfinite(py):
        return np.nan
    dists = []
    for x, y, w, h in boxes:
        cx = min(max(px, x), x + w)
        cy = min(max(py, y), y + h)
        dists.append(math.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    diag = math.sqrt(height ** 2 + width ** 2)
    return float(min(dists) / max(diag, 1.0))


def get_peak_bboxes(heatmap_resized, threshold_pct=0.60, max_boxes=2, min_area_px=8):
    """Return high-activation connected-component boxes in image coordinates."""
    binary = heatmap_to_binary_mask(heatmap_resized, threshold_pct=threshold_pct).astype(np.uint8)
    if binary.sum() == 0:
        return []
    labeled, n_comp = scipy_label(binary)
    boxes = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if len(coords) < min_area_px:
            continue
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        boxes.append((float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:max_boxes]


def annotate_pneumonia_location(
    ax,
    heatmap_resized,
    threshold_pct=0.60,
    label_text="Predicted focus",
    color=PRED_COLOR,
    fontsize=9,
):
    boxes = get_peak_bboxes(heatmap_resized, threshold_pct=threshold_pct)
    for i, (x0, y0, bw, bh) in enumerate(boxes):
        rect = mpatches.FancyBboxPatch(
            (x0, y0),
            bw,
            bh,
            linewidth=2.6,
            edgecolor=color,
            facecolor="none",
            boxstyle="square,pad=1",
        )
        ax.add_patch(rect)
        ax.text(
            x0 + bw / 2.0,
            max(2, y0 - 6),
            label_text if i == 0 else f"Focus {i+1}",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec=color, lw=0.8, alpha=0.75),
        )
    return len(boxes)



def compute_localization_metrics(heatmap_resized, gt_boxes, threshold_pct=0.60,
                                 heatmap_valid=None):
    """Localization metrics against the union of radiologist bounding boxes."""
    hm = np.asarray(heatmap_resized, dtype=np.float32)
    h, w = hm.shape[:2]
    gt_mask = boxes_to_mask(gt_boxes, h, w)
    audit = compute_heatmap_audit(hm)
    valid = audit["heatmap_valid"] if heatmap_valid is None else bool(heatmap_valid)
    metrics = {
        "gt_box_count": int(len(gt_boxes or [])),
        "heatmap_valid": bool(valid),
        "peak_x": np.nan, "peak_y": np.nan,
        "pred_area_ratio": np.nan,
        "mean_activation": audit["heatmap_mean"],
        "max_activation": audit["heatmap_max"],
        "pointing_hit": np.nan,
        "iou_at_thr": np.nan,
        "localization_score": np.nan,
        "energy_inside_gt": np.nan,
        "peak_distance_norm": np.nan,
        "gt_coverage_at_thr": np.nan,
        "activation_precision_at_thr": np.nan,
        "max_activation_inside_gt": np.nan,
    }
    if gt_mask.sum() == 0 or not valid:
        return metrics
    pred_mask = heatmap_to_binary_mask(hm, threshold_pct=threshold_pct)
    px, py = peak_point(hm)
    metrics["peak_x"], metrics["peak_y"] = px, py
    metrics["pred_area_ratio"] = float(pred_mask.mean())
    if np.isfinite(px) and np.isfinite(py):
        px_i, py_i = int(px), int(py)
        metrics["pointing_hit"] = float(
            0 <= px_i < w and 0 <= py_i < h and bool(gt_mask[py_i, px_i])
        )
    metrics["iou_at_thr"] = mask_iou(pred_mask, gt_mask)
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    metrics["gt_coverage_at_thr"] = inter / max(float(gt_mask.sum()), 1.0)
    metrics["activation_precision_at_thr"] = inter / max(float(pred_mask.sum()), 1.0)
    thresholds = np.linspace(0.10, 0.90, 17)
    ious = [mask_iou(heatmap_to_binary_mask(hm, threshold_pct=t), gt_mask) for t in thresholds]
    finite_ious = [v for v in ious if np.isfinite(v)]
    metrics["localization_score"] = float(max(finite_ious)) if finite_ious else np.nan
    total_energy = float(np.nansum(np.clip(hm, 0, None)))
    if total_energy > 1e-8:
        metrics["energy_inside_gt"] = float(np.clip(np.nansum(np.clip(hm[gt_mask], 0, None)) / total_energy, 0.0, 1.0))
    metrics["max_activation_inside_gt"] = float(np.nanmax(hm[gt_mask])) if gt_mask.any() else np.nan
    metrics["peak_distance_norm"] = distance_to_nearest_box(px, py, gt_boxes, h, w)
    return metrics


def summarize_localization_metrics(metrics_df, n_boot=2000, seed=42):
    """Q1 summary with mean/median, IQR, and bootstrap CIs."""
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    rows = []
    group_defs = [("all_gt_positive", metrics_df)]
    if "category" in metrics_df.columns:
        for cat in ["TP", "FN"]:
            sub = metrics_df[metrics_df["category"] == cat]
            if len(sub):
                group_defs.append((cat, sub))
    metric_cols = [
        "pointing_hit", "iou_at_thr", "localization_score",
        "energy_inside_gt", "peak_distance_norm", "pred_area_ratio",
        "gt_coverage_at_thr", "activation_precision_at_thr",
        "max_activation_inside_gt", "outside_lung_ratio_raw",
        "outside_lung_ratio_gated", "inside_lung_energy_raw",
        "inside_lung_energy_gated", "heatmap_entropy",
        "heatmap_nonzero_ratio",
    ]
    for group_i, (name, sub) in enumerate(group_defs):
        valid_mask = sub.get("heatmap_valid", pd.Series(True, index=sub.index)).astype(bool)
        valid = sub[valid_mask].copy()
        row = {
            "group": name,
            "n_total": int(len(sub)),
            "n_valid_heatmaps": int(len(valid)),
            "invalid_heatmap_rate": float(1.0 - len(valid) / max(len(sub), 1)),
            "lung_mask_success_rate": float(
                pd.to_numeric(
                    sub.get("lung_mask_ok", pd.Series(False, index=sub.index)),
                    errors="coerce",
                ).fillna(0).mean()
            ),
        }
        for col_i, col in enumerate(metric_cols):
            if col not in valid.columns:
                continue
            vals = pd.to_numeric(valid[col], errors="coerce").dropna()
            row[f"{col}_n"] = int(len(vals))
            if not len(vals):
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_q1"] = float(vals.quantile(0.25))
            row[f"{col}_q3"] = float(vals.quantile(0.75))
            row[f"{col}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25))
            mean_lo, mean_hi = _bootstrap_stat_ci(
                vals, stat="mean", n_boot=n_boot,
                seed=int(seed) + group_i * 100 + col_i,
            )
            med_lo, med_hi = _bootstrap_stat_ci(
                vals, stat="median", n_boot=n_boot,
                seed=int(seed) + 10000 + group_i * 100 + col_i,
            )
            row[f"{col}_mean_ci95_low"] = mean_lo
            row[f"{col}_mean_ci95_high"] = mean_hi
            row[f"{col}_median_ci95_low"] = med_lo
            row[f"{col}_median_ci95_high"] = med_hi
        rows.append(row)
    return pd.DataFrame(rows)

def _load_lung_mask_array(patient_id, lung_mask_dir, image_size):
    mask, _mode = load_lung_mask_for_patient(
        patient_id, lung_mask_dir, image_size, allow_ellipse=False
    )
    return mask

def _center_bias_heatmap(image_size, sigma_frac=0.25):
    """Validation-locked 2-D Gaussian image-center baseline."""
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    cy = cx = (image_size - 1) / 2.0
    sigma = max(float(sigma_frac) * float(image_size), 1.0)
    hm = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    return normalize_minmax(hm).astype(np.float32)


def build_lesion_prevalence_prior(reference_df, gt_box_map, image_size, reports_dir=None):
    """Build a training-derived lesion-location prior without test labels."""
    acc = np.zeros((int(image_size), int(image_size)), dtype=np.float64)
    n = 0
    if reference_df is not None and len(reference_df):
        for _, row in reference_df.iterrows():
            if int(row.get("label", 0)) != 1:
                continue
            boxes = scale_boxes_to_image(
                str(row["patientId"]), row["image_path"], gt_box_map, image_size
            )
            if not boxes:
                continue
            acc += boxes_to_mask(boxes, image_size, image_size).astype(np.float64)
            n += 1
    prior = normalize_minmax(acc / max(n, 1)).astype(np.float32)
    if reports_dir is not None:
        reports_dir = ensure_dir(reports_dir)
        np.save(reports_dir / "xai_training_lesion_prevalence_prior.npy", prior)
        (reports_dir / "xai_training_lesion_prevalence_prior.json").write_text(
            json.dumps({
                "source_split": "train", "test_set_used": False,
                "n_positive_with_boxes": int(n), "image_size": int(image_size),
            }, indent=2), encoding="utf-8"
        )
    return prior, n


def _lung_prior_heatmap(lung_mask):
    """Smooth anatomical prior: distance from the lung boundary."""
    if lung_mask is None:
        return None
    lm = np.asarray(lung_mask, dtype=bool)
    if not lm.any():
        return None
    hm = distance_transform_edt(lm).astype(np.float32)
    mx = float(hm.max())
    return hm / mx if mx > 0 else None



def compute_chance_baselines(metrics_df, test_df, gt_box_map, image_size,
                             use_lung_mask=False, lung_mask_dir=None,
                             threshold_pct=0.60, seed=42, reports_dir=None,
                             n_boot=2000, random_repeats=5,
                             center_sigma_frac=0.25,
                             lesion_prevalence_map=None,
                             map_policy="raw", soft_lung_outside_weight=0.20):
    """Strong paired baselines evaluated on exactly the same positive test cases."""
    if metrics_df is None or len(metrics_df) == 0 or "sample_index" not in metrics_df.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    center = _center_bias_heatmap(image_size, sigma_frac=center_sigma_frac)
    per_sample = []
    metric_names = ["pointing_hit", "iou_at_thr", "localization_score", "energy_inside_gt"]
    for _, r in metrics_df.iterrows():
        try:
            idx = int(r["sample_index"])
        except Exception:
            continue
        row = test_df.iloc[idx]
        pid = str(row["patientId"])
        gt_boxes = scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
        if len(gt_boxes) == 0:
            continue
        lm = _load_lung_mask_array(pid, lung_mask_dir, image_size) if use_lung_mask else None
        # Repeated random baseline: average metric over independent maps per case.
        random_metrics = {k: [] for k in metric_names}
        for _rep in range(max(1, int(random_repeats))):
            random_hm = rng.random((image_size, image_size)).astype(np.float32)
            random_hm = _apply_lung_constraint_policy(
                random_hm, lm, policy=map_policy,
                outside_weight=soft_lung_outside_weight,
            )
            met = compute_localization_metrics(random_hm, gt_boxes, threshold_pct=threshold_pct)
            for k in metric_names:
                random_metrics[k].append(met.get(k, np.nan))
        rec = {"sample_index": idx, "patientId": pid, "method": "random_noise"}
        rec.update({k: float(np.nanmean(v)) for k, v in random_metrics.items()})
        per_sample.append(rec)

        baseline_maps = [
            ("image_center", center),
            ("lung_prior", _lung_prior_heatmap(lm)),
            ("lesion_prevalence_prior", lesion_prevalence_map),
        ]
        for name, hm in baseline_maps:
            if hm is None:
                continue
            hm = _apply_lung_constraint_policy(
                hm, lm, policy=map_policy,
                outside_weight=soft_lung_outside_weight,
            )
            met = compute_localization_metrics(hm, gt_boxes, threshold_pct=threshold_pct)
            rec = {"sample_index": idx, "patientId": pid, "method": name}
            rec.update({k: met.get(k, np.nan) for k in metric_names})
            per_sample.append(rec)
    per_df = pd.DataFrame(per_sample)
    if len(per_df) == 0:
        return pd.DataFrame()
    rows = []
    valid_model = metrics_df[
        metrics_df.get("heatmap_valid", pd.Series(True, index=metrics_df.index)).astype(bool)
    ].copy()
    groups = [("model_all_gt_positive", valid_model)] + list(per_df.groupby("method"))
    for method, sub in groups:
        row = {"method": method, "n": int(len(sub))}
        for metric_i, metric in enumerate(metric_names):
            vals = pd.to_numeric(sub.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{metric}_q1"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"{metric}_q3"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            lo, hi = _bootstrap_stat_ci(vals, stat="mean", n_boot=n_boot, seed=int(seed) + metric_i)
            row[f"{metric}_mean_ci95_low"] = lo
            row[f"{metric}_mean_ci95_high"] = hi
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    comparisons = []
    model_cols = valid_model[["sample_index"] + metric_names].copy()
    model_cols["sample_index"] = pd.to_numeric(model_cols["sample_index"], errors="coerce")
    for method, base_sub in per_df.groupby("method"):
        merged = model_cols.merge(
            base_sub[["sample_index"] + metric_names], on="sample_index", how="inner",
            suffixes=("_model", "_baseline"),
        )
        for metric_i, metric in enumerate(metric_names):
            point, lo, hi, n_pair = _paired_bootstrap_diff_ci(
                merged[f"{metric}_model"], merged[f"{metric}_baseline"],
                n_boot=n_boot, seed=int(seed) + 1000 + metric_i,
            )
            comparisons.append({
                "baseline": method, "metric": metric, "n_paired": n_pair,
                "mean_difference_model_minus_baseline": point,
                "ci95_low": lo, "ci95_high": hi,
                "mean_superiority": bool(np.isfinite(point) and point > 0),
                "ci_superiority": bool(np.isfinite(lo) and lo > 0),
            })
    comparison_df = pd.DataFrame(comparisons)
    if reports_dir is not None:
        reports_dir = ensure_dir(reports_dir)
        per_df.to_csv(reports_dir / "xai_localization_baselines_per_sample.csv", index=False)
        summary_df.to_csv(reports_dir / "xai_localization_baselines.csv", index=False)
        comparison_df.to_csv(reports_dir / "xai_localization_baseline_comparison.csv", index=False)
        print(f"[OK] Strong XAI baselines saved under: {reports_dir}")
    return summary_df


def update_qc_with_baseline_results(reports_dir, mode="report"):
    """Attach scientific baseline evidence without aborting a technically valid run."""
    reports_dir = Path(reports_dir)
    qc_path = reports_dir / "xai_quality_control.json"
    cmp_path = reports_dir / "xai_localization_baseline_comparison.csv"
    if not qc_path.exists() or not cmp_path.exists():
        return None
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    cmp_df = pd.read_csv(cmp_path)
    required_metrics = ["pointing_hit", "localization_score"]
    expected_baselines = ["random_noise", "image_center", "lung_prior", "lesion_prevalence_prior"]
    relevant = cmp_df[cmp_df["metric"].isin(required_metrics)].copy()
    observed_baselines = set(relevant.get("baseline", pd.Series(dtype=str)).astype(str))
    presence = all(name in observed_baselines for name in expected_baselines)
    per_baseline = {}
    for name in expected_baselines:
        sub = relevant[relevant["baseline"].astype(str) == name]
        per_baseline[name] = {
            "present": bool(len(sub)),
            "mean_superiority_pass": bool(len(sub) == len(required_metrics) and sub["mean_superiority"].astype(bool).all()),
            "ci_superiority_pass": bool(len(sub) == len(required_metrics) and sub["ci_superiority"].astype(bool).all()),
        }
    mean_pass = bool(presence and all(v["mean_superiority_pass"] for v in per_baseline.values()))
    ci_pass = bool(presence and all(v["ci_superiority_pass"] for v in per_baseline.values()))
    mode_l = str(mode).lower()
    baseline_pass = ci_pass if mode_l == "ci" else mean_pass if mode_l == "mean" else True
    technical_pass = bool(qc.get("technical_qc_pass", qc.get("qc_pass", False)))
    qc.update({
        "baseline_qc_mode": mode_l,
        "baseline_presence_pass": presence,
        "baseline_methods_required": expected_baselines,
        "baseline_per_method": per_baseline,
        "baseline_mean_superiority_pass": mean_pass,
        "baseline_ci_superiority_pass": ci_pass,
        "baseline_superiority_qc_pass": baseline_pass,
        "baseline_qc_pass": baseline_pass,
        "localization_claim_ready": bool(technical_pass and baseline_pass),
        "validated_localization_claim": bool(technical_pass and baseline_pass),
        "xai_analysis_complete": True,
        "paper_usage": "validated_localization" if technical_pass and baseline_pass else "qualitative_or_limited_localization_claim",
        "qc_pass": technical_pass,
    })
    for name, vals in per_baseline.items():
        safe = name.replace("lesion_prevalence_prior", "lesion_prior")
        qc[f"model_beats_{safe}_mean"] = vals["mean_superiority_pass"]
        qc[f"model_beats_{safe}_ci"] = vals["ci_superiority_pass"]
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def count_gt_positive_with_boxes(df, gt_box_map, image_size):
    """Count positives with usable boxes without re-reading every DICOM header."""
    del image_size  # retained for API compatibility
    n = 0
    for _, row in df.iterrows():
        try:
            if int(row.get("label", 0)) != 1:
                continue
            if len(gt_box_map.get(str(row["patientId"]), [])) > 0:
                n += 1
        except Exception:
            continue
    return int(n)


def resolve_xai_run_mode(requested_mode, xai_eval_max):
    """Resolve and validate QUICK versus FINAL semantics."""
    requested = str(requested_mode or "auto").strip().lower()
    if requested == "auto":
        mode = "QUICK" if int(xai_eval_max or 0) > 0 else "FINAL"
    else:
        mode = requested.upper()
    if mode not in {"QUICK", "FINAL"}:
        raise ValueError(f"Unsupported XAI run mode: {requested_mode}")
    if mode == "FINAL" and int(xai_eval_max or 0) > 0:
        raise ValueError(
            "FINAL mode requires --xai_eval_max 0. A positive sample limit is a QUICK precheck "
            "and cannot produce final/Q1 artifacts."
        )
    return mode


def _json_safe(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _df_to_markdown_simple(df):
    """Dependency-free Markdown table; avoids requiring optional tabulate."""
    if df is None or len(df) == 0:
        return "_No rows available._"
    frame = df.copy()
    def fmt(v):
        if pd.isna(v):
            return "NA"
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.4f}"
        return str(v).replace("|", "\\|")
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines)


def write_xai_run_manifest(path, *, run_mode, xai_eval_max, expected_gt_positive,
                           min_final_coverage, min_final_valid_cases, args):
    payload = {
        "xai_schema_version": XAI_SCHEMA_VERSION,
        "run_mode": str(run_mode),
        "artifact_status": "PRECHECK_ONLY" if str(run_mode) == "QUICK" else "FINAL_CANDIDATE",
        "xai_eval_max": int(xai_eval_max),
        "expected_gt_positive_with_boxes": int(expected_gt_positive),
        "minimum_final_coverage": float(min_final_coverage),
        "minimum_final_valid_cases": int(min_final_valid_cases),
        "quick_outputs_are_not_final": bool(str(run_mode) == "QUICK"),
        "decision_threshold": float(args.decision_threshold),
        "cam_layer_policy": str(args.cam_layer_policy),
        "cam_method_candidates": str(args.cam_method_candidates),
        "cam_map_policy_candidates": str(args.cam_map_policy_candidates),
        "cam_threshold_candidates": str(args.cam_threshold_candidates),
        "baseline_qc_mode": str(args.baseline_qc_mode),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")
    return payload



def update_qc_with_final_guardrails(reports_dir, *, run_mode, expected_gt_positive,
                                    min_final_coverage, min_final_valid_cases,
                                    layer_manifest_path, require_baseline=True,
                                    allow_baseline_report_only=False):
    """Separate pipeline validity from scientific localization-claim readiness."""
    reports_dir = Path(reports_dir)
    qc_path = reports_dir / "xai_quality_control.json"
    if not qc_path.exists():
        return None
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    n_eval = int(qc.get("n_evaluated", 0)); n_valid = int(qc.get("n_valid_heatmaps", 0))
    expected = max(int(expected_gt_positive), 1)
    eval_coverage = n_eval / expected; valid_coverage = n_valid / expected
    layer_ok = False; layer_reason = "manifest_missing"
    layer_path = Path(layer_manifest_path)
    if layer_path.exists():
        try:
            layer = json.loads(layer_path.read_text(encoding="utf-8"))
            layer_ok = bool(
                str(layer.get("selection_split", "")).lower() == "validation"
                and layer.get("test_set_used_for_selection") is False
                and bool(layer.get("selected_models") or layer.get("selected_layers"))
                and np.isfinite(float(layer.get("selected_threshold", 0.5)))
                and bool(layer.get("validation_baseline_superiority_pass", False))
            )
            layer_reason = "ok" if layer_ok else "manifest_not_validation_locked_or_not_superior_to_strong_priors"
        except Exception as exc:
            layer_reason = f"manifest_parse_error:{exc}"
    mode = str(run_mode).upper()
    coverage_pass = bool(
        mode != "FINAL" or (
            eval_coverage >= float(min_final_coverage)
            and valid_coverage >= float(min_final_coverage)
            and n_valid >= int(min_final_valid_cases)
        )
    )
    baseline_mode = str(qc.get("baseline_qc_mode", "report")).lower()
    baseline_pass = bool(qc.get("baseline_superiority_qc_pass", qc.get("baseline_qc_pass", False)))
    baseline_enforcement_pass = bool(
        not require_baseline or (
            baseline_pass and (allow_baseline_report_only or baseline_mode in {"mean", "ci"})
        )
    )
    technical_pass = bool(qc.get("technical_qc_pass", qc.get("qc_pass", False)))
    validation_config_pass = bool(mode != "FINAL" or layer_ok)
    analysis_complete = bool(qc.get("xai_analysis_complete", True))
    localization_ready = bool(
        mode == "FINAL" and technical_pass and coverage_pass
        and validation_config_pass and baseline_enforcement_pass and analysis_complete
    )
    qc.update({
        "run_mode": mode,
        "artifact_status": "QUICK_PRECHECK_NOT_FOR_PAPER" if mode == "QUICK" else "FINAL_Q1_CANDIDATE",
        "expected_gt_positive_with_boxes": int(expected_gt_positive),
        "evaluation_coverage": float(eval_coverage),
        "valid_heatmap_coverage": float(valid_coverage),
        "minimum_final_coverage": float(min_final_coverage),
        "minimum_final_valid_cases": int(min_final_valid_cases),
        "sample_coverage_pass": coverage_pass,
        "validation_configuration_lock_pass": validation_config_pass,
        "validation_layer_lock_pass": validation_config_pass,
        "validation_layer_lock_reason": layer_reason,
        "baseline_enforcement_required": bool(require_baseline),
        "baseline_enforcement_pass": baseline_enforcement_pass,
        "technical_qc_pass": technical_pass,
        "pipeline_completed": True,
        "xai_analysis_complete": analysis_complete,
        "localization_claim_ready": localization_ready,
        "validated_localization_claim": localization_ready,
        "final_q1_ready": localization_ready,
        "paper_usage": "validated_localization" if localization_ready else "qualitative_or_limited_localization_claim",
        "qc_pass": technical_pass,
    })
    qc_path.write_text(json.dumps(qc, indent=2, default=_json_safe), encoding="utf-8")
    return qc


def build_xai_q1_results_table(reports_dir, *, run_mode, expected_gt_positive,
                               layer_manifest_path, probability_source_path):
    """Create CSV, JSON, and Markdown summaries."""
    reports_dir = Path(reports_dir)
    summary_path = reports_dir / "xai_localization_summary.csv"
    qc_path = reports_dir / "xai_quality_control.json"
    cmp_path = reports_dir / "xai_localization_baseline_comparison.csv"
    baseline_path = reports_dir / "xai_localization_baselines.csv"
    if not summary_path.exists() or not qc_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(cmp_path) if cmp_path.exists() else pd.DataFrame()
    baselines = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    layer = json.loads(Path(layer_manifest_path).read_text(encoding="utf-8")) if Path(layer_manifest_path).exists() else {}
    source = json.loads(Path(probability_source_path).read_text(encoding="utf-8")) if Path(probability_source_path).exists() else {}

    selected_cols = [
        "group", "n_total", "n_valid_heatmaps", "invalid_heatmap_rate",
        "pointing_hit_mean", "pointing_hit_mean_ci95_low", "pointing_hit_mean_ci95_high",
        "iou_at_thr_mean", "iou_at_thr_median", "iou_at_thr_q1", "iou_at_thr_q3",
        "iou_at_thr_mean_ci95_low", "iou_at_thr_mean_ci95_high",
        "localization_score_mean", "localization_score_median",
        "localization_score_mean_ci95_low", "localization_score_mean_ci95_high",
        "energy_inside_gt_mean", "energy_inside_gt_median",
        "energy_inside_gt_mean_ci95_low", "energy_inside_gt_mean_ci95_high",
        "peak_distance_norm_mean", "peak_distance_norm_median",
        "gt_coverage_at_thr_mean", "activation_precision_at_thr_mean",
        "outside_lung_ratio_raw_mean", "outside_lung_ratio_raw_median",
        "outside_lung_ratio_gated_mean",
    ]
    for col in selected_cols:
        if col not in summary.columns:
            summary[col] = np.nan
    final_metrics = summary[selected_cols].copy()

    prefix = "XAI_Q1_FINAL" if str(run_mode).upper() == "FINAL" else "XAI_QUICK_PRECHECK"
    csv_path = reports_dir / f"{prefix}_METRICS.csv"
    json_path = reports_dir / f"{prefix}_QC.json"
    md_path = reports_dir / f"{prefix}_RESULTS_TABLE.md"
    final_metrics.to_csv(csv_path, index=False)

    payload = {
        "xai_schema_version": XAI_SCHEMA_VERSION,
        "run_mode": str(run_mode).upper(),
        "expected_gt_positive_with_boxes": int(expected_gt_positive),
        "final_q1_ready": bool(qc.get("final_q1_ready", False)),
        "quality_control": qc,
        "probability_source": source,
        "cam_configuration_selection": layer,
        "baseline_methods": baselines.to_dict(orient="records") if len(baselines) else [],
        "paired_baseline_comparison": comparison.to_dict(orient="records") if len(comparison) else [],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=_json_safe), encoding="utf-8")

    verdict = "PASS — Q1 final XAI guardrails satisfied" if qc.get("final_q1_ready", False) else (
        "QUICK PRECHECK ONLY — NOT A FINAL PAPER RESULT" if str(run_mode).upper() == "QUICK"
        else "LIMITED CLAIM — analysis complete, validated localization guardrails not satisfied"
    )
    lines = [
        "# AURA-CXR XAI Results and Quality-Control Table",
        "",
        f"**Verdict:** {verdict}",
        "",
        f"- Run mode: `{str(run_mode).upper()}`",
        f"- Expected GT-positive cases with usable boxes: `{expected_gt_positive}`",
        f"- Evaluated cases: `{qc.get('n_evaluated', 0)}`",
        f"- Valid heatmaps: `{qc.get('n_valid_heatmaps', 0)}`",
        f"- Evaluation coverage: `{qc.get('evaluation_coverage', float('nan')):.3f}`",
        f"- Valid heatmap coverage: `{qc.get('valid_heatmap_coverage', float('nan')):.3f}`",
        f"- Stacked source: `{source.get('stacked_source', 'unknown')}`",
        f"- Validation configuration lock: `{qc.get('validation_configuration_lock_pass', False)}`",
        f"- Technical QC: `{qc.get('technical_qc_pass', False)}`",
        f"- Validated localization claim: `{qc.get('localization_claim_ready', False)}`",
        f"- Baseline QC mode/pass: `{qc.get('baseline_qc_mode', 'unknown')}` / `{qc.get('baseline_enforcement_pass', False)}`",
        "",
        "## Localization metrics",
        "",
        _df_to_markdown_simple(final_metrics),
        "",
        "## Required interpretation",
        "",
        "- Raw outside-lung attribution is a shortcut-bias diagnostic and is not hidden by lung gating.",
        "- Gated outside-lung attribution must remain near zero.",
        "- Fixed-threshold IoU and Max-IoU are both reported; Max-IoU must not replace fixed-threshold IoU.",
        "- QUICK outputs are diagnostic only and must not be cited as final test-set evidence.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] XAI summary table saved: {md_path}")
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path), "verdict": verdict}

def explain_failure(row, loc_score_threshold=0.10):
    label = int(row.get("label", -1))
    pred = int(row.get("pred", -1))
    cat = str(row.get("category", ""))
    gt_count = int(row.get("gt_box_count", 0))
    point = row.get("pointing_hit", np.nan)
    iou = row.get("iou_at_thr", np.nan)
    loc = row.get("localization_score", np.nan)
    energy = row.get("energy_inside_gt", np.nan)
    area = row.get("pred_area_ratio", np.nan)

    if cat == "FP":
        return "False positive: radiologist CSV has no pneumonia box, but model produced a high pneumonia probability. Heatmap likely follows non-lesion structure, projection artifact, device, border, or texture confounder."
    if cat == "FN":
        if gt_count > 0:
            return "False negative: pneumonia exists in GT, but model probability is below threshold. Check whether activation is weak, diffuse, or shifted away from the radiologist box."
        return "False negative by label, but no usable bbox was found in labels_csv for localization checking."
    if label == 1 and gt_count > 0:
        if np.isfinite(point) and int(point) == 0:
            return "Peak outside GT: Pointing Game failed; GradCAM focuses outside the radiologist lesion box. This is a localization failure even if classification is correct."
        if np.isfinite(loc) and float(loc) < loc_score_threshold:
            return "Low localization score: no heatmap threshold overlaps the radiologist box well; activation is likely displaced or too diffuse."
        if np.isfinite(iou) and float(iou) < loc_score_threshold:
            return "Low IoU@threshold: selected activation region only weakly overlaps the GT box; threshold choice or diffuse attention may be responsible."
        if np.isfinite(energy) and float(energy) < 0.20:
            return "Low energy inside GT: most activation mass lies outside the annotated lesion despite a possible peak hit."
    if np.isfinite(area) and float(area) > 0.45:
        return "Diffuse heatmap: high activation covers a large fraction of the image, making localization unreliable."
    return "Borderline case: classification and localization are not obviously wrong, but review is useful because metrics are weak or uncertain."


# -----------------------------------------------------------------------------
# Qualitative GradCAM grid with GT overlay and readable labels
# -----------------------------------------------------------------------------


def save_contrastive_gradcam_grid(
    model,
    preprocess_fn,
    sample_images_raw,
    sample_preds,
    sample_trues,
    candidate_layers,
    path,
    image_size=224,
    proba_list=None,
    patient_ids=None,
    image_paths=None,
    gt_boxes_list=None,
    heatmap_threshold_pct=0.60,
    paper_dpi=600,
    save_pdf=False,
    debug=False,
    sample_indices=None,
    models_pp_layers=None,
    lung_mask_dir=None,
    use_lung_mask=False,
    allow_ellipse=False,
    localization_cam_method="GradCAM",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
):
    """Compact paper-ready CAM audit with all metrics below the images."""
    n = len(sample_images_raw)
    if n == 0:
        print("[WARN] No XAI samples selected.")
        return pd.DataFrame()

    fig = plt.figure(figsize=(18.2, max(6.0, 2.78 * n + 1.25)))
    gs = gridspec.GridSpec(
        n, 5, figure=fig,
        width_ratios=[0.42, 1.0, 1.0, 1.0, 1.06],
        left=0.025, right=0.995, top=0.925, bottom=0.035,
        wspace=0.075, hspace=0.50,
    )
    metric_rows = []
    titles = [
        "Original + GT",
        "Pneumonia CAM",
        "Signed contrastive CAM",
        "Selected localization",
    ]

    for r, (raw_img, pred, true) in enumerate(zip(sample_images_raw, sample_preds, sample_trues)):
        raw_rgb = np.repeat(raw_img[..., None], 3, axis=-1) if raw_img.ndim == 2 else raw_img
        raw_u8 = to_uint8(raw_rgb)
        preproc = preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32))).numpy()
        hm_pneu, hm_npneu, _positive_contrast, used_layer, used_method = compute_contrastive_gradcam_robust(
            model, preproc, candidate_layers, debug=(debug and r == 0)
        )
        signed = normalize_minmax(hm_pneu) - normalize_minmax(hm_npneu)
        if models_pp_layers:
            loc_raw, loc_method, details = compute_class_localization_map(
                models_pp_layers, raw_rgb, image_size, class_index=1,
                smooth_sigma_frac=cam_smooth_sigma_frac,
                force_method=localization_cam_method, min_grid=cam_min_grid,
            )
        else:
            loc_raw = resize_heatmap(hm_pneu, image_size, image_size)
            loc_method = used_method
            details = []

        pid = patient_ids[r] if patient_ids is not None and r < len(patient_ids) else str(r)
        manifest = _CAM_CONFIG_LOCK or {}
        selected_policy = str(manifest.get("selected_map_policy", "hard" if use_lung_mask else "raw"))
        selected_threshold = float(manifest.get("selected_threshold", heatmap_threshold_pct))
        outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
        lung_mode, lung_mask = "off", None
        if use_lung_mask:
            lung_mask, lung_mode = load_lung_mask_for_patient(
                pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
            )
        loc_map = _apply_lung_constraint_policy(loc_raw, lung_mask, selected_policy, outside_weight)
        gt_boxes = gt_boxes_list[r] if gt_boxes_list is not None and r < len(gt_boxes_list) else []
        audit = compute_heatmap_audit(loc_map, lung_mask)
        metrics = compute_localization_metrics(
            loc_map, gt_boxes, threshold_pct=selected_threshold,
            heatmap_valid=audit["heatmap_valid"],
        )
        cat = _get_category(true, pred)
        sample_idx = sample_indices[r] if sample_indices is not None and r < len(sample_indices) else r
        prob = proba_list[r] if proba_list is not None else np.nan
        metric_rows.append({
            "sample_index": int(sample_idx), "patientId": pid, "label": int(true),
            "pred": int(pred), "category": cat, "prob_pneumonia": prob,
            "used_layer": used_layer, "used_method": used_method,
            "localization_map": loc_method, "lung_gate_mode": lung_mode,
            "heatmap_valid": audit["heatmap_valid"], **metrics,
        })

        pneu_overlay, _ = overlay_heatmap_rgba(raw_u8, hm_pneu, alpha=0.55)
        signed_overlay, _ = overlay_signed_heatmap(raw_u8, signed, alpha=0.48)
        loc_overlay, _ = overlay_heatmap_rgba(raw_u8, loc_map, alpha=0.58)
        imgs = [raw_u8, pneu_overlay, signed_overlay, loc_overlay]

        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis("off")
        label_ax.text(
            0.52, 0.5,
            f"{pid[:8]} | {cat}\nGT={CLASS_NAMES[int(true)]}\nP(pneu)={format_metric(prob)}",
            ha="center", va="center", rotation=90,
            fontsize=8.2, fontweight="bold",
            color=CATEGORY_COLORS.get(cat, "black"),
        )

        row_axes = [fig.add_subplot(gs[r, c]) for c in range(1, 5)]
        for c, (ax, im) in enumerate(zip(row_axes, imgs)):
            ax.imshow(im)
            _style_image_axis(
                ax,
                title=titles[c] if r == 0 else None,
                border_color=CATEGORY_COLORS.get(cat, "black"),
                title_size=10.5,
            )

        draw_boxes(row_axes[0], gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        draw_boxes(row_axes[1], gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        draw_boxes(row_axes[3], gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        annotate_pneumonia_location(
            row_axes[3], loc_map, threshold_pct=selected_threshold,
            label_text="CAM focus", color=PRED_COLOR, fontsize=7.5,
        )

        footer_metrics = (
            f"PG={format_metric(metrics.get('pointing_hit'), 0)} | "
            f"IoU={format_metric(metrics.get('iou_at_thr'))} | "
            f"Max-IoU={format_metric(metrics.get('localization_score'))} | "
            f"EnergyGT={format_metric(metrics.get('energy_inside_gt'))}\n"
            f"map={_compact_xai_text(loc_method, 34)} | "
            f"policy={selected_policy} | mask={_compact_xai_text(lung_mode, 30)}"
        )
        _add_metric_footer(row_axes[3], footer_metrics, fontsize=6.8, y=-0.12, width=58)

    df = pd.DataFrame(metric_rows)
    valid = df[df["heatmap_valid"].astype(bool)] if len(df) else df
    pg = pd.to_numeric(valid.get("pointing_hit"), errors="coerce").mean() if len(valid) else np.nan
    miou = pd.to_numeric(valid.get("iou_at_thr"), errors="coerce").mean() if len(valid) else np.nan
    mloc = pd.to_numeric(valid.get("localization_score"), errors="coerce").mean() if len(valid) else np.nan
    fig.suptitle(
        f"Validation-Selected CNN Pneumonia CAM Audit — n={n}, valid={len(valid)}, "
        f"PG={format_metric(pg)}, IoU={format_metric(miou)}, Max-IoU={format_metric(mloc)}",
        fontsize=14.5, fontweight="bold", y=0.978,
    )
    fig.text(
        0.51, 0.008,
        "Signed contrastive CAM: red supports pneumonia; blue supports non-pneumonia. "
        "Technical metrics are reported below the selected-localization panel.",
        ha="center", va="bottom", fontsize=8.8,
    )
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] Q1 qualitative localization grid saved: {Path(path)}")
    return df


def _select_balanced_cam_pdf_indices(test_df, y_true, y_pred, n_cases):
    """Deterministically select up to n_cases with balanced TP/TN/FP/FN coverage.

    Selection is for qualitative reporting only and never changes the locked model,
    threshold, validation-selected CAM configuration, or quantitative XAI cohort.
    """
    n_cases = max(0, int(n_cases))
    if n_cases == 0:
        return []
    cats = {
        "TP": (1, 1),
        "TN": (0, 0),
        "FP": (0, 1),
        "FN": (1, 0),
    }
    quota_base = n_cases // 4
    quota_extra = n_cases % 4
    quotas = {k: quota_base + (1 if i < quota_extra else 0) for i, k in enumerate(cats)}
    selected, used = [], set()
    for cat, (tgt_t, tgt_p) in cats.items():
        need = quotas[cat]
        for idx in range(len(test_df)):
            if need <= 0:
                break
            if idx in used:
                continue
            if int(y_true[idx]) != tgt_t or int(y_pred[idx]) != tgt_p:
                continue
            path = str(test_df.iloc[idx].get("image_path", ""))
            if not path or not Path(path).exists():
                continue
            selected.append(idx)
            used.add(idx)
            need -= 1
    if len(selected) < n_cases:
        for idx in range(len(test_df)):
            if len(selected) >= n_cases:
                break
            if idx in used:
                continue
            path = str(test_df.iloc[idx].get("image_path", ""))
            if not path or not Path(path).exists():
                continue
            selected.append(idx)
            used.add(idx)
    return selected[:n_cases]




def _safe_metric_num(val, default=0.0):
    try:
        f = float(val)
        return f if np.isfinite(f) else float(default)
    except Exception:
        return float(default)


def _cam_quality_score_from_metrics_row(row):
    """Composite localization score for ranking qualitative CAM examples.

    Higher is better. The score rewards hit/overlap/coverage and lightly penalizes
    diffuse outside-lung activation and large peak-to-box distance.
    """
    pointing = _safe_metric_num(row.get("pointing_hit"), 0.0)
    iou = _safe_metric_num(row.get("iou_at_thr"), 0.0)
    loc = _safe_metric_num(row.get("localization_score"), 0.0)
    energy = _safe_metric_num(row.get("energy_inside_gt"), 0.0)
    outside = _safe_metric_num(
        row.get("outside_lung_ratio_selected", row.get("outside_lung_ratio_gated", 0.0)), 0.0
    )
    peak_dist = _safe_metric_num(row.get("peak_distance_norm"), 1.0)
    heatmap_valid = 1.0 if bool(row.get("heatmap_valid", True)) else 0.0
    score = (
        0.35 * pointing +
        0.25 * iou +
        0.20 * loc +
        0.15 * energy +
        0.05 * heatmap_valid -
        0.10 * outside -
        0.05 * peak_dist
    )
    return float(score)


def _rank_final_locked_cam_metrics(metrics_df, require_valid=True):
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    df = metrics_df.copy()
    if "label" in df.columns:
        df = df[pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int) == 1].copy()
    if require_valid and "heatmap_valid" in df.columns:
        df = df[df["heatmap_valid"].astype(bool)].copy()
    if len(df) == 0:
        return df
    df["cam_quality_score"] = df.apply(_cam_quality_score_from_metrics_row, axis=1)
    sort_cols = ["cam_quality_score", "pointing_hit", "iou_at_thr", "localization_score", "energy_inside_gt"]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(sort_cols, ascending=[False, False, False, False, False], na_position="last")
    return df.reset_index(drop=True)


def _select_top_cam_pdf_indices_from_metrics(metrics_df, n_cases, require_valid=True):
    ranked = _rank_final_locked_cam_metrics(metrics_df, require_valid=require_valid)
    if len(ranked) == 0:
        return []
    n_cases = max(0, int(n_cases))
    if n_cases == 0:
        return []
    idxs = pd.to_numeric(ranked.get("sample_index"), errors="coerce").dropna().astype(int).tolist()
    return idxs[:n_cases]


def _freeze_jsonable(obj):
    return json.dumps(_json_safe(obj), sort_keys=True)


def _collect_oracle_candidate_pool(cam_manifest, topk_model_candidates=2, topk_fusion_candidates=5):
    """Build a practical per-case oracle pool from validation-eligible CAM candidates.

    The pool is exploratory only. It is derived from validation-ranked candidate summaries
    saved in the manifest, not from test-time reselection.
    """
    cam_manifest = cam_manifest or {}
    selected_models = cam_manifest.get("selected_models", {}) or {}
    per_model = {}
    for model_name, selected_cfg in selected_models.items():
        model_pool = []
        seen = set()
        if selected_cfg:
            key = _freeze_jsonable(selected_cfg)
            model_pool.append(selected_cfg)
            seen.add(key)
        audit = (cam_manifest.get("models", {}) or {}).get(model_name, {}) or {}
        for row in audit.get("candidate_summary", [])[:50]:
            cfg = {
                "layers": list(row.get("layers") or []),
                "methods": list(row.get("methods") or []),
                "layer_weights": [float(x) for x in (row.get("layer_weights") or [1.0])],
                "configuration_type": row.get("configuration_type"),
                "validation_objective": float(row.get("selection_objective", np.nan)),
            }
            key = _freeze_jsonable(cfg)
            if key in seen:
                continue
            model_pool.append(cfg)
            seen.add(key)
            if len(model_pool) >= max(1, int(topk_model_candidates)):
                break
        per_model[model_name] = model_pool

    fusion_pool = []
    seen_fusion = set()
    selected_fusion = {
        "model_weights": cam_manifest.get("selected_model_weights", {}),
        "map_policy": cam_manifest.get("selected_map_policy", "raw"),
        "heatmap_threshold_pct": cam_manifest.get("selected_threshold", 0.60),
    }
    fusion_pool.append(selected_fusion)
    seen_fusion.add(_freeze_jsonable(selected_fusion))
    for row in (cam_manifest.get("fusion_candidate_summary", []) or [])[:100]:
        cfg = {
            "model_weights": {k: float(v) for k, v in (row.get("model_weights") or {}).items()},
            "map_policy": str(row.get("map_policy", cam_manifest.get("selected_map_policy", "raw"))),
            "heatmap_threshold_pct": float(row.get("heatmap_threshold_pct", cam_manifest.get("selected_threshold", 0.60))),
        }
        key = _freeze_jsonable(cfg)
        if key in seen_fusion:
            continue
        fusion_pool.append(cfg)
        seen_fusion.add(key)
        if len(fusion_pool) >= max(1, int(topk_fusion_candidates)):
            break
    return per_model, fusion_pool


def save_oracle_best_cam_multipage_pdf(
    test_df,
    y_true,
    y_pred,
    probas,
    indices,
    models_pp_layers,
    gt_box_map,
    output_pdf,
    reports_dir,
    image_size=224,
    lung_mask_dir=None,
    use_lung_mask=False,
    allow_ellipse=False,
    localization_cam_method="validation_selected",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
    cases_per_page=4,
    topk_model_candidates=2,
    topk_fusion_candidates=5,
):
    """Exploratory oracle PDF: best per-case CAM chosen from a candidate pool.

    This export is for qualitative appendix/showcase only and must not replace the
    primary validation-locked CAM analysis in the thesis text.
    """
    if not indices:
        print("[CAM-PDF][ORACLE][WARN] No cases selected for oracle PDF.")
        return pd.DataFrame()
    manifest = _CAM_CONFIG_LOCK or {}
    per_model_pool, fusion_pool = _collect_oracle_candidate_pool(
        manifest,
        topk_model_candidates=topk_model_candidates,
        topk_fusion_candidates=topk_fusion_candidates,
    )
    if not per_model_pool or not fusion_pool:
        print("[CAM-PDF][ORACLE][WARN] Candidate pools are empty; oracle PDF skipped.")
        return pd.DataFrame()

    model_lookup = {_model_cam_key(m): (m, pp, cand) for (m, pp, cand) in models_pp_layers}
    model_names = [name for name in per_model_pool.keys() if name in model_lookup]
    if not model_names:
        print("[CAM-PDF][ORACLE][WARN] No selected CNN models matched the runtime model list.")
        return pd.DataFrame()

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = ensure_dir(reports_dir)
    cases_per_page = max(1, int(cases_per_page))
    rows = []

    with PdfPages(output_pdf) as pdf:
        for page_start in range(0, len(indices), cases_per_page):
            page_indices = indices[page_start:page_start + cases_per_page]
            nrows = len(page_indices)
            fig, axes = plt.subplots(nrows, 2, figsize=(8.27, max(5.5, 3.35 * nrows + 0.8)))
            if nrows == 1:
                axes = np.asarray([axes])
            fig.suptitle(
                f"AURA-CXR — Oracle Best-per-Case CAM (exploratory) | cases {page_start + 1}–{page_start + nrows} of {len(indices)}\\n"
                f"candidate pool: top-{int(topk_model_candidates)} model CAM configs/model × top-{int(topk_fusion_candidates)} fusion configs",
                fontsize=11.5, fontweight="bold", y=0.995,
            )

            for rr, idx in enumerate(page_indices):
                row = test_df.iloc[int(idx)]
                path = str(row["image_path"])
                pid = str(row["patientId"])
                raw_img = load_dicom_grayscale(path, image_size)
                raw_rgb = np.repeat(raw_img[..., None], 3, axis=-1) if raw_img.ndim == 2 else raw_img
                raw_u8 = to_uint8(raw_rgb)
                gt_boxes = scale_boxes_to_image(pid, path, gt_box_map, image_size)
                lung_mode, lung_mask = "off", None
                if use_lung_mask:
                    lung_mask, lung_mode = load_lung_mask_for_patient(
                        pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
                    )

                model_cache = {}
                best = None
                for fusion_cfg in fusion_pool:
                    for cfg_bundle in itertools.product(*[per_model_pool[name] for name in model_names]):
                        maps, ws, details_all = [], [], []
                        missing_required = []
                        for name, model_cfg in zip(model_names, cfg_bundle):
                            cache_key = (name, _freeze_jsonable(model_cfg))
                            if cache_key not in model_cache:
                                model, preprocess_fn, cand = model_lookup[name]
                                hm, details = _compute_model_cam_from_config(
                                    model, preprocess_fn, cand, raw_rgb, image_size,
                                    model_cfg, allow_fallback=False, warn_on_failure=False,
                                )
                                model_cache[cache_key] = (hm, details)
                            hm, details = model_cache[cache_key]
                            details_all.extend([{**d, "model": name} for d in details])
                            weight = float((fusion_cfg.get("model_weights") or {}).get(name, 0.0))
                            if weight <= 0:
                                continue
                            if hm is None:
                                missing_required.append(name)
                                continue
                            maps.append(hm)
                            ws.append(weight)
                        if missing_required or not maps:
                            continue
                        w = np.asarray(ws, dtype=float)
                        w = w / w.sum() if w.sum() > 0 else np.ones(len(maps), dtype=float) / len(maps)
                        fused = normalize_minmax(np.tensordot(w, np.stack(maps), axes=(0, 0)).astype(np.float32))
                        policy = str(fusion_cfg.get("map_policy", manifest.get("selected_map_policy", "raw")))
                        outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
                        loc_map = _apply_lung_constraint_policy(fused, lung_mask, policy, outside_weight)
                        audit = compute_heatmap_audit(loc_map, lung_mask)
                        threshold = float(fusion_cfg.get("heatmap_threshold_pct", manifest.get("selected_threshold", 0.60)))
                        metrics = compute_localization_metrics(
                            loc_map, gt_boxes, threshold_pct=threshold,
                            heatmap_valid=audit["heatmap_valid"],
                        )
                        eval_row = {
                            "outside_lung_ratio_selected": audit.get("outside_lung_ratio", np.nan),
                            "peak_distance_norm": metrics.get("peak_distance_norm", np.nan),
                            "heatmap_valid": bool(audit["heatmap_valid"]),
                            **metrics,
                        }
                        score = _cam_quality_score_from_metrics_row(eval_row)
                        candidate = {
                            "score": score,
                            "loc_map": loc_map,
                            "metrics": metrics,
                            "audit": audit,
                            "fusion_cfg": fusion_cfg,
                            "model_cfgs": {name: cfg for name, cfg in zip(model_names, cfg_bundle)},
                            "details": details_all,
                        }
                        if best is None or candidate["score"] > best["score"]:
                            best = candidate

                if best is None:
                    print(f"[CAM-PDF][ORACLE][WARN] No valid oracle candidate for sample index {idx} ({pid}).")
                    continue

                true = int(y_true[int(idx)])
                pred = int(y_pred[int(idx)])
                prob = float(probas[int(idx)])
                cat = _get_category(true, pred)
                overlay, _ = overlay_heatmap_rgba(raw_u8, best["loc_map"], alpha=0.58)
                ax0, ax1 = axes[rr, 0], axes[rr, 1]
                ax0.imshow(raw_u8)
                ax1.imshow(overlay)
                _style_image_axis(ax0, title="Original + GT", border_color=CATEGORY_COLORS.get(cat, "black"), title_size=9.0)
                _style_image_axis(ax1, title="Oracle best-per-case CAM", border_color=CATEGORY_COLORS.get(cat, "black"), title_size=9.0)
                draw_boxes(ax0, gt_boxes, color=GT_COLOR, label="GT", fontsize=6.5)
                draw_boxes(ax1, gt_boxes, color=GT_COLOR, label="GT", fontsize=6.5)
                annotate_pneumonia_location(
                    ax1, best["loc_map"], threshold_pct=float(best["fusion_cfg"].get("heatmap_threshold_pct", manifest.get("selected_threshold", 0.60))),
                    label_text="CAM focus", color=PRED_COLOR, fontsize=6.5,
                )
                weight_text = ", ".join(f"{k}={float(v):.2f}" for k, v in (best["fusion_cfg"].get("model_weights") or {}).items())
                metric_text = (
                    f"#{page_start + rr + 1:03d} | {pid[:12]} | {cat} | GT={CLASS_NAMES[true]} | P(pneu)={prob:.3f}\\n"
                    f"score={best['score']:.4f} | PG={format_metric(best['metrics'].get('pointing_hit'), 0)} | IoU={format_metric(best['metrics'].get('iou_at_thr'))} | "
                    f"Max-IoU={format_metric(best['metrics'].get('localization_score'))} | EnergyGT={format_metric(best['metrics'].get('energy_inside_gt'))}\\n"
                    f"policy={best['fusion_cfg'].get('map_policy')} | thr={float(best['fusion_cfg'].get('heatmap_threshold_pct', np.nan)):.2f} | weights: {weight_text}"
                )
                ax1.text(0.5, -0.12, metric_text, transform=ax1.transAxes, ha="center", va="top", fontsize=6.4, wrap=True)
                rows.append({
                    "pdf_order": page_start + rr + 1,
                    "sample_index": int(idx),
                    "patientId": pid,
                    "label": true,
                    "pred": pred,
                    "category": cat,
                    "prob_pneumonia": prob,
                    "cam_quality_score": float(best["score"]),
                    "localization_map": "oracle_best_per_case",
                    "lung_gate_mode": lung_mode,
                    "heatmap_valid": bool(best["audit"]["heatmap_valid"]),
                    "selected_map_policy": str(best["fusion_cfg"].get("map_policy")),
                    "selected_threshold": float(best["fusion_cfg"].get("heatmap_threshold_pct", np.nan)),
                    "selected_model_weights": json.dumps(best["fusion_cfg"].get("model_weights", {}), sort_keys=True),
                    "selected_model_configs": json.dumps(best["model_cfgs"], default=_json_safe, sort_keys=True),
                    **best["metrics"],
                })

            fig.subplots_adjust(top=0.93, bottom=0.06, hspace=0.48, wspace=0.10)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)

    print(f"[CAM-PDF][ORACLE][OK] Multipage oracle CAM PDF saved: {output_pdf} | n={len(rows)} | per_page={cases_per_page}")
    return pd.DataFrame(rows)

def save_fused_cam_multipage_pdf(
    test_df,
    y_true,
    y_pred,
    probas,
    indices,
    models_pp_layers,
    gt_box_map,
    output_pdf,
    image_size=224,
    lung_mask_dir=None,
    use_lung_mask=False,
    allow_ellipse=False,
    heatmap_threshold_pct=0.60,
    localization_cam_method="GradCAM",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
    cases_per_page=4,
):
    """Write a multipage four-stage Hybrid CNN-ViT CAM comparison.

    Each case is shown horizontally as Before XAI, After XRV, After EVA-X, and
    After Fused Hybrid CNN-ViT. This export is qualitative only; it does not subsample
    or modify the quantitative XAI evaluation cohort.
    """
    if not indices:
        print("[CAM-PDF][WARN] No cases selected for multipage PDF.")
        return pd.DataFrame()
    if not models_pp_layers:
        raise RuntimeError("[CAM-PDF] models_pp_layers is required for fused hybrid CNN-ViT CAM export.")

    manifest = _CAM_CONFIG_LOCK or {}
    selected_policy = str(manifest.get("selected_map_policy", "hard" if use_lung_mask else "raw"))
    selected_threshold = float(manifest.get("selected_threshold", heatmap_threshold_pct))
    outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
    selected_weights = manifest.get("selected_model_weights", {}) or {}
    weight_text = ", ".join(f"{k}={float(v):.2f}" for k, v in selected_weights.items()) if selected_weights else "manifest weights"

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cases_per_page = max(1, int(cases_per_page))
    rows = []

    with PdfPages(output_pdf) as pdf:
        for page_start in range(0, len(indices), cases_per_page):
            page_indices = indices[page_start:page_start + cases_per_page]
            nrows = len(page_indices)
            fig, axes = plt.subplots(nrows, 4, figsize=(11.69, max(5.5, 2.95 * nrows + 0.9)))
            if nrows == 1:
                axes = np.asarray([axes])
            fig.suptitle(
                f"AURA-CXR — Hybrid CNN-ViT XAI: Before → XRV → EVA-X → Fused | cases {page_start + 1}–{page_start + nrows} of {len(indices)}\n"
                f"validation-locked fusion: {weight_text}",
                fontsize=11.2, fontweight="bold", y=0.995,
            )

            for rr, idx in enumerate(page_indices):
                row = test_df.iloc[idx]
                path = str(row["image_path"])
                pid = str(row["patientId"])
                raw_img = load_dicom_grayscale(path, image_size)
                raw_rgb = np.repeat(raw_img[..., None], 3, axis=-1) if raw_img.ndim == 2 else raw_img
                raw_u8 = to_uint8(raw_rgb)

                component_maps, loc_raw, loc_method, component_details, details = compute_dual_cnn_component_maps(
                    models_pp_layers,
                    raw_rgb,
                    image_size,
                    smooth_sigma_frac=cam_smooth_sigma_frac,
                    force_method=localization_cam_method,
                    min_grid=cam_min_grid,
                )
                xrv_key, eva_key = _resolve_dual_cnn_visual_branches(component_maps)
                if xrv_key is None or eva_key is None:
                    raise RuntimeError(
                        f"[CAM-PDF] Expected XRV and EVA-X component maps; got keys={list(component_maps)}"
                    )
                lung_mode, lung_mask = "off", None
                if use_lung_mask:
                    lung_mask, lung_mode = load_lung_mask_for_patient(
                        pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
                    )
                xrv_map = _apply_lung_constraint_policy(component_maps[xrv_key], lung_mask, selected_policy, outside_weight)
                eva_map = _apply_lung_constraint_policy(component_maps[eva_key], lung_mask, selected_policy, outside_weight)
                loc_map = _apply_lung_constraint_policy(loc_raw, lung_mask, selected_policy, outside_weight)
                gt_boxes = scale_boxes_to_image(pid, path, gt_box_map, image_size)
                audit = compute_heatmap_audit(loc_map, lung_mask)
                metrics = compute_localization_metrics(
                    loc_map, gt_boxes, threshold_pct=selected_threshold,
                    heatmap_valid=audit["heatmap_valid"],
                )
                true = int(y_true[idx])
                pred = int(y_pred[idx])
                prob = float(probas[idx])
                cat = _get_category(true, pred)
                xrv_overlay, _ = overlay_heatmap_rgba(raw_u8, xrv_map, alpha=0.58)
                eva_overlay, _ = overlay_heatmap_rgba(raw_u8, eva_map, alpha=0.58)
                fused_overlay, _ = overlay_heatmap_rgba(raw_u8, loc_map, alpha=0.58)

                row_axes = [axes[rr, c] for c in range(4)]
                panel_images = [raw_u8, xrv_overlay, eva_overlay, fused_overlay]
                panel_titles = [
                    "Before XAI",
                    "After XRV",
                    "After EVA-X",
                    "After Fused Hybrid CNN-ViT",
                ]
                for cc, (ax, panel_img) in enumerate(zip(row_axes, panel_images)):
                    ax.imshow(panel_img)
                    _style_image_axis(
                        ax,
                        title=panel_titles[cc] if rr == 0 else None,
                        border_color=CATEGORY_COLORS.get(cat, "black"),
                        title_size=8.5,
                    )
                    draw_boxes(ax, gt_boxes, color=GT_COLOR, label="GT", fontsize=5.8)
                annotate_pneumonia_location(
                    row_axes[3], loc_map, threshold_pct=selected_threshold,
                    label_text="Fused CAM focus", color=PRED_COLOR, fontsize=5.8,
                )
                info = (
                    f"#{page_start + rr + 1:03d} | {pid[:12]} | {cat} | "
                    f"GT={CLASS_NAMES[true]} | P(pneu)={prob:.3f}\n"
                    f"PG={format_metric(metrics.get('pointing_hit'), 0)} | "
                    f"IoU={format_metric(metrics.get('iou_at_thr'))} | "
                    f"Max-IoU={format_metric(metrics.get('localization_score'))} | "
                    f"EnergyGT={format_metric(metrics.get('energy_inside_gt'))} | "
                    f"valid={bool(audit['heatmap_valid'])}"
                )
                row_axes[3].text(
                    0.5, -0.10, info, transform=row_axes[3].transAxes,
                    ha="center", va="top", fontsize=6.2, wrap=True,
                )
                rows.append({
                    "pdf_order": page_start + rr + 1,
                    "sample_index": int(idx),
                    "patientId": pid,
                    "label": true,
                    "pred": pred,
                    "category": cat,
                    "prob_pneumonia": prob,
                    "localization_map": loc_method,
                    "xrv_model_key": xrv_key,
                    "eva_model_key": eva_key,
                    "xrv_cam_valid": bool(compute_heatmap_audit(xrv_map, lung_mask)["heatmap_valid"]),
                    "eva_cam_valid": bool(compute_heatmap_audit(eva_map, lung_mask)["heatmap_valid"]),
                    "lung_gate_mode": lung_mode,
                    "heatmap_valid": bool(audit["heatmap_valid"]),
                    **metrics,
                })

            fig.subplots_adjust(top=0.93, bottom=0.065, hspace=0.44, wspace=0.06)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)

    print(f"[CAM-PDF][OK] Multipage fused-CAM PDF saved: {output_pdf} | n={len(rows)} | per_page={cases_per_page}")
    return pd.DataFrame(rows)


def save_dual_cnn_four_stage_grid(
    test_df,
    y_true,
    y_pred,
    probas,
    indices,
    models_pp_layers,
    gt_box_map,
    path,
    image_size=224,
    lung_mask_dir=None,
    use_lung_mask=False,
    allow_ellipse=False,
    heatmap_threshold_pct=0.60,
    localization_cam_method="validation_selected",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
    paper_dpi=600,
    save_pdf=False,
    dataset_name="RSNA locked test",
    show_gt_boxes=True,
):
    """Paper-ready four-stage XAI comparison: Before, XRV, EVA-X, and fused.

    Each row is one CXR case and all four panels are aligned horizontally so the
    branch-specific attention patterns and their final weighted fusion can be
    compared directly. This export is qualitative only and does not change the
    locked classifier, CAM configuration, or quantitative XAI metrics.
    """
    indices = [int(i) for i in indices]
    if not indices:
        print("[4-STAGE-XAI][WARN] No cases selected.")
        return pd.DataFrame()
    if not models_pp_layers:
        raise RuntimeError("[4-STAGE-XAI] models_pp_layers is required.")

    manifest = _CAM_CONFIG_LOCK or {}
    selected_policy = str(manifest.get("selected_map_policy", "hard" if use_lung_mask else "raw"))
    selected_threshold = float(manifest.get("selected_threshold", heatmap_threshold_pct))
    outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
    selected_weights = manifest.get("selected_model_weights", {}) or {}

    nrows = len(indices)
    fig, axes = plt.subplots(nrows, 4, figsize=(13.2, max(3.6, 3.0 * nrows + 1.0)))
    if nrows == 1:
        axes = np.asarray([axes])

    rows = []
    column_titles = [
        "Before XAI\nOriginal CXR",
        "After XRV\nDenseNet121-XRV",
        "After EVA-X\nEVA-X-S",
        "After Fused CNN-ViT\nValidation-locked fusion",
    ]

    for rr, idx in enumerate(indices):
        row = test_df.iloc[idx]
        image_path = str(row["image_path"])
        pid = str(row["patientId"])
        raw_img = load_dicom_grayscale(image_path, image_size)
        raw_rgb = np.repeat(raw_img[..., None], 3, axis=-1) if raw_img.ndim == 2 else raw_img
        raw_u8 = to_uint8(raw_rgb)

        component_maps, fused_raw, fused_method, component_details, fused_details = compute_dual_cnn_component_maps(
            models_pp_layers,
            raw_rgb,
            image_size,
            smooth_sigma_frac=cam_smooth_sigma_frac,
            force_method=localization_cam_method,
            min_grid=cam_min_grid,
        )
        xrv_key, eva_key = _resolve_dual_cnn_visual_branches(component_maps)
        if xrv_key is None or eva_key is None:
            raise RuntimeError(
                f"[4-STAGE-XAI] Expected two CAM-capable CNN branches; got keys={list(component_maps)}"
            )

        lung_mode, lung_mask = "off", None
        if use_lung_mask:
            lung_mask, lung_mode = load_lung_mask_for_patient(
                pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
            )

        xrv_map = _apply_lung_constraint_policy(component_maps[xrv_key], lung_mask, selected_policy, outside_weight)
        eva_map = _apply_lung_constraint_policy(component_maps[eva_key], lung_mask, selected_policy, outside_weight)
        fused_map = _apply_lung_constraint_policy(fused_raw, lung_mask, selected_policy, outside_weight)

        gt_boxes = scale_boxes_to_image(pid, image_path, gt_box_map, image_size)
        fused_audit = compute_heatmap_audit(fused_map, lung_mask)
        fused_metrics = compute_localization_metrics(
            fused_map,
            gt_boxes,
            threshold_pct=selected_threshold,
            heatmap_valid=fused_audit["heatmap_valid"],
        )

        true = int(y_true[idx])
        pred = int(y_pred[idx])
        prob = float(probas[idx])
        cat = _get_category(true, pred)

        xrv_overlay, _ = overlay_heatmap_rgba(raw_u8, xrv_map, alpha=0.58)
        eva_overlay, _ = overlay_heatmap_rgba(raw_u8, eva_map, alpha=0.58)
        fused_overlay, _ = overlay_heatmap_rgba(raw_u8, fused_map, alpha=0.58)
        panel_images = [raw_u8, xrv_overlay, eva_overlay, fused_overlay]

        for cc, (ax, panel_img) in enumerate(zip(axes[rr], panel_images)):
            ax.imshow(panel_img)
            _style_image_axis(
                ax,
                title=column_titles[cc] if rr == 0 else None,
                border_color=CATEGORY_COLORS.get(cat, "black"),
                title_size=10.0,
            )
            if show_gt_boxes:
                draw_boxes(ax, gt_boxes, color=GT_COLOR, label="GT", fontsize=6.2)

        annotate_pneumonia_location(
            axes[rr, 3],
            fused_map,
            threshold_pct=selected_threshold,
            label_text="Fused CAM focus",
            color=PRED_COLOR,
            fontsize=6.2,
        )

        xrv_weight = float(selected_weights.get(xrv_key, np.nan)) if selected_weights else np.nan
        eva_weight = float(selected_weights.get(eva_key, np.nan)) if selected_weights else np.nan
        if show_gt_boxes:
            row_text = (
                f"{pid[:12]} | {cat} | GT={CLASS_NAMES[true]} | P(pneu)={prob:.3f} | "
                f"PG={format_metric(fused_metrics.get('pointing_hit'), 0)} | "
                f"IoU={format_metric(fused_metrics.get('iou_at_thr'))} | "
                f"EnergyGT={format_metric(fused_metrics.get('energy_inside_gt'))} | "
                f"w(XRV)={format_metric(xrv_weight, 2)} | w(EVA)={format_metric(eva_weight, 2)}"
            )
        else:
            row_text = (
                f"{pid[:24]} | {cat} | GT={CLASS_NAMES[true]} | P(pneu)={prob:.3f} | "
                f"w(XRV)={format_metric(xrv_weight, 2)} | w(EVA)={format_metric(eva_weight, 2)}"
            )
        axes[rr, 3].text(
            0.5, -0.105, row_text,
            transform=axes[rr, 3].transAxes,
            ha="center", va="top", fontsize=6.4, wrap=True,
        )

        rows.append({
            "figure_order": rr + 1,
            "sample_index": int(idx),
            "patientId": pid,
            "label": true,
            "pred": pred,
            "category": cat,
            "prob_pneumonia": prob,
            "xrv_model_key": xrv_key,
            "eva_model_key": eva_key,
            "xrv_cam_valid": bool(compute_heatmap_audit(xrv_map, lung_mask)["heatmap_valid"]),
            "eva_cam_valid": bool(compute_heatmap_audit(eva_map, lung_mask)["heatmap_valid"]),
            "fused_cam_valid": bool(fused_audit["heatmap_valid"]),
            "fused_localization_map": fused_method,
            "lung_gate_mode": lung_mode,
            "selected_map_policy": selected_policy,
            "selected_threshold": selected_threshold,
            "xrv_weight": xrv_weight,
            "eva_weight": eva_weight,
            **fused_metrics,
        })

    fig.suptitle(
        f"Hybrid CNN-ViT XAI Progression ({dataset_name}): Before XAI → XRV → EVA-X → Fused Hybrid CNN-ViT",
        fontsize=14.0, fontweight="bold", y=0.992,
    )
    footer_text = (
        "All four panels in each row use the same CXR and GT box. Branch CAMs reuse the validation-selected CAM configuration; "
        "the fused panel is the same validation-locked Hybrid CNN-ViT localization map used by the quantitative XAI pipeline."
        if show_gt_boxes else
        "All four panels in each row use the same external-cohort CXR. Branch CAMs reuse the validation-selected CAM configuration; "
        "the fused panel is the same validation-locked Hybrid CNN-ViT localization map. Kermany has no RSNA-style lesion boxes, so the figure is qualitative only."
    )
    fig.text(
        0.5, 0.006,
        footer_text,
        ha="center", va="bottom", fontsize=8.0,
    )
    fig.subplots_adjust(top=0.93, bottom=0.075, hspace=0.46, wspace=0.06)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[4-STAGE-XAI][OK] Four-stage XAI grid saved: {Path(path)} | n={len(rows)}")
    return pd.DataFrame(rows)




def build_kermany_df(image_dir, max_samples=0):
    """Build a dataframe for the official Kermany external test split.

    ``image_dir`` may point either to the dataset root containing ``test/`` or
    directly to the ``test`` directory itself.
    """
    root = Path(image_dir)
    base = root if (root / "NORMAL").is_dir() and (root / "PNEUMONIA").is_dir() else root / "test"
    rows = []
    for label, cls_name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        cls_dir = base / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Kermany class directory not found: {cls_dir}. Expected <image_dir>/test/NORMAL and PNEUMONIA, or image_dir itself to be the test folder."
            )
        files = []
        for pat in ("*.jpeg", "*.jpg", "*.png", "*.bmp", "*.dcm"):
            files.extend(cls_dir.rglob(pat))
        for fp in sorted(set(files)):
            patient_id = f"KERMANY_{cls_name}_{fp.stem}"
            rows.append({"patientId": patient_id, "image_path": str(fp), "label": int(label), "source": "Kermany"})
    df = pd.DataFrame(rows)
    if df.empty or len(df["label"].unique()) < 2:
        raise ValueError("Kermany test set is empty or incomplete")
    if max_samples and len(df) > int(max_samples):
        sampled = []
        for _, g in df.groupby("label"):
            n = max(1, round(int(max_samples) * len(g) / len(df)))
            sampled.append(g.sample(min(n, len(g)), random_state=42))
        df = pd.concat(sampled, ignore_index=True)
    return df.reset_index(drop=True)


def _resolve_train_script_path(user_path=""):
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"--train_script not found: {user_path}")
    here = Path(__file__).resolve().parent
    candidates = [
        here / "train_aura.py",
        here / "train_aura (3).py",
        Path.cwd() / "train_aura.py",
        Path.cwd() / "train_aura (3).py",
        Path.cwd() / "ardex_cxr.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not auto-detect train_aura.py for Kermany radiomics inference. "
        "Provide --train_script explicitly."
    )


def _load_train_module_for_external(train_script_path):
    return _import_module_from_path(train_script_path, "aura_train_external_xai")


def _radiomics_feature_config_from_deployment(deployment):
    cfg = dict(deployment.get("radiomics_selection", {}).get("feature_config", {}))
    required = ["wavelet_levels", "glcm_distances", "glcm_angles_deg", "lbp_radius", "lbp_n_points"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise RuntimeError(f"Locked deployment lacks radiomics feature configuration: {missing}")
    return cfg


def _predict_selected_model_probabilities(df, model_name, model_obj, preprocess_fn, batch_size=16, image_size=224):
    probs = []
    for start in range(0, len(df), max(1, int(batch_size))):
        sub = df.iloc[start:start + max(1, int(batch_size))]
        rgb = np.stack([
            np.repeat(load_dicom_grayscale(p, image_size)[..., None], 3, axis=-1)
            for p in sub["image_path"].astype(str).tolist()
        ]).astype(np.float32)
        if getattr(model_obj, "_is_torch_adapter", False):
            out = model_obj.predict_numpy(rgb, mc_passes=1)
        else:
            xb = preprocess_fn(tf.constant(rgb))
            out = np.asarray(model_obj(xb, training=False), dtype=np.float32)
        out = normalize_binary_probability(out, len(sub), f"external {model_name}")
        probs.append(out[:, 1])
    return np.concatenate(probs).astype(np.float32)


def _infer_kermany_stacked_probability(df, selected_dl_models, cnn_models, radiomics_model_path,
                                       deployment, meta_clf, args):
    dl_probs = {}
    for name in selected_dl_models:
        dl_probs[name] = _predict_selected_model_probabilities(
            df=df,
            model_name=name,
            model_obj=cnn_models[name]["model"],
            preprocess_fn=cnn_models[name]["preprocess"],
            batch_size=args.batch_size,
            image_size=args.image_size,
        )

    train_script_path = _resolve_train_script_path(getattr(args, "train_script", ""))
    train_module = _load_train_module_for_external(train_script_path)
    radiomics_model = joblib.load(radiomics_model_path)
    selected_estimator = deployment.get("radiomics_selection", {}).get("selected_estimator")
    if selected_estimator and hasattr(train_module, "assert_radiomics_estimator_family"):
        train_module.assert_radiomics_estimator_family(radiomics_model, selected_estimator)
    cfg = _radiomics_feature_config_from_deployment(deployment)
    feats = []
    for p in df["image_path"].astype(str).tolist():
        gray = load_dicom_grayscale(p, args.image_size)
        feat, _ = train_module.extract_adaptive_wavelet_glcm_lbp(
            gray,
            distances=[int(x) for x in cfg["glcm_distances"]],
            angles_deg=[float(x) for x in cfg["glcm_angles_deg"]],
            wavelet_levels=int(cfg["wavelet_levels"]),
            lbp_radius=int(cfg["lbp_radius"]),
            lbp_n_points=int(cfg["lbp_n_points"]),
        )
        feats.append(feat)
    Xrad = np.stack(feats).astype(np.float32)
    expected_dim = int(cfg.get("feature_dimension", Xrad.shape[1]))
    if Xrad.shape[1] != expected_dim:
        raise RuntimeError(f"Kermany radiomics feature dimension mismatch: {Xrad.shape[1]} != {expected_dim}")
    rad_probs = normalize_binary_probability(radiomics_model.predict_proba(Xrad), len(df), "external radiomics")[:, 1]

    Xmeta = np.column_stack([dl_probs[selected_dl_models[0]], dl_probs[selected_dl_models[1]], rad_probs])
    stacked_probs = meta_clf.predict_proba(Xmeta)[:, 1].astype(np.float32)
    soft_weights = deployment.get("soft_voting_weights", {}) or {}
    soft_probs = np.zeros(len(df), dtype=np.float32)
    if soft_weights:
        members = list(selected_dl_models) + ["radiomics"]
        for m in members:
            src = rad_probs if m == "radiomics" else dl_probs[m]
            soft_probs += float(soft_weights.get(m, 0.0)) * src
    return {
        "stacked": stacked_probs,
        "soft": soft_probs,
        "radiomics": rad_probs,
        **dl_probs,
    }


def _select_kermany_four_stage_indices(df, y_true, y_pred, probas, n=4):
    """Prespecified label/path-based case selection; NEVER rank test cases by CAM or probability."""
    import hashlib
    n = max(1, int(n))
    def stable_key(i):
        return hashlib.sha256(("kermany-xai-42:" + Path(df.iloc[i]["image_path"]).name).encode()).hexdigest()
    positives = sorted((i for i in range(len(df)) if int(y_true[i]) == 1), key=stable_key)
    negatives = sorted((i for i in range(len(df)) if int(y_true[i]) == 0), key=stable_key)
    # First half from each label if available, independent of observed model performance.
    npos = min(len(positives), (n + 1) // 2)
    nneg = min(len(negatives), n - npos)
    selected = positives[:npos] + negatives[:nneg]
    for i in positives[npos:] + negatives[nneg:]:
        if len(selected) >= n:
            break
        selected.append(i)
    return selected[:n]


def _load_locked_kermany_predictions(results_dir, external_df, kermany_dir):
    """Reuse the locked MC30 per-image probabilities, aligned by path.

    Recomputing a single deterministic forward pass would NOT reproduce the frozen
    A8 external metrics, so it is intentionally not offered as a silent fallback.
    """
    results = Path(results_dir)
    reports = results / "reports"
    stage19 = reports / "external_validation_kermany_predictions.csv"
    legacy = reports / "external_validation_kermany_predictive_uncertainty.csv"
    test_root = Path(kermany_dir).resolve()
    if not ((test_root / "NORMAL").is_dir() and (test_root / "PNEUMONIA").is_dir()):
        test_root = test_root / "test"
    if stage19.exists():
        recorded = pd.read_csv(stage19)
        col = "stacked_selected_pair_probability"
        if "relative_path" not in recorded or col not in recorded:
            raise RuntimeError(f"Stage-19 predictions have missing required columns: {stage19}")
        recorded = recorded[["relative_path", "label", col]].copy()
        recorded["relative_path"] = recorded["relative_path"].astype(str).map(lambda x: Path(x).as_posix())
        current = external_df.copy()
        current["relative_path"] = current["image_path"].map(
            lambda x: Path(x).resolve().relative_to(test_root).as_posix()
        )
        aligned = current.merge(recorded, how="left", on="relative_path", suffixes=("", "_locked"), validate="one_to_one")
        if len(aligned) != len(current) or aligned[col].isna().any() or (aligned["label"] != aligned["label_locked"]).any():
            raise RuntimeError("Kermany Stage-19 prediction alignment / labels mismatch; refusing to guess probabilities")
        return aligned[col].to_numpy(dtype=np.float32), str(stage19)
    if legacy.exists():
        recorded = pd.read_csv(legacy)
        col = "stacked_selected_pair_mean_probability"
        if "image_path" not in recorded or col not in recorded or "label" not in recorded:
            raise RuntimeError(f"Legacy external prediction report is missing columns: {legacy}")
        recorded = recorded[["image_path", "label", col]].copy()
        recorded["image_path"] = recorded["image_path"].astype(str).map(lambda x: str(Path(x).resolve()))
        current = external_df.copy()
        current["image_path"] = current["image_path"].map(lambda x: str(Path(x).resolve()))
        aligned = current.merge(recorded, how="left", on="image_path", suffixes=("", "_locked"), validate="one_to_one")
        if len(aligned) != len(current) or aligned[col].isna().any() or (aligned["label"] != aligned["label_locked"]).any():
            raise RuntimeError("Legacy Kermany prediction alignment mismatch; refusing to guess probabilities")
        return aligned[col].to_numpy(dtype=np.float32), str(legacy)
    raise FileNotFoundError(
        "Kermany external four-panel visualization needs existing LOCKED per-image external predictions. "
        "First run external_validation_kermany_stage19.py (preferred), or external_validation_kermany.py, "
        "using the official 624-image Kermany test split; no model refitting or probability re-estimation is performed here."
    )


def maybe_generate_kermany_four_stage_artifacts(args, xai_output_dir, xai_reports, deployment,
                                                 radiomics_model_path, selected_dl_models,
                                                 cnn_models, meta_clf, models_pp_layers):
    if not getattr(args, "kermany_dir", ""):
        return None
    print("\n[2a-K/6] External Kermany four-stage Hybrid CNN-ViT XAI...")
    kermany_df = build_kermany_df(args.kermany_dir, getattr(args, "kermany_max_samples", 0))
    y_true = kermany_df["label"].to_numpy(dtype=int)
    probs, probability_source = _load_locked_kermany_predictions(
        args.results_dir, kermany_df, args.kermany_dir
    )
    y_pred = (probs >= float(args.decision_threshold)).astype(np.int32)
    n_cases = int(getattr(args, "kermany_four_stage_n", 4) or 4)
    indices = _select_kermany_four_stage_indices(kermany_df, y_true, y_pred, probs, n=n_cases)
    metrics_df = save_dual_cnn_four_stage_grid(
        test_df=kermany_df, y_true=y_true, y_pred=y_pred, probas=probs,
        indices=indices, models_pp_layers=models_pp_layers, gt_box_map={},
        path=xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused_kermany.png",
        image_size=args.image_size, lung_mask_dir=None, use_lung_mask=False, allow_ellipse=False,
        heatmap_threshold_pct=args.heatmap_threshold_pct,
        localization_cam_method=args.localization_cam_method,
        cam_min_grid=args.cam_min_grid, cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
        paper_dpi=args.paper_dpi, save_pdf=args.save_pdf,
        dataset_name="Kermany external test", show_gt_boxes=False,
    )
    if len(metrics_df):
        import shutil
        shutil.copy2(xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused_kermany.png",
                     xai_output_dir / "hybrid_cnn_vit_four_panel_kermany.png")
        metrics_df["probability_source"] = probability_source
        metrics_df.to_csv(xai_reports / "xai_dual_cnn_four_stage_manifest_kermany.csv", index=False)
        metrics_df.to_csv(xai_reports / "xai_hybrid_cnn_vit_four_stage_manifest_kermany.csv", index=False)
    full_df = kermany_df.copy()
    full_df["stacked_prob_pneumonia"] = probs
    full_df["stacked_pred"] = y_pred
    full_df["probability_source"] = probability_source
    full_df.to_csv(xai_reports / "kermany_external_four_stage_predictions.csv", index=False)
    print(f"[4-STAGE-XAI][KERMANY][OK] n={len(metrics_df)} saved under {xai_output_dir}")
    return metrics_df


def load_detailed_class_map(detailed_class_info_csv):
    """RSNA stage_2_detailed_class_info.csv -> {patientId: class_string}.
    Classes: 'Normal', 'No Lung Opacity / Not Normal', 'Lung Opacity'.
    """
    if not detailed_class_info_csv:
        return None
    p = Path(detailed_class_info_csv)
    if not p.exists():
        print(f"[COMPARE][WARN] detailed_class_info not found: {p}; 3-way figure will be skipped.")
        return None
    try:
        d = pd.read_csv(p)
        col = "class" if "class" in d.columns else d.columns[-1]
        d = d[["patientId", col]].dropna().drop_duplicates("patientId")
        return dict(zip(d["patientId"].astype(str), d[col].astype(str)))
    except Exception as exc:
        print(f"[COMPARE][WARN] could not read detailed_class_info ({exc}); skipping 3-way.")
        return None


def _rank_positive_indices(test_df, y_true, y_pred, probas, gt_box_map, image_size,
                           xai_metrics_df=None):
    """TP pneumonia cases, best-localized first (needs GT boxes)."""
    order = []
    if xai_metrics_df is not None and len(xai_metrics_df) and "sample_index" in xai_metrics_df.columns:
        d = xai_metrics_df.copy()
        d = d[(d["label"] == 1) & (d["pred"] == 1) & (d["gt_box_count"] > 0)]
        for c in ["localization_score", "iou_at_thr", "prob_pneumonia"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.sort_values(by=["localization_score", "iou_at_thr", "prob_pneumonia"],
                          ascending=False, na_position="last")
        order = [int(i) for i in d["sample_index"].tolist()]
    if not order:
        cand = [i for i in range(len(test_df))
                if int(y_true[i]) == 1 and int(y_pred[i]) == 1
                and len(scale_boxes_to_image(str(test_df.iloc[i]["patientId"]),
                                             test_df.iloc[i]["image_path"], gt_box_map, image_size)) > 0]
        cand.sort(key=lambda i: float(probas[i, 1]) if probas is not None else 0.0, reverse=True)
        order = cand
    return order


def _rank_confident_negative(test_df, y_true, y_pred, probas, patient_filter=None):
    """Correctly-classified negatives, most confident first (lowest P(pneu))."""
    cand = []
    for i in range(len(test_df)):
        if int(y_true[i]) != 0 or int(y_pred[i]) != 0:
            continue
        if patient_filter is not None and not patient_filter(str(test_df.iloc[i]["patientId"])):
            continue
        cand.append(i)
    cand.sort(key=lambda i: float(probas[i, 1]) if probas is not None else 1.0)
    return cand



def save_curated_class_comparison(
    model, preprocess_fn, test_df, y_true, y_pred, probas,
    gt_box_map, column_specs, path, image_size=224,
    use_lung_mask=False, lung_mask_dir=None, heatmap_threshold_pct=0.60,
    n_per_class=2, paper_dpi=600, save_pdf=False, suptitle=None,
    allow_ellipse=False, cam_min_grid=16, localization_cam_method="validation_selected",
    models_pp_layers=None,
):
    """Compact class-comparison figure with provenance placed below panels."""
    rows = []
    for title, idxs in column_specs:
        for i in idxs[:max(n_per_class, 1)]:
            rows.append((title, int(i)))
    if not rows:
        print("[COMPARE][WARN] No samples available.")
        return

    fig = plt.figure(figsize=(14.8, max(5.2, 2.85 * len(rows) + 0.9)))
    gs = gridspec.GridSpec(
        len(rows), 4, figure=fig,
        width_ratios=[0.46, 1.0, 1.0, 1.05],
        left=0.025, right=0.995, top=0.925, bottom=0.04,
        wspace=0.08, hspace=0.45,
    )
    col_titles = ["Original + GT", "Raw selected CAM", "Anatomical-policy CAM"]

    for r, (class_title, idx) in enumerate(rows):
        row = test_df.iloc[idx]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(row["image_path"], image_size)
        raw_rgb = np.repeat(raw[..., None], 3, axis=-1)
        raw_u8 = to_uint8(raw_rgb)
        if models_pp_layers:
            hm_r, method, details = compute_class_localization_map(
                models_pp_layers, raw_rgb, image_size, class_index=1,
                force_method=localization_cam_method, min_grid=cam_min_grid,
            )
            layer = "|".join(sorted({str(d.get("layer", "")) for d in details if d.get("status") == "ok"}))
        else:
            preproc = preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32))).numpy()
            layers = _resolution_aware_gradcam_layers(model, min_grid=cam_min_grid, max_layers=5)
            hm, layer, method = compute_gradcam_robust(
                model, preproc, None, class_index=1, fixed_layer=layers[0],
                force_method=localization_cam_method, allow_method_fallback=True,
                allow_layer_fallback=False,
            )
            hm_r = resize_heatmap(hm, image_size, image_size)

        manifest = _CAM_CONFIG_LOCK or {}
        policy = str(manifest.get("selected_map_policy", "hard" if use_lung_mask else "raw"))
        outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
        if use_lung_mask:
            lung_mask, gate_mode = load_lung_mask_for_patient(
                pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
            )
        else:
            lung_mask, gate_mode = None, "off"
        hm_g = _apply_lung_constraint_policy(hm_r, lung_mask, policy, outside_weight)
        gt_boxes = scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
        raw_overlay, _ = overlay_heatmap_rgba(raw_u8, hm_r, alpha=0.55)
        gated_overlay, _ = overlay_heatmap_rgba(raw_u8, hm_g, alpha=0.55)
        p = float(probas[idx, 1]) if probas is not None else np.nan
        cat = _get_category(y_true[idx], y_pred[idx])

        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis("off")
        label_ax.text(
            0.52, 0.5, f"{class_title}\n{pid[:8]}\nP={format_metric(p, 2)}",
            rotation=90, ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=CATEGORY_COLORS.get(cat, "black"),
        )
        row_axes = [fig.add_subplot(gs[r, c]) for c in range(1, 4)]
        for c, (ax, img) in enumerate(zip(row_axes, [raw_u8, raw_overlay, gated_overlay])):
            ax.imshow(img)
            _style_image_axis(
                ax, title=col_titles[c] if r == 0 else None,
                border_color=CATEGORY_COLORS.get(cat, "black"), title_size=10.2,
            )
            draw_boxes(ax, gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)

        footer = (
            f"{_compact_xai_text(method, 24)} | layer={_compact_xai_text(layer, 34)}\n"
            f"policy={policy} | mask={_compact_xai_text(gate_mode, 28)}"
        )
        _add_metric_footer(row_axes[2], footer, fontsize=6.9, y=-0.12, width=55)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.978)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] Class-comparison figure saved: {Path(path)}")

def make_class_comparison_figures(
    model, preprocess_fn, test_df, y_true, y_pred, probas, gt_box_map,
    out_dir, image_size=224, use_lung_mask=False, lung_mask_dir=None,
    heatmap_threshold_pct=0.60, n_per_class=2, paper_dpi=600, save_pdf=False,
    xai_metrics_df=None, detailed_class_info_csv=None, allow_ellipse=False,
    cam_min_grid=16, localization_cam_method="validation_selected",
    models_pp_layers=None,
):
    """Build BOTH the 2-way (Non-Pneumonia vs Pneumonia) and, if detailed class
    info is available, the 3-way (Normal / No Lung Opacity-Not Normal / Pneumonia)
    curated comparison figures for the paper."""
    out_dir = ensure_dir(out_dir)
    pos_idx = _rank_positive_indices(test_df, y_true, y_pred, probas, gt_box_map,
                                     image_size, xai_metrics_df=xai_metrics_df)

    # --- 2-way ---
    neg_idx = _rank_confident_negative(test_df, y_true, y_pred, probas)
    save_curated_class_comparison(
        model, preprocess_fn, test_df, y_true, y_pred, probas, gt_box_map,
        column_specs=[("Non-Pneumonia", neg_idx), ("Pneumonia", pos_idx)],
        path=out_dir / "n4_compare_2way_nonpneumonia_vs_pneumonia.png",
        image_size=image_size, use_lung_mask=use_lung_mask, lung_mask_dir=lung_mask_dir,
        heatmap_threshold_pct=heatmap_threshold_pct, n_per_class=n_per_class,
        paper_dpi=paper_dpi, save_pdf=save_pdf,
        suptitle="Validation-Selected Pneumonia CAM: Non-Pneumonia vs Pneumonia",
        allow_ellipse=allow_ellipse, cam_min_grid=cam_min_grid,
        localization_cam_method=localization_cam_method,
        models_pp_layers=models_pp_layers,
    )

    # --- 3-way (needs detailed class info) ---
    dmap = load_detailed_class_map(detailed_class_info_csv)
    if dmap is not None:
        def is_normal(pid):
            return dmap.get(str(pid), "").strip().lower() == "normal"

        def is_nolung(pid):
            return "no lung opacity" in dmap.get(str(pid), "").strip().lower()

        normal_idx = _rank_confident_negative(test_df, y_true, y_pred, probas, patient_filter=is_normal)
        nolung_idx = _rank_confident_negative(test_df, y_true, y_pred, probas, patient_filter=is_nolung)
        save_curated_class_comparison(
            model, preprocess_fn, test_df, y_true, y_pred, probas, gt_box_map,
            column_specs=[("Normal", normal_idx),
                          ("No Lung Opacity /\nNot Normal", nolung_idx),
                          ("Pneumonia", pos_idx)],
            path=out_dir / "n4_compare_3way_normal_nonpneu_pneumonia.png",
            image_size=image_size, use_lung_mask=use_lung_mask, lung_mask_dir=lung_mask_dir,
            heatmap_threshold_pct=heatmap_threshold_pct, n_per_class=n_per_class,
            paper_dpi=paper_dpi, save_pdf=save_pdf,
            suptitle="Post-hoc negative subcategories vs Pneumonia (binary classifier)",
            allow_ellipse=allow_ellipse, cam_min_grid=cam_min_grid,
            localization_cam_method=localization_cam_method,
            models_pp_layers=models_pp_layers,
        )


# -----------------------------------------------------------------------------
# Quantitative XAI evaluation over a dataset/subset
# -----------------------------------------------------------------------------

def select_xai_indices(df, y_true, y_pred, max_samples=0):
    n = len(df)
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= n:
        return list(range(n))
    max_samples = int(max_samples)
    categories = np.array([_get_category(t, p) for t, p in zip(y_true, y_pred)])
    selected = []
    # Prioritize error and positive classes for localization analysis, then fill.
    order = ["FN", "FP", "TP", "TN"]
    per_group = max(1, max_samples // len(order))
    for cat in order:
        idxs = np.where(categories == cat)[0].tolist()
        selected.extend(idxs[:per_group])
    selected = list(dict.fromkeys(selected))
    if len(selected) < max_samples:
        for idx in range(n):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]



def init_agg_bucket(image_size):
    return {
        "n": 0,
        "raw_sum": np.zeros((image_size, image_size), dtype=np.float64),
        "raw_cam_sum": np.zeros((image_size, image_size), dtype=np.float64),
        "loc_sum": np.zeros((image_size, image_size), dtype=np.float64),
        "gt_mask_sum": np.zeros((image_size, image_size), dtype=np.float64),
        "lung_mask_sum": np.zeros((image_size, image_size), dtype=np.float64),
    }


def add_to_agg(bucket, raw_img, loc_map, gt_boxes, lung_mask=None, raw_cam=None):
    audit = compute_heatmap_audit(loc_map)
    if not audit["heatmap_valid"]:
        return
    bucket["n"] += 1
    bucket["raw_sum"] += np.asarray(raw_img, dtype=np.float64)
    if raw_cam is not None:
        bucket["raw_cam_sum"] += np.asarray(raw_cam, dtype=np.float64)
    bucket["loc_sum"] += np.asarray(loc_map, dtype=np.float64)
    bucket["gt_mask_sum"] += boxes_to_mask(
        gt_boxes, raw_img.shape[0], raw_img.shape[1]
    ).astype(np.float64)
    if lung_mask is not None:
        bucket["lung_mask_sum"] += np.asarray(lung_mask, dtype=np.float64)

def save_xai_agg_state(agg, path):
    """Persist aggregate heatmap accumulators for interruption-safe XAI resume."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(agg, tmp)
    tmp.replace(path)



def load_xai_agg_state(path, image_size):
    path = Path(path)
    if path.exists():
        try:
            agg = joblib.load(path)
            if isinstance(agg, dict) and all(
                isinstance(v, dict) and "loc_sum" in v for v in agg.values()
            ):
                return agg
        except Exception as exc:
            print(f"[WARN] Could not load XAI aggregate resume state {path}: {exc}")
    return {key: init_agg_bucket(image_size) for key in ["GT_1", "TP", "FN"]}


def evaluate_xai_localization_dataset(
    model,
    preprocess_fn,
    test_df,
    y_true,
    y_pred,
    probas,
    candidate_layers,
    gt_box_map,
    image_size,
    max_samples=0,
    heatmap_threshold_pct=0.60,
    reports_dir=None,
    use_lung_mask=False,
    lung_mask_dir=None,
    resume=True,
    force=False,
    save_every=10,
    models_pp_layers=None,
    lung_gate=True,
    allow_ellipse=False,
    localization_cam_method="GradCAM",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
    n_boot=2000,
    seed=42,
    max_invalid_rate=0.02,
    max_gated_outside_lung_ratio=0.001,
    strict_qc=True,
    run_mode="FINAL",
    expected_gt_positive=None,
    min_final_coverage=0.95,
    min_final_valid_cases=1000,
    min_effective_lung_mask_coverage=0.99,
    max_lung_mask_fallback_rate=0.05,
    cam_config_manifest=None,
):
    indices_all = select_xai_indices(test_df, y_true, y_pred, max_samples=max_samples)
    indices = []
    for i in indices_all:
        row_i = test_df.iloc[i]
        boxes = scale_boxes_to_image(
            str(row_i["patientId"]), row_i["image_path"], gt_box_map, image_size
        )
        if int(y_true[i]) == 1 and len(boxes) > 0:
            indices.append(i)
    print(f"[INFO] Quantitative XAI evaluation GT-positive samples: {len(indices)} / {len(test_df)}")

    reports_dir = ensure_dir(reports_dir) if reports_dir is not None else None
    metrics_path = reports_dir / "xai_localization_per_sample.csv" if reports_dir is not None else None
    summary_path = reports_dir / "xai_localization_summary.csv" if reports_dir is not None else None
    qc_path = reports_dir / "xai_quality_control.json" if reports_dir is not None else None
    agg_state_path = reports_dir / "xai_localization_agg_state.joblib" if reports_dir is not None else None

    records, processed_indices = [], set()
    can_resume = bool(resume) and (not force) and metrics_path is not None and metrics_path.exists() and agg_state_path is not None and agg_state_path.exists()
    if can_resume:
        try:
            prev_df = pd.read_csv(metrics_path)
            required = {
                "xai_schema_version", "localization_map_role", "heatmap_valid",
                "outside_lung_ratio_raw", "outside_lung_ratio_gated", "deployment_lock_id",
            }
            if not required.issubset(prev_df.columns):
                raise ValueError("stale_xai_cache_columns")
            if set(prev_df["xai_schema_version"].astype(str).unique()) != {XAI_SCHEMA_VERSION}:
                raise ValueError("stale_xai_schema_version")
            if set(prev_df["deployment_lock_id"].astype(str).unique()) != {str(_XAI_DEPLOYMENT_LOCK_ID)}:
                raise ValueError("stale_deployment_lock_id")
            records = prev_df.to_dict(orient="records")
            processed_indices = set(pd.to_numeric(prev_df["sample_index"], errors="coerce").dropna().astype(int))
            agg = load_xai_agg_state(agg_state_path, image_size)
            print(f"[RESUME-XAI] Loaded {len(records)} current-schema rows; skipping completed samples.")
        except Exception as exc:
            print(f"[RESUME-XAI] Existing cache is incompatible ({exc}); recomputing Q1 XAI metrics.")
            records, processed_indices = [], set()
            agg = {key: init_agg_bucket(image_size) for key in ["GT_1", "TP", "FN"]}
    else:
        agg = {key: init_agg_bucket(image_size) for key in ["GT_1", "TP", "FN"]}

    n_new = 0
    for idx in tqdm(indices, desc="  XAI-localization"):
        if int(idx) in processed_indices:
            continue
        row = test_df.iloc[idx]
        path = row["image_path"]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(path, image_size)
        raw_rgb = np.repeat(raw[..., None], 3, axis=-1)

        # Primary localization evidence: Pneumonia-class CAM only.
        if models_pp_layers:
            loc_raw, loc_method, cam_details = compute_class_localization_map(
                models_pp_layers,
                raw_rgb,
                image_size,
                class_index=1,
                smooth_sigma_frac=cam_smooth_sigma_frac,
                force_method=localization_cam_method,
                min_grid=cam_min_grid,
            )
        else:
            preproc = preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32))).numpy()
            layers = _resolution_aware_gradcam_layers(model, min_grid=cam_min_grid, max_layers=5)
            hm_cls, layer_name, method = compute_gradcam_robust(
                model, preproc, candidate_layers, class_index=1,
                fixed_layer=layers[0], force_method=localization_cam_method,
                allow_method_fallback=True, allow_layer_fallback=False,
            )
            loc_raw = resize_heatmap(hm_cls, image_size, image_size)
            loc_method = method
            cam_details = [{"model": getattr(model, "name", "cnn"), "layer": layer_name, "method": method}]

        manifest = cam_config_manifest or _CAM_CONFIG_LOCK or {}
        selected_policy = str(manifest.get("selected_map_policy", "hard" if lung_gate else "raw"))
        selected_threshold = float(manifest.get("selected_threshold", heatmap_threshold_pct))
        outside_weight = float(manifest.get("soft_lung_outside_weight", 0.20))
        lung_mask = None; lung_mode = "off"
        if use_lung_mask:
            lung_mask, lung_mode = load_lung_mask_for_patient(
                pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
            )
        loc_map = _apply_lung_constraint_policy(
            loc_raw, lung_mask, policy=selected_policy, outside_weight=outside_weight
        )
        hard_gated_map = _apply_lung_constraint_policy(
            loc_raw, lung_mask, policy="hard", outside_weight=outside_weight
        ) if lung_mask is not None else loc_raw
        raw_audit = compute_heatmap_audit(loc_raw, lung_mask=lung_mask)
        selected_audit = compute_heatmap_audit(loc_map, lung_mask=lung_mask)
        gated_audit = compute_heatmap_audit(hard_gated_map, lung_mask=lung_mask)
        gt_boxes = scale_boxes_to_image(pid, path, gt_box_map, image_size)
        metrics = compute_localization_metrics(
            loc_map, gt_boxes, threshold_pct=selected_threshold,
            heatmap_valid=selected_audit["heatmap_valid"],
        )
        true_i, pred_i = int(y_true[idx]), int(y_pred[idx])
        cat = _get_category(true_i, pred_i)
        ok_details = [d for d in cam_details if d.get("status", "ok") == "ok"]
        rec = {
            "xai_schema_version": XAI_SCHEMA_VERSION,
            "deployment_lock_id": str(_XAI_DEPLOYMENT_LOCK_ID),
            "model_checkpoint_sha256": json.dumps(_XAI_MODEL_HASHES, sort_keys=True),
            "localization_map_role": "primary_pneumonia_class_cam",
            "sample_index": int(idx),
            "sample_id": row.get("sample_id", np.nan),
            "patientId": pid,
            "label": true_i,
            "pred": pred_i,
            "category": cat,
            "prob_pneumonia": float(probas[idx, 1]) if probas is not None else np.nan,
            "localization_map": loc_method,
            "cam_models_valid": int(len(ok_details)),
            "cam_models_attempted": int(len(models_pp_layers or [model])),
            "cam_layers": "|".join(str(d.get("layer", "")) for d in ok_details),
            "cam_methods": "|".join(str(d.get("method", "")) for d in ok_details),
            "lung_gate_mode": lung_mode,
            "lung_mask_ok": bool(lung_mask is not None),
            "heatmap_threshold_pct": float(selected_threshold),
            "outside_lung_ratio_raw": raw_audit["outside_lung_ratio"],
            "inside_lung_energy_raw": raw_audit["inside_lung_energy"],
            "outside_lung_ratio_selected": selected_audit["outside_lung_ratio"],
            "outside_lung_ratio_gated": gated_audit["outside_lung_ratio"],
            "inside_lung_energy_gated": gated_audit["inside_lung_energy"],
            "cam_map_policy": selected_policy,
            "cam_fusion_rule": "validation_selected_model_and_layer_weights",
            "cam_layer_selection_split": "validation",
            "cam_test_set_used_for_layer_selection": False,
            "heatmap_entropy": selected_audit["heatmap_entropy"],
            "heatmap_nonzero_ratio": selected_audit["heatmap_nonzero_ratio"],
            "lung_mask_area_ratio": gated_audit["lung_mask_area_ratio"],
            **metrics,
        }
        rec["failure_reason"] = explain_failure(rec)
        records.append(rec)

        if selected_audit["heatmap_valid"]:
            add_to_agg(agg["GT_1"], raw, loc_map, gt_boxes, lung_mask, raw_cam=loc_raw)
            if cat in agg:
                add_to_agg(agg[cat], raw, loc_map, gt_boxes, lung_mask, raw_cam=loc_raw)

        n_new += 1
        if reports_dir is not None and n_new % max(int(save_every), 1) == 0:
            partial_df = pd.DataFrame(records)
            partial_df.to_csv(metrics_path, index=False)
            summarize_localization_metrics(partial_df, n_boot=0, seed=seed).to_csv(summary_path, index=False)
            save_xai_agg_state(agg, agg_state_path)
            print(f"[RESUME-XAI] checkpoint saved: {len(partial_df)} rows")

    metrics_df = pd.DataFrame(records)
    summary_df = summarize_localization_metrics(metrics_df, n_boot=n_boot, seed=seed)
    invalid_rate = float((~metrics_df["heatmap_valid"].astype(bool)).mean()) if len(metrics_df) else 1.0
    lung_success = float(metrics_df["lung_mask_ok"].astype(bool).mean()) if len(metrics_df) else 0.0
    outside_vals = pd.to_numeric(metrics_df.get("outside_lung_ratio_raw"), errors="coerce").dropna()
    gated_outside_vals = pd.to_numeric(metrics_df.get("outside_lung_ratio_gated"), errors="coerce").dropna()
    gated_outside_max = float(gated_outside_vals.max()) if len(gated_outside_vals) else np.nan
    expected_n = int(expected_gt_positive) if expected_gt_positive is not None else int(len(indices))
    expected_n = max(expected_n, 1)
    n_evaluated = int(len(metrics_df))
    n_valid = int(metrics_df["heatmap_valid"].astype(bool).sum()) if len(metrics_df) else 0
    evaluation_coverage = n_evaluated / expected_n
    valid_heatmap_coverage = n_valid / expected_n
    mode_upper = str(run_mode).upper()
    sample_coverage_pass = bool(
        mode_upper != "FINAL"
        or (
            evaluation_coverage >= float(min_final_coverage)
            and valid_heatmap_coverage >= float(min_final_coverage)
            and n_valid >= int(min_final_valid_cases)
        )
    )
    lung_modes = metrics_df.get("lung_gate_mode", pd.Series(dtype=str)).fillna("missing").astype(str) if len(metrics_df) else pd.Series(dtype=str)
    lung_mode_counts = lung_modes.value_counts(dropna=False).to_dict() if len(lung_modes) else {}
    lung_missing_count = int((~metrics_df["lung_mask_ok"].astype(bool)).sum()) if len(metrics_df) else 0
    pretrained_mask_count = int(lung_modes.str.startswith("pretrained").sum()) if len(lung_modes) else 0
    ellipse_fallback_count = int(lung_modes.str.contains("ellipse", case=False, regex=False).sum()) if len(lung_modes) else 0
    effective_lung_mask_coverage = float(lung_success)
    pretrained_lung_mask_rate = float(pretrained_mask_count / n_evaluated) if n_evaluated else 0.0
    lung_mask_fallback_rate = float(ellipse_fallback_count / n_evaluated) if n_evaluated else 0.0
    selected_policy_for_qc = str((_CAM_CONFIG_LOCK or {}).get("selected_map_policy", "raw")).lower()
    primary_map_requires_mask = bool(use_lung_mask and selected_policy_for_qc in {"soft", "hard"})
    lung_mask_qc_pass = bool(
        (not primary_map_requires_mask)
        or (
            effective_lung_mask_coverage >= float(min_effective_lung_mask_coverage)
            and lung_mask_fallback_rate <= float(max_lung_mask_fallback_rate)
        )
    )

    # Sensitivity summary by mask provenance. This prevents a small fallback
    # subset from being silently mixed into the primary result.
    if reports_dir is not None and len(metrics_df):
        sensitivity_frames = []
        group_specs = [
            ("all_effective_masks", metrics_df[metrics_df["lung_mask_ok"].astype(bool)]),
            ("pretrained_only", metrics_df[lung_modes.str.startswith("pretrained")]),
            ("ellipse_fallback_only", metrics_df[lung_modes.str.contains("ellipse", case=False, regex=False)]),
            ("missing_or_invalid_mask", metrics_df[~metrics_df["lung_mask_ok"].astype(bool)]),
        ]
        for subset_name, subset_df in group_specs:
            if len(subset_df) == 0:
                continue
            subset_summary = summarize_localization_metrics(
                subset_df, n_boot=min(int(n_boot), 1000), seed=seed
            )
            subset_summary.insert(0, "lung_mask_subset", subset_name)
            subset_summary.insert(1, "n_subset", int(len(subset_df)))
            sensitivity_frames.append(subset_summary)
        if sensitivity_frames:
            pd.concat(sensitivity_frames, ignore_index=True).to_csv(
                reports_dir / "xai_lung_mask_sensitivity_summary.csv", index=False
            )
        (reports_dir / "xai_lung_mask_provenance.json").write_text(
            json.dumps({
                "n_evaluated": n_evaluated,
                "pretrained_mask_count": pretrained_mask_count,
                "ellipse_fallback_count": ellipse_fallback_count,
                "missing_or_invalid_count": lung_missing_count,
                "effective_lung_mask_coverage": effective_lung_mask_coverage,
                "pretrained_lung_mask_rate": pretrained_lung_mask_rate,
                "lung_mask_fallback_rate": lung_mask_fallback_rate,
                "max_lung_mask_fallback_rate_allowed": float(max_lung_mask_fallback_rate),
                "minimum_effective_lung_mask_coverage": float(min_effective_lung_mask_coverage),
                "mode_counts": lung_mode_counts,
                "primary_scientific_cohort": "pretrained_and_adaptive_pretrained_masks",
                "ellipse_fallback_role": "sensitivity_analysis_only",
                "primary_map_requires_lung_mask": primary_map_requires_mask,
            }, indent=2), encoding="utf-8"
        )

    quick_diagnostic_pass = bool(
        invalid_rate <= float(max_invalid_rate)
        and n_valid == n_evaluated
        and (
            not use_lung_mask
            or not np.isfinite(gated_outside_max)
            or gated_outside_max <= float(max_gated_outside_lung_ratio)
        )
    )
    final_technical_candidate_pass = bool(
        invalid_rate <= float(max_invalid_rate)
        and lung_mask_qc_pass
        and sample_coverage_pass
        and (not use_lung_mask or (np.isfinite(gated_outside_max) and gated_outside_max <= float(max_gated_outside_lung_ratio)))
    )
    qc = {
        "xai_schema_version": XAI_SCHEMA_VERSION,
        "control_flow_revision": "STRONG_BASELINE_CV_V8",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_mode": mode_upper,
        "artifact_status": "QUICK_PRECHECK_NOT_FOR_PAPER" if mode_upper == "QUICK" else "FINAL_Q1_CANDIDATE",
        "expected_gt_positive_with_boxes": expected_n,
        "n_evaluated": n_evaluated,
        "n_valid_heatmaps": n_valid,
        "evaluation_coverage": float(evaluation_coverage),
        "valid_heatmap_coverage": float(valid_heatmap_coverage),
        "minimum_final_coverage": float(min_final_coverage),
        "minimum_final_valid_cases": int(min_final_valid_cases),
        "sample_coverage_pass": sample_coverage_pass,
        "invalid_heatmap_rate": invalid_rate,
        "max_invalid_rate_allowed": float(max_invalid_rate),
        "lung_mask_success_rate": lung_success,
        "effective_lung_mask_coverage": effective_lung_mask_coverage,
        "pretrained_lung_mask_rate": pretrained_lung_mask_rate,
        "pretrained_lung_mask_count": pretrained_mask_count,
        "lung_mask_fallback_rate": lung_mask_fallback_rate,
        "ellipse_fallback_count": ellipse_fallback_count,
        "max_lung_mask_fallback_rate_allowed": float(max_lung_mask_fallback_rate),
        "minimum_effective_lung_mask_coverage": float(min_effective_lung_mask_coverage),
        "lung_mask_qc_pass": lung_mask_qc_pass,
        "primary_map_requires_lung_mask": primary_map_requires_mask,
        "lung_mask_fallback_is_technical_blocker": primary_map_requires_mask,
        "lung_mask_missing_count": lung_missing_count,
        "lung_mask_mode_counts": lung_mode_counts,
        "quick_diagnostic_qc_pass": quick_diagnostic_pass,
        "final_technical_qc_candidate_pass": final_technical_candidate_pass,
        "quick_qc_is_nonfatal": bool(mode_upper == "QUICK"),
        "outside_lung_ratio_raw_mean": float(outside_vals.mean()) if len(outside_vals) else np.nan,
        "outside_lung_ratio_raw_median": float(outside_vals.median()) if len(outside_vals) else np.nan,
        "outside_lung_ratio_raw_q1": float(outside_vals.quantile(0.25)) if len(outside_vals) else np.nan,
        "outside_lung_ratio_raw_q3": float(outside_vals.quantile(0.75)) if len(outside_vals) else np.nan,
        "outside_lung_ratio_gated_mean": float(gated_outside_vals.mean()) if len(gated_outside_vals) else np.nan,
        "outside_lung_ratio_gated_max": gated_outside_max,
        "max_gated_outside_lung_ratio_allowed": float(max_gated_outside_lung_ratio),
        "primary_map": "Validation-selected weighted multi-layer CNN Pneumonia-class CAM",
        "selected_map_policy": selected_policy_for_qc,
        "selected_threshold": float((_CAM_CONFIG_LOCK or {}).get("selected_threshold", heatmap_threshold_pct)),
        "technical_qc_pass": final_technical_candidate_pass,
        "xai_analysis_complete": True,
        "contrastive_map_role": "supplementary_qualitative_only",
        "qc_pass": final_technical_candidate_pass,
    }
    if reports_dir is not None:
        metrics_df.to_csv(metrics_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        save_xai_agg_state(agg, agg_state_path)
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
        print(f"[OK] XAI localization metrics saved: {metrics_path}")
        print(f"[OK] XAI localization summary saved: {summary_path}")
        print(f"[OK] XAI quality control saved: {qc_path}")
    if not qc["qc_pass"]:
        if mode_upper == "QUICK":
            print(
                "[WARN] QUICK precheck QC did not satisfy FINAL thresholds. "
                "This is diagnostic-only and will not stop the workflow."
            )
        elif strict_qc:
            raise RuntimeError(
                "FINAL XAI technical quality control failed: " + json.dumps(qc, default=str)
            )
    return metrics_df, summary_df, agg


def _unpaired_bootstrap_diff_ci(a, b, n_boot=2000, seed=42):
    a=np.asarray(pd.to_numeric(pd.Series(a),errors="coerce").dropna(),dtype=float)
    b=np.asarray(pd.to_numeric(pd.Series(b),errors="coerce").dropna(),dtype=float)
    if len(a)==0 or len(b)==0:
        return np.nan,np.nan,np.nan
    point=float(a.mean()-b.mean())
    if int(n_boot)<=0:
        return point,point,point
    rng=np.random.default_rng(int(seed)); diffs=np.empty(int(n_boot),dtype=float)
    for i in range(int(n_boot)):
        diffs[i]=rng.choice(a,size=len(a),replace=True).mean()-rng.choice(b,size=len(b),replace=True).mean()
    return point,float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))


def _cliffs_delta(a, b):
    a = np.asarray(pd.to_numeric(pd.Series(a), errors="coerce").dropna(), dtype=float)
    b = np.asarray(pd.to_numeric(pd.Series(b), errors="coerce").dropna(), dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Rank-based O((n+m)log(n+m)) implementation.
    b_sorted = np.sort(b)
    greater = sum(np.searchsorted(b_sorted, x, side="left") for x in a)
    less = sum(len(b_sorted) - np.searchsorted(b_sorted, x, side="right") for x in a)
    return float((greater - less) / (len(a) * len(b)))


def save_tp_fn_localization_statistics(metrics_df, reports_dir, n_boot=2000, seed=42):
    if metrics_df is None or len(metrics_df) == 0 or "category" not in metrics_df.columns:
        return pd.DataFrame()
    rows = []
    metrics = ["pointing_hit", "iou_at_thr", "localization_score", "energy_inside_gt", "peak_distance_norm"]
    tp = metrics_df[(metrics_df["category"] == "TP") & metrics_df["heatmap_valid"].astype(bool)]
    fn = metrics_df[(metrics_df["category"] == "FN") & metrics_df["heatmap_valid"].astype(bool)]
    for i, metric in enumerate(metrics):
        a = pd.to_numeric(tp.get(metric), errors="coerce").dropna()
        b = pd.to_numeric(fn.get(metric), errors="coerce").dropna()
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided") if len(a) and len(b) else (np.nan, np.nan)
        except Exception:
            u, p = np.nan, np.nan
        point, lo, hi = _unpaired_bootstrap_diff_ci(
            a, b, n_boot=n_boot, seed=int(seed) + i
        )
        rows.append({
            "metric": metric, "n_tp": int(len(a)), "n_fn": int(len(b)),
            "tp_mean": float(a.mean()) if len(a) else np.nan,
            "fn_mean": float(b.mean()) if len(b) else np.nan,
            "mean_difference_tp_minus_fn": point,
            "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
            "mann_whitney_u": float(u) if np.isfinite(u) else np.nan,
            "mann_whitney_p": float(p) if np.isfinite(p) else np.nan,
            "cliffs_delta_tp_vs_fn": _cliffs_delta(a, b),
        })
    out = pd.DataFrame(rows)
    reports_dir = ensure_dir(reports_dir)
    out.to_csv(reports_dir / "xai_tp_vs_fn_statistics.csv", index=False)
    (reports_dir / "xai_tp_vs_fn_statistics.json").write_text(
        json.dumps(out.to_dict(orient="records"), indent=2, default=_json_safe), encoding="utf-8"
    )
    return out



def _mean_cnn_probability(models_pp_layers, raw_rgb):
    """Probability target for CAM faithfulness.

    V26 hotfix: when the final CAM lock provides inter-model weights, the
    perturbation target uses the SAME selected hybrid CNN-ViT weights as the fused
    CAM (e.g. XRV=0.25, EVA-X=0.75) instead of silently averaging both CNNs
    0.5/0.5.  This keeps deletion/retention aligned with the final V18 CAM
    configuration.  The unweighted mean is retained only as a compatibility
    fallback when no lock is available.
    """
    probs=[]; keys=[]
    for model,preprocess_fn,_cand in models_pp_layers:
        if getattr(model,"_is_torch_adapter",False):
            pred=model.predict_numpy(raw_rgb[np.newaxis],mc_passes=1)
        else:
            x=preprocess_fn(tf.constant(raw_rgb[np.newaxis].astype(np.float32)))
            pred=np.asarray(model(x,training=False),dtype=np.float32)
        if pred.ndim==2 and pred.shape[1]>1:
            probs.append(float(pred[0,1])); keys.append(_model_cam_key(model))
        elif pred.size:
            probs.append(float(pred.ravel()[0])); keys.append(_model_cam_key(model))
    if not probs:
        return np.nan
    locked_weights = ((_CAM_CONFIG_LOCK or {}).get("selected_model_weights") or {})
    if locked_weights and all(k in locked_weights for k in keys):
        w=np.asarray([float(locked_weights[k]) for k in keys],dtype=np.float64)
        if np.all(np.isfinite(w)) and np.all(w > 0) and float(w.sum()) > 0:
            w=w/float(w.sum())
            return float(np.dot(np.asarray(probs,dtype=np.float64),w))
    return float(np.mean(probs))



def evaluate_xai_faithfulness(metrics_df, test_df, models_pp_layers, image_size,
                              reports_dir, max_samples=256, top_fraction=0.10,
                              cam_smooth_sigma_frac=0.005, seed=42,
                              use_lung_mask=False, lung_mask_dir=None, allow_ellipse=False):
    """Deletion/retention faithfulness audit for the CNN explanation target."""
    if metrics_df is None or len(metrics_df) == 0 or int(max_samples) == 0:
        return pd.DataFrame()
    valid = metrics_df[metrics_df["heatmap_valid"].astype(bool)].copy()
    if max_samples > 0 and len(valid) > int(max_samples):
        valid = valid.sample(n=int(max_samples), random_state=int(seed))
    rows = []
    policy = (_CAM_CONFIG_LOCK or {}).get("selected_map_policy", "raw")
    outside_weight = float((_CAM_CONFIG_LOCK or {}).get("soft_lung_outside_weight", 0.20))
    for _, rec in tqdm(valid.iterrows(), total=len(valid), desc="  XAI-faithfulness"):
        idx = int(rec["sample_index"]); row = test_df.iloc[idx]
        raw = load_dicom_grayscale(row["image_path"], image_size)
        raw_rgb = np.repeat(raw[..., None], 3, axis=-1)
        cam_raw, _tag, _details = compute_class_localization_map(
            models_pp_layers, raw_rgb, image_size,
            smooth_sigma_frac=cam_smooth_sigma_frac,
            force_method="validation_selected",
        )
        pid=str(row["patientId"])
        mask,_mask_mode=load_lung_mask_for_patient(
            pid,lung_mask_dir,image_size,allow_ellipse=allow_ellipse
        ) if use_lung_mask else (None,"off")
        cam = _apply_lung_constraint_policy(cam_raw, mask, policy=policy, outside_weight=outside_weight)
        cutoff = float(np.quantile(cam, max(0.0, 1.0 - float(top_fraction))))
        focus = cam >= cutoff
        fill = float(np.mean(raw))
        deleted = raw.copy(); deleted[focus] = fill
        retained = np.full_like(raw, fill); retained[focus] = raw[focus]
        p0 = _mean_cnn_probability(models_pp_layers, raw_rgb)
        pdel = _mean_cnn_probability(models_pp_layers, np.repeat(deleted[..., None], 3, axis=-1))
        pkeep = _mean_cnn_probability(models_pp_layers, np.repeat(retained[..., None], 3, axis=-1))
        rows.append({
            "sample_index": idx, "patientId": str(row["patientId"]),
            "prob_original": p0, "prob_deleted_top_cam": pdel,
            "prob_retained_top_cam": pkeep,
            "deletion_confidence_drop": p0 - pdel if np.isfinite(p0) and np.isfinite(pdel) else np.nan,
            "retention_probability_ratio": pkeep / max(p0, 1e-8) if np.isfinite(p0) and np.isfinite(pkeep) else np.nan,
            "top_fraction": float(top_fraction),
        })
    out = pd.DataFrame(rows)
    reports_dir = ensure_dir(reports_dir)
    out.to_csv(reports_dir / "xai_faithfulness_per_sample.csv", index=False)
    summary = {}
    for col in ["deletion_confidence_drop", "retention_probability_ratio"]:
        vals = pd.to_numeric(out.get(col), errors="coerce").dropna()
        lo, hi = _bootstrap_stat_ci(vals, stat="mean", n_boot=2000, seed=seed)
        summary[col] = {"n": int(len(vals)), "mean": float(vals.mean()) if len(vals) else np.nan, "ci95_low": lo, "ci95_high": hi}
    (reports_dir / "xai_faithfulness_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
    )
    return out


def run_classifier_head_randomization_sanity(models_pp_layers, val_df, y_val,
                                             image_size, reports_dir,
                                             n_samples=0, seed=42):
    """Optional classifier-head randomization check; failures are reported, not hidden."""
    if int(n_samples) <= 0:
        return pd.DataFrame()
    positive = [i for i in range(len(val_df)) if int(y_val[i]) == 1]
    rng = np.random.default_rng(int(seed)); positive = list(rng.permutation(positive))[:int(n_samples)]
    rows = []
    for model, preprocess_fn, cand in models_pp_layers:
        name = _model_cam_key(model)
        cfg = (_CAM_CONFIG_LOCK or {}).get("selected_models", {}).get(name)
        if not cfg:
            continue
        try:
            randomized = tf.keras.models.clone_model(model)
            randomized.set_weights(model.get_weights())
            setattr(randomized, "_aura_cam_name", name)
            dense = next((l for l in reversed(randomized.layers) if isinstance(l, tf.keras.layers.Dense)), None)
            if dense is None:
                raise RuntimeError("final_dense_not_found")
            weights = dense.get_weights()
            new_weights = []
            for w in weights:
                scale = float(np.std(w)) if np.std(w) > 0 else 0.05
                new_weights.append(rng.normal(0.0, scale, size=w.shape).astype(w.dtype))
            dense.set_weights(new_weights)
            rand_cand = _get_candidate_conv_layers(randomized)
            for idx in positive:
                raw = load_dicom_grayscale(val_df.iloc[idx]["image_path"], image_size)
                rgb = np.repeat(raw[..., None], 3, axis=-1)
                trained_map, _ = _compute_model_cam_from_config(model, preprocess_fn, cand, rgb, image_size, cfg, allow_fallback=False)
                randomized_map, _ = _compute_model_cam_from_config(randomized, preprocess_fn, rand_cand, rgb, image_size, cfg, allow_fallback=False)
                corr = np.nan
                if trained_map is not None and randomized_map is not None:
                    corr = spearmanr(trained_map.ravel(), randomized_map.ravel(), nan_policy="omit").statistic
                rows.append({"model": name, "sample_index": int(idx), "spearman_trained_vs_randomized_head": corr})
            del randomized; gc.collect()
        except Exception as exc:
            rows.append({"model": name, "sample_index": np.nan, "error": str(exc), "spearman_trained_vs_randomized_head": np.nan})
    out = pd.DataFrame(rows)
    reports_dir = ensure_dir(reports_dir)
    out.to_csv(reports_dir / "xai_model_randomization_sanity.csv", index=False)
    valid = pd.to_numeric(out.get("spearman_trained_vs_randomized_head"), errors="coerce").dropna()
    summary = {
        "n_valid": int(len(valid)),
        "median_spearman": float(valid.median()) if len(valid) else np.nan,
        "sanity_pass": bool(len(valid) and float(valid.median()) < 0.80),
        "criterion": "median Spearman trained-vs-randomized-head < 0.80",
    }
    (reports_dir / "xai_model_randomization_sanity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Aggregate heatmaps and failure analysis figures
# -----------------------------------------------------------------------------

def average_bucket_image(bucket, key):
    n = int(bucket.get("n", 0))
    if n <= 0:
        return None
    return (bucket[key] / float(n)).astype(np.float32)



def save_xai_aggregate_heatmaps(agg, path, paper_dpi=600, save_pdf=False, run_label=""):
    entries = [(k, t) for k, t in [
        ("GT_1", "All GT-positive cases"),
        ("TP", "True Positive"),
        ("FN", "False Negative"),
    ] if k in agg and int(agg[k].get("n", 0)) > 0]
    if not entries:
        print("[WARN] No valid primary localization maps for aggregate heatmaps.")
        return
    n_cols = len(entries)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.9))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title) in zip(axes, entries):
        bucket = agg[key]
        raw_avg = average_bucket_image(bucket, "raw_sum")
        hm_avg = normalize_minmax(average_bucket_image(bucket, "loc_sum"))
        gt_avg = normalize_minmax(average_bucket_image(bucket, "gt_mask_sum"))
        lung_avg = normalize_minmax(average_bucket_image(bucket, "lung_mask_sum"))
        overlay, _ = overlay_heatmap_rgba(to_uint8(raw_avg), hm_avg, alpha=0.55)
        ax.imshow(overlay)
        if gt_avg.max() > 0:
            ax.contour(gt_avg, levels=[0.25], colors=[GT_COLOR], linewidths=2.0)
        if lung_avg.max() > 0:
            ax.contour(lung_avg, levels=[0.50], colors=["white"], linewidths=1.0, linestyles="--")
        _style_image_axis(ax, title=f"{title}\nn={bucket['n']}", title_size=12.5)
    prefix = f"{run_label} — " if run_label else ""
    fig.suptitle(
        prefix + "Aggregate Validation-Selected Pneumonia CAM",
        fontsize=14.5, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.025,
        "Cyan contour = radiologist GT prevalence; white dashed contour = lung-field prevalence.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.82, bottom=0.08, wspace=0.08)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] Aggregate primary localization heatmaps saved: {Path(path)}")

def save_xai_aggregate_prior_audit(agg, path, paper_dpi=600, save_pdf=False, run_label=""):
    """Show whether the aggregate CAM exceeds a simple lung-field prior."""
    if "GT_1" not in agg or int(agg["GT_1"].get("n", 0)) <= 0:
        print("[WARN] Aggregate prior audit skipped: no GT-positive aggregate bucket.")
        return
    bucket = agg["GT_1"]
    raw_cam = normalize_minmax(average_bucket_image(bucket, "raw_cam_sum"))
    gated_cam = normalize_minmax(average_bucket_image(bucket, "loc_sum"))
    lung_prior = normalize_minmax(average_bucket_image(bucket, "lung_mask_sum"))
    gt_prev = normalize_minmax(average_bucket_image(bucket, "gt_mask_sum"))
    residual = gated_cam - lung_prior
    max_abs = float(np.nanmax(np.abs(residual))) if residual.size else 1.0
    max_abs = max(max_abs, 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(14.6, 9.2))
    axes = np.asarray(axes).ravel()
    panels = [
        (raw_cam, "Mean raw CAM", CMAP_HEATMAP, 0.0, 1.0),
        (lung_prior, "Mean lung-field prior", "viridis", 0.0, 1.0),
        (gated_cam, "Mean gated CAM", CMAP_HEATMAP, 0.0, 1.0),
        (gt_prev, "Radiologist GT prevalence", "viridis", 0.0, 1.0),
        (residual, "Gated CAM − lung prior", "coolwarm", -max_abs, max_abs),
    ]
    for ax, (img, title, cmap, vmin, vmax) in zip(axes[:5], panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        _style_image_axis(ax, title=title, title_size=11.5)
        fig.colorbar(im, ax=ax, fraction=0.043, pad=0.025)
    axes[5].axis("off")
    axes[5].text(
        0.04, 0.72,
        "Interpretation\n\n"
        "Residual > 0:\nCAM exceeds the lung prior\n\n"
        "Residual < 0:\nlung prior exceeds the CAM\n\n"
        "All maps use the same GT-positive cohort.",
        ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FAFAFA", ec="#90A4AE", lw=1.2),
    )
    prefix = f"{run_label} — " if run_label else ""
    fig.suptitle(
        prefix + f"Aggregate Attribution-vs-Anatomical-Prior Audit (GT-positive, n={bucket['n']})",
        fontsize=14.5, fontweight="bold", y=0.975,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.045, wspace=0.16, hspace=0.18)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] Aggregate prior audit saved: {Path(path)}")

def select_failure_cases(metrics_df, n_failure=6, loc_score_threshold=0.10):
    """Balanced audit with metric-specific, non-overclaiming group labels."""
    if metrics_df is None or len(metrics_df) == 0:
        return pd.DataFrame()
    df = metrics_df.copy()
    df = df[df.get("heatmap_valid", True).astype(bool)].copy()
    for col in ["localization_score", "iou_at_thr", "prob_pneumonia"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    n = max(int(n_failure), 3)
    n_fn = max(1, n // 3)
    n_worst = max(1, n // 3)
    n_best = max(1, n - n_fn - n_worst)
    parts = []
    fn = df[df["category"] == "FN"].sort_values(
        ["localization_score", "prob_pneumonia"], ascending=[True, True]
    ).head(n_fn).copy()
    fn["audit_group"] = "FN — Lowest Max-IoU"
    parts.append(fn)
    tp = df[df["category"] == "TP"].copy()
    worst = tp.sort_values(["iou_at_thr", "localization_score"], ascending=[True, True]).head(n_worst).copy()
    worst["audit_group"] = "TP — Lowest fixed-threshold IoU"
    parts.append(worst)
    best = tp.sort_values(["localization_score", "iou_at_thr"], ascending=[False, False]).head(n_best).copy()
    best["audit_group"] = "TP — Highest Max-IoU"
    parts.append(best)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out.head(n)


def save_failure_analysis_grid(
    model, preprocess_fn, test_df, metrics_df, candidate_layers, gt_box_map,
    path, image_size=224, heatmap_threshold_pct=0.60,
    loc_score_threshold=0.10, n_failure=6, paper_dpi=600, save_pdf=False,
    models_pp_layers=None, lung_mask_dir=None, use_lung_mask=False,
    allow_ellipse=False, localization_cam_method="GradCAM", cam_min_grid=16,
    cam_smooth_sigma_frac=0.005, run_label="",
):
    cases = select_failure_cases(metrics_df, n_failure=n_failure,
                                 loc_score_threshold=loc_score_threshold)
    if len(cases) == 0:
        print("[WARN] No cases available for localization audit grid.")
        return
    n = len(cases)
    fig = plt.figure(figsize=(18.0, max(6.0, 2.8 * n + 1.1)))
    gs = gridspec.GridSpec(
        n, 5, figure=fig,
        width_ratios=[0.42, 1.0, 1.0, 1.0, 1.18],
        left=0.025, right=0.995, top=0.925, bottom=0.035,
        hspace=0.32, wspace=0.07,
    )
    for r, (_, case) in enumerate(cases.iterrows()):
        idx = int(case["sample_index"])
        row = test_df.iloc[idx]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(row["image_path"], image_size)
        raw_rgb = np.repeat(raw[..., None], 3, axis=-1)
        raw_u8 = to_uint8(raw_rgb)
        loc_raw, _method, _details = compute_class_localization_map(
            models_pp_layers or [(model, preprocess_fn, candidate_layers)],
            raw_rgb, image_size, class_index=1,
            smooth_sigma_frac=cam_smooth_sigma_frac,
            force_method=localization_cam_method, min_grid=cam_min_grid,
        )
        if use_lung_mask:
            loc_map, lung_mode, _lung = gate_heatmap_to_lungs(
                loc_raw, pid, lung_mask_dir, image_size,
                allow_ellipse=allow_ellipse,
            )
        else:
            loc_map, lung_mode = loc_raw, "off"
        gt_boxes = scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
        overlay, _ = overlay_heatmap_rgba(raw_u8, loc_map, alpha=0.58)
        pred_mask = heatmap_to_binary_mask(loc_map, threshold_pct=heatmap_threshold_pct)
        gt_mask = boxes_to_mask(gt_boxes, image_size, image_size)
        cat = str(case.get("category", ""))
        border = CATEGORY_COLORS.get(cat, "black")

        label_ax = fig.add_subplot(gs[r, 0])
        label_ax.axis("off")
        label_ax.text(
            0.52, 0.5, f"{pid[:8]} | {cat}\nP={format_metric(case.get('prob_pneumonia'))}",
            rotation=90, ha="center", va="center",
            fontsize=8.3, fontweight="bold", color=border,
        )

        ax0 = fig.add_subplot(gs[r, 1]); ax0.imshow(raw_u8)
        draw_boxes(ax0, gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        _style_image_axis(ax0, title="Original + GT" if r == 0 else None, border_color=border, title_size=10)

        ax1 = fig.add_subplot(gs[r, 2]); ax1.imshow(overlay)
        draw_boxes(ax1, gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        annotate_pneumonia_location(
            ax1, loc_map, threshold_pct=heatmap_threshold_pct,
            label_text="CAM focus", color=PRED_COLOR, fontsize=7.5,
        )
        _style_image_axis(ax1, title="Selected CNN CAM" if r == 0 else None, border_color=border, title_size=10)

        ax2 = fig.add_subplot(gs[r, 3])
        compare = np.zeros((image_size, image_size, 3), dtype=np.float32)
        compare[..., 0] = pred_mask.astype(np.float32)
        compare[..., 1] = gt_mask.astype(np.float32)
        compare[..., 2] = gt_mask.astype(np.float32)
        ax2.imshow(compare)
        _style_image_axis(ax2, title="CAM mask vs GT" if r == 0 else None, border_color=border, title_size=10)
        if r == 0:
            ax2.text(0.5, -0.08, "red=CAM | cyan=GT", transform=ax2.transAxes,
                     ha="center", va="top", fontsize=7.5, clip_on=False)

        ax3 = fig.add_subplot(gs[r, 4]); ax3.axis("off")
        text = (
            f"{case.get('audit_group', '')}\n"
            f"PG={format_metric(case.get('pointing_hit'), 0)} | "
            f"IoU@{heatmap_threshold_pct:.2f}={format_metric(case.get('iou_at_thr'))}\n"
            f"Max-IoU={format_metric(case.get('localization_score'))} | "
            f"EnergyGT={format_metric(case.get('energy_inside_gt'))}\n"
            f"Outside lung: raw={format_metric(case.get('outside_lung_ratio_raw'))}, "
            f"gated={format_metric(case.get('outside_lung_ratio_gated'))}\n"
            f"Peak distance={format_metric(case.get('peak_distance_norm'))} | "
            f"mask={_compact_xai_text(lung_mode, 26)}"
        )
        ax3.text(
            0.02, 0.5, textwrap.fill(text, width=55, replace_whitespace=False),
            ha="left", va="center", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.42", fc="#FAFAFA", ec=border, lw=1.2),
        )

    prefix = f"{run_label} — " if run_label else ""
    fig.suptitle(
        prefix + "Localization Audit: FN Lowest Max-IoU, TP Lowest Fixed-IoU, and TP Highest Max-IoU",
        fontsize=14.2, fontweight="bold", y=0.978,
    )
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    print(f"[OK] Balanced localization audit saved: {Path(path)}")

def _select_explainer_indices(metrics_df, n_samples=4):
    if metrics_df is None or len(metrics_df) == 0 or n_samples <= 0:
        return []
    order = []
    for cat in ["TP", "FN", "FP", "TN"]:
        sub = metrics_df[metrics_df["category"] == cat]
        if len(sub):
            order.extend(sub["sample_index"].astype(int).head(max(1, n_samples // 4)).tolist())
    if len(order) < n_samples:
        for idx in metrics_df["sample_index"].astype(int).tolist():
            if idx not in order:
                order.append(idx)
            if len(order) >= n_samples:
                break
    return order[:n_samples]



def save_lime_explanations(
    model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
    image_size=224, n_samples=4, lime_num_samples=2000,
    lime_segments=120, paper_dpi=600, save_pdf=False,
    lung_mask_dir=None, use_lung_mask=False, allow_ellipse=False,
    repeats=3, reports_dir=None,
):
    if n_samples <= 0:
        print("[INFO] LIME skipped because n_lime=0.")
        return
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries, slic
    except Exception as exc:
        print(f"[WARN] LIME skipped: {exc}")
        return
    indices = _select_explainer_indices(metrics_df, n_samples=n_samples)
    if not indices:
        return

    def predict_fn(images):
        arr = np.asarray(images, dtype=np.float32)
        if arr.max() > 2.0:
            arr /= 255.0
        if getattr(model, "_is_torch_adapter", False):
            return model.predict_numpy(arr, mc_passes=1)
        return model(preprocess_fn(tf.constant(arr)), training=False).numpy()

    fig = plt.figure(figsize=(12.8, max(5.2, 3.05 * len(indices) + 0.9)))
    gs = gridspec.GridSpec(
        len(indices), 3, figure=fig,
        width_ratios=[0.42, 1.0, 1.05],
        left=0.03, right=0.995, top=0.91, bottom=0.04,
        wspace=0.09, hspace=0.46,
    )
    stability_rows = []
    for r, idx in enumerate(indices):
        row = test_df.iloc[idx]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(row["image_path"], image_size)
        raw_rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32)
        gt_boxes = scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
        lung_mask, lung_mode = load_lung_mask_for_patient(
            pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
        ) if use_lung_mask else (None, "off")
        segments = slic(raw_rgb, n_segments=int(lime_segments), compactness=10,
                        sigma=1, start_label=1)
        masks = []
        for rep in range(max(1, int(repeats))):
            try:
                explainer = lime_image.LimeImageExplainer(random_state=42 + rep)
                explanation = explainer.explain_instance(
                    raw_rgb, predict_fn, labels=(1,), top_labels=None,
                    hide_color=0, num_samples=int(lime_num_samples),
                    segmentation_fn=lambda _img, seg=segments: seg,
                )
                _temp, mask = explanation.get_image_and_mask(
                    label=1, positive_only=True, num_features=10, hide_rest=False
                )
                binary = mask > 0
                if lung_mask is not None:
                    binary &= lung_mask
                masks.append(binary)
            except Exception as exc:
                print(f"[WARN] LIME failed for {pid}, repeat={rep}: {exc}")
        if masks:
            consensus = np.mean(np.stack(masks), axis=0) >= 0.5
            jaccards = []
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    union = np.logical_or(masks[i], masks[j]).sum()
                    jaccards.append(np.logical_and(masks[i], masks[j]).sum() / union if union else 1.0)
            stability = float(np.mean(jaccards)) if jaccards else 1.0
        else:
            consensus = np.zeros((image_size, image_size), dtype=bool)
            stability = np.nan
        gt_mask = boxes_to_mask(gt_boxes, image_size, image_size)
        union = np.logical_or(consensus, gt_mask).sum()
        iou = float(np.logical_and(consensus, gt_mask).sum() / union) if union else np.nan
        stability_rows.append({
            "patientId": pid, "lime_repeats": int(repeats),
            "lime_stability_jaccard": stability, "lime_gt_iou": iou,
            "lung_mask_mode": lung_mode,
        })

        label_ax = fig.add_subplot(gs[r, 0]); label_ax.axis("off")
        label_ax.text(0.52, 0.5, pid[:8], rotation=90, ha="center", va="center",
                      fontsize=8.5, fontweight="bold")
        ax0 = fig.add_subplot(gs[r, 1]); ax0.imshow(raw_rgb, cmap="gray")
        draw_boxes(ax0, gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        _style_image_axis(ax0, title="Original + GT" if r == 0 else None, title_size=10.5)
        ax1 = fig.add_subplot(gs[r, 2]); ax1.imshow(mark_boundaries(raw_rgb, consensus.astype(np.int32)))
        draw_boxes(ax1, gt_boxes, color=GT_COLOR, label="GT", fontsize=7.5)
        _style_image_axis(ax1, title="LIME consensus" if r == 0 else None, title_size=10.5)
        _add_metric_footer(
            ax1,
            f"stability={format_metric(stability)} | GT-IoU={format_metric(iou)} | "
            f"mask={_compact_xai_text(lung_mode, 28)}",
            fontsize=7.2, y=-0.12, width=52,
        )

    fig.suptitle("LIME Stability Audit — Pneumonia Class", fontsize=14.2, fontweight="bold", y=0.968)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06)
    plt.close(fig)
    if reports_dir is not None:
        pd.DataFrame(stability_rows).to_csv(Path(reports_dir) / "xai_lime_stability.csv", index=False)
    print(f"[OK] LIME stability explanations saved: {Path(path)}")

def _extract_shap_class_values(shap_values, class_index=1):
    if isinstance(shap_values, (list, tuple)):
        if len(shap_values) > class_index:
            return np.asarray(shap_values[class_index])
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    # Common newer SHAP format: (n, h, w, c, outputs)
    if arr.ndim == 5 and arr.shape[-1] > class_index:
        return arr[..., class_index]
    # Sometimes: (outputs, n, h, w, c)
    if arr.ndim == 5 and arr.shape[0] > class_index:
        return arr[class_index]
    return arr



def _extract_torch_shap_class_values(shap_values, class_index=1, expected_n=None):
    """Normalize SHAP PyTorch outputs to (N, C, H, W) for one class."""
    if isinstance(shap_values, (list, tuple)):
        if not shap_values:
            raise ValueError("SHAP returned an empty list")
        chosen = shap_values[class_index] if len(shap_values) > class_index else shap_values[-1]
        arr = np.asarray(chosen)
    else:
        arr = np.asarray(shap_values)

    # Newer SHAP may return (N, C, H, W, outputs).
    if arr.ndim == 5 and arr.shape[-1] > class_index:
        arr = arr[..., class_index]
    # Alternative layout: (outputs, N, C, H, W).
    elif arr.ndim == 5 and arr.shape[0] > class_index:
        if expected_n is None or arr.shape[1] == int(expected_n):
            arr = arr[class_index]

    if arr.ndim != 4:
        raise ValueError(f"Expected class SHAP tensor with 4 dimensions, got {arr.shape}")
    if expected_n is not None and arr.shape[0] != int(expected_n):
        raise ValueError(f"SHAP sample mismatch: got {arr.shape[0]}, expected {expected_n}")
    return arr.astype(np.float32)


def _torch_shap_signed_maps(adapter, bg_raw, x_raw, class_index=1):
    """Exact Gradient SHAP for the selected PyTorch model; no proxy substitution."""
    import torch
    import shap

    if not getattr(adapter, "_is_torch_adapter", False):
        raise TypeError("Torch SHAP requires TorchCamAdapter")

    # The adapter applies the exact deployment preprocessing. SHAP then receives
    # tensors in the native model input space, preserving differentiability.
    bg_tensor = adapter._tensor(bg_raw).detach()
    x_tensor = adapter._tensor(x_raw).detach()
    model = adapter.model
    model.eval()

    # Retry with smaller background sets only on memory pressure. This does not
    # alter the explained model or samples; it only reduces the SHAP reference set.
    sizes = []
    for n in [len(bg_tensor), min(8, len(bg_tensor)), min(4, len(bg_tensor))]:
        if n > 0 and n not in sizes:
            sizes.append(n)
    last_exc = None
    for n_bg in sizes:
        try:
            background = bg_tensor[:n_bg]
            explainer = shap.GradientExplainer(model, background)
            values = explainer.shap_values(x_tensor)
            vals = _extract_torch_shap_class_values(
                values, class_index=class_index, expected_n=len(x_tensor)
            )
            # Convert NCHW to one signed spatial attribution map per sample.
            signed = vals.sum(axis=1)
            return signed.astype(np.float32), int(n_bg)
        except RuntimeError as exc:
            last_exc = exc
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            last_exc = exc
            raise
    raise RuntimeError(f"PyTorch Gradient SHAP failed: {last_exc}")


def save_shap_explanations(
    model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
    image_size=224, n_samples=4, background_n=16, paper_dpi=600,
    save_pdf=False, background_df=None, lung_mask_dir=None,
    use_lung_mask=False, allow_ellipse=False, reports_dir=None,
):
    if n_samples <= 0:
        print("[INFO] SHAP skipped because n_shap=0.")
        return
    try:
        import shap  # noqa: F401
    except Exception as exc:
        print(f"[WARN] SHAP skipped: {exc}")
        return

    indices = _select_explainer_indices(metrics_df, n_samples=n_samples)
    if not indices:
        print("[WARN] SHAP skipped because no eligible samples were selected.")
        return

    bg_df = background_df if background_df is not None and len(background_df) else test_df
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(
        len(bg_df), size=min(max(1, int(background_n)), len(bg_df)), replace=False
    )
    bg_raw = []
    for idx in bg_indices:
        raw = load_dicom_grayscale(bg_df.iloc[int(idx)]["image_path"], image_size)
        bg_raw.append(np.repeat(raw[..., None], 3, axis=-1))
    bg_raw = np.stack(bg_raw).astype(np.float32)

    x_raw, pids, gt_boxes_list, lung_masks = [], [], [], []
    for idx in indices:
        row = test_df.iloc[int(idx)]
        pid = str(row["patientId"])
        raw = load_dicom_grayscale(row["image_path"], image_size)
        x_raw.append(np.repeat(raw[..., None], 3, axis=-1))
        pids.append(pid)
        gt_boxes_list.append(
            scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
        )
        lm, _ = load_lung_mask_for_patient(
            pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
        ) if use_lung_mask else (None, "off")
        lung_masks.append(lm)
    x_raw = np.stack(x_raw).astype(np.float32)

    backend = "pytorch" if getattr(model, "_is_torch_adapter", False) else "tensorflow"
    used_background_n = len(bg_raw)
    try:
        if backend == "pytorch":
            signed_maps, used_background_n = _torch_shap_signed_maps(
                model, bg_raw, x_raw, class_index=1
            )
        else:
            bg_pre = preprocess_fn(tf.constant(bg_raw)).numpy()
            x_pre = preprocess_fn(tf.constant(x_raw)).numpy()
            explainer = shap.GradientExplainer(model, bg_pre)
            vals = _extract_shap_class_values(
                explainer.shap_values(x_pre), class_index=1
            )
            if vals.shape[0] != x_raw.shape[0] and vals.ndim >= 5:
                vals = vals[0]
            if vals.shape[0] != x_raw.shape[0]:
                raise ValueError(f"SHAP output shape not understood: {vals.shape}")
            signed_maps = []
            for r in range(len(indices)):
                sample_vals = np.asarray(vals[r], dtype=np.float32)
                # TensorFlow image inputs are NHWC.
                signed_maps.append(sample_vals.sum(axis=-1))
            signed_maps = np.stack(signed_maps).astype(np.float32)
    except Exception as exc:
        print(f"[WARN] Exact {backend} Gradient SHAP failed: {type(exc).__name__}: {exc}")
        if reports_dir is not None:
            Path(reports_dir).mkdir(parents=True, exist_ok=True)
            Path(reports_dir, "xai_shap_scope.json").write_text(
                json.dumps({
                    "status": "failed",
                    "backend": backend,
                    "method": "shap.GradientExplainer",
                    "proxy_used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }, indent=2), encoding="utf-8"
            )
        return

    processed_maps = []
    for r, sm in enumerate(signed_maps):
        sm = np.asarray(sm, dtype=np.float32)
        if sm.shape != (int(image_size), int(image_size)):
            # Resize signed maps without clipping their sign.
            factor = (int(image_size) / sm.shape[0], int(image_size) / sm.shape[1])
            sm = zoom(sm, factor, order=1).astype(np.float32)
            sm = sm[:int(image_size), :int(image_size)]
        if lung_masks[r] is not None:
            lm = np.asarray(lung_masks[r], dtype=bool)
            if lm.shape != sm.shape:
                lm = resize_binary_mask(lm, sm.shape[0], sm.shape[1])
            sm = sm * lm.astype(np.float32)
        processed_maps.append(sm)
    signed_maps = processed_maps

    finite_parts = [np.abs(m).ravel() for m in signed_maps if np.isfinite(m).any()]
    global_abs = np.concatenate(finite_parts) if finite_parts else np.asarray([1.0])
    vmax = float(np.percentile(global_abs[np.isfinite(global_abs)], 99.0))
    vmax = max(vmax, 1e-8)

    fig = plt.figure(figsize=(13.2, max(5.2, 3.05 * len(indices) + 0.9)))
    gs = gridspec.GridSpec(
        len(indices), 3, figure=fig,
        width_ratios=[0.42, 1.0, 1.08],
        left=0.03, right=0.93, top=0.91, bottom=0.04,
        wspace=0.09, hspace=0.46,
    )
    metric_rows, shap_axes = [], []
    im = None
    for r, sm in enumerate(signed_maps):
        norm = np.clip(sm / vmax, -1, 1)
        raw_gray = x_raw[r][..., 0]
        label_ax = fig.add_subplot(gs[r, 0]); label_ax.axis("off")
        label_ax.text(
            0.52, 0.5, pids[r][:8], rotation=90, ha="center", va="center",
            fontsize=8.5, fontweight="bold"
        )
        ax0 = fig.add_subplot(gs[r, 1]); ax0.imshow(raw_gray, cmap="gray")
        draw_boxes(ax0, gt_boxes_list[r], color=GT_COLOR, label="GT", fontsize=7.5)
        _style_image_axis(ax0, title="Original + GT" if r == 0 else None, title_size=10.5)
        ax1 = fig.add_subplot(gs[r, 2]); ax1.imshow(raw_gray, cmap="gray")
        im = ax1.imshow(norm, cmap="coolwarm", vmin=-1, vmax=1, alpha=0.58)
        draw_boxes(ax1, gt_boxes_list[r], color=GT_COLOR, label="GT", fontsize=7.5)
        _style_image_axis(ax1, title="Signed SHAP" if r == 0 else None, title_size=10.5)
        shap_axes.append(ax1)

        gt_mask = boxes_to_mask(gt_boxes_list[r], image_size, image_size)
        abs_sm = np.abs(sm)
        total = float(abs_sm.sum())
        energy_gt = (
            float(abs_sm[gt_mask].sum() / total)
            if total > 1e-8 and gt_mask.any() else np.nan
        )
        metric_rows.append({
            "patientId": pids[r],
            "shap_abs_energy_inside_gt": energy_gt,
            "background_split": "train" if background_df is not None else "fallback",
            "backend": backend,
            "method": "shap.GradientExplainer",
            "proxy_used": False,
        })
        _add_metric_footer(
            ax1,
            f"red supports pneumonia | blue supports non-pneumonia | "
            f"|SHAP| energy in GT={format_metric(energy_gt)}",
            fontsize=7.1, y=-0.12, width=55,
        )

    if im is not None:
        fig.colorbar(
            im, ax=shap_axes, fraction=0.022, pad=0.018,
            label="Signed attribution (global symmetric scale)"
        )
    display_backend = "PyTorch" if backend == "pytorch" else "TensorFlow"
    fig.suptitle(
        f"Signed Gradient SHAP — Exact Selected {display_backend} Model",
        fontsize=14.2, fontweight="bold", y=0.968
    )
    _save_xai_figure(
        fig, path, paper_dpi=paper_dpi, save_pdf=save_pdf, pad_inches=0.06
    )
    plt.close(fig)

    if reports_dir is not None:
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metric_rows).to_csv(
            reports_dir / "xai_shap_metrics.csv", index=False
        )
        (reports_dir / "xai_shap_scope.json").write_text(
            json.dumps({
                "status": "completed",
                "backend": backend,
                "method": "shap.GradientExplainer",
                "explained_model": str(getattr(model, "name", _model_cam_key(model))),
                "exact_selected_model": True,
                "proxy_used": False,
                "background_split": "train" if background_df is not None else "fallback",
                "background_n_requested": int(background_n),
                "background_n_used": int(used_background_n),
                "n_explained": int(len(indices)),
                "class_index": 1,
            }, indent=2), encoding="utf-8"
        )
    print(f"[OK] Exact {display_backend} Gradient SHAP saved: {Path(path)}")

def save_roc_pr_curves(models_dict, path, paper_dpi=600):
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models_dict)))

    for ax, mode in zip(axes, ["ROC", "PR"]):
        for (name, (y_true, y_prob)), c in zip(models_dict.items(), colors):
            if mode == "ROC":
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ax.plot(fpr, tpr, color=c, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves", fontweight="bold")
            else:
                prec, rec, _ = precision_recall_curve(y_true, y_prob)
                ap = float(trapz_fn(prec[::-1], rec[::-1]))
                ax.plot(rec, prec, color=c, label=f"{name} (AP={ap:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision–Recall Curves", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8.2, loc="lower right", framealpha=0.95)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("ROC and Precision–Recall Curves — Locked Test Set", fontsize=14.2, fontweight="bold", y=0.97)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.12, wspace=0.18)
    _save_xai_figure(fig, path, paper_dpi=paper_dpi, save_pdf=False, pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] ROC+PR saved: {Path(path)}")

def load_keras_model_compat(path: Path):
    try:
        return tf.keras.models.load_model(str(path), compile=False, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(str(path), compile=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--dicom_dir", required=False, default="")
    parser.add_argument("--labels_csv", required=False, default="")
    parser.add_argument("--stage", default="xai", help="xai or lungmask")
    parser.add_argument("--pair_manifest", default="",
                        help="Locked validation-only DL pair manifest. Default: <results_dir>/reports/dl_pair_selection_manifest.json")
    parser.add_argument("--xrv_script", default="train_xrv_backbone.py")
    parser.add_argument("--eva_x_script", default="train_eva_x_backbone.py")
    parser.add_argument("--eva_x_repo", default="")
    parser.add_argument("--eva_x_checkpoint", default="")
    parser.add_argument("--eva_mean", type=float, nargs=3, default=(0.5,0.5,0.5))
    parser.add_argument("--eva_std", type=float, nargs=3, default=(0.5,0.5,0.5))
    parser.add_argument("--require_selected_pair", action="store_true", default=True,
                        help="Require a locked two-DL development-OOF selection manifest for final XAI.")
    parser.add_argument("--allow_nonselected_xai", action="store_true",
                        help="Diagnostic only: allow legacy EfficientNet/ResNet XAI when no selected-pair manifest exists.")
    parser.add_argument("--use_lung_mask", action="store_true")
    parser.add_argument("--no_lung_gate", action="store_true",
                        help="Matikan lung gating pada peta lokalisasi XAI (default: aktif).")
    parser.add_argument("--no_ellipse_fallback", action="store_true",
                        help="Disable ellipse fallback. Ellipse is used only when --allow_ellipse_fallback is explicit.")
    parser.add_argument("--lung_mask_method", choices=["pretrained", "ellipse"], default="pretrained",
                        help="pretrained (torchxrayvision PSPNet, thesis default) or ellipse heuristic prior.")
    parser.add_argument("--allow_ellipse_fallback", action="store_true",
                        help="If the pretrained lung segmenter cannot load, degrade to the ellipse prior instead of stopping.")
    parser.add_argument("--lung_seg_input_size", type=int, default=512,
                        help="Input resolution fed to the pretrained lung segmentation model.")
    parser.add_argument("--repair_heuristic_lung_masks", action="store_true",
                        help="Re-run pretrained deterministic TTA recovery for cached ellipse/unknown masks.")
    parser.add_argument("--no_baselines", action="store_true",
                        help="Skip random/center-bias chance baselines for XAI localization.")
    parser.add_argument("--detailed_class_info", default="",
                        help="Path to RSNA stage_2_detailed_class_info.csv for the 3-way paper figure. If empty, auto-detected next to labels_csv.")
    parser.add_argument("--comparison_n_per_class", type=int, default=2,
                        help="Representative examples per class in the paper comparison figures.")
    parser.add_argument("--kermany_dir", default="",
                        help="Optional path to the Kermany external test root (or directly its test directory). When set, a four-stage Kermany Hybrid CNN-ViT XAI figure is also exported.")
    parser.add_argument("--rsna_four_stage_n", type=int, default=4,
                        help="Deterministically sampled RSNA GT-positive four-panel XAI rows.")
    parser.add_argument("--kermany_four_stage_n", type=int, default=4,
                        help="Number of Kermany external-test cases shown in the four-stage XAI grid.")
    parser.add_argument("--kermany_max_samples", type=int, default=0,
                        help="Optional cap for Kermany XAI sample pool. 0 means the full official test cohort; probabilities always come from the existing locked external-validation predictions.")
    parser.add_argument("--train_script", default="",
                        help="Legacy optional path; external XAI reuses frozen Stage-19 probabilities and does not rerun radiomics.")
    parser.add_argument("--no_comparison", action="store_true",
                        help="Skip the curated Non-Pneumonia-vs-Pneumonia (and 3-way) paper figures.")
    parser.add_argument("--use_clahe", action="store_true", help="Reserved preprocessing flag for controlled CLAHE experiments")
    parser.add_argument("--chunk_id", type=int, default=None)
    parser.add_argument("--n_chunks", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--wavelet_levels", type=int, default=3,
                        help="Radiomic wavelet level used by train_aura.py; L=3 expects 746 features.")
    parser.add_argument("--allow_legacy_radiomic_cache", action="store_true",
                        help="Explicitly allow legacy adaptive_wavelet_features.pkl for non-final archived experiments only.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_xai", type=int, default=8, help="Qualitative GradCAM rows.")
    parser.add_argument("--cam_pdf_n", type=int, default=100,
                        help="Number of qualitative fused-CAM cases exported to the multipage PDF. Does not limit quantitative XAI evaluation.")
    parser.add_argument("--cam_pdf_per_page", type=int, default=4,
                        help="Number of CAM cases per page in the multipage PDF.")
    parser.add_argument("--cam_pdf_filename", default="xai_dual_cnn_fused_cam_100_samples.pdf",
                        help="Filename for the multipage qualitative fused-CAM PDF under plots/xai.")
    parser.add_argument("--no_cam_multipage_pdf", action="store_true",
                        help="Disable the dedicated multipage fused-CAM PDF export.")
    parser.add_argument("--cam_pdf_mode", choices=["balanced", "top_final_locked", "oracle_best_per_case", "both"], default="balanced",
                        help="PDF export mode: balanced = representative 100-case PDF (V19 behaviour); top_final_locked = top-scoring cases under the final locked CAM; oracle_best_per_case = exploratory per-case best CAM; both = generate final-locked and oracle PDFs.")
    parser.add_argument("--cam_pdf_metric", choices=["composite"], default="composite",
                        help="Ranking metric for top CAM PDF selection. Current implementation uses a composite localization score.")
    parser.add_argument("--cam_pdf_final_locked_filename", default="xai_dual_cnn_final_locked_top100.pdf",
                        help="Filename for the top final-locked CAM PDF under plots/xai.")
    parser.add_argument("--cam_pdf_oracle_filename", default="xai_dual_cnn_oracle_best_per_case_top100.pdf",
                        help="Filename for the exploratory oracle best-per-case CAM PDF under plots/xai.")
    parser.add_argument("--cam_pdf_oracle_topk_model_candidates", type=int, default=2,
                        help="Per selected CNN, number of top validation candidate CAM configurations considered in oracle exploratory PDF generation.")
    parser.add_argument("--cam_pdf_oracle_topk_fusion_candidates", type=int, default=5,
                        help="Number of top validation fusion candidates considered in oracle exploratory PDF generation.")
    parser.add_argument("--mc_n", type=int, default=30)
    parser.add_argument("--decision_threshold", type=float, default=0.40,
                        help="Threshold P(Pneumonia). Use 0.35-0.40 to favor sensitivity.")
    parser.add_argument("--gpu40", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--debug_gradcam", action="store_true", help="Print mean activation per layer for first XAI sample.")
    parser.add_argument("--paper_dpi", type=int, default=600, help="DPI for paper figures.")
    parser.add_argument("--save_pdf", action="store_true", help="Also save PDF versions of XAI figures.")
    parser.add_argument("--heatmap_threshold_pct", type=float, default=0.60,
                        help="Heatmap threshold as pct of max activation for IoU/predicted focus boxes.")
    parser.add_argument("--loc_score_threshold", type=float, default=0.10,
                        help="Threshold used to flag localization failure in failure analysis.")
    parser.add_argument("--xai_eval_max", type=int, default=0,
                        help="Max samples for quantitative XAI evaluation. 0 means all test samples.")
    parser.add_argument("--run_mode", choices=["auto", "quick", "final"], default="auto",
                        help="Hard artifact mode. auto: QUICK when xai_eval_max>0, otherwise FINAL.")
    parser.add_argument("--final_min_coverage", type=float, default=0.95,
                        help="Minimum evaluated and valid-heatmap coverage required in FINAL mode.")
    parser.add_argument("--final_min_valid_cases", type=int, default=1000,
                        help="Minimum valid GT-positive heatmaps required in FINAL mode.")
    parser.add_argument("--allow_final_baseline_report_only", action="store_true",
                        help="Allow FINAL mode with baseline_qc_mode=report. Not recommended for Q1.")
    parser.add_argument("--n_failure", type=int, default=6, help="Balanced localization-audit cases to plot.")
    parser.add_argument("--n_lime", type=int, default=4, help="Number of LIME examples. Use 0 to skip.")
    parser.add_argument("--lime_num_samples", type=int, default=2000, help="Perturbations per LIME repeat.")
    parser.add_argument("--lime_segments", type=int, default=120, help="SLIC superpixels for LIME.")
    parser.add_argument("--n_shap", type=int, default=4, help="Number of SHAP examples. Use 0 to skip.")
    parser.add_argument("--shap_background", type=int, default=16, help="Background samples for SHAP GradientExplainer.")
    parser.add_argument("--best_lime_pdf_n", type=int, default=100,
                        help="Number of highest-quality LIME explanations exported to the exploratory multipage PDF.")
    parser.add_argument("--best_lime_candidate_pool", type=int, default=150,
                        help="Candidate cases preselected by final-locked CAM quality before LIME ranking. Higher values improve search coverage but increase runtime.")
    parser.add_argument("--best_lime_pdf_per_page", type=int, default=4,
                        help="LIME cases per page in the best-LIME PDF.")
    parser.add_argument("--best_lime_pdf_filename", default="xai_lime_best_top100_exploratory.pdf",
                        help="Filename for exploratory best-LIME PDF under plots/xai.")
    parser.add_argument("--best_shap_pdf_n", type=int, default=100,
                        help="Number of highest-quality SHAP explanations exported to the exploratory multipage PDF.")
    parser.add_argument("--best_shap_candidate_pool", type=int, default=150,
                        help="Candidate cases preselected by final-locked CAM quality before SHAP ranking. Higher values improve search coverage but increase runtime.")
    parser.add_argument("--best_shap_pdf_per_page", type=int, default=4,
                        help="SHAP cases per page in the best-SHAP PDF.")
    parser.add_argument("--best_shap_pdf_filename", default="xai_shap_best_top100_exploratory.pdf",
                        help="Filename for exploratory best-SHAP PDF under plots/xai.")
    parser.add_argument("--no_best_lime_shap_pdfs", action="store_true",
                        help="Disable exploratory best-LIME and best-SHAP multipage PDF exports.")
    parser.add_argument("--no_resume_xai", action="store_true",
                        help="Disable XAI/MC-Dropout resume caches. Default: resume enabled.")
    parser.add_argument("--force_recompute_mc", action="store_true",
                        help="Ignore saved MC-Dropout probabilities and recompute CNN probabilities.")
    parser.add_argument("--xai_save_every", type=int, default=10,
                        help="Save quantitative XAI resume checkpoint every N new samples.")
    parser.add_argument("--localization_cam_method", choices=["validation_selected", "GradCAM", "GradCAM++", "LayerCAM", "HiResCAM", "auto"], default="validation_selected",
                        help="Primary CAM. validation_selected locks method(s) using validation boxes only.")
    parser.add_argument("--cam_method_candidates", default="GradCAM,LayerCAM,HiResCAM",
                        help="Validation-only CAM methods. Stable default excludes GradCAM++; opt in explicitly only for compatibility experiments.")
    parser.add_argument("--cam_map_policy_candidates", default="raw,soft",
                        help="Primary validation policies. Q1 default excludes hard gating; hard remains supplementary only.")
    parser.add_argument("--cam_threshold_candidates", default="0.25,0.35,0.45,0.55,0.65,0.75",
                        help="Validation-only fixed heatmap thresholds.")
    parser.add_argument("--cam_model_weight_grid", default="0.25,0.50,0.75",
                        help=("Two-CNN CAM fusion weight candidates selected on validation. "
                              "Endpoint weights 0 and 1 are excluded by default so both locked CNNs contribute."))
    parser.add_argument("--cam_min_model_weight", type=float, default=0.25,
                        help=("Minimum contribution required from each selected CNN during validation-only CAM fusion selection. "
                              "For two CNNs, each candidate must lie in [min_weight, 1-min_weight]."))
    parser.add_argument("--allow_partial_cam_fusion", action="store_true",
                        help=("Compatibility escape hatch: allow final CAM fusion to fall back to the available CNN when another CAM fails. "
                              "Disabled by default; final thesis runs should keep strict hybrid CNN-ViT contribution."))
    parser.add_argument("--cam_selection_metric_size", type=int, default=96,
                        help="Downsampled resolution for validation localization selection metrics.")
    parser.add_argument("--soft_lung_outside_weight", type=float, default=0.20,
                        help="Outside-lung multiplier for the soft anatomical constraint.")
    parser.add_argument("--cam_min_grid", type=int, default=16,
                        help="Minimum spatial CAM grid for deterministic architecture-level layer selection.")
    parser.add_argument("--cam_layer_policy", choices=["validation_fixed", "architecture_fixed"], default="validation_fixed",
                        help="Lock one layer per CNN. Q1 default selects on validation GT only and saves a manifest.")
    parser.add_argument("--cam_layer_val_samples", type=int, default=256,
                        help="Validation-positive selection subset. 0 uses all; selected configuration is confirmed on all positives.")
    parser.add_argument("--cam_validation_folds", type=int, default=5,
                        help="Deterministic folds used to penalize unstable CAM configurations during validation selection.")
    parser.add_argument("--cam_selection_bootstrap", type=int, default=500,
                        help="Paired bootstrap replicates for strong-baseline superiority during CAM selection.")
    parser.add_argument("--no_cam_validation_confirm_all", action="store_true",
                        help="Disable all-positive validation confirmation. Not recommended for final Q1 runs.")
    parser.add_argument("--allow_ellipse_in_cam_validation", action="store_true",
                        help="Allow ellipse fallback cases in CAM configuration selection. Disabled by default.")
    parser.add_argument("--cam_layer_candidates", type=int, default=5,
                        help="Architecture-level candidate layers evaluated per CNN on validation.")
    parser.add_argument("--cam_candidate_pilot_samples", type=int, default=12,
                        help="Initial validation cases used to reject systematically invalid layer-method candidates early.")
    parser.add_argument("--cam_candidate_pilot_min_valid_rate", type=float, default=0.80,
                        help="Minimum valid-map rate in the pilot screen before full candidate evaluation.")
    parser.add_argument("--cam_candidate_min_valid_rate", type=float, default=0.95,
                        help="Minimum full-validation valid-map rate required for CAM candidate eligibility.")
    parser.add_argument("--allow_depthwise_cam_candidates", action="store_true",
                        help="Explicitly include DepthwiseConv2D layers. Disabled by default due to repeated degenerate CAMs.")
    parser.add_argument("--force_cam_layer_selection", action="store_true",
                        help="Ignore an existing validation CAM-layer manifest and select again.")
    parser.add_argument("--cam_smooth_sigma_frac", type=float, default=0.005,
                        help="Small Gaussian smoothing sigma as fraction of image size.")
    parser.add_argument("--xai_bootstrap", type=int, default=2000,
                        help="Bootstrap replicates for 95%% CI of XAI metrics.")
    parser.add_argument("--max_invalid_heatmap_rate", type=float, default=0.02,
                        help="Maximum allowed invalid/zero heatmap rate before Q1 QC fails.")
    parser.add_argument("--max_gated_outside_lung_ratio", type=float, default=0.001,
                        help="Maximum allowed outside-lung energy ratio after gating.")
    parser.add_argument("--min_effective_lung_mask_coverage", type=float, default=0.99,
                        help="Minimum FINAL coverage after pretrained recovery and explicit fallback.")
    parser.add_argument("--max_lung_mask_fallback_rate", type=float, default=0.05,
                        help="Maximum fraction allowed to use an explicitly recorded ellipse fallback.")
    parser.add_argument("--fail_on_technical_qc", action="store_true",
                        help="Abort immediately on FINAL technical QC failure. Default completes all reports and records not-ready status.")
    parser.add_argument("--random_baseline_repeats", type=int, default=5,
                        help="Independent random maps per case for the random baseline.")
    parser.add_argument("--center_baseline_sigma_frac", type=float, default=0.25,
                        help="Fixed Gaussian center-prior sigma as fraction of image size.")
    parser.add_argument("--faithfulness_eval_max", type=int, default=256,
                        help="Valid positive cases for deletion/retention faithfulness; 0 skips.")
    parser.add_argument("--faithfulness_only", action="store_true",
                        help="Run only the immutable final hybrid CNN-ViT faithfulness audit using existing full-cohort XAI localization metrics.")
    parser.add_argument("--faithfulness_artifact_dir", default="faithfulness_v18_dual_cnn_full256",
                        help="Dedicated reports subdirectory for immutable full-256 faithfulness artifacts.")
    parser.add_argument("--faithfulness_top_fraction", type=float, default=0.10,
                        help="Top CAM fraction used by deletion/retention faithfulness.")
    parser.add_argument("--xai_sanity_samples", type=int, default=0,
                        help="Validation cases for classifier-head randomization sanity; 0 skips.")
    parser.add_argument("--fail_on_localization_claim_qc", action="store_true",
                        help="Abort only when scientific localization-claim guardrails fail. Default completes reports.")
    parser.add_argument("--baseline_qc_mode", choices=["report", "mean", "ci"], default="mean",
                        help="Baseline QC: report only, require positive mean difference, or require paired CI > 0.")
    parser.add_argument("--allow_xai_qc_fail", action="store_true",
                        help="Continue despite failed XAI QC; not recommended for final paper outputs.")
    parser.add_argument("--allow_soft_fallback", action="store_true",
                        help="Permit soft-voting fallback when final stacked probabilities are unavailable. Disabled for Q1 final runs.")
    parser.add_argument("--lime_repeats", type=int, default=3,
                        help="Repeated LIME runs per image for stability analysis.")
    parser.add_argument("--use_mixed_precision_xai", action="store_true",
                        help="Opt in to mixed precision for XAI. Default float32 improves gradient stability.")
    parser.add_argument("--enable_xla_xai", action="store_true",
                        help="Opt in to XLA during XAI. Disabled by default for stable gradients.")
    args = parser.parse_args()
    if int(args.image_size) != 224:
        raise ValueError("Q1 selected-pair XAI is locked to --image_size 224 for all candidate backbones.")

    print(f"[ENV] Python/TF/NumPy/Pandas: {os.sys.version.split()[0]} / {tf.__version__} / {np.__version__} / {pd.__version__}")
    if args.use_clahe:
        print("[WARN] --use_clahe is not applied during XAI because final models were trained with the standard preprocessing. This prevents explanation-time distribution shift.")
    if tuple(map(int, os.sys.version.split()[0].split('.')[:2])) >= (3, 13):
        print("[ENV] Python 3.13 mode: evaluation prefers .keras, then .h5 if needed.")

    if args.gpu40:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[GPU] CUDA_VISIBLE_DEVICES={args.gpu_id}")
        policy = "mixed_float16" if args.use_mixed_precision_xai else "float32"
        tf.keras.mixed_precision.set_global_policy(policy)
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.config.optimizer.set_jit(bool(args.enable_xla_xai))
        print(f"[GPU] XAI policy={policy} | XLA={'enabled' if args.enable_xla_xai else 'disabled'}")

    results = Path(args.results_dir)
    plots = ensure_dir(results / "plots")
    xai_dir = ensure_dir(results / "xai")
    reports = ensure_dir(results / "reports")
    probs_dir = ensure_dir(results / "probs")
    run_mode = resolve_xai_run_mode(args.run_mode, args.xai_eval_max)
    if not (0 < float(args.final_min_coverage) <= 1.0):
        raise ValueError("--final_min_coverage must be in (0,1].")
    if int(args.final_min_valid_cases) < 1:
        raise ValueError("--final_min_valid_cases must be >= 1.")
    if not (0 < float(args.min_effective_lung_mask_coverage) <= 1.0):
        raise ValueError("--min_effective_lung_mask_coverage must be in (0,1].")
    if not (0 <= float(args.max_lung_mask_fallback_rate) <= 1.0):
        raise ValueError("--max_lung_mask_fallback_rate must be in [0,1].")
    if run_mode == "QUICK":
        xai_output_dir = ensure_dir(xai_dir / "QUICK_PRECHECK")
        xai_reports = ensure_dir(reports / "xai_quick_precheck")
        xai_plots = ensure_dir(plots / "xai_quick_precheck")
        print("=" * 78)
        print("[QUICK PRECHECK] Diagnostic subset only — outputs are NOT final paper evidence.")
        print("=" * 78)
    else:
        xai_output_dir = xai_dir
        xai_reports = reports
        xai_plots = plots
        if args.baseline_qc_mode == "report" and not args.allow_final_baseline_report_only:
            raise ValueError(
                "FINAL mode requires --baseline_qc_mode mean or ci. "
                "Use --allow_final_baseline_report_only only for non-paper diagnostics."
            )
        if args.no_baselines:
            raise ValueError(
                "FINAL mode requires random, image-center, and lung-prior baselines. "
                "Remove --no_baselines; baseline evidence is mandatory for Q1 final artifacts."
            )
        print("=" * 78)
        print("[FINAL MODE] Full GT-positive evaluation with hard Q1 guardrails.")
        print("=" * 78)

    print("[INFO] Loading splits...")
    test_df = pd.read_csv(results / "splits" / "test.csv")
    val_df = pd.read_csv(results / "splits" / "val.csv")
    train_path = results / "splits" / "train.csv"
    train_df = pd.read_csv(train_path) if train_path.exists() else pd.DataFrame()
    split_frames = [test_df, val_df] + ([train_df] if len(train_df) else [])
    if args.dicom_dir:
        dicom_dir = Path(args.dicom_dir)
        for df in split_frames:
            df["image_path"] = df["patientId"].astype(str).map(lambda x: str(dicom_dir / f"{x}.dcm"))
    elif any("image_path" not in df.columns for df in split_frames):
        raise ValueError("Provide --dicom_dir, or use split CSVs that contain image_path.")
    test_df = test_df[test_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    val_df = val_df[val_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if len(train_df):
        train_df = train_df[train_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    y_test = test_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    print(f"[INFO] test={len(test_df)} | val={len(val_df)} | train={len(train_df)}")

    lung_mask_dir = results / "cache" / "lung_masks"
    if args.stage.lower() == "lungmask":
        build_lung_mask_cache(test_df, lung_mask_dir, image_size=args.image_size,
                              chunk_id=args.chunk_id, n_chunks=args.n_chunks,
                              method=args.lung_mask_method,
                              allow_ellipse_fallback=args.allow_ellipse_fallback,
                              seg_input_size=args.lung_seg_input_size,
                              repair_heuristic_masks=args.repair_heuristic_lung_masks)
        manifest_path = results / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"stages": {}}
        manifest.setdefault("stages", {})["lungmask"] = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts": {"lung_mask_dir": str(lung_mask_dir)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[STAGE] lungmask complete: {lung_mask_dir}")
        return

    if args.use_lung_mask:
        mask_df = pd.concat([val_df[val_df["label"] == 1], test_df], ignore_index=True).drop_duplicates("patientId")
        existing = sum((lung_mask_dir / f"{pid}.npy").exists() for pid in mask_df["patientId"].astype(str))
        print(
            f"[LUNGMASK] Validation+test cache coverage {existing}/{len(mask_df)}; "
            "validating and completing cache."
        )
        build_lung_mask_cache(
            mask_df, lung_mask_dir, image_size=args.image_size,
            method=args.lung_mask_method,
            allow_ellipse_fallback=args.allow_ellipse_fallback,
            seg_input_size=args.lung_seg_input_size,
            repair_heuristic_masks=args.repair_heuristic_lung_masks,
        )

    if not args.labels_csv:
        raise ValueError("--labels_csv is required for XAI localization ground-truth boxes")
    gt_box_map = load_radiologist_bboxes(args.labels_csv)
    expected_gt_positive = count_gt_positive_with_boxes(test_df, gt_box_map, args.image_size)
    print(f"[XAI RUN] mode={run_mode} | expected GT-positive with boxes={expected_gt_positive}")
    write_xai_run_manifest(
        xai_reports / "xai_run_manifest.json",
        run_mode=run_mode,
        xai_eval_max=args.xai_eval_max,
        expected_gt_positive=expected_gt_positive,
        min_final_coverage=args.final_min_coverage,
        min_final_valid_cases=args.final_min_valid_cases,
        args=args,
    )


    print("[INFO] Loading exact models and probabilities from the locked deployment manifest...")
    global _XAI_DEPLOYMENT_LOCK_ID, _XAI_MODEL_HASHES
    model_dir=results/"models"; pair_manifest=load_pair_manifest(results,strict=True); deployment=load_deployment_manifest(results,strict=True)
    _XAI_DEPLOYMENT_LOCK_ID=str(deployment["deployment_lock_id"])
    _XAI_MODEL_HASHES={name:rec.get("sha256","") for name,rec in deployment.get("base_model_artifacts",{}).items() if name in pair_manifest.get("selected_dl_models",[])}
    if not pair_manifest.get("final_test_evaluated_once"):
        raise RuntimeError("Final XAI requires the completed one-time locked-test fusion")
    selected_dl_models=list(pair_manifest["selected_dl_models"])
    args.decision_threshold=float(pair_manifest["threshold"])
    print(f"[PAIR-XAI] Locked selected DL pair: {pair_manifest['selected_pair_display']}; threshold={args.decision_threshold:.6f}")
    radiomics_model_path=resolve_locked_model_artifact(results,"radiomics");radiomics_pipe=joblib.load(radiomics_model_path)
    radiomics_label=deployment.get("radiomics_selection", {}).get(
        "selected_display_name", "MRFO-optimized radiomics branch (best learner)"
    )

    def identity_preprocess(x): return x
    cnn_models={};cnn_specs=[]
    for name in selected_dl_models:
        path=resolve_locked_model_artifact(results,name)
        if name=="efficientnetv2": model_obj=load_keras_model_compat(path);pp=preprocess_efficientnet
        elif name=="resnet50": model_obj=load_keras_model_compat(path);pp=preprocess_resnet
        elif name in {"xrv","eva_x"}: model_obj=load_torch_cam_adapter(name,path,args);pp=identity_preprocess
        else: raise ValueError(name)
        try:
            setattr(model_obj,"_aura_cam_name",name)
            setattr(model_obj,"_aura_checkpoint_sha256",sha256_file(path))
        except Exception:pass
        cnn_models[name]={"model":model_obj,"preprocess":pp,"path":path,"checkpoint_sha256":sha256_file(path)}
        cnn_specs.append((name,path,pp));print(f"[PAIR-XAI] exact development-refit {name}: {path.name}")
    primary_name=selected_dl_models[0];primary_model=cnn_models[primary_name]["model"];primary_preprocess=cnn_models[primary_name]["preprocess"]
    meta_path=Path(deployment["meta_learner_artifact"]["path"]);meta_clf=joblib.load(meta_path)
    xai_pair_scope={"schema":XAI_SCHEMA_VERSION,"selection_lock_id":pair_manifest["selection_lock_id"],
        "deployment_lock_id":deployment["deployment_lock_id"],"selected_dl_models":selected_dl_models,
        "selected_pair_display":pair_manifest["selected_pair_display"],"cam_models":selected_dl_models,
        "classification_source":"exact locked stacked probabilities","cam_scope":"all selected DL members",
        "cam_fusion_constraint_requested":"all selected CNNs must contribute positive weight",
        "nonselected_models_substituted":False,"cam_model_artifact_policy":"exact development refit only",
        "checkpoint_sha256":{n:cnn_models[n]["checkpoint_sha256"] for n in selected_dl_models},
        "decision_threshold_source":"development_oof_manifest","test_set_used_for_selection":False}
    (xai_reports/"xai_selected_pair_scope.json").write_text(json.dumps(xai_pair_scope,indent=2),encoding="utf-8")

    dev=ensure_development_table(results);val_mask=dev["original_split"].astype(str).eq("validation").to_numpy()
    def two(p):
        p=normalize_binary_probability(p);return np.column_stack([1-p,p]).astype(np.float32)
    rad_oof,_=load_oof_artifact(results,"radiomics",dev);radiomics_val_proba=two(rad_oof[val_mask])
    radiomics_test_proba=load_probability_array(resolve_probability_file(results,"radiomics","test",True),len(test_df),"locked radiomics test")
    radiomics_test_pred=predict_from_proba(radiomics_test_proba,float(pair_manifest["individual_thresholds"]["radiomics"]))
    for name in selected_dl_models:
        oof,_=load_oof_artifact(results,name,dev);val_mean=two(oof[val_mask])
        test_path=resolve_probability_file(results,name,"test",True);test_mean=load_probability_array(test_path,len(test_df),f"locked {name} test")
        std_path=Path(str(test_path).replace("_mean.npy","_std.npy"))
        test_std=np.load(std_path) if std_path.exists() else np.zeros_like(test_mean,dtype=np.float32)
        if test_std.ndim==1:test_std=np.column_stack([test_std,test_std])
        val_std=np.zeros_like(val_mean,dtype=np.float32);val_ci=np.zeros(len(val_mean),dtype=np.float32)
        width_path=Path(str(test_path).replace("_mean.npy","_predictive_interval_width_95.npy"))
        test_ci=np.load(width_path) if width_path.exists() else (2*1.96*test_std[:,1] if test_std.ndim==2 else np.zeros(len(test_mean),dtype=np.float32))
        thr=float(pair_manifest["individual_thresholds"][name])
        cnn_models[name].update({"val_mean":val_mean,"val_std":val_std,"val_ci":val_ci,
            "test_mean":test_mean,"test_std":test_std,"test_ci":test_ci,"test_pred":predict_from_proba(test_mean,thr),"threshold":thr})
    soft_val_proba=load_probability_array(probs_dir/"soft_val.npy",len(val_df),"soft development validation")
    stacked_val_proba=load_probability_array(probs_dir/"stacked_val.npy",len(val_df),"stacked development validation")
    soft_test_proba=load_probability_array(resolve_probability_file(results,"soft","test",True),len(test_df),"locked soft test")
    stacked_test_proba=load_probability_array(resolve_probability_file(results,"stacked","test",True),len(test_df),"locked stacked test")
    soft_test_pred=predict_from_proba(soft_test_proba,float(pair_manifest["soft_voting_threshold"]))
    stacked_test_pred=predict_from_proba(stacked_test_proba,args.decision_threshold)
    ensemble_member_names=["radiomics"]+selected_dl_models;stacked_source="locked_deployment_manifest"
    models_pp_layers=[(cnn_models[n]["model"],cnn_models[n]["preprocess"],_get_candidate_conv_layers(cnn_models[n]["model"])) for n in selected_dl_models]
    candidate_layers=_get_candidate_conv_layers(primary_model)
    print(f"[INFO] Exact selected models loaded; primary={primary_name}; candidate layers={len(candidate_layers)}")
    # Training-derived lesion-location prior is built before CAM selection so
    # validation optimization can explicitly penalize prevalence-prior mimicry.
    lesion_prior, lesion_prior_n = build_lesion_prevalence_prior(
        train_df, gt_box_map, args.image_size, reports_dir=xai_reports
    )
    cam_layer_manifest = configure_cam_layer_lock(
        models_pp_layers=models_pp_layers,
        val_df=val_df,
        y_val=y_val,
        gt_box_map=gt_box_map,
        image_size=args.image_size,
        reports_dir=reports,
        lung_mask_dir=lung_mask_dir,
        use_lung_mask=bool(args.use_lung_mask and not args.no_lung_gate),
        allow_ellipse=bool(args.allow_ellipse_fallback and not args.no_ellipse_fallback),
        policy=args.cam_layer_policy,
        val_samples=args.cam_layer_val_samples,
        candidate_count=args.cam_layer_candidates,
        force_method=args.localization_cam_method,
        method_candidates=args.cam_method_candidates,
        map_policy_candidates=args.cam_map_policy_candidates,
        candidate_pilot_samples=args.cam_candidate_pilot_samples,
        candidate_pilot_min_valid_rate=args.cam_candidate_pilot_min_valid_rate,
        candidate_min_valid_rate=args.cam_candidate_min_valid_rate,
        allow_depthwise_candidates=args.allow_depthwise_cam_candidates,
        threshold_candidates=args.cam_threshold_candidates,
        model_weight_grid=args.cam_model_weight_grid,
        cam_min_model_weight=args.cam_min_model_weight,
        require_all_selected_cnn_contributions=not args.allow_partial_cam_fusion,
        soft_lung_outside_weight=args.soft_lung_outside_weight,
        selection_metric_size=args.cam_selection_metric_size,
        lesion_prevalence_map=lesion_prior,
        validation_folds=args.cam_validation_folds,
        selection_bootstrap=args.cam_selection_bootstrap,
        validation_confirm_all=not args.no_cam_validation_confirm_all,
        allow_ellipse_in_validation=args.allow_ellipse_in_cam_validation,
        min_grid=args.cam_min_grid,
        heatmap_threshold_pct=args.heatmap_threshold_pct,
        force_reselect=args.force_cam_layer_selection,
        seed=42,
    )
    try:
        with (xai_reports / "xai_probability_source.json").open("w", encoding="utf-8") as f:
            json.dump({
                "stacked_source": stacked_source,
                "xai_models_source": "exact_locked_development_refit_selected_pair",
                "selected_dl_models": selected_dl_models,
                "selected_pair_display": pair_manifest.get("selected_pair_display") if pair_manifest else None,
                "cam_models": [x[0] for x in cnn_specs],
                "nonselected_models_substituted": False,
                "ensemble_members": ensemble_member_names,
                "selection_lock_id": pair_manifest["selection_lock_id"],
                "deployment_lock_id": deployment["deployment_lock_id"],
                "model_checkpoint_sha256": _XAI_MODEL_HASHES,
                "stacked_val_shape": list(stacked_val_proba.shape),
                "stacked_test_shape": list(stacked_test_proba.shape),
                "decision_threshold": float(args.decision_threshold),
                "xai_schema_version": XAI_SCHEMA_VERSION,
                "primary_localization_map": "Validation-selected constrained weighted hybrid CNN-ViT Pneumonia-class attribution",
                "cam_fusion_rule": "validation_selected_positive_weight_dual_cnn_model_and_layer_weights",
                "cam_layer_policy": args.cam_layer_policy,
                "cam_layer_selection_split": "validation",
                "cam_test_set_used_for_layer_selection": False,
                "cam_selected_layers": cam_layer_manifest.get("selected_layers", {}),
                "cam_selected_models": cam_layer_manifest.get("selected_models", {}),
                "cam_selected_model_weights": cam_layer_manifest.get("selected_model_weights", {}),
                "cam_min_model_weight": cam_layer_manifest.get("cam_min_model_weight"),
                "cam_require_all_selected_cnn_contributions": cam_layer_manifest.get("require_all_selected_cnn_contributions"),
                "cam_fusion_constraint": cam_layer_manifest.get("fusion_constraint"),
                "cam_selected_map_policy": cam_layer_manifest.get("selected_map_policy"),
                "cam_selected_threshold": cam_layer_manifest.get("selected_threshold"),
                "cam_validation_confirmation": cam_layer_manifest.get("validation_confirmation"),
                "cam_validation_baseline_superiority_pass": cam_layer_manifest.get("validation_baseline_superiority_pass"),
                "localization_cam_method": args.localization_cam_method,
                "cam_min_grid": int(args.cam_min_grid),
                "cam_candidate_pilot_samples": int(args.cam_candidate_pilot_samples),
                "cam_candidate_pilot_min_valid_rate": float(args.cam_candidate_pilot_min_valid_rate),
                "cam_candidate_min_valid_rate": float(args.cam_candidate_min_valid_rate),
                "allow_depthwise_cam_candidates": bool(args.allow_depthwise_cam_candidates),
                "cam_smooth_sigma_frac": float(args.cam_smooth_sigma_frac),
                "lung_mask_enabled": bool(args.use_lung_mask),
                "lung_mask_method": args.lung_mask_method if args.use_lung_mask else "off",
                "ellipse_fallback_explicit": bool(args.allow_ellipse_fallback),
                "xai_numeric_policy": "mixed_float16" if args.use_mixed_precision_xai else "float32",
                "xla_enabled": bool(args.enable_xla_xai),
                "run_mode": run_mode,
                "quick_outputs_are_not_final": bool(run_mode == "QUICK"),
                "expected_gt_positive_with_boxes": int(expected_gt_positive),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }, f, indent=2)
    except Exception as exc:
        print(f"[WARN] Could not save XAI probability-source audit: {exc}")

    if args.faithfulness_only:
        print("\n" + "=" * 78)
        print("[FAITHFULNESS-ONLY V26] Immutable hybrid CNN-ViT full-256 audit")
        print("=" * 78)
        if int(args.faithfulness_eval_max) != 256:
            raise ValueError("--faithfulness_only requires --faithfulness_eval_max 256 for the thesis-final audit.")
        if run_mode != "FINAL":
            raise ValueError("--faithfulness_only must run with --run_mode final and --xai_eval_max 0.")

        # Validate the final V18 hybrid CNN-ViT lock before any perturbation is run.
        expected_schema = "q1_fixed_best_radiomics_locked_pair_v18_dual_cnn_cam"
        if str(XAI_SCHEMA_VERSION) != expected_schema:
            raise RuntimeError(f"Unexpected XAI schema: {XAI_SCHEMA_VERSION!r}; expected {expected_schema!r}")
        if set(selected_dl_models) != {"xrv", "eva_x"}:
            raise RuntimeError(f"Faithfulness final requires selected DL pair xrv+eva_x, got {selected_dl_models}")
        if cam_layer_manifest.get("selection_split") != "validation":
            raise RuntimeError("CAM selection split is not validation.")
        if bool(cam_layer_manifest.get("test_set_used_for_selection")):
            raise RuntimeError("Locked internal test was used for CAM selection; refusing final faithfulness audit.")
        if cam_layer_manifest.get("fusion_constraint") != "all_selected_cnn_positive_contribution":
            raise RuntimeError("Strict positive-contribution hybrid CNN-ViT fusion constraint is missing.")
        if not bool(cam_layer_manifest.get("require_all_selected_cnn_contributions")):
            raise RuntimeError("Both selected CNNs must be required for final faithfulness.")
        selected_weights = cam_layer_manifest.get("selected_model_weights", {}) or {}
        if abs(float(selected_weights.get("xrv", -1)) - 0.25) > 1e-9 or abs(float(selected_weights.get("eva_x", -1)) - 0.75) > 1e-9:
            raise RuntimeError(f"Unexpected final model weights: {selected_weights}; expected xrv=0.25, eva_x=0.75")
        if str(cam_layer_manifest.get("selected_map_policy")) != "raw":
            raise RuntimeError(f"Unexpected map policy: {cam_layer_manifest.get('selected_map_policy')}")
        if abs(float(cam_layer_manifest.get("selected_threshold", -1)) - 0.25) > 1e-9:
            raise RuntimeError(f"Unexpected final heatmap threshold: {cam_layer_manifest.get('selected_threshold')}")
        for model_name in ("xrv", "eva_x"):
            cfg=(cam_layer_manifest.get("selected_models", {}) or {}).get(model_name, {})
            if list(cfg.get("methods", [])) != ["LayerCAM", "LayerCAM"]:
                raise RuntimeError(f"{model_name} final method is not two-layer LayerCAM: {cfg.get('methods')}")
            lw=[float(x) for x in cfg.get("layer_weights", [])]
            if len(lw) != 2 or abs(lw[0]-0.75) > 1e-9 or abs(lw[1]-0.25) > 1e-9:
                raise RuntimeError(f"{model_name} final layer weights are not 0.75/0.25: {lw}")

        source_metrics = reports / "xai_localization_per_sample.csv"
        if not source_metrics.exists():
            raise FileNotFoundError(
                f"Missing {source_metrics}. Run the final V18 hybrid CNN-ViT XAI localization first; "
                "faithfulness-only deliberately refuses to regenerate/overwrite localization evidence."
            )
        faith_metrics = pd.read_csv(source_metrics)
        if len(faith_metrics) != int(expected_gt_positive):
            raise RuntimeError(
                f"Final localization source is incomplete: rows={len(faith_metrics)} "
                f"but expected_gt_positive={expected_gt_positive}."
            )
        if "xai_schema_version" in faith_metrics.columns:
            schemas=set(faith_metrics["xai_schema_version"].dropna().astype(str).unique())
            if schemas != {expected_schema}:
                raise RuntimeError(f"Localization source schema mismatch: {sorted(schemas)}")
        if "cam_layer_selection_split" in faith_metrics.columns:
            splits=set(faith_metrics["cam_layer_selection_split"].dropna().astype(str).str.lower().unique())
            if splits != {"validation"}:
                raise RuntimeError(f"Localization source has non-validation CAM selection split(s): {sorted(splits)}")
        if "cam_test_set_used_for_layer_selection" in faith_metrics.columns:
            test_flags=faith_metrics["cam_test_set_used_for_layer_selection"].astype(str).str.lower().isin(["true","1","yes"])
            if bool(test_flags.any()):
                raise RuntimeError("Localization source indicates test-set use during CAM layer selection.")
        if "heatmap_valid" not in faith_metrics.columns:
            raise RuntimeError("Localization source is missing heatmap_valid.")
        valid_flags=faith_metrics["heatmap_valid"].astype(str).str.lower().isin(["true","1","yes"])
        n_valid=int(valid_flags.sum())
        if n_valid < 256:
            raise RuntimeError(f"Only {n_valid} valid heatmaps are available; 256 are required.")

        artifact_dir = ensure_dir(reports / str(args.faithfulness_artifact_dir))
        print(f"[FAITHFULNESS-ONLY] source metrics: {source_metrics} ({len(faith_metrics)} rows; valid={n_valid})")
        print(f"[FAITHFULNESS-ONLY] immutable artifact directory: {artifact_dir}")
        faith_df = evaluate_xai_faithfulness(
            faith_metrics, test_df, models_pp_layers, args.image_size, artifact_dir,
            max_samples=256,
            top_fraction=args.faithfulness_top_fraction,
            cam_smooth_sigma_frac=args.cam_smooth_sigma_frac, seed=42,
            use_lung_mask=args.use_lung_mask, lung_mask_dir=lung_mask_dir,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
        )
        per_path=artifact_dir / "xai_faithfulness_per_sample.csv"
        sum_path=artifact_dir / "xai_faithfulness_summary.json"
        if len(faith_df) != 256 or not per_path.exists() or not sum_path.exists():
            raise RuntimeError(f"Faithfulness run incomplete: rows={len(faith_df)}")
        summary=json.loads(sum_path.read_text(encoding="utf-8"))
        for metric_name in ("deletion_confidence_drop", "retention_probability_ratio"):
            if int(summary.get(metric_name, {}).get("n", -1)) != 256:
                raise RuntimeError(f"Faithfulness summary {metric_name} does not have n=256: {summary.get(metric_name)}")

        # Immutable aliases at reports root are intentionally different from the legacy
        # generic filenames that SHAP/LIME maintenance runs may touch.
        immutable_per = reports / "xai_faithfulness_v18_dual_cnn_full256_per_sample.csv"
        immutable_sum = reports / "xai_faithfulness_v18_dual_cnn_full256_summary.json"
        shutil.copy2(per_path, immutable_per)
        shutil.copy2(sum_path, immutable_sum)
        manifest = {
            "schema": "aura_cxr_faithfulness_v26_immutable_full256",
            "xai_schema_version": XAI_SCHEMA_VERSION,
            "status": "VERIFIED",
            "n": 256,
            "seed": 42,
            "top_fraction": float(args.faithfulness_top_fraction),
            "probability_target": "validation-selected hybrid CNN-ViT probability fusion using the same inter-model weights as final CAM",
            "selected_dl_models": selected_dl_models,
            "selected_model_weights": selected_weights,
            "selected_models": cam_layer_manifest.get("selected_models", {}),
            "selected_map_policy": cam_layer_manifest.get("selected_map_policy"),
            "selected_threshold": cam_layer_manifest.get("selected_threshold"),
            "fusion_constraint": cam_layer_manifest.get("fusion_constraint"),
            "selection_split": cam_layer_manifest.get("selection_split"),
            "test_set_used_for_selection": bool(cam_layer_manifest.get("test_set_used_for_selection")),
            "selection_lock_id": pair_manifest.get("selection_lock_id"),
            "deployment_lock_id": deployment.get("deployment_lock_id"),
            "model_checkpoint_sha256": _XAI_MODEL_HASHES,
            "source_localization_metrics": str(source_metrics),
            "source_localization_metrics_sha256": sha256_file(source_metrics),
            "per_sample_path": str(immutable_per),
            "per_sample_sha256": sha256_file(immutable_per),
            "summary_path": str(immutable_sum),
            "summary_sha256": sha256_file(immutable_sum),
            "summary": summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "editorial_marker_can_be_removed": True,
        }
        manifest_path = reports / "xai_faithfulness_v18_dual_cnn_full256_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8")
        (artifact_dir / "FAITHFULNESS_VERIFIED_OK.txt").write_text(
            "VERIFIED: final V18 strict hybrid CNN-ViT faithfulness completed on n=256.\n"
            f"Deletion confidence drop mean={summary['deletion_confidence_drop']['mean']:.10f}\n"
            f"Retention probability ratio mean={summary['retention_probability_ratio']['mean']:.10f}\n"
            "The thesis editorial REQUIRES_RERUN marker may now be removed after running the verifier.\n",
            encoding="utf-8",
        )
        print("[FAITHFULNESS-ONLY][VERIFIED] n=256")
        print(json.dumps(summary, indent=2))
        print(f"[FAITHFULNESS-ONLY] {immutable_sum}")
        print(f"[FAITHFULNESS-ONLY] {immutable_per}")
        print(f"[FAITHFULNESS-ONLY] {manifest_path}")
        return

    print("\n[1/6] ROC + PR curves...")
    curve_dict = {radiomics_label: (y_test, radiomics_test_proba[:, 1])}
    for name, _, _ in cnn_specs:
        label = DISPLAY_NAMES.get(name, name) + " (MC-Dropout)"
        curve_dict[label] = (y_test, cnn_models[name]["test_mean"][:, 1])
    curve_dict["Soft Voting"] = (y_test, soft_test_proba[:, 1])
    curve_dict["Stacked Ensemble"] = (y_test, stacked_test_proba[:, 1])
    save_roc_pr_curves(curve_dict, xai_plots / "roc_pr_curves.png", paper_dpi=args.paper_dpi)

    print(f"\n[2/6] Qualitative Contrastive GradCAM ({args.n_xai} samples)...")
    qualitative_metrics_df = pd.DataFrame()
    try:
        xai_imgs, xai_preds, xai_trues, xai_probas = [], [], [], []
        xai_pids, xai_paths, xai_gt_boxes, xai_indices = [], [], [], []
        cat_need = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}
        collected = {k: 0 for k in cat_need}
        per_cat = max(1, args.n_xai // 4)
        used_indices = set()

        for idx in range(len(test_df)):
            if len(xai_imgs) >= args.n_xai:
                break
            t = int(y_test[idx])
            p = int(stacked_test_pred[idx])
            cat = next((k for k, (tt, pp) in cat_need.items() if tt == t and pp == p and collected[k] < per_cat), None)
            if cat is None:
                continue
            path = test_df.iloc[idx]["image_path"]
            if not Path(path).exists():
                continue
            pid = str(test_df.iloc[idx]["patientId"])
            xai_imgs.append(load_dicom_grayscale(path, args.image_size))
            xai_preds.append(p)
            xai_trues.append(t)
            xai_probas.append(float(stacked_test_proba[idx, 1]))
            xai_pids.append(pid)
            xai_paths.append(path)
            xai_gt_boxes.append(scale_boxes_to_image(pid, path, gt_box_map, args.image_size))
            xai_indices.append(idx)
            collected[cat] += 1
            used_indices.add(idx)

        for idx in range(len(test_df)):
            if len(xai_imgs) >= args.n_xai:
                break
            if idx in used_indices:
                continue
            path = test_df.iloc[idx]["image_path"]
            if not Path(path).exists():
                continue
            pid = str(test_df.iloc[idx]["patientId"])
            xai_imgs.append(load_dicom_grayscale(path, args.image_size))
            xai_preds.append(int(stacked_test_pred[idx]))
            xai_trues.append(int(y_test[idx]))
            xai_probas.append(float(stacked_test_proba[idx, 1]))
            xai_pids.append(pid)
            xai_paths.append(path)
            xai_gt_boxes.append(scale_boxes_to_image(pid, path, gt_box_map, args.image_size))
            xai_indices.append(idx)

        print(f"  Samples: {len(xai_imgs)} | TP={collected['TP']} TN={collected['TN']} FP={collected['FP']} FN={collected['FN']}")
        qualitative_metrics_df = save_contrastive_gradcam_grid(
            primary_model,
            primary_preprocess,
            xai_imgs,
            xai_preds,
            xai_trues,
            candidate_layers,
            xai_output_dir / "n4_contrastive_gradcam_paper.png",
            image_size=args.image_size,
            proba_list=xai_probas,
            patient_ids=xai_pids,
            image_paths=xai_paths,
            gt_boxes_list=xai_gt_boxes,
            heatmap_threshold_pct=args.heatmap_threshold_pct,
            paper_dpi=args.paper_dpi,
            save_pdf=args.save_pdf,
            debug=args.debug_gradcam,
            sample_indices=xai_indices,
            models_pp_layers=models_pp_layers,
            lung_mask_dir=lung_mask_dir,
            use_lung_mask=args.use_lung_mask,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            localization_cam_method=args.localization_cam_method,
            cam_min_grid=args.cam_min_grid,
            cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
        )
        qualitative_metrics_df.to_csv(xai_reports / "xai_qualitative_grid_metrics.csv", index=False)

        # Prespecified RSNA figure case selection uses patient IDs and GT boxes ONLY.
        # No predicted class, probability or test-CAM quality enters case selection.
        import hashlib
        four_stage_n = max(1, int(getattr(args, "rsna_four_stage_n", 4)))
        four_stage_indices = [
            i for i in range(len(test_df))
            if int(y_test[i]) == 1
            and len(scale_boxes_to_image(
                str(test_df.iloc[i]["patientId"]),
                str(test_df.iloc[i]["image_path"]), gt_box_map, args.image_size
            )) > 0
        ]
        four_stage_indices.sort(key=lambda i: hashlib.sha256(
            ("rsna-xai-42:" + str(test_df.iloc[i]["patientId"])).encode()).hexdigest()
        )
        four_stage_metrics_df = save_dual_cnn_four_stage_grid(
            test_df=test_df,
            y_true=y_test,
            y_pred=stacked_test_pred,
            probas=stacked_test_proba[:, 1],
            indices=four_stage_indices[:four_stage_n],
            models_pp_layers=models_pp_layers,
            gt_box_map=gt_box_map,
            path=xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused.png",
            image_size=args.image_size,
            lung_mask_dir=lung_mask_dir,
            use_lung_mask=args.use_lung_mask,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            heatmap_threshold_pct=args.heatmap_threshold_pct,
            localization_cam_method=args.localization_cam_method,
            cam_min_grid=args.cam_min_grid,
            cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
            paper_dpi=args.paper_dpi,
            save_pdf=args.save_pdf,
            dataset_name="RSNA locked test",
            show_gt_boxes=True,
        )
        four_stage_metrics_df.to_csv(
            xai_reports / "xai_dual_cnn_four_stage_manifest.csv", index=False
        )
        four_stage_metrics_df.to_csv(
            xai_reports / "xai_dual_cnn_four_stage_manifest_rsna.csv", index=False
        )
        four_stage_metrics_df.to_csv(
            xai_reports / "xai_hybrid_cnn_vit_four_stage_manifest_rsna.csv", index=False
        )
        # Backward-compatible legacy file plus an explicit RSNA alias.
        rsna_alias = xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused_rsna.png"
        try:
            shutil.copy2(xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused.png", rsna_alias)
            shutil.copy2(rsna_alias, xai_output_dir / "hybrid_cnn_vit_four_panel_rsna.png")
            if args.save_pdf:
                pdf_src = (xai_output_dir / "n4_dual_cnn_before_xrv_eva_fused.png").with_suffix(".pdf")
                if pdf_src.exists():
                    shutil.copy2(pdf_src, rsna_alias.with_suffix(".pdf"))
        except Exception:
            pass
        maybe_generate_kermany_four_stage_artifacts(
            args=args,
            xai_output_dir=xai_output_dir,
            xai_reports=xai_reports,
            deployment=deployment,
            radiomics_model_path=radiomics_model_path,
            selected_dl_models=selected_dl_models,
            cnn_models=cnn_models,
            meta_clf=meta_clf,
            models_pp_layers=models_pp_layers,
        )
    except Exception as e:
        print(f"[WARN] Qualitative GradCAM failed: {e}")
        import traceback
        traceback.print_exc()

    if (not args.no_cam_multipage_pdf) and int(args.cam_pdf_n) > 0 and str(args.cam_pdf_mode).lower() == "balanced":
        print(f"\n[2b/6] Multipage fused hybrid CNN-ViT CAM PDF ({int(args.cam_pdf_n)} samples; balanced representative mode)...")
        try:
            cam_pdf_indices = _select_balanced_cam_pdf_indices(
                test_df, y_test, stacked_test_pred, int(args.cam_pdf_n)
            )
            cam_pdf_metrics_df = save_fused_cam_multipage_pdf(
                test_df=test_df,
                y_true=y_test,
                y_pred=stacked_test_pred,
                probas=stacked_test_proba[:, 1],
                indices=cam_pdf_indices,
                models_pp_layers=models_pp_layers,
                gt_box_map=gt_box_map,
                output_pdf=xai_plots / str(args.cam_pdf_filename),
                image_size=args.image_size,
                lung_mask_dir=lung_mask_dir,
                use_lung_mask=args.use_lung_mask,
                allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
                heatmap_threshold_pct=args.heatmap_threshold_pct,
                localization_cam_method=args.localization_cam_method,
                cam_min_grid=args.cam_min_grid,
                cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
                cases_per_page=args.cam_pdf_per_page,
            )
            cam_pdf_metrics_df.to_csv(
                xai_reports / "xai_dual_cnn_cam_pdf_100_samples_manifest.csv", index=False
            )
        except Exception as exc:
            print(f"[CAM-PDF][WARN] Multipage CAM PDF export failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n[3/6] Quantitative XAI localization metrics...")
    xai_metrics_df = pd.DataFrame()
    xai_summary_df = pd.DataFrame()
    agg = None
    try:
        # Validation-selected weighted multi-layer CAM from the available Keras CNNs.
        # Exact layers are locked using validation GT only.
        # Lung gating is active only when --use_lung_mask is explicit.
        lung_gate_on = bool(args.use_lung_mask and not args.no_lung_gate)
        allow_ellipse_on = bool(args.allow_ellipse_fallback and not args.no_ellipse_fallback)
        xai_metrics_df, xai_summary_df, agg = evaluate_xai_localization_dataset(
            primary_model,
            primary_preprocess,
            test_df,
            y_test,
            stacked_test_pred,
            stacked_test_proba,
            candidate_layers,
            gt_box_map,
            image_size=args.image_size,
            max_samples=args.xai_eval_max,
            heatmap_threshold_pct=args.heatmap_threshold_pct,
            reports_dir=xai_reports,
            use_lung_mask=args.use_lung_mask,
            lung_mask_dir=lung_mask_dir,
            resume=not args.no_resume_xai,
            force=False,
            save_every=args.xai_save_every,
            models_pp_layers=models_pp_layers,
            lung_gate=lung_gate_on,
            allow_ellipse=allow_ellipse_on,
            localization_cam_method=args.localization_cam_method,
            cam_min_grid=args.cam_min_grid,
            cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
            n_boot=args.xai_bootstrap,
            seed=42,
            max_invalid_rate=args.max_invalid_heatmap_rate,
            max_gated_outside_lung_ratio=args.max_gated_outside_lung_ratio,
            strict_qc=bool(run_mode == "FINAL" and args.fail_on_technical_qc and not args.allow_xai_qc_fail),
            run_mode=run_mode,
            expected_gt_positive=expected_gt_positive,
            min_final_coverage=args.final_min_coverage,
            min_final_valid_cases=args.final_min_valid_cases,
            min_effective_lung_mask_coverage=args.min_effective_lung_mask_coverage,
            max_lung_mask_fallback_rate=args.max_lung_mask_fallback_rate,
            cam_config_manifest=cam_layer_manifest,
        )
        if len(xai_summary_df):
            print("\n[XAI Summary]")
            print(xai_summary_df.to_string(index=False))
        if not args.no_baselines and xai_metrics_df is not None and len(xai_metrics_df):
            base_df = compute_chance_baselines(
                xai_metrics_df, test_df, gt_box_map, image_size=args.image_size,
                use_lung_mask=args.use_lung_mask, lung_mask_dir=lung_mask_dir,
                threshold_pct=args.heatmap_threshold_pct, reports_dir=xai_reports,
                n_boot=args.xai_bootstrap, seed=42,
                random_repeats=args.random_baseline_repeats,
                center_sigma_frac=args.center_baseline_sigma_frac,
                lesion_prevalence_map=lesion_prior,
                map_policy=cam_layer_manifest.get("selected_map_policy", "raw"),
                soft_lung_outside_weight=args.soft_lung_outside_weight,
            )
            if len(base_df):
                print("\n[XAI Chance/Lung-Prior Baselines vs Model]")
                print(base_df.to_string(index=False))
                updated_qc = update_qc_with_baseline_results(
                    xai_reports, mode=args.baseline_qc_mode
                )
                if updated_qc is not None:
                    print("[XAI BASELINE QC]", json.dumps(updated_qc, indent=2, default=str))
                    if not updated_qc.get("baseline_superiority_qc_pass", False):
                        print("[WARN] Baseline superiority did not pass. Reports will complete, but validated localization claims are disabled.")
        save_tp_fn_localization_statistics(
            xai_metrics_df, xai_reports, n_boot=args.xai_bootstrap, seed=42
        )
        evaluate_xai_faithfulness(
            xai_metrics_df, test_df, models_pp_layers, args.image_size, xai_reports,
            max_samples=args.faithfulness_eval_max,
            top_fraction=args.faithfulness_top_fraction,
            cam_smooth_sigma_frac=args.cam_smooth_sigma_frac, seed=42,
            use_lung_mask=args.use_lung_mask, lung_mask_dir=lung_mask_dir,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
        )
        run_classifier_head_randomization_sanity(
            models_pp_layers, val_df, y_val, args.image_size, xai_reports,
            n_samples=args.xai_sanity_samples, seed=42,
        )
    except Exception as e:
        print(f"[ERROR] Quantitative XAI evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        if run_mode == "FINAL" and args.fail_on_technical_qc and not args.allow_xai_qc_fail:
            raise
        print("[WARN] XAI diagnostic/QC issue preserved in logs; continuing so all scientific reports are generated.")

    final_qc = update_qc_with_final_guardrails(
        xai_reports,
        run_mode=run_mode,
        expected_gt_positive=expected_gt_positive,
        min_final_coverage=args.final_min_coverage,
        min_final_valid_cases=args.final_min_valid_cases,
        layer_manifest_path=reports / "xai_cam_configuration_validation.json",
        require_baseline=bool(run_mode == "FINAL"),
        allow_baseline_report_only=args.allow_final_baseline_report_only,
    )
    table_info = build_xai_q1_results_table(
        xai_reports,
        run_mode=run_mode,
        expected_gt_positive=expected_gt_positive,
        layer_manifest_path=reports / "xai_cam_configuration_validation.json",
        probability_source_path=xai_reports / "xai_probability_source.json",
    )
    if final_qc is not None:
        print("[XAI FINAL GUARDRAILS]", json.dumps(final_qc, indent=2, default=str))
        if run_mode == "FINAL" and not final_qc.get("final_q1_ready", False):
            print("[WARN] FINAL analysis completed, but validated localization-claim Q1 guardrails did not pass.")
            if args.fail_on_localization_claim_qc:
                raise RuntimeError("FINAL XAI localization-claim guardrails failed; reports were preserved.")

    if (not args.no_cam_multipage_pdf) and int(args.cam_pdf_n) > 0 and xai_metrics_df is not None and len(xai_metrics_df):
        cam_pdf_mode = str(args.cam_pdf_mode).lower()
        if cam_pdf_mode in {"top_final_locked", "both", "oracle_best_per_case"}:
            print(f"\n[3b/6] Ranked CAM PDF export mode={cam_pdf_mode} ({int(args.cam_pdf_n)} samples)...")
        try:
            if cam_pdf_mode in {"top_final_locked", "both"}:
                final_locked_indices = _select_top_cam_pdf_indices_from_metrics(
                    xai_metrics_df, int(args.cam_pdf_n), require_valid=True
                )
                final_locked_manifest_df = save_fused_cam_multipage_pdf(
                    test_df=test_df,
                    y_true=y_test,
                    y_pred=stacked_test_pred,
                    probas=stacked_test_proba[:, 1],
                    indices=final_locked_indices,
                    models_pp_layers=models_pp_layers,
                    gt_box_map=gt_box_map,
                    output_pdf=xai_plots / str(args.cam_pdf_final_locked_filename),
                    image_size=args.image_size,
                    lung_mask_dir=lung_mask_dir,
                    use_lung_mask=args.use_lung_mask,
                    allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
                    heatmap_threshold_pct=args.heatmap_threshold_pct,
                    localization_cam_method=args.localization_cam_method,
                    cam_min_grid=args.cam_min_grid,
                    cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
                    cases_per_page=args.cam_pdf_per_page,
                )
                if len(final_locked_manifest_df):
                    ranked_map = _rank_final_locked_cam_metrics(xai_metrics_df, require_valid=True)[["sample_index", "cam_quality_score"]].copy()
                    final_locked_manifest_df = final_locked_manifest_df.merge(ranked_map, on="sample_index", how="left")
                    final_locked_manifest_df.to_csv(
                        xai_reports / "xai_dual_cnn_final_locked_top100_manifest.csv", index=False
                    )
            if cam_pdf_mode in {"oracle_best_per_case", "both"}:
                oracle_pool_df = _rank_final_locked_cam_metrics(xai_metrics_df, require_valid=True)
                oracle_indices = pd.to_numeric(oracle_pool_df.get("sample_index"), errors="coerce").dropna().astype(int).tolist()
                oracle_manifest_df = save_oracle_best_cam_multipage_pdf(
                    test_df=test_df,
                    y_true=y_test,
                    y_pred=stacked_test_pred,
                    probas=stacked_test_proba[:, 1],
                    indices=oracle_indices[: int(args.cam_pdf_n)],
                    models_pp_layers=models_pp_layers,
                    gt_box_map=gt_box_map,
                    output_pdf=xai_plots / str(args.cam_pdf_oracle_filename),
                    reports_dir=xai_reports,
                    image_size=args.image_size,
                    lung_mask_dir=lung_mask_dir,
                    use_lung_mask=args.use_lung_mask,
                    allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
                    localization_cam_method=args.localization_cam_method,
                    cam_min_grid=args.cam_min_grid,
                    cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
                    cases_per_page=args.cam_pdf_per_page,
                    topk_model_candidates=args.cam_pdf_oracle_topk_model_candidates,
                    topk_fusion_candidates=args.cam_pdf_oracle_topk_fusion_candidates,
                )
                if len(oracle_manifest_df):
                    oracle_manifest_df.to_csv(
                        xai_reports / "xai_dual_cnn_oracle_best_per_case_top100_manifest.csv", index=False
                    )
        except Exception as exc:
            print(f"[CAM-PDF][WARN] Ranked/oracle CAM PDF export failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n[4/6] Aggregate heatmaps and failure analysis...")
    try:
        if agg is not None:
            save_xai_aggregate_heatmaps(
                agg,
                xai_output_dir / "n4_aggregate_heatmaps.png",
                paper_dpi=args.paper_dpi,
                save_pdf=args.save_pdf,
                run_label="QUICK PRECHECK — NOT FINAL" if run_mode == "QUICK" else "FINAL FULL TEST",
            )
            save_xai_aggregate_prior_audit(
                agg,
                xai_output_dir / "n4_aggregate_prior_audit.png",
                paper_dpi=args.paper_dpi,
                save_pdf=args.save_pdf,
                run_label="QUICK PRECHECK — NOT FINAL" if run_mode == "QUICK" else "FINAL FULL TEST",
            )
        if xai_metrics_df is not None and len(xai_metrics_df):
            save_failure_analysis_grid(
                primary_model,
                primary_preprocess,
                test_df,
                xai_metrics_df,
                candidate_layers,
                gt_box_map,
                xai_output_dir / "n4_failure_analysis.png",
                image_size=args.image_size,
                heatmap_threshold_pct=args.heatmap_threshold_pct,
                loc_score_threshold=args.loc_score_threshold,
                n_failure=args.n_failure,
                paper_dpi=args.paper_dpi,
                save_pdf=args.save_pdf,
                models_pp_layers=models_pp_layers,
                lung_mask_dir=lung_mask_dir,
                use_lung_mask=args.use_lung_mask,
                allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
                localization_cam_method=args.localization_cam_method,
                cam_min_grid=args.cam_min_grid,
                cam_smooth_sigma_frac=args.cam_smooth_sigma_frac,
                run_label="QUICK PRECHECK — NOT FINAL" if run_mode == "QUICK" else "FINAL FULL TEST",
            )
    except Exception as e:
        print(f"[WARN] Aggregate/failure analysis failed: {e}")
        import traceback
        traceback.print_exc()

    if not args.no_comparison:
        print("\n[4b/6] Curated class-comparison figures for the paper...")
        try:
            detailed_csv = args.detailed_class_info
            if not detailed_csv and args.labels_csv:
                guess = Path(args.labels_csv).with_name("stage_2_detailed_class_info.csv")
                detailed_csv = str(guess) if guess.exists() else ""
            make_class_comparison_figures(
                primary_model, primary_preprocess, test_df, y_test,
                stacked_test_pred, stacked_test_proba, gt_box_map,
                out_dir=xai_output_dir, image_size=args.image_size,
                use_lung_mask=args.use_lung_mask, lung_mask_dir=lung_mask_dir,
                heatmap_threshold_pct=args.heatmap_threshold_pct,
                n_per_class=args.comparison_n_per_class,
                paper_dpi=args.paper_dpi, save_pdf=args.save_pdf,
                xai_metrics_df=xai_metrics_df, detailed_class_info_csv=detailed_csv,
                allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
                cam_min_grid=args.cam_min_grid,
                localization_cam_method=args.localization_cam_method,
                models_pp_layers=models_pp_layers,
            )
        except Exception as e:
            print(f"[WARN] Class-comparison figures failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n[5/6] Optional SHAP and LIME explainers...")
    try:
        if xai_metrics_df is None or len(xai_metrics_df) == 0:
            # Fallback to the qualitative selection when full metrics fail.
            xai_metrics_df = qualitative_metrics_df.rename(columns={"row": "sample_index"})
            if "sample_index" not in xai_metrics_df.columns:
                xai_metrics_df["sample_index"] = list(range(min(len(test_df), len(xai_metrics_df))))
        save_lime_explanations(
            primary_model,
            primary_preprocess,
            test_df,
            xai_metrics_df,
            gt_box_map,
            xai_output_dir / "n4_lime_explanations.png",
            image_size=args.image_size,
            n_samples=args.n_lime,
            lime_num_samples=args.lime_num_samples,
            lime_segments=args.lime_segments,
            paper_dpi=args.paper_dpi,
            save_pdf=args.save_pdf,
            lung_mask_dir=lung_mask_dir,
            use_lung_mask=args.use_lung_mask,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            repeats=args.lime_repeats,
            reports_dir=xai_reports,
        )
        save_shap_explanations(
            primary_model,
            primary_preprocess,
            test_df,
            xai_metrics_df,
            gt_box_map,
            xai_output_dir / "n4_shap_explanations.png",
            image_size=args.image_size,
            n_samples=args.n_shap,
            background_n=args.shap_background,
            paper_dpi=args.paper_dpi,
            save_pdf=args.save_pdf,
            background_df=train_df,
            lung_mask_dir=lung_mask_dir,
            use_lung_mask=args.use_lung_mask,
            allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            reports_dir=xai_reports,
        )
        if not args.no_best_lime_shap_pdfs:
            print(f"[XAI-PDF] Building exploratory best-LIME PDF: top {int(args.best_lime_pdf_n)} from candidate pool {int(args.best_lime_candidate_pool)}")
            save_best_lime_multipage_pdf(
                primary_model, primary_preprocess, test_df, xai_metrics_df, gt_box_map,
                xai_plots / str(args.best_lime_pdf_filename),
                reports_dir=xai_reports, image_size=args.image_size,
                n_output=args.best_lime_pdf_n, candidate_pool=args.best_lime_candidate_pool,
                cases_per_page=args.best_lime_pdf_per_page,
                lime_num_samples=args.lime_num_samples, lime_segments=args.lime_segments,
                repeats=args.lime_repeats, lung_mask_dir=lung_mask_dir,
                use_lung_mask=args.use_lung_mask,
                allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            )
            print(f"[XAI-PDF] Building exploratory best-SHAP PDF: top {int(args.best_shap_pdf_n)} from candidate pool {int(args.best_shap_candidate_pool)}")
            save_best_shap_multipage_pdf(
                primary_model, primary_preprocess, test_df, xai_metrics_df, gt_box_map,
                xai_plots / str(args.best_shap_pdf_filename),
                reports_dir=xai_reports, image_size=args.image_size,
                n_output=args.best_shap_pdf_n, candidate_pool=args.best_shap_candidate_pool,
                cases_per_page=args.best_shap_pdf_per_page,
                background_n=args.shap_background, background_df=train_df,
                lung_mask_dir=lung_mask_dir, use_lung_mask=args.use_lung_mask,
                allow_ellipse=args.allow_ellipse_fallback and not args.no_ellipse_fallback,
            )
    except Exception as e:
        print(f"[WARN] Optional explainers failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[6/6] test_predictions_full.csv...")
    pred_df = test_df[["sample_id", "patientId", "label"]].copy().reset_index(drop=True)
    pred_df["decision_threshold"] = float(pair_manifest["threshold"])
    pred_df["radiomics_prob"] = radiomics_test_proba[:, 1]
    pred_df["radiomics_pred"] = radiomics_test_pred
    for name, _, _ in cnn_specs:
        prefix = "effnetv2" if name == "efficientnetv2" else name
        info = cnn_models[name]
        pred_df[f"{prefix}_prob"] = info["test_mean"][:, 1]
        pred_df[f"{prefix}_std"] = info["test_std"][:, 1]
        pred_df[f"{prefix}_predictive_interval_width_95"] = info["test_ci"]
        pred_df[f"{prefix}_ci95"] = info["test_ci"]  # legacy alias
        pred_df[f"{prefix}_pred"] = info["test_pred"]
    primary = cnn_models[primary_name]
    pred_df["cnn_mean_prob"] = primary["test_mean"][:, 1]
    pred_df["cnn_pred"] = primary["test_pred"]
    pred_df["soft_prob"] = soft_test_proba[:, 1]
    pred_df["soft_pred"] = soft_test_pred
    pred_df["stacked_prob"] = stacked_test_proba[:, 1]
    pred_df["stacked_pred"] = stacked_test_pred
    pred_df["category"] = [_get_category(int(t), int(p)) for t, p in zip(y_test, stacked_test_pred)]

    if xai_metrics_df is not None and len(xai_metrics_df):
        merge_cols = [
            "patientId",
            "pointing_hit",
            "iou_at_thr",
            "localization_score",
            "energy_inside_gt",
            "peak_distance_norm",
            "pred_area_ratio",
            "heatmap_valid",
            "outside_lung_ratio_raw",
            "gt_coverage_at_thr",
            "activation_precision_at_thr",
            "failure_reason",
        ]
        available = [c for c in merge_cols if c in xai_metrics_df.columns]
        xai_for_merge = xai_metrics_df[available].drop_duplicates("patientId")
        rename_map = {c: f"xai_{c}" for c in available if c != "patientId"}
        pred_df = pred_df.merge(xai_for_merge.rename(columns=rename_map), on="patientId", how="left")

    # Risk-score columns intentionally removed; proposal keeps bbox only for XAI ground truth.

    pred_df.to_csv(xai_reports / "test_predictions_full.csv", index=False)
    print(f"[OK] CSV saved ({len(pred_df)} rows): {xai_reports / 'test_predictions_full.csv'}")

    print("\n" + "=" * 60 + "\nFINAL TEST SET RESULTS\n" + "=" * 60)
    summary_items = [(radiomics_label, radiomics_test_pred, radiomics_test_proba)]
    for name, _, _ in cnn_specs:
        label = DISPLAY_NAMES.get(name, name)
        summary_items.append((label, cnn_models[name]["test_pred"], cnn_models[name]["test_mean"]))
    summary_items.extend([
        ("Soft Voting", soft_test_pred, soft_test_proba),
        ("Stacked Ensemble", stacked_test_pred, stacked_test_proba),
    ])
    for name, preds, probas in summary_items:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        au = roc_auc_score(y_test, probas[:, 1]) if len(np.unique(y_test)) > 1 else 0.0
        sen, spe = sens_spec(y_test, preds)
        print(f"  {name:28s} | Acc={acc:.4f} | F1={f1:.4f} | AUC={au:.4f} | Sens={sen:.4f} | Spec={spe:.4f}")

    print(f"\n[XAI RUN COMPLETE] mode={run_mode} | output_dir={xai_output_dir} | reports_dir={xai_reports}")
    if run_mode == "QUICK":
        print("[IMPORTANT] QUICK PRECHECK outputs are diagnostic only and MUST NOT be cited as final paper results.")
    else:
        qc_final_path = xai_reports / "xai_quality_control.json"
        if qc_final_path.exists():
            qc_final = json.loads(qc_final_path.read_text(encoding="utf-8"))
            print(f"[FINAL Q1 VERDICT] final_q1_ready={qc_final.get('final_q1_ready', False)}")
    print(
        "\n[OK] Done. Key outputs:\n"
        f"  {xai_plots / 'roc_pr_curves.png'}\n"
        f"  {xai_output_dir / 'n4_contrastive_gradcam_paper.png'}\n"
        f"  {xai_output_dir / 'n4_dual_cnn_before_xrv_eva_fused.png'}\n"
        f"  {xai_reports / 'xai_localization_per_sample.csv'}\n"
        f"  {xai_reports / 'xai_localization_summary.csv'}\n"
        f"  {xai_reports / 'xai_quality_control.json'}\n"
        f"  {xai_output_dir / 'n4_aggregate_heatmaps.png'}\n"
        f"  {xai_output_dir / 'n4_aggregate_prior_audit.png'}\n"
        f"  {reports / 'xai_cam_configuration_validation.json'}\n"
        f"  {xai_reports / 'xai_localization_baseline_comparison.csv'}\n"
        f"  {xai_output_dir / 'n4_failure_analysis.png'}\n"
        f"  {xai_output_dir / 'n4_lime_explanations.png'}\n"
        f"  {xai_output_dir / 'n4_shap_explanations.png'}\n"
        f"  {xai_reports / 'test_predictions_full.csv'}"
    )


# =========================
# Pagination wrappers for larger paper sample sets
# =========================
_ORIG_SAVE_CONTRASTIVE_GRADCAM_GRID = save_contrastive_gradcam_grid
_ORIG_SAVE_FAILURE_ANALYSIS_GRID = save_failure_analysis_grid
_ORIG_SAVE_LIME_EXPLANATIONS = save_lime_explanations
_ORIG_SAVE_SHAP_EXPLANATIONS = save_shap_explanations


def _xai_page_size_for_path(path):
    name = str(Path(path).name).lower()
    if 'contrastive_gradcam' in name:
        return 10
    if 'failure_analysis' in name:
        return 10
    if 'lime' in name or 'shap' in name:
        return 10
    return 10


def _xai_num_parts(total_n, page_size):
    total_n = int(total_n)
    page_size = max(1, int(page_size))
    return (total_n + page_size - 1) // page_size


def _xai_part_path(path, part_idx, total_parts):
    p = Path(path)
    if int(total_parts) <= 1:
        return p
    return p.with_name(f"{p.stem}_part{int(part_idx):02d}{p.suffix}")


def _xai_slice_optional(obj, start, end):
    if obj is None:
        return None
    try:
        return obj[start:end]
    except Exception:
        return [obj[i] for i in range(start, end)]


def _xai_ordered_metrics_subset(metrics_df, selected_indices):
    if metrics_df is None or len(metrics_df) == 0:
        return metrics_df
    idxs = [int(i) for i in selected_indices]
    order_map = {int(v): i for i, v in enumerate(idxs)}
    out = metrics_df[metrics_df['sample_index'].astype(int).isin(idxs)].copy()
    if len(out) == 0:
        return out
    out['__order'] = out['sample_index'].astype(int).map(order_map)
    out = out.sort_values('__order').drop(columns='__order')
    return out


def _xai_write_paginated_manifest(base_path, total_items, total_parts, page_size, kind, page_files):
    base_path = Path(base_path)
    manifest_path = base_path.with_name(f"{base_path.stem}_pagination_manifest.json")
    payload = {
        'kind': kind,
        'base_file': str(base_path),
        'total_items': int(total_items),
        'page_size': int(page_size),
        'total_parts': int(total_parts),
        'page_files': [str(Path(p)) for p in page_files],
    }
    try:
        manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"[OK] Pagination manifest saved: {manifest_path}")
    except Exception as exc:
        print(f"[WARN] Could not save pagination manifest for {base_path.name}: {exc}")


def save_contrastive_gradcam_grid(
    model,
    preprocess_fn,
    sample_images_raw,
    sample_preds,
    sample_trues,
    candidate_layers,
    path,
    image_size=224,
    proba_list=None,
    patient_ids=None,
    image_paths=None,
    gt_boxes_list=None,
    heatmap_threshold_pct=0.60,
    paper_dpi=600,
    save_pdf=False,
    debug=False,
    sample_indices=None,
    models_pp_layers=None,
    lung_mask_dir=None,
    use_lung_mask=False,
    allow_ellipse=False,
    localization_cam_method="GradCAM",
    cam_min_grid=16,
    cam_smooth_sigma_frac=0.005,
):
    total_n = len(sample_images_raw)
    page_size = _xai_page_size_for_path(path)
    total_parts = _xai_num_parts(total_n, page_size)
    if total_parts <= 1:
        return _ORIG_SAVE_CONTRASTIVE_GRADCAM_GRID(
            model, preprocess_fn, sample_images_raw, sample_preds, sample_trues,
            candidate_layers, path, image_size=image_size, proba_list=proba_list,
            patient_ids=patient_ids, image_paths=image_paths, gt_boxes_list=gt_boxes_list,
            heatmap_threshold_pct=heatmap_threshold_pct, paper_dpi=paper_dpi,
            save_pdf=save_pdf, debug=debug, sample_indices=sample_indices,
            models_pp_layers=models_pp_layers, lung_mask_dir=lung_mask_dir,
            use_lung_mask=use_lung_mask, allow_ellipse=allow_ellipse,
            localization_cam_method=localization_cam_method, cam_min_grid=cam_min_grid,
            cam_smooth_sigma_frac=cam_smooth_sigma_frac,
        )
    all_df = []
    page_files = []
    for part_idx in range(total_parts):
        start = part_idx * page_size
        end = min(total_n, (part_idx + 1) * page_size)
        part_path = _xai_part_path(path, part_idx + 1, total_parts)
        page_files.append(part_path)
        df = _ORIG_SAVE_CONTRASTIVE_GRADCAM_GRID(
            model, preprocess_fn,
            _xai_slice_optional(sample_images_raw, start, end),
            _xai_slice_optional(sample_preds, start, end),
            _xai_slice_optional(sample_trues, start, end),
            candidate_layers, part_path, image_size=image_size,
            proba_list=_xai_slice_optional(proba_list, start, end),
            patient_ids=_xai_slice_optional(patient_ids, start, end),
            image_paths=_xai_slice_optional(image_paths, start, end),
            gt_boxes_list=_xai_slice_optional(gt_boxes_list, start, end),
            heatmap_threshold_pct=heatmap_threshold_pct, paper_dpi=paper_dpi,
            save_pdf=save_pdf, debug=debug,
            sample_indices=_xai_slice_optional(sample_indices, start, end),
            models_pp_layers=models_pp_layers, lung_mask_dir=lung_mask_dir,
            use_lung_mask=use_lung_mask, allow_ellipse=allow_ellipse,
            localization_cam_method=localization_cam_method, cam_min_grid=cam_min_grid,
            cam_smooth_sigma_frac=cam_smooth_sigma_frac,
        )
        if isinstance(df, pd.DataFrame) and len(df):
            df = df.copy()
            df['page'] = int(part_idx + 1)
            all_df.append(df)
    _xai_write_paginated_manifest(path, total_n, total_parts, page_size, 'qualitative_cam_grid', page_files)
    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        csv_path = Path(path).with_name(f"{Path(path).stem}_all_pages_metrics.csv")
        try:
            combined.to_csv(csv_path, index=False)
            print(f"[OK] Combined qualitative metrics saved: {csv_path}")
        except Exception as exc:
            print(f"[WARN] Could not save combined qualitative metrics: {exc}")
        return combined
    return pd.DataFrame()


def save_failure_analysis_grid(
    model, preprocess_fn, test_df, metrics_df, candidate_layers, gt_box_map,
    path, image_size=224, heatmap_threshold_pct=0.60,
    loc_score_threshold=0.10, n_failure=6, paper_dpi=600, save_pdf=False,
    models_pp_layers=None, lung_mask_dir=None, use_lung_mask=False,
    allow_ellipse=False, localization_cam_method="GradCAM", cam_min_grid=16,
    cam_smooth_sigma_frac=0.005, run_label="",
):
    page_size = _xai_page_size_for_path(path)
    cases = select_failure_cases(metrics_df, n_failure=n_failure, loc_score_threshold=loc_score_threshold)
    total_n = len(cases)
    total_parts = _xai_num_parts(total_n, page_size)
    if total_parts <= 1:
        return _ORIG_SAVE_FAILURE_ANALYSIS_GRID(
            model, preprocess_fn, test_df, metrics_df, candidate_layers, gt_box_map,
            path, image_size=image_size, heatmap_threshold_pct=heatmap_threshold_pct,
            loc_score_threshold=loc_score_threshold, n_failure=n_failure,
            paper_dpi=paper_dpi, save_pdf=save_pdf, models_pp_layers=models_pp_layers,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, localization_cam_method=localization_cam_method,
            cam_min_grid=cam_min_grid, cam_smooth_sigma_frac=cam_smooth_sigma_frac,
            run_label=run_label,
        )
    page_files = []
    for part_idx in range(total_parts):
        start = part_idx * page_size
        end = min(total_n, (part_idx + 1) * page_size)
        part_cases = cases.iloc[start:end].copy()
        part_path = _xai_part_path(path, part_idx + 1, total_parts)
        page_files.append(part_path)
        part_label = f"{run_label} — part {part_idx + 1}/{total_parts}" if run_label else f"part {part_idx + 1}/{total_parts}"
        _ORIG_SAVE_FAILURE_ANALYSIS_GRID(
            model, preprocess_fn, test_df, part_cases, candidate_layers, gt_box_map,
            part_path, image_size=image_size, heatmap_threshold_pct=heatmap_threshold_pct,
            loc_score_threshold=loc_score_threshold, n_failure=len(part_cases),
            paper_dpi=paper_dpi, save_pdf=save_pdf, models_pp_layers=models_pp_layers,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, localization_cam_method=localization_cam_method,
            cam_min_grid=cam_min_grid, cam_smooth_sigma_frac=cam_smooth_sigma_frac,
            run_label=part_label,
        )
    _xai_write_paginated_manifest(path, total_n, total_parts, page_size, 'failure_analysis', page_files)


def save_lime_explanations(
    model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
    image_size=224, n_samples=4, lime_num_samples=1000, lime_segments=100,
    paper_dpi=600, save_pdf=False, lung_mask_dir=None, use_lung_mask=False,
    allow_ellipse=False, repeats=1, reports_dir=None,
):
    selected = _select_explainer_indices(metrics_df, n_samples=n_samples)
    page_size = _xai_page_size_for_path(path)
    total_n = len(selected)
    total_parts = _xai_num_parts(total_n, page_size)
    if total_parts <= 1:
        return _ORIG_SAVE_LIME_EXPLANATIONS(
            model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
            image_size=image_size, n_samples=n_samples, lime_num_samples=lime_num_samples,
            lime_segments=lime_segments, paper_dpi=paper_dpi, save_pdf=save_pdf,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, repeats=repeats, reports_dir=reports_dir,
        )
    page_files = []
    for part_idx in range(total_parts):
        chunk = selected[part_idx * page_size: min(total_n, (part_idx + 1) * page_size)]
        part_metrics = _xai_ordered_metrics_subset(metrics_df, chunk)
        part_path = _xai_part_path(path, part_idx + 1, total_parts)
        page_files.append(part_path)
        page_reports = None
        if reports_dir is not None:
            page_reports = Path(reports_dir) / 'lime_pages' / f'part{part_idx + 1:02d}'
            page_reports.mkdir(parents=True, exist_ok=True)
        _ORIG_SAVE_LIME_EXPLANATIONS(
            model, preprocess_fn, test_df, part_metrics, gt_box_map, part_path,
            image_size=image_size, n_samples=len(chunk), lime_num_samples=lime_num_samples,
            lime_segments=lime_segments, paper_dpi=paper_dpi, save_pdf=save_pdf,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, repeats=repeats, reports_dir=page_reports,
        )
    _xai_write_paginated_manifest(path, total_n, total_parts, page_size, 'lime', page_files)


def save_shap_explanations(
    model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
    image_size=224, n_samples=4, background_n=16, paper_dpi=600,
    save_pdf=False, background_df=None, lung_mask_dir=None,
    use_lung_mask=False, allow_ellipse=False, reports_dir=None,
):
    selected = _select_explainer_indices(metrics_df, n_samples=n_samples)
    page_size = _xai_page_size_for_path(path)
    total_n = len(selected)
    total_parts = _xai_num_parts(total_n, page_size)
    if total_parts <= 1:
        return _ORIG_SAVE_SHAP_EXPLANATIONS(
            model, preprocess_fn, test_df, metrics_df, gt_box_map, path,
            image_size=image_size, n_samples=n_samples, background_n=background_n,
            paper_dpi=paper_dpi, save_pdf=save_pdf, background_df=background_df,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, reports_dir=reports_dir,
        )
    page_files = []
    for part_idx in range(total_parts):
        chunk = selected[part_idx * page_size: min(total_n, (part_idx + 1) * page_size)]
        part_metrics = _xai_ordered_metrics_subset(metrics_df, chunk)
        part_path = _xai_part_path(path, part_idx + 1, total_parts)
        page_files.append(part_path)
        page_reports = None
        if reports_dir is not None:
            page_reports = Path(reports_dir) / 'shap_pages' / f'part{part_idx + 1:02d}'
            page_reports.mkdir(parents=True, exist_ok=True)
        _ORIG_SAVE_SHAP_EXPLANATIONS(
            model, preprocess_fn, test_df, part_metrics, gt_box_map, part_path,
            image_size=image_size, n_samples=len(chunk), background_n=background_n,
            paper_dpi=paper_dpi, save_pdf=save_pdf, background_df=background_df,
            lung_mask_dir=lung_mask_dir, use_lung_mask=use_lung_mask,
            allow_ellipse=allow_ellipse, reports_dir=page_reports,
        )
    _xai_write_paginated_manifest(path, total_n, total_parts, page_size, 'shap', page_files)


def _best_xai_candidate_indices(metrics_df, candidate_pool):
    ranked = _rank_final_locked_cam_metrics(metrics_df, require_valid=True)
    if ranked is None or len(ranked) == 0:
        return []
    idxs = pd.to_numeric(ranked.get("sample_index"), errors="coerce").dropna().astype(int).tolist()
    return idxs[:max(1, int(candidate_pool))]


def _lime_case_explanation(model, preprocess_fn, test_df, idx, gt_box_map, image_size,
                           lime_num_samples, lime_segments, repeats,
                           lung_mask_dir=None, use_lung_mask=False, allow_ellipse=False):
    from lime import lime_image
    from skimage.segmentation import slic
    row = test_df.iloc[int(idx)]
    pid = str(row["patientId"])
    raw = load_dicom_grayscale(row["image_path"], image_size)
    raw_rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32)
    gt_boxes = scale_boxes_to_image(pid, row["image_path"], gt_box_map, image_size)
    lung_mask, lung_mode = load_lung_mask_for_patient(
        pid, lung_mask_dir, image_size, allow_ellipse=allow_ellipse
    ) if use_lung_mask else (None, "off")
    segments = slic(raw_rgb, n_segments=int(lime_segments), compactness=10, sigma=1, start_label=1)

    def predict_fn(images):
        arr = np.asarray(images, dtype=np.float32)
        if arr.max() > 2.0:
            arr /= 255.0
        if getattr(model, "_is_torch_adapter", False):
            return model.predict_numpy(arr, mc_passes=1)
        return model(preprocess_fn(tf.constant(arr)), training=False).numpy()

    masks = []
    for rep in range(max(1, int(repeats))):
        explainer = lime_image.LimeImageExplainer(random_state=42 + rep)
        explanation = explainer.explain_instance(
            raw_rgb, predict_fn, labels=(1,), top_labels=None, hide_color=0,
            num_samples=int(lime_num_samples),
            segmentation_fn=lambda _img, seg=segments: seg,
        )
        _temp, mask = explanation.get_image_and_mask(
            label=1, positive_only=True, num_features=10, hide_rest=False
        )
        binary = mask > 0
        if lung_mask is not None:
            binary &= lung_mask
        masks.append(binary)
    consensus = np.mean(np.stack(masks), axis=0) >= 0.5 if masks else np.zeros((image_size, image_size), bool)
    jaccards = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            union = np.logical_or(masks[i], masks[j]).sum()
            jaccards.append(np.logical_and(masks[i], masks[j]).sum() / union if union else 1.0)
    stability = float(np.mean(jaccards)) if jaccards else 1.0
    gt_mask = boxes_to_mask(gt_boxes, image_size, image_size)
    union = np.logical_or(consensus, gt_mask).sum()
    iou = float(np.logical_and(consensus, gt_mask).sum() / union) if union else np.nan
    gt_coverage = float(np.logical_and(consensus, gt_mask).sum() / max(gt_mask.sum(), 1)) if gt_mask.any() else np.nan
    precision = float(np.logical_and(consensus, gt_mask).sum() / max(consensus.sum(), 1)) if consensus.any() else 0.0
    score = 0.45 * _safe_metric_num(iou) + 0.30 * _safe_metric_num(stability) + 0.15 * _safe_metric_num(gt_coverage) + 0.10 * _safe_metric_num(precision)
    return {
        "sample_index": int(idx), "patientId": pid, "raw_rgb": raw_rgb,
        "gt_boxes": gt_boxes, "consensus": consensus, "stability": stability,
        "gt_iou": iou, "gt_coverage": gt_coverage, "precision": precision,
        "quality_score": float(score), "lung_mask_mode": lung_mode,
    }


def save_best_lime_multipage_pdf(model, preprocess_fn, test_df, metrics_df, gt_box_map,
                                 output_pdf, reports_dir, image_size=224, n_output=100,
                                 candidate_pool=150, cases_per_page=4, lime_num_samples=2000,
                                 lime_segments=120, repeats=3, lung_mask_dir=None,
                                 use_lung_mask=False, allow_ellipse=False):
    if int(n_output) <= 0:
        print("[XAI-PDF][SKIP] n_output=0; expensive explanation ranking skipped.")
        return pd.DataFrame()
    try:
        from skimage.segmentation import mark_boundaries
        import lime  # noqa: F401
    except Exception as exc:
        print(f"[BEST-LIME][WARN] Dependencies unavailable: {exc}")
        return pd.DataFrame()
    indices = _best_xai_candidate_indices(metrics_df, candidate_pool)
    results = []
    for idx in tqdm(indices, desc="  Best-LIME candidates"):
        try:
            results.append(_lime_case_explanation(
                model, preprocess_fn, test_df, idx, gt_box_map, image_size,
                lime_num_samples, lime_segments, repeats, lung_mask_dir,
                use_lung_mask, allow_ellipse,
            ))
        except Exception as exc:
            print(f"[BEST-LIME][WARN] idx={idx}: {type(exc).__name__}: {exc}")
    results = sorted(results, key=lambda r: r["quality_score"], reverse=True)[:max(0, int(n_output))]
    if not results:
        return pd.DataFrame()
    output_pdf = Path(output_pdf); output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cases_per_page = max(1, int(cases_per_page)); manifest_rows = []
    with PdfPages(output_pdf) as pdf:
        for start in range(0, len(results), cases_per_page):
            page = results[start:start+cases_per_page]
            fig, axes = plt.subplots(len(page), 2, figsize=(8.27, max(5.5, 3.35*len(page)+0.8)))
            if len(page) == 1: axes = np.asarray([axes])
            fig.suptitle(
                f"AURA-CXR — Best LIME Explanations (exploratory) | {start+1}–{start+len(page)} of {len(results)}\n"
                "ranked after explanation generation; not used for primary quantitative claims",
                fontsize=11.2, fontweight="bold", y=0.995,
            )
            for rr, rec in enumerate(page):
                raw = rec["raw_rgb"]
                ax0, ax1 = axes[rr,0], axes[rr,1]
                ax0.imshow(raw, cmap="gray"); draw_boxes(ax0, rec["gt_boxes"], color=GT_COLOR, label="GT", fontsize=6.5)
                _style_image_axis(ax0, title="Original + GT", title_size=9.0)
                ax1.imshow(mark_boundaries(raw, rec["consensus"].astype(np.int32)))
                draw_boxes(ax1, rec["gt_boxes"], color=GT_COLOR, label="GT", fontsize=6.5)
                _style_image_axis(ax1, title="Best LIME consensus", title_size=9.0)
                info = (f"#{start+rr+1:03d} | {rec['patientId'][:12]} | score={rec['quality_score']:.4f} | "
                        f"IoU={format_metric(rec['gt_iou'])} | stability={format_metric(rec['stability'])} | "
                        f"coverage={format_metric(rec['gt_coverage'])} | precision={format_metric(rec['precision'])}")
                ax1.text(0.5,-0.105,info,transform=ax1.transAxes,ha="center",va="top",fontsize=5.8,linespacing=1.15,wrap=True)
                manifest_rows.append({k:v for k,v in rec.items() if k not in {"raw_rgb","gt_boxes","consensus"}} | {"pdf_order":start+rr+1})
            fig.subplots_adjust(top=0.93,bottom=0.06,hspace=0.45,wspace=0.10)
            pdf.savefig(fig,bbox_inches="tight",pad_inches=0.06); plt.close(fig)
    df = pd.DataFrame(manifest_rows)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(reports_dir)/"xai_lime_best_top100_manifest.csv", index=False)
    print(f"[BEST-LIME][OK] {output_pdf} | n={len(df)}")
    return df


def _compute_signed_shap_maps_for_indices(model, preprocess_fn, test_df, indices, image_size,
                                          background_n, gt_box_map, background_df=None,
                                          lung_mask_dir=None, use_lung_mask=False, allow_ellipse=False):
    import shap  # noqa: F401
    bg_df = background_df if background_df is not None and len(background_df) else test_df
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(bg_df), size=min(max(1,int(background_n)),len(bg_df)), replace=False)
    bg_raw=[]
    for idx in bg_indices:
        raw=load_dicom_grayscale(bg_df.iloc[int(idx)]["image_path"], image_size)
        bg_raw.append(np.repeat(raw[...,None],3,axis=-1))
    bg_raw=np.stack(bg_raw).astype(np.float32)
    x_raw=[]; pids=[]; gt_boxes_list=[]; lung_masks=[]
    for idx in indices:
        row=test_df.iloc[int(idx)]; pid=str(row["patientId"])
        raw=load_dicom_grayscale(row["image_path"],image_size)
        x_raw.append(np.repeat(raw[...,None],3,axis=-1)); pids.append(pid)
        gt_boxes_list.append(scale_boxes_to_image(pid,row["image_path"],gt_box_map,image_size))
        lm,_=load_lung_mask_for_patient(pid,lung_mask_dir,image_size,allow_ellipse=allow_ellipse) if use_lung_mask else (None,"off")
        lung_masks.append(lm)
    x_raw=np.stack(x_raw).astype(np.float32)
    backend="pytorch" if getattr(model,"_is_torch_adapter",False) else "tensorflow"
    if backend=="pytorch":
        signed_maps,_used=_torch_shap_signed_maps(model,bg_raw,x_raw,class_index=1)
    else:
        import shap
        bg_pre=preprocess_fn(tf.constant(bg_raw)).numpy(); x_pre=preprocess_fn(tf.constant(x_raw)).numpy()
        explainer=shap.GradientExplainer(model,bg_pre)
        vals=_extract_shap_class_values(explainer.shap_values(x_pre),class_index=1)
        signed_maps=[]
        for r in range(len(indices)):
            signed_maps.append(np.asarray(vals[r],dtype=np.float32).sum(axis=-1))
        signed_maps=np.stack(signed_maps).astype(np.float32)
    processed=[]
    for r,sm in enumerate(signed_maps):
        sm=np.asarray(sm,dtype=np.float32)
        if sm.shape != (int(image_size),int(image_size)):
            factor=(int(image_size)/sm.shape[0],int(image_size)/sm.shape[1]); sm=zoom(sm,factor,order=1).astype(np.float32)[:image_size,:image_size]
        if lung_masks[r] is not None:
            lm=np.asarray(lung_masks[r],dtype=bool)
            if lm.shape!=sm.shape: lm=resize_binary_mask(lm,sm.shape[0],sm.shape[1])
            sm=sm*lm.astype(np.float32)
        processed.append(sm)
    return x_raw,pids,gt_boxes_list,processed,backend


def save_best_shap_multipage_pdf(model, preprocess_fn, test_df, metrics_df, gt_box_map,
                                 output_pdf, reports_dir, image_size=224, n_output=100,
                                 candidate_pool=150, cases_per_page=4, background_n=16,
                                 background_df=None, lung_mask_dir=None, use_lung_mask=False,
                                 allow_ellipse=False):
    if int(n_output) <= 0:
        print("[XAI-PDF][SKIP] n_output=0; expensive explanation ranking skipped.")
        return pd.DataFrame()
    try:
        import shap  # noqa: F401
    except Exception as exc:
        print(f"[BEST-SHAP][WARN] SHAP unavailable: {exc}")
        return pd.DataFrame()
    indices=_best_xai_candidate_indices(metrics_df,candidate_pool)
    if not indices: return pd.DataFrame()
    try:
        x_raw,pids,gt_boxes_list,signed_maps,backend=_compute_signed_shap_maps_for_indices(
            model,preprocess_fn,test_df,indices,image_size,background_n,gt_box_map,background_df,
            lung_mask_dir,use_lung_mask,allow_ellipse)
    except Exception as exc:
        print(f"[BEST-SHAP][WARN] candidate batch failed: {type(exc).__name__}: {exc}")
        return pd.DataFrame()
    rows=[]
    for j,(idx,pid,boxes,sm) in enumerate(zip(indices,pids,gt_boxes_list,signed_maps)):
        gt_mask=boxes_to_mask(boxes,image_size,image_size); abs_sm=np.abs(sm); total=float(abs_sm.sum())
        energy=float(abs_sm[gt_mask].sum()/total) if total>1e-8 and gt_mask.any() else np.nan
        pos=np.clip(sm,0,None); pos_total=float(pos.sum())
        pos_gt=float(pos[gt_mask].sum()/pos_total) if pos_total>1e-8 and gt_mask.any() else np.nan
        concentration=float(np.percentile(abs_sm,99)/(np.mean(abs_sm)+1e-8)) if np.isfinite(abs_sm).any() else 0.0
        concentration_norm=float(np.tanh(concentration/10.0))
        score=0.70*_safe_metric_num(energy)+0.25*_safe_metric_num(pos_gt)+0.05*concentration_norm
        rows.append({"sample_index":int(idx),"patientId":pid,"gt_boxes":boxes,"raw_rgb":x_raw[j],"signed_map":sm,
                     "shap_abs_energy_inside_gt":energy,"shap_positive_energy_inside_gt":pos_gt,
                     "shap_concentration":concentration,"quality_score":float(score),"backend":backend})
    rows=sorted(rows,key=lambda r:r["quality_score"],reverse=True)[:max(0,int(n_output))]
    if not rows: return pd.DataFrame()
    global_abs=np.concatenate([np.abs(r["signed_map"]).ravel() for r in rows if np.isfinite(r["signed_map"]).any()])
    vmax=max(float(np.percentile(global_abs[np.isfinite(global_abs)],99.0)),1e-8)
    output_pdf=Path(output_pdf); output_pdf.parent.mkdir(parents=True,exist_ok=True); cases_per_page=max(1,int(cases_per_page)); manifest=[]
    with PdfPages(output_pdf) as pdf:
        for start in range(0,len(rows),cases_per_page):
            page=rows[start:start+cases_per_page]
            fig,axes=plt.subplots(len(page),2,figsize=(8.27,max(5.5,3.35*len(page)+0.8)))
            if len(page)==1: axes=np.asarray([axes])
            fig.suptitle(f"AURA-CXR — Best Signed SHAP Explanations (exploratory) | {start+1}–{start+len(page)} of {len(rows)}\\nranked after explanation generation; not used for primary quantitative claims",fontsize=11.2,fontweight="bold",y=0.995)
            im=None
            for rr,rec in enumerate(page):
                raw=rec["raw_rgb"][...,0]; norm=np.clip(rec["signed_map"]/vmax,-1,1)
                ax0,ax1=axes[rr,0],axes[rr,1]; ax0.imshow(raw,cmap="gray"); draw_boxes(ax0,rec["gt_boxes"],color=GT_COLOR,label="GT",fontsize=6.5); _style_image_axis(ax0,title="Original + GT",title_size=9.0)
                ax1.imshow(raw,cmap="gray"); im=ax1.imshow(norm,cmap="coolwarm",vmin=-1,vmax=1,alpha=0.58); draw_boxes(ax1,rec["gt_boxes"],color=GT_COLOR,label="GT",fontsize=6.5); _style_image_axis(ax1,title="Best signed SHAP",title_size=9.0)
                info=(f"#{start+rr+1:03d} | {rec['patientId'][:12]} | score={rec['quality_score']:.4f}\n"
                      f"|SHAP| in GT={format_metric(rec['shap_abs_energy_inside_gt'])} | +SHAP in GT={format_metric(rec['shap_positive_energy_inside_gt'])}")
                ax1.text(0.5,-0.11,info,transform=ax1.transAxes,ha="center",va="top",fontsize=6.5,wrap=True)
                manifest.append({k:v for k,v in rec.items() if k not in {"raw_rgb","gt_boxes","signed_map"}} | {"pdf_order":start+rr+1})
            if im is not None:
                # V25: dedicated colorbar axis prevents collision with the SHAP panels.
                cax = fig.add_axes([0.905, 0.14, 0.014, 0.72])
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("Signed attribution", fontsize=8.0, labelpad=8)
                cb.ax.tick_params(labelsize=7.0, pad=2)
            fig.subplots_adjust(left=0.06, right=0.845, top=0.93, bottom=0.07, hspace=0.56, wspace=0.12); pdf.savefig(fig,bbox_inches="tight",pad_inches=0.08); plt.close(fig)
    df=pd.DataFrame(manifest); Path(reports_dir).mkdir(parents=True,exist_ok=True); df.to_csv(Path(reports_dir)/"xai_shap_best_top100_manifest.csv",index=False)
    print(f"[BEST-SHAP][OK] {output_pdf} | n={len(df)}")
    return df


if __name__ == "__main__":
    main()

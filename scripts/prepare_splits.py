#!/usr/bin/env python3
"""Materialize the released patientId-level (image-level) RSNA split for the training scripts.

The training scripts read ``<results_dir>/splits/{train,val,test}.csv`` with the
columns produced by ``train_aura.build_samples_dataframe`` (label, sample_id,
image_path and bounding-box aggregates used only as XAI ground truth). The
repository ships only patient identifiers, labels and fold assignments
(``data/splits/``), so this script rebuilds the full split files from the
official RSNA label file and a local DICOM directory while keeping the released
patient membership, row order, sample_id and labels exactly. It also places the
released five-fold assignment in ``<results_dir>/splits/development_oof_folds.csv``;
``aura_dl_pair_selection.make_patient_oof_folds`` regenerates the folds with the
recorded seed and stops if they differ from this file.

Example:
    python scripts/prepare_splits.py \
        --labels_csv data/raw/rsna/stage_2_train_labels.csv \
        --dicom_dir data/raw/rsna/stage_2_train_images \
        --results_dir runs/aura_cxr
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "data" / "splits"
SPLITS = ("train", "val", "test")
EXPECTED_ROWS = {"train": 18678, "val": 2668, "test": 5338}


def build_patient_table(labels_csv: Path, dicom_dir: Path) -> pd.DataFrame:
    """Same aggregation as train_aura.build_samples_dataframe (one row per patientId)."""
    labels = pd.read_csv(labels_csv)
    if "patientId" not in labels.columns or "Target" not in labels.columns:
        raise ValueError("labels_csv must contain RSNA columns: patientId and Target")
    df = labels.groupby("patientId", as_index=False)["Target"].max()
    df["label"] = df["Target"].astype(int)
    df["sample_id"] = np.arange(len(df))
    df["image_path"] = df["patientId"].astype(str).map(lambda x: str(dicom_dir / f"{x}.dcm"))
    if {"x", "y", "width", "height"}.issubset(labels.columns):
        bb = labels[["patientId", "x", "y", "width", "height"]].copy()
        bb["bbox_area"] = bb["width"].fillna(0) * bb["height"].fillna(0)
        bb["bbox_cx"] = bb["x"].fillna(0) + bb["width"].fillna(0) / 2
        bb["bbox_cy"] = bb["y"].fillna(0) + bb["height"].fillna(0) / 2
        agg = bb.groupby("patientId").agg(
            bbox_area_max=("bbox_area", "max"),
            bbox_cx_mean=("bbox_cx", "mean"),
            bbox_cy_mean=("bbox_cy", "mean"),
        ).reset_index()
        df = df.merge(agg, on="patientId", how="left")
        df["bbox_area_max"] = df["bbox_area_max"].fillna(0.0)
        df["bbox_cx_mean"] = df["bbox_cx_mean"].fillna(512.0)
        df["bbox_cy_mean"] = df["bbox_cy_mean"].fillna(512.0)
    else:
        raise ValueError("labels_csv must contain the RSNA box columns x, y, width and height")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels_csv", required=True, help="RSNA stage_2_train_labels.csv")
    ap.add_argument("--dicom_dir", required=True, help="directory with <patientId>.dcm files")
    ap.add_argument("--results_dir", required=True, help="run directory used by all training scripts")
    ap.add_argument("--skip_dicom_check", action="store_true",
                    help="do not require the DICOM files to exist (dry run only)")
    ap.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = ap.parse_args()

    dicom_dir = Path(args.dicom_dir)
    table = build_patient_table(Path(args.labels_csv), dicom_dir)
    released = {s: pd.read_csv(RELEASED / f"{s}.csv") for s in SPLITS}

    ids = pd.concat([released[s]["patientId"] for s in SPLITS]).astype(str)
    if ids.duplicated().any():
        raise AssertionError("released splits overlap at patientId level")
    if len(table) != len(ids) or set(table["patientId"].astype(str)) != set(ids):
        raise AssertionError(
            f"labels_csv has {len(table)} patients; the released split covers {len(ids)}. "
            "Use the RSNA stage-2 training label file (26,684 patientIds).")
    if not args.skip_dicom_check:
        missing = int((~table["image_path"].map(lambda p: Path(p).is_file())).sum())
        if missing:
            raise FileNotFoundError(f"{missing} DICOM files are missing under {dicom_dir}")
    table["exists"] = True

    out = Path(args.results_dir) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    indexed = table.set_index(table["patientId"].astype(str))
    for split in SPLITS:
        rel = released[split]
        target = out / f"{split}.csv"
        if target.exists() and not args.force:
            raise FileExistsError(f"{target} exists; use a new results directory or --force")
        full = indexed.loc[rel["patientId"].astype(str)].reset_index(drop=True)
        if len(full) != EXPECTED_ROWS[split]:
            raise AssertionError(f"{split}: {len(full)} rows, expected {EXPECTED_ROWS[split]}")
        if not np.array_equal(full["label"].to_numpy(), rel["label"].to_numpy()):
            raise AssertionError(f"{split}: labels differ from the released split")
        if not np.array_equal(full["sample_id"].to_numpy(), rel["sample_id"].to_numpy()):
            raise AssertionError(f"{split}: sample_id differs from the released split")
        full.to_csv(target, index=False)
        print(f"[SPLITS] {split}: {len(full)} patients ({int(full['label'].sum())} positive) -> {target}")

    folds_target = out / "development_oof_folds.csv"
    if folds_target.exists() and not args.force:
        raise FileExistsError(f"{folds_target} exists; use a new results directory or --force")
    shutil.copyfile(RELEASED / "development_oof_folds.csv", folds_target)
    print(f"[SPLITS] released five-fold assignment -> {folds_target}")


if __name__ == "__main__":
    main()

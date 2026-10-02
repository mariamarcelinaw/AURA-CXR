#!/usr/bin/env python3
"""DenseNet121-XRV candidate with patient-level development OOF support.

Q1 protocol:
- image size fixed at 224;
- outer OOF holdout is never used for early stopping;
- early stopping uses an inner split drawn only from the outer training folds;
- locked-test inference is deferred until the DL pair is locked;
- test/external labels are never used for candidate or pair selection.
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from aura_dl_pair_selection import (
    ensure_development_table,
    make_patient_oof_folds,
    save_oof_artifact,
    two_col,
    load_pair_manifest, create_or_refresh_deployment_manifest, load_deployment_manifest,
    resolve_locked_model_artifact, register_probability_artifact, sha256_file,
)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_dicom_xrv(path, size, xrv):
    import pydicom
    from skimage.transform import resize as skresize
    dcm = pydicom.dcmread(str(path))
    img = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, "RescaleSlope", 1.0)); intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = img * slope + intercept
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img = xrv.datasets.normalize((img * 255.0).astype(np.float32), 255)
    if img.shape != (size, size):
        img = skresize(img, (size, size), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    return img[None, ...]


class DicomDataset:
    def __init__(self, df, size, xrv):
        self.paths = df["image_path"].astype(str).tolist()
        self.y = df["label"].to_numpy(dtype=np.int64)
        self.size = int(size); self.xrv = xrv
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        import torch
        return torch.from_numpy(load_dicom_xrv(self.paths[i], self.size, self.xrv)), int(self.y[i])


def build_model(xrv, torch, dropout=0.3, weights="densenet121-res224-all"):
    # Training keeps the pretrained default. Locked external inference passes
    # weights=None because the development-refit state_dict already contains
    # the complete backbone and must not trigger a network download.
    base = xrv.models.DenseNet(weights=weights)
    feat_dim = base.classifier.in_features if hasattr(base, "classifier") else 1024
    class XRVBinary(torch.nn.Module):
        def __init__(self, backbone, fdim):
            super().__init__(); self.backbone = backbone
            self.drop = torch.nn.Dropout(dropout); self.head = torch.nn.Linear(fdim, 2)
        def forward(self, x):
            import torch.nn.functional as F
            f = self.backbone.features(x)
            if f.dim() == 4:
                f = F.adaptive_avg_pool2d(F.relu(f, inplace=False), 1).flatten(1)
            return self.head(self.drop(f))
    return XRVBinary(base, feat_dim)


def make_loader(df, args, xrv, shuffle=False):
    from torch.utils.data import DataLoader
    return DataLoader(DicomDataset(df, args.image_size, xrv), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=bool(args.gpu), persistent_workers=args.num_workers > 0)


def deterministic_predict(model, loader, torch, device):
    import torch.nn.functional as F
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(F.softmax(model(xb.to(device, non_blocking=True)), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


PREDICTIVE_INTERVAL_Z95 = 1.96
PREDICTIVE_UNCERTAINTY_DEFINITION = "two_sided_95_predictive_interval_width_equals_2_times_1.96_times_mc_probability_std"


def predictive_interval_width_95(std_probability):
    return (2.0 * PREDICTIVE_INTERVAL_Z95 * np.asarray(std_probability, dtype=np.float32)).astype(np.float32)


def mc_predict(model, loader, torch, mc_passes, device):
    import torch.nn.functional as F
    def enable_dropout():
        model.eval()
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout): mod.train()
    draws = []
    with torch.no_grad():
        for _ in range(int(mc_passes)):
            enable_dropout(); batch = []
            for xb, _ in loader:
                batch.append(F.softmax(model(xb.to(device, non_blocking=True)), dim=1).cpu().numpy())
            draws.append(np.concatenate(batch, axis=0))
    arr = np.stack(draws, axis=0)
    mean = arr.mean(0); std = arr.std(0)
    return mean, std, predictive_interval_width_95(std[:, 1])



def train_model(train_df, monitor_df, args, xrv, torch, device, save_path: Path):
    import torch.nn.functional as F
    model = build_model(xrv, torch, dropout=args.dropout).to(device)
    train_loader = make_loader(train_df, args, xrv, shuffle=True)
    monitor_loader = make_loader(monitor_df, args, xrv, shuffle=False)
    y = train_df["label"].to_numpy(dtype=int)
    weights = torch.tensor([len(y)/(2*max((y==0).sum(),1)), len(y)/(2*max((y==1).sum(),1))], dtype=torch.float32, device=device)
    def focal_ce(logits, target):
        logp = F.log_softmax(logits, dim=1); p = logp.exp()
        ce = F.nll_loss(logp, target, weight=weights, reduction="none")
        pt = p.gather(1, target[:, None]).squeeze(1)
        return (((1-pt)**args.focal_gamma) * ce).mean()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_auc = -np.inf; best_state = None; stale = 0; best_epoch = 0; updates = 0
    for ep in range(args.epochs):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True); loss = focal_ce(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); updates += 1
            total += float(loss.detach().cpu()) * len(xb)
        sched.step(); pred = deterministic_predict(model, monitor_loader, torch, device)[:, 1]
        auc = roc_auc_score(monitor_df["label"].to_numpy(dtype=int), pred)
        print(f"[XRV] epoch={ep+1}/{args.epochs} loss={total/max(len(train_df),1):.5f} inner_val_auc={auc:.5f}")
        if auc > best_auc + 1e-5:
            best_auc = auc; stale = 0; best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= args.patience:
            print(f"[XRV] early stop at epoch {ep+1}"); break
    if best_state is None: raise RuntimeError("XRV training produced no best state")
    model.load_state_dict(best_state); save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "inner_val_auc": float(best_auc), "best_epoch": int(best_epoch),
        "epochs_ran": int(ep + 1), "optimizer_updates": int(updates), "image_size": 224,
        "parameter_count": int(sum(p.numel() for p in model.parameters())), "dropout": float(args.dropout),
    }, save_path)
    return model, float(best_auc)



def _inner_split(outer_train: pd.DataFrame, seed: int, fraction: float):
    tr_idx, va_idx = train_test_split(np.arange(len(outer_train)), test_size=fraction,
                                      stratify=outer_train["label"].to_numpy(dtype=int), random_state=seed)
    return outer_train.iloc[tr_idx].reset_index(drop=True), outer_train.iloc[va_idx].reset_index(drop=True)


def repair_paths(df, dicom_dir):
    df = df.copy()
    if dicom_dir:
        d = Path(dicom_dir)
        df["image_path"] = df["patientId"].astype(str).map(lambda p: str(d / f"{p}.dcm"))
    missing = ~df["image_path"].astype(str).map(lambda x: Path(x).exists())
    if missing.any():
        raise FileNotFoundError(f"Missing {int(missing.sum())} DICOM paths in XRV input")
    return df


def run_oof(args, xrv, torch, device):
    root = Path(args.results_dir); splits = root / "splits"
    train_df = repair_paths(pd.read_csv(splits / "train.csv"), args.dicom_dir)
    val_df = repair_paths(pd.read_csv(splits / "val.csv"), args.dicom_dir)
    dev = ensure_development_table(root, train_df, val_df)
    folds = make_patient_oof_folds(dev, args.oof_folds, args.seed, args.force_oof, root)
    oof = np.full(len(dev), np.nan, dtype=np.float64)
    parameter_count = None
    for fold in sorted(np.unique(folds)):
        outer_train = dev.loc[folds != fold].reset_index(drop=True)
        hold = dev.loc[folds == fold].reset_index(drop=True)
        inner_train, inner_val = _inner_split(outer_train, args.seed + int(fold), args.inner_val_fraction)
        save_path = root / "models" / "oof" / "xrv" / f"fold_{int(fold)}.pt"
        model, auc = train_model(inner_train, inner_val, args, xrv, torch, device, save_path)
        if parameter_count is None:
            parameter_count = int(sum(p.numel() for p in model.parameters()))
        hold_mean, _, _ = mc_predict(model, make_loader(hold, args, xrv), torch, args.mc_passes, device)
        oof[np.where(folds == fold)[0]] = hold_mean[:, 1]
        print(f"[OOF][xrv] fold={fold} outer_hold={len(hold)} inner_auc={auc:.5f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("XRV OOF has missing rows")
    save_oof_artifact(root, "xrv", oof, dev, folds,
                      source="five outer folds; inner validation used for early stopping")
    (root / "reports" / "xrv_oof_protocol.json").write_text(json.dumps({
        "selection_data": "development_oof", "outer_holdout_used_for_early_stopping": False,
        "inner_validation_fraction": args.inner_val_fraction, "test_labels_used": False,
        "locked_test_images_accessed": False, "locked_test_inference": "deferred_until_pair_lock",
        "external_labels_used": False, "image_size": 224, "parameter_count": int(parameter_count) if parameter_count is not None else None, "candidate_training_budget": {"max_epochs": int(args.epochs), "patience": int(args.patience), "mc_passes": int(args.mc_passes)}, "created_at": datetime.now().isoformat()
    }, indent=2), encoding="utf-8")



def _median_oof_best_epoch(root: Path) -> int:
    vals = []
    for path in sorted((root / "models" / "oof" / "xrv").glob("fold_*.pt")):
        ck = __import__("torch").load(path, map_location="cpu")
        if ck.get("best_epoch"): vals.append(int(ck["best_epoch"]))
    if not vals: raise FileNotFoundError("XRV OOF checkpoints with best_epoch are required for development refit")
    return max(1, int(round(float(np.median(vals)))))


def train_fixed_epochs_xrv(dev_df, args, xrv, torch, device, epochs, save_path):
    import torch.nn.functional as F
    model = build_model(xrv, torch, dropout=args.dropout).to(device)
    dl = make_loader(dev_df, args, xrv, shuffle=True); y = dev_df.label.to_numpy(int)
    weights = torch.tensor([len(y)/(2*max((y==0).sum(),1)), len(y)/(2*max((y==1).sum(),1))], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs),1)); updates=0
    for ep in range(int(epochs)):
        model.train(); total=0.0
        for xb,yb in dl:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); logit=model(xb); logp=F.log_softmax(logit,1); pp=logp.exp()
            ce=F.nll_loss(logp,yb,weight=weights,reduction="none"); pt=pp.gather(1,yb[:,None]).squeeze(1)
            loss=(((1-pt)**args.focal_gamma)*ce).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); updates+=1; total+=float(loss.detach().cpu())*len(xb)
        sched.step(); print(f"[XRV][REFIT] epoch={ep+1}/{epochs} loss={total/max(len(dev_df),1):.5f}")
    save_path.parent.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({"state_dict":state,"refit_epochs":int(epochs),"optimizer_updates":int(updates),"image_size":224,
                "parameter_count":int(sum(p.numel() for p in model.parameters())),"dropout":float(args.dropout)},save_path)
    return model


def run_refit(args, xrv, torch, device):
    root=Path(args.results_dir); pair=load_pair_manifest(root,strict=True)
    if pair.get("final_test_evaluated_once"): raise RuntimeError("Cannot refit after locked test")
    if "xrv" not in pair["selected_dl_models"]:
        print("[XRV][REFIT] XRV not selected; skipped."); return
    tr=repair_paths(pd.read_csv(root/"splits"/"train.csv"),args.dicom_dir)
    va=repair_paths(pd.read_csv(root/"splits"/"val.csv"),args.dicom_dir)
    dev=ensure_development_table(root,tr,va); epochs=_median_oof_best_epoch(root)
    path=root/"models"/"xrv_development_refit.pt"
    model=train_fixed_epochs_xrv(dev,args,xrv,torch,device,epochs,path)
    (root/"reports"/"xrv_development_refit_manifest.json").write_text(json.dumps({
        "schema":"aura_cxr_dl_pair_selection_q1_v17","model":"xrv","training_scope":"full_development_train_plus_validation",
        "epoch_policy":"median_best_epoch_from_outer_OOF_inner_validation","refit_epochs":int(epochs),
        "parameter_count":int(sum(p.numel() for p in model.parameters())),"model_path":str(path),"model_sha256":sha256_file(path),
        "test_images_accessed":False,"created_at":datetime.now().isoformat(timespec="seconds")},indent=2),encoding="utf-8")
    create_or_refresh_deployment_manifest(root,require_complete=False)
    print(f"[XRV][REFIT] saved {path}")



def _run_exploratory_fold_ensemble_test(args, xrv, torch, device, test_df):
    """Post-lock exploratory five-fold ensemble for A9/A10 only."""
    root = Path(args.results_dir)
    fold_paths = sorted((root / "models" / "oof" / "xrv").glob("fold_*.pt"))
    if not fold_paths:
        raise FileNotFoundError("Missing XRV outer-fold checkpoints for exploratory A9/A10")
    infer = test_df.copy(); infer["label"] = 0
    draws = []
    for path in fold_paths:
        ck = torch.load(path, map_location="cpu")
        dropout = float(ck.get("dropout", args.dropout)) if isinstance(ck, dict) else float(args.dropout)
        model = build_model(xrv, torch, dropout=dropout).to(device)
        model.load_state_dict(ck.get("state_dict", ck))
        mean, _, _ = mc_predict(model, make_loader(infer, args, xrv), torch, args.mc_passes, device)
        draws.append(mean[:, 1])
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    arr = np.stack(draws, axis=0)
    pdir = root / "probs"; pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / "xrv_exploratory_cvensemble_test.npy"
    np.save(path, two_col(arr.mean(axis=0)))
    (root / "reports" / "xrv_exploratory_test_manifest.json").write_text(json.dumps({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "role": "exploratory_nonselected_models",
        "method": "five_outer_fold_checkpoint_ensemble",
        "generated_after_pair_lock": True,
        "test_labels_used": False,
        "test_set_used_for_selection": False,
        "fold_model_paths": [str(x) for x in fold_paths],
        "probability_path": str(path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"[XRV][EXPLORATORY] saved five-fold ensemble probabilities for n={len(test_df)}")


def run_locked_test(args, xrv, torch, device):
    root = Path(args.results_dir)
    pair = load_pair_manifest(root, strict=True)
    if pair.get("final_test_evaluated_once"):
        raise RuntimeError("Locked test already evaluated")
    test_path = root / "splits" / "test.csv"
    test_cols = [c for c in pd.read_csv(test_path, nrows=0).columns if c != "label"]
    test_df = repair_paths(pd.read_csv(test_path, usecols=test_cols), args.dicom_dir)

    if args.exploratory_test:
        _run_exploratory_fold_ensemble_test(args, xrv, torch, device, test_df)

    if "xrv" not in pair["selected_dl_models"]:
        print("[XRV][LOCKED-TEST] XRV not selected; exact deployment inference skipped.")
        return
    load_deployment_manifest(root, strict=True)
    model_path = resolve_locked_model_artifact(root, "xrv")
    infer = test_df.copy(); infer["label"] = 0
    ck = torch.load(model_path, map_location="cpu")
    checkpoint_dropout = float(ck.get("dropout", args.dropout)) if isinstance(ck, dict) else float(args.dropout)
    model = build_model(xrv, torch, dropout=checkpoint_dropout).to(device)
    model.load_state_dict(ck.get("state_dict", ck))
    mean, std, predictive_width = mc_predict(model, make_loader(infer, args, xrv), torch, args.mc_passes, device)
    pdir = root / "probs"; pdir.mkdir(parents=True, exist_ok=True)
    prob_path = pdir / "xrv_development_refit_test_mean.npy"
    std_path = pdir / "xrv_development_refit_test_std.npy"
    width_path = pdir / "xrv_development_refit_test_predictive_interval_width_95.npy"
    np.save(prob_path, mean.astype(np.float32)); np.save(std_path, std.astype(np.float32))
    np.save(width_path, predictive_width.astype(np.float32))
    register_probability_artifact(root, "xrv", "test", prob_path, model_path)
    (root / "reports" / "xrv_locked_test_inference.json").write_text(json.dumps({
        "schema": "aura_cxr_dl_pair_selection_q1_v17",
        "source_model": str(model_path),
        "source_sha256": sha256_file(model_path),
        "development_refit_only": True,
        "test_inference_started_after_pair_lock": True,
        "test_labels_used": False,
        "test_set_used_for_selection": False,
        "predictive_uncertainty_definition": PREDICTIVE_UNCERTAINTY_DEFINITION,
        "predictive_interval_width_path": str(width_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    print(f"[XRV][LOCKED-TEST] saved exact refit probabilities for n={len(test_df)}")


def run_standard(args, xrv, torch, device):
    root = Path(args.results_dir); sp = root / "splits"
    train_df = repair_paths(pd.read_csv(sp / "train.csv"), args.dicom_dir)
    val_df = repair_paths(pd.read_csv(sp / "val.csv"), args.dicom_dir)
    test_df = repair_paths(pd.read_csv(sp / "test.csv"), args.dicom_dir)
    model, auc = train_model(train_df, val_df, args, xrv, torch, device, root / "models" / "xrv_densenet121_final.pt")
    pdir = root / "probs"; pdir.mkdir(parents=True, exist_ok=True)
    for split, df in [("val", val_df), ("test", test_df)]:
        mean, std, ci = mc_predict(model, make_loader(df, args, xrv), torch, args.mc_passes, device)
        np.save(pdir / f"xrv_{split}_mean.npy", mean.astype(np.float32))
        np.save(pdir / f"xrv_{split}_std.npy", std.astype(np.float32))
        np.save(pdir / f"xrv_{split}_ci.npy", ci.astype(np.float32))
    print(f"[XRV] standard training complete; best validation AUC={auc:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True); ap.add_argument("--dicom_dir", default="")
    ap.add_argument("--mode", choices=["standard", "oof", "refit", "test_infer"], default="oof")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--mc_passes", type=int, default=30); ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--num_workers", type=int, default=4); ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--oof_folds", type=int, default=5); ap.add_argument("--inner_val_fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--force_oof", action="store_true")
    ap.add_argument("--exploratory_test", action="store_true", help="After pair lock only: infer XRV test probabilities even when XRV was not selected.")
    args = ap.parse_args()
    if args.image_size != 224:
        raise ValueError("DenseNet121-XRV Q1 candidate is locked to 224x224")
    seed_all(args.seed)
    import torch, torchxrayvision as xrv
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"[XRV] mode={args.mode} device={device} image_size=224")
    if args.mode == "oof": run_oof(args, xrv, torch, device)
    elif args.mode == "refit": run_refit(args, xrv, torch, device)
    elif args.mode == "test_infer": run_locked_test(args, xrv, torch, device)
    else: run_standard(args, xrv, torch, device)


if __name__ == "__main__":
    main()

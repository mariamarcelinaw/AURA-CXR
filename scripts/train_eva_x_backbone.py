#!/usr/bin/env python3
"""EVA-X-S candidate trainer with strict patient-level development OOF predictions.

Requires the official EVA-X repository (eva_x.py) and EVA-X-S checkpoint. The
official repository exposes ``eva_x_small_patch16(pretrained=...)`` at 224x224.
This script replaces the pretraining head with a binary head and follows the same
outer-OOF/inner-validation protocol as the XRV candidate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from aura_dl_pair_selection import (
    ensure_development_table, make_patient_oof_folds, save_oof_artifact, two_col, load_pair_manifest,
    create_or_refresh_deployment_manifest, load_deployment_manifest, resolve_locked_model_artifact,
    register_probability_artifact, sha256_file,
)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def import_eva_x(repo_dir: Path):
    path = Path(repo_dir) / "eva_x.py"
    if not path.exists():
        raise FileNotFoundError(f"Official EVA-X eva_x.py not found: {path}")
    spec = importlib.util.spec_from_file_location("aura_eva_x_official", str(path))
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    if not hasattr(mod, "eva_x_small_patch16"):
        raise AttributeError("Official EVA-X module lacks eva_x_small_patch16")
    return mod


def load_dicom_rgb(path: str, size: int):
    import pydicom
    from PIL import Image
    d = pydicom.dcmread(path); a = d.pixel_array.astype(np.float32)
    a = a * float(getattr(d, "RescaleSlope", 1.0)) + float(getattr(d, "RescaleIntercept", 0.0))
    if getattr(d, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1": a = a.max() - a
    a -= a.min(); a = a / a.max() if a.max() > 0 else np.zeros_like(a)
    im = Image.fromarray((a*255).astype(np.uint8), mode="L").resize((size,size), Image.Resampling.BILINEAR)
    g = np.asarray(im, dtype=np.float32) / 255.0
    return np.repeat(g[None, ...], 3, axis=0)


class EvaDataset:
    def __init__(self, df, size, mean, std):
        self.df=df.reset_index(drop=True); self.size=size
        self.mean=np.asarray(mean,dtype=np.float32)[:,None,None]; self.std=np.asarray(std,dtype=np.float32)[:,None,None]
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        import torch
        r=self.df.iloc[i]; x=(load_dicom_rgb(str(r.image_path),self.size)-self.mean)/self.std
        return torch.from_numpy(x.astype(np.float32)), int(r.label)


def repair_paths(df,dicom_dir):
    df=df.copy()
    if dicom_dir:
        d=Path(dicom_dir); df["image_path"]=df["patientId"].astype(str).map(lambda p:str(d/f"{p}.dcm"))
    miss=~df["image_path"].astype(str).map(lambda x:Path(x).exists())
    if miss.any(): raise FileNotFoundError(f"Missing {int(miss.sum())} EVA-X DICOM files")
    return df


def loader(df,args,shuffle=False):
    from torch.utils.data import DataLoader
    ds=EvaDataset(df,args.image_size,args.mean,args.std)
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,num_workers=args.num_workers,
                      pin_memory=args.gpu,persistent_workers=args.num_workers>0)


def build_model(args,eva_mod,torch):
    model=eva_mod.eva_x_small_patch16(pretrained=str(Path(args.pretrained_checkpoint)))
    if not hasattr(model,"head") or not hasattr(model.head,"in_features"):
        raise AttributeError("EVA-X model.head.in_features unavailable; verify official repository/timm version")
    in_features=int(model.head.in_features)
    model.head=torch.nn.Sequential(torch.nn.Dropout(args.dropout),torch.nn.Linear(in_features,2))
    return model


PREDICTIVE_INTERVAL_Z95 = 1.96
PREDICTIVE_UNCERTAINTY_DEFINITION = "two_sided_95_predictive_interval_width_equals_2_times_1.96_times_mc_probability_std"


def predictive_interval_width_95(std_probability):
    return (2.0 * PREDICTIVE_INTERVAL_Z95 * np.asarray(std_probability, dtype=np.float32)).astype(np.float32)


def predict(model,dl,torch,device,mc=1):
    import torch.nn.functional as F
    draws=[]
    with torch.no_grad():
        for _ in range(max(1,int(mc))):
            model.eval()
            if mc>1:
                for m in model.modules():
                    if isinstance(m,torch.nn.Dropout):m.train()
            batch=[]
            for xb,_ in dl: batch.append(F.softmax(model(xb.to(device,non_blocking=True)),1).cpu().numpy())
            draws.append(np.concatenate(batch))
    a=np.stack(draws);mean=a.mean(0);std=a.std(0)
    return mean,std,predictive_interval_width_95(std[:,1])



def train_model(train_df,monitor_df,args,eva_mod,torch,device,save_path):
    import torch.nn.functional as F
    model=build_model(args,eva_mod,torch).to(device); tr=loader(train_df,args,True); va=loader(monitor_df,args,False)
    y=train_df.label.to_numpy(int); w=torch.tensor([len(y)/(2*max((y==0).sum(),1)),len(y)/(2*max((y==1).sum(),1))],dtype=torch.float32,device=device)
    def loss_fn(logits,target):
        logp=F.log_softmax(logits,1); pp=logp.exp(); ce=F.nll_loss(logp,target,weight=w,reduction="none")
        pt=pp.gather(1,target[:,None]).squeeze(1); return (((1-pt)**args.focal_gamma)*ce).mean()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1))
    best=-np.inf; state=None; stale=0; best_epoch=0; updates=0
    for ep in range(args.epochs):
        model.train();total=0
        for xb,yb in tr:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();updates+=1
            total+=float(loss.detach().cpu())*len(xb)
        sched.step();vp=predict(model,va,torch,device,1)[0][:,1];auc=roc_auc_score(monitor_df.label.to_numpy(int),vp)
        print(f"[EVA-X] epoch={ep+1}/{args.epochs} loss={total/max(len(train_df),1):.5f} inner_val_auc={auc:.5f}")
        if auc>best+1e-5:
            best=auc;stale=0;best_epoch=ep+1;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=args.patience:break
    if state is None:raise RuntimeError("No EVA-X best state")
    model.load_state_dict(state);save_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"inner_val_auc":float(best),"best_epoch":int(best_epoch),"epochs_ran":int(ep+1),
                "optimizer_updates":int(updates),"image_size":224,"checkpoint":str(args.pretrained_checkpoint),
                "parameter_count":int(sum(p.numel() for p in model.parameters())),"dropout":float(args.dropout)},save_path)
    return model,best



def inner_split(df,seed,fraction):
    a,b=train_test_split(np.arange(len(df)),test_size=fraction,stratify=df.label.to_numpy(int),random_state=seed)
    return df.iloc[a].reset_index(drop=True),df.iloc[b].reset_index(drop=True)


def run_oof(args,eva_mod,torch,device):
    root=Path(args.results_dir);sp=root/"splits"
    tr=repair_paths(pd.read_csv(sp/"train.csv"),args.dicom_dir);va=repair_paths(pd.read_csv(sp/"val.csv"),args.dicom_dir)
    dev=ensure_development_table(root,tr,va);folds=make_patient_oof_folds(dev,args.oof_folds,args.seed,args.force_oof,root)
    oof=np.full(len(dev),np.nan)
    parameter_count=None
    for f in sorted(np.unique(folds)):
        outer=dev.loc[folds!=f].reset_index(drop=True);hold=dev.loc[folds==f].reset_index(drop=True)
        it,iv=inner_split(outer,args.seed+int(f),args.inner_val_fraction)
        model,auc=train_model(it,iv,args,eva_mod,torch,device,root/"models"/"oof"/"eva_x"/f"fold_{int(f)}.pt")
        if parameter_count is None: parameter_count=int(sum(p.numel() for p in model.parameters()))
        hp,_,_=predict(model,loader(hold,args),torch,device,args.mc_passes)
        oof[np.where(folds==f)[0]]=hp[:,1]
        print(f"[OOF][eva_x] fold={f} held={len(hold)} inner_auc={auc:.5f}")
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not np.all(np.isfinite(oof)):raise RuntimeError("EVA-X OOF incomplete")
    save_oof_artifact(root,"eva_x",oof,dev,folds,source="official EVA-X-S; outer OOF with inner early stopping")
    (root/"reports"/"eva_x_oof_protocol.json").write_text(json.dumps({
        "model":"EVA-X-S","official_repo":str(args.eva_x_repo),"pretrained_checkpoint":str(args.pretrained_checkpoint),
        "selection_data":"development_oof","outer_holdout_used_for_early_stopping":False,"test_labels_used":False,
        "locked_test_images_accessed":False,"locked_test_inference":"deferred_until_pair_lock",
        "external_labels_used":False,"image_size":224,"parameter_count":int(parameter_count) if parameter_count is not None else None,"candidate_training_budget":{"max_epochs":int(args.epochs),"patience":int(args.patience),"mc_passes":int(args.mc_passes)},"created_at":datetime.now().isoformat()
    },indent=2),encoding="utf-8")



def _median_oof_best_epoch(root):
    import torch
    vals=[]
    for path in sorted((root/"models"/"oof"/"eva_x").glob("fold_*.pt")):
        ck=torch.load(path,map_location="cpu")
        if ck.get("best_epoch"):vals.append(int(ck["best_epoch"]))
    if not vals:raise FileNotFoundError("EVA-X OOF checkpoints with best_epoch required")
    return max(1,int(round(float(np.median(vals)))))


def train_fixed_epochs_eva(dev,args,eva_mod,torch,device,epochs,path):
    import torch.nn.functional as F
    model=build_model(args,eva_mod,torch).to(device);dl=loader(dev,args,True);y=dev.label.to_numpy(int)
    w=torch.tensor([len(y)/(2*max((y==0).sum(),1)),len(y)/(2*max((y==1).sum(),1))],dtype=torch.float32,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(int(epochs),1));updates=0
    for ep in range(int(epochs)):
        model.train();total=0
        for xb,yb in dl:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            logits=model(xb);logp=F.log_softmax(logits,1);pp=logp.exp();ce=F.nll_loss(logp,yb,weight=w,reduction="none");pt=pp.gather(1,yb[:,None]).squeeze(1)
            loss=(((1-pt)**args.focal_gamma)*ce).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();updates+=1;total+=float(loss.detach().cpu())*len(xb)
        sched.step();print(f"[EVA-X][REFIT] epoch={ep+1}/{epochs} loss={total/max(len(dev),1):.5f}")
    state={k:v.detach().cpu() for k,v in model.state_dict().items()};path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":state,"refit_epochs":int(epochs),"optimizer_updates":int(updates),"image_size":224,
                "checkpoint":str(args.pretrained_checkpoint),"parameter_count":int(sum(p.numel() for p in model.parameters())),"dropout":float(args.dropout)},path)
    return model


def run_refit(args,eva_mod,torch,device):
    root=Path(args.results_dir);pair=load_pair_manifest(root,strict=True)
    if pair.get("final_test_evaluated_once"):raise RuntimeError("Cannot refit after locked test")
    if "eva_x" not in pair["selected_dl_models"]:print("[EVA-X][REFIT] not selected; skipped.");return
    tr=repair_paths(pd.read_csv(root/"splits"/"train.csv"),args.dicom_dir);va=repair_paths(pd.read_csv(root/"splits"/"val.csv"),args.dicom_dir)
    dev=ensure_development_table(root,tr,va);epochs=_median_oof_best_epoch(root);path=root/"models"/"eva_x_development_refit.pt"
    model=train_fixed_epochs_eva(dev,args,eva_mod,torch,device,epochs,path)
    (root/"reports"/"eva_x_development_refit_manifest.json").write_text(json.dumps({
        "schema":"aura_cxr_dl_pair_selection_q1_v17","model":"eva_x","training_scope":"full_development_train_plus_validation",
        "epoch_policy":"median_best_epoch_from_outer_OOF_inner_validation","refit_epochs":int(epochs),
        "parameter_count":int(sum(p.numel() for p in model.parameters())),"model_path":str(path),"model_sha256":sha256_file(path),
        "official_repo":str(args.eva_x_repo),"pretrained_checkpoint":str(args.pretrained_checkpoint),
        "test_images_accessed":False,"created_at":datetime.now().isoformat(timespec="seconds")},indent=2),encoding="utf-8")
    create_or_refresh_deployment_manifest(root,require_complete=False);print(f"[EVA-X][REFIT] saved {path}")



def _run_exploratory_fold_ensemble_test(args,eva_mod,torch,device,test_df):
    root=Path(args.results_dir);paths=sorted((root/"models"/"oof"/"eva_x").glob("fold_*.pt"))
    if not paths:raise FileNotFoundError("Missing EVA-X outer-fold checkpoints for exploratory A9/A10")
    infer=test_df.copy();infer["label"]=0;draw=[]
    for path in paths:
        ck=torch.load(path,map_location="cpu")
        if isinstance(ck,dict) and "dropout" in ck:args.dropout=float(ck["dropout"])
        model=build_model(args,eva_mod,torch).to(device);model.load_state_dict(ck.get("state_dict",ck))
        mean,_,_=predict(model,loader(infer,args,False),torch,device,args.mc_passes);draw.append(mean[:,1])
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    arr=np.stack(draw);pdir=root/"probs";pdir.mkdir(parents=True,exist_ok=True)
    path=pdir/"eva_x_exploratory_cvensemble_test.npy";np.save(path,two_col(arr.mean(0)))
    (root/"reports"/"eva_x_exploratory_test_manifest.json").write_text(json.dumps({
        "schema":"aura_cxr_dl_pair_selection_q1_v17","role":"exploratory_nonselected_models",
        "method":"five_outer_fold_checkpoint_ensemble","generated_after_pair_lock":True,
        "test_labels_used":False,"test_set_used_for_selection":False,
        "fold_model_paths":[str(x) for x in paths],"probability_path":str(path),
        "created_at":datetime.now().isoformat(timespec="seconds")},indent=2),encoding="utf-8")
    print(f"[EVA-X][EXPLORATORY] saved five-fold ensemble probabilities for n={len(test_df)}")


def run_locked_test(args,eva_mod,torch,device):
    root=Path(args.results_dir);pair=load_pair_manifest(root,strict=True)
    if pair.get("final_test_evaluated_once"):raise RuntimeError("Locked test already evaluated")
    test_path=root/"splits"/"test.csv";test_cols=[c for c in pd.read_csv(test_path,nrows=0).columns if c!="label"]
    te=repair_paths(pd.read_csv(test_path,usecols=test_cols),args.dicom_dir)
    if args.exploratory_test:_run_exploratory_fold_ensemble_test(args,eva_mod,torch,device,te)
    if "eva_x" not in pair["selected_dl_models"]:
        print("[EVA-X][LOCKED-TEST] not selected; exact deployment inference skipped.");return
    load_deployment_manifest(root,strict=True);model_path=resolve_locked_model_artifact(root,"eva_x")
    infer=te.copy();infer["label"]=0;ck=torch.load(model_path,map_location="cpu")
    if isinstance(ck,dict) and "dropout" in ck:args.dropout=float(ck["dropout"])
    model=build_model(args,eva_mod,torch).to(device);model.load_state_dict(ck.get("state_dict",ck))
    mean,std,predictive_width=predict(model,loader(infer,args,False),torch,device,args.mc_passes);pdir=root/"probs";pdir.mkdir(parents=True,exist_ok=True)
    prob_path=pdir/"eva_x_development_refit_test_mean.npy";std_path=pdir/"eva_x_development_refit_test_std.npy"
    width_path=pdir/"eva_x_development_refit_test_predictive_interval_width_95.npy"
    np.save(prob_path,mean.astype(np.float32));np.save(std_path,std.astype(np.float32));np.save(width_path,predictive_width.astype(np.float32));register_probability_artifact(root,"eva_x","test",prob_path,model_path)
    (root/"reports"/"eva_x_locked_test_inference.json").write_text(json.dumps({
        "schema":"aura_cxr_dl_pair_selection_q1_v17","source_model":str(model_path),"source_sha256":sha256_file(model_path),
        "development_refit_only":True,"test_inference_started_after_pair_lock":True,"test_labels_used":False,
        "test_set_used_for_selection":False,"predictive_uncertainty_definition":PREDICTIVE_UNCERTAINTY_DEFINITION,
        "predictive_interval_width_path":str(width_path),"created_at":datetime.now().isoformat(timespec="seconds")},indent=2),encoding="utf-8")
    print(f"[EVA-X][LOCKED-TEST] saved exact refit probabilities for n={len(te)}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results_dir",required=True);ap.add_argument("--dicom_dir",default="")
    ap.add_argument("--mode",choices=["oof","refit","test_infer"],default="oof")
    ap.add_argument("--eva_x_repo",required=True,help="Path to official hustvl/EVA-X checkout containing eva_x.py")
    ap.add_argument("--pretrained_checkpoint",required=True,help="Official EVA-X-S checkpoint eva_x_s_16.pt")
    ap.add_argument("--image_size",type=int,default=224);ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--patience",type=int,default=12);ap.add_argument("--batch_size",type=int,default=24)
    ap.add_argument("--lr",type=float,default=1e-4);ap.add_argument("--weight_decay",type=float,default=0.05)
    ap.add_argument("--dropout",type=float,default=0.2);ap.add_argument("--focal_gamma",type=float,default=2.0)
    ap.add_argument("--mc_passes",type=int,default=30);ap.add_argument("--num_workers",type=int,default=4)
    ap.add_argument("--oof_folds",type=int,default=5);ap.add_argument("--inner_val_fraction",type=float,default=0.10)
    ap.add_argument("--mean",type=float,nargs=3,default=(0.5,0.5,0.5));ap.add_argument("--std",type=float,nargs=3,default=(0.5,0.5,0.5))
    ap.add_argument("--seed",type=int,default=42);ap.add_argument("--force_oof",action="store_true");ap.add_argument("--gpu",action="store_true")
    ap.add_argument("--exploratory_test",action="store_true",help="After pair lock only: infer EVA-X test probabilities even when not selected.")
    args=ap.parse_args()
    if args.image_size!=224:raise ValueError("Official EVA-X-S Q1 candidate is locked to 224x224")
    if not Path(args.pretrained_checkpoint).exists():raise FileNotFoundError(args.pretrained_checkpoint)
    import torch
    seed_all(args.seed);device="cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    eva_mod=import_eva_x(Path(args.eva_x_repo));print(f"[EVA-X] mode={args.mode} device={device} official_repo={args.eva_x_repo}")
    if args.mode=="oof":run_oof(args,eva_mod,torch,device)
    elif args.mode=="refit":run_refit(args,eva_mod,torch,device)
    else:run_locked_test(args,eva_mod,torch,device)


if __name__=="__main__":main()
